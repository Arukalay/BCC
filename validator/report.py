"""Сборка итогового отчёта в markdown по ТЗ (шесть разделов, на русском).

Публичный интерфейс:
    build(meta, result) -> str   готовый отчёт
    summarize(result) -> dict    числа сводки и вывод (считает код, не модель)

Отчёт рассчитан на чтение человеком: требования без замечаний перечислены одной строкой,
у каждого нарушения — номер, критичность, критерий, цитата-доказательство и исправление;
цитата не повторяется, если совпадает с цитатой самого требования. Порядок детерминирован:
критичность, затем порядок в документе.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from .prompts import DEFECT_TYPE, DEFECT_TYPES
from .schemas import AnalysisResult, Requirement, RunMeta

__all__ = ["build", "summarize"]

SEVERITIES = ("critical", "major", "minor")
_RANK = {s: i for i, s in enumerate(SEVERITIES)}
LABEL = {"critical": "Критично", "major": "Существенно", "minor": "Замечание"}
_FAR = 10**6          # позиция для находок, не привязанных к требованию: в конец
TOP_FIXES = 10        # сколько пунктов в «Что исправить в первую очередь»
REFS_IN_SECTION = 6   # сколько затронутых требований перечислять в разделе 4 (остальные — «и ещё N»)
REFS_IN_SUMMARY = 3   # то же в коротких ссылках сводки и итога

LEGEND = (
    "> **Как читать отчёт.** Агент пронумеровал требования (R-01, R-02, …), рядом указан номер из документа, "
    "если он был. Нарушения: **Критично** — по формулировке нельзя однозначно разработать или протестировать; "
    "**Существенно** — вероятно приведёт к разночтению; **Замечание** — не мешает реализации. "
    "Текст `моноширинным шрифтом` — дословная цитата из документа."
)


# ---------------------------------------------------------------- форматирование

def _flat(text: object) -> str:
    """Схлопывает переносы и повторные пробелы в один пробел."""
    return " ".join(str(text).split())


def _cell(text: object) -> str:
    """Безопасно для ячейки markdown-таблицы: без переносов, `|` экранирован."""
    return _flat(text).replace("|", "\\|")


def _quotes(quote: str) -> list[str]:
    """Цитата может состоять из нескольких мест через ' || ' (например, противоречие)."""
    return [p for p in (_flat(x) for x in str(quote).split(" || ")) if p]


def _code(text: str, table: bool = False) -> str:
    """Цитата как code-span: `*`, `_`, `|` и обратные кавычки внутри не ломают вёрстку.

    Забор из обратных кавычек длиннее самой длинной серии внутри; если текст начинается или
    кончается обратной кавычкой, добавляются пробелы-прокладки. В ячейке таблицы `|` экранируется.
    """
    s = _flat(text)
    if table:
        s = s.replace("|", "\\|")
    fence = "`" * (max((len(m) for m in re.findall(r"`+", s)), default=0) + 1)
    pad = " " if s.startswith("`") or s.endswith("`") else ""
    return f"{fence}{pad}{s}{pad}{fence}"


def _first_sentence(text: str) -> str:
    """Первое предложение — для кратких списков в сводке."""
    s = _flat(text)
    m = re.search(r"(?<=[.!?])\s+(?=[A-ZА-ЯЁ«\"(])", s)
    return s[:m.start()] if m else s


def _quote_lines(quote: str, indent: str) -> list[str]:
    parts = _quotes(quote)
    if len(parts) == 1:
        return [f"{indent}- Цитата: {_code(parts[0])}"]
    return [f"{indent}- Цитата {n}: {_code(p)}" for n, p in enumerate(parts, 1)]


def _req_label(rid: str, by_id: dict[str, Requirement]) -> str:
    """«R-03 (Ф-1)» — ID в отчёте и, если есть, номер в документе."""
    r = by_id.get(rid)
    return f"{rid} ({r.doc_id})" if r is not None and r.doc_id else rid


def _req_list(ids: list[str], by_id: dict[str, Requirement], limit: int) -> str:
    """Список требований для ссылки; длинный обрезается: «R-01, R-02, R-03 и ещё 5»."""
    labels = [_req_label(r, by_id) for r in ids]
    if len(labels) <= limit:
        return ", ".join(labels)
    return f"{', '.join(labels[:limit])} и ещё {len(labels) - limit}"


# ---------------------------------------------------------------- данные

@dataclass
class _Item:
    """Нарушение в едином виде: по требованию (R-03.1) или по набору (S-02)."""
    severity: str
    order: int            # позиция в документе (для сортировки)
    seq: int              # исходный порядок: сортировка стабильна
    group: str            # критерий (требование) или вид (набор)
    problem: str
    fix: str
    quote: str
    req_ids: list[str] = field(default_factory=list)
    ref: str = ""
    is_set: bool = False

    @property
    def key(self) -> tuple[int, int, int]:
        return (_RANK[self.severity], self.order, self.seq)

    @property
    def defect_type(self) -> str:
        return DEFECT_TYPE.get(self.group, "прочие")


def _collect(result: AnalysisResult):
    """(нарушения по требованиям, группы нарушений набора по виду, все нарушения по важности)."""
    index = {r.report_id: i for i, r in enumerate(result.requirements)}
    seq = 0
    per_req: dict[str, list[_Item]] = {}
    for check in result.checks:
        for f in check.findings:
            seq += 1
            per_req.setdefault(check.req_id, []).append(_Item(
                f.severity, index.get(check.req_id, _FAR), seq, f.criterion, f.problem, f.fix, f.quote,
                [check.req_id]))
    for rid, items in per_req.items():
        items.sort(key=lambda i: i.key)
        for n, it in enumerate(items, 1):
            it.ref = f"{rid}.{n}"

    set_items: list[_Item] = []
    for f in result.set_findings:
        seq += 1
        order = min((index.get(r, _FAR) for r in f.related_req_ids), default=_FAR)
        set_items.append(_Item(f.severity, order, seq, f.kind, f.problem, f.fix, f.quote,
                               list(f.related_req_ids), is_set=True))
    groups: dict[str, list[_Item]] = {}
    for it in sorted(set_items, key=lambda i: i.key):
        groups.setdefault(it.group, []).append(it)
    ordered = sorted(groups.items(), key=lambda kv: (min(_RANK[i.severity] for i in kv[1]), kv[0].lower()))
    n = 0
    for _, items in ordered:
        for it in items:
            n += 1
            it.ref = f"S-{n:02d}"

    everything = sorted([i for items in per_req.values() for i in items] + set_items, key=lambda i: i.key)
    return per_req, ordered, everything


def summarize(result: AnalysisResult) -> dict:
    """Все числа сводки и вывод. Считаются кодом из структурированных данных."""
    ids = [r.report_id for r in result.requirements]
    checks = {c.req_id: c for c in result.checks}
    req_counts = Counter(f.severity for c in result.checks for f in c.findings)
    set_counts = Counter(f.severity for f in result.set_findings)
    total = {s: req_counts[s] + set_counts[s] for s in SEVERITIES}
    checked = [rid for rid in ids if rid in checks]
    with_violations = [rid for rid in checked if checks[rid].findings]
    verdict = "stop" if total["critical"] else "caveats" if total["major"] else "ok"
    return {
        "requirements": len(ids),
        "checked": len(checked),
        "unchecked": len(ids) - len(checked),
        "with_violations": len(with_violations),
        "clean": len(checked) - len(with_violations),
        "req": {s: req_counts[s] for s in SEVERITIES},
        "set": {s: set_counts[s] for s in SEVERITIES},
        "total": total,
        "total_all": sum(total.values()),
        "verdict": verdict,
    }


# ---------------------------------------------------------------- разделы

def _passport(meta: RunMeta) -> list[str]:
    return [
        "## 1. Паспорт прогона", "",
        "| Параметр | Значение |", "|---|---|",
        f"| Проверяемый документ | {_cell(meta.document_name)} |",
        f"| Дата прогона | {_cell(meta.date)} |",
        f"| Модель | {_cell(meta.model)} |",
        f"| Версия агента | {_cell(meta.agent_version)} |",
        "",
    ]


def _slicing(result: AnalysisResult) -> list[str]:
    reqs = result.requirements
    out = ["## 2. Как агент нарезал требования", ""]
    if not reqs:
        return out + ["Агент не выделил ни одного требования.", ""]
    numbered = sum(1 for r in reqs if r.doc_id)
    out += [
        f"Агент выделил **{len(reqs)}** требований: с номером в документе — {numbered}, без номера — "
        f"{len(reqs) - numbered}. Требованиям без номера агент присвоил свой идентификатор и сохранил дословную цитату.",
        "",
        "| ID в отчёте | ID в документе | Раздел | Краткая формулировка | Цитата (для требований без номера) |",
        "|---|---|---|---|---|",
    ]
    for r in reqs:
        doc_id = _cell(r.doc_id) if r.doc_id else "—"
        quote = "—" if r.doc_id else _code(r.quote, table=True)
        out.append(f"| {_cell(r.report_id)} | {doc_id} | {_cell(r.section)} | {_cell(r.short)} | {quote} |")
    return out + [""]


def _finding_lines(it: _Item, own_quote: str | None) -> list[str]:
    head = f"- **{it.ref} · {LABEL[it.severity]} · {_flat(it.group)}.** {_flat(it.problem)}"
    lines = [head]
    if own_quote is None or _quotes(it.quote) != _quotes(own_quote):
        lines += _quote_lines(it.quote, "  ")
    lines.append(f"  - Как исправить: {_flat(it.fix)}")
    return lines


def _per_requirement(result: AnalysisResult, per_req: dict[str, list[_Item]], stats: dict) -> list[str]:
    out = ["## 3. Проверка каждого требования", ""]
    by_id = {r.report_id: r for r in result.requirements}
    checks = {c.req_id for c in result.checks}
    if not result.requirements:
        return out + ["Проверять нечего: требования не выделены.", ""]
    line = (f"Проверено **{stats['checked']}** из {stats['requirements']}: с нарушениями — "
            f"**{stats['with_violations']}**, без замечаний — {stats['clean']}")
    if stats["unchecked"]:
        line += f", проверка не выполнена — {stats['unchecked']}"
    out += [line + ".", ""]

    for r in result.requirements:
        items = per_req.get(r.report_id)
        if not items:
            continue
        out += [f"### {_req_label(r.report_id, by_id)} · {_flat(r.short)}", "",
                f"> {_code(r.quote)}", "", f"Раздел: {_flat(r.section)}", ""]
        for it in items:
            out += _finding_lines(it, r.quote)
        out.append("")

    orphans = [rid for rid in per_req if rid not in by_id]  # находки к неизвестному требованию (не должно быть)
    for rid in orphans:
        out += [f"### {rid} · требования нет в таблице нарезки", ""]
        for it in per_req[rid]:
            out += _finding_lines(it, None)
        out.append("")

    clean = [_req_label(r.report_id, by_id) for r in result.requirements
             if r.report_id in checks and not per_req.get(r.report_id)]
    unchecked = [_req_label(r.report_id, by_id) for r in result.requirements if r.report_id not in checks]
    if clean:
        out += [f"**Без замечаний ({len(clean)}):** {', '.join(clean)}.", ""]
    if unchecked:
        out += [f"**Проверка не выполнена ({len(unchecked)}):** {', '.join(unchecked)} — модель не вернула "
                "результат; эти требования нужно проверить вручную или повторить прогон.", ""]
    return out


def _set_section(result: AnalysisResult, groups) -> list[str]:
    by_id = {r.report_id: r for r in result.requirements}
    out = ["## 4. Проверка набора требований целиком", ""]
    if not groups:
        return out + ["Замечаний к набору нет.", ""]
    total = sum(len(items) for _, items in groups)
    out += [f"Замечаний к набору: **{total}**. Здесь то, что видно только при взгляде на документ целиком: "
            "противоречия между пунктами, пробелы, шаблон, термины, прослеживаемость.", ""]
    for kind, items in groups:
        title = _flat(kind)
        out += [f"### {title[:1].upper()}{title[1:]} ({len(items)})", ""]
        for it in items:
            out.append(f"- **{it.ref} · {LABEL[it.severity]}.** {_flat(it.problem)}")
            where = _req_list(it.req_ids, by_id, REFS_IN_SECTION) or "—"
            out.append(f"  - Затронуты требования: {where}")
            out += _quote_lines(it.quote, "  ")
            out.append(f"  - Как исправить: {_flat(it.fix)}")
        out.append("")
    return out


def _where(it: _Item, by_id: dict[str, Requirement]) -> str:
    """Короткая ссылка на нарушение: «R-03.1 (Ф-1)» или «S-02 · R-03 (Ф-1), R-04»."""
    if it.is_set:
        rel = _req_list(it.req_ids, by_id, REFS_IN_SUMMARY)
        return f"{it.ref} · {rel}" if rel else it.ref
    r = by_id.get(it.req_ids[0])
    return f"{it.ref} ({r.doc_id})" if r is not None and r.doc_id else it.ref


def _summary(result: AnalysisResult, everything: list[_Item], stats: dict) -> list[str]:
    by_id = {r.report_id: r for r in result.requirements}
    s = stats
    out = [
        "## 5. Сводка", "",
        f"- Требований выделено: **{s['requirements']}**, с нарушениями: **{s['with_violations']}**, "
        f"без замечаний: {s['clean']}" + (f", не проверено: {s['unchecked']}" if s["unchecked"] else "") + ".",
        f"- Нарушений всего: **{s['total_all']}** — по отдельным требованиям {sum(s['req'].values())}, "
        f"по набору целиком {sum(s['set'].values())}.", "",
        "| Где найдено | Критично | Существенно | Замечание | Всего |", "|---|---|---|---|---|",
        f"| В отдельных требованиях | {s['req']['critical']} | {s['req']['major']} | {s['req']['minor']} | {sum(s['req'].values())} |",
        f"| В наборе целиком | {s['set']['critical']} | {s['set']['major']} | {s['set']['minor']} | {sum(s['set'].values())} |",
        f"| **Итого** | **{s['total']['critical']}** | **{s['total']['major']}** | **{s['total']['minor']}** | **{s['total_all']}** |",
        "",
    ]
    critical = [i for i in everything if i.severity == "critical"]
    out += [f"### Критичные дефекты ({len(critical)})", ""]
    if not critical:
        return out + ["Критичных дефектов нет.", ""]
    for kind in DEFECT_TYPES:
        items = [i for i in critical if i.defect_type == kind]
        if not items:
            continue
        out += [f"**{kind[:1].upper()}{kind[1:]} ({len(items)})**", ""]
        out += [f"- {_where(i, by_id)}: {_first_sentence(i.problem)}" for i in items]
        out.append("")
    return out


def _conclusion(result: AnalysisResult, everything: list[_Item], stats: dict) -> list[str]:
    by_id = {r.report_id: r for r in result.requirements}
    s = stats
    out = ["## 6. Итог", ""]
    if s["verdict"] == "stop":
        out.append(f"**Начинать разработку и тестирование по этому документу нельзя.** Найдено критичных "
                   f"дефектов: {s['total']['critical']} — пока они не исправлены, разработчики и тестировщики "
                   "поймут требования по-разному.")
        top = [i for i in everything if i.severity == "critical"]
    elif s["verdict"] == "caveats":
        out.append(f"**Начинать можно с оговорками.** Критичных дефектов нет, существенных — {s['total']['major']}: "
                   "их лучше закрыть до старта или явно принять как риск.")
        top = [i for i in everything if i.severity == "major"]
    else:
        out.append(f"**Начинать разработку и тестирование можно.** Критичных и существенных дефектов нет "
                   f"(замечаний: {s['total']['minor']}).")
        top = []
    if s["unchecked"]:
        out += ["", f"Оговорка: {s['unchecked']} из {s['requirements']} требований не проверены, вывод неполный."]
    if top:
        out += ["", "### Что исправить в первую очередь", ""]
        out += [f"{n}. **{_where(it, by_id)}** — {_flat(it.fix)}" for n, it in enumerate(top[:TOP_FIXES], 1)]
        if len(top) > TOP_FIXES:
            out += ["", f"Остальные нарушения той же критичности ({len(top) - TOP_FIXES}) — в разделах 3 и 4."]
    out += ["", "---", "", "Отчёт сформирован агентом автоматически. Числа в сводке и вывод посчитаны кодом по "
            "результатам проверки; каждая цитата сверена с текстом документа."]
    return out


# ---------------------------------------------------------------- вход

def build(meta: RunMeta, result: AnalysisResult) -> str:
    """Собирает markdown-отчёт на русском по ТЗ (шесть разделов)."""
    per_req, groups, everything = _collect(result)
    stats = summarize(result)
    lines = [f"# Отчёт о проверке требований: {_flat(meta.document_name)}", "", LEGEND, ""]
    lines += _passport(meta)
    lines += _slicing(result)
    lines += _per_requirement(result, per_req, stats)
    lines += _set_section(result, groups)
    lines += _summary(result, everything, stats)
    lines += _conclusion(result, everything, stats)
    return "\n".join(lines).rstrip() + "\n"
