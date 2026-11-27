"""Конвейер анализа: нарезка → проверка требований → проверка набора → проверка цитат.

Публичный интерфейс: run(text) -> AnalysisResult.

Главный инвариант: каждая цитата в результате (Requirement.quote, Finding.quote,
SetFinding.quote) — фрагмент ИСХОДНОГО текста документа (только пробелы схлопнуты), его
можно найти в документе поиском. Недостоверные цитаты исправляются кодом, затем один раз
моделью, иначе запись отбрасывается.

Устойчивость: ошибка, которую повтор не исправит (llm.LLMFatalError: квота, ключ, модель),
останавливает прогон сразу. Плохой ответ модели на одну пачку не роняет прогон: пачка делится,
а требование, которое так и не удалось проверить, попадает в отчёт как «проверка не выполнена».
"""
from __future__ import annotations

import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterator, NamedTuple, TypeVar

from pydantic import BaseModel

from validator import prompts
from validator.llm import LLMFatalError
from validator.loader import normalize
from validator.schemas import (
    AnalysisResult,
    Finding,
    Requirement,
    RequirementCheck,
    SetFinding,
)

log = logging.getLogger(__name__)

BATCH_SIZE = 4            # требований в одном вызове проверки
MAX_WORKERS = 2           # параллельных вызовов (бесплатная модель легко даёт 429)
MAX_CHUNK_CHARS = 30_000  # длиннее — нарезаем документ по разделам
MIN_QUOTE = 3             # короче цитату считаем недостоверной
SAME_PROBLEM = 0.6        # доля общих основ слов, начиная с которой два описания проблемы — дубль
SAME_PROBLEM_CROSS = 0.3  # то же для замечаний к набору из разных проходов на пересекающихся цитатах
SEVERITY_RANK = {"critical": 0, "major": 1, "minor": 2}
_NO_ID = {"", "null", "none", "нет", "-", "—"}


# ---------- схемы ответов модели (внутренние, контракт schemas.py не меняют) ----------

class RawRequirement(BaseModel):
    doc_id: str | None
    section: str
    quote: str
    short: str


class SlicingResult(BaseModel):
    requirements: list[RawRequirement]


class BatchChecks(BaseModel):
    checks: list[RequirementCheck]


class SetResult(BaseModel):
    findings: list[SetFinding]


class QuoteFix(BaseModel):
    index: int
    quote: str


class QuoteFixes(BaseModel):
    fixes: list[QuoteFix]


class MergeGroups(BaseModel):
    groups: list[list[int]]


# ---------- обращение к модели ----------

class _Ask:
    """Обёртка над llm.ask_json: подставляет system-промпт и считает вызовы."""

    def __init__(self, ask_json: Callable[..., BaseModel]) -> None:
        self._ask_json = ask_json
        self._lock = threading.Lock()
        self.calls = 0

    def __call__(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        with self._lock:
            self.calls += 1
        return self._ask_json(prompt, schema, system=prompts.SYSTEM_PROMPT)


# ---------- проверка цитат ----------

_FOLD = str.maketrans({  # замены 1 символ → 1 символ, длина строки не меняется
    "«": '"', "»": '"', "„": '"', "“": '"', "”": '"', "‘": "'", "’": "'",
    "–": "-", "—": "-", "−": "-", "‑": "-", " ": " ", "ё": "е", "Ё": "Е",
})
_MARKUP = set("*`|#>_\\")  # разметка markdown: модель вправе её опустить


def _squash(s: str) -> tuple[str, list[int]]:
    """Приводит строку к «сравнимому» виду и возвращает карту индексов в исходную.

    Убирает разметку, схлопывает пробелы, сворачивает регистр, кавычки и тире.
    Результат[i] соответствует исходной строке s[карта[i]].
    """
    chars: list[str] = []
    idx: list[int] = []
    for i, ch in enumerate(s.translate(_FOLD)):
        if ch in _MARKUP:
            continue
        if ch.isspace():
            if not chars or chars[-1] == " ":
                continue
            ch = " "
        else:
            low = ch.lower()
            ch = low if len(low) == 1 else ch
        chars.append(ch)
        idx.append(i)
    if chars and chars[-1] == " ":
        chars.pop()
        idx.pop()
    return "".join(chars), idx


def _norm_map(text: str) -> tuple[str, list[int]] | None:
    """normalize(text) вместе с картой «индекс в нормализованном тексте → индекс в исходном».

    Работает, пока loader.normalize заменяет символы один к одному и схлопывает пробелы.
    Если это не так, возвращает None, и цитаты останутся в нормализованном виде.
    """
    chars: list[str] = []
    idx: list[int] = []
    for i, ch in enumerate(text):
        if ch.isspace():
            if chars and chars[-1] != " ":
                chars.append(" ")
                idx.append(i)
            continue
        folded = normalize(ch) or ch
        if len(folded) != 1:
            return None
        chars.append(folded)
        idx.append(i)
    if chars and chars[-1] == " ":
        chars.pop()
        idx.pop()
    return "".join(chars), idx


_INLINE_MARKUP = re.compile(r"\*\*|`")     # жирный шрифт и код markdown: в отображённом документе их не видно
_LIST_MARKER = re.compile(r"^[*+-]\s+")    # маркер пункта списка в начале цитаты


def _readable(fragment: str) -> str:
    """Цитата для отчёта: текст документа без символов разметки markdown, пробелы схлопнуты."""
    return " ".join(_LIST_MARKER.sub("", _INLINE_MARKUP.sub("", fragment).strip()).split())


class Located(NamedTuple):
    """Цитата, найденная в документе."""
    quote: str   # для отчёта: исходный текст документа без разметки (части через « || »)
    pos: int     # начало в нормализованном тексте — порядок в документе
    end: int     # конец в нормализованном тексте — по нему склеиваются вложенные дубли нарезки


class _Doc:
    """Исходный документ и поиск в нём дословных цитат."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.norm = normalize(text)
        self._squashed, self._idx = _squash(self.norm)
        mapped = _norm_map(text)
        self._orig = mapped[1] if mapped and mapped[0] == self.norm else None

    def original(self, start: int, end: int) -> str:
        """Текст документа для нормализованного диапазона [start, end) в читаемом виде (_readable).

        В отчёт идут исходные символы (тире, кавычки-ёлочки), чтобы цитату находил поиск по документу.
        """
        if self._orig is None or end <= start:
            return _readable(self.norm[start:end])
        return _readable(self.text[self._orig[start]:self._orig[end - 1] + 1])

    def _find(self, quote: str) -> tuple[str, int] | None:
        """Один фрагмент → (точный фрагмент нормализованного текста, позиция) или None."""
        exact = normalize(quote)
        if len(exact) >= MIN_QUOTE:
            pos = self.norm.find(exact)
            if pos >= 0:
                return exact, pos
        sq, _ = _squash(quote)
        if len(sq) < MIN_QUOTE:
            return None
        p = self._squashed.find(sq)
        if p < 0:
            return None
        start = self._idx[p]
        end = self._idx[p + len(sq) - 1] + 1
        return self.norm[start:end], start

    def resolve(self, quote: str) -> Located | None:
        """Цитата поля (возможно несколько мест через « || ») → Located или None.

        None, если хотя бы одна часть не найдена в документе.
        """
        parts = [p for p in re.split(r"\s*\|\|\s*", quote.strip()) if p.strip()]
        found = [self._find(p) for p in parts]
        if not found or any(f is None for f in found):
            return None
        spans = [(pos, pos + len(frag)) for frag, pos in found]
        return Located(" || ".join(self.original(a, b) for a, b in spans),
                       min(a for a, _ in spans), max(b for _, b in spans))


def _verify(doc: _Doc, ask: _Ask, quotes: list[str], contexts: list[str]) -> list[Located | None]:
    """Проверяет цитаты по документу. Недостоверные один раз отдаёт модели на исправление."""
    result = [doc.resolve(q) for q in quotes]
    bad = [i for i, r in enumerate(result) if r is None]
    if not bad:
        return result
    log.info("Цитат не найдено в документе: %d из %d, запрос исправления", len(bad), len(quotes))
    entries = [(i, quotes[i], contexts[i]) for i in bad]
    try:
        fixes = ask(prompts.build_quote_fix_prompt(doc.text, entries), QuoteFixes)
    except LLMFatalError:
        raise
    except Exception as exc:  # без исправления недостоверные записи просто отбрасываются
        log.warning("Исправление цитат не удалось (%s): %d записей будет отброшено", exc, len(bad))
        return result
    for fix in fixes.fixes:
        if fix.index in bad and fix.quote.strip():
            result[fix.index] = doc.resolve(fix.quote)
    still = sum(1 for i in bad if result[i] is None)
    log.info("После исправления недостоверных цитат: %d", still)
    return result


# ---------- шаг 1: нарезка ----------

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _blocks(text: str) -> list[str]:
    """Смысловые блоки текста подряд: от заголовка markdown до следующего, а если заголовков нет
    (текст из PDF или TXT) — абзацы, разделённые пустой строкой. Склейка блоков = исходный текст."""
    lines = text.splitlines(keepends=True)
    by_headings = any(_HEADING.match(line) for line in lines)
    blocks: list[str] = []
    cur: list[str] = []
    for line in lines:
        starts_block = bool(_HEADING.match(line)) if by_headings else not line.strip()
        if starts_block and cur:
            blocks.append("".join(cur))
            cur = []
        cur.append(line)
    if cur:
        blocks.append("".join(cur))
    return blocks


def _pieces(block: str) -> list[str]:
    """Блок длиннее MAX_CHUNK_CHARS режется по строкам, сверхдлинная строка — по символам."""
    if len(block) <= MAX_CHUNK_CHARS:
        return [block]
    out: list[str] = []
    cur = ""
    for line in block.splitlines(keepends=True):
        for i in range(0, len(line), MAX_CHUNK_CHARS):
            part = line[i:i + MAX_CHUNK_CHARS]
            if cur and len(cur) + len(part) > MAX_CHUNK_CHARS:
                out.append(cur)
                cur = ""
            cur += part
    if cur:
        out.append(cur)
    return out


def _chunks(text: str) -> list[tuple[str, str]]:
    """Делит длинный документ на части не длиннее MAX_CHUNK_CHARS по границам блоков (_blocks).

    Возвращает (подсказка, где начинается часть, текст части); склейка частей = исходный текст.
    Подсказка — последний заголовок второго уровня или, если заголовков нет, первая строка части.
    """
    if len(text) <= MAX_CHUNK_CHARS:
        return [("", text)]
    by_headings = any(_HEADING.match(line) for line in text.splitlines())
    chunks: list[tuple[str, str]] = []
    section, cur, cur_hint = "начало документа", "", ""
    for block in _blocks(text):
        heading = _HEADING.match(block.splitlines()[0]) if block.strip() else None
        if heading and len(heading.group(1)) == 2:
            section = heading.group(2)
        for piece in _pieces(block):
            if cur and len(cur) + len(piece) > MAX_CHUNK_CHARS:
                chunks.append((cur_hint, cur))
                cur = ""
            if not cur:
                first_line = next((line.strip() for line in piece.splitlines() if line.strip()), "")
                cur_hint = section if by_headings else (first_line[:80] or "фрагмент документа")
            cur += piece
    if cur:
        chunks.append((cur_hint, cur))
    return chunks


def _slice(doc: _Doc, ask: _Ask) -> list[Requirement]:
    chunks = _chunks(doc.text)
    raws: list[RawRequirement] = []
    for n, (hint, part) in enumerate(chunks, 1):
        res = ask(prompts.build_slicing_prompt(part, hint or None), SlicingResult)
        log.info("Нарезка: фрагмент %d/%d, найдено %d", n, len(chunks), len(res.requirements))
        raws.extend(res.requirements)

    verified = _verify(doc, ask, [r.quote for r in raws], [r.short for r in raws])
    found: list[tuple[Located, RawRequirement]] = []
    for raw, loc in zip(raws, verified):
        if loc is None:
            log.warning("Требование отброшено, цитата не найдена в документе: %s", raw.short)
            continue
        found.append((loc, raw))

    # Порядок документа. Цитата, целиком лежащая внутри другой (или совпадающая с ней), —
    # дубль: он возникает при повторах модели и при слиянии фрагментов.
    found.sort(key=lambda t: (t[0].pos, t[0].pos - t[0].end))
    kept: list[tuple[Located, RawRequirement]] = []
    max_end = -1
    for loc, raw in found:
        if loc.end <= max_end:
            log.info("Требование отброшено как дубль: %s", raw.short)
            continue
        max_end = loc.end
        kept.append((loc, raw))

    return [
        Requirement(
            report_id=f"R-{n:02d}",
            doc_id=None if (raw.doc_id or "").strip().lower() in _NO_ID else raw.doc_id.strip(),
            section=raw.section.strip(),
            quote=loc.quote,
            short=raw.short.strip(),
        )
        for n, (loc, raw) in enumerate(kept, 1)
    ]


# ---------- шаг 2: проверка требований ----------

def _norm_id(raw: str) -> str:
    """Идентификатор требования из ответа модели к виду R-01 («R-1», «r-01», «R-01 (Ф-1)» → «R-01»)."""
    m = re.search(r"R\s*-?\s*(\d+)", raw, re.IGNORECASE)
    return f"R-{int(m.group(1)):02d}" if m else raw.strip()


def _check_batch(text: str, batch: list[Requirement], ask: _Ask) -> list[RequirementCheck]:
    """Проверяет пачку. Если вызов упал, делит пополам; недостающие проверки запрашивает снова.

    Требование, которое не удалось проверить даже поодиночке, пропускается (в отчёте:
    «проверка не выполнена»), чтобы один плохой ответ модели не лишал прогон отчёта.
    """
    ids = {r.report_id for r in batch}
    try:
        res = ask(prompts.build_check_prompt(text, batch), BatchChecks)
        by_id: dict[str, RequirementCheck] = {}
        for c in res.checks:
            cid = _norm_id(c.req_id)
            if cid in ids and cid not in by_id:
                by_id[cid] = c.model_copy(update={"req_id": cid})
        if not by_id:
            raise ValueError("в ответе нет проверок для запрошенных требований")
    except LLMFatalError:
        raise  # квота, ключ, недоступность: деление пачки не поможет, только умножит вызовы
    except Exception as exc:
        if len(batch) == 1:
            log.error("Требование %s не проверено, в отчёте будет «проверка не выполнена»: %s",
                      batch[0].report_id, exc)
            return []
        log.warning("Пачка из %d требований не обработана (%s), делю пополам", len(batch), exc)
        mid = len(batch) // 2
        return _check_batch(text, batch[:mid], ask) + _check_batch(text, batch[mid:], ask)

    checks = [by_id[r.report_id] for r in batch if r.report_id in by_id]
    missing = [r for r in batch if r.report_id not in by_id]
    if missing:
        log.warning("Модель пропустила %d требований из пачки, повторяю", len(missing))
        checks += _check_batch(text, missing, ask)
    order = {r.report_id: n for n, r in enumerate(batch)}
    return sorted(checks, key=lambda c: order[c.req_id])


def _map_fail_fast(fn: Callable[[T], R], items: list[T]) -> Iterator[R]:
    """Параллельный map в порядке items. При первой ошибке отменяет задачи, ещё не начатые.

    Обычный pool.map при ошибке всё равно дожидается всей очереди: если модель недоступна,
    прогон тянулся бы до конца очереди вместо того, чтобы остановиться сразу.
    """
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(fn, item) for item in items]
        try:
            for future in futures:
                yield future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            raise


def _check_all(doc: _Doc, reqs: list[Requirement], ask: _Ask) -> list[RequirementCheck]:
    batches = [reqs[i:i + BATCH_SIZE] for i in range(0, len(reqs), BATCH_SIZE)]
    checks: list[RequirementCheck] = []
    for n, got in enumerate(_map_fail_fast(lambda b: _check_batch(doc.text, b, ask), batches), 1):
        checks.extend(got)
        log.info("Проверка требований: пачка %d/%d, готово %d/%d",
                 n, len(batches), len(checks), len(reqs))
    return checks


# ---------- шаг 3: проверка набора ----------

def _check_set(doc: _Doc, reqs: list[Requirement], ask: _Ask) -> list[SetFinding]:
    """Проверка набора проходами prompts.SET_PASSES; дубли между проходами убирает _finalize."""
    passes = list(prompts.SET_PASSES)

    def one(focus: str) -> list[SetFinding]:
        res = ask(prompts.build_set_prompt(doc.text, reqs, focus=focus), SetResult)
        log.info("Проверка набора, проход «%s»: получено %d замечаний",
                 prompts.SET_PASSES[focus]["title"], len(res.findings))
        return res.findings

    found: list[SetFinding] = []
    for part in _map_fail_fast(one, passes):  # порядок проходов фиксирован, результат детерминирован
        found.extend(part)
    return found


# ---------- шаг 4: проверка цитат в находках и сборка результата ----------

def _canon(value: str, known: list[str]) -> str:
    """Приводит название критерия/вида к каноническому написанию, если оно известно."""
    v = value.strip()
    return next((k for k in known if k.lower() == v.lower()), v)


def _stems(text: str) -> set[str]:
    """Основы значимых слов (первые 4 буквы): грубо сглаживает русские окончания."""
    words = re.findall(r"\w+", text.lower().replace("ё", "е"))
    return {w[:4] for w in words if len(w) > 3}


def _same_problem(a: str, b: str, threshold: float = SAME_PROBLEM) -> bool:
    """Два описания проблемы говорят об одном и том же (достаточно общих слов)."""
    sa, sb = _stems(a), _stems(b)
    if not sa or not sb:
        return a.strip().lower() == b.strip().lower()
    return len(sa & sb) / len(sa | sb) >= threshold


def _quotes_overlap(a: str, b: str) -> bool:
    """Цитаты (возможно, из двух частей через « || ») указывают на одно место: одна часть содержит другую."""
    pa = [p for p in a.split(" || ") if p]
    pb = [p for p in b.split(" || ") if p]
    return any(x in y or y in x for x in pa for y in pb)


def _drop_cross_pass_duplicates(items: list[SetFinding]) -> list[SetFinding]:
    """Убирает повторы между проходами проверки набора.

    Проходы перекрываются: одну проблему (например, пустой блок согласующих) разные проходы
    описывают с чуть разными цитатами и видами, и _unique их не склеивает. Повтор — это
    пересекающиеся цитаты плюс похожее описание проблемы (порог мягче, чем внутри вида).
    items отсортированы по важности, выживает первое; сравнение идёт и с уже отброшенными,
    чтобы цепочка A≈B≈C не рвалась.
    """
    kept: list[SetFinding] = []
    seen: list[SetFinding] = []
    for s in items:
        dup = any(_quotes_overlap(o.quote, s.quote) and _same_problem(o.problem, s.problem, SAME_PROBLEM_CROSS)
                  for o in seen)
        seen.append(s)
        if dup:
            log.info("Повтор замечания к набору между проходами отброшен (%s)", s.kind)
            continue
        kept.append(s)
    return kept


T = TypeVar("T")
R = TypeVar("R")


def _unique(cands: list[tuple[tuple[int, int], str, str, str, T]]) -> list[T]:
    """Сортирует находки по рангу и убирает повторы.

    cands: (ранг, критерий или вид, цитата, problem, сама находка). Повтор — это
    тот же критерий, та же цитата и по смыслу та же проблема. Разные проблемы с
    одной цитатой остаются: на один фрагмент законно приходится несколько дефектов
    (особенно у «пробелов», которые привязаны к ближайшему фрагменту). Выживает
    более критичная находка, при равенстве — первая из ответа модели.
    """
    kept: list[T] = []
    seen: list[tuple[str, str, str]] = []
    for _, label, quote, problem, item in sorted(cands, key=lambda c: c[0]):
        if any(lb == label and q == quote and _same_problem(p, problem) for lb, q, p in seen):
            continue
        seen.append((label, quote, problem))
        kept.append(item)
    return kept


def _finalize(
    doc: _Doc,
    ask: _Ask,
    reqs: list[Requirement],
    checks: list[RequirementCheck],
    set_findings: list[SetFinding],
) -> tuple[list[RequirementCheck], list[SetFinding]]:
    found = [f for c in checks for f in c.findings] + list(set_findings)
    verified = _verify(doc, ask, [f.quote for f in found], [f.problem for f in found])
    ok = {id(f): v for f, v in zip(found, verified)}
    known_ids = {r.report_id for r in reqs}

    def rank(f: Finding | SetFinding, pos: int) -> tuple[int, int]:
        return SEVERITY_RANK[f.severity], pos

    by_req = {c.req_id: c for c in checks}
    final_checks: list[RequirementCheck] = []
    for r in reqs:
        if r.report_id not in by_req:  # проверка не выполнена: в отчёте это будет видно
            continue
        cands: list[tuple[tuple[int, int], str, str, str, Finding]] = []
        for f in by_req[r.report_id].findings:
            v = ok[id(f)]
            if v is None or not f.problem.strip():
                log.warning("Находка отброшена (%s, %s): цитата не найдена в документе",
                            r.report_id, f.criterion)
                continue
            quote, pos = v.quote, v.pos
            criterion = _canon(f.criterion, prompts.CRITERIA)
            cands.append((rank(f, pos), criterion, quote, f.problem, f.model_copy(update={
                "req_id": r.report_id, "criterion": criterion, "quote": quote,
            })))
        findings = _unique(cands)
        final_checks.append(RequirementCheck(
            req_id=r.report_id,
            verdict="violations" if findings else "ok",
            findings=findings,
        ))

    set_cands: list[tuple[tuple[int, int], str, str, str, SetFinding]] = []
    for s in set_findings:
        v = ok[id(s)]
        if v is None or not s.problem.strip():
            log.warning("Замечание к набору отброшено (%s): цитата не найдена в документе", s.kind)
            continue
        quote, pos = v.quote, v.pos
        kind = _canon(s.kind, prompts.SET_KINDS)
        ids = list(dict.fromkeys(rid for rid in map(_norm_id, s.related_req_ids) if rid in known_ids))
        set_cands.append((rank(s, pos), kind, quote, s.problem, s.model_copy(update={
            "kind": kind, "quote": quote, "related_req_ids": ids,
        })))
    return final_checks, _merge_same_defects(_drop_cross_pass_duplicates(_unique(set_cands)), ask)


def _merge_same_defects(items: list[SetFinding], ask: _Ask) -> list[SetFinding]:
    """Склеивает замечания к набору, которые описывают один и тот же дефект разными словами.

    Похожесть по словам повторы от разных дефектов не отличает, поэтому группы повторов
    определяет модель по короткому списку замечаний (без документа). Код проверяет номера;
    в группе остаётся первое замечание — items уже отсортированы по важности, — а списки
    затронутых требований объединяются. Если вызов не удался, замечания остаются как были.
    """
    if len(items) < 2:
        return items
    try:
        res = ask(prompts.build_merge_prompt([(i, s.kind, s.problem, s.fix) for i, s in enumerate(items)]), MergeGroups)
    except LLMFatalError:
        raise
    except Exception as exc:
        log.warning("Склейка повторов по смыслу не удалась (%s), замечания к набору оставлены как есть", exc)
        return items
    parent = list(range(len(items)))   # объединение групп; корень — меньший номер, то есть более важное замечание

    def root(i: int) -> int:
        while parent[i] != i:
            i = parent[i]
        return i

    for group in res.groups:
        members = sorted({i for i in group if isinstance(i, int) and 0 <= i < len(items)})
        for i in members[1:]:
            a, b = sorted((root(members[0]), root(i)))
            parent[b] = a
    ids_of: dict[int, list[str]] = {}
    for i, item in enumerate(items):
        ids_of.setdefault(root(i), []).extend(item.related_req_ids)
    merged = [item.model_copy(update={"related_req_ids": list(dict.fromkeys(ids_of[i]))})
              for i, item in enumerate(items) if root(i) == i]
    if len(merged) < len(items):
        log.info("Склеено повторов по смыслу: %d замечаний к набору", len(items) - len(merged))
    return merged


# ---------- публичный интерфейс ----------

def run(text: str, ask_json: Callable[..., BaseModel] | None = None) -> AnalysisResult:
    """Анализирует документ с требованиями.

    Args:
        text: текст документа (результат loader.load).
        ask_json: функция вида ask_json(prompt, schema, system=None) -> BaseModel.
            По умолчанию llm.ask_json; параметр нужен для подмены в тестах.

    Returns:
        AnalysisResult: нарезанные требования (в порядке документа), проверка каждого
        и замечания к набору. Находки отсортированы по критичности, затем по порядку
        в документе. Все цитаты дословно присутствуют в тексте.
    """
    if ask_json is None:
        from validator.llm import ask_json as default_ask_json
        ask_json = default_ask_json
    ask = _Ask(ask_json)
    doc = _Doc(text)
    log.info("Версия промптов: %s", prompts.PROMPTS_VERSION)

    log.info("Шаг 1/4: нарезка на требования")
    reqs = _slice(doc, ask)
    if not reqs:
        raise RuntimeError("Нарезка не дала ни одного требования с достоверной цитатой")
    log.info("Нарезано требований: %d", len(reqs))

    log.info("Шаг 2/4: проверка каждого требования")
    checks = _check_all(doc, reqs, ask)

    log.info("Шаг 3/4: проверка набора требований")
    set_findings = _check_set(doc, reqs, ask)

    log.info("Шаг 4/4: проверка цитат в находках")
    checks, set_findings = _finalize(doc, ask, reqs, checks, set_findings)

    n_find = sum(len(c.findings) for c in checks)
    unchecked = len(reqs) - len(checks)
    if unchecked:
        log.warning("Не удалось проверить требований: %d (в отчёте отмечены)", unchecked)
    log.info("Готово: требований %d, находок %d, замечаний к набору %d, вызовов модели %d",
             len(reqs), n_find, len(set_findings), ask.calls)
    return AnalysisResult(requirements=reqs, checks=checks, set_findings=set_findings)
