"""Сверка отчёта агента с эталоном и проверка цитат (без LLM).

Запуск:
    python tests/compare_reference.py docs/requirements.md output/report.md

Печатает:
  1. таблицу «дефект из эталона (ID) — найден / пропущен — где в отчёте»;
  2. «лишнее»: находки отчёта, цитата которых не пересекается ни с одним дефектом эталона
     (либо дефект, которого нет в эталоне, либо выдумка модели; смотрим руками);
  3. долю найденного (всего, по типу, по критичности);
  4. цитаты отчёта, которых нет в документе.

«Найден» = цитата какой-либо находки отчёта пересекается с цитатой дефекта эталона:
либо одна содержит другую, либо у них общий фрагмент не короче MIN_COMMON символов и не
короче половины более короткой цитаты (агент мог процитировать только часть фразы).
Перед сравнением убираются `*` и обратные кавычки (в цитатах агента остаётся разметка
исходного markdown, например «Ф-3.** Когда расходы…») и схлопываются пробелы.
Это эвристика: несколько дефектов эталона делят одну цитату (например, предложения Ф-3),
поэтому «найден» означает «отчёт задел это место» и даёт верхнюю оценку доли найденного.
Пограничные случаи нужно смотреть глазами.
"""
from __future__ import annotations

import sys
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import find_bad_quotes, normalize, parse_reference, parse_report_findings  # noqa: E402

REFERENCE = Path(__file__).resolve().parent / "reference_defects.md"
MIN_OVERLAP = 6   # слишком короткие цитаты не сравниваем
MIN_COMMON = 15   # общий фрагмент двух цитат (если ни одна не содержит другую), символов
MIN_SHARE = 0.5   # ...и не меньше этой доли более короткой цитаты (иначе совпадут «Ответ в случае, если…» у разных кодов)


def _plain(text: str) -> str:
    """Нормализованный текст цитаты без разметки markdown (`*` и обратных кавычек)."""
    return normalize(text.replace("*", "").replace("`", ""))


def overlaps(a: str, b: str) -> bool:
    """Цитаты относятся к одному месту документа (см. описание в начале файла)."""
    na, nb = _plain(a), _plain(b)
    short, long_ = sorted((na, nb), key=len)
    if len(short) < MIN_OVERLAP:
        return False
    if short in long_:
        return True
    common = SequenceMatcher(None, short, long_, autojunk=False).find_longest_match(0, len(short), 0, len(long_)).size
    return common >= max(MIN_COMMON, MIN_SHARE * len(short))


def compare(report_md: str, reference_path: Path = REFERENCE):
    """Возвращает (строки эталона со статусом, лишние находки отчёта)."""
    findings = parse_report_findings(report_md)
    ref = parse_reference(reference_path)
    rows = []
    for row in ref:
        hits = [f["ref"] for f in findings
                if any(overlaps(q, rq) for q in f["quotes"] for rq in row["quotes"])]
        rows.append({**row, "found": bool(hits), "where": ", ".join(hits) or "—"})
    extra = [f for f in findings
             if not any(overlaps(q, rq) for q in f["quotes"] for r in ref for rq in r["quotes"])]
    return rows, extra


def _share(rows) -> str:
    found = sum(r["found"] for r in rows)
    return f"{found} из {len(rows)} ({100 * found / max(len(rows), 1):.0f}%)"


def render(rows, extra, bad_quotes) -> str:
    out = ["| ID | Тип | Критичность | Дефект | Статус | Где в отчёте |", "|---|---|---|---|---|---|"]
    for r in rows:
        status = "найден" if r["found"] else "ПРОПУЩЕН"
        short = r["defect"][:70].replace("|", "/")
        out.append(f"| {r['id']} | {r['kind']} | {r['severity']} | {short} | {status} | {r['where']} |")
    out += ["", f"**Найдено: {_share(rows)}** (по совпадению цитат: верхняя оценка, смысл сверять вручную, см. tests/evaluation.md)", ""]
    for kind in ("внутри требования", "набор"):
        sub = [r for r in rows if r["kind"] == kind]
        out.append(f"- {kind}: {_share(sub)}")
    for sev in ("критично", "существенно", "замечание"):
        sub = [r for r in rows if r["severity"] == sev]
        out.append(f"- {sev}: {_share(sub)}")
    out += ["", f"### Лишнее ({len(extra)}): находки без дефекта в эталоне", ""]
    if extra:
        out += ["| Находка | Цитата | Проблема |", "|---|---|---|"]
        for f in extra:
            q = " || ".join(f["quotes"])[:70].replace("|", "/")
            out.append(f"| {f['ref']} | {q} | {f['problem'][:80].replace('|', '/')} |")
    else:
        out.append("Нет.")
    out += ["", f"### Цитаты отчёта, которых нет в документе ({len(bad_quotes)})", ""]
    out += [f"- `{q[:100]}`" for q in bad_quotes] or ["Нет."]
    return "\n".join(out) + "\n"


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    doc_text = Path(argv[1]).read_text(encoding="utf-8")
    report_md = Path(argv[2]).read_text(encoding="utf-8")
    rows, extra = compare(report_md)
    bad = find_bad_quotes(report_md, doc_text)
    print(render(rows, extra, bad))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
