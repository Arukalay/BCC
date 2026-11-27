"""Общие функции проверки цитат и эталона (без LLM)."""
from __future__ import annotations

import re
from pathlib import Path

from validator.loader import normalize as _loader_normalize


def normalize(text: str) -> str:
    """Нормализация для сверки цитат: как у агента (пробелы, кавычки, тире), без невидимых символов
    и без разметки markdown (`**`, обратные кавычки) — в отчёте агент показывает цитаты без неё."""
    return _loader_normalize(re.sub("[\u200b\ufeff]|\\*\\*|`", "", text))


def quote_in_text(quote: str, text: str) -> bool:
    """Каждая часть цитаты (разделитель ' || ') должна встречаться в тексте."""
    parts = [p for p in quote.split(" || ") if p.strip()]
    if not parts:
        return False
    doc = normalize(text)
    return all(normalize(p) in doc for p in parts)


# Разметка отчёта (validator/report.py):
#   "> `цитата требования`"                              — цитата самого требования (раздел 3)
#   "- **R-03.1 · Критично · полнота.** Проблема…"      — нарушение; у набора: "- **S-02 · Критично.** …"
#   "  - Цитата: `…`" / "  - Цитата 2: `…`"              — цитата-доказательство (у нарушения требования —
#                                                          только если отличается от цитаты требования)
_QUOTE_LINE = re.compile(r"^\s+- Цитата(?: \d+)?: (`+)(.*?)\1\s*$")
_REQ_QUOTE = re.compile(r"^> (`+)(.*?)\1\s*$")
_FINDING_HEAD = re.compile(r"^- \*\*(R-\d+\.\d+|S-\d+) · .*?\.\*\* (.*)$")
_SLICING_ROW = re.compile(r"^\|\s*R-\d+\s*\|")
_UNESCAPED_PIPE = re.compile(r"(?<!\\)\|")


def _unwrap(m: re.Match) -> str:
    return m.group(2).strip()


def _row_quote(line: str) -> str | None:
    """Цитата из последней колонки строки таблицы нарезки или None (если там «—»).

    Ячейки делятся по неэкранированному `|`: внутри цитаты `|` записан как `\\|`, а сама цитата
    может быть в двойных обратных кавычках (когда она содержит ``)."""
    cells = [c.strip() for c in _UNESCAPED_PIPE.split(line.strip())]  # ['', 'R-01', '—', раздел, кратко, цитата, '']
    if len(cells) < 7:
        return None
    m = re.fullmatch(r"(`+)(.*)\1", cells[-2])
    return m.group(2).strip().replace("\\|", "|") if m else None


def extract_report_quotes(report_md: str) -> list[str]:
    """Все цитаты отчёта: таблица нарезки, цитаты требований и цитаты-доказательства нарушений."""
    quotes: list[str] = []
    for line in report_md.splitlines():
        m = _QUOTE_LINE.match(line) or _REQ_QUOTE.match(line)
        if m:
            quotes.append(_unwrap(m))
        elif _SLICING_ROW.match(line):
            q = _row_quote(line)
            if q:
                quotes.append(q)
    return quotes


def parse_report_findings(report_md: str) -> list[dict]:
    """Нарушения отчёта: [{'ref': 'R-03.1', 'quotes': [...], 'problem': '...'}].

    У нарушения требования без своей строки цитаты цитата совпадает с цитатой требования."""
    findings: list[dict] = []
    current: dict | None = None
    req_quote: str | None = None
    for line in report_md.splitlines():
        if line.startswith("#"):
            current, req_quote = None, None
            continue
        m = _REQ_QUOTE.match(line)
        if m:
            req_quote = _unwrap(m)
            continue
        head = _FINDING_HEAD.match(line)
        if head:
            current = {"ref": head.group(1), "quotes": [], "problem": head.group(2).strip(), "_req": req_quote}
            findings.append(current)
            continue
        m = _QUOTE_LINE.match(line) if current is not None else None
        if m:
            current["quotes"].append(_unwrap(m))
    for f in findings:
        own = f.pop("_req")
        if not f["quotes"] and own and f["ref"].startswith("R-"):
            f["quotes"] = [own]
    return findings


def find_bad_quotes(report_md: str, doc_text: str) -> list[str]:
    """Цитаты отчёта, которых нет в документе (кандидаты в выдумку модели)."""
    return [q for q in extract_report_quotes(report_md) if not quote_in_text(q, doc_text)]


def parse_reference(path: Path) -> list[dict]:
    """Читает tests/reference_defects.md: список словарей с ключами id, section, quotes, ..."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not re.match(r"^\|\s*D-\d+", line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split(" | ")]
        if len(cells) != 7:
            raise ValueError(f"Неверная строка эталона: {line[:80]}")
        quotes = [q.strip().strip("«»") for q in cells[2].split(" // ")]
        rows.append({"id": cells[0], "section": cells[1], "quotes": quotes,
                     "defect": cells[3], "criterion": cells[4], "severity": cells[5], "kind": cells[6]})
    return rows
