"""Проверки самого эталона: формат, уникальность, цитаты есть в выданном документе."""
from pathlib import Path

import pytest

from tests.helpers import parse_reference, quote_in_text

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "tests" / "reference_defects.md"
DOC = ROOT / "docs" / "requirements.md"


def test_reference_format():
    rows = parse_reference(REF)
    assert len(rows) >= 40
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))
    assert {r["severity"] for r in rows} <= {"критично", "существенно", "замечание"}
    assert all(r["quotes"] and all(r["quotes"]) for r in rows)
    assert {r["kind"] for r in rows} <= {"внутри требования", "набор"}


@pytest.mark.skipif(not DOC.exists(), reason="docs/requirements.md отсутствует")
def test_reference_quotes_exist_in_document():
    text = DOC.read_text(encoding="utf-8")
    missing = [(r["id"], q) for r in parse_reference(REF) for q in r["quotes"] if not quote_in_text(q, text)]
    assert missing == []
