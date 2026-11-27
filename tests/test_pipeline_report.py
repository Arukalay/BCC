"""Конвейер на моке модели (tests/mock_llm.py) -> report.build, без реальной модели.

Проверяет, что ничего не теряется по дороге: каждая находка попадает в отчёт,
цитаты в отчёте дословно есть в документе, числа сводки совпадают с данными.
"""
import re
from pathlib import Path

import pytest

from tests import mock_llm
from tests.helpers import find_bad_quotes, parse_report_findings
from validator import pipeline
from validator.report import build, summarize
from validator.schemas import RunMeta

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "requirements.md"

pytestmark = pytest.mark.skipif(not DOC.exists(), reason="docs/requirements.md отсутствует")


@pytest.fixture(scope="module")
def run_result():
    text = DOC.read_text(encoding="utf-8")
    return text, pipeline.run(text, ask_json=mock_llm.MockLLM())


@pytest.fixture(scope="module")
def report(run_result):
    meta = RunMeta(document_name="requirements.md", date="2026-11-27", model="mock", agent_version="test")
    return build(meta, run_result[1])


def test_report_has_six_sections(report):
    for n, title in enumerate(["Паспорт прогона", "Как агент нарезал требования", "Проверка каждого требования",
                               "Проверка набора требований целиком", "Сводка", "Итог"], 1):
        assert f"## {n}. {title}" in report


def test_every_requirement_is_in_slicing_table_and_in_section_3(run_result, report):
    sec3 = report.split("## 3.")[1].split("## 4.")[0]
    listed = " ".join(line for line in sec3.splitlines() if line.startswith(("**Без замечаний", "**Проверка не выполнена")))
    checks = {c.req_id: c for c in run_result[1].checks}
    for r in run_result[1].requirements:
        assert f"| {r.report_id} |" in report
        if checks.get(r.report_id) and checks[r.report_id].findings:
            assert f"### {r.report_id}" in sec3                  # с нарушениями — отдельный блок
        else:
            assert re.search(rf"\b{r.report_id}\b", listed)       # без замечаний — одной строкой


def test_no_finding_lost(run_result, report):
    result = run_result[1]
    expected = sum(len(c.findings) for c in result.checks) + len(result.set_findings)
    assert expected > 0
    assert len(parse_report_findings(report)) == expected


def test_quotes_in_report_exist_in_document(run_result, report):
    # цитаты конвейера содержат разметку исходного markdown (например «Ф-3.** Когда…»), в отчёте это code-span
    assert find_bad_quotes(report, run_result[0]) == []


def test_summary_numbers_match_pipeline_result(run_result):
    result = run_result[1]
    s = summarize(result)
    assert s["requirements"] == len(result.requirements)
    assert s["total_all"] == sum(len(c.findings) for c in result.checks) + len(result.set_findings)
    assert s["with_violations"] == sum(1 for c in result.checks if c.findings)
    assert s["unchecked"] == 0  # конвейер гарантирует проверку каждого требования
