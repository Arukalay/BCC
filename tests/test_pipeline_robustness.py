"""Устойчивость конвейера и точность цитат (без LLM)."""
from pathlib import Path

import pytest

from tests import mock_llm
from validator import pipeline
from validator.llm import LLMError, LLMFatalError
from validator.report import build
from validator.schemas import RunMeta

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "requirements.md"
pytestmark = pytest.mark.skipif(not DOC.exists(), reason="docs/requirements.md отсутствует")


@pytest.fixture(scope="module")
def doc():
    return pipeline._Doc(DOC.read_text(encoding="utf-8"))


def test_quote_keeps_original_dashes_and_guillemets(doc):
    """Модель пишет обычные кавычки и дефис, а в отчёт идёт текст документа: его находит поиск (Ctrl+F)."""
    quote = doc.resolve('поле "Порог, ₸"').quote
    assert quote == "поле «Порог, ₸»"
    quote = doc.resolve("от 1 000 до 1 000 000 ₸ включительно").quote
    assert quote in doc.text
    quote = doc.resolve("Если результат 200 или 201 - успешный.").quote
    assert quote == "Если результат 200 или 201 — успешный."


def test_quote_spanning_lines_is_flattened_but_original(doc):
    quote = doc.resolve("Ответ в случае, если параметры запроса заполнены неверно.").quote
    assert quote == "Ответ в случае, если параметры запроса заполнены неверно."
    two = doc.resolve("На карту приходится максимум один такой push || независимо от истории предыдущей настройки").quote
    assert " || " in two and all(part in doc.text for part in two.split(" || "))


@pytest.mark.parametrize("raw,expected", [
    ("R-01", "R-01"), ("R-1", "R-01"), ("r-07", "R-07"), ("R-03 (Ф-1)", "R-03"), (" R 12 ", "R-12"), ("R-105", "R-105"),
])
def test_requirement_ids_from_model_are_normalized(raw, expected):
    assert pipeline._norm_id(raw) == expected


class _WrongIds(mock_llm.MockLLM):
    """Мок, который в ответе проверки пачки пишет идентификаторы без ведущего нуля (R-1 вместо R-01)."""

    def _BatchChecks(self, prompt):
        res = super()._BatchChecks(prompt)
        for c in res.checks:
            c.req_id = c.req_id.replace("R-0", "R-")
        return res


def test_ids_without_leading_zero_are_accepted():
    result = pipeline.run(DOC.read_text(encoding="utf-8"), ask_json=_WrongIds())
    assert [c.req_id for c in result.checks] == [r.report_id for r in result.requirements]


class _NeverChecks(mock_llm.MockLLM):
    """Мок, который не может проверить ни одно требование (ответ без нужных id) и не может исправить цитаты."""

    def _BatchChecks(self, prompt):
        from validator.pipeline import BatchChecks
        return BatchChecks(checks=[])

    def _QuoteFixes(self, prompt):
        raise LLMError("модель вернула невалидный JSON")


def test_unchecked_requirements_do_not_kill_the_run():
    text = DOC.read_text(encoding="utf-8")
    result = pipeline.run(text, ask_json=_NeverChecks())
    assert result.requirements and result.checks == []        # прогон дошёл до конца
    report = build(RunMeta(document_name="d.md", date="2026-11-27", model="m", agent_version="v"), result)
    assert "проверка не выполнена" in report and "## 6. Итог" in report


def test_fatal_error_during_quote_fix_still_stops_the_run():
    class FatalFix(mock_llm.MockLLM):
        def _QuoteFixes(self, prompt):
            raise LLMFatalError("квота")

    with pytest.raises(LLMFatalError):
        pipeline.run(DOC.read_text(encoding="utf-8"), ask_json=FatalFix())
