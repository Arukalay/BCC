"""Сквозной тест `python -m validator` без сети: loader -> pipeline -> report.

Модель Gemini подменена моком из tests/mock_llm.py через llm.set_llm_client, так что
проверяется вся цепочка, включая вызов модели (подпись ask_json со `system=`) и CLI.
"""
from pathlib import Path

import pytest

from tests import mock_llm
from tests.helpers import find_bad_quotes, parse_report_findings
from validator import cli, llm, loader

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "requirements.md"

pytestmark = pytest.mark.skipif(not DOC.exists(), reason="docs/requirements.md отсутствует")


class _MockClient(llm.LLMClient):
    """Провайдер LLM для тестов: отдаёт ответы мока, как настоящий GeminiClient."""

    def __init__(self) -> None:
        self.mock = mock_llm.MockLLM()

    def model_name(self) -> str:
        return "mock-model"

    def ask_json(self, prompt, schema, system=None):
        return self.mock(prompt, schema, system=system)


@pytest.fixture()
def client():
    c = _MockClient()
    llm.set_llm_client(c)
    try:
        yield c
    finally:
        llm.set_llm_client(None)


def test_cli_full_chain_on_real_document(tmp_path, client):
    out = tmp_path / "out" / "report.md"
    assert cli.main([str(DOC), str(out)]) == 0
    report = out.read_text(encoding="utf-8")

    for n, title in enumerate(["Паспорт прогона", "Как агент нарезал требования", "Проверка каждого требования",
                               "Проверка набора требований целиком", "Сводка", "Итог"], 1):
        assert f"## {n}. {title}" in report
    assert "requirements.md" in report and "mock-model" in report   # паспорт: файл и модель из llm.model_name()
    assert len(parse_report_findings(report)) > 0
    assert find_bad_quotes(report, DOC.read_text(encoding="utf-8")) == []
    assert len(client.mock.calls) <= 25                              # число вызовов модели не взрывается


def test_cli_full_chain_from_pdf(tmp_path, client):
    """PDF — обязательный вход по ТЗ: loader.load -> конвейер -> отчёт на PDF-образце (собран C из requirements.md)."""
    pdf = Path(__file__).resolve().parent / "data" / "requirements_sample.pdf"
    out = tmp_path / "report.md"
    assert cli.main([str(pdf), str(out)]) == 0
    report = out.read_text(encoding="utf-8")
    assert "## 6. Итог" in report and "requirements_sample.pdf" in report
    # цитаты в отчёте есть в тексте, который отдал loader (с маркерами страниц «--- Страница N ---»)
    assert find_bad_quotes(report, loader.load(str(pdf))) == []
