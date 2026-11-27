"""Базовые тесты ядра (loader, schemas, llm, cli) для проверки инфраструктуры."""

from pathlib import Path
import sys
import types
import pytest

from validator import cli, llm, loader
from validator.loader import DocumentLoadError
from validator.schemas import AnalysisResult


def test_normalize_quotes_dashes_and_spaces():
    raw = "  «Уведомление»  —  это\n\tpush-сообщение   ‘раз в месяц’  "
    assert loader.normalize(raw) == '"Уведомление" - это push-сообщение \'раз в месяц\''


def test_load_markdown_and_unknown_extension(tmp_path: Path):
    md_file = tmp_path / "req.md"
    md_file.write_text("# Заголовок\nТекст требования.", encoding="utf-8")
    assert "Заголовок" in loader.load(str(md_file))

    bad_file = tmp_path / "req.xlsx"
    bad_file.write_text("data", encoding="utf-8")
    with pytest.raises(ValueError, match="Поддерживаемые форматы"):
        loader.load(str(bad_file))


def test_load_docx_preserves_headings_and_tables(tmp_path: Path):
    docx = pytest.importorskip("docx")
    doc = docx.Document()
    doc.add_heading("Часть A", level=1)
    doc.add_paragraph("Ф-1: Порог от 1 000 до 1 000 000 тенге.")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Параметр"
    table.cell(0, 1).text = "Тип"
    table.cell(1, 0).text = "cardId"
    table.cell(1, 1).text = "string"

    docx_path = tmp_path / "sample.docx"
    doc.save(str(docx_path))

    loaded = loader.load(str(docx_path))
    assert "# Часть A" in loaded
    assert "| Параметр | Тип |" in loaded
    assert "| cardId | string |" in loaded


def test_load_pdf_empty_raises_ocr_hint(tmp_path: Path):
    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    pdf_path = tmp_path / "scan.pdf"
    with pdf_path.open("wb") as f:
        writer.write(f)

    with pytest.raises(DocumentLoadError, match="OCR"):
        loader.load(str(pdf_path))


def test_llm_ask_json_provider_interface():
    class FakeClient(llm.LLMClient):
        def model_name(self) -> str:
            return "fake-gemini"

        def ask_json(self, prompt, schema, system=None):
            return schema.model_validate({"requirements": [], "checks": [], "set_findings": []})

    llm.set_llm_client(FakeClient())
    try:
        res = llm.ask_json("test", AnalysisResult)
        assert isinstance(res, AnalysisResult)
        assert llm.model_name() == "fake-gemini"
    finally:
        llm.set_llm_client(None)


def test_gemini_client_parse_retry_and_429_backoff(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    monkeypatch.setattr("validator.llm.RETRY_DELAYS_SECONDS", (0, 0))

    # Подменяем google.genai в sys.modules для изолированного unit-теста без сети
    fake_google = types.ModuleType("google")
    fake_genai = types.ModuleType("google.genai")
    fake_types = types.ModuleType("google.genai.types")

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeResp:
        def __init__(self, text):
            self.text = text

    attempts = {"count": 0}

    class FakeModels:
        def generate_content(self, model, contents, config):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("429 RESOURCE_EXHAUSTED")
            if attempts["count"] == 2:
                return FakeResp("not a valid json at all")
            return FakeResp('{"requirements": [], "checks": [], "set_findings": []}')

    class FakeGenaiClient:
        def __init__(self, api_key):
            self.models = FakeModels()

    fake_types.GenerateContentConfig = FakeGenerateContentConfig
    fake_genai.Client = FakeGenaiClient
    fake_genai.types = fake_types
    fake_google.genai = fake_genai

    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)

    client = llm.GeminiClient()
    result = client.ask_json("Проверь требования", AnalysisResult)
    assert isinstance(result, AnalysisResult)
    assert attempts["count"] == 3


def test_cli_end_to_end_smoke(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    inp = tmp_path / "reqs.md"
    out = tmp_path / "nested" / "report.md"
    inp.write_text("# Требования\nФ-1: Система отправляет push.", encoding="utf-8")

    code = cli.main([str(inp), str(out)])
    assert code == 0
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "reqs.md" in content
    assert "gemini-2.5-flash" in content
