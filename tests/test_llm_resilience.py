"""Устойчивость к сбоям модели: квоты, повторы, таймаут, быстрый отказ конвейера (без сети).

Тексты ошибок взяты из реальных ответов Gemini API (27.11.2026): суточная квота бесплатного
тарифа у новых моделей — 20 запросов в сутки, и повторять такой запрос бесполезно.
"""
import sys
import types
from pathlib import Path

import pytest

from validator import cli, llm, pipeline
from validator.llm import LLMError, LLMFatalError
from validator.schemas import AnalysisResult

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "requirements.md"

DAILY_429 = (
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota', "
    "'status': 'RESOURCE_EXHAUSTED', 'details': [{'violations': [{'quotaMetric': "
    "'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': "
    "'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaValue': '20'}]}, "
    "{'retryDelay': '65314s'}]}}"
)
MINUTE_429 = (
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'details': [{'violations': [{'quotaId': "
    "'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', 'quotaValue': '10'}]}, {'retryDelay': '3s'}]}}"
)
OVERLOAD_503 = "503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently experiencing high demand.'}}"
OK_JSON = '{"requirements": [], "checks": [], "set_findings": []}'


@pytest.fixture()
def fake_gemini(monkeypatch):
    """Подменяет google.genai: ответы берутся из списка (исключение — бросается, строка — возвращается)."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.setattr(llm, "RETRY_DELAYS_SECONDS", (0, 0))
    slept: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: slept.append(s))

    state = types.SimpleNamespace(script=[], calls=0, configs=[], slept=slept)

    class Config:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            state.configs.append(kwargs)

    class Resp:
        def __init__(self, text):
            self.text = text

    class Models:
        def generate_content(self, model, contents, config):
            state.calls += 1
            item = state.script[min(state.calls - 1, len(state.script) - 1)]
            if isinstance(item, Exception):
                raise item
            return Resp(item)

    class Client:
        def __init__(self, api_key):
            self.models = Models()

    google, genai, gtypes = types.ModuleType("google"), types.ModuleType("google.genai"), types.ModuleType("google.genai.types")
    gtypes.GenerateContentConfig = Config
    genai.Client, genai.types = Client, gtypes
    google.genai = genai
    for name, mod in (("google", google), ("google.genai", genai), ("google.genai.types", gtypes)):
        monkeypatch.setitem(sys.modules, name, mod)
    return state


def test_daily_quota_fails_immediately_without_retries(fake_gemini):
    fake_gemini.script = [RuntimeError(DAILY_429)]
    with pytest.raises(LLMFatalError) as info:
        llm.GeminiClient().ask_json("q", AnalysisResult)
    assert fake_gemini.calls == 1 and fake_gemini.slept == []   # ни одного повтора и ни одной паузы
    text = str(info.value)
    assert "суточная квота" in text and "test-model" in text and "20 запросов" in text
    assert "GEMINI_MODEL" in text and "18 ч" in text             # что делать и когда сбросится квота


def test_per_minute_429_is_retried_and_respects_server_delay(fake_gemini):
    fake_gemini.script = [RuntimeError(MINUTE_429), OK_JSON]
    assert isinstance(llm.GeminiClient().ask_json("q", AnalysisResult), AnalysisResult)
    assert fake_gemini.calls == 2
    assert fake_gemini.slept == [3.0]                            # сервер попросил 3 с, а в графике было 0


def test_server_retry_delay_is_capped(fake_gemini):
    fake_gemini.script = [RuntimeError(MINUTE_429.replace("'3s'", "'900s'")), OK_JSON]
    llm.GeminiClient().ask_json("q", AnalysisResult)
    assert fake_gemini.slept == [llm.MAX_SERVER_RETRY_DELAY_SECONDS]


def test_overload_is_retried_then_becomes_fatal(fake_gemini):
    fake_gemini.script = [RuntimeError(OVERLOAD_503)]
    with pytest.raises(LLMFatalError):
        llm.GeminiClient().ask_json("q", AnalysisResult)
    assert fake_gemini.calls == len(llm.RETRY_DELAYS_SECONDS) + 1


def test_request_has_timeout(fake_gemini):
    fake_gemini.script = [OK_JSON]
    llm.GeminiClient().ask_json("q", AnalysisResult)
    assert fake_gemini.configs[0]["http_options"] == {"timeout": llm.REQUEST_TIMEOUT_SECONDS * 1000}


def test_fatal_error_is_an_llm_error_and_cli_reports_it(tmp_path, monkeypatch, capsys):
    assert issubclass(LLMFatalError, LLMError)

    def boom(text):
        raise LLMFatalError("Исчерпана суточная квота")

    monkeypatch.setattr(cli.pipeline, "run", boom)
    src = tmp_path / "r.md"
    src.write_text("# Док\nТекст.", encoding="utf-8")
    assert cli.main([str(src), str(tmp_path / "out.md")]) == 3
    assert "Исчерпана суточная квота" in capsys.readouterr().err


@pytest.mark.skipif(not DOC.exists(), reason="docs/requirements.md отсутствует")
def test_pipeline_stops_at_first_fatal_error_instead_of_splitting_batches():
    """Раньше любая ошибка делила пачку пополам: при исчерпанной квоте это были сотни вызовов по минуте."""
    from tests import mock_llm
    mock = mock_llm.MockLLM()

    def ask(prompt, schema, system=None):
        if schema.__name__ == "BatchChecks":
            mock.calls.append("BatchChecks")
            raise LLMFatalError("квота")
        return mock(prompt, schema, system=system)

    with pytest.raises(LLMFatalError):
        pipeline.run(DOC.read_text(encoding="utf-8"), ask_json=ask)
    assert mock.calls.count("BatchChecks") <= pipeline.MAX_WORKERS * 2   # раньше ≈ 2 × число требований
    assert "SetResult" not in mock.calls                                 # до проверки набора не дошли
