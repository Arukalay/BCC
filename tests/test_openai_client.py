"""Клиент OpenAI (Responses API + Structured Outputs) без сети: сетевой вызов подменён, ошибки настоящие из SDK."""
import types

import openai
import pytest

try:  # openai 3.x построен на httpx2, более ранние версии на httpx
    import httpx2 as httpx
except ImportError:  # pragma: no cover
    import httpx

from validator import llm, pipeline
from validator.llm import LLMError, LLMFatalError, OpenAIClient
from validator.schemas import AnalysisResult

EMPTY = AnalysisResult(requirements=[], checks=[], set_findings=[])


def _resp(status: int, headers: dict | None = None) -> httpx.Response:
    return httpx.Response(status, headers=headers or {}, request=httpx.Request("POST", "https://api.openai.com/v1/responses"))


def _err(cls, status: int, message: str = "m", body: dict | None = None, headers: dict | None = None):
    return cls(message, response=_resp(status, headers), body=body)


@pytest.fixture()
def client(monkeypatch):
    """OpenAIClient с поддельным `responses.parse`: скрипт из результатов (исключение бросается, остальное возвращается)."""
    monkeypatch.setattr(llm, "RETRY_DELAYS_SECONDS", (0, 0))
    slept: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: slept.append(s))
    c = OpenAIClient(api_key="test-key", model="test-model")
    state = types.SimpleNamespace(script=[], calls=[], slept=slept, models=[])

    def parse(**kwargs):
        state.calls.append(kwargs)
        item = state.script[min(len(state.calls) - 1, len(state.script) - 1)]
        if isinstance(item, Exception):
            raise item
        return item

    c._client = types.SimpleNamespace(
        responses=types.SimpleNamespace(parse=parse),
        models=types.SimpleNamespace(list=lambda: [types.SimpleNamespace(id=i) for i in state.models]),
    )
    c.state = state
    return c


def ok(obj=EMPTY):
    return types.SimpleNamespace(output_parsed=obj, status="completed", output=[])


def test_returns_parsed_object_and_sends_system_prompt(client):
    client.state.script = [ok()]
    assert client.ask_json("вопрос", AnalysisResult, system="правила") is EMPTY
    call = client.state.calls[0]
    assert call["model"] == "test-model" and call["input"] == "вопрос"
    assert call["instructions"] == "правила" and call["text_format"] is AnalysisResult
    assert "temperature" not in call          # по умолчанию не передаётся: часть моделей его не принимает


def test_temperature_only_when_configured(monkeypatch):
    monkeypatch.setenv("OPENAI_TEMPERATURE", "0.2")
    c = OpenAIClient(api_key="k", model="m")
    assert c._temperature == 0.2


def test_insufficient_quota_is_fatal_without_retries(client):
    client.state.script = [_err(openai.RateLimitError, 429, body={"code": "insufficient_quota", "message": "x"})]
    with pytest.raises(LLMFatalError) as info:
        client.ask_json("q", AnalysisResult)
    assert len(client.state.calls) == 1 and client.state.slept == []
    assert "баланс" in str(info.value) and "insufficient_quota" in str(info.value)


def test_rate_limit_is_retried_with_server_delay(client):
    client.state.script = [_err(openai.RateLimitError, 429, body={"code": "rate_limit_exceeded"}, headers={"retry-after": "3"}), ok()]
    assert client.ask_json("q", AnalysisResult) is EMPTY
    assert len(client.state.calls) == 2 and client.state.slept == [3.0]


def test_server_errors_and_timeouts_are_retried_then_fatal(client):
    client.state.script = [_err(openai.InternalServerError, 503)]
    with pytest.raises(LLMFatalError):
        client.ask_json("q", AnalysisResult)
    assert len(client.state.calls) == len(llm.RETRY_DELAYS_SECONDS) + 1

    client.state.calls.clear()
    client.state.script = [openai.APITimeoutError(request=httpx.Request("POST", "https://x")), ok()]
    assert client.ask_json("q", AnalysisResult) is EMPTY
    assert len(client.state.calls) == 2


@pytest.mark.parametrize("cls,status,needle", [
    (openai.AuthenticationError, 401, "OPENAI_API_KEY"),
    (openai.PermissionDeniedError, 403, "OPENAI_API_KEY"),
    (openai.NotFoundError, 404, "OPENAI_MODEL"),
    (openai.BadRequestError, 400, "отклонил"),
])
def test_unrecoverable_http_errors_are_fatal_and_not_retried(client, cls, status, needle):
    client.state.script = [_err(cls, status, message="Unsupported parameter: temperature")]
    with pytest.raises(LLMFatalError) as info:
        client.ask_json("q", AnalysisResult)
    assert len(client.state.calls) == 1 and needle in str(info.value)


def test_refusal_is_a_plain_error_so_pipeline_may_split_the_batch(client):
    refusal = types.SimpleNamespace(
        output_parsed=None, status="completed",
        output=[types.SimpleNamespace(content=[types.SimpleNamespace(type="refusal", refusal="нельзя")])])
    client.state.script = [refusal]
    with pytest.raises(LLMError) as info:
        client.ask_json("q", AnalysisResult)
    assert not isinstance(info.value, LLMFatalError) and "отказалась" in str(info.value)
    assert len(client.state.calls) == llm.MAX_JSON_PARSE_RETRIES + 1
    assert "ВАЖНО" in client.state.calls[-1]["input"]       # повторы идут с напоминанием про схему


def test_truncated_answer_is_not_retried(client):
    cut = types.SimpleNamespace(output_parsed=None, status="incomplete",
                                incomplete_details=types.SimpleNamespace(reason="max_output_tokens"), output=[])
    client.state.script = [cut]
    with pytest.raises(LLMError) as info:
        client.ask_json("q", AnalysisResult)
    assert len(client.state.calls) == 1 and "max_output_tokens" in str(info.value)


def test_missing_model_and_key_fail_with_instructions(monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setattr(llm, "load_dotenv", lambda *a, **k: False)
    with pytest.raises(LLMFatalError, match="OPENAI_MODEL"):
        OpenAIClient(api_key="k").ask_json("q", AnalysisResult)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        OpenAIClient()


def test_model_list_hides_non_text_models(client):
    client.state.models = ["gpt-x", "text-embedding-3", "gpt-x-mini", "whisper-1", "gpt-image-2", "gpt-x"]
    assert client.list_available_models() == ["gpt-x", "gpt-x-mini"]


def test_factory_and_passport_use_openai(monkeypatch):
    monkeypatch.setattr(llm, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_MODEL", "my-model")
    llm.set_llm_client(None)
    try:
        assert isinstance(llm.create_llm_client(), OpenAIClient)
        assert llm.model_name() == "my-model"
    finally:
        llm.set_llm_client(None)


def test_check_llm_prints_models_even_when_model_is_not_set(client, monkeypatch):
    """Так пользователь выбирает OPENAI_MODEL: список должен появиться и при ошибке проверки."""
    client._model = ""
    client.state.models = ["gpt-a", "gpt-b"]
    llm.set_llm_client(client)
    try:
        ok_, message = llm.check_llm()
    finally:
        llm.set_llm_client(None)
    assert not ok_ and "OPENAI_MODEL" in message and "gpt-a, gpt-b" in message


def test_all_pipeline_schemas_fit_openai_strict_mode():
    """Strict mode: все поля обязательные, без значений по умолчанию. Если схема изменится, тест это поймает."""
    to_strict = pytest.importorskip("openai.lib._pydantic").to_strict_json_schema
    for cls in (llm._HealthCheckResponse, AnalysisResult, pipeline.SlicingResult, pipeline.BatchChecks,
                pipeline.SetResult, pipeline.QuoteFixes):
        schema = to_strict(cls)
        assert schema["additionalProperties"] is False and set(schema["required"]) == set(schema["properties"]), cls
