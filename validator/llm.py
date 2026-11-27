"""Абстракция клиента LLM и реализация для Google Gemini API (Structured Outputs)."""

from abc import ABC, abstractmethod
import logging
import os
import time
from typing import TypeVar

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs) -> bool:
        return False
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_LLM_PROVIDER = "gemini"
RETRY_DELAYS_SECONDS = (2, 5, 15, 30)
MAX_JSON_PARSE_RETRIES = 2


class LLMError(Exception):
    """Ошибка взаимодействия с языковой моделью."""


class _HealthCheckResponse(BaseModel):
    """Служебная схема для проверки доступности модели (`--check-llm`)."""

    status: str = Field(description="Должно быть 'ok'")
    message: str = Field(description="Краткое подтверждение на русском")


class LLMClient(ABC):
    """Базовый интерфейс провайдера LLM для подменяемости реализации."""

    @abstractmethod
    def model_name(self) -> str:
        """Возвращает идентификатор используемой модели."""

    @abstractmethod
    def ask_json(
        self,
        prompt: str,
        schema: type[T],
        system: str | None = None,
    ) -> T:
        """Отправляет запрос в LLM и возвращает валидированный экземпляр Pydantic-схемы."""

    def check_connection(self) -> tuple[bool, str]:
        """Проверяет работоспособность модели тестовым structured-output запросом."""
        try:
            res = self.ask_json(
                prompt="Ответь строго в формате JSON со status='ok' и коротким приветствием.",
                schema=_HealthCheckResponse,
                system="Ты сервис проверки доступности API. Возвращай валидный JSON.",
            )
            return True, f"Модель '{self.model_name()}' доступна (ответ: {res.status} — {res.message})."
        except Exception as exc:
            return False, f"Ошибка проверки модели '{self.model_name()}': {exc}"


class GeminiClient(LLMClient):
    """Реализация LLMClient на базе официального SDK `google-genai`."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 0.1,
    ) -> None:
        load_dotenv()
        self._api_key = (api_key or os.getenv("GEMINI_API_KEY") or "").strip()
        self._model = (model or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL).strip()
        self._temperature = temperature

        if not self._api_key or self._api_key == "your_api_key_here":
            raise LLMError(
                "Не задан GEMINI_API_KEY. Скопируйте .env.example в .env и укажите ключ "
                "из Google AI Studio (https://aistudio.google.com/)."
            )

        if not (0.0 <= self._temperature <= 0.2):
            raise ValueError("Температура для детерминированного анализа должна быть в пределах 0.0..0.2")

        try:
            from google import genai
        except ImportError as exc:
            raise LLMError(
                "Библиотека 'google-genai' не установлена. Выполните: pip install google-genai"
            ) from exc

        self._client = genai.Client(api_key=self._api_key)

    def model_name(self) -> str:
        """Возвращает имя активной модели Gemini."""
        return self._model

    def list_available_models(self) -> list[str]:
        """Возвращает список доступных моделей Gemini на текущем API-ключе."""
        try:
            models: list[str] = []
            for m in self._client.models.list():
                name = getattr(m, "name", "") or ""
                if "gemini" in name.lower():
                    models.append(name.removeprefix("models/"))
            return sorted(set(models))
        except Exception as exc:
            logger.warning("Не удалось получить список моделей через models.list(): %s", exc)
            return []

    @staticmethod
    def _extract_status_code(exc: Exception) -> int | None:
        """Извлекает HTTP-код ошибки из исключения google-genai."""
        for attr in ("code", "status_code"):
            val = getattr(exc, attr, None)
            if isinstance(val, int):
                return val
        msg = str(exc)
        for code in (429, 500, 502, 503, 504, 400, 401, 403, 404):
            if str(code) in msg:
                return code
        return None

    def _call_with_http_retries(self, prompt: str, schema: type[BaseModel], system: str | None) -> str:
        """Выполняет запрос к Gemini API с экспоненциальными повторами при 429 и 5xx."""
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=self._temperature,
            response_mime_type="application/json",
            response_schema=schema,
            system_instruction=system,
        )

        last_exc: Exception | None = None
        total_attempts = len(RETRY_DELAYS_SECONDS) + 1

        for attempt in range(1, total_attempts + 1):
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=config,
                )
                text = getattr(response, "text", None)
                if not text or not text.strip():
                    raise LLMError(f"Модель '{self._model}' вернула пустой ответ.")
                return text
            except LLMError:
                raise
            except Exception as exc:
                last_exc = exc
                status_code = self._extract_status_code(exc)

                if status_code in (401, 403):
                    raise LLMError(
                        f"Ошибка авторизации Gemini API (HTTP {status_code}). "
                        "Проверьте корректность GEMINI_API_KEY в .env."
                    ) from exc

                if status_code == 404:
                    raise LLMError(
                        f"Модель '{self._model}' не найдена или недоступна (HTTP 404). "
                        f"Проверьте GEMINI_MODEL в .env (рекомендуется '{DEFAULT_GEMINI_MODEL}')."
                    ) from exc

                is_retryable = status_code == 429 or (
                    status_code is not None and 500 <= status_code < 600
                )
                if is_retryable and attempt < total_attempts:
                    delay = RETRY_DELAYS_SECONDS[attempt - 1]
                    reason = (
                        "превышен лимит запросов (429)"
                        if status_code == 429
                        else f"временная ошибка сервера ({status_code})"
                    )
                    logger.warning(
                        "Gemini API: %s. Попытка %d/%d, повтор через %d сек...",
                        reason,
                        attempt,
                        total_attempts,
                        delay,
                    )
                    time.sleep(delay)
                    continue

                if status_code == 429:
                    raise LLMError(
                        f"Исчерпан лимит запросов (429) к модели '{self._model}' после "
                        f"{total_attempts} попыток. Подождите минуту или проверьте дневную квоту AI Studio."
                    ) from exc

                raise LLMError(
                    f"Ошибка обращения к Gemini API ({self._model}): {exc}"
                ) from exc

        raise LLMError(f"Не удалось получить ответ от Gemini API: {last_exc}")

    def ask_json(
        self,
        prompt: str,
        schema: type[T],
        system: str | None = None,
    ) -> T:
        """Запрашивает структурированный JSON и парсит его в Pydantic-схему.

        При невалидном JSON выполняет до 2 уточняющих повторов.
        """
        current_prompt = prompt
        last_validation_error: Exception | None = None

        for parse_attempt in range(MAX_JSON_PARSE_RETRIES + 1):
            raw_json = self._call_with_http_retries(
                prompt=current_prompt,
                schema=schema,
                system=system,
            )
            cleaned_json = raw_json.strip()
            # На случай если модель всё же обернула JSON в ```json ... ```
            if cleaned_json.startswith("```"):
                lines = cleaned_json.splitlines()
                if len(lines) >= 3:
                    cleaned_json = "\n".join(lines[1:-1]).strip()

            try:
                return schema.model_validate_json(cleaned_json)
            except ValidationError as exc:
                last_validation_error = exc
                logger.warning(
                    "Невалидный JSON от модели (попытка разбора %d/%d): %s",
                    parse_attempt + 1,
                    MAX_JSON_PARSE_RETRIES + 1,
                    exc,
                )
                current_prompt = (
                    f"{prompt}\n\n"
                    "ВАЖНО: предыдущий ответ не прошёл валидацию JSON-схемы. "
                    "Верни только валидный JSON по схеме, без лишнего текста и markdown-разметки."
                )

        raise LLMError(
            f"Модель '{self._model}' не вернула валидный JSON по схеме {schema.__name__} "
            f"после {MAX_JSON_PARSE_RETRIES + 1} попыток: {last_validation_error}"
        )


class _StubUnsupportedProviderClient(LLMClient):
    """Заглушка для будущих провайдеров (OpenAI, Anthropic, Ollama)."""

    def __init__(self, provider: str) -> None:
        self._provider = provider

    def model_name(self) -> str:
        return f"{self._provider}:unconfigured"

    def ask_json(self, prompt: str, schema: type[T], system: str | None = None) -> T:
        raise NotImplementedError(
            f"Провайдер '{self._provider}' пока не реализован. "
            "Используйте LLM_PROVIDER=gemini в файле .env."
        )


_default_client: LLMClient | None = None


def create_llm_client(provider: str | None = None) -> LLMClient:
    """Фабрика клиента LLM по переменной окружения LLM_PROVIDER."""
    load_dotenv()
    selected = (provider or os.getenv("LLM_PROVIDER") or DEFAULT_LLM_PROVIDER).strip().lower()
    if selected == "gemini":
        return GeminiClient()
    if selected in ("openai", "anthropic", "ollama"):
        return _StubUnsupportedProviderClient(selected)
    raise LLMError(
        f"Неизвестный провайдер LLM_PROVIDER='{selected}'. Поддерживается: gemini."
    )


def get_llm_client() -> LLMClient:
    """Возвращает лениво инициализированный синглтон клиента LLM."""
    global _default_client
    if _default_client is None:
        _default_client = create_llm_client()
    return _default_client


def set_llm_client(client: LLMClient | None) -> None:
    """Позволяет подменить активный клиент LLM (удобно для тестов и заглушек)."""
    global _default_client
    _default_client = client


def model_name() -> str:
    """Возвращает имя текущей модели для паспорта прогона."""
    load_dotenv()
    if _default_client is not None:
        return _default_client.model_name()
    return (os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL).strip()


def ask_json(
    prompt: str,
    schema: type[T],
    system: str | None = None,
) -> T:
    """Публичный интерфейс A -> B: выполняет запрос в LLM и возвращает Pydantic-модель."""
    client = get_llm_client()
    return client.ask_json(prompt=prompt, schema=schema, system=system)


def check_llm() -> tuple[bool, str]:
    """Проверяет соединение с LLM и возвращает (успех, диагностическое сообщение)."""
    client = get_llm_client()
    ok, message = client.check_connection()
    if ok and isinstance(client, GeminiClient):
        available = client.list_available_models()
        if available:
            message += f"\nДоступные модели Gemini на ключе: {', '.join(available)}"
    return ok, message
