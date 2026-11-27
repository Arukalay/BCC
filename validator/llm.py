"""Клиенты LLM: Google Gemini и OpenAI со структурированным выводом (ответ строго по Pydantic-схеме).

Провайдер выбирается переменной LLM_PROVIDER (.env). Временные сбои (429 по минутному лимиту, 5xx,
таймаут) повторяются с паузами; ошибки, которые повтор не исправит (ключ, баланс, суточная квота,
модель), сразу поднимают LLMFatalError, и конвейер останавливается с понятным сообщением.
"""

from abc import ABC, abstractmethod
import logging
import os
import re
import time
from typing import TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"  # gemini-2.5-* отдают 404 новым ключам (проверено 27.11.2026)
DEFAULT_LLM_PROVIDER = "gemini"
RETRY_DELAYS_SECONDS = (2, 5, 15, 30)
MAX_JSON_PARSE_RETRIES = 2
REQUEST_TIMEOUT_SECONDS = 120  # без таймаута зависший запрос блокирует прогон навсегда
MAX_SERVER_RETRY_DELAY_SECONDS = 60  # дольше этого по подсказке сервера (retryDelay) не ждём


class LLMError(Exception):
    """Ошибка взаимодействия с языковой моделью."""


class LLMFatalError(LLMError):
    """Ошибка, которую повтор не исправит: суточная квота, неверный ключ, модель недоступна.

    Конвейер не должен ни повторять такой вызов, ни пытаться «обойти» его (например, делить пачку):
    нужно сразу остановить прогон и показать причину.
    """


def _is_daily_quota(message: str) -> bool:
    """429 из-за суточной квоты (ждать до сброса часы, повторы бесполезны)."""
    low = message.lower()
    return "perday" in low or "per day" in low or "requestsperday" in low


def _server_retry_delay(message: str) -> float | None:
    """Пауза, которую просит сервер («retryDelay': '34s'»), в секундах."""
    m = re.search(r"retryDelay\W+(\d+(?:\.\d+)?)s", message)
    return float(m.group(1)) if m else None


def _quota_limit(message: str) -> str:
    m = re.search(r"quotaValue\W+(\d+)", message) or re.search(r"limit:\s*(\d+)", message)
    return m.group(1) if m else "?"


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
            http_options={"timeout": int(REQUEST_TIMEOUT_SECONDS * 1000)},  # миллисекунды
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
                    raise LLMFatalError(
                        f"Ошибка авторизации Gemini API (HTTP {status_code}). "
                        "Проверьте корректность GEMINI_API_KEY в .env."
                    ) from exc

                if status_code == 404:
                    raise LLMFatalError(
                        f"Модель '{self._model}' не найдена или недоступна (HTTP 404; для новых ключей "
                        f"часть старых моделей отключена). Проверьте GEMINI_MODEL в .env: список доступных "
                        f"моделей покажет `python -m validator --check-llm` на рабочей модели."
                    ) from exc

                if status_code == 429 and _is_daily_quota(str(exc)):
                    wait = _server_retry_delay(str(exc))
                    hours = f" (сброс примерно через {wait / 3600:.0f} ч)" if wait else ""
                    raise LLMFatalError(
                        f"Исчерпана суточная квота бесплатного тарифа для модели '{self._model}': "
                        f"{_quota_limit(str(exc))} запросов в сутки на проект{hours}. Повтор не поможет. "
                        "Укажите другую модель в GEMINI_MODEL (.env) или подождите до сброса квоты."
                    ) from exc

                is_retryable = status_code == 429 or (
                    status_code is not None and 500 <= status_code < 600
                )
                if is_retryable and attempt < total_attempts:
                    delay = RETRY_DELAYS_SECONDS[attempt - 1]
                    asked = _server_retry_delay(str(exc))  # сервер сам знает, когда минутный лимит сбросится
                    if asked is not None:
                        delay = max(delay, min(asked, MAX_SERVER_RETRY_DELAY_SECONDS))
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
                    raise LLMFatalError(
                        f"Исчерпан лимит запросов (429) к модели '{self._model}' после "
                        f"{total_attempts} попыток. Подождите минуту или проверьте дневную квоту AI Studio."
                    ) from exc

                raise LLMFatalError(
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


_OPENAI_NOT_CHAT = ("embedding", "tts", "whisper", "audio", "image", "realtime", "transcribe",
                    "moderation", "dall-e", "sora", "search", "computer-use")


class OpenAIClient(LLMClient):
    """Реализация LLMClient на базе официального SDK `openai` (Responses API, Structured Outputs).

    Настройки (.env): OPENAI_API_KEY, OPENAI_MODEL (обязательна: названия моделей быстро меняются,
    список доступных на ключе печатает `python -m validator --check-llm`), OPENAI_TEMPERATURE (по
    умолчанию не передаётся: часть моделей не принимает этот параметр).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> None:
        load_dotenv()
        self._api_key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
        self._model = (model or os.getenv("OPENAI_MODEL") or "").strip()
        raw_temp = temperature if temperature is not None else os.getenv("OPENAI_TEMPERATURE", "").strip()
        self._temperature = float(raw_temp) if raw_temp not in (None, "") else None

        if not self._api_key or self._api_key == "your_api_key_here":
            raise LLMError(
                "Не задан OPENAI_API_KEY. Укажите ключ из https://platform.openai.com/api-keys в файле .env "
                "(и LLM_PROVIDER=openai)."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError("Библиотека 'openai' не установлена. Выполните: pip install openai") from exc

        # max_retries=0: повторы делаем сами (см. _call_with_retries), иначе паузы SDK перемножатся с нашими
        self._client = OpenAI(api_key=self._api_key, timeout=REQUEST_TIMEOUT_SECONDS, max_retries=0)

    def model_name(self) -> str:
        return self._model or "openai:модель-не-задана"

    def list_available_models(self) -> list[str]:
        """Идентификаторы текстовых моделей, доступных на ключе (без embedding, аудио, изображений)."""
        try:
            ids = [m.id for m in self._client.models.list()]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Не удалось получить список моделей OpenAI: %s", exc)
            return []
        return sorted(i for i in set(ids) if not any(bad in i.lower() for bad in _OPENAI_NOT_CHAT))

    @staticmethod
    def _error_code(exc: Exception) -> str:
        body = getattr(exc, "body", None)
        inner = body.get("error", body) if isinstance(body, dict) else {}
        return str(getattr(exc, "code", None) or (inner.get("code") if isinstance(inner, dict) else "") or "")

    @staticmethod
    def _retry_after(exc: Exception) -> float | None:
        headers = getattr(getattr(exc, "response", None), "headers", None)
        try:
            value = headers.get("retry-after") if headers is not None else None
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _call_with_retries(self, prompt: str, schema: type[BaseModel], system: str | None):
        """Запрос с повторами при временных ошибках; всё, что повтор не исправит, — LLMFatalError."""
        import openai

        kwargs: dict = {"model": self._model, "input": prompt, "text_format": schema}
        if system:
            kwargs["instructions"] = system
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature

        total_attempts = len(RETRY_DELAYS_SECONDS) + 1
        for attempt in range(1, total_attempts + 1):
            try:
                return self._client.responses.parse(**kwargs)
            except ValidationError:
                raise
            except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
                raise LLMFatalError(
                    f"Ошибка авторизации OpenAI (HTTP {exc.status_code}). Проверьте OPENAI_API_KEY в .env."
                ) from exc
            except openai.NotFoundError as exc:
                raise LLMFatalError(
                    f"Модель '{self._model}' не найдена или недоступна на этом ключе. Список доступных моделей "
                    "покажет `python -m validator --check-llm`; выберите одну и укажите в OPENAI_MODEL (.env)."
                ) from exc
            except openai.BadRequestError as exc:
                raise LLMFatalError(f"OpenAI отклонил запрос к модели '{self._model}': {exc.message}") from exc
            except openai.RateLimitError as exc:
                if self._error_code(exc) == "insufficient_quota":
                    raise LLMFatalError(
                        "Недостаточно средств или квоты на аккаунте OpenAI (insufficient_quota). Повтор не поможет: "
                        "пополните баланс на https://platform.openai.com/settings/organization/billing."
                    ) from exc
                reason, delay = "превышен лимит запросов (429)", self._retry_after(exc)
                error: Exception = exc
            except (openai.APITimeoutError, openai.APIConnectionError) as exc:
                reason, delay, error = f"нет ответа или связи ({type(exc).__name__})", None, exc
            except openai.APIStatusError as exc:
                if exc.status_code < 500:
                    raise LLMFatalError(f"Ошибка OpenAI (HTTP {exc.status_code}, {self._model}): {exc.message}") from exc
                reason, delay, error = f"временная ошибка сервера ({exc.status_code})", None, exc

            if attempt >= total_attempts:
                raise LLMFatalError(
                    f"OpenAI недоступен: {reason}, {total_attempts} попыток исчерпано ({self._model}): {error}"
                ) from error
            pause = RETRY_DELAYS_SECONDS[attempt - 1]
            if delay is not None:
                pause = max(pause, min(delay, MAX_SERVER_RETRY_DELAY_SECONDS))
            logger.warning("OpenAI API: %s. Попытка %d/%d, повтор через %d сек...", reason, attempt, total_attempts, pause)
            time.sleep(pause)
        raise LLMFatalError("Не удалось получить ответ от OpenAI API.")  # pragma: no cover

    @staticmethod
    def _why_empty(resp) -> str:
        """Почему в ответе нет разобранного объекта (отказ модели, обрыв по длине)."""
        status = getattr(resp, "status", None)
        details = getattr(resp, "incomplete_details", None)
        if status == "incomplete":
            return f"ответ оборван ({getattr(details, 'reason', 'неизвестная причина')})"
        for item in getattr(resp, "output", None) or []:
            for part in getattr(item, "content", None) or []:
                if getattr(part, "type", "") == "refusal":
                    return f"модель отказалась отвечать: {getattr(part, 'refusal', '')}"
        return "в ответе нет структурированных данных"

    def ask_json(self, prompt: str, schema: type[T], system: str | None = None) -> T:
        if not self._model:
            raise LLMFatalError(
                "Не задана модель OPENAI_MODEL в .env. Список доступных на ключе моделей покажет "
                "`python -m validator --check-llm`."
            )
        current_prompt = prompt
        last_problem = ""
        for parse_attempt in range(MAX_JSON_PARSE_RETRIES + 1):
            try:
                resp = self._call_with_retries(current_prompt, schema, system)
            except ValidationError as exc:
                last_problem = str(exc)
            else:
                parsed = getattr(resp, "output_parsed", None)
                if parsed is not None:
                    return parsed
                last_problem = self._why_empty(resp)
                if getattr(resp, "status", None) == "incomplete":
                    break  # повтор того же запроса оборвётся так же; пусть конвейер решает (например, делит пачку)
            logger.warning("OpenAI: ответ не разобран (попытка %d/%d): %s", parse_attempt + 1,
                           MAX_JSON_PARSE_RETRIES + 1, last_problem)
            current_prompt = (
                f"{prompt}\n\nВАЖНО: предыдущий ответ не прошёл проверку схемы. "
                "Верни только валидный JSON по схеме, без лишнего текста и markdown-разметки."
            )
        raise LLMError(f"Модель '{self._model}' не вернула валидный ответ по схеме {schema.__name__}: {last_problem}")


_default_client: LLMClient | None = None
PROVIDERS: dict[str, type[LLMClient]] = {"gemini": GeminiClient, "openai": OpenAIClient}


def create_llm_client(provider: str | None = None) -> LLMClient:
    """Фабрика клиента LLM по переменной окружения LLM_PROVIDER."""
    load_dotenv()
    selected = (provider or os.getenv("LLM_PROVIDER") or DEFAULT_LLM_PROVIDER).strip().lower()
    if selected not in PROVIDERS:
        raise LLMError(
            f"Неизвестный провайдер LLM_PROVIDER='{selected}'. Поддерживаются: {', '.join(PROVIDERS)}."
        )
    return PROVIDERS[selected]()


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
    if (os.getenv("LLM_PROVIDER") or DEFAULT_LLM_PROVIDER).strip().lower() == "openai":
        return (os.getenv("OPENAI_MODEL") or "openai:модель-не-задана").strip()
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
    if isinstance(client, OpenAIClient):  # список нужен и при ошибке: так выбирают значение OPENAI_MODEL
        available = client.list_available_models()
        if available:
            message += f"\nДоступные модели OpenAI на ключе: {', '.join(available)}"
    return ok, message
