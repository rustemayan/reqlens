"""Общий клиент LLM: ретраи, смена модели при исчерпании квоты, кэш, валидация JSON.

Провайдер (Gemini / OpenAI-совместимый / Anthropic) реализует только один
метод `generate`. Всё, что делает прогон надёжным, живёт здесь:

* экспоненциальная пауза и повтор при 429/5xx (с учётом retryDelay от API);
* переход на запасную модель, если у текущей кончилась дневная квота или её нет;
* проверка ответа по JSON Schema и повторный запрос при невалидном JSON;
* сигнал об обрезке ответа (пайплайн дробит пакет требований пополам);
* дисковый кэш: повторный прогон не тратит квоту и даёт тот же результат.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .jsonutil import JSONExtractError, extract_json, validate

log = logging.getLogger("reqlens.llm")


class LLMError(Exception):
    """Ошибка вызова модели (этап можно пропустить)."""


class LLMFatalError(LLMError):
    """Продолжать прогон бессмысленно: следующие этапы упадут так же."""


class LLMAuthError(LLMFatalError):
    """Ключ неверен / нет доступа / регион не поддерживается."""


class LLMConnectionError(LLMFatalError):
    """Нет связи с API (сеть, DNS, прокси, TLS-сертификат)."""


class LLMTruncated(LLMError):
    """Ответ обрезан по лимиту токенов."""


class ModelUnavailable(LLMError):
    """Модель недоступна (нет такой модели, исчерпана дневная квота) — пробуем следующую."""


class RetryableError(LLMError):
    def __init__(self, message: str, retry_after: Optional[float] = None, network: bool = False):
        super().__init__(message)
        self.retry_after = retry_after
        self.network = network


def map_transport_error(status: int, message: str) -> Optional[LLMError]:
    """Общая обработка ошибок транспорта для всех провайдеров."""
    if status == -2:
        return LLMConnectionError(
            f"{message}. Если вы за корпоративным прокси или антивирусом с проверкой HTTPS, "
            "укажите путь к корневому сертификату в SSL_CERT_FILE."
        )
    if status == 0:
        return RetryableError(message, network=True)
    return None


@dataclass
class RawResponse:
    text: str
    finish_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    truncated: bool = False


@dataclass
class CallRecord:
    stage: str
    model: str
    cached: bool
    seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    attempts: int = 1
    ok: bool = True
    note: str = ""


class Provider(ABC):
    name = "provider"

    def __init__(self, settings):
        self.settings = settings

    @abstractmethod
    def generate(self, model: str, system: str, user: str, schema: Optional[dict]) -> RawResponse:
        ...


@dataclass
class LLMClient:
    provider: Provider
    settings: Any
    records: List[CallRecord] = field(default_factory=list)
    used_models: List[str] = field(default_factory=list)
    _last_call: float = 0.0
    _active: int = 0

    @property
    def models(self) -> List[str]:
        return [self.settings.model] + list(self.settings.fallback_models)

    @property
    def model(self) -> str:
        return self.models[min(self._active, len(self.models) - 1)]

    # ----------------------------------------------------------------- cache
    def _cache_key(self, system: str, user: str, schema: Optional[dict]) -> str:
        blob = json.dumps(
            {
                "provider": self.settings.provider,
                "model": self.settings.model,
                "system": system,
                "user": user,
                "schema": schema,
                "v": 1,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return Path(self.settings.cache_dir) / f"{key[:2]}" / f"{key}.json"

    def _cache_get(self, key: str) -> Optional[dict]:
        if not self.settings.use_cache:
            return None
        p = self._cache_path(key)
        if p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
        return None

    def _cache_put(self, key: str, record: dict) -> None:
        if not self.settings.use_cache:
            return
        p = self._cache_path(key)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError as exc:
            log.warning("не удалось записать кэш: %s", exc)

    # ------------------------------------------------------------------ call
    def complete_json(self, stage: str, system: str, user: str, schema: Optional[dict]) -> Any:
        key = self._cache_key(system, user, schema)
        hit = self._cache_get(key)
        if hit is not None and hit.get("truncated"):
            # в прошлый раз ответ на этот запрос обрезался — не тратим квоту повторно
            raise LLMTruncated("ответ обрезан (из кэша)")
        if hit is not None and "data" in hit:
            self.records.append(CallRecord(stage, hit.get("model", "?"), True, 0.0, note="кэш"))
            self._note_model(hit.get("model", self.model))
            return hit["data"]

        last_exc: Optional[Exception] = None
        while self._active < len(self.models):
            model = self.models[self._active]
            t0 = time.monotonic()
            try:
                data, raw, attempts = self._call_model(model, system, user, schema)
            except LLMTruncated:
                self.records.append(CallRecord(stage, model, False, time.monotonic() - t0, ok=False, note="обрезан"))
                self._cache_put(key, {"model": model, "stage": stage, "truncated": True})
                raise
            except ModelUnavailable as exc:
                log.warning("модель %s недоступна (%s) — переключаюсь на запасную", model, exc)
                self.records.append(CallRecord(stage, model, False, time.monotonic() - t0, ok=False, note=str(exc)[:200]))
                last_exc = exc
                self._active += 1
                continue
            except RetryableError as exc:
                self.records.append(CallRecord(stage, model, False, time.monotonic() - t0, ok=False, note=str(exc)[:200]))
                if exc.network:
                    # запасные модели живут на том же сервере — смена модели не поможет
                    raise LLMConnectionError(
                        f"нет связи с API {self.settings.provider_title} ({exc}). "
                        "Проверьте интернет, VPN или прокси (HTTPS_PROXY). Без модели можно запустить с --offline."
                    )
                log.warning("модель %s не ответила после повторов (%s) — пробую запасную", model, exc)
                last_exc = exc
                self._active += 1
                continue
            rec = CallRecord(
                stage,
                model,
                False,
                time.monotonic() - t0,
                raw.input_tokens,
                raw.output_tokens,
                attempts,
            )
            self.records.append(rec)
            self._note_model(model)
            self._cache_put(key, {"model": model, "stage": stage, "data": data})
            return data
        # все модели перебраны — откатываемся к основной для следующих этапов
        self._active = 0
        raise LLMError(f"ни одна модель не ответила: {last_exc}")

    def _note_model(self, model: str) -> None:
        if model and model not in self.used_models:
            self.used_models.append(model)

    def _throttle(self) -> None:
        gap = self.settings.min_interval - (time.monotonic() - self._last_call)
        if gap > 0:
            time.sleep(gap)
        self._last_call = time.monotonic()

    def _call_model(self, model: str, system: str, user: str, schema: Optional[dict]):
        attempts = 0
        json_repairs = 0
        prompt = user
        while True:
            attempts += 1
            self._throttle()
            try:
                raw = self.provider.generate(model, system, prompt, schema)
            except RetryableError as exc:
                limit = min(self.settings.max_retries, 2) if exc.network else self.settings.max_retries
                if attempts > limit:
                    raise
                delay = exc.retry_after if exc.retry_after else min(60.0, 2.0 * (2 ** (attempts - 1)))
                delay = min(delay, 120.0) + random.uniform(0, 1.0)
                log.info("повтор через %.0f с (%s)", delay, exc)
                time.sleep(delay)
                continue

            try:
                data = extract_json(raw.text)
            except JSONExtractError as exc:
                if raw.truncated:
                    raise LLMTruncated(f"ответ модели обрезан ({raw.finish_reason})")
                if json_repairs < 2:
                    json_repairs += 1
                    log.info("невалидный JSON от модели (%s) — прошу переформировать", exc)
                    prompt = user + (
                        "\n\nВАЖНО: предыдущая попытка вернула невалидный ответ "
                        f"({exc}). Верни ТОЛЬКО валидный JSON строго по схеме, без markdown и пояснений."
                    )
                    continue
                raise LLMError(f"модель {model} не вернула валидный JSON: {exc}")

            if schema:
                errors = validate(data, schema)
                # ошибки верхнего уровня (нет массива results и т.п.) — просим переделать;
                # ошибки отдельных элементов пайплайн отфильтрует сам
                fatal = [e for e in errors if "[" not in e.split(":", 1)[0]]
                if fatal and json_repairs < 2:
                    json_repairs += 1
                    log.info("ответ не соответствует схеме (%s) — прошу исправить", "; ".join(fatal[:3]))
                    prompt = user + (
                        "\n\nВАЖНО: предыдущий ответ не соответствует JSON-схеме: "
                        + "; ".join(fatal[:5])
                        + ". Верни ТОЛЬКО валидный JSON строго по схеме."
                    )
                    continue
                if errors:
                    log.debug("мелкие расхождения со схемой (будут отфильтрованы): %s", errors[:5])
            return data, raw, attempts


def make_client(settings) -> LLMClient:
    prov = settings.provider
    if prov == "gemini":
        from .gemini import GeminiProvider

        p: Provider = GeminiProvider(settings)
    elif prov in ("openai", "ollama"):
        from .openai_compat import OpenAICompatProvider

        p = OpenAICompatProvider(settings)
    elif prov == "anthropic":
        from .anthropic import AnthropicProvider

        p = AnthropicProvider(settings)
    else:
        raise LLMError(f"неизвестный провайдер {prov}")
    return LLMClient(provider=p, settings=settings)


def summarize_records(records: List[CallRecord]) -> Dict[str, Any]:
    live = [r for r in records if not r.cached and r.ok]
    return {
        "calls": len([r for r in records if r.ok]),
        "live_calls": len(live),
        "cached_calls": len([r for r in records if r.cached]),
        "failed_calls": len([r for r in records if not r.ok]),
        "input_tokens": sum(r.input_tokens for r in live),
        "output_tokens": sum(r.output_tokens for r in live),
        "llm_seconds": round(sum(r.seconds for r in records), 1),
    }
