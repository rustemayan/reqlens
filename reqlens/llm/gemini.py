"""Google Gemini API (generateContent, REST v1beta) без SDK."""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from .base import LLMAuthError, ModelUnavailable, Provider, RawResponse, RetryableError, map_transport_error
from .http import HTTPError, post_json

log = logging.getLogger("reqlens.llm")

# Способы попросить JSON — от нового формата API к старым. Если API отвергает
# поле (400 INVALID_ARGUMENT), клиент автоматически переходит к следующему.
_FORMAT_MODES = ("responseFormat", "responseJsonSchema", "responseMimeType", "plain")


def _parse_delay(value: str) -> Optional[float]:
    m = re.match(r"^\s*([\d.]+)\s*s\s*$", value or "")
    return float(m.group(1)) if m else None


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, settings):
        super().__init__(settings)
        self._mode = 0

    def _payload(self, system: str, user: str, schema: Optional[dict]) -> dict:
        gen: dict = {"maxOutputTokens": self.settings.max_output_tokens}
        if self.settings.temperature is not None:
            gen["temperature"] = self.settings.temperature
        if self.settings.thinking_level:
            gen["thinkingConfig"] = {"thinkingLevel": self.settings.thinking_level}
        mode = _FORMAT_MODES[self._mode]
        if mode == "responseFormat" and schema:
            gen["responseFormat"] = {"text": {"mimeType": "application/json", "schema": schema}}
        elif mode == "responseJsonSchema" and schema:
            gen["responseMimeType"] = "application/json"
            gen["responseJsonSchema"] = schema
        elif mode in ("responseMimeType", "responseFormat", "responseJsonSchema"):
            gen["responseMimeType"] = "application/json"
        return {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": gen,
        }

    def generate(self, model: str, system: str, user: str, schema: Optional[dict]) -> RawResponse:
        url = f"{self.settings.base_url}/v1beta/models/{model}:generateContent"
        headers = {"x-goog-api-key": self.settings.api_key}
        while True:
            payload = self._payload(system, user, schema)
            try:
                data, _ = post_json(url, payload, headers, self.settings.timeout)
                break
            except HTTPError as exc:
                if exc.status == 400 and "thinking" in exc.body.lower() and self.settings.thinking_level:
                    log.info("модель не поддерживает thinkingLevel — отключаю")
                    self.settings.thinking_level = ""
                    continue
                if self._is_format_rejection(exc) and self._mode < len(_FORMAT_MODES) - 1:
                    self._mode += 1
                    log.info("Gemini отклонил формат structured output, пробую режим %s", _FORMAT_MODES[self._mode])
                    continue
                raise self._map_error(exc, model)
        return self._parse(data)

    @staticmethod
    def _is_format_rejection(exc: HTTPError) -> bool:
        if exc.status != 400:
            return False
        body = exc.body.lower()
        markers = ("responseformat", "response_format", "responsejsonschema", "response_json_schema",
                   "responsemimetype", "response_mime_type", "unknown name", "invalid json payload", "schema")
        return any(m in body for m in markers)

    def _map_error(self, exc: HTTPError, model: str) -> Exception:
        status = exc.status
        body = exc.body
        transport = map_transport_error(status, body)
        if transport:
            return transport
        try:
            err = json.loads(body).get("error", {})
        except (json.JSONDecodeError, AttributeError):
            err = {}
        message = err.get("message") or body[:300]
        details = err.get("details") or []
        if status == 429:
            per_day = "perday" in body.lower().replace("_", "").replace(" ", "")
            delay = None
            for d in details:
                if isinstance(d, dict) and "retryDelay" in d:
                    delay = _parse_delay(str(d.get("retryDelay")))
            if per_day or (delay is not None and delay > 300):
                return ModelUnavailable(f"дневная квота модели {model} исчерпана")
            return RetryableError(f"лимит запросов в минуту ({message[:120]})", retry_after=delay or 20.0)
        if status == 404:
            return ModelUnavailable(f"модель {model} не найдена: {message[:150]}")
        if status in (401, 403):
            return LLMAuthError(f"нет доступа к Gemini API: {message[:200]}. Проверьте GEMINI_API_KEY.")
        if status == 400:
            low = message.lower()
            if "api key" in low or "api_key" in low:
                return LLMAuthError(f"неверный ключ Gemini: {message[:200]}. Проверьте GEMINI_API_KEY в .env.")
            if "location" in low or "region" in low or "not supported" in low and "user" in low:
                return LLMAuthError(f"Gemini API недоступен из вашего региона: {message[:200]}. Используйте VPN или другого провайдера.")
            return ModelUnavailable(f"запрос отклонён моделью {model}: {message[:200]}")
        if status in (0, -1) or status >= 500:
            return RetryableError(f"временная ошибка Gemini ({status}): {message[:150]}")
        return ModelUnavailable(f"HTTP {status}: {message[:200]}")

    @staticmethod
    def _parse(data: dict) -> RawResponse:
        cands = data.get("candidates") or []
        usage = data.get("usageMetadata") or {}
        in_tok = int(usage.get("promptTokenCount") or 0)
        out_tok = int(usage.get("candidatesTokenCount") or 0) + int(usage.get("thoughtsTokenCount") or 0)
        if not cands:
            fb = data.get("promptFeedback") or {}
            raise RetryableError(f"пустой ответ Gemini (blockReason={fb.get('blockReason', '?')})")
        cand = cands[0]
        finish = str(cand.get("finishReason") or "")
        parts = (cand.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and not p.get("thought"))
        if not text.strip() and finish not in ("MAX_TOKENS",):
            raise RetryableError(f"Gemini вернул пустой текст (finishReason={finish})")
        return RawResponse(
            text=text,
            finish_reason=finish,
            input_tokens=in_tok,
            output_tokens=out_tok,
            truncated=finish == "MAX_TOKENS",
        )
