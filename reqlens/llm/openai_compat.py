"""OpenAI-совместимый Chat Completions API: OpenAI, DeepSeek, OpenRouter, Groq,
Together, vLLM, LM Studio и Ollama (через /v1)."""

from __future__ import annotations

import json
import logging
from typing import Optional

from .base import LLMAuthError, ModelUnavailable, Provider, RawResponse, RetryableError, map_transport_error
from .http import HTTPError, post_json

log = logging.getLogger("reqlens.llm")


class OpenAICompatProvider(Provider):
    name = "openai"

    def __init__(self, settings):
        super().__init__(settings)
        self._json_mode = True
        self._send_temperature = settings.temperature is not None
        self._max_tokens_field = "max_tokens"

    def generate(self, model: str, system: str, user: str, schema: Optional[dict]) -> RawResponse:
        url = f"{self.settings.base_url}/chat/completions"
        headers = {}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        sys_text = system
        if schema:
            sys_text += "\n\nJSON-схема ответа:\n" + json.dumps(schema, ensure_ascii=False)
        while True:
            payload: dict = {
                "model": model,
                "messages": [
                    {"role": "system", "content": sys_text},
                    {"role": "user", "content": user},
                ],
                self._max_tokens_field: self.settings.max_output_tokens,
            }
            if self._json_mode:
                payload["response_format"] = {"type": "json_object"}
            if self._send_temperature:
                payload["temperature"] = self.settings.temperature
            try:
                data, _ = post_json(url, payload, headers, self.settings.timeout)
                break
            except HTTPError as exc:
                low = exc.body.lower()
                if exc.status == 400 and "temperature" in low and self._send_temperature:
                    self._send_temperature = False
                    continue
                if exc.status == 400 and "max_tokens" in low and self._max_tokens_field == "max_tokens":
                    self._max_tokens_field = "max_completion_tokens"
                    continue
                if exc.status == 400 and "response_format" in low and self._json_mode:
                    self._json_mode = False
                    continue
                raise self._map_error(exc, model)

        choices = data.get("choices") or []
        if not choices:
            raise RetryableError("пустой ответ API (нет choices)")
        msg = choices[0].get("message") or {}
        finish = str(choices[0].get("finish_reason") or "")
        usage = data.get("usage") or {}
        return RawResponse(
            text=msg.get("content") or "",
            finish_reason=finish,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            truncated=finish == "length",
        )

    @staticmethod
    def _map_error(exc: HTTPError, model: str) -> Exception:
        s = exc.status
        transport = map_transport_error(s, exc.body)
        if transport:
            return transport
        if s == 429:
            if "insufficient_quota" in exc.body:
                return ModelUnavailable("квота аккаунта исчерпана")
            ra = exc.headers.get("retry-after")
            try:
                delay = float(ra) if ra else None
            except ValueError:
                delay = None
            return RetryableError("лимит запросов", retry_after=delay)
        if s in (401, 403):
            return LLMAuthError(f"нет доступа к API ({s}): {exc.body[:200]}. Проверьте ключ.")
        if s == 404:
            return ModelUnavailable(f"модель {model} не найдена")
        if s in (0, -1) or s >= 500:
            return RetryableError(f"временная ошибка API ({s}): {exc.body[:150]}")
        return ModelUnavailable(f"HTTP {s}: {exc.body[:200]}")
