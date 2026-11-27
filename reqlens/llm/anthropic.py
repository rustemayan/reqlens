"""Anthropic Messages API без SDK."""

from __future__ import annotations

import json
from typing import Optional

from .base import LLMAuthError, ModelUnavailable, Provider, RawResponse, RetryableError, map_transport_error
from .http import HTTPError, post_json


class AnthropicProvider(Provider):
    name = "anthropic"

    def generate(self, model: str, system: str, user: str, schema: Optional[dict]) -> RawResponse:
        url = f"{self.settings.base_url}/v1/messages"
        headers = {"x-api-key": self.settings.api_key, "anthropic-version": "2023-06-01"}
        sys_text = system
        if schema:
            sys_text += "\n\nJSON-схема ответа:\n" + json.dumps(schema, ensure_ascii=False)
        payload: dict = {
            "model": model,
            "max_tokens": min(self.settings.max_output_tokens, 32000),
            "system": sys_text,
            "messages": [{"role": "user", "content": user}],
        }
        if self.settings.temperature is not None:
            payload["temperature"] = self.settings.temperature
        try:
            data, _ = post_json(url, payload, headers, self.settings.timeout)
        except HTTPError as exc:
            s = exc.status
            transport = map_transport_error(s, exc.body)
            if transport:
                raise transport
            if s == 429 or s == 529:
                ra = exc.headers.get("retry-after")
                return_delay = float(ra) if ra and ra.replace(".", "", 1).isdigit() else None
                raise RetryableError("лимит запросов Anthropic", retry_after=return_delay)
            if s in (401, 403):
                raise LLMAuthError(f"нет доступа к Anthropic API: {exc.body[:200]}")
            if s == 404:
                raise ModelUnavailable(f"модель {model} не найдена")
            if s in (0, -1) or s >= 500:
                raise RetryableError(f"временная ошибка Anthropic ({s})")
            raise ModelUnavailable(f"HTTP {s}: {exc.body[:200]}")
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage") or {}
        stop = str(data.get("stop_reason") or "")
        return RawResponse(
            text=text,
            finish_reason=stop,
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            truncated=stop == "max_tokens",
        )
