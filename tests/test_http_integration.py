"""Интеграционный тест HTTP-слоя: локальный сервер имитирует Gemini, OpenAI и Anthropic API."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from reqlens.config import Settings
from reqlens.llm import make_client
from reqlens.llm.base import LLMAuthError

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}


class Handler(BaseHTTPRequestHandler):
    log = []

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        Handler.log.append((self.path, dict(self.headers), payload))
        if self.path.startswith("/v1beta/models/"):
            if self.headers.get("x-goog-api-key") != "good":
                return self._send(400, {"error": {"code": 400, "message": "API key not valid. Please pass a valid API key."}})
            if "bad-model" in self.path:
                return self._send(404, {"error": {"code": 404, "message": "models/bad-model is not found"}})
            gen = payload["generationConfig"]
            if "responseFormat" in gen:
                return self._send(400, {"error": {"code": 400, "message": 'Invalid JSON payload received. Unknown name "responseFormat"'}})
            return self._send(200, {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": '{"ok": true}'}]}}],
                                    "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2}})
        if self.path == "/v1/chat/completions":
            assert self.headers.get("Authorization") == "Bearer sk"
            return self._send(200, {"choices": [{"finish_reason": "stop", "message": {"content": '{"ok": true}'}}],
                                    "usage": {"prompt_tokens": 3, "completion_tokens": 2}})
        if self.path == "/v1/messages":
            assert self.headers.get("x-api-key") == "ak"
            return self._send(200, {"content": [{"type": "text", "text": '{"ok": true}'}], "stop_reason": "end_turn",
                                    "usage": {"input_tokens": 3, "output_tokens": 2}})
        self._send(404, {"error": "unknown"})


@pytest.fixture(scope="module")
def server():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _s(tmp_path, provider, base, key, model="m", fallbacks=()):
    return Settings(provider=provider, model=model, api_key=key, base_url=base, cache_dir=tmp_path,
                    use_cache=False, min_interval=0, max_retries=1, fallback_models=list(fallbacks), timeout=10)


def test_gemini_real_http_path_with_fallback(tmp_path, server):
    Handler.log.clear()
    c = make_client(_s(tmp_path, "gemini", server, "good", model="bad-model", fallbacks=["gemini-ok"]))
    assert c.complete_json("t", "sys", "user", SCHEMA) == {"ok": True}
    assert c.used_models == ["gemini-ok"]
    paths = [p for p, _, _ in Handler.log]
    assert paths[0].endswith("bad-model:generateContent") and paths[-1].endswith("gemini-ok:generateContent")
    last = Handler.log[-1][2]["generationConfig"]
    assert last["responseMimeType"] == "application/json" and "responseJsonSchema" in last


def test_gemini_bad_key(tmp_path, server):
    c = make_client(_s(tmp_path, "gemini", server, "wrong"))
    with pytest.raises(LLMAuthError):
        c.complete_json("t", "sys", "user", SCHEMA)


def test_openai_compatible_http(tmp_path, server):
    c = make_client(_s(tmp_path, "openai", server + "/v1", "sk"))
    assert c.complete_json("t", "sys", "user", SCHEMA) == {"ok": True}
    payload = Handler.log[-1][2]
    assert payload["response_format"] == {"type": "json_object"}


def test_anthropic_http(tmp_path, server):
    c = make_client(_s(tmp_path, "anthropic", server, "ak"))
    assert c.complete_json("t", "sys", "user", SCHEMA) == {"ok": True}
