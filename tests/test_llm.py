import json

import pytest

from reqlens.config import Settings
from reqlens.llm import base
from reqlens.llm.base import LLMClient, LLMTruncated, ModelUnavailable, Provider, RawResponse, RetryableError
from reqlens.llm.gemini import GeminiProvider
from reqlens.llm.http import HTTPError
from reqlens.llm.jsonutil import JSONExtractError, extract_json, validate

SCHEMA = {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "string"}}}, "required": ["items"]}


def _settings(tmp_path, **kw):
    s = Settings(provider="gemini", model="m1", api_key="k", cache_dir=tmp_path / "c", min_interval=0, max_retries=2,
                 fallback_models=["m2"])
    for k, v in kw.items():
        setattr(s, k, v)
    return s


class Scripted(Provider):
    def __init__(self, settings, script):
        super().__init__(settings)
        self.script = list(script)
        self.seen = []

    def generate(self, model, system, user, schema):
        self.seen.append((model, user))
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(base.time, "sleep", lambda s: None)


def test_extract_json_variants():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Ответ: {"a": [1, 2,],}') == {"a": [1, 2]}
    with pytest.raises(JSONExtractError):
        extract_json("нет json")


def test_validate_reports_errors():
    assert validate({"items": ["x"]}, SCHEMA) == []
    errs = validate({"items": [1]}, SCHEMA)
    assert errs and "$.items[0]" in errs[0]
    assert validate({}, SCHEMA)[0].startswith("$: нет обязательного поля")


def test_retry_then_success_and_cache(tmp_path):
    s = _settings(tmp_path)
    p = Scripted(s, [RetryableError("429", 0.0), RawResponse('{"items": ["a"]}', input_tokens=5, output_tokens=2)])
    c = LLMClient(p, s)
    assert c.complete_json("st", "sys", "user", SCHEMA) == {"items": ["a"]}
    assert c.records[-1].attempts == 2
    # второй вызов — из кэша, провайдер не вызывается
    assert c.complete_json("st", "sys", "user", SCHEMA) == {"items": ["a"]}
    assert c.records[-1].cached and len(p.seen) == 2


def test_fallback_model_on_quota(tmp_path):
    s = _settings(tmp_path)
    p = Scripted(s, [ModelUnavailable("дневная квота"), RawResponse('{"items": []}')])
    c = LLMClient(p, s)
    assert c.complete_json("st", "sys", "u", SCHEMA) == {"items": []}
    assert [m for m, _ in p.seen] == ["m1", "m2"]
    assert c.used_models == ["m2"]


def test_invalid_json_is_reasked(tmp_path):
    s = _settings(tmp_path)
    p = Scripted(s, [RawResponse("извините, не могу"), RawResponse('{"items": ["ok"]}')])
    c = LLMClient(p, s)
    assert c.complete_json("st", "sys", "u", SCHEMA) == {"items": ["ok"]}
    assert "предыдущая попытка вернула невалидный ответ" in p.seen[1][1]


def test_schema_violation_is_reasked(tmp_path):
    s = _settings(tmp_path)
    p = Scripted(s, [RawResponse('{"wrong": 1}'), RawResponse('{"items": []}')])
    c = LLMClient(p, s)
    assert c.complete_json("st", "sys", "u", SCHEMA) == {"items": []}


def test_network_failure_fails_fast_without_model_switch(tmp_path):
    from reqlens.llm.base import LLMConnectionError

    s = _settings(tmp_path, max_retries=4)
    p = Scripted(s, [RetryableError("сеть", network=True)] * 10)
    with pytest.raises(LLMConnectionError):
        LLMClient(p, s).complete_json("st", "sys", "u", SCHEMA)
    assert len(p.seen) == 3 and {m for m, _ in p.seen} == {"m1"}  # 1 + 2 повтора, без смены модели


def test_tls_error_is_fatal(tmp_path):
    from reqlens.llm.base import LLMConnectionError

    prov = GeminiProvider(_settings(tmp_path))
    assert isinstance(prov._map_error(HTTPError(-2, "CERTIFICATE_VERIFY_FAILED"), "x"), LLMConnectionError)


def test_truncation_raises(tmp_path):
    s = _settings(tmp_path)
    p = Scripted(s, [RawResponse('{"items": ["a', finish_reason="MAX_TOKENS", truncated=True)])
    with pytest.raises(LLMTruncated):
        LLMClient(p, s).complete_json("st", "sys", "u", SCHEMA)


# ------------------------------------------------------------- Gemini REST
def test_gemini_format_ladder_and_parse(tmp_path, monkeypatch):
    s = _settings(tmp_path)
    prov = GeminiProvider(s)
    calls = []

    def fake_post(url, payload, headers, timeout):
        calls.append(payload)
        gen = payload["generationConfig"]
        if "responseFormat" in gen:
            raise HTTPError(400, json.dumps({"error": {"code": 400, "message": 'Invalid JSON payload received. Unknown name "responseFormat"'}}))
        return {
            "candidates": [{"finishReason": "STOP", "content": {"parts": [
                {"text": "размышления", "thought": True}, {"text": '{"items": ["x"]}'}]}}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 4, "thoughtsTokenCount": 6},
        }, {}

    monkeypatch.setattr("reqlens.llm.gemini.post_json", fake_post)
    raw = prov.generate("gemini-x", "sys", "user", SCHEMA)
    assert raw.text == '{"items": ["x"]}'
    assert raw.output_tokens == 10
    assert "responseJsonSchema" in calls[1]["generationConfig"]
    assert calls[0]["systemInstruction"]["parts"][0]["text"] == "sys"
    assert "temperature" not in calls[0]["generationConfig"]  # Gemini 3: температура по умолчанию


@pytest.mark.parametrize(
    "status,body,exc",
    [
        (429, {"error": {"message": "Quota exceeded", "details": [{"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}}, ModelUnavailable),
        (429, {"error": {"message": "Rate", "details": [{"@type": "RetryInfo", "retryDelay": "17s"}]}}, RetryableError),
        (404, {"error": {"message": "models/x is not found"}}, ModelUnavailable),
        (503, {"error": {"message": "overloaded"}}, RetryableError),
    ],
)
def test_gemini_error_mapping(tmp_path, status, body, exc):
    prov = GeminiProvider(_settings(tmp_path))
    err = prov._map_error(HTTPError(status, json.dumps(body)), "x")
    assert isinstance(err, exc)
    if isinstance(err, RetryableError) and status == 429:
        assert err.retry_after == 17.0


def test_gemini_bad_key_is_auth_error(tmp_path):
    from reqlens.llm.base import LLMAuthError

    prov = GeminiProvider(_settings(tmp_path))
    body = json.dumps({"error": {"code": 400, "message": "API key not valid. Please pass a valid API key."}})
    assert isinstance(prov._map_error(HTTPError(400, body), "x"), LLMAuthError)
