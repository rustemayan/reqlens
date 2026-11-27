import json
import os

import pytest

from reqlens import cli
from reqlens.config import ConfigError, load_dotenv, load_settings


def test_cli_offline_writes_reports(tmp_path, example_path):
    out = tmp_path / "out" / "report.md"
    code = cli.main([str(example_path), str(out), "--offline", "-q"])
    assert code == 0
    assert out.is_file() and out.with_suffix(".json").is_file()
    data = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert data["schema"] == "reqlens.audit/1"
    assert "## 6. Итог" in out.read_text(encoding="utf-8")


def test_cli_fail_on_critical(tmp_path, example_path):
    out = tmp_path / "r.md"
    assert cli.main(["-i", str(example_path), "-o", str(out), "--offline", "-q", "--fail-on", "critical"]) == 2


def test_cli_missing_file(tmp_path):
    assert cli.main([str(tmp_path / "nope.md"), str(tmp_path / "r.md"), "--offline", "-q"]) == 2


@pytest.fixture
def clean_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith(("GEMINI_", "GOOGLE_API", "OPENAI_", "ANTHROPIC_", "OLLAMA_", "LLM_", "REQLENS_")):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr("reqlens.config.load_dotenv", lambda *a, **k: [])
    return monkeypatch


def test_placeholder_key_is_rejected(clean_env):
    clean_env.setenv("GEMINI_API_KEY", "your_gemini_api_key_here")
    with pytest.raises(ConfigError):
        load_settings()


def test_gemini_defaults(clean_env):
    clean_env.setenv("GEMINI_API_KEY", "AIza-test")
    s = load_settings()
    assert s.provider == "gemini" and s.model == "gemini-3.5-flash"
    assert s.temperature is None and "gemini-3-flash-preview" in s.fallback_models


def test_openai_compatible(clean_env):
    clean_env.setenv("OPENAI_API_KEY", "sk-test")
    clean_env.setenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    clean_env.setenv("OPENAI_MODEL", "deepseek-chat")
    s = load_settings()
    assert s.provider == "openai" and s.base_url == "https://api.deepseek.com/v1" and s.temperature == 0.1


def test_dotenv_parsing(tmp_path, monkeypatch):
    monkeypatch.delenv("RL_TEST_A", raising=False)
    monkeypatch.delenv("RL_TEST_B", raising=False)
    p = tmp_path / ".env"
    p.write_text('# comment\nRL_TEST_A="quoted value"\nexport RL_TEST_B=plain # trailing\n', encoding="utf-8")
    load_dotenv([p])
    assert os.environ["RL_TEST_A"] == "quoted value"
    assert os.environ["RL_TEST_B"] == "plain"
