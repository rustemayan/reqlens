"""Настройки агента: .env + переменные окружения + аргументы командной строки."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODELS = {
    "gemini": "gemini-3.5-flash",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-5-5",
    "ollama": "qwen2.5:14b",
}
# Если у основной модели кончилась дневная квота бесплатного тарифа или её нет
# в регионе — агент автоматически переходит к следующей модели из списка.
DEFAULT_GEMINI_FALLBACKS = ["gemini-3-flash-preview", "gemini-3.1-flash-lite", "gemini-2.5-flash"]

_PLACEHOLDERS = ("your_", "xxx", "<", "changeme", "sk-...", "")


class ConfigError(Exception):
    pass


def load_dotenv(paths: Optional[List[Path]] = None) -> List[Path]:
    """Читает .env (KEY=VALUE). Уже заданные переменные окружения не перезаписывает."""
    loaded = []
    for p in paths or [Path.cwd() / ".env", PROJECT_ROOT / ".env"]:
        if not p.is_file() or p in loaded:
            continue
        for raw in p.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[7:]
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if value and value[0] in "\"'" and value[-1:] == value[0]:
                value = value[1:-1]
            elif value.startswith("#"):
                value = ""
            elif " #" in value:
                value = value.split(" #", 1)[0].strip()
            if key and key not in os.environ:
                os.environ[key] = value
        loaded.append(p)
    return loaded


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _is_real_key(value: str) -> bool:
    v = value.strip()
    return bool(v) and not any(v.lower().startswith(p) for p in _PLACEHOLDERS if p)


@dataclass
class Settings:
    provider: str
    model: str
    api_key: str = ""
    base_url: str = ""
    fallback_models: List[str] = field(default_factory=list)
    temperature: Optional[float] = None
    max_output_tokens: int = 32768
    timeout: float = 300.0
    max_retries: int = 4
    min_interval: float = 0.0
    batch_size: int = 10
    cache_dir: Path = PROJECT_ROOT / ".reqlens_cache"
    use_cache: bool = True
    thinking_level: str = ""
    review_pass: bool = True

    @property
    def provider_title(self) -> str:
        return {
            "gemini": "Google Gemini API",
            "openai": "OpenAI-совместимый API",
            "anthropic": "Anthropic API",
            "ollama": "Ollama (локально)",
        }.get(self.provider, self.provider)

    def describe(self) -> Dict[str, str]:
        return {
            "provider": self.provider_title,
            "model": self.model,
            "fallback": ", ".join(self.fallback_models) or "—",
        }


def detect_provider() -> str:
    explicit = _env("LLM_PROVIDER").lower()
    if explicit:
        if explicit not in DEFAULT_MODELS:
            raise ConfigError(f"Неизвестный LLM_PROVIDER={explicit}. Допустимо: {', '.join(DEFAULT_MODELS)}")
        return explicit
    if _is_real_key(_env("GEMINI_API_KEY")) or _is_real_key(_env("GOOGLE_API_KEY")):
        return "gemini"
    if _is_real_key(_env("OPENAI_API_KEY")):
        return "openai"
    if _is_real_key(_env("ANTHROPIC_API_KEY")):
        return "anthropic"
    if _env("OLLAMA_MODEL") or _env("OLLAMA_BASE_URL"):
        return "ollama"
    raise ConfigError(
        "Не найден ключ LLM. Скопируйте .env.example в .env и укажите GEMINI_API_KEY "
        "(бесплатно: https://aistudio.google.com/ → Get API key). "
        "Либо запустите без модели: --offline (только статические правила)."
    )


def load_settings(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    batch_size: Optional[int] = None,
    use_cache: Optional[bool] = None,
    review_pass: Optional[bool] = None,
) -> Settings:
    load_dotenv()
    prov = (provider or "").lower() or detect_provider()
    if prov not in DEFAULT_MODELS:
        raise ConfigError(f"Неизвестный провайдер: {prov}")

    if prov == "gemini":
        key = _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY")
        mdl = model or _env("GEMINI_MODEL") or DEFAULT_MODELS["gemini"]
        fb_raw = _env("GEMINI_FALLBACK_MODELS")
        fallbacks = [m.strip() for m in fb_raw.split(",") if m.strip()] if fb_raw else list(DEFAULT_GEMINI_FALLBACKS)
        base = _env("GEMINI_BASE_URL") or "https://generativelanguage.googleapis.com"
        min_interval = 4.0
    elif prov == "openai":
        key = _env("OPENAI_API_KEY")
        mdl = model or _env("OPENAI_MODEL") or DEFAULT_MODELS["openai"]
        fallbacks = [m.strip() for m in _env("OPENAI_FALLBACK_MODELS").split(",") if m.strip()]
        base = _env("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        min_interval = 0.0
    elif prov == "anthropic":
        key = _env("ANTHROPIC_API_KEY")
        mdl = model or _env("ANTHROPIC_MODEL") or DEFAULT_MODELS["anthropic"]
        fallbacks = [m.strip() for m in _env("ANTHROPIC_FALLBACK_MODELS").split(",") if m.strip()]
        base = _env("ANTHROPIC_BASE_URL") or "https://api.anthropic.com"
        min_interval = 0.0
    else:  # ollama
        key = _env("OLLAMA_API_KEY", "ollama")
        mdl = model or _env("OLLAMA_MODEL") or DEFAULT_MODELS["ollama"]
        fallbacks = []
        base = (_env("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        min_interval = 0.0

    if prov != "ollama" and not _is_real_key(key):
        env_name = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}[prov]
        raise ConfigError(f"Для провайдера {prov} не задан ключ {env_name} (в .env или в окружении).")

    fallbacks = [m for m in fallbacks if m != mdl]
    temp_raw = _env("LLM_TEMPERATURE")
    temperature: Optional[float] = float(temp_raw) if temp_raw else None
    if temperature is None and prov != "gemini":
        temperature = 0.1  # для Gemini 3.x Google рекомендует оставлять температуру по умолчанию

    def _num(name: str, default: float) -> float:
        raw = _env(name)
        try:
            return float(raw) if raw else default
        except ValueError:
            raise ConfigError(f"{name} должно быть числом, получено: {raw!r}")

    cache_dir = Path(_env("REQLENS_CACHE_DIR")) if _env("REQLENS_CACHE_DIR") else PROJECT_ROOT / ".reqlens_cache"

    return Settings(
        provider=prov,
        model=mdl,
        api_key=key,
        base_url=base.rstrip("/"),
        fallback_models=fallbacks,
        temperature=temperature,
        max_output_tokens=int(_num("LLM_MAX_OUTPUT_TOKENS", 32768 if prov == "gemini" else 16000)),
        timeout=_num("LLM_TIMEOUT", 300.0),
        max_retries=int(_num("LLM_MAX_RETRIES", 4)),
        min_interval=_num("LLM_MIN_INTERVAL", min_interval),
        batch_size=int(batch_size or _num("REQLENS_BATCH_SIZE", 10)),
        cache_dir=cache_dir,
        use_cache=use_cache if use_cache is not None else _env("REQLENS_CACHE", "1") not in ("0", "false", "no"),
        thinking_level=_env("GEMINI_THINKING_LEVEL"),
        review_pass=review_pass if review_pass is not None else _env("REQLENS_REVIEW", "1") not in ("0", "false", "no"),
    )
