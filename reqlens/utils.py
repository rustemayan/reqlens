"""Вспомогательные функции: версия агента из git, форматирование."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict

from . import __product__, __version__
from .config import PROJECT_ROOT


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *args],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def agent_version() -> Dict[str, str]:
    commit = _git("rev-parse", "--short", "HEAD")
    tag = _git("describe", "--tags", "--exact-match") if commit else ""
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)reports")) if commit else False
    label = f"{__product__} {__version__}"
    if tag:
        label += f", тег {tag}"
    if commit:
        label += f", commit {commit}" + (" (+ незакоммиченные изменения)" if dirty else "")
    else:
        label += " (вне git-репозитория — commit не определён)"
    return {"version": __version__, "commit": commit, "tag": tag, "dirty": "yes" if dirty else "no", "label": label}


def human_size(n: int) -> str:
    if n < 1024:
        return f"{n} Б"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} КБ"
    return f"{n / 1024 / 1024:.1f} МБ"


def plural(n: int, one: str, few: str, many: str) -> str:
    n10, n100 = n % 10, n % 100
    if n10 == 1 and n100 != 11:
        return one
    if 2 <= n10 <= 4 and not 12 <= n100 <= 14:
        return few
    return many


def rel_path(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(p)
