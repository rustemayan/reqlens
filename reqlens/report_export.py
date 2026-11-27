"""Экспорт результата: JSON (для интеграций и CI) и HTML (для чтения и демо)."""

from __future__ import annotations

import html
import json
from dataclasses import asdict
from typing import Any, Dict

from .models import AuditResult


def to_json_dict(res: AuditResult) -> Dict[str, Any]:
    doc = res.document
    return {
        "schema": "reqlens.audit/1",
        "document": {
            "title": doc.title,
            "file": doc.path.name,
            "format": doc.fmt,
            "lines": doc.line_count,
            "sha256": doc.sha256,
        },
        "meta": res.meta,
        "verdict": asdict(res.verdict) if res.verdict else None,
        "requirements": [r.to_dict() for r in res.requirements],
        "findings": [f.to_dict() for f in res.findings],
        "glossary": [
            {"concept": g.concept, "variants": [e.to_dict() for e in g.variants], "recommendation": g.recommendation}
            for g in res.glossary
        ],
        "traceability": [asdict(t) for t in res.trace],
        "missing_scenarios": [asdict(m) for m in res.missing],
        "priorities": [asdict(p) for p in res.priorities],
        "stages": [asdict(s) for s in res.stages],
        "stats": {k: v for k, v in res.stats.items() if k != "rejected"},
        "rejected_findings": [{"title": t, "reason": r} for t, r in res.stats.get("rejected", [])],
        "warnings": res.warnings,
    }


def write_json(res: AuditResult, path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(to_json_dict(res), fh, ensure_ascii=False, indent=2)


_CSS = """
:root{--fg:#1f2328;--muted:#59636e;--bg:#fff;--line:#d1d9e0;--soft:#f6f8fa;--accent:#0b5cad}
@media (prefers-color-scheme: dark){:root{--fg:#e6edf3;--muted:#9198a1;--bg:#0d1117;--line:#30363d;--soft:#151b23;--accent:#4493f8}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 -apple-system,"Segoe UI",Roboto,Arial,sans-serif}
main{max-width:1180px;margin:0 auto;padding:32px 24px 80px}
h1{font-size:28px;margin:0 0 12px}h2{margin-top:44px;padding-bottom:6px;border-bottom:1px solid var(--line)}
h3{margin-top:28px}h4{margin:26px 0 6px;font-size:16px}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
blockquote{margin:12px 0;padding:8px 14px;border-left:4px solid var(--accent);background:var(--soft);color:var(--fg)}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:13.5px;display:block;overflow-x:auto}
th,td{border:1px solid var(--line);padding:6px 8px;vertical-align:top}th{background:var(--soft);text-align:left}
td:first-child{white-space:nowrap}
code{background:var(--soft);padding:1px 4px;border-radius:4px;font-size:90%}
ul{padding-left:22px}li{margin:3px 0}
.toolbar{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:8px 24px;z-index:5;font-size:14px}
.toolbar a{margin-right:14px}
"""


def write_html(markdown_text: str, path, title: str) -> bool:
    """Конвертирует Markdown-отчёт в автономный HTML. Возвращает False, если нет пакета markdown."""
    try:
        import markdown  # type: ignore
    except ImportError:
        return False
    body = markdown.markdown(markdown_text, extensions=["tables", "sane_lists"], output_format="html5")
    nav = ('<div class="toolbar"><a href="#top">↑ Наверх</a>'
           '<a href="#s1">Паспорт</a><a href="#s2">Нарезка</a><a href="#s3">Требования</a>'
           '<a href="#s4">Набор</a><a href="#s5">Сводка</a><a href="#s6">Итог</a><a href="#s7">Вопросы</a></div>')
    for n in range(1, 8):
        body = body.replace(f"<h2>{n}. ", f'<h2 id="s{n}">{n}. ', 1)
    doc = (
        "<!DOCTYPE html><html lang=\"ru\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head>"
        f"<body><a id=\"top\"></a>{nav}<main>{body}</main></body></html>"
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return True
