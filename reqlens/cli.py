"""Командная строка: один запуск — один отчёт.

    python main.py docs/requirements.md reports/report.md
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional

from . import __product__, __version__
from .config import ConfigError, load_settings
from .llm import LLMFatalError, make_client
from .loaders import SUPPORTED, LoaderError
from .pipeline import run_audit
from .report_export import write_html, write_json
from .report_md import render_markdown, render_questions
from .verdict import VERDICT_LABEL


def _utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="reqlens",
        description="ReqLens — ИИ-агент статической проверки требований (ISO/IEC/IEEE 29148, IEEE 830). "
        "Читает документ, проверяет каждое требование и набор целиком, пишет отчёт.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Примеры:\n"
            "  python main.py examples/requirements.md reports/report.md\n"
            "  python main.py spec.pdf report.md --model gemini-3-flash-preview\n"
            "  python main.py spec.docx report.md --provider openai\n"
            "  python main.py spec.md report.md --offline      # без LLM, только правила\n"
            f"\nФорматы входа: {', '.join(sorted(SUPPORTED))}"
        ),
    )
    p.add_argument("input_pos", nargs="?", metavar="INPUT", help="путь к документу требований")
    p.add_argument("output_pos", nargs="?", metavar="OUTPUT", help="путь к файлу отчёта (.md)")
    p.add_argument("-i", "--input", dest="input_opt", help="путь к документу требований")
    p.add_argument("-o", "--output", dest="output_opt", help="путь к файлу отчёта (.md)")
    p.add_argument("--provider", choices=["gemini", "openai", "anthropic", "ollama"], help="провайдер LLM (по умолчанию — по ключу в .env)")
    p.add_argument("--model", help="модель (по умолчанию из .env)")
    p.add_argument("--offline", action="store_true", help="без LLM: только статические правила и структурная нарезка")
    p.add_argument("--batch-size", type=int, help="требований в одном запросе к модели (по умолчанию 10)")
    p.add_argument("--no-cache", action="store_true", help="не использовать кэш ответов модели")
    p.add_argument("--no-review", action="store_true", help="пропустить проход ревьюера-скептика (экономит 1 запрос)")
    p.add_argument("--no-html", action="store_true", help="не создавать HTML-версию отчёта")
    p.add_argument("--fail-on", choices=["critical", "major", "never"], default="never",
                   help="код выхода 2, если есть дефекты этого уровня (для CI)")
    p.add_argument("-v", "--verbose", action="store_true", help="подробный журнал")
    p.add_argument("-q", "--quiet", action="store_true", help="без прогресса")
    p.add_argument("--version", action="version", version=f"{__product__} {__version__}")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    _utf8_console()
    args = build_parser().parse_args(argv)
    src = args.input_opt or args.input_pos
    if not src:
        build_parser().print_help()
        return 1
    src_path = Path(src)
    out_arg = args.output_opt or args.output_pos
    out_path = Path(out_arg) if out_arg else Path("reports") / f"{src_path.stem}_report.md"
    if out_path.suffix.lower() not in (".md", ".markdown"):
        out_path = out_path.with_suffix(".md")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="      ! %(message)s",
        stream=sys.stderr,
    )
    logging.getLogger("MARKDOWN").setLevel(logging.WARNING)
    say = (lambda m: None) if args.quiet else (lambda m: print(m, file=sys.stderr, flush=True))
    say(f"{__product__} {__version__} — статическая проверка требований")

    llm = None
    batch = args.batch_size or 10
    review = not args.no_review
    if not args.offline:
        try:
            settings = load_settings(
                provider=args.provider,
                model=args.model,
                batch_size=args.batch_size,
                use_cache=False if args.no_cache else None,
                review_pass=False if args.no_review else None,
            )
        except ConfigError as exc:
            print(f"Ошибка настройки: {exc}", file=sys.stderr)
            return 2
        llm = make_client(settings)
        batch = settings.batch_size
        review = settings.review_pass
        fb = f" (запасные: {', '.join(settings.fallback_models)})" if settings.fallback_models else ""
        say(f"Модель: {settings.model} · {settings.provider_title}{fb}")
    else:
        say("Режим --offline: без LLM (только статические правила)")

    try:
        res = run_audit(src_path, llm, batch_size=batch, review_pass=review, progress=say)
    except LoaderError as exc:
        print(f"Ошибка чтения документа: {exc}", file=sys.stderr)
        return 2
    except LLMFatalError as exc:
        print(f"Ошибка доступа к модели: {exc}", file=sys.stderr)
        return 3

    out_path.parent.mkdir(parents=True, exist_ok=True)
    md = render_markdown(res)
    out_path.write_text(md, encoding="utf-8")
    json_path = out_path.with_suffix(".json")
    write_json(res, json_path)
    q_path = out_path.with_name(out_path.stem + "_questions.md")
    q_path.write_text(render_questions(res), encoding="utf-8")
    html_path = out_path.with_suffix(".html")
    html_ok = False if args.no_html else write_html(md, html_path, f"Отчёт ReqLens — {res.document.title}")

    cnt = Counter(f.severity for f in res.findings)
    v = res.verdict
    say("")
    say(f"Требований: {len(res.requirements)} · дефектов: {len(res.findings)} "
        f"(критичных {cnt['critical']}, существенных {cnt['major']}, замечаний {cnt['minor']})")
    if v:
        say(f"Разработка: {VERDICT_LABEL[v.development]} · Тестирование: {VERDICT_LABEL[v.testing]}")
    for wmsg in res.warnings:
        say(f"Предупреждение: {wmsg}")
    say(f"Отчёт: {out_path}")
    say(f"JSON:  {json_path}")
    say(f"Вопросы для заинтересованных сторон: {q_path}")
    if html_ok:
        say(f"HTML:  {html_path}")

    if args.fail_on == "critical" and cnt["critical"]:
        return 2
    if args.fail_on == "major" and (cnt["critical"] or cnt["major"]):
        return 2
    return 0
