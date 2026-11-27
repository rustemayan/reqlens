"""Нарезка документа на требования.

Основной путь — модель (понимает смысл и делит абзацы на независимые
утверждения). Результат модели проверяется кодом: каждая цитата должна найтись
в документе, иначе требование берётся по диапазону строк или отбрасывается.
Запасной путь — структурная нарезка по разметке (списки, таблицы, абзацы):
используется в режиме --offline и если модель недоступна.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .document import Document, normalize
from .models import Requirement
from .schemas import KINDS

_HEADING = re.compile(r"^\s{0,3}#{1,6}\s")
_BULLET = re.compile(r"^\s*([*\-+•]|\d+[.)])\s+")
_DOC_ID = re.compile(r"^\s*(?:[*\-+•]\s+)?\**\s*([A-ZА-ЯЁ][A-Za-zА-Яа-яЁё]{0,7}[-‐–]\d+(?:\.\d+)*)\.?\s*\**")
_NUMBERED = re.compile(r"^\s*(\d+)[.)]\s+")
_SEPARATOR = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$")
_SKIP_SECTIONS = re.compile(r"(оглавлен|содержани|согласующ|ревью требований|история изменений)", re.I)
_META_SECTIONS = re.compile(r"(общие сведения|паспорт|информация о документе)", re.I)

_KIND_BY_SECTION = [
    (r"нефункционал", "nonfunctional"),
    (r"функционал", "functional"),
    (r"данн", "data"),
    (r"интеграц", "integration"),
    (r"при[её]мк", "acceptance"),
    (r"(ui|ux|дизайн|интерфейс|экран)", "ui"),
    (r"(bpmn|процесс)", "process"),
    (r"архитектур", "architecture"),
    (r"(sequence|последовательн)", "interaction"),
    (r"цел", "goal"),
    (r"тестир", "testing"),
    (r"(api|rest|параметр|ответ|статус|логика|правила|схема|запрос|request|response)", "api"),
]


def kind_for(section: str) -> str:
    s = section.lower()
    last = s.split(" › ")[-1] if s else ""
    for pat, kind in _KIND_BY_SECTION:
        if re.search(pat, last):
            return kind
    for pat, kind in _KIND_BY_SECTION:
        if re.search(pat, s):
            return kind
    return "other"


def _clean(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^\s*[*\-+•]\s+", "", t)
    t = t.replace("**", "").replace("`", "")
    if t.startswith("|"):
        t = " | ".join(c.strip() for c in t.strip("|").split("|") if c.strip())
    return re.sub(r"\s+", " ", t).strip()


def _summary(text: str, words: int = 14) -> str:
    w = _clean(text).split()
    s = " ".join(w[:words])
    return s + ("…" if len(w) > words else "")


def short_section(section: str) -> str:
    parts = [p for p in section.split(" › ") if p]
    if not parts:
        return "—"
    if len(parts) == 1:
        return parts[0]
    head = re.match(r"(Часть\s+[A-ZА-Я0-9]+)", parts[0])
    prefix = head.group(1) + " › " if head else ""
    return prefix + parts[-1]


def structural_segment(doc: Document) -> List[Requirement]:
    units: List[Tuple[int, int, str, str]] = []  # start, end, text, doc_id
    para: List[int] = []
    para_id = [""]
    lines = doc.lines
    i = 0

    def flush_para():
        if para:
            text = " ".join(lines[n - 1].text.strip() for n in para)
            units.append((para[0], para[-1], text, para_id[0]))
            para.clear()
        para_id[0] = ""

    while i < len(lines):
        ln = lines[i]
        t = ln.text.strip()
        if t.startswith("```"):
            flush_para()
            j = i + 1
            while j < len(lines) and not lines[j].text.strip().startswith("```"):
                j += 1
            body = [lines[k].text.strip() for k in range(i + 1, min(j, len(lines)))]
            if body and not _SKIP_SECTIONS.search(ln.section):
                units.append((i + 2, min(j, len(lines)), " ".join(body), ""))
            i = j + 1
            continue
        if not t or _HEADING.match(t) or t in ("---", "***") or _SKIP_SECTIONS.search(ln.section):
            flush_para()
            i += 1
            continue
        if t.startswith("|"):
            flush_para()
            nxt = lines[i + 1].text.strip() if i + 1 < len(lines) else ""
            is_header = bool(_SEPARATOR.match(nxt))
            cells = [c.strip() for c in t.strip("|").split("|")]
            if _SEPARATOR.match(t) or is_header or not any(cells):
                i += 1
                continue
            if _META_SECTIONS.search(ln.section) and len(cells) == 2:
                i += 1
                continue
            units.append((ln.no, ln.no, t, ""))
            i += 1
            continue
        if _BULLET.match(t):
            flush_para()
            m = _DOC_ID.match(t)
            did = m.group(1) if m else ""
            if not did:
                nm = _NUMBERED.match(t)
                did = f"п. {nm.group(1)}" if nm else ""
            units.append((ln.no, ln.no, t, did))
            i += 1
            continue
        m = _DOC_ID.match(t)
        if m and re.match(r"^\s*\**\s*[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё]{0,7}[-‐–]\d+", t):
            # «Ф-1. …» без маркера списка (например, текст из PDF) — начало нового требования
            flush_para()
            para_id[0] = m.group(1)
        para.append(ln.no)
        i += 1
    flush_para()

    # аннотация под заголовком документа (до первого раздела) и связки вроде «или» — не требования
    has_sections = any(ln.section and ln.section != doc.title for ln in lines)
    reqs: List[Requirement] = []
    for start, end, text, did in units:
        section = doc.section_of(start)
        if has_sections and (not section or section == doc.title):
            continue
        if len(_clean(text).split()) < 2:
            continue
        reqs.append(Requirement(
            id="",
            doc_id=did,
            section=section,
            line_start=start,
            line_end=end,
            quote=_clean(text),
            summary=_summary(text),
            kind=kind_for(section),
            page=doc.line(start).page if doc.line(start) else None,
        ))
    return number(reqs)


def number(reqs: List[Requirement]) -> List[Requirement]:
    reqs.sort(key=lambda r: (r.line_start, r.line_end))
    width = 3 if len(reqs) >= 100 else 2
    for i, r in enumerate(reqs, start=1):
        r.id = f"REQ-{i:0{width}d}"
    return reqs


def from_llm(doc: Document, data: dict) -> Tuple[List[Requirement], Dict[str, int]]:
    """Проверяет нарезку модели по тексту документа."""
    stats = {"proposed": 0, "verified": 0, "by_range": 0, "dropped": 0, "duplicates": 0}
    seen = set()
    reqs: List[Requirement] = []
    for item in data.get("requirements") or []:
        if not isinstance(item, dict):
            continue
        stats["proposed"] += 1
        quote = str(item.get("quote") or "").strip()
        try:
            ls = int(item.get("line_start") or 0)
            le = int(item.get("line_end") or ls)
        except (TypeError, ValueError):
            ls, le = 0, 0
        if le < ls:
            ls, le = le, ls
        ev = doc.locate(quote, hint_line=ls or None) if quote else None
        if ev:
            stats["verified"] += 1
            # диапазон модели принимаем, если он содержит найденную цитату и не слишком широк
            if ls and ls <= ev.line_start and le >= ev.line_end and le - ls <= 12:
                start, end = ls, le
            else:
                start, end = ev.line_start, ev.line_end
            text = quote if ev.match == "exact" else ev.quote
        elif doc.line(ls) and doc.line(le) and le - ls <= 12 and doc.text_range(ls, le).strip():
            stats["by_range"] += 1
            start, end = ls, le
            text = _clean(doc.text_range(ls, le))
        else:
            stats["dropped"] += 1
            continue
        key = (start, end, normalize(text)[:80])
        if key in seen:
            stats["duplicates"] += 1
            continue
        seen.add(key)
        kind = str(item.get("kind") or "")
        section = doc.section_of(start) or str(item.get("section") or "")
        reqs.append(Requirement(
            id="",
            doc_id=_clean_doc_id(str(item.get("doc_id") or "")),
            section=section,
            line_start=start,
            line_end=end,
            quote=_clean(text),
            summary=str(item.get("summary") or _summary(text)).strip(),
            kind=kind if kind in KINDS else kind_for(section),
            page=doc.line(start).page if doc.line(start) else None,
        ))
    return number(reqs), stats


def _clean_doc_id(did: str) -> str:
    d = did.strip().strip("*[]()").strip()
    if d.lower() in ("", "-", "—", "нет", "без номера", "none", "null", "n/a"):
        return ""
    return d


def coverage(doc: Document, reqs: List[Requirement]) -> float:
    """Доля содержательных строк вне служебных разделов, попавших в какое-либо требование."""
    content = [n for n in doc.content_line_numbers() if not _SKIP_SECTIONS.search(doc.section_of(n))
               and not _META_SECTIONS.search(doc.section_of(n))]
    if not content:
        return 1.0
    covered = set()
    for r in reqs:
        covered.update(range(r.line_start, r.line_end + 1))
    return len([n for n in content if n in covered]) / len(content)


def find_requirement(reqs: List[Requirement], line: int) -> Optional[Requirement]:
    best = None
    for r in reqs:
        if r.line_start <= line <= r.line_end:
            if best is None or (r.line_end - r.line_start) < (best.line_end - best.line_start):
                best = r
    return best
