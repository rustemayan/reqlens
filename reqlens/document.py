"""Модель документа: строки с номерами, разделы и поиск цитат.

Поиск цитат — основа принципа доказательности: любое замечание агента
должно ссылаться на текст, который действительно есть в документе.
Номер строки ставит код, а не модель.
"""

from __future__ import annotations

import difflib
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_LINE_PREFIX_RE = re.compile(r"^\s*L?\d{1,5}\s*[|:]\s?")
_ELLIPSIS_RE = re.compile(r"\s*(?:\.\.\.|…)\s*")


@dataclass
class Line:
    no: int
    text: str
    page: Optional[int] = None
    section: str = ""


@dataclass
class Evidence:
    quote: str
    line_start: int
    line_end: int
    section: str = ""
    page: Optional[int] = None
    match: str = "exact"  # exact | fuzzy | rule

    def where(self) -> str:
        if self.line_start == self.line_end:
            loc = f"строка {self.line_start}"
        else:
            loc = f"строки {self.line_start}–{self.line_end}"
        if self.page:
            loc = f"стр. {self.page}, {loc}"
        return loc

    def to_dict(self) -> dict:
        return {
            "quote": self.quote,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "section": self.section,
            "page": self.page,
            "match": self.match,
        }


def normalize(text: str) -> str:
    """Нормализация для сравнения цитат: регистр, кавычки, тире, разметка, пробелы."""
    s = text.replace(" ", " ").replace(" ", " ").replace(" ", " ")
    s = re.sub(r"[*_`#>]+", "", s)
    s = s.replace("|", " ")
    s = re.sub(r"[«»“”„\"'’‘`]", '"', s)
    s = re.sub(r"[‐‑‒–—―−]", "-", s)
    s = s.replace("ё", "е").replace("Ё", "Е")
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .;,:")


def _tokens(text: str) -> set:
    return {t for t in re.findall(r"[0-9a-zа-я]+", normalize(text)) if len(t) >= 3}


def clean_quote(quote: str) -> str:
    """Убирает префиксы номеров строк (L012|), которые модель могла скопировать."""
    parts = [_LINE_PREFIX_RE.sub("", p) for p in quote.splitlines()]
    q = " ".join(p.strip() for p in parts if p.strip())
    return q.strip().strip("«»\"' ")


@dataclass
class Document:
    path: Path
    title: str
    fmt: str
    lines: List[Line]
    raw_bytes: int = 0
    sha256: str = ""
    _norm_joined: str = field(default="", repr=False)
    _norm_index: List[Tuple[int, int]] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------ build
    @classmethod
    def from_text(
        cls,
        path: Path,
        text: str,
        fmt: str,
        pages: Optional[Sequence[Optional[int]]] = None,
        raw: Optional[bytes] = None,
    ) -> "Document":
        raw_lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        # убираем хвостовые пустые строки, номера остаются стабильными
        while raw_lines and not raw_lines[-1].strip():
            raw_lines.pop()
        lines: List[Line] = []
        stack: List[Tuple[int, str]] = []
        title = ""
        for i, t in enumerate(raw_lines, start=1):
            m = _HEADING_RE.match(t)
            if m:
                level = len(m.group(1))
                name = re.sub(r"[*_`]", "", m.group(2)).strip()
                if not title and level <= 2:
                    title = name
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, name))
            section = " › ".join(n for lvl, n in stack if lvl >= 2) or (stack[-1][1] if stack else "")
            page = pages[i - 1] if pages and i - 1 < len(pages) else None
            lines.append(Line(no=i, text=t.rstrip(), page=page, section=section))
        data = raw if raw is not None else text.encode("utf-8")
        doc = cls(
            path=path,
            title=title or path.stem,
            fmt=fmt,
            lines=lines,
            raw_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
        )
        doc._build_index()
        return doc

    def _build_index(self) -> None:
        parts: List[str] = []
        index: List[Tuple[int, int]] = []
        pos = 0
        for ln in self.lines:
            n = normalize(ln.text)
            index.append((pos, ln.no))
            parts.append(n)
            pos += len(n) + 1
        self._norm_joined = " ".join(parts)
        self._norm_index = index

    # ---------------------------------------------------------------- queries
    @property
    def line_count(self) -> int:
        return len(self.lines)

    def line(self, no: int) -> Optional[Line]:
        if 1 <= no <= len(self.lines):
            return self.lines[no - 1]
        return None

    def section_of(self, no: int) -> str:
        ln = self.line(no)
        return ln.section if ln else ""

    def text_range(self, start: int, end: int) -> str:
        start = max(1, start)
        end = min(len(self.lines), max(start, end))
        return "\n".join(self.lines[i - 1].text for i in range(start, end + 1))

    def numbered_text(self) -> str:
        """Текст документа с номерами строк — так модель может ссылаться на строки."""
        width = max(3, len(str(len(self.lines))))
        out = []
        for ln in self.lines:
            prefix = f"L{ln.no:0{width}d}|"
            if ln.page:
                prefix = f"L{ln.no:0{width}d}[p{ln.page}]|"
            out.append(f"{prefix} {ln.text}" if ln.text else prefix)
        return "\n".join(out)

    def content_line_numbers(self) -> List[int]:
        """Строки с содержанием (без пустых, заголовков и разделителей таблиц)."""
        res = []
        for ln in self.lines:
            t = ln.text.strip()
            if not t or _HEADING_RE.match(t) or re.fullmatch(r"[|:\-\s]+", t) or t in ("---", "```", "```text", "```json"):
                continue
            res.append(ln.no)
        return res

    def _offset_to_line(self, offset: int) -> int:
        lo, hi = 0, len(self._norm_index) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._norm_index[mid][0] <= offset:
                lo = mid
            else:
                hi = mid - 1
        return self._norm_index[lo][1]

    # ---------------------------------------------------------------- locate
    def locate(self, quote: str, hint_line: Optional[int] = None) -> Optional[Evidence]:
        """Находит цитату в документе. Возвращает Evidence с номерами строк или None.

        1) точное совпадение после нормализации (ближайшее к подсказке модели);
        2) цитата с многоточием — все фрагменты по порядку в окне до 15 строк;
        3) нечёткое совпадение (покрытие ≥ 0.85) — тогда цитатой становится
           реальный текст строки документа, а не формулировка модели.
        """
        q = clean_quote(quote)
        nq = normalize(q)
        if len(nq) < 3:
            return None

        found = self._find_exact(nq, hint_line)
        if found:
            s, e = found
            return self._evidence(q, s, e, "exact")

        fragments = [f for f in _ELLIPSIS_RE.split(q) if len(normalize(f)) >= 6]
        if len(fragments) >= 2:
            spans = []
            for frag in fragments:
                f = self._find_exact(normalize(frag), hint_line if not spans else spans[-1][1])
                if not f:
                    spans = []
                    break
                spans.append(f)
            if spans:
                s = min(x[0] for x in spans)
                e = max(x[1] for x in spans)
                if e - s <= 15:
                    return self._evidence(q, s, e, "exact")

        return self._find_fuzzy(q, nq, hint_line)

    def _find_exact(self, nq: str, hint_line: Optional[int]) -> Optional[Tuple[int, int]]:
        hay = self._norm_joined
        hits = []
        start = 0
        while True:
            idx = hay.find(nq, start)
            if idx < 0:
                break
            hits.append(idx)
            start = idx + 1
            if len(hits) > 50:
                break
        if not hits:
            return None
        spans = [(self._offset_to_line(i), self._offset_to_line(i + len(nq) - 1)) for i in hits]
        if hint_line:
            spans.sort(key=lambda sp: min(abs(sp[0] - hint_line), abs(sp[1] - hint_line)))
        return spans[0]

    def _find_fuzzy(self, q: str, nq: str, hint_line: Optional[int]) -> Optional[Evidence]:
        if len(nq) < 15:
            return None
        qtok = _tokens(q)
        if not qtok:
            return None
        candidates = []
        n = len(self.lines)
        for size in (1, 2, 3):
            for i in range(0, n - size + 1):
                chunk = self.lines[i : i + size]
                txt = " ".join(c.text for c in chunk)
                if not txt.strip():
                    continue
                tok = _tokens(txt)
                if not tok:
                    continue
                overlap = len(qtok & tok) / len(qtok)
                if overlap >= 0.6:
                    candidates.append((overlap, i + 1, i + size, txt))
        if not candidates:
            return None
        candidates.sort(key=lambda c: (-c[0], c[2] - c[1]))
        best = None
        for overlap, s, e, txt in candidates[:12]:
            nt = normalize(txt)
            sm = difflib.SequenceMatcher(None, nq, nt, autojunk=False)
            covered = sum(b.size for b in sm.get_matching_blocks())
            coverage = covered / max(1, len(nq))
            dist = abs(s - hint_line) if hint_line else 0
            score = (coverage, -dist, -(e - s))
            if coverage >= 0.85 and (best is None or score > best[0]):
                best = (score, s, e)
        if not best:
            return None
        _, s, e = best
        real = " ".join(self.lines[i - 1].text.strip() for i in range(s, e + 1))
        real = re.sub(r"\s*\|\s*", " | ", real).strip(" |")
        return self._evidence(real, s, e, "fuzzy")

    def _evidence(self, quote: str, s: int, e: int, match: str) -> Evidence:
        ln = self.line(s)
        return Evidence(
            quote=quote,
            line_start=s,
            line_end=e,
            section=ln.section if ln else "",
            page=ln.page if ln else None,
            match=match,
        )

    def evidence_for_lines(self, s: int, e: Optional[int] = None, quote: Optional[str] = None) -> Evidence:
        """Evidence для статических правил: цитата берётся прямо из строк документа."""
        e = e or s
        txt = quote if quote is not None else " ".join(self.lines[i - 1].text.strip() for i in range(s, e + 1))
        txt = re.sub(r"\s+", " ", txt).strip()
        ev = self._evidence(txt, s, e, "rule")
        return ev
