"""Загрузчики входных документов: Markdown, TXT, PDF, DOCX, HTML.

DOCX и HTML читаются стандартной библиотекой (без зависимостей), PDF — через
pypdf (запасной вариант — pdfminer.six). Структура документа (заголовки,
списки, таблицы) по возможности переводится в Markdown, чтобы разделы
определялись одинаково для всех форматов.
"""

from __future__ import annotations

import html
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Optional, Tuple
from xml.etree import ElementTree as ET

from .document import Document


class LoaderError(Exception):
    pass


SUPPORTED = {
    ".md": "Markdown",
    ".markdown": "Markdown",
    ".txt": "Текст",
    ".text": "Текст",
    ".pdf": "PDF",
    ".docx": "DOCX",
    ".html": "HTML",
    ".htm": "HTML",
}


def load_document(path: Path) -> Document:
    path = Path(path)
    if not path.exists():
        raise LoaderError(f"Файл не найден: {path}")
    if path.is_dir():
        raise LoaderError(f"Ожидался файл, а не папка: {path}")
    ext = path.suffix.lower()
    raw = path.read_bytes()
    if not raw.strip():
        raise LoaderError(f"Файл пустой: {path}")

    if ext == ".pdf":
        text, pages = _read_pdf(path)
        doc = Document.from_text(path, text, "PDF", pages=pages, raw=raw)
    elif ext == ".docx":
        doc = Document.from_text(path, _read_docx(path), "DOCX", raw=raw)
    elif ext in (".html", ".htm"):
        doc = Document.from_text(path, _read_html(_decode(raw)), "HTML", raw=raw)
    else:
        # .md, .txt и любые неизвестные текстовые форматы
        doc = Document.from_text(path, _decode(raw), SUPPORTED.get(ext, "Текст"), raw=raw)

    if not any(ln.text.strip() for ln in doc.lines):
        raise LoaderError(
            f"Не удалось извлечь текст из {path.name}. Если это скан PDF без текстового слоя, "
            "сначала распознайте его (OCR) или передайте документ в формате DOCX/MD."
        )
    return doc


def _decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "cp1251"):
        try:
            text = raw.decode(enc)
            if enc == "utf-16" and not raw.startswith((b"\xff\xfe", b"\xfe\xff")):
                continue
            return text
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


# --------------------------------------------------------------------- PDF
def _read_pdf(path: Path) -> Tuple[str, List[Optional[int]]]:
    page_texts: List[str] = []
    errors = []
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        for p in reader.pages:
            try:
                page_texts.append(p.extract_text(extraction_mode="layout") or "")
            except TypeError:  # старые версии pypdf
                page_texts.append(p.extract_text() or "")
    except ImportError:
        errors.append("pypdf не установлен")
    except Exception as exc:  # битый PDF
        errors.append(f"pypdf: {exc}")

    if not any(t.strip() for t in page_texts):
        try:
            from pdfminer.high_level import extract_text  # type: ignore

            whole = extract_text(str(path)) or ""
            page_texts = whole.split("\f")
        except ImportError:
            errors.append("pdfminer.six не установлен")
        except Exception as exc:
            errors.append(f"pdfminer: {exc}")

    if not any(t.strip() for t in page_texts):
        hint = "; ".join(errors) if errors else "в PDF нет текстового слоя"
        raise LoaderError(f"Не удалось прочитать PDF {path.name}: {hint}. Установите зависимости: pip install -r requirements.txt")

    lines: List[str] = []
    pages: List[Optional[int]] = []
    for pno, t in enumerate(page_texts, start=1):
        for raw_line in t.splitlines():
            line = raw_line.replace("\t", " ").rstrip()
            line = re.sub(r" {3,}", "  ", line)
            lines.append(line)
            pages.append(pno)
    # схлопываем серии пустых строк
    out_l, out_p = [], []
    blank = 0
    for l, p in zip(lines, pages):
        if not l.strip():
            blank += 1
            if blank > 1:
                continue
        else:
            blank = 0
        out_l.append(l.strip())
        out_p.append(p)
    return "\n".join(_guess_headings(out_l)), out_p


def _guess_headings(lines: List[str]) -> List[str]:
    """В PDF нет разметки заголовков. Короткая строка с заглавной буквы, без точки
    в конце, окружённая пустыми строками, считается заголовком раздела."""
    out = list(lines)
    first = True
    for i, t in enumerate(lines):
        s = t.strip()
        if not s:
            continue
        prev_blank = i == 0 or not lines[i - 1].strip()
        next_blank = i + 1 >= len(lines) or not lines[i + 1].strip()
        words = s.split()
        if (
            prev_blank
            and next_blank
            and len(words) <= 8
            and s[0].isupper()
            and not re.search(r"[.,:;!?)»\"]$", s)
            and "  " not in s
            and not re.search(r"\d{2,}", s)
        ):
            out[i] = ("# " if first else "## ") + s
            first = False
    return out


# -------------------------------------------------------------------- DOCX
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _read_docx(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as z:
            xml = z.read("word/document.xml")
            styles = {}
            if "word/styles.xml" in z.namelist():
                st = ET.fromstring(z.read("word/styles.xml"))
                for s in st.iter(f"{_W}style"):
                    sid = s.get(f"{_W}styleId")
                    name_el = s.find(f"{_W}name")
                    if sid and name_el is not None:
                        styles[sid] = (name_el.get(f"{_W}val") or "").lower()
    except (zipfile.BadZipFile, KeyError) as exc:
        raise LoaderError(f"Файл {path.name} не похож на корректный DOCX: {exc}")

    root = ET.fromstring(xml)
    body = root.find(f"{_W}body")
    out: List[str] = []
    if body is None:
        return ""
    for el in body:
        if el.tag == f"{_W}p":
            out.append(_docx_paragraph(el, styles))
        elif el.tag == f"{_W}tbl":
            out.extend(_docx_table(el))
            out.append("")
    return "\n".join(out)


def _docx_text(el) -> str:
    parts = []
    for node in el.iter():
        if node.tag == f"{_W}t" and node.text:
            parts.append(node.text)
        elif node.tag == f"{_W}tab":
            parts.append("\t")
        elif node.tag in (f"{_W}br", f"{_W}cr"):
            parts.append(" ")
    return "".join(parts).strip()


def _docx_paragraph(p, styles) -> str:
    text = _docx_text(p)
    if not text:
        return ""
    ppr = p.find(f"{_W}pPr")
    style_name = ""
    is_list = False
    if ppr is not None:
        ps = ppr.find(f"{_W}pStyle")
        if ps is not None:
            sid = ps.get(f"{_W}val") or ""
            style_name = styles.get(sid, sid.lower())
        is_list = ppr.find(f"{_W}numPr") is not None
    m = re.search(r"(heading|заголовок)\s*(\d)", style_name)
    if m:
        return "#" * min(6, int(m.group(2)) + 1) + " " + text
    if style_name == "title":
        return "# " + text
    if is_list or "list" in style_name:
        return "* " + text
    return text


def _docx_table(tbl) -> List[str]:
    rows = []
    for tr in tbl.iter(f"{_W}tr"):
        cells = [_docx_text(tc).replace("|", "/") for tc in tr.findall(f"{_W}tc")]
        rows.append("| " + " | ".join(cells) + " |")
    if rows:
        ncols = rows[0].count("|") - 1
        rows.insert(1, "| " + " | ".join([":---"] * max(1, ncols)) + " |")
    return rows


# -------------------------------------------------------------------- HTML
class _HTMLToMarkdown(HTMLParser):
    BLOCK = {"p", "div", "section", "article", "br", "tr", "ul", "ol", "table", "pre", "blockquote"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: List[str] = []
        self.buf: List[str] = []
        self.skip = 0
        self.in_cell = False
        self.row: List[str] = []
        self.prefix = ""
        self.table_rows = 0

    def _flush(self) -> None:
        text = re.sub(r"\s+", " ", "".join(self.buf)).strip()
        self.buf = []
        if text:
            self.out.append(self.prefix + text)
        self.prefix = ""

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "head"):
            self.skip += 1
            return
        if re.fullmatch(r"h[1-6]", tag):
            self._flush()
            self.prefix = "#" * int(tag[1]) + " "
        elif tag == "li":
            self._flush()
            self.prefix = "* "
        elif tag in ("td", "th"):
            self.in_cell = True
            self.buf = []
        elif tag == "tr":
            self._flush()
            self.row = []
        elif tag == "table":
            self._flush()
            self.table_rows = 0
        elif tag in self.BLOCK:
            self._flush()

    def handle_endtag(self, tag):
        if tag in ("script", "style", "head"):
            self.skip = max(0, self.skip - 1)
            return
        if tag in ("td", "th"):
            self.row.append(re.sub(r"\s+", " ", "".join(self.buf)).strip().replace("|", "/"))
            self.buf = []
            self.in_cell = False
        elif tag == "tr":
            if self.row:
                self.out.append("| " + " | ".join(self.row) + " |")
                if self.table_rows == 0:
                    self.out.append("| " + " | ".join([":---"] * len(self.row)) + " |")
                self.table_rows += 1
            self.row = []
        elif tag == "table":
            self.table_rows = 0
            self._flush()
        elif re.fullmatch(r"h[1-6]", tag) or tag in self.BLOCK or tag == "li":
            self._flush()

    def handle_data(self, data):
        if not self.skip:
            self.buf.append(data)


def _read_html(text: str) -> str:
    parser = _HTMLToMarkdown()
    parser.feed(text)
    parser._flush()
    return html.unescape("\n".join(parser.out))
