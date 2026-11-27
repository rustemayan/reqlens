import zipfile

import pytest

from reqlens.loaders import LoaderError, load_document

_DOCX_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Функциональные требования</w:t></w:r></w:p>
<w:p><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr></w:pPr><w:r><w:t>Ф-1. Отчёт формируется быстро.</w:t></w:r></w:p>
<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Parameter</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Format</w:t></w:r></w:p></w:tc></w:tr>
<w:tr><w:tc><w:p><w:r><w:t>amount</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>integer</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
</w:body></w:document>"""

_STYLES_XML = """<?xml version="1.0" encoding="UTF-8"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:style w:styleId="Heading1"><w:name w:val="heading 1"/></w:style></w:styles>"""


def test_markdown(example_doc):
    assert example_doc.fmt == "Markdown"
    assert example_doc.line_count == 198
    assert len(example_doc.sha256) == 64


def test_docx_stdlib(tmp_path):
    p = tmp_path / "spec.docx"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("word/document.xml", _DOCX_XML)
        z.writestr("word/styles.xml", _STYLES_XML)
    doc = load_document(p)
    texts = [l.text for l in doc.lines]
    assert "## Функциональные требования" in texts
    assert "* Ф-1. Отчёт формируется быстро." in texts
    assert "| amount | integer |" in texts
    assert doc.locate("Отчёт формируется быстро").line_start == 2


def test_html(tmp_path):
    p = tmp_path / "spec.html"
    p.write_text("<html><body><h2>Данные</h2><p>Хранить сумму в тиынах.</p>"
                 "<table><tr><th>Parameter</th><th>Format</th></tr><tr><td>amount</td><td>string</td></tr></table>"
                 "<script>var x=1;</script></body></html>", encoding="utf-8")
    doc = load_document(p)
    texts = [l.text for l in doc.lines]
    assert "## Данные" in texts
    assert "| Parameter | Format |" in texts and "| :--- | :--- |" in texts
    assert not any("var x" in t for t in texts)


def test_pdf(tmp_path):
    reportlab = pytest.importorskip("reportlab")
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    p = tmp_path / "spec.pdf"
    c = canvas.Canvas(str(p), pagesize=A4)
    c.drawString(72, 800, "Functional requirements")
    c.drawString(72, 770, "F-1. The report is generated quickly.")
    c.showPage()
    c.drawString(72, 800, "F-2. Amounts are stored in cents.")
    c.save()
    doc = load_document(p)
    assert doc.fmt == "PDF"
    ev = doc.locate("Amounts are stored in cents")
    assert ev is not None and ev.page == 2


def test_missing_and_empty(tmp_path):
    with pytest.raises(LoaderError):
        load_document(tmp_path / "nope.md")
    empty = tmp_path / "empty.md"
    empty.write_text("   \n", encoding="utf-8")
    with pytest.raises(LoaderError):
        load_document(empty)


def test_cp1251_text(tmp_path):
    p = tmp_path / "legacy.txt"
    p.write_bytes("Требование: порог в тенге".encode("cp1251"))
    assert load_document(p).lines[0].text == "Требование: порог в тенге"
