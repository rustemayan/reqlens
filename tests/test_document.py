from reqlens.document import normalize


def test_normalize_unifies_quotes_dashes_markup():
    assert normalize("**Ф-1.** «Порог» — 1 000 ₸") == normalize('Ф-1. "порог" - 1 000 ₸')
    assert normalize("ёлка") == "елка"


def test_locate_exact_with_hint(example_doc):
    ev = example_doc.locate("Месяц определяется по дате совершения покупки в Алматы", hint_line=40)
    assert ev and ev.line_start == 38 and ev.match == "exact"


def test_locate_strips_line_prefix(example_doc):
    ev = example_doc.locate("L104| **REST type** | GET")
    assert ev and ev.line_start == 104


def test_locate_ellipsis_fragments(example_doc):
    ev = example_doc.locate("Покупка 31 января в 23:58 на 4 000 ₸ … Февральский счётчик вырос на 4 000 ₸")
    assert ev and ev.line_start == 64


def test_locate_fuzzy_replaces_quote_with_document_text(example_doc):
    ev = example_doc.locate("Хранить идентификатор карты, месяц, идентификатор операции, сумму покупки в тиинах и порог в тенге")
    assert ev and ev.line_start == 49 and ev.match == "fuzzy"
    assert "тиынах" in ev.quote  # подставлен реальный текст документа


def test_locate_rejects_fabricated_quote(example_doc):
    assert example_doc.locate("Сервис отправляет push в течение 5 минут после покупки") is None
    assert example_doc.locate("Система отправляет SMS при каждом входе") is None


def test_sections_and_title(example_doc):
    assert example_doc.title == "Уведомление о расходах по карте"
    assert example_doc.section_of(37).endswith("Функциональные требования")
    assert "Часть B" in example_doc.section_of(104)


def test_numbered_text_has_prefixes(example_doc):
    text = example_doc.numbered_text()
    assert "L037| * **Ф-1.**" in text
