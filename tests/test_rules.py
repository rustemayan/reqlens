from reqlens.document import Document
from reqlens.rules import run_rules


def _by_rule(findings):
    out = {}
    for f in findings:
        out.setdefault(f.rule_id, []).append(f)
    return out


def test_rules_on_example(example_doc):
    found = _by_rule(run_rules(example_doc))
    # подложенные организаторами дефекты находятся детерминированно
    assert found["R06"][0].severity == "critical"          # GET для сохранения
    assert found["R06"][0].evidence[0].line_start == 104
    assert found["R07"][0].evidence[1].line_start == 171    # 200 + ERROR
    assert {e.line_start for e in found["R08"][0].evidence} >= {49}  # тиыны vs тенге
    lines_r09 = {e.line_start for e in found["R09"][0].evidence}
    assert {143, 156, 144, 157} <= lines_r09               # integer vs string
    assert any(e.line_start == 57 for f in found["R01"] for e in f.evidence)  # «заполняются тестировщиками»
    assert found["R03"] and found["R04"] and found["R10"] and found["R11"]


def test_rules_do_not_fire_on_clean_text(tmp_path):
    text = (
        "# Спецификация\n\n## API\n\n| Поле | Значение |\n| :--- | :--- |\n| REST type | PUT |\n"
        "| Request URL | /v1/limits |\n\n## Данные\n\nВсе суммы хранятся в тиынах; 1 тенге = 100 тиын.\n"
    )
    doc = Document.from_text(tmp_path / "clean.md", text, "Markdown")
    assert run_rules(doc) == []


def test_vague_words(tmp_path):
    doc = Document.from_text(tmp_path / "v.md", "## НФТ\n\nОтчёт должен формироваться быстро и удобно.\n", "Markdown")
    f = [x for x in run_rules(doc) if x.rule_id == "R05"]
    assert f and "быстро" in f[0].title
