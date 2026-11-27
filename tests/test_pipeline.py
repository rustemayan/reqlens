import json

from fake_llm import make_fake_client

from reqlens.pipeline import run_audit
from reqlens.report_export import to_json_dict
from reqlens.report_md import render_markdown


def _run(tmp_path, example_path, **kw):
    client = make_fake_client(tmp_path, **kw)
    return client, run_audit(example_path, client, batch_size=8)


def test_full_pipeline_with_fake_llm(tmp_path, example_path):
    client, res = _run(tmp_path, example_path)

    # нарезка: выдуманное требование и дубль отброшены, недостающее дополнено структурно
    assert all(r.doc_id != "Ф-9" for r in res.requirements)
    assert [r.doc_id for r in res.requirements if r.doc_id.startswith("Ф-")] == ["Ф-1", "Ф-2", "Ф-3", "Ф-4"]

    titles = [f.title for f in res.findings]
    # находка с выдуманной цитатой отброшена и записана в журнал
    assert "Не описаны покупки в валюте" not in titles
    assert any(t == "Не описаны покупки в валюте" for t, _ in res.stats["rejected"])
    # дубль правила GET и находки LLM склеен
    get = [f for f in res.findings if f.rule_id == "R06"]
    assert len(get) == 1 and get[0].source == "rule+llm"
    assert not any("Метод GET используется" in t for t in titles)
    # противоречия найдены и цитаты привязаны к строкам
    contra = next(f for f in res.findings if "Ф-4" in f.title)
    assert {e.line_start for e in contra.evidence} == {39, 40}
    # сбои модели пережиты: битый JSON, 429, обрезанный ответ
    assert client.provider.calls.count("segment") == 2
    assert res.verdict.development == "blocked"
    # глоссарий без выдуманного варианта, трассировка без несуществующих REQ
    assert res.glossary and all(e.line_start != 1 for e in res.glossary[0].variants)
    assert all(t.source.startswith("REQ-") and t.source != "REQ-777" for t in res.trace)
    assert res.missing and "REQ-555" not in res.missing[0].req_ids
    # id финальные и уникальные
    ids = [f.id for f in res.findings]
    assert len(ids) == len(set(ids)) and ids[0] == "D-001"


def test_report_has_all_required_sections(tmp_path, example_path):
    _, res = _run(tmp_path, example_path)
    md = render_markdown(res)
    for heading in [
        "## 1. Паспорт прогона",
        "## 2. Как агент нарезал требования",
        "## 3. Проверка каждого требования",
        "## 4. Проверка набора требований целиком",
        "## 5. Сводка",
        "## 6. Итог",
        "## 7. Вопросы для уточнения",
    ]:
        assert heading in md
    assert "fake-model" in md and "Версия агента" in md
    assert "| ID в отчёте | ID в документе |" in md
    assert "Что исправить в первую очередь" in md
    data = to_json_dict(res)
    json.dumps(data, ensure_ascii=False)
    assert data["verdict"]["development"] == "blocked"


def test_second_run_uses_cache(tmp_path, example_path):
    client1, res1 = _run(tmp_path, example_path)
    client2 = make_fake_client(tmp_path)
    res2 = run_audit(example_path, client2, batch_size=8)
    assert client2.provider.calls == []  # всё из кэша
    assert [f.title for f in res1.findings] == [f.title for f in res2.findings]


def test_offline_mode(example_path):
    res = run_audit(example_path, None)
    assert res.requirements and res.findings
    assert all(f.source == "rule" for f in res.findings)
    assert res.verdict.development == "blocked"
    assert "только статические правила" in res.meta["mode"]
