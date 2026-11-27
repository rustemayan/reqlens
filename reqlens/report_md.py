"""Отчёт в Markdown. Структура повторяет состав отчёта из ТЗ хакатона:
паспорт → нарезка → проверка каждого требования → проверка набора → сводка →
итог, плюс вопросы для уточнения и методика."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, List

from .criteria import (
    ADDRESSEE_RU,
    ADDRESSEES,
    ALL,
    INDIVIDUAL,
    SET,
    SEVERITY_DEFINITION,
    SEVERITY_ICON,
    SEVERITY_RANK,
    SEVERITY_RU,
    SUMMARY_GROUPS,
)
from .document import Evidence
from .models import AuditResult, Finding, Requirement
from .rules import RULE_DESCRIPTIONS
from .schemas import KIND_RU
from .segmenter import short_section
from .utils import human_size, plural
from .verdict import VERDICT_LABEL

TRACE_ICON = {"covered": "✅ покрыто", "partial": "⚠️ частично", "missing": "❌ нет"}


# ------------------------------------------------------------------ utils
def cell(text: str, limit: int = 0) -> str:
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    if limit and len(t) > limit:
        t = t[: limit - 1].rstrip() + "…"
    return t.replace("|", "\\|") or "—"


def show_quote(text: str) -> str:
    t = str(text or "").replace("**", "").strip()
    if t.startswith("|"):
        t = " | ".join(c.strip() for c in t.strip().strip("|").split("|") if c.strip())
    t = re.sub(r"^\s*[*\-+]\s+", "", t)
    return re.sub(r"\s+", " ", t)


def anchor(fid: str) -> str:
    return fid.lower()


def link(fid: str) -> str:
    return f"[{fid}](#{anchor(fid)})"


def where(e: Evidence) -> str:
    return e.where()


def worst(findings: List[Finding]) -> str:
    if not findings:
        return ""
    return min(findings, key=lambda f: SEVERITY_RANK.get(f.severity, 9)).severity


def sev_badge(sev: str) -> str:
    return f"{SEVERITY_ICON.get(sev, '')} {SEVERITY_RU.get(sev, sev)}"


def source_label(f: Finding) -> str:
    if f.source == "rule":
        return f"статическое правило {f.rule_id}"
    if f.source == "rule+llm":
        return f"статическое правило {f.rule_id} + анализ LLM"
    return "анализ LLM"


# ---------------------------------------------------------------- finding
def render_finding(f: Finding, reqs_by_id: Dict[str, Requirement]) -> List[str]:
    out = [f'<a id="{anchor(f.id)}"></a>']
    out.append(f"**{SEVERITY_ICON.get(f.severity, '')} {f.id} · {ALL[f.criterion].name}** — **{f.title}**")
    out.append("")
    places = []
    for e in f.evidence[:5]:
        places.append(f"{where(e)}: «{show_quote(e.quote)}»")
    out.append("- **Где в документе:** " + "; ".join(places))
    if f.explanation:
        out.append(f"- **В чём проблема:** {f.explanation}")
    if f.recommendation:
        out.append(f"- **Как исправить:** {f.recommendation}")
    if f.question:
        out.append(f"- **Уточняющий вопрос** ({ADDRESSEE_RU.get(f.question.to, f.question.to)}): {f.question.text}")
    if f.test_idea:
        out.append(f"- **Как проверить (тест-идея):** {f.test_idea}")
    tail = [f"критичность: {SEVERITY_RU.get(f.severity, f.severity)}", f"источник: {source_label(f)}", f"критерий: {ALL[f.criterion].ref}"]
    rel = [r for r in f.req_ids if r in reqs_by_id]
    if rel:
        tail.insert(0, "требования: " + ", ".join(f"[{r}](#{r.lower()})" for r in rel))
    out.append("- *" + " · ".join(tail) + "*")
    out.append("")
    return out


def _questions_block(questions: List[Finding], with_links: bool) -> List[str]:
    out: List[str] = []
    qn = 0
    for to in ADDRESSEES:
        group = sorted([f for f in questions if f.question and f.question.to == to], key=lambda f: (f.rank, f.first_line))
        if not group:
            continue
        out.append(f"### {ADDRESSEE_RU[to]}")
        out.append("")
        for f in group:
            qn += 1
            ref = link(f.id) if with_links else f.id
            where_s = ", ".join(f"стр. {e.line_start}" for e in f.evidence[:3])
            out.append(f"{qn}. {SEVERITY_ICON[f.severity]} {f.question.text} — *{ref}: {f.title}; {where_s}*")
            if not with_links:
                out.append("   - Ответ: ")
        out.append("")
    return out


def render_questions(res: AuditResult) -> str:
    """Опросник для заинтересованных сторон: можно сразу отправить бизнесу, архитектору, ИБ."""
    doc = res.document
    questions = [f for f in res.findings if f.question]
    crit = len([f for f in questions if f.severity == "critical"])
    L = [
        f"# Вопросы по требованиям «{doc.title}»",
        "",
        f"По итогам автоматического ревью документа `{doc.path.name}` ({res.meta.get('started_at', '')}) агент ReqLens "
        f"сформировал {len(questions)} {plural(len(questions), 'вопрос', 'вопроса', 'вопросов')}, из них {crit} — по "
        "критичным дефектам, без ответа на которые разработку начинать нельзя. Пожалуйста, впишите ответы под вопросами. "
        "Номера строк указаны по проверяемому документу; подробности по каждому дефекту (D-…) — в полном отчёте.",
        "",
    ]
    L.extend(_questions_block(questions, with_links=False))
    L.append(f"*Сформировано: {res.meta['agent']['label']}.*")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------- render
def render_markdown(res: AuditResult) -> str:
    doc = res.document
    reqs = res.requirements
    reqs_by_id = {r.id: r for r in reqs}
    meta = res.meta
    findings = res.findings
    sev_count = Counter(f.severity for f in findings)
    req_level = [f for f in findings if f.scope == "requirement"]
    set_level = [f for f in findings if f.scope == "set"]
    questions = [f for f in findings if f.question]
    reqs_with = {r for f in req_level for r in f.req_ids}
    v = res.verdict

    L: List[str] = []
    w = L.append

    w("# Отчёт о статической проверке требований")
    w("")
    w(f"> Документ «{doc.title}». Отчёт сформирован автоматически агентом **{meta['agent']['label']}** "
      "и не редактировался вручную. Каждое замечание ссылается на номер строки и цитату проверяемого документа; "
      "цитаты сверены с исходным текстом программно.")
    w("")

    # ---------------------------------------------------------- 1 passport
    w("## 1. Паспорт прогона")
    w("")
    w("| Параметр | Значение |")
    w("| :--- | :--- |")
    w(f"| Проверяемый документ | «{cell(doc.title)}» — файл `{doc.path.name}` |")
    w(f"| Формат и объём | {doc.fmt} · {doc.line_count} {plural(doc.line_count, 'строка', 'строки', 'строк')} · {human_size(doc.raw_bytes)} |")
    w(f"| Контрольная сумма документа | SHA-256 `{doc.sha256[:16]}…` |")
    off = meta.get("utc_offset", "")
    off_s = f" (UTC{off[:3]}:{off[3:]})" if off else ""
    w(f"| Дата и время прогона | {meta['started_at']}{off_s} |")
    if meta.get("model") and meta["model"] != "—":
        used = meta.get("used_models") or [meta["model"]]
        model_s = ", ".join(f"`{m}`" for m in used)
        if used != [meta["model"]]:
            model_s += f" (основная — `{meta['model']}`)"
        w(f"| Модель | {model_s} · {meta['provider']} |")
    else:
        w("| Модель | не использовалась (офлайн-режим) |")
    w(f"| Версия агента | {cell(meta['agent']['label'])} |")
    w(f"| Режим | {meta['mode']}" + (" · ревью-скептик: включено" if meta.get("review_pass") else "") + " |")
    llm = meta.get("llm") or {}
    dur = f"{meta['duration_s']} с"
    if llm:
        dur += f" · вызовов LLM: {llm.get('calls', 0)} (из кэша: {llm.get('cached_calls', 0)})"
        if llm.get("input_tokens"):
            dur += f" · токены: {llm['input_tokens']:,} вход / {llm['output_tokens']:,} выход".replace(",", " ")
    w(f"| Длительность анализа | {dur} |")
    w("| Критерии | ISO/IEC/IEEE 29148:2018 п. 5.2.5, 5.2.6; IEEE Std 830-1998 п. 4.3 |")
    w("")
    if v:
        w(f"> **Коротко.** Разработка: **{VERDICT_LABEL[v.development]}**. Тестирование: **{VERDICT_LABEL[v.testing]}**. "
          f"Требований: {len(reqs)}. Дефектов: {len(findings)} "
          f"(🔴 {sev_count['critical']} · 🟡 {sev_count['major']} · 🟢 {sev_count['minor']}). "
          f"Уточняющих вопросов: {len(questions)}.")
        if v.summary:
            w(">")
            w(f"> {v.summary}")
        w("")
    if res.warnings:
        w("> ⚠️ **Предупреждения прогона:** " + " ".join(res.warnings))
        w("")

    # ---------------------------------------------------------- 2 slicing
    numbered = [r for r in reqs if r.doc_id]
    w("## 2. Как агент нарезал требования")
    w("")
    w(f"Выделено **{len(reqs)}** {plural(len(reqs), 'требование', 'требования', 'требований')}: "
      f"{len(numbered)} с идентификатором в документе, {len(reqs) - len(numbered)} без идентификатора "
      "(агент присвоил свой ID и сохранил цитату). "
      f"Нарезка покрывает {meta.get('coverage', 0):.0%} содержательных строк документа "
      "(без служебной шапки, чек-листа и блока согласующих).")
    w("")
    w("| ID в отчёте | ID в документе | Раздел | Строки | Тип | Краткая формулировка | Цитата из документа |")
    w("| :--- | :--- | :--- | :---: | :--- | :--- | :--- |")
    for r in reqs:
        w(f"| [{r.id}](#{r.id.lower()}) | {cell(r.doc_id) if r.doc_id else '— (присвоен агентом)'} | "
          f"{cell(short_section(r.section), 45)} | {r.lines_label} | {KIND_RU.get(r.kind, r.kind)} | "
          f"{cell(r.summary, 110)} | «{cell(show_quote(r.quote), 140)}» |")
    w("")

    # ------------------------------------------------- 3 each requirement
    w("## 3. Проверка каждого требования")
    w("")
    w("### 3.1. Матрица соответствия критериям")
    w("")
    w("Критерии ISO/IEC/IEEE 29148 §5.2.5 и IEEE 830 §4.3. Обозначения: ✅ — нарушений нет, "
      "🟢 — замечание, 🟡 — существенный дефект, 🔴 — критичный дефект.")
    w("")
    w("| ID | ID в док. | " + " | ".join(c.short for c in INDIVIDUAL) + " | Итог |")
    w("| :--- | :--- | " + " | ".join(":---:" for _ in INDIVIDUAL) + " | :--- |")
    for r in reqs:
        rf = res.findings_for(r.id)
        cells = []
        for c in INDIVIDUAL:
            s = worst([f for f in rf if f.criterion == c.id])
            cells.append(SEVERITY_ICON[s] if s else "✅")
        total = f"{SEVERITY_ICON[worst(rf)]} {len(rf)}" if rf else "✅ 0"
        w(f"| [{r.id}](#{r.id.lower()}) | {cell(r.doc_id) if r.doc_id else '—'} | " + " | ".join(cells) + f" | {total} |")
    w("")
    w("Расшифровка столбцов: " + "; ".join(f"**{c.short}** — {c.name}" for c in INDIVIDUAL) + ".")
    w("")

    w("### 3.2. Разбор по требованиям")
    w("")
    for r in reqs:
        w(f'<a id="{r.id.lower()}"></a>')
        title = f"{r.id} · {r.doc_id}" if r.doc_id else f"{r.id} · без номера в документе"
        w(f"#### {title} — {r.summary}")
        w("")
        w(f"*Раздел:* {short_section(r.section)} · *строки:* {r.lines_label} · *тип:* {KIND_RU.get(r.kind, r.kind)}")
        w("")
        w(f"> «{show_quote(r.quote)}»")
        w("")
        primary = [f for f in res.findings_for(r.id, primary_only=True)]
        related = [f for f in res.findings_for(r.id) if f not in primary]
        in_set = [f for f in set_level if r.id in f.req_ids]
        if r.assessment:
            w(f"**Оценка агента:** {r.assessment}")
            w("")
        if not primary and not related:
            w("✅ **Нарушений по критериям отдельного требования не выявлено.**")
            w("")
        for f in primary:
            L.extend(render_finding(f, reqs_by_id))
        if related:
            w("Также затрагивают это требование: " + "; ".join(
                f"{SEVERITY_ICON[f.severity]} {link(f.id)} ({f.primary_req}) — {f.title}" for f in related) + ".")
            w("")
        if in_set:
            w("В проверке набора: " + "; ".join(f"{SEVERITY_ICON[f.severity]} {link(f.id)} — {f.title}" for f in in_set) + ".")
            w("")

    # ------------------------------------------------------------- 4 set
    w("## 4. Проверка набора требований целиком")
    w("")
    w("Свойства спецификации как системы по ISO/IEC/IEEE 29148 §5.2.6 и IEEE 830 §4.3.")
    w("")
    for i, c in enumerate(SET, start=1):
        cf = res.set_findings(c.id)
        cf.sort(key=lambda f: (f.rank, f.first_line))
        cnt = Counter(f.severity for f in cf)
        status = " · ".join(f"{SEVERITY_ICON[s]} {cnt[s]}" for s in ("critical", "major", "minor") if cnt[s])
        if not status:
            status = "⚠️ см. таблицу терминов" if (c.id == "set_comprehensible" and res.glossary) else "✅ нарушений не выявлено"
        w(f"### 4.{i}. {c.name} — {status}")
        w("")
        w(f"*{c.ref}.* {c.definition}")
        w("")
        for f in cf:
            L.extend(render_finding(f, reqs_by_id))
        if c.id == "set_comprehensible" and res.glossary:
            w("**Терминология: одно понятие — разные названия**")
            w("")
            w("| Понятие | Как названо в документе | Рекомендация |")
            w("| :--- | :--- | :--- |")
            for g in res.glossary:
                variants = "; ".join(f"«{cell(show_quote(e.quote), 60)}» (стр. {e.line_start})" for e in g.variants[:6])
                w(f"| {cell(g.concept)} | {variants} | {cell(g.recommendation)} |")
            w("")
        if c.id == "traceable" and res.trace:
            w("**Матрица прослеживаемости: цель / функция → интерфейс → данные → приёмка**")
            w("")
            w("| Требование | UI | API | Данные | Критерии приёмки | Статус | Чего не хватает |")
            w("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            for t in res.trace:
                w(f"| [{t.source}](#{t.source.lower()}) {cell(t.title, 60)} | {cell(t.ui)} | {cell(t.api)} | {cell(t.data)} | "
                  f"{cell(t.acceptance)} | {TRACE_ICON.get(t.status, t.status)} | {cell(t.note)} |")
            w("")
    n_set = len(SET)
    w(f"### 4.{n_set + 1}. Непокрытые сценарии и предлагаемые критерии приёмки")
    w("")
    if res.missing:
        w("Сценарии, которые документ не описывает, но которые обязательно встретятся в разработке и эксплуатации. "
          "Для каждого агент предлагает критерий приёмки — его можно сразу перенести в раздел «Критерии приёмки».")
        w("")
        w("| № | Сценарий | Почему важно | Предлагаемый критерий приёмки | Связь |")
        w("| :---: | :--- | :--- | :--- | :--- |")
        for i, m in enumerate(res.missing, start=1):
            ac = f"**Дано:** {cell(m.given)}<br>**Когда:** {cell(m.when)}<br>**Тогда:** {cell(m.then)}"
            rel = ", ".join(f"[{x}](#{x.lower()})" for x in m.req_ids) or "—"
            w(f"| {i} | {cell(m.scenario)} | {cell(m.why)} | {ac} | {rel} |")
        w("")
    else:
        w("Анализ непокрытых сценариев не выполнялся (нужен этап LLM)." if not res.meta.get("llm") else "Непокрытых сценариев не выявлено.")
        w("")

    # --------------------------------------------------------- 5 summary
    w("## 5. Сводка")
    w("")
    w("| Показатель | Значение |")
    w("| :--- | :--- |")
    w(f"| Требований выделено | {len(reqs)} (с ID в документе: {len(numbered)}, без ID: {len(reqs) - len(numbered)}) |")
    w(f"| Требований без замечаний | {len([r for r in reqs if r.id not in reqs_with])} |")
    w(f"| Требований с замечаниями | {len([r for r in reqs if r.id in reqs_with])} |")
    w(f"| Всего нарушений | **{len(findings)}** (уровень требования: {len(req_level)}, уровень набора: {len(set_level)}) |")
    w(f"| 🔴 Критичных (блокеры) | **{sev_count['critical']}** |")
    w(f"| 🟡 Существенных | {sev_count['major']} |")
    w(f"| 🟢 Замечаний | {sev_count['minor']} |")
    w(f"| Уточняющих вопросов | {len(questions)} |")
    w(f"| Предложено новых критериев приёмки | {len(res.missing)} |")
    w("")
    w("**Распределение по критериям**")
    w("")
    w("| Критерий | Стандарт | 🔴 | 🟡 | 🟢 | Всего |")
    w("| :--- | :--- | :---: | :---: | :---: | :---: |")
    by_c = defaultdict(Counter)
    for f in findings:
        by_c[f.criterion][f.severity] += 1
    for c in INDIVIDUAL + SET:
        if by_c[c.id]:
            cc = by_c[c.id]
            w(f"| {c.name} | {c.ref} | {cc['critical'] or ''} | {cc['major'] or ''} | {cc['minor'] or ''} | {sum(cc.values())} |")
    w("")

    crit = sorted([f for f in findings if f.severity == "critical"], key=lambda f: f.first_line)
    w("### Критичные дефекты — блокеры разработки")
    w("")
    if not crit:
        w("Критичных дефектов не выявлено.")
        w("")
    used = set()
    for group, cids in SUMMARY_GROUPS:
        items = [f for f in crit if f.criterion in cids and f.id not in used]
        if not items:
            continue
        w(f"**{group} формулировки**" if group in ("Противоречивые", "Непроверяемые", "Неполные", "Двусмысленные") else f"**{group}**")
        w("")
        for f in items:
            used.add(f.id)
            lines = ", ".join(str(e.line_start) for e in f.evidence[:4])
            reqs_s = ", ".join(f.req_ids[:4]) or "набор"
            w(f"- {SEVERITY_ICON['critical']} {link(f.id)} — {f.title} *(строки {lines}; {reqs_s})*")
        w("")
    major = [f for f in findings if f.severity == "major"]
    if major:
        w("### Существенные дефекты по категориям")
        w("")
        for group, cids in SUMMARY_GROUPS:
            items = [f for f in major if f.criterion in cids]
            if items:
                w(f"- **{group}** ({len(items)}): " + "; ".join(f"{link(f.id)} {f.title}" for f in items))
        w("")

    # ---------------------------------------------------------- 6 verdict
    w("## 6. Итог")
    w("")
    if v:
        w("| Вопрос | Вердикт | Обоснование |")
        w("| :--- | :--- | :--- |")
        w(f"| Можно ли начинать **разработку**? | **{VERDICT_LABEL[v.development]}** | {cell(v.development_reason)} |")
        w(f"| Можно ли начинать **тестирование** (тест-дизайн)? | **{VERDICT_LABEL[v.testing]}** | {cell(v.testing_reason)} |")
        w("")
        if v.summary:
            w(v.summary)
            w("")
    w("### Что исправить в первую очередь")
    w("")
    for i, p in enumerate(res.priorities, start=1):
        w(f"{i}. **{p.action}** — " + ", ".join(link(x) for x in p.finding_ids))
    w("")
    w("**Критерий готовности документа:** закрыты все 🔴 критичные дефекты, получены ответы на уточняющие вопросы "
      "раздела 7 к бизнесу и архитектору, раздел «Критерии приёмки» дополнен сценариями из п. 4."
      f"{n_set + 1}, заполнены версия документа и блок согласующих.")
    w("")

    # -------------------------------------------------------- 7 questions
    w("## 7. Вопросы для уточнения")
    w("")
    w("Вопросы сгруппированы по адресатам и отсортированы по критичности: ответ на каждый закрывает связанный дефект.")
    w("")
    L.extend(_questions_block(questions, with_links=True))

    # --------------------------------------------------------- appendices
    w("## Приложение A. Методика проверки")
    w("")
    w("1. **Чтение документа** (Markdown, TXT, PDF, DOCX, HTML) и нумерация строк — все ссылки отчёта ведут на эти номера.")
    w("2. **Статические правила** — детерминированные проверки типовых дефектов (список ниже); работают без модели.")
    w("3. **Нарезка на требования** моделью с программной проверкой: каждая цитата требования ищется в тексте документа.")
    w("4. **Проверка каждого требования** пакетами по критериям ISO 29148 §5.2.5 / IEEE 830 §4.3 с контекстом всего документа.")
    w("5. **Проверка набора** — два прохода: согласованность и терминология; полнота, прослеживаемость и непокрытые сценарии.")
    w("6. **Проверка доказательств** — цитаты всех находок сверяются с документом; находки без подтверждённой цитаты отбрасываются.")
    w("7. **Ревью-скептик** — отдельный проход модели удаляет ложные срабатывания, объединяет дубли и калибрует критичность.")
    w("8. **Вердикт** вычисляется детерминированно по критичности дефектов; модель формулирует только обоснование.")
    w("")
    w("**Критерии**")
    w("")
    w("| ID | Критерий | Стандарт | Что проверяется |")
    w("| :--- | :--- | :--- | :--- |")
    for c in INDIVIDUAL + SET:
        w(f"| `{c.id}` | {c.name} | {c.ref} | {cell(c.check)} |")
    w("")
    w("**Шкала критичности**")
    w("")
    for s in ("critical", "major", "minor"):
        w(f"- {SEVERITY_ICON[s]} **{SEVERITY_RU[s]}** — {SEVERITY_DEFINITION[s]}")
    w("")
    st = res.stats
    w("**Контроль доказательности этого прогона**")
    w("")
    w(f"- Цитат проверено: {st.get('quotes_total', 0)} — точных совпадений {st.get('quotes_exact', 0)}, "
      f"нечётких (заменены текстом документа) {st.get('quotes_fuzzy', 0)}, отклонено {st.get('quotes_rejected', 0)}.")
    w(f"- Находок отброшено без подтверждённой цитаты: {st.get('findings_unverified', 0)}; "
      f"удалено ревьюером как ложные: {st.get('dropped_review', 0)}.")
    w(f"- Объединено дублей: автоматически {st.get('merged_auto', 0)}, ревьюером {st.get('merged_review', 0)}; "
      f"критичность скорректирована ревьюером: {st.get('severity_changed_review', 0)}.")
    rejected = st.get("rejected") or []
    if rejected:
        w("- Отброшенные находки (для прозрачности): " + "; ".join(f"«{cell(t, 70)}» — {cell(r, 80)}" for t, r in rejected[:12])
          + (" …" if len(rejected) > 12 else "") + ".")
    w("")
    w("**Статические правила**")
    w("")
    fired = res.stats.get("rules_fired") or Counter(f.rule_id for f in findings if f.rule_id)
    w("| Правило | Что ищет | Сработало |")
    w("| :--- | :--- | :---: |")
    for rid, desc in RULE_DESCRIPTIONS.items():
        w(f"| {rid} | {desc} | {'да' if fired.get(rid) else '—'} |")
    w("")

    w("## Приложение B. Журнал прогона")
    w("")
    w("| Этап | Время, с | Статус | Детали |")
    w("| :--- | ---: | :--- | :--- |")
    for s in res.stages:
        w(f"| {s.name} | {s.seconds:.1f} | {s.status} | {cell(s.detail, 160)} |")
    w("")
    w("---")
    w(f"*Сформировано: {meta['agent']['label']} · {meta['started_at']}.*")
    w("")
    return "\n".join(L)
