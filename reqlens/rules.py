"""Детерминированные статические правила.

Правила не зависят от модели и всегда дают проверяемые ссылки на строки.
Они страхуют LLM от пропуска типовых дефектов и работают в режиме --offline.
Правила общие (не подогнаны под конкретный документ): ищут паттерны,
характерные для любых ТЗ и спецификаций API.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from .document import Document, Evidence, normalize
from .models import Finding, Question


@dataclass
class Table:
    start: int
    header: List[str]
    rows: List[Tuple[int, List[str]]]
    section: str


def _split_row(text: str) -> List[str]:
    t = text.strip()
    if t.startswith("|"):
        t = t[1:]
    if t.endswith("|"):
        t = t[:-1]
    return [c.strip() for c in t.split("|")]


def parse_tables(doc: Document) -> List[Table]:
    tables: List[Table] = []
    i = 0
    lines = doc.lines
    while i < len(lines):
        t = lines[i].text.strip()
        nxt = lines[i + 1].text.strip() if i + 1 < len(lines) else ""
        if t.startswith("|") and re.fullmatch(r"\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?", nxt):
            header = [_clean_cell(c) for c in _split_row(t)]
            rows = []
            j = i + 2
            while j < len(lines) and lines[j].text.strip().startswith("|"):
                rows.append((lines[j].no, [_clean_cell(c) for c in _split_row(lines[j].text)]))
                j += 1
            tables.append(Table(lines[i].no, header, rows, lines[i].section))
            i = j
            continue
        i += 1
    return tables


def _clean_cell(c: str) -> str:
    return re.sub(r"[*`]", "", c).strip()


_counter = {"n": 0}


def _finding(
    rule_id: str,
    criterion: str,
    severity: str,
    title: str,
    evidence: List[Evidence],
    explanation: str,
    recommendation: str,
    q_to: str,
    q_text: str,
    test_idea: str = "",
) -> Finding:
    _counter["n"] += 1
    return Finding(
        uid=f"R{_counter['n']}",
        scope="set",
        criterion=criterion,
        severity=severity,
        title=title,
        evidence=evidence,
        explanation=explanation,
        recommendation=recommendation,
        question=Question(q_to, q_text),
        test_idea=test_idea,
        source="rule",
        rule_id=rule_id,
        stage="rules",
    )


# ----------------------------------------------------------------- rules
_PLACEHOLDER = re.compile(
    r"(заполня\w*\s+(тестировщик|аналитик|разработчик|позже|позднее|в\s+процессе)\w*|\bTBD\b|\bTODO\b|\bуточняется\b|"
    r"будет\s+(описан|добавлен|уточн|определ)\w*|\bзаглушка\b|lorem ipsum|\bне\s+заполнен[аоы]?\b)",
    re.I,
)


def rule_placeholders(doc: Document, tables: List[Table]) -> List[Finding]:
    out = []
    for ln in doc.lines:
        m = _PLACEHOLDER.search(ln.text)
        if not m:
            continue
        low = ln.text.lower()
        ev = [doc.evidence_for_lines(ln.no)]
        if "верси" in low or "дата" in low:
            out.append(_finding(
                "R01", "modifiable", "minor", "Не указаны версия и дата документа", ev,
                "Без версии и даты невозможно понять, по какой редакции велась разработка и тестирование, "
                "и отследить изменения требований (IEEE 830 §4.3.7).",
                "Заполнить версию, дату и историю изменений; фиксировать версию при каждом согласовании.",
                "analyst", "Какая версия и дата у этой редакции документа и где ведётся история изменений?",
            ))
        else:
            section = ln.section.split(" › ")[-1] if ln.section else "документа"
            out.append(_finding(
                "R01", "set_complete", "major", f"Раздел «{section}» содержит заглушку вместо содержания", ev,
                f"Обнаружена заглушка «{m.group(0)}». Обязательный раздел не заполнен — по нему нельзя "
                "спланировать работу и проверить результат; набор требований неполон (ISO 29148 §5.2.6).",
                "Заполнить раздел до передачи в разработку: для тестирования — перечень тест-сценариев, "
                "связанных с ID требований, включая граничные и негативные случаи.",
                "qa", f"Кто и к какому сроку заполнит раздел «{section}», и будет ли он согласован до старта разработки?",
            ))
    return out


_EMPTY_VALUE = re.compile(r"^\s*[`]?[—–\-]?[`]?\s*$")
_INLINE_EMPTY = re.compile(r"([A-Za-zА-Яа-я][\w ]{1,20}?)\s*:\s*[`]?\s*[—–]\s*[`]?\s*(?=$|[|.;,])")


def rule_empty_fields(doc: Document, tables: List[Table]) -> List[Finding]:
    ev: List[Evidence] = []
    names: List[str] = []
    for t in tables:
        if len(t.header) != 2:
            continue
        for no, cells in t.rows:
            if len(cells) >= 2 and cells[0] and _EMPTY_VALUE.match(cells[1]):
                ev.append(doc.evidence_for_lines(no))
                names.append(cells[0])
            elif len(cells) >= 2:
                for m in _INLINE_EMPTY.finditer(cells[1]):
                    ev.append(doc.evidence_for_lines(no))
                    names.append(f"{cells[0]}: {m.group(1).strip()}")
    if not ev:
        return []
    listed = ", ".join(f"«{n}»" for n in names[:8])
    return [_finding(
        "R02", "set_complete", "minor", "Не заполнены обязательные поля паспорта/спецификации", ev[:6],
        f"Поля без значения: {listed}. Без них нельзя развернуть и проверить решение в целевом окружении.",
        "Заполнить значения или явно указать «не применимо» с причиной.",
        "analyst", f"Когда будут определены значения полей {listed}?",
    )]


_YESNO = re.compile(r"Да\s*/\s*Нет", re.I)
_MARKED = re.compile(r"(\[x\]|\[х\]|✓|✔|☑|\bДа\b(?!\s*/))", re.I)


def rule_unfilled_checklist(doc: Document, tables: List[Table]) -> List[Finding]:
    rows = []
    for t in tables:
        for no, cells in t.rows:
            joined = " | ".join(cells[1:])
            if _YESNO.search(joined) and not _MARKED.search(_YESNO.sub("", joined)):
                rows.append(no)
    if len(rows) < 2:
        return []
    ev = [doc.evidence_for_lines(rows[0]), doc.evidence_for_lines(rows[-1])]
    return [_finding(
        "R03", "set_validatable", "minor", f"Чек-лист ревью требований не заполнен ({len(rows)} пунктов без отметок)", ev,
        "Ни один пункт чек-листа не отмечен — нет подтверждения, что требования прошли ревью, архитектура и ИБ "
        "согласованы, предусмотрен откат. Документ не провалидирован заинтересованными сторонами.",
        "Пройти чек-лист: проставить отметки «Создано/Проверено», для «Нет» — указать план устранения.",
        "analyst", "Кто проводит ревью по чек-листу и какие пункты сейчас фактически не выполнены (откат, ИБ, архитектура)?",
    )]


def rule_empty_tables(doc: Document, tables: List[Table]) -> List[Finding]:
    out = []
    for t in tables:
        if t.rows and all(not any(c for c in cells) for _, cells in t.rows):
            hdr_line = t.start
            ev = [doc.evidence_for_lines(hdr_line)]
            section = t.section.split(" › ")[-1] if t.section else "таблица"
            out.append(_finding(
                "R04", "modifiable", "minor", f"Таблица «{section}» не заполнена", ev,
                f"Таблица с колонками «{' / '.join(h for h in t.header if h)}» пуста — не зафиксированы ответственные "
                "и факт согласования.",
                "Заполнить таблицу: подразделения, ответственные, даты согласования.",
                "business", "Кто из подразделений должен согласовать документ и в какие сроки?",
            ))
    return out


_VAGUE = re.compile(
    r"\b(быстр(о|ый|ая|ое)|оперативн\w*|своевременн\w*|удобн\w*|интуитивн\w*|достаточн\w*|надёжн\w*|надежн\w*|"
    r"эффективн\w*|оптимальн\w*|максимальн\w+|по возможности|при необходимости|как правило|"
    r"и т\.\s?д\.?|и т\.\s?п\.?|и др\.|гибк\w+|легко|приемлем\w*|разумн\w*|user[- ]friendly|asap|"
    r"в кратчайшие сроки|в разумные сроки|высок\w+ производительн\w+)\b",
    re.I,
)


def rule_vague_words(doc: Document, tables: List[Table]) -> List[Finding]:
    out = []
    for ln in doc.lines:
        m = _VAGUE.search(ln.text)
        if m and not ln.text.lstrip().startswith("#"):
            word = m.group(0)
            out.append(_finding(
                "R05", "unambiguous", "major", f"Размытая формулировка «{word}»", [doc.evidence_for_lines(ln.no)],
                f"Слово «{word}» допускает разные толкования и не даёт измеримого критерия (ISO 29148 §5.2.5).",
                "Заменить измеримым значением: число, единица измерения, условие и перцентиль.",
                "analyst", f"Какое конкретное измеримое значение стоит за «{word}»?",
            ))
        if len(out) >= 8:
            break
    return out


_METHOD = re.compile(r"(rest\s*type|http\s*method|method|метод(?!\w))[^|\n]*\|?\s*`?(GET|POST|PUT|PATCH|DELETE)\b`?", re.I)
_MUTATION = re.compile(
    r"(save|update|create|delete|remove|\bset[A-Z_]|\bset\b|add[A-Z_]|edit|change|patch|сохран\w*|измен\w*|удал\w*|"
    r"созда\w*|добав\w*|замен\w*|запис\w*)",
    re.I,
)


def _part_of(section: str) -> str:
    return section.split(" › ")[0] if section else ""


def rule_get_mutation(doc: Document, tables: List[Table]) -> List[Finding]:
    out = []
    for ln in doc.lines:
        m = _METHOD.search(ln.text)
        if not m or m.group(2).upper() != "GET":
            continue
        part = _part_of(ln.section)
        mut_lines = []
        for other in doc.lines:
            if _part_of(other.section) != part or other.no == ln.no:
                continue
            txt = other.text
            if ("/" in txt or "url" in txt.lower() or "description" in txt.lower() or re.match(r"\s*\d+\.", txt)) and _MUTATION.search(txt):
                mut_lines.append(other.no)
        if not mut_lines:
            continue
        ev = [doc.evidence_for_lines(ln.no)] + [doc.evidence_for_lines(n) for n in mut_lines[:2]]
        out.append(_finding(
            "R06", "conforming", "critical", "Изменяющая операция объявлена через HTTP GET", ev,
            "GET по RFC 9110 (§9.2.1) — безопасный метод без побочных эффектов. Сохранение через GET: запрос могут "
            "повторить или закэшировать прокси/браузер, предзагрузить ссылку; параметры попадают в URL, логи и историю. "
            "Риск непреднамеренного изменения настроек клиента и утечки данных.",
            "Использовать POST или PUT (для полной замены настройки — PUT), параметры передавать в JSON-теле; "
            "для чтения настройки описать отдельный GET.",
            "architect", "Подтверждаете замену метода сохранения на PUT/POST с телом JSON и отдельный GET для чтения настройки?",
            "Повторить запрос сохранения дважды и через кэширующий прокси — состояние не должно меняться неконтролируемо.",
        ))
        break
    return out


_STATUS_MARK = re.compile(r"^\s*(#{1,6}\s*)?\**\s*([1-5]\d\d)\b")
_ERROR_BODY = re.compile(r'("result"\s*:\s*"(error|fail\w*)"|"status"\s*:\s*"error"|"success"\s*:\s*false)', re.I)


def rule_200_with_error(doc: Document, tables: List[Table]) -> List[Finding]:
    current: Optional[Tuple[str, int]] = None
    hits = []
    for ln in doc.lines:
        m = _STATUS_MARK.match(ln.text)
        if m:
            current = (m.group(2), ln.no)
            continue
        if current and current[0].startswith("2") and _ERROR_BODY.search(ln.text):
            hits.append((current[1], ln.no))
    if not hits:
        return []
    head, body = hits[0]
    ev = [doc.evidence_for_lines(head), doc.evidence_for_lines(body)]
    return [_finding(
        "R07", "conforming", "critical", "Ошибка возвращается с HTTP-кодом успеха 2xx", ev,
        "Бизнес-ошибка маскируется под успешный ответ: мониторинг, ретраи и клиент по коду считают операцию успешной. "
        "Для одной и той же ошибки валидации неясно, вернётся 2xx с ERROR или 4xx — ожидаемый результат теста неоднозначен.",
        "Для ошибок валидации возвращать 400 (или 422) с единым телом ошибки {code, message}; "
        "2xx — только при успешном сохранении.",
        "architect", "Какой HTTP-код должен вернуться при недопустимой сумме: 200 с result=ERROR или 400? Нужно одно правило.",
        "Отправить сумму на 1 меньше нижней границы и проверить код ответа и тело.",
    )]


_MINOR_UNIT = re.compile(r"\b(ти[ыи]н\w*|тийын\w*|копе[ей]\w*|cents?\b|minor units?)", re.I)
_MAJOR_UNIT = re.compile(r"(\bтенге\b|₸|\bKZT\b|\bруб\w*|\bUSD\b|\bдоллар\w*)", re.I)
_CONVERSION = re.compile(r"(1\s*(тенге|₸)\s*=\s*100|×\s*100|\*\s*100|/\s*100|пересч[её]т\w*\s+(в|из)\s+ти|конверт\w+\s+ти)", re.I)


def rule_mixed_money_units(doc: Document, tables: List[Table]) -> List[Finding]:
    if any(_CONVERSION.search(ln.text) for ln in doc.lines):
        return []
    minor_lines = [ln for ln in doc.lines if _MINOR_UNIT.search(ln.text)]
    if not minor_lines:
        return []
    major_lines = [
        ln for ln in doc.lines
        if _MAJOR_UNIT.search(ln.text) and re.search(r"(сумм|порог|лимит|amount|limit|баланс|цен)", ln.text, re.I)
    ]
    if not major_lines:
        return []
    ev = [doc.evidence_for_lines(minor_lines[0].no)]
    for ln in major_lines:
        if ln.no != minor_lines[0].no and _part_of(ln.section) != _part_of(minor_lines[0].section):
            ev.append(doc.evidence_for_lines(ln.no))
            break
    else:
        other = [ln for ln in major_lines if ln.no != minor_lines[0].no]
        if other:
            ev.append(doc.evidence_for_lines(other[0].no))
    return [_finding(
        "R08", "set_consistent", "major", "Денежные суммы в разных единицах без правила пересчёта", ev,
        "Одни суммы указаны в минорных единицах (тиыны), другие — в основных (тенге). При сравнении расходов с порогом "
        "без явного правила пересчёта возможна ошибка в 100 раз — уведомление придёт не тогда.",
        "Зафиксировать единицу хранения и сравнения (рекомендуется минорные единицы, целое число) и правило "
        "конвертации на границах API: порог_тиын = порог_тенге × 100.",
        "architect", "В каких единицах сравниваются расходы и порог и где выполняется пересчёт тенге ↔ тиыны?",
        "Порог 50 000 ₸, расходы 4 999 999 тиын + покупка 1 тиын — push должен прийти ровно на 5 000 000 тиын.",
    )]


_PARAM_HDR = re.compile(r"^(parameter|параметр|поле|field|name|имя)$", re.I)
_FORMAT_HDR = re.compile(r"^(format|формат|тип|type)$", re.I)


def rule_type_mismatch(doc: Document, tables: List[Table]) -> List[Finding]:
    seen: Dict[str, List[Tuple[str, int]]] = {}
    for t in tables:
        hdr = [h.lower() for h in t.header]
        try:
            pi = next(i for i, h in enumerate(hdr) if _PARAM_HDR.match(h))
            fi = next(i for i, h in enumerate(hdr) if _FORMAT_HDR.match(h))
        except StopIteration:
            continue
        for no, cells in t.rows:
            if len(cells) <= max(pi, fi):
                continue
            name = cells[pi].strip("` ").lower()
            fmt = cells[fi].strip("` ").lower()
            if name and fmt:
                seen.setdefault(name, []).append((fmt, no))
    diffs = {n: v for n, v in seen.items() if len({f for f, _ in v}) > 1}
    if not diffs:
        return []
    ev: List[Evidence] = []
    parts = []
    for name, vals in list(diffs.items())[:4]:
        fmts = " → ".join(f for f, _ in vals)
        parts.append(f"`{name}`: {fmts}")
        for _, no in vals[:2]:
            if len(ev) < 6:
                ev.append(doc.evidence_for_lines(no))
    return [_finding(
        "R09", "set_consistent", "major", "Один и тот же параметр имеет разные типы в запросе и ответе", ev,
        "Несовпадение типов: " + "; ".join(parts) + ". Клиент и автотесты по контракту не смогут однозначно "
        "сериализовать и сравнивать значения.",
        "Выбрать один тип для каждого параметра (число для сумм, boolean для признаков) и описать контракт в OpenAPI.",
        "developer", "Какой тип является контрактным для " + ", ".join(f"`{n}`" for n in diffs) + " — и в запросе, и в ответе?",
    )]


_SENSITIVE = re.compile(r"(card|pan|cardnumber|account|iban|\biin\b|иин|token|password|passwd|secret|phone|cvv|cvc|email)", re.I)


def rule_sensitive_in_query(doc: Document, tables: List[Table]) -> List[Finding]:
    ev: List[Evidence] = []
    names = []
    for t in tables:
        if "query" not in t.section.lower() and "query" not in " ".join(t.header).lower():
            continue
        hdr = [h.lower() for h in t.header]
        pi = next((i for i, h in enumerate(hdr) if _PARAM_HDR.match(h)), 0)
        for no, cells in t.rows:
            if len(cells) > pi and _SENSITIVE.search(cells[pi]):
                ev.append(doc.evidence_for_lines(no))
                names.append(cells[pi].strip("` "))
    if not ev:
        return []
    return [_finding(
        "R10", "conforming", "major", "Чувствительные данные передаются в query-строке URL", ev[:3],
        f"Параметры {', '.join(f'`{n}`' for n in names)} передаются в URL. Query-строка попадает в журналы веб-серверов, "
        "прокси, APM и историю — это риск утечки идентификаторов карт/клиента (требования PCI DSS и банковской ИБ).",
        "Передавать параметры в теле запроса (POST/PUT), использовать внутренний токен карты вместо номера, "
        "маскировать идентификаторы в логах; согласовать с ИБ.",
        "security", f"Является ли {', '.join(names)} внутренним токеном (не PAN) и допустима ли его передача в URL по политике ИБ?",
    )]


def rule_body_contradiction(doc: Document, tables: List[Table]) -> List[Finding]:
    for i, ln in enumerate(doc.lines):
        if "content-type" in ln.text.lower() and re.search(r"(body|тело)", ln.section, re.I):
            window = doc.lines[i : i + 4]
            for other in window[1:]:
                if re.search(r"(не\s+использ|отсутству|not\s+used|no\s+body)", other.text, re.I):
                    ev = [doc.evidence_for_lines(ln.no), doc.evidence_for_lines(other.no)]
                    return [_finding(
                        "R11", "correct", "minor", "Указан Content-Type для запроса без тела", ev,
                        "Content-Type описывает тело запроса; если тела нет, заголовок вводит в заблуждение и "
                        "указывает на незавершённую смену метода.",
                        "Либо убрать Content-Type, либо (рекомендуется вместе с переходом на PUT/POST) описать JSON-тело.",
                        "analyst", "У метода будет тело запроса в JSON или параметры остаются в URL?",
                    )]
    return []


RULES: List[Callable[[Document, List[Table]], List[Finding]]] = [
    rule_placeholders,
    rule_empty_fields,
    rule_unfilled_checklist,
    rule_empty_tables,
    rule_vague_words,
    rule_get_mutation,
    rule_200_with_error,
    rule_mixed_money_units,
    rule_type_mismatch,
    rule_sensitive_in_query,
    rule_body_contradiction,
]

RULE_DESCRIPTIONS = {
    "R01": "Заглушки и незаполненные разделы («TBD», «заполняются …», «не заполнен»)",
    "R02": "Пустые поля паспорта/спецификации («—»)",
    "R03": "Незаполненный чек-лист (Да / Нет без отметок)",
    "R04": "Пустые таблицы (согласующие и т.п.)",
    "R05": "Размытые слова («быстро», «удобно», «и т. д.» …)",
    "R06": "Изменяющая операция через HTTP GET",
    "R07": "Ошибка с HTTP-кодом успеха 2xx",
    "R08": "Суммы в разных единицах (тиыны/тенге) без правила пересчёта",
    "R09": "Разные типы одного параметра в разных таблицах",
    "R10": "Чувствительные данные в query-строке",
    "R11": "Content-Type при отсутствии тела запроса",
}


def run_rules(doc: Document) -> List[Finding]:
    _counter["n"] = 0
    tables = parse_tables(doc)
    findings: List[Finding] = []
    for rule in RULES:
        try:
            findings.extend(rule(doc, tables))
        except Exception as exc:  # правило не должно ронять прогон
            import logging

            logging.getLogger("reqlens").warning("правило %s упало: %s", rule.__name__, exc)
    return findings
