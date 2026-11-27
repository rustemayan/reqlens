"""Итоговый вердикт. Решение принимается детерминированно по найденным дефектам
(чтобы его нельзя было «уговорить»), модель добавляет только обоснование."""

from __future__ import annotations

from typing import List

from .models import Finding, Priority, Requirement, Verdict

_TEST_BLOCKING = {"verifiable", "unambiguous", "consistent", "set_consistent", "set_validatable", "correct"}

VERDICT_LABEL = {
    "blocked": "⛔ Нельзя начинать",
    "conditional": "⚠️ Можно начинать условно",
    "ready": "✅ Можно начинать",
}


def build_verdict(findings: List[Finding], reqs: List[Requirement], summary: str = "",
                  dev_reason: str = "", test_reason: str = "") -> Verdict:
    crit = [f for f in findings if f.severity == "critical"]
    major = [f for f in findings if f.severity == "major"]
    acceptance_ids = {r.id for r in reqs if r.kind == "acceptance"}

    if crit:
        dev = "blocked"
    elif major:
        dev = "conditional"
    else:
        dev = "ready"

    test_crit = [f for f in crit if f.criterion in _TEST_BLOCKING or set(f.req_ids) & acceptance_ids]
    test_major = [f for f in major if f.criterion in _TEST_BLOCKING | {"complete", "set_complete"}]
    if test_crit:
        test = "blocked"
    elif test_major or crit:
        test = "conditional"
    else:
        test = "ready"

    def _list(items: List[Finding], n: int = 4) -> str:
        return "; ".join(f"{f.id} — {f.title}" for f in items[:n])

    if not dev_reason:
        if dev == "blocked":
            dev_reason = f"Есть {len(crit)} критичных дефектов, которые приведут к переделке: {_list(crit)}."
        elif dev == "conditional":
            dev_reason = f"Критичных дефектов нет, но {len(major)} существенных нужно закрыть до спринта: {_list(major, 3)}."
        else:
            dev_reason = "Критичных и существенных дефектов не выявлено."
    if not test_reason:
        if test == "blocked":
            test_reason = ("Ожидаемые результаты нельзя вывести однозначно — противоречивые или непроверяемые "
                           f"формулировки: {_list(test_crit)}.")
        elif test == "conditional":
            test_reason = ("Тест-дизайн можно начинать для однозначных требований; по остальным нужны ответы "
                           "на вопросы из раздела 7.")
        else:
            test_reason = "Требования проверяемы, ожидаемые результаты определены."
    return Verdict(dev, test, dev_reason, test_reason, summary)


def fallback_priorities(findings: List[Finding], limit: int = 7) -> List[Priority]:
    ranked = sorted(findings, key=lambda f: (f.rank, -len(f.req_ids), f.first_line))
    out = []
    for f in ranked[:limit]:
        if f.severity == "minor" and len(out) >= 3:
            break
        action = f.recommendation or f.title
        if len(action) > 220:
            action = action[:217].rstrip() + "…"
        out.append(Priority([f.id], action))
    return out
