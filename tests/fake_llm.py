"""Фейковый провайдер LLM для тестов: имитирует ответы модели по этапам,
включая типичные сбои (битый JSON, 429, обрезка ответа, выдуманные цитаты)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

from reqlens.config import Settings
from reqlens.llm.base import LLMClient, Provider, RawResponse, RetryableError


def _f(criterion, severity, title, evidence, related=(), q_to="analyst", q="Уточните правило?", test=""):
    return {
        "criterion": criterion,
        "severity": severity,
        "title": title,
        "evidence": [{"quote": qt, "line": ln} for qt, ln in evidence],
        "related_req_ids": list(related),
        "explanation": f"Объяснение: {title}.",
        "recommendation": f"Исправить: {title}.",
        "question": {"to": q_to, "text": q},
        "test_idea": test,
    }


@dataclass
class FakeProvider(Provider):
    settings: Settings
    fail_first_json: bool = True
    rate_limit_once: bool = True
    truncate_big_batches: bool = True
    calls: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.name = "fake"

    def generate(self, model, system, user, schema) -> RawResponse:
        stage = self._stage(user)
        self.calls.append(stage)
        if stage == "check" and self.rate_limit_once:
            self.rate_limit_once = False
            raise RetryableError("429 fake", retry_after=0.01)
        if stage == "segment":
            if self.fail_first_json:
                self.fail_first_json = False
                return RawResponse(text="Вот результат: {\"requirements\": [ {\"doc_id\": ", finish_reason="STOP")
            return self._ok(self._segment(user))
        if stage == "check":
            ids = re.findall(r"^### (REQ-\d+)", user, re.M)
            if self.truncate_big_batches and len(ids) > 6:
                return RawResponse(text='{"results": [{"req_id": "REQ-01", "assess', finish_reason="MAX_TOKENS", truncated=True)
            return self._ok(self._check(ids, user))
        if stage == "consistency":
            return self._ok(self._consistency())
        if stage == "completeness":
            return self._ok(self._completeness(user))
        if stage == "review":
            return self._ok(self._review(user))
        return self._ok({})

    @staticmethod
    def _ok(data) -> RawResponse:
        return RawResponse(text="```json\n" + json.dumps(data, ensure_ascii=False) + "\n```", finish_reason="STOP",
                           input_tokens=1000, output_tokens=300)

    @staticmethod
    def _stage(user: str) -> str:
        if "нарезать документ" in user:
            return "segment"
        if "ПАКЕТ ДЛЯ ПРОВЕРКИ" in user:
            return "check"
        if "на согласованность" in user:
            return "consistency"
        if "ПОЛНОТУ набора" in user:
            return "completeness"
        if "ревьюер-скептик" in user:
            return "review"
        return "other"

    # ---------------------------------------------------------- stages
    def _segment(self, user: str) -> dict:
        # берём строки документа из промпта и «нарезаем» их как модель
        lines = dict((int(n), t) for n, t in re.findall(r"^L(\d+)\|\s?(.*)$", user, re.M))
        reqs = []
        for n, t in sorted(lines.items()):
            s = t.strip()
            if re.match(r"^\*\s+\*\*Ф-\d", s):
                did = re.search(r"Ф-\d+", s).group(0)
                reqs.append({"doc_id": did, "section": "Функциональные требования", "line_start": n, "line_end": n,
                             "quote": re.sub(r"[*]", "", s).strip(), "summary": f"Функциональное требование {did}", "kind": "functional"})
            elif s.startswith("| Расходы") or s.startswith("| Покупка"):
                reqs.append({"doc_id": "", "section": "Критерии приемки", "line_start": n, "line_end": n,
                             "quote": s.strip("| "), "summary": "Критерий приёмки", "kind": "acceptance"})
            elif s.startswith("Сервис передаёт push"):
                reqs.append({"doc_id": "", "section": "НФТ", "line_start": n, "line_end": n, "quote": s,
                             "summary": "Время передачи push", "kind": "nonfunctional"})
            elif s.startswith("Хранить идентификатор"):
                reqs.append({"doc_id": "", "section": "Данные", "line_start": n, "line_end": n,
                             "quote": "Хранить идентификатор карты, месяц, идентификатор операции",
                             "summary": "Состав хранимых данных", "kind": "data"})
            elif "**REST type**" in s:
                reqs.append({"doc_id": "", "section": "Общие параметры", "line_start": n, "line_end": n,
                             "quote": "REST type | GET", "summary": "Метод сохранения порога — GET", "kind": "api"})
            elif s.startswith("Приложение читает"):
                reqs.append({"doc_id": "", "section": "Интеграция", "line_start": n, "line_end": n,
                             "quote": "Приложение читает и сохраняет настройки через дочерний документ",
                             "summary": "Чтение и сохранение настроек через API", "kind": "integration"})
        # выдуманное требование — должно быть отброшено проверкой цитат
        reqs.append({"doc_id": "Ф-9", "section": "Выдумка", "line_start": 999, "line_end": 999,
                     "quote": "Система отправляет SMS при каждом входе", "summary": "Выдумка", "kind": "functional"})
        # дубль
        if reqs:
            reqs.append(dict(reqs[0]))
        return {"document_title": "Уведомление о расходах по карте", "requirements": reqs}

    def _check(self, ids: List[str], user: str) -> dict:
        # сопоставление id → doc_id по промпту
        meta = dict(re.findall(r"^### (REQ-\d+) \[([^\]]+)\]", user, re.M))
        results = []
        for rid in ids:
            did = meta.get(rid, "")
            findings = []
            if did == "Ф-3":
                findings.append(_f("consistent", "critical", "Лимит одного push в месяц противоречит Ф-4",
                                   [("На карту приходится максимум один такой push за календарный месяц", 39),
                                    ("следующее достижение вызывает push независимо от истории предыдущей настройки", 40)],
                                   q_to="business", q="Сбрасывается ли лимит одного push при смене порога?",
                                   test="Порог 10 000, push отправлен; поднять до 20 000; потратить ещё — сколько push?"))
                findings.append(_f("verifiable", "major", "Получение push клиентом не контролируется системой",
                                   [("клиент получает push", 39)], q_to="business",
                                   q="Фиксируем факт передачи push провайдеру или факт доставки?"))
            if did == "Ф-2":
                findings.append(_f("unambiguous", "major", "Не определён термин «проведённые покупки»",
                                   [("В расходы входят проведённые покупки", 38)], q_to="business",
                                   q="Считаем по авторизации или по клирингу?"))
                # выдуманная цитата — находка должна быть отброшена
                findings.append(_f("complete", "major", "Не описаны покупки в валюте",
                                   [("Покупки в иностранной валюте конвертируются по курсу ЦБ", 38)]))
            if "Покупка 31 января" in user and rid == ids[0] and did == "":
                pass
            results.append({"req_id": rid, "assessment": f"Оценка {rid}.", "findings": findings})
        # критерий приёмки — противоречие с Ф-2 (перефразированная цитата → нечёткое совпадение)
        for rid in ids:
            block = re.search(rf"^### {rid} .*?\n«(.*?)»", user, re.M | re.S)
            if block and "31 января" in block.group(1):
                for r in results:
                    if r["req_id"] == rid:
                        r["findings"].append(_f("consistent", "critical", "Пример приёмки противоречит правилу месяца Ф-2",
                                                [("Покупка 31 января в 23:58 на 4 000 ₸", 64),
                                                 ("Месяц определяется по дате совершения покупки", 38)],
                                                q_to="business", q="Месяц — по дате покупки или по дате проведения?"))
        # дубль правила R06 (GET) — должен склеиться автоматически
        for rid in ids:
            block = re.search(rf"^### {rid} .*?\n«(.*?)»", user, re.M | re.S)
            if block and "REST type" in block.group(1):
                for r in results:
                    if r["req_id"] == rid:
                        r["findings"].append(_f("conforming", "critical", "Метод GET используется для изменения настройки порога",
                                                [("REST type | GET", 104)], q_to="architect",
                                                q="Меняем GET на PUT /alerts/limit?"))
        return {"results": results}

    def _consistency(self) -> dict:
        return {
            "findings": [
                _f("set_consistent", "critical", "Часть A требует чтения настроек, а API описывает только сохранение",
                   [("Приложение читает и сохраняет настройки через дочерний документ", 52),
                    ("Сохранение порога. Функциональные требования: чтение порога", 103)], q_to="architect",
                   q="Кто описывает метод чтения настройки и суммы за месяц?"),
            ],
            "glossary": [
                {"concept": "Порог уведомления", "variants": [
                    {"quote": "личного бюджета", "line": 23},
                    {"quote": "Месячный порог расходов по карте", "line": 14},
                    {"quote": "/alerts/saveLimit", "line": 108},
                    {"quote": "несуществующий термин из головы модели", "line": 1},
                ], "recommendation": "Закрепить термин «порог уведомления»."},
            ],
        }

    def _completeness(self, user: str) -> dict:
        func = re.findall(r"^- (REQ-\d+) \[Ф-\d\]", user, re.M)
        return {
            "findings": [
                _f("traceable", "major", "Не у всех функциональных требований есть критерии приёмки",
                   [("Критерии приемки", 59)], q_to="qa", q="Кто допишет критерии приёмки для Ф-1 и Ф-4?"),
                _f("set_complete", "major", "Повтор находки про заглушку тестирования",
                   [("заполняются тестировщиками", 57)], q_to="qa", q="Когда заполнят?"),
            ],
            "traceability": [
                {"source": rid, "ui": "REQ-99, —", "api": "", "data": "", "acceptance": "—",
                 "status": "partial", "note": "Нет критерия приёмки"} for rid in func
            ] + [{"source": "REQ-777", "ui": "", "api": "", "data": "", "acceptance": "", "status": "missing", "note": ""}],
            "missing_scenarios": [
                {"scenario": "Возврат покупки прошлого месяца", "why": "Меняет закрытый месяц",
                 "related_req_ids": func[:1] + ["REQ-555"], "given": "Покупка 10 000 ₸ 30 января",
                 "when": "Возврат 5 000 ₸ 3 февраля", "then": "Январский счётчик уменьшен на 5 000 ₸, февральский не изменился"},
            ],
        }

    def _review(self, user: str) -> dict:
        ids = re.findall(r"^- id=(\S+) \|", user, re.M)
        decisions = []
        for i, fid in enumerate(ids):
            if i == 1:
                decisions.append({"id": fid, "action": "drop", "severity": "minor", "reason": "Придирка"})
            elif i == 2 and len(ids) > 3:
                decisions.append({"id": fid, "action": "merge", "merge_into": ids[3], "severity": "major", "reason": "Дубль"})
            else:
                decisions.append({"id": fid, "action": "keep", "severity": "major" if i == 4 else "", "reason": "Верно"})
        return {
            "decisions": decisions,
            "priorities": [{"finding_ids": ids[:2], "action": "Устранить противоречия правил push и месяца"},
                           {"finding_ids": ["NOPE"], "action": "Несуществующая находка"}],
            "summary": "Документ не готов к разработке: есть противоречия в ключевых правилах.",
            "development_reason": "Противоречия Ф-3/Ф-4 и Ф-2/приёмки.",
            "testing_reason": "Ожидаемые результаты неоднозначны.",
        }


def make_fake_client(tmp_path, **kwargs) -> LLMClient:
    settings = Settings(provider="gemini", model="fake-model", api_key="test", cache_dir=tmp_path / "cache",
                        min_interval=0.0, max_retries=3, fallback_models=["fake-fallback"])
    return LLMClient(provider=FakeProvider(settings=settings, **kwargs), settings=settings)
