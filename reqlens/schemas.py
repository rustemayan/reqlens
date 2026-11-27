"""JSON-схемы ответов модели. Используются дважды: передаются в модель как
structured output и проверяют её ответ (reqlens.llm.jsonutil.validate)."""

from __future__ import annotations

from .criteria import ADDRESSEES, INDIVIDUAL_IDS, SET_IDS, SEVERITIES

KINDS = [
    "goal", "functional", "nonfunctional", "ui", "process", "architecture", "interaction",
    "data", "integration", "acceptance", "api", "testing", "constraint", "other",
]

KIND_RU = {
    "goal": "Цель",
    "functional": "Функциональное",
    "nonfunctional": "Нефункциональное",
    "ui": "UI/UX",
    "process": "Процесс (BPMN)",
    "architecture": "Архитектура",
    "interaction": "Взаимодействие",
    "data": "Данные",
    "integration": "Интеграция",
    "acceptance": "Критерий приёмки",
    "api": "API",
    "testing": "Тестирование",
    "constraint": "Ограничение",
    "other": "Прочее",
}

_EVIDENCE = {
    "type": "object",
    "properties": {
        "quote": {"type": "string", "description": "Дословная цитата из документа (3–25 слов) без префикса Lxxx|"},
        "line": {"type": "integer", "description": "Номер строки документа, где находится цитата"},
    },
    "required": ["quote", "line"],
}

_QUESTION = {
    "type": "object",
    "properties": {
        "to": {"type": "string", "enum": ADDRESSEES},
        "text": {"type": "string"},
    },
    "required": ["to", "text"],
}


def _finding(criteria_ids):
    return {
        "type": "object",
        "properties": {
            "criterion": {"type": "string", "enum": criteria_ids},
            "severity": {"type": "string", "enum": SEVERITIES},
            "title": {"type": "string", "description": "Суть дефекта в 4–12 словах"},
            "evidence": {"type": "array", "items": _EVIDENCE},
            "related_req_ids": {"type": "array", "items": {"type": "string"}},
            "explanation": {"type": "string", "description": "Почему это дефект и к чему приведёт"},
            "recommendation": {"type": "string", "description": "Конкретная исправленная формулировка или что добавить"},
            "question": _QUESTION,
            "test_idea": {"type": "string", "description": "Тест/граничный случай, который выявит проблему"},
        },
        "required": ["criterion", "severity", "title", "evidence", "explanation", "recommendation", "question"],
    }


SEGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "document_title": {"type": "string"},
        "requirements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "Идентификатор из документа (Ф-1, п. 2) или пустая строка"},
                    "section": {"type": "string"},
                    "line_start": {"type": "integer"},
                    "line_end": {"type": "integer"},
                    "quote": {"type": "string", "description": "Дословный текст требования из документа"},
                    "summary": {"type": "string", "description": "Краткая формулировка, до 15 слов"},
                    "kind": {"type": "string", "enum": KINDS},
                },
                "required": ["doc_id", "section", "line_start", "line_end", "quote", "summary", "kind"],
            },
        },
    },
    "required": ["document_title", "requirements"],
}

CHECK_REQ_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "req_id": {"type": "string"},
                    "assessment": {"type": "string", "description": "Итоговая оценка требования, 1–2 предложения"},
                    "findings": {"type": "array", "items": _finding(INDIVIDUAL_IDS)},
                },
                "required": ["req_id", "assessment", "findings"],
            },
        }
    },
    "required": ["results"],
}

CHECK_SET_CONSISTENCY_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {"type": "array", "items": _finding(SET_IDS)},
        "glossary": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "concept": {"type": "string"},
                    "variants": {"type": "array", "items": _EVIDENCE},
                    "recommendation": {"type": "string"},
                },
                "required": ["concept", "variants", "recommendation"],
            },
        },
    },
    "required": ["findings", "glossary"],
}

CHECK_SET_COMPLETENESS_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {"type": "array", "items": _finding(SET_IDS)},
        "traceability": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "REQ-id функции или цели"},
                    "ui": {"type": "string"},
                    "api": {"type": "string"},
                    "data": {"type": "string"},
                    "acceptance": {"type": "string"},
                    "status": {"type": "string", "enum": ["covered", "partial", "missing"]},
                    "note": {"type": "string"},
                },
                "required": ["source", "ui", "api", "data", "acceptance", "status", "note"],
            },
        },
        "missing_scenarios": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "scenario": {"type": "string"},
                    "why": {"type": "string"},
                    "related_req_ids": {"type": "array", "items": {"type": "string"}},
                    "given": {"type": "string"},
                    "when": {"type": "string"},
                    "then": {"type": "string"},
                },
                "required": ["scenario", "why", "related_req_ids", "given", "when", "then"],
            },
        },
    },
    "required": ["findings", "traceability", "missing_scenarios"],
}

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "action": {"type": "string", "enum": ["keep", "drop", "merge"]},
                    "merge_into": {"type": "string"},
                    "severity": {"type": "string", "enum": SEVERITIES},
                    "reason": {"type": "string"},
                },
                "required": ["id", "action", "severity", "reason"],
            },
        },
        "priorities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "finding_ids": {"type": "array", "items": {"type": "string"}},
                    "action": {"type": "string"},
                },
                "required": ["finding_ids", "action"],
            },
        },
        "summary": {"type": "string"},
        "development_reason": {"type": "string"},
        "testing_reason": {"type": "string"},
    },
    "required": ["decisions", "priorities", "summary", "development_reason", "testing_reason"],
}
