"""Извлечение JSON из ответа модели и валидация по JSON Schema (подмножество).

Схемы в reqlens.schemas — единый источник правды: они же уходят в модель
(structured output), и ими же проверяется ответ.
"""

from __future__ import annotations

import json
import re
from typing import Any, List


class JSONExtractError(ValueError):
    pass


def extract_json(text: str) -> Any:
    """Достаёт JSON-объект из текста: убирает ```json-ограждения, мусор до/после,
    висячие запятые. Бросает JSONExtractError, если JSON не найден."""
    if text is None:
        raise JSONExtractError("пустой ответ модели")
    s = text.strip()
    fence = re.search(r"```(?:json|JSON)?\s*(.*?)```", s, re.S)
    if fence:
        s = fence.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    start = min([i for i in (s.find("{"), s.find("[")) if i >= 0], default=-1)
    if start < 0:
        raise JSONExtractError("в ответе нет JSON")
    end = max(s.rfind("}"), s.rfind("]"))
    if end <= start:
        raise JSONExtractError("JSON в ответе обрезан")
    candidate = s[start : end + 1]
    for attempt in (candidate, _remove_trailing_commas(candidate)):
        try:
            return json.loads(attempt)
        except json.JSONDecodeError as exc:
            last = exc
    raise JSONExtractError(f"невалидный JSON: {last.msg} (позиция {last.pos})")


def _remove_trailing_commas(s: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", s)


def validate(data: Any, schema: dict, path: str = "$") -> List[str]:
    """Минимальный валидатор JSON Schema: type, properties, required, items, enum.
    Возвращает список ошибок (пустой — данные валидны)."""
    errors: List[str] = []
    _validate(data, schema, path, errors)
    return errors


_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
}


def _validate(data: Any, schema: dict, path: str, errors: List[str]) -> None:
    if len(errors) > 30:
        return
    t = schema.get("type")
    if t:
        types = t if isinstance(t, list) else [t]
        ok = False
        for tt in types:
            if tt == "null" and data is None:
                ok = True
            elif tt in _TYPES:
                py = _TYPES[tt]
                if isinstance(data, bool) and tt in ("integer", "number"):
                    continue
                if isinstance(data, py):
                    ok = True
        if not ok:
            errors.append(f"{path}: ожидался тип {t}, получено {type(data).__name__}")
            return
    if "enum" in schema and data not in schema["enum"]:
        errors.append(f"{path}: значение {data!r} не из {schema['enum']}")
    if isinstance(data, dict):
        for req in schema.get("required", []):
            if req not in data:
                errors.append(f"{path}: нет обязательного поля '{req}'")
        for key, sub in schema.get("properties", {}).items():
            if key in data:
                _validate(data[key], sub, f"{path}.{key}", errors)
    if isinstance(data, list) and "items" in schema:
        for i, item in enumerate(data):
            _validate(item, schema["items"], f"{path}[{i}]", errors)
