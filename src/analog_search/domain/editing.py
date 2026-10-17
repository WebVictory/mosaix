"""A specialist's edits of the knowledge: a draft over the active release, published as a new release.

The knowledge lives in the database as releases; research/*.yaml only seeds an empty base. A specialist
does not edit a release: they put edits into a draft — a field of an attribute (level, comparison, step
tolerance, status, substitution note) or the trigger words of a category. The draft is the active
release with the edits applied; published, it becomes a new release, and the history is parsed with it.
A release is named by its content: undoing every edit gives back the release it started from.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from enum import StrEnum
from typing import Any

from analog_search.domain.knowledge import (
    Attribute,
    AttributeStatus,
    Category,
    Comparison,
    Knowledge,
    Level,
    normalize,
)

MAX_TOLERANCE = 5
MAX_TEXT = 500


class Field(StrEnum):
    LEVEL = "level"
    COMPARISON = "comparison"
    STEP_TOLERANCE = "step_tolerance"
    STATUS = "status"
    SUBSTITUTION = "substitution"
    TRIGGERS = "triggers"
    ANTI_TRIGGERS = "anti_triggers"


ATTRIBUTE_FIELDS = (Field.LEVEL, Field.COMPARISON, Field.STEP_TOLERANCE, Field.STATUS, Field.SUBSTITUTION)
CATEGORY_FIELDS = (Field.TRIGGERS, Field.ANTI_TRIGGERS)
RULE_FIELDS = (Field.STEP_TOLERANCE, Field.SUBSTITUTION)  # kept in Attribute.rule


class InvalidEdit(ValueError):
    pass


@dataclass(frozen=True)
class Edit:
    """One change of the draft: a field of a category (attribute None) or of one of its attributes."""

    category: str
    attribute: str | None
    field: Field
    value: Any
    previous: Any = None  # in the release the draft started from
    author: str | None = None
    reason: str | None = None
    id: int | None = None
    created_at: datetime | None = None

    @property
    def target(self) -> tuple[str, str | None, Field]:
        return self.category, self.attribute, self.field


def current(knowledge: Knowledge, category_code: str, attribute_code: str | None, field: Field) -> Any:
    """The field's value in the knowledge; InvalidEdit if there is no such category, attribute or field."""
    category = _category(knowledge, category_code)
    if attribute_code is None:
        if field not in CATEGORY_FIELDS:
            raise InvalidEdit(f"У категории нет поля {field}")
        return list(getattr(category, field.value))
    if field not in ATTRIBUTE_FIELDS:
        raise InvalidEdit(f"У характеристики нет поля {field}")
    attribute = _attribute(category, attribute_code)
    if field in RULE_FIELDS:
        default = 1 if field is Field.STEP_TOLERANCE else None
        return attribute.rule.get(field.value, default)
    return getattr(attribute, field.value).value


def validated(knowledge: Knowledge, edit: Edit) -> Edit:
    """The edit with its value in canonical form and the value it replaces; InvalidEdit if it makes no sense."""
    previous = current(knowledge, edit.category, edit.attribute, edit.field)
    return replace(edit, value=_value(knowledge, edit), previous=previous)


def apply(knowledge: Knowledge, edits: list[Edit]) -> Knowledge:
    """The knowledge with the edits, named by its content."""
    if not edits:
        return knowledge
    by_category: dict[str, list[Edit]] = {}
    for edit in edits:
        _category(knowledge, edit.category)
        by_category.setdefault(edit.category, []).append(edit)
    categories = tuple(_edited(c, by_category.get(c.code, [])) for c in knowledge.categories)
    digest = fingerprint(categories)
    major = knowledge.version.split("-", 1)[0]
    return Knowledge(version=f"{major}-{digest[:8]}", sha256=digest, categories=categories)


def fingerprint(categories: tuple[Category, ...]) -> str:
    """sha256 of the content: the same knowledge, the same release."""
    plain = json.dumps([asdict(c) for c in categories], ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(plain.encode()).hexdigest()


def _edited(category: Category, edits: list[Edit]) -> Category:
    if not edits:
        return category
    fields = {e.field.value: tuple(e.value) for e in edits if e.attribute is None}
    attributes = tuple(_edited_attribute(a, [e for e in edits if e.attribute == a.code]) for a in category.attributes)
    return replace(category, **fields, attributes=attributes)


def _edited_attribute(attribute: Attribute, edits: list[Edit]) -> Attribute:
    if not edits:
        return attribute
    rule = dict(attribute.rule)
    changes: dict[str, Any] = {}
    for edit in edits:
        match edit.field:
            case Field.LEVEL:
                changes["level"] = Level(edit.value)
            case Field.COMPARISON:
                changes["comparison"] = Comparison(edit.value)
            case Field.STATUS:
                changes["status"] = AttributeStatus(edit.value)
            case Field.STEP_TOLERANCE | Field.SUBSTITUTION if edit.value is None:
                rule.pop(edit.field.value, None)
            case Field.STEP_TOLERANCE | Field.SUBSTITUTION:
                rule[edit.field.value] = edit.value
    return replace(attribute, **changes, rule=rule)


def _value(knowledge: Knowledge, edit: Edit) -> Any:
    value = edit.value
    match edit.field:
        case Field.LEVEL:
            return _member(Level, value, "уровень")
        case Field.COMPARISON:
            comparison = _member(Comparison, value, "сравнение")
            attribute = _attribute(_category(knowledge, edit.category), edit.attribute or "")
            if comparison == Comparison.STEP and not _numbers(attribute):
                raise InvalidEdit(
                    f"Сравнение «{comparison}» требует числового ряда значений, у «{attribute.name}» его нет"
                )
            return comparison
        case Field.STATUS:
            return _member(AttributeStatus, value, "статус")
        case Field.STEP_TOLERANCE:
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_TOLERANCE:
                raise InvalidEdit(f"Допуск — целое число ступеней ряда от 0 до {MAX_TOLERANCE}")
            return value
        case Field.SUBSTITUTION:
            text = " ".join(str(value or "").split())
            if len(text) > MAX_TEXT:
                raise InvalidEdit(f"Пояснение длиннее {MAX_TEXT} символов")
            return text or None
        case Field.TRIGGERS | Field.ANTI_TRIGGERS:
            return _words(knowledge, edit, value)
    raise InvalidEdit(f"Неизвестное поле {edit.field}")


def _words(knowledge: Knowledge, edit: Edit, value: Any) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(w, str) for w in value):
        raise InvalidEdit("Слова-признаки — список строк")
    words = list(dict.fromkeys(w for w in (normalize(v) for v in value) if w))
    if any(len(w) > 100 for w in words):
        raise InvalidEdit("Слово-признак длиннее 100 символов")
    category = _category(knowledge, edit.category)
    if edit.field is Field.TRIGGERS and not words and category.group is not None:
        raise InvalidEdit(f"У «{category.name}» не останется слов-признаков: её позиции перестанут распознаваться")
    return words


def _member(kind: type[StrEnum], value: Any, what: str) -> str:
    try:
        return kind(value).value
    except ValueError:
        raise InvalidEdit(f"Недопустимое значение «{value}» поля {what}") from None


def _numbers(attribute: Attribute) -> bool:
    series = attribute.rule.get("series") or []
    return bool(series) and all(isinstance(s, int | float) and not isinstance(s, bool) for s in series)


def _category(knowledge: Knowledge, code: str) -> Category:
    category = next((c for c in knowledge.categories if c.code == code), None)
    if category is None:
        raise InvalidEdit(f"Нет категории {code}")
    return category


def _attribute(category: Category, code: str) -> Attribute:
    attribute = next((a for a in category.attributes if a.code == code), None)
    if attribute is None:
        raise InvalidEdit(f"У «{category.name}» нет характеристики {code}")
    return attribute
