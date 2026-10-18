"""Rows of the procurement history: as read from the workbook and as parsed with a knowledge release.

A row is parsed by rules alone (classifier, extractors); what the choice model reads on top of that is added by
services/knowledge.py and kept across releases by `carried`.
"""

from __future__ import annotations

from typing import Any

from domain.features import Extractors
from domain.frozen import Frozen
from domain.knowledge import SERVICE_FLAG, Classifier, Knowledge, normalize


class PositionRecord(Frozen):
    """A history row as the customer wrote it."""

    row: int  # Excel row number: the id the customer sees
    name: str
    price: float | None


class ParsedPosition(Frozen):
    """A history row with what the parser made of it, ready to store."""

    source_row: int
    name: str
    price: float | None
    search_text: str
    category_code: str | None
    attributes: dict[str, Any]
    flags: tuple[str, ...]
    read_by_llm: tuple[str, ...] = ()  # attributes of `attributes` the choice model read, not the rules


def parse(knowledge: Knowledge, records: list[PositionRecord]) -> list[ParsedPosition]:
    classifier = Classifier(knowledge.categories)
    extractors = Extractors(knowledge.categories)
    parsed = []
    for record in records:
        match = classifier.classify(record.name)
        category = match.category if match else None
        text = normalize(record.name)
        parsed.append(
            ParsedPosition(
                source_row=record.row,
                name=record.name,
                price=record.price,
                search_text=text,
                category_code=category.code if category else None,
                attributes=extractors.extract(category, text) if category else {},
                flags=(SERVICE_FLAG,) if category and category.is_service else (),
            )
        )
    return parsed


def reparsed(knowledge: Knowledge, stored: list[ParsedPosition]) -> list[ParsedPosition]:
    """Stored rows parsed by the rules of another release, from their names."""
    return parse(knowledge, [PositionRecord(row=p.source_row, name=p.name, price=p.price) for p in stored])


def carried(parsed: list[ParsedPosition], stored: list[ParsedPosition]) -> list[ParsedPosition]:
    """What the model read before, added where the rules still leave it unread in the same category."""
    before = {p.source_row: p for p in stored}
    result = []
    for position in parsed:
        old = before.get(position.source_row)
        if old is None or not old.read_by_llm or old.category_code != position.category_code:
            result.append(position)
            continue
        kept = {code: old.attributes[code] for code in old.read_by_llm if code not in position.attributes}
        result.append(
            position.model_copy(
                update={"attributes": {**position.attributes, **kept}, "read_by_llm": tuple(sorted(kept))}
            )
            if kept
            else position
        )
    return result
