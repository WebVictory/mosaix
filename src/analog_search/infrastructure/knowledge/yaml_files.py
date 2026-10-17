"""research/categories.yaml + attributes.yaml → Knowledge.

The files are written for the specialist, in Russian; this is the only place that knows their keys.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from analog_search.domain.knowledge import (
    Attribute,
    AttributeStatus,
    Category,
    Comparison,
    Knowledge,
    Level,
    Provenance,
    slug,
)

FILES = ("categories.yaml", "attributes.yaml")

COMPARISONS = {
    "равно": Comparison.EQUAL,
    "не_хуже": Comparison.NOT_WORSE,
    "не_больше": Comparison.NOT_MORE,
    "ступень": Comparison.STEP,
    "включает": Comparison.INCLUDES,
    "без_порядка": Comparison.UNORDERED,
}
PROVENANCES = {
    "источник": Provenance.SOURCE,
    "здравый смысл": Provenance.COMMON_SENSE,
    "догадка": Provenance.GUESS,
    "специалист": Provenance.SPECIALIST,
}
STATUSES = {
    "предложено": AttributeStatus.PROPOSED,
    "одобрено": AttributeStatus.APPROVED,
    "отклонено": AttributeStatus.REJECTED,
}
# Everything else of an attribute goes to its rule as is
RULE_KEYS = {"написания": "spellings", "ряд": "series", "порядок": "order", "допуск_ступеней": "step_tolerance",
             "замена": "substitution", "величина": "quantity"}  # fmt: skip


@dataclass(frozen=True)
class YamlKnowledge:
    """The yaml files of a directory as the seed of the knowledge base."""

    directory: Path

    @property
    def location(self) -> str:
        return str(self.directory)

    def read(self) -> Knowledge:
        return read(self.directory)


def read(directory: Path) -> Knowledge:
    paths = [directory / name for name in FILES]
    digest = hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest()
    categories_doc, attributes_doc = (yaml.safe_load(p.read_text(encoding="utf-8")) for p in paths)

    attributes = {
        entry["категория"]: tuple(_attribute(a) for a in entry["характеристики"])
        for entry in attributes_doc["категории"]
    }
    groups = dict.fromkeys(entry["группа"] for entry in categories_doc["категории"])
    categories = [Category(code=slug(g), name=g, group=None) for g in groups]
    categories += [
        Category(
            code=slug(entry["имя"]),
            name=entry["имя"],
            group=entry["группа"],
            triggers=tuple(entry["триггеры"]),
            anti_triggers=tuple(entry["анти_триггеры"]),
            synonyms=tuple(entry["синонимы"]),
            attributes=attributes.get(entry["имя"], ()),
        )
        for entry in categories_doc["категории"]
    ]
    codes = [c.code for c in categories]
    if len(set(codes)) != len(codes):
        raise ValueError(f"category codes collide: {sorted(c for c in codes if codes.count(c) > 1)}")
    if unknown := set(attributes) - {c.name for c in categories}:
        raise ValueError(f"attributes.yaml names unknown categories: {sorted(unknown)}")
    return Knowledge(version=f"{categories_doc['версия']}-{digest[:8]}", sha256=digest, categories=tuple(categories))


def _attribute(entry: dict[str, Any]) -> Attribute:
    return Attribute(
        code=entry["характеристика"],
        name=entry["величина"],
        unit=entry["единица"],
        level=Level(entry["уровень"]),
        comparison=COMPARISONS[entry["сравнение"]],
        provenance=PROVENANCES[entry["уверенность"]],
        status=STATUSES[entry["статус"]],
        source_ref=entry["источник"],
        rule={RULE_KEYS[k]: v for k, v in entry.items() if k in RULE_KEYS and v is not None},
    )
