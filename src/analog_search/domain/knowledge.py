"""Our taxonomy as plain data: categories with triggers, attribute schemas with substitution rules.

The specialist edits it in research/*.yaml; a load turns one state of those files into a
knowledge release, and every search result refers to the release it was made with.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from analog_search.domain.frozen import Frozen
from analog_search.domain.spoken import spoken_sizes


class Level(StrEnum):
    HARD = "hard"  # mismatch excludes the candidate
    SOFT = "soft"  # mismatch lowers the score
    INFO = "info"  # shown, not scored


class Comparison(StrEnum):
    EQUAL = "equal"
    NOT_WORSE = "not_worse"
    NOT_MORE = "not_more"
    STEP = "step"
    INCLUDES = "includes"
    UNORDERED = "unordered"


class Provenance(StrEnum):
    SOURCE = "source"  # standard, catalogue
    COMMON_SENSE = "common_sense"
    GUESS = "guess"
    SPECIALIST = "specialist"


class AttributeStatus(StrEnum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"


class Attribute(Frozen):
    code: str  # сечение_жилы
    name: str
    unit: str | None
    level: Level
    comparison: Comparison
    provenance: Provenance
    status: AttributeStatus
    source_ref: str | None
    rule: dict[str, Any]  # spellings, value series, order, step tolerance, substitution note


class Category(Frozen):
    code: str  # kabel_silovoy
    name: str  # кабель силовой
    group: str | None  # None for a group itself
    triggers: tuple[str, ...] = ()
    anti_triggers: tuple[str, ...] = ()
    synonyms: tuple[str, ...] = ()
    attributes: tuple[Attribute, ...] = ()

    def __eq__(self, other: object) -> bool:
        """A category is what names and recognizes it; its attribute schema is not compared."""
        return isinstance(other, Category) and self._identity == other._identity

    def __hash__(self) -> int:
        return hash(self._identity)

    @property
    def _identity(self) -> tuple[Any, ...]:
        return self.code, self.name, self.group, self.triggers, self.anti_triggers, self.synonyms

    @property
    def group_code(self) -> str | None:
        return None if self.group is None else slug(self.group)

    @property
    def is_service(self) -> bool:
        """Transport, loading, surcharges: rows of an estimate, not goods."""
        return self.group == SERVICE_GROUP


SERVICE_GROUP = "служебные строки"
SERVICE_FLAG = "service"  # positions.flags of such rows: never a candidate
OTHER_MARK = "(проч"  # «труба (прочая)»: a catch-all yields to a sibling that also matched


class Knowledge(Frozen):
    version: str
    sha256: str
    categories: tuple[Category, ...]  # groups first, then categories in file order


_TRANSLIT = str.maketrans(
    {
        "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ж": "zh", "з": "z", "и": "i",
        "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s",
        "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch",
        "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
    }
)  # fmt: skip


def slug(name: str) -> str:
    """«кабель силовой» → kabel_silovoy; «СКУД / турникет» → skud_turniket."""
    latin = normalize(name).translate(_TRANSLIT)
    return re.sub(r"[^a-z0-9]+", "_", latin).strip("_")


_NUMBERING = re.compile(r"^\s*\d+(\.\d+)*[.)]\s*")
_TIMES = re.compile(r"(?<=\d)(?:\s*[xх×*]\s*|\s+на\s+)(?=\d)")  # one spelling of a size: «7*1.5», «7 на 1.5»
_DECIMAL_COMMA = re.compile(r"(?<=\d),(?=\d)")


def normalize(text: str) -> str:
    """Lower case, ё → е, no leading «1.2)» numbering, «7 х 1,5», «7 на 1,5», «семь на полтора» → «7x1.5»,
    single spaces."""
    text = spoken_sizes(text.lower().replace("ё", "е").replace("’", "'"))
    text = _DECIMAL_COMMA.sub(".", _TIMES.sub("x", _NUMBERING.sub("", text)))
    return " ".join(text.split())


@dataclass(frozen=True)
class Match:
    category: Category
    trigger: str


class Classifier:
    """Category of a row by trigger words; the rule of research/categories.yaml.

    A trigger matches from a word start. The earliest match wins, a longer trigger breaks a tie.
    An anti-trigger of the category cancels a match that it overlaps. A catch-all «(прочее)»
    category yields to a sibling matched anywhere in the text: a category of the same group, or one named
    by the same head noun («труба гофрированная» for «труба (прочая)», whatever group it is filed under).
    """

    def __init__(self, categories: tuple[Category, ...]):
        self._triggers = [
            (trigger, category, re.compile(r"(?<![а-яa-z0-9])" + re.escape(trigger)))
            for category in categories
            for trigger in category.triggers
        ]
        self._anti = {
            category.code: [re.compile(re.escape(a)) for a in category.anti_triggers] for category in categories
        }

    def classify(self, name: str) -> Match | None:
        text = normalize(name)
        best = self._earliest(text, lambda _: True, any_occurrence=True)
        if best is None or OTHER_MARK not in best.category.name:
            return best
        group, code = best.category.group, best.category.code
        head = best.category.name.split(OTHER_MARK)[0].strip()
        sibling = self._earliest(
            text,
            lambda c: c.code != code and (c.group == group or c.name.startswith(head + " ")),
            any_occurrence=False,
        )
        return sibling or best

    def _earliest(self, text: str, accept: Callable[[Category], bool], any_occurrence: bool) -> Match | None:
        """Earliest, then longest trigger match; any_occurrence=False looks at the first occurrence only."""
        best: tuple[tuple[int, int], Match] | None = None
        for trigger, category, pattern in self._triggers:
            if not accept(category):
                continue
            for found in pattern.finditer(text):
                if not self._cancelled(category, text, found.start()):
                    key = (found.start(), -len(trigger))
                    if best is None or key < best[0]:
                        best = (key, Match(category, trigger))
                    break
                if not any_occurrence:
                    break
        return best[1] if best else None

    def _cancelled(self, category: Category, text: str, position: int) -> bool:
        return any(
            anti.start() <= position < anti.end()
            for pattern in self._anti[category.code]
            for anti in pattern.finditer(text)
        )
