"""What the rules could not read, read by a choice model: the category of a query, attributes from their series.

Rules read the usual spellings; people type in their own words, with typos and numbers in words
(«кабел контрольный семь жил по полтора»). The model does not write anything: it picks
- a category of a query no trigger word recognized, or NONE, among the nearest: half by spelling (trigrams
  catch typos: «трансформатр»), half by meaning (embeddings catch other words: «бесперебойник»);
- an attribute value among its series in attributes.yaml, «не указано» (NONE) or «указано, но не из ряда»
  (OTHER), so it cannot invent a size the specialist never listed. A row of the base is asked about its
  hard attributes only; a query, about soft ones too: they order what the hard rules let through.
A pick is kept only if
- a number is stated in the name, in digits or in words: «ТМЛ 50» → 50 мм² passes, 70 for the same name
  does not; «по два с половиной» states 2.5. Being found in the text is a check of its own, so the model
  need only be fairly sure: confidence ≥ min_stated;
- a qualitative value («медь» from the mark ТМЛ) or a category has nothing to check against but the
  confidence: ≥ min_confidence.
Everything else stays unread: a false hard value rejects a right candidate.
A sure NONE is silence. A sure answer that was not kept — OTHER, or a value not stated in the name —
means the name states the attribute in a way nobody read: «ТМЛ 300» with no 300 in the series must not
pass for «ТМЛ, any size», so for a query such an attribute is unsure and nothing found for it is confirmed.
An answer the model is not sure of, or none at all (the model is down, the cache has none), adds nothing:
the attribute stays as the rules left it, so a hesitant model never does worse than no model.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import zip_longest
from typing import Any, Protocol

from analog_search.domain.features import Extractors, FeatureExtractor, Value, format_number
from analog_search.domain.knowledge import Attribute, Category, Level, normalize
from analog_search.domain.spoken import spoken_numbers
from analog_search.domain.trigrams import trigrams
from analog_search.domain.vectors import cosine

log = logging.getLogger(__name__)

NONE = "NONE"
OTHER = "OTHER"
CATEGORY = "категория"  # the code of the category question
# «0,4 кВ» stated, 400 В chosen: the same number in another unit of the same quantity
UNIT_FACTORS = (1.0, 1e3, 1e-3, 1e6, 1e-6)


class ReaderError(Exception):
    """The choice model is unreachable or answered out of contract."""


@dataclass(frozen=True)
class Question:
    code: str
    instructions: str
    options: dict[str, str]  # option id → what the model reads; NONE and OTHER among them
    values: dict[str, Value]  # option id → the value stored when it is chosen


@dataclass(frozen=True)
class Choice:
    option: str
    confidence: float


@dataclass(frozen=True)
class Reading:
    values: dict[str, Value]  # verified picks
    unsure: frozenset[str] = frozenset()  # the name may state these, but no value passed the checks
    unanswered: frozenset[str] = frozenset()  # no sure answer: read by rules only


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        """A vector per text, in order; raises ReaderError."""
        ...


class ChoiceReader(Protocol):
    def choose(self, state: str, questions: list[Question]) -> dict[str, Choice]:
        """Answers by question code; raises ReaderError."""
        ...


class SeriesValue(FeatureExtractor):
    """Comparer of an attribute only the model reads: values are items of its series, the text is not parsed."""

    def __init__(self, attribute: Attribute):
        self.numeric = _numbers(attribute)
        self._order = list(attribute.rule.get("order") or [])

    def extract(self, text: str) -> Value | None:
        return None

    def key(self, value: Value) -> Any:
        return self._order.index(value) if value in self._order else value


@dataclass(frozen=True)
class Option:
    """An item of an attribute's series as a value to choose: by the model or a specialist."""

    id: str
    label: str
    value: Value


def options(attribute: Attribute, extractor: FeatureExtractor | None) -> list[Option]:
    """The attribute's series as values to choose from, [] when a chosen one would not compare like a read one.

    The series must be all numbers or all words; where an extractor exists, it must accept every item as its
    own value (FeatureExtractor.accepts).
    """
    series = attribute.rule.get("series") or []
    numbers, words = _numbers(attribute), all(isinstance(s, str) for s in series)
    if not series or not (numbers or words):
        return []
    values: list[Value] = [float(s) if numbers else s for s in series]
    if extractor is not None and not all(extractor.accepts(v) for v in values):
        return []
    return [
        Option(
            f"v{i}",
            f"{format_number(v)} {attribute.unit}" if isinstance(v, float) and attribute.unit else str(v),
            v,
        )
        for i, v in enumerate(values)
    ]


def question(attribute: Attribute, extractor: FeatureExtractor | None) -> Question | None:
    """The attribute as a choice, or None when the model may not read it: only scored attributes with options,
    and where an extractor exists, only one of the series' own numbers or items."""
    choices = options(attribute, extractor)
    if attribute.level is Level.INFO or not choices:
        return None
    if extractor is not None and not (extractor.numeric or extractor.series_items):
        return None
    spellings = attribute.rule.get("spellings") or []
    instructions = (
        f"{attribute.name.capitalize()}{f' ({attribute.unit})' if attribute.unit else ''} этой позиции. "
        + (f"Как это пишут: {'; '.join(map(str, spellings))}. " if spellings else "")
        + "Выбери значение, только если оно прямо указано в названии или однозначно следует из марки; "
        f"если указано значение, которого нет среди вариантов, — {OTHER}; если не указано — {NONE}."
    )
    return Question(
        attribute.code,
        instructions,
        {**{o.id: o.label for o in choices}, OTHER: "указано значение не из списка", NONE: "в названии не указано"},
        {o.id: o.value for o in choices},
    )


def verify(
    question: Question, choice: Choice, text: str, min_confidence: float, min_stated: float | None = None
) -> Value | None:
    """The chosen value, if it passes the checks of the module docstring; text is normalized."""
    if choice.option == NONE or choice.option not in question.values:
        return None
    value = question.values[choice.option]
    if isinstance(value, float):
        stated = _stated(value, text) and choice.confidence >= min(min_confidence, min_stated or min_confidence)
        return value if stated else None
    return value if choice.confidence >= min_confidence else None


class ChoiceReading:
    """Asks the model about what the rules left unread in a name and keeps the verified answers."""

    def __init__(
        self,
        reader: ChoiceReader,
        categories: tuple[Category, ...],
        min_confidence: float,
        min_stated: float | None = None,
        nearest: int = 8,
        embedder: Embedder | None = None,
    ):
        self._reader = reader
        self._embedder = embedder
        self._min_confidence = min_confidence
        self._min_stated = min_stated
        self._nearest = nearest
        extractors = Extractors(categories)
        self._questions: dict[str, list[tuple[Level, Question]]] = {}
        self._comparers: dict[str, dict[str, FeatureExtractor]] = {}
        for category in categories:
            of = extractors.of(category)
            readable = [(a, q) for a in category.attributes if (q := question(a, of.get(a.code)))]
            self._questions[category.code] = [(a.level, q) for a, q in readable]
            self._comparers[category.code] = {a.code: SeriesValue(a) for a, _ in readable if a.code not in of}
        # a category is recognized by its name, trigger words and synonyms, each spelled as trigrams
        self._spellings = [
            (category, trigrams(normalize(spelling)))
            for category in categories
            if category.group is not None and not category.is_service
            for spelling in (category.name, *category.triggers, *category.synonyms)
        ]
        # by meaning: one text per category, embedded on first use
        self._described = list(dict.fromkeys(category for category, _ in self._spellings))
        self._vectors: list[list[float]] | None = None

    def readable(self, category: Category, soft: bool = False) -> list[str]:
        return [q.code for level, q in self._questions.get(category.code, []) if soft or level is Level.HARD]

    def category(self, name: str) -> Category | None:
        """The category of a name no trigger word matched: the model picks among the nearest by spelling."""
        candidates = self.nearest_categories(name)
        if not candidates:
            return None
        options = {f"c{i}": c.name for i, c in enumerate(candidates)}
        asked = Question(
            CATEGORY,
            "Что это за продукция? Выбери вид, к которому позиция относится по назначению, даже если "
            f"в названии опечатки или другие слова; если ни один вид не подходит — {NONE}.",
            {**options, NONE: "ни один не подходит"},
            {},
        )
        choice = self._reader.choose(f"Позиция закупки: «{name}»", [asked]).get(CATEGORY)
        if choice is None or choice.confidence < self._min_confidence or choice.option not in options:
            return None
        return candidates[int(choice.option[1:])]

    def nearest_categories(self, name: str) -> list[Category]:
        """By spelling and by meaning in turn, without repeats; by spelling alone when there are no embeddings."""
        by_meaning = self._nearest_by_meaning(name)
        nearest: dict[str, Category] = {}
        for pair in zip_longest(self._nearest_by_spelling(name), by_meaning):
            for category in pair:
                if category is not None and len(nearest) < self._nearest:
                    nearest.setdefault(category.code, category)
        return list(nearest.values())

    def _nearest_by_spelling(self, name: str) -> list[Category]:
        """Categories whose spelling is most contained in the name: «трансформатр» still shares most of
        «трансформатор силовой»."""
        grams = trigrams(normalize(name))
        best: dict[str, tuple[float, Category]] = {}
        for category, spelling in self._spellings:
            score = len(grams & spelling) / len(spelling) if spelling else 0.0
            if score > best.get(category.code, (0.0, category))[0]:
                best[category.code] = (score, category)
        ranked = sorted(best.values(), key=lambda pair: -pair[0])
        return [category for _, category in ranked[: self._nearest]]

    def _nearest_by_meaning(self, name: str) -> list[Category]:
        """Categories whose description is closest to the name in meaning; none if embeddings are unavailable."""
        if self._embedder is None:
            return []
        try:
            if self._vectors is None:
                self._vectors = self._embedder.embed([_description(c) for c in self._described])
            [query] = self._embedder.embed([name])
        except ReaderError as error:
            log.warning("категории по смыслу недоступны, только по написанию: %s", error)
            return []
        ranked = sorted(zip(self._described, self._vectors, strict=True), key=lambda pair: -cosine(query, pair[1]))
        return [category for category, _ in ranked[: self._nearest]]

    def comparers(self, category: Category) -> dict[str, FeatureExtractor]:
        """For attributes no extractor reads: how to compare the values the model chose."""
        return self._comparers.get(category.code, {})

    def read(
        self,
        category: Category,
        name: str,
        known: Mapping[str, Value],
        skip: frozenset[str] = frozenset(),
        soft: bool = False,
    ) -> Reading:
        """Values of the readable attributes not in known; skip — stated in a form nobody may guess at;
        soft — ask about soft attributes too (a query)."""
        questions = [
            q
            for level, q in self._questions.get(category.code, [])
            if (soft or level is Level.HARD) and q.code not in known and q.code not in skip
        ]
        if not questions:
            return Reading({})
        choices = self._reader.choose(f"Категория: {category.name}. Позиция закупки: «{name}»", questions)
        text = normalize(name)
        values = {
            q.code: value
            for q in questions
            if (choice := choices.get(q.code))
            and (value := verify(q, choice, text, self._min_confidence, self._min_stated)) is not None
        }
        answered = {
            q.code
            for q in questions
            if q.code in values or ((c := choices.get(q.code)) and c.confidence >= self._min_confidence)
        }
        unsure = frozenset(c for c in answered if c not in values and choices[c].option != NONE)
        return Reading(values, unsure, frozenset(q.code for q in questions) - answered)


def _description(category: Category) -> str:
    return f"{category.name}: {', '.join((*category.triggers, *category.synonyms))}"


def _numbers(attribute: Attribute) -> bool:
    series = attribute.rule.get("series") or []
    return bool(series) and all(isinstance(s, int | float) and not isinstance(s, bool) for s in series)


def _stated(value: float, text: str) -> bool:
    numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)] + spoken_numbers(text)
    return any(math.isclose(value, n * f, rel_tol=1e-6) for n in numbers for f in UNIT_FACTORS)
