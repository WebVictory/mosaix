"""Search for a position and its analogs in the procurement history.

query → normalize, embed → category (trigger words) → attribute values (extractors of that category)
      → candidates: every row of the category, and the nearest rows of the whole base by spelling
        (trigrams) and by meaning (embeddings) → rules + score → the model judges what the rules
        could not settle → gate: top-K above the threshold, or «не найдено» with the closest row and the
        reason; rows rejected by a hard rule are never shown as found, nor shown to the model.
Rows of other categories are checked by the query's rules read from their names: a mis-parsed
«Кабель ВВГ 5х95» is still a cable 5х95, a «Кабель ВВГ 3х2,5» never passes for it.
Rows that pass but whose hard attributes could not be compared are shown, yet the answer is not «найдено»
unless the model, judging them, is sure they fit: nobody checked them, so a buyer has to.
What specialists confirmed comes first and wins: the query is keyed by what was read from it, and the golden
card of that key (domain/cards.py) says which rows are the same position, which may replace it and which
must never be offered.
A query is typed in the user's own words. What the rules leave unread is asked of the choice model, if one
is configured: the category when no trigger word matched («кабел контрольный», «автомат легранд»), and the
attributes the extractors did not read («семь жил по полтора»). When the model or the embeddings fail or
are unsure, the search goes on with what the rules read and the trigrams found.
A specialist sees the understanding as a shop shows a category: every attribute it has, stated or not, and
may correct it — another category, a value of an attribute's series, an attribute cleared. The query is then
read again with the correction on top, and the corrected structure is what the search and the golden card
key go by.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from enum import StrEnum
from functools import cached_property
from typing import Any, Protocol

from pydantic import Field

from domain.cards import Card, CardStore, card_key, confirm
from domain.completion import Completer
from domain.features import Extractors, FeatureExtractor, Value
from domain.frozen import Frozen
from domain.gate import Gatekeeper, NotFound
from domain.judging import Judge
from domain.knowledge import Attribute, Category, Classifier, Knowledge, Level, normalize
from domain.matching import (
    BY_TEXT,
    Assessment,
    Candidate,
    Matcher,
    Requirements,
)
from domain.reading import ChoiceReading, Embedder, Option, ReaderError, Reading, SeriesValue, options
from domain.rules import SearchRules
from domain.spelling import Fix, Spelled, Speller

log = logging.getLogger(__name__)


class InvalidQuery(ValueError):
    """Empty or meaningless query: the caller's mistake, not a miss."""


class CandidateStore(Protocol):
    def in_category(
        self, knowledge_version: str, category_code: str, text: str, vector: list[float] | None = None
    ) -> list[Candidate]:
        """Every row of the category; with the query's vector, their meaning similarity too."""
        ...

    def nearest(self, text: str, limit: int, vector: list[float] | None = None) -> list[Candidate]:
        """The closest rows of the whole base: `limit` by spelling and, with a vector, `limit` by meaning."""
        ...

    def positions(self, ids: list[int], text: str, vector: list[float] | None = None) -> list[Candidate]:
        """The rows by id, the service ones aside: those a specialist linked to the query's card."""
        ...

    def stats(self) -> dict[str, Any]: ...

    def names(self) -> list[str]:
        """Names of the goods rows: the words a query is spelled against."""
        ...


class Source(StrEnum):
    """Who read a category or a value of the query."""

    RULES = "rules"
    LLM = "llm"  # the choice model
    SPECIALIST = "specialist"  # a correction


class Correction(Frozen):
    """What a specialist corrected in the understanding of a query."""

    category: str | None = None  # a category code in place of the one read
    values: dict[str, str | None] = Field(default_factory=dict)  # attribute → option id; None — not stated


class Facet(Frozen):
    """An attribute of the query's category, as a shop shows a filter: its value, if stated, and the others."""

    attribute: Attribute
    value: str | None  # None — not stated, or not read
    source: Source | None
    declined: bool  # stated in a form nobody reads: what is found for it is not confirmed
    options: list[Option]  # what a specialist may set it to; [] — not settable
    option: str | None  # the value among the options


class Understanding(Requirements):
    """What was read from a query: the same structure the rows of the base were parsed into, and who read it."""

    text: str
    category_source: Source | None = None
    corrected: frozenset[str] = frozenset()  # values a specialist set or cleared

    @property
    def key(self) -> str:
        """Its golden card's key."""
        return card_key(self.category, self.values, self.text)


class SearchResult(Frozen):
    query: str
    knowledge_version: str
    category: Category | None
    facets: list[Facet]
    category_source: Source | None = None
    candidates: list[Assessment] = Field(default_factory=list)
    not_found: NotFound | None = None
    card: Card | None = None  # the typical position specialists confirmed rows for

    @property
    def found(self) -> bool:
        return bool(self.candidates) and self.not_found is None


class Suggestions(Frozen):
    """What is offered while a query is typed: the query with its typing mistakes fixed, its last word
    finished, the categories it names and the names of the history that hold it."""

    text: str
    fixes: tuple[Fix, ...]
    completions: list[str] = Field(default_factory=list)
    categories: list[Category] = Field(default_factory=list)
    names: list[str] = Field(default_factory=list)


class _Pool(Frozen):
    """The candidates of one search, assessed: those of the query's category, the near ones of other
    categories, and the nearest rows by text the «не найдено» explanation points to."""

    in_category: list[Assessment]
    foreign: list[Assessment]
    nearest: list[Candidate]

    @property
    def assessed(self) -> list[Assessment]:
        return self.in_category + self.foreign


class SearchService:
    def __init__(
        self,
        knowledge: Knowledge,
        store: CandidateStore,
        rules: SearchRules,
        reading: ChoiceReading | None = None,
        reading_mode: str = "off",
        embedder: Embedder | None = None,
        judge: Judge | None = None,
        cards: CardStore | None = None,
    ):
        self.knowledge = knowledge
        self.rules = rules
        self._store = store
        self._classifier = Classifier(knowledge.categories)
        self._extractors = Extractors(knowledge.categories)
        self._matcher = Matcher(rules.scoring)
        self._gate = Gatekeeper(rules.gate, self._matcher)
        self._reading = reading
        self._reading_mode = reading_mode if reading else "off"
        self._embedder = embedder
        self._judge = judge
        self._cards = cards

    def search(self, query: str, correction: Correction | None = None) -> SearchResult:
        """The steps of ARCHITECTURE.md §2, in their order."""
        read = self.understand(query, correction)  # 4. the query parsed
        card = self._cards.card(read.key) if self._cards else None
        vector = self._vector(query)
        nearest, in_category = self._candidates(read, vector)  # 5. candidates
        pool = self._apply_rules(read, nearest, in_category)  # 6. rules
        if card:
            pool = self._apply_decisions(read, card, pool, vector)  # 7. the specialists' decisions
        passing = self._passing(query, read.category, pool)  # 8. threshold, the model's check
        result = SearchResult(
            query=query,
            knowledge_version=self.knowledge.version,
            category=read.category,
            facets=self._facets(read),
            category_source=read.category_source,
            card=card,
        )
        return self._answer(result, pool, passing)  # top-K, or «не найдено» with the reason

    def understand(self, query: str, correction: Correction | None = None) -> Understanding:
        """The query read the way the rows of the base were: category, then attribute values; a specialist's
        correction on top."""
        text = normalize(query)
        if not re.search(r"[а-яa-z]", text) or len(query) > self.rules.query.max_length:
            raise InvalidQuery(
                f"Запрос должен содержать слова и быть не длиннее {self.rules.query.max_length} символов"
            )

        correction = correction or Correction()
        category, source = self._category(query, correction.category)
        if category is None:
            if correction.values:
                raise InvalidQuery("Характеристики правятся только у запроса с категорией")
            return Understanding(text=text)
        extractors = self._comparers(category)
        values = self._extractors.extract(category, text)
        declined = self._extractors.declined(category, text)
        read = self._read(category, query, values, declined)
        if read.unanswered:  # what the model did not answer is compared as without it
            rules = self._extractors.of(category)
            extractors = {code: e for code, e in extractors.items() if code in rules or code not in read.unanswered}
        understood = Understanding(
            text=text,
            category=category,
            category_source=source,
            extractors=extractors,
            values={**values, **read.values},
            declined=declined | read.unsure,
            by_llm=frozenset(read.values),
        )
        return self._corrected(understood, category, correction.values) if correction.values else understood

    def assess(self, query: str, position_ids: list[int]) -> tuple[Category | None, list[Assessment]]:
        """Rows checked against the query as the search checks them, but with no card and no judge: what the
        rules say about rows a specialist has decided on (services/editor.py)."""
        read = self.understand(query)
        candidates = self._store.positions(position_ids, read.text) if position_ids else []
        if read.category is None:
            return None, [self._matcher.assess(BY_TEXT, c) for c in candidates]
        return read.category, [self._assess(read, c) for c in candidates]

    def read_by_rules(self, name: str) -> tuple[Category | None, dict[str, Value]]:
        """A name read by the rules alone, as the rows of the history are: for names of a reference book."""
        match = self._classifier.classify(name)
        if match is None or match.category.is_service:
            return None, {}
        return match.category, self._extractors.extract(match.category, normalize(name))

    def suggest(self, query: str) -> Suggestions:
        """While a query is typed: its spelling fixed, its last word finished, the categories it names and the
        names of the history that hold it — all of the query as fixed."""
        if len(query.strip()) < self.rules.query.min_suggested:
            return Suggestions(text=query, fixes=())
        spelled = self.spell(query[: self.rules.query.max_length], typing=True)
        return Suggestions(
            text=spelled.text,
            fixes=spelled.fixes,
            completions=self._completer.words(spelled.text, self.rules.query.completed),
            categories=self._completer.categories(spelled.text, self.rules.query.completed),
            names=self._completer.names(spelled.text, self.rules.query.suggested),
        )

    def spell(self, query: str, typing: bool = False) -> Spelled:
        """The query with the words unknown to the base fixed: a mistyped letter, the English layout."""
        return self._speller.spell(query, typing)

    def categories(self) -> list[Category]:
        """What a query's category may be corrected to: every category of products."""
        return [c for c in self.knowledge.categories if c.group is not None and not c.is_service]

    def stats(self) -> dict[str, Any]:
        """What the search works with; a warning if the base was parsed with another knowledge release."""
        stored = self._store.stats()
        covered, total = self._extractors.coverage()
        stale = sorted(set(stored["parsed_with"]) - {self.knowledge.version})
        warning = f"Позиции разобраны другой версией знаний ({', '.join(stale)}): выполните `positions load`"
        return {
            **stored,
            "knowledge_version": self.knowledge.version,
            "categories": len(self.knowledge.categories),
            "attributes": total,
            "attributes_with_extractor": covered,
            "hard_attributes_without_extractor": self.unread_hard_attributes(),
            "hard_attributes_read_by_llm": self.unread_hard_attributes() - self.unread_hard_attributes(reading=True),
            "llm": self._reading_mode,
            "threshold": self.rules.gate.threshold,
            "warnings": [warning] if stale else [],
        }

    def unread_hard_attributes(self, reading: bool = False) -> int:
        """Hard rules no extractor (and, with reading, not the model either) enforces:
        candidates of such a category are never «the same»."""
        return sum(
            a.level is Level.HARD and a.code not in (self._comparers(c) if reading else self._extractors.of(c))
            for c in self.knowledge.categories
            for a in c.attributes
        )

    def _candidates(self, read: Understanding, vector: list[float] | None) -> tuple[list[Candidate], list[Candidate]]:
        """The nearest rows of the whole base by spelling and meaning, and every row of the query's category."""
        nearest = self._store.nearest(read.text, self.rules.candidates.nearest_k, vector)
        if read.category is None:
            return nearest, []
        return nearest, self._store.in_category(self.knowledge.version, read.category.code, read.text, vector)

    def _apply_rules(self, read: Understanding, nearest: list[Candidate], in_category: list[Candidate]) -> _Pool:
        category = read.category
        if category is None:  # nothing to apply rules to: the whole base by text, against the same threshold
            return _Pool(in_category=[self._matcher.assess(BY_TEXT, c) for c in nearest], foreign=[], nearest=nearest)
        return _Pool(
            in_category=[self._assess(read, c) for c in in_category],
            # read by the query's rules, not the model: a row's name is not asked about per query
            foreign=[self._assess(read, self._as(category, c)) for c in nearest if c.category_code != category.code],
            nearest=nearest,
        )

    def _apply_decisions(self, read: Understanding, card: Card, pool: _Pool, vector: list[float] | None) -> _Pool:
        """The specialists' verdicts applied; a row they linked but the search did not reach is fetched."""
        reached = {a.candidate.id for a in pool.assessed}
        missing = [link.position_id for link in card.links if link.position_id not in reached]
        fetched = [self._assess(read, c) for c in self._store.positions(missing, read.text, vector)] if missing else []
        return pool.model_copy(
            update={
                "in_category": [confirm(a, card.decision(a.candidate.id)) for a in pool.in_category],
                "foreign": [confirm(a, card.decision(a.candidate.id)) for a in pool.foreign + fetched],
            }
        )

    def _passing(self, query: str, category: Category | None, pool: _Pool) -> list[Assessment]:
        """What a specialist confirmed, and what is above the threshold and the model did not turn down."""
        return self._gate.confirmed(pool.assessed) + self._judged(query, category, self._gate.open(pool.assessed))

    def _answer(self, result: SearchResult, pool: _Pool, passing: list[Assessment]) -> SearchResult:
        candidates, not_found = self._gate.answer(
            result.category, passing, pool.assessed, pool.in_category, pool.nearest
        )
        return result.model_copy(update={"candidates": candidates, "not_found": not_found})

    def _category(self, query: str, code: str | None) -> tuple[Category | None, Source | None]:
        if code is not None:
            chosen = next((c for c in self.categories() if c.code == code), None)
            if chosen is None:
                raise InvalidQuery(f"Нет категории «{code}»")
            return chosen, Source.SPECIALIST
        match = self._classifier.classify(query)
        if match is not None:
            return (None, None) if match.category.is_service else (match.category, Source.RULES)
        guessed = self._guess(query)
        return guessed, Source.LLM if guessed else None

    def _guess(self, query: str) -> Category | None:
        if self._reading is None:
            return None
        try:
            return self._reading.category(query)
        except ReaderError as error:
            log.warning("query category by rules only: %s", error)
            return None

    def _read(self, category: Category, query: str, values: dict[str, Value], declined: frozenset[str]) -> Reading:
        if self._reading is None:
            return Reading({})
        try:
            return self._reading.read(category, query, values, declined, soft=True)
        except ReaderError as error:
            log.warning("query read by rules only: %s", error)
            return Reading({}, unanswered=frozenset(self._reading.readable(category, soft=True)))

    def _corrected(
        self, read: Understanding, category: Category, corrections: Mapping[str, str | None]
    ) -> Understanding:
        """The understanding with a specialist's values: an option of the series, or nothing stated."""
        attributes = {a.code: a for a in category.attributes}
        rules = self._extractors.of(category)
        values, extractors = dict(read.values), dict(read.extractors)
        for code, option_id in corrections.items():
            attribute = attributes.get(code)
            if attribute is None:
                raise InvalidQuery(f"В категории «{category.name}» нет характеристики «{code}»")
            if option_id is None:
                values.pop(code, None)
                continue
            option = next((o for o in self._options(attribute, rules) if o.id == option_id), None)
            if option is None:
                raise InvalidQuery(f"У характеристики «{attribute.name}» нет значения «{option_id}»")
            values[code] = option.value
            extractors.setdefault(code, SeriesValue(attribute))  # compared even when only the model reads it
        corrected = frozenset(corrections)
        return read.model_copy(
            update={
                "extractors": extractors,
                "values": values,
                "declined": read.declined - corrected,
                "by_llm": read.by_llm - corrected,
                "corrected": corrected,
            }
        )

    @staticmethod
    def _options(attribute: Attribute, rules: dict[str, FeatureExtractor]) -> list[Option]:
        return [] if attribute.level is Level.INFO else options(attribute, rules.get(attribute.code))

    def _comparers(self, category: Category) -> dict[str, FeatureExtractor]:
        extractors = self._extractors.of(category)
        return {**extractors, **self._reading.comparers(category)} if self._reading else extractors

    def _vector(self, query: str) -> list[float] | None:
        if self._embedder is None:
            return None
        try:
            [vector] = self._embedder.embed([query])
        except ReaderError as error:
            log.warning("candidates by spelling only: %s", error)
            return None
        return vector

    def _assess(self, read: Understanding, candidate: Candidate) -> Assessment:
        category = read.category
        foreign = category is not None and candidate.category_code != category.code
        if category is not None and foreign:
            candidate = self._as(category, candidate)
        return self._matcher.assess(read, candidate, foreign)

    def _as(self, category: Category, candidate: Candidate) -> Candidate:
        """A row of another category with the attributes the query's rules read from its name."""
        return candidate.model_copy(
            update={
                "attributes": self._extractors.extract(category, normalize(candidate.name)),
                "read_by_llm": frozenset(),
            }
        )

    def _judged(self, query: str, category: Category | None, passing: list[Assessment]) -> list[Assessment]:
        if self._judge is None:
            return passing
        try:
            return self._judge.judge(query, category.name if category else None, passing)
        except ReaderError as error:
            log.warning("candidates as the rules left them: %s", error)
            return passing

    def _facets(self, read: Understanding) -> list[Facet]:
        """Every attribute of the query's category: what was read, by whom, and what it may be set to."""
        category = read.category
        if category is None:
            return []
        rules = self._extractors.of(category)

        def facet(attribute: Attribute) -> Facet:
            code = attribute.code
            choices = self._options(attribute, rules)
            if code not in read.values:
                return Facet(
                    attribute=attribute,
                    value=None,
                    source=None,
                    declined=code in read.declined,
                    options=choices,
                    option=None,
                )
            value = read.values[code]
            source = (
                Source.SPECIALIST if code in read.corrected else Source.LLM if code in read.by_llm else Source.RULES
            )
            option = next((o.id for o in choices if o.value == value), None)
            return Facet(
                attribute=attribute,
                value=read.extractors[code].format(value, attribute.unit),
                source=source,
                declined=False,
                options=choices,
                option=option,
            )

        return [facet(a) for a in category.attributes]

    @cached_property
    def _names(self) -> list[str]:
        return self._store.names()

    @cached_property
    def _speller(self) -> Speller:
        words = [w for c in self.knowledge.categories for w in (c.name, *c.triggers, *c.synonyms)]
        return Speller([*self._names, *words])

    @cached_property
    def _completer(self) -> Completer:
        return Completer(self._names, self.categories())
