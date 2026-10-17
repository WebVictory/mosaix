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
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import cached_property
from typing import Any, Protocol

from analog_search.domain.cards import Card, CardStore, card_key, confirm
from analog_search.domain.completion import Completer
from analog_search.domain.features import Extractors, FeatureExtractor, Value
from analog_search.domain.judging import Judge
from analog_search.domain.knowledge import Attribute, Category, Classifier, Knowledge, Level, normalize
from analog_search.domain.matching import Assessment, Candidate, Decision, Kind, Matcher, Outcome, ScoringConfig
from analog_search.domain.reading import ChoiceReading, Embedder, Option, ReaderError, Reading, SeriesValue, options
from analog_search.domain.spelling import Fix, Spelled, Speller

log = logging.getLogger(__name__)

MAX_QUERY_LENGTH = 500
SUGGESTED = 6  # names of the history offered while a query is typed
COMPLETED = 3  # ways to finish the word being typed, categories offered
MIN_SUGGESTED = 3  # characters typed before anything is offered
KIND_ORDER = (Kind.SAME, Kind.ANALOG, Kind.TEXT, Kind.UNVERIFIED)


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


class Reason(StrEnum):
    UNKNOWN_CATEGORY = "unknown_category"
    CATEGORY_ABSENT = "category_absent"
    HARD_MISMATCH = "hard_mismatch"
    LOW_SCORE = "low_score"
    UNVERIFIED = "unverified"
    DISMISSED = "dismissed"


@dataclass(frozen=True)
class NotFound:
    reason: Reason
    message: str
    closest: Assessment | None


class Source(StrEnum):
    """Who read a category or a value of the query."""

    RULES = "rules"
    LLM = "llm"  # the choice model
    SPECIALIST = "specialist"  # a correction


@dataclass(frozen=True)
class Correction:
    """What a specialist corrected in the understanding of a query."""

    category: str | None = None  # a category code in place of the one read
    values: Mapping[str, str | None] = field(default_factory=dict)  # attribute → option id; None — not stated


@dataclass(frozen=True)
class Facet:
    """An attribute of the query's category, as a shop shows a filter: its value, if stated, and the others."""

    attribute: Attribute
    value: str | None  # None — not stated, or not read
    source: Source | None
    declined: bool  # stated in a form nobody reads: what is found for it is not confirmed
    options: list[Option]  # what a specialist may set it to; [] — not settable
    option: str | None  # the value among the options


@dataclass(frozen=True)
class Understanding:
    """What was read from a query: the same structure the rows of the base were parsed into."""

    text: str
    category: Category | None
    category_source: Source | None
    extractors: dict[str, FeatureExtractor]  # comparers of what can be read, by the rules or the model
    values: dict[str, Value]
    declined: frozenset[str]  # stated in a form nobody reads («3х2,5+1х1,5»)
    by_llm: frozenset[str]
    corrected: frozenset[str] = frozenset()  # values a specialist set or cleared

    @property
    def key(self) -> str:
        """Its golden card's key."""
        return card_key(self.category, self.values, self.text)


@dataclass(frozen=True)
class SearchResult:
    query: str
    knowledge_version: str
    category: Category | None
    facets: list[Facet]
    category_source: Source | None = None
    candidates: list[Assessment] = field(default_factory=list)
    not_found: NotFound | None = None
    card: Card | None = None  # the typical position specialists confirmed rows for

    @property
    def found(self) -> bool:
        return bool(self.candidates) and self.not_found is None


@dataclass(frozen=True)
class Suggestions:
    """What is offered while a query is typed: the query with its typing mistakes fixed, its last word
    finished, the categories it names and the names of the history that hold it."""

    text: str
    fixes: tuple[Fix, ...]
    completions: list[str] = field(default_factory=list)
    categories: list[Category] = field(default_factory=list)
    names: list[str] = field(default_factory=list)


class SearchService:
    def __init__(
        self,
        knowledge: Knowledge,
        store: CandidateStore,
        config: ScoringConfig,
        reading: ChoiceReading | None = None,
        reading_mode: str = "off",
        embedder: Embedder | None = None,
        judge: Judge | None = None,
        cards: CardStore | None = None,
    ):
        self.knowledge = knowledge
        self.config = config
        self._store = store
        self._classifier = Classifier(knowledge.categories)
        self._extractors = Extractors(knowledge.categories)
        self._matcher = Matcher(config)
        self._reading = reading
        self._reading_mode = reading_mode if reading else "off"
        self._embedder = embedder
        self._judge = judge
        self._cards = cards

    def understand(self, query: str, correction: Correction | None = None) -> Understanding:
        """The query read the way the rows of the base were: category, then attribute values; a specialist's
        correction on top."""
        text = normalize(query)
        if not re.search(r"[а-яa-z]", text) or len(query) > MAX_QUERY_LENGTH:
            raise InvalidQuery(f"Запрос должен содержать слова и быть не длиннее {MAX_QUERY_LENGTH} символов")

        correction = correction or Correction()
        category, source = self._category(query, correction.category)
        if category is None:
            if correction.values:
                raise InvalidQuery("Характеристики правятся только у запроса с категорией")
            return Understanding(text, None, None, {}, {}, frozenset(), frozenset())
        extractors = self._comparers(category)
        values = self._extractors.extract(category, text)
        declined = self._extractors.declined(category, text)
        read = self._read(category, query, values, declined)
        if read.unanswered:  # what the model did not answer is compared as without it
            rules = self._extractors.of(category)
            extractors = {code: e for code, e in extractors.items() if code in rules or code not in read.unanswered}
        understood = Understanding(
            text,
            category,
            source,
            extractors,
            {**values, **read.values},
            declined | read.unsure,
            frozenset(read.values),
        )
        return self._corrected(understood, category, correction.values) if correction.values else understood

    def read_by_rules(self, name: str) -> tuple[Category | None, dict[str, Value]]:
        """A name read by the rules alone, as the rows of the history are: for names of a reference book."""
        match = self._classifier.classify(name)
        if match is None or match.category.is_service:
            return None, {}
        return match.category, self._extractors.extract(match.category, normalize(name))

    def assess(self, query: str, position_ids: list[int]) -> tuple[Category | None, list[Assessment]]:
        """Rows checked against the query as the search checks them, but with no card and no judge: what the
        rules say about rows a specialist has decided on (services/editor.py)."""
        read = self.understand(query)
        candidates = self._store.positions(position_ids, read.text) if position_ids else []
        if read.category is None:
            return None, [self._matcher.assess(None, {}, {}, c) for c in candidates]
        return read.category, [self._assess(read, c) for c in candidates]

    def search(self, query: str, correction: Correction | None = None) -> SearchResult:
        read = self.understand(query, correction)
        category = read.category
        card = self._cards.card(read.key) if self._cards else None
        result = SearchResult(
            query=query,
            knowledge_version=self.knowledge.version,
            category=category,
            facets=self._facets(read),
            category_source=read.category_source,
            card=card,
        )

        vector = self._vector(query)
        nearest = self._store.nearest(read.text, self.config.nearest_k, vector)
        if category is None:  # nothing to apply rules to: the whole base by text, against the same threshold
            assessed = [self._matcher.assess(None, {}, {}, c) for c in nearest]
            foreign: list[Assessment] = []
        else:
            pool = self._store.in_category(self.knowledge.version, category.code, read.text, vector)
            assessed = [self._assess(read, c) for c in pool]
            # read by the query's rules, not the model: a row's name is not asked about per query
            foreign = [self._assess(read, self._as(category, c)) for c in nearest if c.category_code != category.code]
        if card:
            assessed, foreign = self._confirmed(read, card, assessed, foreign, vector)

        confirmed = [a for a in assessed + foreign if a.confirmed in (Decision.SAME, Decision.ANALOG)]
        open_ = [a for a in assessed + foreign if a.confirmed is None]
        passing = confirmed + self._judged(
            query, category, [a for a in open_ if not a.rejected and a.score >= self.config.threshold]
        )
        accepted = sorted(
            (a for a in passing if not a.dismissed),
            key=lambda a: (KIND_ORDER.index(a.kind), a.confirmed is None, -a.score),
        )[: self.config.top_k]
        if any(a.kind is not Kind.UNVERIFIED for a in accepted):
            return replace(result, candidates=accepted)
        if accepted:
            return replace(result, candidates=accepted, not_found=self._unverified(category, accepted[0]))
        dismissed = passing + [  # what would have passed but a specialist turned down
            a for a in assessed + foreign if a.dismissed and not a.rejected and a.score >= self.config.threshold
        ]
        if dismissed:
            return replace(result, not_found=self._dismissed(max(dismissed, key=lambda a: a.score)))
        return replace(result, not_found=self._explain(category, assessed, nearest))

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
            "threshold": self.config.threshold,
            "warnings": [warning] if stale else [],
        }

    def spell(self, query: str, typing: bool = False) -> Spelled:
        """The query with the words unknown to the base fixed: a mistyped letter, the English layout."""
        return self._speller.spell(query, typing)

    def suggest(self, query: str) -> Suggestions:
        """While a query is typed: its spelling fixed, its last word finished, the categories it names and the
        names of the history that hold it — all of the query as fixed."""
        if len(query.strip()) < MIN_SUGGESTED:
            return Suggestions(query, ())
        spelled = self.spell(query[:MAX_QUERY_LENGTH], typing=True)
        return Suggestions(
            spelled.text,
            spelled.fixes,
            completions=self._completer.words(spelled.text, COMPLETED),
            categories=self._completer.categories(spelled.text, COMPLETED),
            names=self._completer.names(spelled.text, SUGGESTED),
        )

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

    def unread_hard_attributes(self, reading: bool = False) -> int:
        """Hard rules no extractor (and, with reading, not the model either) enforces:
        candidates of such a category are never «the same»."""
        return sum(
            a.level is Level.HARD and a.code not in (self._comparers(c) if reading else self._extractors.of(c))
            for c in self.knowledge.categories
            for a in c.attributes
        )

    def _comparers(self, category: Category) -> dict[str, FeatureExtractor]:
        extractors = self._extractors.of(category)
        return {**extractors, **self._reading.comparers(category)} if self._reading else extractors

    def categories(self) -> list[Category]:
        """What a query's category may be corrected to: every category of products."""
        return [c for c in self.knowledge.categories if c.group is not None and not c.is_service]

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
        return replace(
            read,
            extractors=extractors,
            values=values,
            declined=read.declined - corrected,
            by_llm=read.by_llm - corrected,
            corrected=corrected,
        )

    @staticmethod
    def _options(attribute: Attribute, rules: dict[str, FeatureExtractor]) -> list[Option]:
        return [] if attribute.level is Level.INFO else options(attribute, rules.get(attribute.code))

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

    def _assess(self, read: Understanding, candidate: Candidate) -> Assessment:
        category = read.category
        foreign = category is not None and candidate.category_code != category.code
        if foreign:
            candidate = self._as(category, candidate)
        return self._matcher.assess(
            category, read.extractors, read.values, candidate, read.declined, read.by_llm, foreign
        )

    def _confirmed(
        self,
        read: Understanding,
        card: Card,
        assessed: list[Assessment],
        foreign: list[Assessment],
        vector: list[float] | None,
    ) -> tuple[list[Assessment], list[Assessment]]:
        """The specialists' verdicts applied; a row they linked but the search did not reach is fetched."""
        reached = {a.candidate.id for a in assessed + foreign}
        missing = [link.position_id for link in card.links if link.position_id not in reached]
        fetched = [self._assess(read, c) for c in self._store.positions(missing, read.text, vector)] if missing else []
        return (
            [confirm(a, card.decision(a.candidate.id)) for a in assessed],
            [confirm(a, card.decision(a.candidate.id)) for a in foreign + fetched],
        )

    def _as(self, category: Category, candidate: Candidate) -> Candidate:
        """A row of another category with the attributes the query's rules read from its name."""
        return replace(
            candidate, attributes=self._extractors.extract(category, normalize(candidate.name)), read_by_llm=frozenset()
        )

    def _vector(self, query: str) -> list[float] | None:
        if self._embedder is None:
            return None
        try:
            [vector] = self._embedder.embed([query])
        except ReaderError as error:
            log.warning("candidates by spelling only: %s", error)
            return None
        return vector

    def _judged(self, query: str, category: Category | None, passing: list[Assessment]) -> list[Assessment]:
        if self._judge is None:
            return passing
        try:
            return self._judge.judge(query, category.name if category else None, passing)
        except ReaderError as error:
            log.warning("candidates as the rules left them: %s", error)
            return passing

    @staticmethod
    def _dismissed(closest: Assessment) -> NotFound:
        by = "специалист отклонил" if closest.confirmed is Decision.NOT_ANALOG else "ИИ-проверка сочла"
        return NotFound(
            Reason.DISMISSED,
            f"Похожие по тексту позиции есть, и правила их не отсекли, но {by} каждую другим товаром.",
            closest,
        )

    @staticmethod
    def _unverified(category: Category | None, closest: Assessment) -> NotFound:
        unchecked = [a.name for a in closest.unreadable] + [
            c.attribute.name for c in closest.checks_with(Outcome.MISSING) if c.attribute.level is Level.HARD
        ]
        return NotFound(
            Reason.UNVERIFIED,
            f"В категории «{category.name if category else ''}» есть похожие позиции, но жёсткие характеристики "
            f"не удалось сравнить ({', '.join(unchecked)}): совпадение не подтверждено, проверьте вручную.",
            closest,
        )

    def _explain(self, category: Category | None, assessed: list[Assessment], nearest: list[Candidate]) -> NotFound:
        closest = max(nearest, key=self._matcher.text, default=None)
        closest_by_text = self._matcher.assess(None, {}, {}, closest) if closest else None
        if category is None:
            return NotFound(
                Reason.UNKNOWN_CATEGORY,
                "Не удалось определить, что это за товар: ни одна категория справочника не узнала запрос, "
                "а по тексту ничего достаточно близкого нет.",
                closest_by_text,
            )
        if not assessed:
            return NotFound(
                Reason.CATEGORY_ABSENT,
                f"Запрос распознан как «{category.name}», но таких позиций в истории закупок нет.",
                closest_by_text,
            )
        passing = [a for a in assessed if not a.rejected]
        if not passing:
            closest = max(assessed, key=lambda a: (len(a.checks_with(Outcome.MATCH)), a.score))
            failed = "; ".join(
                f"{c.attribute.name}: {c.candidate} вместо {c.query}"
                for c in closest.checks_with(Outcome.MISMATCH)
                if c.attribute.level is Level.HARD
            )
            return NotFound(
                Reason.HARD_MISMATCH,
                f"В категории «{category.name}» {len(assessed)} поз., ни одна не проходит жёсткие правила замены. "
                f"Ближайшая не подходит: {failed}.",
                closest,
            )
        closest = max(passing, key=lambda a: a.score)
        return NotFound(
            Reason.LOW_SCORE,
            f"В категории «{category.name}» нет позиции с оценкой выше порога {self.config.threshold}: "
            "характеристики запроса у кандидатов не указаны или текст слишком далёк.",
            closest,
        )

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
                return Facet(attribute, None, None, code in read.declined, choices, None)
            value = read.values[code]
            source = (
                Source.SPECIALIST if code in read.corrected else Source.LLM if code in read.by_llm else Source.RULES
            )
            option = next((o.id for o in choices if o.value == value), None)
            return Facet(attribute, read.extractors[code].format(value, attribute.unit), source, False, choices, option)

        return [facet(a) for a in category.attributes]
