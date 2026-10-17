"""A candidate against a query: the specialist's rules per attribute, a score with its parts, a verdict.

Rules are the attributes of the query's category (level + comparison from attributes.yaml):
a hard mismatch rejects the candidate, a soft one lowers the score, info is only shown.
The score is a weighted sum the UI shows as is:

    score = text_weight · text similarity: by spelling (trigrams) or by meaning (embeddings), the higher
          + features_weight · share of the query's attributes the candidate satisfies
          + category_weight · (same category)
          − penalties (soft mismatches, hard attributes the candidate does not state)

An attribute the candidate satisfies with a margin (IP65 for IP54, 33 Вт for 30) counts inexact_credit
of an identical one: of two candidates that both fit, the one closer to what was asked comes first.

A row of another category found by text («Кабель ВВГ 5х95», which the rules took for something else) is
checked by the query's rules read from its name: a hard mismatch rejects it, otherwise it stays unverified
(the rules never confirmed what it is) unless the choice model, judging it, says it fits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from analog_search.domain.features import FeatureExtractor, Value, same
from analog_search.domain.knowledge import Attribute, Category, Comparison, Level


class Outcome(StrEnum):
    MATCH = "match"
    MISMATCH = "mismatch"
    MISSING = "missing"  # the candidate's name does not state it: not a mismatch, not a confirmation


class Kind(StrEnum):
    SAME = "same"  # every hard attribute is known on both sides and every value is the same
    ANALOG = "analog"  # every hard rule holds, something differs or the query does not ask for it
    UNVERIFIED = "unverified"  # no hard rule broken, but some hard rule could not be checked
    TEXT = "text"  # the query has no category: no rules applied, only the text is similar


class Verdict(StrEnum):
    """The choice model's judgement of a candidate the rules let through (domain/judging.py)."""

    SAME = "same"
    ANALOG = "analog"
    OTHER = "other"  # another product: the candidate is dismissed


class Decision(StrEnum):
    """A specialist's verdict on a candidate for a typical position (domain/cards.py)."""

    SAME = "same"
    ANALOG = "analog"
    NOT_ANALOG = "not_analog"  # never offered for this position, whatever the rules say
    UNCLEAR = "unclear"  # skipped: kept in the decision log, links nothing


@dataclass(frozen=True)
class ScoringConfig:
    text_weight: float = 0.35
    features_weight: float = 0.45
    category_weight: float = 0.2
    soft_share: float = 0.5  # a soft attribute counts this much of a hard one in the features share
    no_features_credit: float = 0.5  # features share when the query states no attribute at all
    inexact_credit: float = 0.75  # a match with a margin, of an identical one
    meaning_floor: float = 0.5  # cosine of unrelated names: meaning similarity is rescaled from it to 1
    soft_mismatch_penalty: float = 0.1
    missing_penalty: float = 0.05  # per hard attribute the candidate does not state
    threshold: float = 0.5  # below — «не найдено»
    top_k: int = 5
    nearest_k: int = 20  # second pass over the whole base, by spelling and as many by meaning
    judge_k: int = 10  # unresolved candidates the choice model judges


@dataclass(frozen=True)
class Candidate:
    """A stored position as the search sees it."""

    id: int
    row: int
    name: str
    price: float | None
    category_code: str | None
    attributes: dict[str, Value]
    similarity: float  # trigram similarity of the normalized names, 0…1
    meaning: float | None = None  # cosine of the embeddings of the query and the name, if both are known
    read_by_llm: frozenset[str] = frozenset()  # attributes the choice model read, not the rules


@dataclass(frozen=True)
class Check:
    attribute: Attribute
    query: str
    candidate: str | None
    outcome: Outcome
    exact: bool
    query_by_llm: bool = False  # the value was read by the choice model, not the rules
    candidate_by_llm: bool = False


@dataclass(frozen=True)
class Assessment:
    candidate: Candidate
    score: float
    breakdown: dict[str, float]
    checks: tuple[Check, ...]
    kind: Kind | None  # None — rejected by a hard rule
    unreadable: tuple[Attribute, ...] = ()  # hard attributes of the category neither rules nor the model read
    verdict: Verdict | None = None  # the choice model's, when it judged the candidate and was sure
    foreign: bool = False  # a row of another category, checked by the query's rules read from its name
    confirmed: Decision | None = None  # a specialist's verdict on this row for the query's typical position

    @property
    def rejected(self) -> bool:
        return self.kind is None

    @property
    def dismissed(self) -> bool:
        return self.verdict is Verdict.OTHER or self.confirmed is Decision.NOT_ANALOG

    def checks_with(self, outcome: Outcome) -> list[Check]:
        return [c for c in self.checks if c.outcome is outcome]


def satisfies(attribute: Attribute, extractor: FeatureExtractor, query: Value, candidate: Value) -> bool:
    """Does the candidate's value meet the requested one under the attribute's comparison?"""
    q, c = extractor.key(query), extractor.key(candidate)
    match attribute.comparison:
        case Comparison.EQUAL | Comparison.UNORDERED:
            return _equal(q, c)
        case Comparison.NOT_WORSE:
            return _at_least(c, q)
        case Comparison.NOT_MORE:
            return _at_least(q, c)
        case Comparison.INCLUDES:
            return c >= q if isinstance(q, frozenset) else _at_least(c, q)
        case Comparison.STEP:
            return _within_steps(attribute, q, c)
    raise ValueError(attribute.comparison)


def _equal(a: Any, b: Any) -> bool:
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, float | int) and isinstance(b, float | int):
        return math.isclose(a, b, rel_tol=1e-6)
    return a == b


def _at_least(a: Any, b: Any) -> bool:
    if isinstance(a, tuple) and isinstance(b, tuple):  # IP54 vs IP65: every digit
        return len(a) == len(b) and all(x >= y for x, y in zip(a, b, strict=True))
    if isinstance(a, float | int) and isinstance(b, float | int):
        return a >= b or math.isclose(a, b, rel_tol=1e-6)
    return a == b  # values without an order: only the same one will do


def _within_steps(attribute: Attribute, q: Any, c: Any) -> bool:
    """±N steps of the attribute's series (2 Мп for 4 Мп), N = step_tolerance, 1 by default."""
    tolerance = int(attribute.rule.get("step_tolerance", 1))
    series = [float(s) for s in attribute.rule.get("series", []) if isinstance(s, int | float)]
    if isinstance(q, float) and isinstance(c, float) and q in series and c in series:
        return abs(series.index(q) - series.index(c)) <= tolerance
    return _equal(q, c)


class Matcher:
    def __init__(self, config: ScoringConfig):
        self.config = config

    def assess(
        self,
        category: Category | None,
        extractors: dict[str, FeatureExtractor],
        query: dict[str, Value],
        candidate: Candidate,
        unread: frozenset[str] = frozenset(),
        read_by_llm: frozenset[str] = frozenset(),
        foreign: bool = False,
    ) -> Assessment:
        """extractors — comparers of the attributes that can be read, by the rules or the choice model;
        unread — attributes the query states in a form no extractor reads («3х2,5+1х1,5»);
        read_by_llm — attributes of the query the choice model read;
        foreign — a row of another category whose attributes were read from its name by the query's rules."""
        config = self.config
        in_category = category is not None and candidate.category_code == category.code
        checked = in_category or (foreign and category is not None)
        checks = self._checks(category, extractors, query, candidate, read_by_llm) if checked else ()

        scored = [c for c in checks if c.attribute.level is not Level.INFO]
        weights = [1.0 if c.attribute.level is Level.HARD else config.soft_share for c in scored]
        if scored:
            credits = [
                w * (1.0 if c.exact else config.inexact_credit)
                for w, c in zip(weights, scored, strict=True)
                if c.outcome is Outcome.MATCH
            ]
            share = sum(credits) / sum(weights)
        elif in_category:
            share = config.no_features_credit
        else:  # a row of another category: the rules do not apply, the text decides
            share = self.text(candidate) if category is None else 0.0
        soft_misses = sum(c.outcome is Outcome.MISMATCH and c.attribute.level is Level.SOFT for c in scored)
        hard_missing = sum(c.outcome is Outcome.MISSING and c.attribute.level is Level.HARD for c in scored)

        breakdown = {
            "text": round(config.text_weight * self.text(candidate), 4),
            "features": round(config.features_weight * share, 4),
            "category": config.category_weight if in_category else 0.0,
            "penalties": round(
                -(config.soft_mismatch_penalty * soft_misses + config.missing_penalty * hard_missing), 4
            ),
        }
        score = round(min(1.0, max(0.0, sum(breakdown.values()))), 4)
        hard_miss = any(c.outcome is Outcome.MISMATCH and c.attribute.level is Level.HARD for c in checks)
        unreadable = (
            tuple(a for a in _hard(category) if a.code not in extractors or a.code in unread) if in_category else ()
        )
        if hard_miss:
            kind = None
        elif foreign:
            kind = Kind.UNVERIFIED
        else:
            kind = self._kind(category, query, candidate, checks, unreadable, unread)
        return Assessment(candidate, score, breakdown, checks, kind, unreadable, foreign=checked and not in_category)

    def text(self, candidate: Candidate) -> float:
        """How close the names are: by spelling or by meaning, the higher; meaning rescaled from meaning_floor."""
        if candidate.meaning is None:
            return candidate.similarity
        floor = self.config.meaning_floor
        return max(candidate.similarity, max(0.0, candidate.meaning - floor) / (1 - floor))

    @staticmethod
    def _checks(
        category: Category,
        extractors: dict[str, FeatureExtractor],
        query: dict[str, Value],
        candidate: Candidate,
        read_by_llm: frozenset[str],
    ) -> tuple[Check, ...]:
        checks = []
        for attribute in category.attributes:
            if attribute.code not in query:
                continue
            extractor = extractors[attribute.code]
            wanted, offered = query[attribute.code], candidate.attributes.get(attribute.code)
            if offered is None:
                outcome, exact = Outcome.MISSING, False
            else:
                ok = satisfies(attribute, extractor, wanted, offered)
                outcome, exact = (Outcome.MATCH if ok else Outcome.MISMATCH), same(wanted, offered)
            checks.append(
                Check(
                    attribute=attribute,
                    query=extractor.format(wanted, attribute.unit),
                    candidate=None if offered is None else extractor.format(offered, attribute.unit),
                    outcome=outcome,
                    exact=exact,
                    query_by_llm=attribute.code in read_by_llm,
                    candidate_by_llm=offered is not None and attribute.code in candidate.read_by_llm,
                )
            )
        return tuple(checks)

    @staticmethod
    def _kind(
        category: Category | None,
        query: dict[str, Value],
        candidate: Candidate,
        checks: tuple[Check, ...],
        unreadable: tuple[Attribute, ...],
        unread: frozenset[str],
    ) -> Kind:
        """An analog needs a confirmed hard rule; «the same» needs every hard attribute compared.

        A hard value the query states but nobody read is not a passed rule: «3х2,5+1х1,5» is not «3х2,5».
        A category without hard attributes has nothing critical to confirm: the text and soft rules decide.
        """
        if category is None or candidate.category_code != category.code:
            return Kind.TEXT
        hard = _hard(category)
        if any(a.code in unread for a in hard) or (hard and not any(c.outcome is Outcome.MATCH for c in checks)):
            return Kind.UNVERIFIED  # only the text is similar: «ТМЛ-50» and «ТМЛ-35» are not compared
        compared = not unreadable and all(a.code in query and a.code in candidate.attributes for a in hard)
        return Kind.SAME if compared and all(c.exact for c in checks) else Kind.ANALOG


def _hard(category: Category | None) -> list[Attribute]:
    return [a for a in category.attributes if a.level is Level.HARD] if category else []
