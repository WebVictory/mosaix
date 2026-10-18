"""The specialist's substitution rules and the score built on them."""

from __future__ import annotations

from typing import Any

import pytest

from app.search_config import SEARCH
from domain.features import FireRating, IPRating, Pattern, Quantity
from domain.knowledge import Attribute, AttributeStatus, Category, Comparison, Level, Provenance
from domain.matching import Candidate, Kind, Matcher, Outcome, Requirements, satisfies


def attribute(code: str, level: Level, comparison: Comparison, **rule: Any) -> Attribute:
    return Attribute(
        code=code,
        name=code,
        unit=None,
        level=level,
        comparison=comparison,
        provenance=Provenance.SPECIALIST,
        status=AttributeStatus.APPROVED,
        source_ref=None,
        rule=rule,
    )


POWER = attribute("мощность", Level.HARD, Comparison.STEP, series=[630, 1000, 1250, 1600], step_tolerance=0)
IP = attribute("степень_защиты", Level.SOFT, Comparison.NOT_WORSE)
FIRE = attribute("исполнение_пожарное", Level.HARD, Comparison.INCLUDES)
CAMERA = attribute("разрешение", Level.SOFT, Comparison.STEP, series=[1, 2, 4, 8])
CURRENT = attribute("номинальный_ток", Level.HARD, Comparison.EQUAL)


@pytest.mark.parametrize(
    ("rule", "extractor", "query", "candidate", "ok"),
    [
        (POWER, Quantity("apparent_power", 1e3), 1000.0, 1000.0, True),
        (POWER, Quantity("apparent_power", 1e3), 1000.0, 1250.0, False),  # tolerance 0: the next step is not it
        (CAMERA, Quantity("resolution", 1), 2.0, 4.0, True),  # one step either way by default
        (CAMERA, Quantity("resolution", 1), 2.0, 8.0, False),
        (IP, IPRating(), "IP54", "IP65", True),  # every digit not worse
        (IP, IPRating(), "IP65", "IP54", False),
        (IP, IPRating(), "IP54", "IP64", True),
        (FIRE, FireRating(), "нг(А)-LS", "нг(А)-FRLS", True),  # FRLS covers LS
        (FIRE, FireRating(), "нг(А)-FRLS", "нг(А)-LS", False),
        (CURRENT, Pattern(r"(\d+)\s*а"), 6.0, 6.0, True),
    ],
)
def test_satisfies(rule: Attribute, extractor: Any, query: Any, candidate: Any, ok: bool) -> None:
    assert satisfies(rule, extractor, query, candidate) is ok


BREAKERS = Category(code="avtomat", name="автомат", group="аппараты", attributes=(CURRENT, IP))
EXTRACTORS = {"номинальный_ток": Pattern(r"(\d+)\s*а"), "степень_защиты": IPRating()}


def candidate(attributes: dict[str, Any], similarity: float = 0.5, category: str = "avtomat") -> Candidate:
    return Candidate(
        id=1, row=1, name="автомат", price=100.0, category_code=category, attributes=attributes, similarity=similarity
    )


def assess(query: dict[str, Any], offered: Candidate):
    return Matcher(SEARCH.scoring).assess(Requirements(category=BREAKERS, extractors=EXTRACTORS, values=query), offered)


def test_same_values_make_a_same_candidate_with_a_full_score_breakdown() -> None:
    result = assess({"номинальный_ток": 6.0}, candidate({"номинальный_ток": 6.0}))

    assert result.kind is Kind.SAME
    assert result.breakdown == {"text": 0.175, "features": 0.45, "category": 0.2, "penalties": 0.0}
    assert result.score == 0.825


def test_hard_mismatch_rejects_whatever_the_score() -> None:
    result = assess({"номинальный_ток": 6.0}, candidate({"номинальный_ток": 10.0}, similarity=1.0))

    assert result.rejected
    assert [c.outcome for c in result.checks] == [Outcome.MISMATCH]


def test_soft_mismatch_and_unstated_hard_value_lower_the_score() -> None:
    result = assess({"номинальный_ток": 6.0, "степень_защиты": "IP65"}, candidate({"степень_защиты": "IP54"}))

    assert result.breakdown["penalties"] == -0.15  # soft mismatch 0.1 + missing hard value 0.05
    assert [c.outcome for c in result.checks] == [Outcome.MISSING, Outcome.MISMATCH]


def test_nothing_hard_confirmed_is_unverified_not_an_analog() -> None:
    """ТМЛ-50 vs ТМЛ-35: the size was not read, only the text is alike."""
    result = assess({"степень_защиты": "IP54"}, candidate({}))

    assert result.kind is Kind.UNVERIFIED and not result.rejected


def test_hard_value_the_query_states_but_nobody_read_is_unverified() -> None:
    """«3х2,5+1х1,5»: the query is not silent about the section, it is unreadable."""
    offered = candidate({"номинальный_ток": 6.0})
    asked = Requirements(
        category=BREAKERS,
        extractors=EXTRACTORS,
        values={"номинальный_ток": 6.0},
        declined=frozenset({"номинальный_ток"}),
    )
    result = Matcher(SEARCH.scoring).assess(asked, offered)

    assert result.kind is Kind.UNVERIFIED
    assert [a.code for a in result.unreadable] == ["номинальный_ток"]


def test_same_needs_every_hard_attribute_read_and_asked() -> None:
    voltage = attribute("напряжение", Level.HARD, Comparison.EQUAL)
    breakers = Category(code="avtomat", name="автомат", group="аппараты", attributes=(CURRENT, voltage))
    matcher = Matcher(SEARCH.scoring)
    query = {"номинальный_ток": 6.0}

    current = {"номинальный_ток": EXTRACTORS["номинальный_ток"]}
    no_reader = matcher.assess(Requirements(category=breakers, extractors=current, values=query), candidate(query))
    unasked = matcher.assess(
        Requirements(category=breakers, extractors={**current, "напряжение": Quantity("voltage", 1)}, values=query),
        candidate({**query, "напряжение": 230.0}),
    )

    assert (no_reader.kind, [a.code for a in no_reader.unreadable]) == (Kind.ANALOG, ["напряжение"])
    assert unasked.kind is Kind.ANALOG  # the query did not ask for 230 В, so it is not «the same»


def test_better_value_is_an_analog_not_the_same() -> None:
    result = assess({"степень_защиты": "IP54"}, candidate({"степень_защиты": "IP65"}))

    assert (result.kind, result.checks[0].outcome, result.checks[0].exact) == (Kind.ANALOG, Outcome.MATCH, False)


def test_row_of_another_category_gets_no_rules_and_no_category_credit() -> None:
    result = assess({"номинальный_ток": 6.0}, candidate({"номинальный_ток": 10.0}, category="lampa"))

    assert (result.checks, result.kind) == ((), Kind.TEXT)
    assert result.breakdown["category"] == result.breakdown["features"] == 0.0
