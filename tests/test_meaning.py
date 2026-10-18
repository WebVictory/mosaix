"""Candidates by meaning, rows of other categories checked by the query's rules, the model as a judge."""

from __future__ import annotations

import pytest

from app.search_config import SEARCH
from domain.gate import Reason
from domain.judging import Judge
from domain.knowledge import Knowledge
from domain.matching import Candidate, Kind, Matcher, Verdict
from domain.positions import ParsedPosition
from domain.reading import NONE, Choice, Question, ReaderError
from repositories.memory import MemoryStore
from services.search import SearchService

CABLE_7X15 = "кабель КВВГнг-LS 7*1.5"
FORMULA = "Кабель ВВГнг(А)-LS 3х2,5+1х1,5"  # not read by the rules: every candidate stays unverified


class VerdictReader:
    """A stand-in for the model as a judge: the same answer about every candidate it is shown."""

    def __init__(self, option: str, confidence: float = 1.0) -> None:
        self.shown: list[str] = []
        self._option, self._confidence = option, confidence

    def choose(self, state: str, questions: list[Question]) -> dict[str, Choice]:
        self.shown += [q.instructions.split("«")[1].split("»")[0] for q in questions]
        return {q.code: Choice(self._option, self._confidence) for q in questions}


class DownEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise ReaderError("недоступно")


def searched(
    knowledge: Knowledge,
    service: SearchService,
    positions: list[ParsedPosition],
    reader: VerdictReader | None = None,
    **kwargs,
) -> SearchService:
    judge = Judge(reader, SEARCH.confidence.min, SEARCH.candidates.judge_k) if reader else None
    return SearchService(knowledge, MemoryStore(knowledge.version, positions), service.rules, judge=judge, **kwargs)


def misparsed(positions: dict[int, ParsedPosition], *rows: int) -> list[ParsedPosition]:
    """The rows as if the rules had not recognized their category."""
    return [positions[r].model_copy(update={"category_code": None, "attributes": {}}) for r in rows]


def test_row_of_another_category_is_checked_by_the_query_rules(
    knowledge: Knowledge, positions: dict[int, ParsedPosition], service: SearchService
) -> None:
    search = searched(knowledge, service, misparsed(positions, 1171, 164))  # КВВГЭнг 7х1,5 and 7х2,5

    result = search.search(CABLE_7X15)

    assert not result.found and result.not_found is not None and result.not_found.reason is Reason.UNVERIFIED
    assert [(a.candidate.row, a.kind, a.foreign) for a in result.candidates] == [(1171, Kind.UNVERIFIED, True)]


def test_the_judge_never_sees_what_the_rules_rejected_nor_confirms_a_category(
    knowledge: Knowledge, positions: dict[int, ParsedPosition], service: SearchService
) -> None:
    reader = VerdictReader(Verdict.SAME)
    search = searched(knowledge, service, misparsed(positions, 1171, 164), reader)

    result = search.search(CABLE_7X15)

    assert reader.shown == [positions[1171].name]  # 7х2,5: other section, rejected by the rules
    assert result.found and [(a.candidate.row, a.kind, a.verdict) for a in result.candidates] == [
        (1171, Kind.ANALOG, Verdict.SAME)  # the rules never said it is a control cable: an analog at most
    ]


@pytest.mark.parametrize(
    ("verdict", "confidence", "found", "kind", "reason"),
    [
        (Verdict.ANALOG, 0.95, True, Kind.ANALOG, None),
        (Verdict.ANALOG, 0.6, False, Kind.UNVERIFIED, Reason.UNVERIFIED),  # unsure: as the rules left it
        (NONE, 1.0, False, Kind.UNVERIFIED, Reason.UNVERIFIED),
    ],
)
def test_the_judge_settles_what_the_rules_could_not(
    knowledge: Knowledge,
    positions: dict[int, ParsedPosition],
    service: SearchService,
    verdict: str,
    confidence: float,
    found: bool,
    kind: Kind | None,
    reason: Reason | None,
) -> None:
    search = searched(knowledge, service, list(positions.values()), VerdictReader(verdict, confidence))

    result = search.search(FORMULA)

    assert result.found is found
    assert (result.not_found.reason if result.not_found else None) is reason
    assert {a.kind for a in result.candidates} == ({kind} if kind else set())


def test_what_the_judge_dismissed_is_not_shown(
    knowledge: Knowledge, positions: dict[int, ParsedPosition], service: SearchService
) -> None:
    reader = VerdictReader(Verdict.OTHER)
    unverified = [a.candidate.row for a in service.search(FORMULA).candidates]

    result = searched(knowledge, service, [positions[r] for r in unverified], reader).search(FORMULA)

    assert not result.found and result.candidates == []
    assert result.not_found is not None and result.not_found.reason is Reason.DISMISSED


def test_the_judge_is_not_asked_about_what_the_rules_confirmed(
    knowledge: Knowledge, positions: dict[int, ParsedPosition], service: SearchService
) -> None:
    reader = VerdictReader(Verdict.OTHER)

    result = searched(knowledge, service, list(positions.values()), reader).search(CABLE_7X15)

    assert positions[1171].name not in reader.shown
    assert [a.candidate.row for a in result.candidates[:2]] == [1171, 2025]


def test_without_embeddings_the_search_goes_on_by_spelling(
    knowledge: Knowledge, positions: dict[int, ParsedPosition], service: SearchService
) -> None:
    search = searched(knowledge, service, list(positions.values()), embedder=DownEmbedder())

    assert [a.candidate.row for a in search.search(CABLE_7X15).candidates] == [
        a.candidate.row for a in service.search(CABLE_7X15).candidates
    ]


def test_names_close_in_meaning_are_close_in_text() -> None:
    matcher = Matcher(SEARCH.scoring.model_copy(update={"meaning_floor": 0.5}))
    candidate = Candidate(
        id=1, row=1, name="ИБП уличный", price=None, category_code=None, attributes={}, similarity=0.1
    )

    assert matcher.text(candidate) == 0.1
    assert matcher.text(candidate.model_copy(update={"meaning": 0.8})) == pytest.approx(0.6)
    assert matcher.text(candidate.model_copy(update={"meaning": 0.3})) == 0.1  # unrelated: spelling decides


def test_nearest_rows_by_spelling_and_by_meaning(positions: dict[int, ParsedPosition]) -> None:
    rows = [positions[r] for r in (1171, 164, 1541)]  # 1541 — ИБП: no trigram in common with «бесперебойник»
    store = MemoryStore("v", rows, {1171: [0.0, 1.0], 164: [0.0, 1.0], 1541: [1.0, 0.0]})

    by_spelling = [c.row for c in store.nearest("кабель квввгэнг 7x1.5", 1)]
    both = {c.row: c.meaning for c in store.nearest("кабель квввгэнг 7x1.5", 1, [1.0, 0.0])}

    assert by_spelling == [1171]
    assert both == {1171: 0.0, 1541: 1.0}
