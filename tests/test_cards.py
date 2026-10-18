"""A specialist's verdicts: golden cards keyed by what a query reads into, and the search answering from them."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.bootstrap import Services
from app.config import PROJECT_ROOT
from domain.cards import Reference, UnknownPosition, card_key
from domain.gate import Reason
from domain.knowledge import Knowledge
from domain.matching import Decision, Kind, Verdict
from domain.positions import ParsedPosition
from repositories.memory import MemoryCards
from services.cards import CardService
from services.search import SearchService
from tests.test_meaning import VerdictReader, searched

CABLE = "кабель КВВГнг-LS 7*1.5"
DUCT = "Воздуховод оцинкованный d=200 толщина 3 мм"  # the rules do not read the wall: 0,55 мм passes
TRANSFORMER = "трансформатор силовой сухой 1000 кВА"  # 1877 is 1250 кВА: rejected by the rules


class Book:
    def __init__(self, *names: str) -> None:
        self._references = [Reference(source="ksr", code=f"01.{i}", name=name) for i, name in enumerate(names)]

    def nearest(self, text: str, limit: int) -> list[Reference]:
        return self._references[:limit]


@pytest.fixture
def cards(positions: dict[int, ParsedPosition]) -> MemoryCards:
    return MemoryCards(list(positions.values()))


@pytest.fixture
def search(knowledge: Knowledge, positions: dict[int, ParsedPosition], service: SearchService, cards: MemoryCards):
    return searched(knowledge, service, list(positions.values()), cards=cards)


def rows(search: SearchService, query: str) -> list[tuple[int, Kind | None, Decision | None]]:
    return [(a.candidate.row, a.kind, a.confirmed) for a in search.search(query).candidates]


def test_queries_read_into_the_same_structure_share_a_card(service: SearchService) -> None:
    assert service.understand(CABLE).key == service.understand("Кабель контрольный КВВГнг-LS 7х1,5 мм2").key
    assert service.understand(CABLE).key != service.understand("кабель КВВГнг-LS 7*2.5").key
    assert card_key(None, {}, "пломба номерная") == "-|пломба номерная"  # nothing read: by the text


def test_confirmed_rows_come_first_and_win_over_the_rules(search: SearchService, cards: MemoryCards) -> None:
    decide = CardService(search, cards).decide
    decide(CABLE, 2025, Decision.SAME)
    decide(TRANSFORMER, 1877, Decision.ANALOG, note="1250 кВА с запасом")

    assert rows(search, "Кабель контрольный КВВГнг-LS 7х1,5 мм2")[:2] == [
        (2025, Kind.SAME, Decision.SAME),
        (1171, Kind.ANALOG, None),
    ]
    result = search.search(TRANSFORMER)
    assert result.found and result.card is not None
    assert [(a.candidate.row, a.kind, a.rejected) for a in result.candidates] == [(1877, Kind.ANALOG, False)]
    assert result.candidates[0].checks_with  # the disagreement with the rule stays visible in the checks
    assert result.card.links[0].note == "1250 кВА с запасом"


def test_what_a_specialist_turned_down_is_not_offered(search: SearchService, cards: MemoryCards) -> None:
    offered = [row for row, _, _ in rows(search, DUCT)]

    for row in offered:
        CardService(search, cards).decide(DUCT, row, Decision.NOT_ANALOG)

    result = search.search(DUCT)
    assert offered and result.candidates == []
    assert result.not_found is not None and result.not_found.reason is Reason.DISMISSED
    assert "специалист" in result.not_found.message


def test_an_unclear_verdict_is_logged_and_links_nothing(search: SearchService, cards: MemoryCards) -> None:
    assert CardService(search, cards).decide(CABLE, 1171, Decision.UNCLEAR) is None
    assert cards.log[-1].decision is Decision.UNCLEAR and cards.cards() == []


def test_the_latest_verdict_on_a_row_holds(search: SearchService, cards: MemoryCards) -> None:
    decide = CardService(search, cards).decide
    decide(CABLE, 1171, Decision.SAME)

    card = decide(CABLE, 1171, Decision.NOT_ANALOG)

    assert card is not None and [(k.row, k.decision) for k in card.links] == [(1171, Decision.NOT_ANALOG)]
    assert [e.decision for e in cards.log] == [Decision.SAME, Decision.NOT_ANALOG]


def test_a_verdict_on_an_unknown_row_is_refused(search: SearchService, cards: MemoryCards) -> None:
    with pytest.raises(UnknownPosition):
        CardService(search, cards).decide(CABLE, 999_999, Decision.SAME)


def test_the_judge_is_not_asked_about_confirmed_rows(
    knowledge: Knowledge, positions: dict[int, ParsedPosition], service: SearchService, cards: MemoryCards
) -> None:
    formula = "Кабель ВВГнг(А)-LS 3х2,5+1х1,5"
    reader = VerdictReader(Verdict.OTHER)
    search = searched(knowledge, service, list(positions.values()), reader, cards=cards)
    CardService(search, cards).decide(formula, 280, Decision.SAME)

    result = search.search(formula)

    assert positions[280].name not in reader.shown
    assert result.found and (result.candidates[0].candidate.row, result.candidates[0].kind) == (280, Kind.SAME)


def test_a_linked_row_the_search_did_not_reach_is_fetched(
    knowledge: Knowledge, positions: dict[int, ParsedPosition], service: SearchService, cards: MemoryCards
) -> None:
    far = positions[1541].model_copy(update={"name": "SKAT-V.12DC-18 исп.5М", "search_text": "skat-v.12dc-18 исп.5м"})
    search = searched(knowledge, service, [*(p for r, p in positions.items() if r != 1541), far], cards=cards)
    CardService(search, cards).decide("пломба номерная пластиковая", 1541, Decision.ANALOG)

    assert rows(search, "пломба номерная пластиковая")[0] == (1541, Kind.ANALOG, Decision.ANALOG)


def test_a_new_card_is_anchored_to_the_ksr_ideal_that_reads_the_same(search: SearchService, cards: MemoryCards) -> None:
    ideal = "Кабель контрольный КВВГнг(А)-LS 7х1,5"  # нг(А)-LS for нг-LS: soft, the hard ones decide
    book = Book("Кабель контрольный КВВГнг(А)-LS 7х2,5", ideal)

    card = CardService(search, cards, book).decide(CABLE, 1171, Decision.ANALOG)

    assert card is not None and card.reference == Reference(source="ksr", code="01.1", name=ideal)


def test_no_ideal_when_the_query_leaves_a_hard_attribute_unread(search: SearchService, cards: MemoryCards) -> None:
    book = Book("Изделия фасонные для воздуховодов из оцинкованной стали, толщина 0,5-0,55 мм, диаметр 200 мм")

    card = CardService(search, cards, book).decide(DUCT, 847, Decision.NOT_ANALOG)

    assert card is not None and card.reference is None  # the wall the query asks for is not read


def test_demo_verdicts_load_once_and_export_as_evaluation_queries(search: SearchService, cards: MemoryCards) -> None:
    service = CardService(search, cards)
    demo = PROJECT_ROOT / "demo" / "decisions.json"

    first, again = service.load(demo, "демо", "positions.xlsx"), service.load(demo, "демо", "positions.xlsx")

    assert first["decisions"] >= 10 and first["unknown_rows"] == 0
    assert again == {"decisions": 0, "kept": first["decisions"], "unknown_rows": 0}
    exported = {q["query"]: q for q in service.export()}
    assert exported[DUCT]["expected_kind"] == "no_match"
    assert exported["ворота откатные 6000х2250"]["expected_ids"] == [159]


def test_api_decisions_and_cards(search: SearchService, cards: MemoryCards) -> None:
    services = Services(search, CardService(search, cards))
    with TestClient(create_app(build=lambda: services), raise_server_exceptions=False) as client:
        candidate = client.post("/api/search", json={"query": CABLE}).json()["candidates"][1]
        decided = {"query": CABLE, "position_id": candidate["id"], "decision": "same", "note": "проверено"}

        card = client.post("/api/decisions", json=decided).json()["card"]
        body = client.post("/api/search", json={"query": CABLE}).json()

        assert [(k["row"], k["decision"], k["note"]) for k in card["links"]] == [
            (candidate["row"], "same", "проверено")
        ]
        assert (body["candidates"][0]["row"], body["candidates"][0]["confirmed"]) == (candidate["row"], "same")
        assert body["card"]["id"] == card["id"] and client.get("/api/cards").json()[0]["id"] == card["id"]
        assert client.post("/api/decisions", json={**decided, "position_id": 999_999}).status_code == 400
        assert client.post("/api/decisions", json={**decided, "decision": "maybe"}).status_code == 422
