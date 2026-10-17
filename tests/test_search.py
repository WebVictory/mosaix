"""Search over the real history held in memory: found rows, honest misses with a reason, the HTTP contract."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from analog_search.app.api import create_app
from analog_search.app.bootstrap import Services
from analog_search.domain.knowledge import Knowledge
from analog_search.domain.matching import Kind
from analog_search.repositories.knowledge import ParsedPosition
from analog_search.repositories.memory import MemoryCards, MemoryStore
from analog_search.services.cards import CardService
from analog_search.services.search import InvalidQuery, Reason, SearchService


def rows(service: SearchService, query: str) -> list[int]:
    return [a.candidate.row for a in service.search(query).candidates]


def test_same_cable_first_and_fire_resistant_one_as_its_analog(service: SearchService) -> None:
    result = service.search("кабель КВВГнг-LS 7*1.5")

    assert result.category is not None and result.category.code == "kabel_kontrolnyy"
    assert [a.candidate.row for a in result.candidates[:2]] == [1171, 2025]  # FRLS covers the requested LS


def test_size_written_the_other_way_round_is_the_same_gate(service: SearchService) -> None:
    assert rows(service, "ворота откатные 6000х2250")[0] == 159


@pytest.mark.parametrize(
    ("query", "reason", "closest"),
    [
        ("трансформатор силовой сухой 1000 кВА", Reason.HARD_MISMATCH, 1877),  # ТСЗ-1250: power is not 1000
        ("Наконечник кабельный ТМЛ 50", Reason.HARD_MISMATCH, None),  # the mark is read: no ring lug of 50 мм²
        ("огнетушитель порошковый ОП-5", Reason.CATEGORY_ABSENT, None),
        ("насос циркуляционный Grundfos UPS 25-60", Reason.UNKNOWN_CATEGORY, None),
    ],
)
def test_miss_names_its_reason(service: SearchService, query: str, reason: Reason, closest: int | None) -> None:
    result = service.search(query)

    assert not result.found and result.not_found is not None
    assert result.not_found.reason is reason
    if closest is not None:
        assert result.not_found.closest is not None and result.not_found.closest.candidate.row == closest


@pytest.mark.parametrize(
    ("query", "closest"),
    [
        ("Кабель ВВГнг(А)-LS 3х2,5+1х1,5", 280),  # the formula is not read: 3х2,5 must not pass for it
    ],
)
def test_unverified_candidates_are_shown_but_not_found(service: SearchService, query: str, closest: int) -> None:
    result = service.search(query)

    assert not result.found and result.not_found is not None
    assert result.not_found.reason is Reason.UNVERIFIED
    assert result.candidates and {a.kind for a in result.candidates} == {Kind.UNVERIFIED}
    assert closest in rows(service, query)


def test_twisted_pair_section_is_compared(service: SearchService) -> None:
    result = service.search("Кабель КПСЭнг(А)-FRLS 2х2х0,75")  # only 2х2х0,5 and 1х2х0,75 exist

    assert not result.found and result.candidates == []  # 2х2х0,5 rejected by the section, 1х2х0,75 by the pairs


@pytest.mark.parametrize("query", ["", "  12 345 ", "кабель " * 100])
def test_query_without_words_or_too_long_is_invalid(service: SearchService, query: str) -> None:
    with pytest.raises(InvalidQuery):
        service.search(query)


def test_stats_warn_when_positions_were_parsed_by_another_release(
    service: SearchService, knowledge: Knowledge, positions: dict[int, ParsedPosition]
) -> None:
    stale = SearchService(knowledge, MemoryStore("0-old", list(positions.values())), service.config)

    assert service.stats()["warnings"] == []
    assert "0-old" in stale.stats()["warnings"][0]


@pytest.fixture
def client(service: SearchService, positions: dict[int, ParsedPosition]):
    services = Services(service, CardService(service, MemoryCards(list(positions.values()))))
    with TestClient(create_app(build=lambda: services), raise_server_exceptions=False) as client:
        yield client


def test_api_search_returns_candidates_with_checks(client: TestClient) -> None:
    body = client.post("/api/search", json={"query": "кабель КВВГнг-LS 7*1.5"}).json()

    assert body["found"] and body["category"] == "кабель контрольный"
    first = body["candidates"][0]
    assert (first["row"], first["kind"]) == (1171, "analog")  # screened нг(А)-LS meets the rules, but is not the same
    assert {(c["code"], c["exact"]) for c in first["matches"]} == {
        ("число_жил", True), ("сечение_жилы", True), ("экран", False), ("исполнение_пожарное", False),
    }  # fmt: skip


def test_api_miss_has_reason_and_closest(client: TestClient) -> None:
    body = client.post("/api/search", json={"query": "трансформатор силовой сухой 1000 кВА"}).json()

    assert body["found"] is False and body["candidates"] == []
    assert body["explain_not_found"]["reason"] == "hard_mismatch"
    assert body["explain_not_found"]["closest"]["kind"] is None


def test_api_unverified_returns_candidates_with_the_reason(client: TestClient) -> None:
    body = client.post("/api/search", json={"query": "Кабель ВВГнг(А)-LS 3х2,5+1х1,5"}).json()

    assert body["found"] is False and body["explain_not_found"]["reason"] == "unverified"
    assert body["candidates"] and {c["kind"] for c in body["candidates"]} == {"unverified"}
    assert body["candidates"][0]["unchecked"]


def test_api_errors(client: TestClient, service: SearchService, monkeypatch: pytest.MonkeyPatch) -> None:
    assert client.post("/api/search", json={"query": "123"}).status_code == 400
    assert client.post("/api/search", json={}).status_code == 422

    def broken(*_: object) -> None:
        raise RuntimeError("store is down")

    monkeypatch.setattr(service, "search", broken)
    response = client.post("/api/search", json={"query": "кабель"})
    assert response.status_code == 500
    assert response.json()["trace_id"] and "store is down" not in response.text


def test_api_stats_and_page(client: TestClient) -> None:
    stats = client.get("/api/stats").json()
    assert stats["positions"] == 2381
    assert stats["hard_attributes_without_extractor"] <= 114  # ratchet: a new hard rule comes with its reader
    assert "Поиск аналогов" in client.get("/").text
