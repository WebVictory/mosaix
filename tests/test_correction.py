"""A specialist corrects what was understood from a query, as a shop's filters: the category, a value of an
attribute's series, an attribute cleared; the search and the golden card go by the corrected structure."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from analog_search.app.api import create_app
from analog_search.app.bootstrap import Services
from analog_search.domain.matching import Decision
from analog_search.repositories.knowledge import ParsedPosition
from analog_search.repositories.memory import MemoryCards
from analog_search.services.cards import CardService
from analog_search.services.search import Correction, InvalidQuery, SearchService, Source

CABLE = "кабель контрольный 7 жил"  # the section is not stated
SECTION_1_5 = "v2"  # 1,5 мм² in the series of сечение_жилы


def rows(service: SearchService, query: str, correction: Correction | None = None) -> list[int]:
    return [a.candidate.row for a in service.search(query, correction).candidates]


def test_every_attribute_of_the_category_is_shown_stated_or_not(service: SearchService) -> None:
    facets = {f.attribute.code: f for f in service.search(CABLE).facets}

    assert list(facets) == ["число_жил", "сечение_жилы", "экран", "исполнение_пожарное", "номинальное_напряжение"]
    cores, section = facets["число_жил"], facets["сечение_жилы"]
    assert (cores.value, cores.source, cores.option) == ("7 шт", Source.RULES, "v2")
    assert (section.value, section.source) == (None, None) and section.options[2].label == "1,5 мм²"
    assert [o.label for o in facets["исполнение_пожарное"].options][:3] == ["нг", "нг(А)", "нг(А)-LS"]
    assert facets["номинальное_напряжение"].options == []  # informational: not compared, nothing to set


def test_a_value_set_by_a_specialist_narrows_the_search(service: SearchService) -> None:
    corrected = Correction(values={"сечение_жилы": SECTION_1_5})

    result = service.search(CABLE, corrected)

    section = next(f for f in result.facets if f.attribute.code == "сечение_жилы")
    assert (section.value, section.source) == ("1,5 мм²", Source.SPECIALIST)
    assert rows(service, CABLE, corrected)[0] == rows(service, "Кабель контрольный 7х1,5")[0]
    assert service.understand(CABLE, corrected).key == service.understand("кабель контрольный 7х1,5").key


def test_a_cleared_value_is_not_compared(service: SearchService) -> None:
    cleared = service.understand("кабель контрольный 7х1,5", Correction(values={"число_жил": None}))

    assert "число_жил" not in cleared.values and cleared.corrected == {"число_жил"}
    assert 57 in rows(service, "кабель контрольный 7х1,5", Correction(values={"число_жил": None}))  # 5х1,5


def test_a_category_set_by_a_specialist_reads_the_query_with_its_rules(service: SearchService) -> None:
    read = service.understand(CABLE, Correction(category="kabel_silovoy"))

    assert read.category is not None and read.category.code == "kabel_silovoy"
    assert read.category_source is Source.SPECIALIST and read.values == {"число_жил": 7.0}


@pytest.mark.parametrize(
    ("correction", "message"),
    [
        (Correction(category="nope"), "Нет категории"),
        (Correction(values={"мощность": "v0"}), "нет характеристики"),
        (Correction(values={"сечение_жилы": "v99"}), "нет значения"),
        (Correction(values={"номинальное_напряжение": "v0"}), "нет значения"),
    ],
)
def test_a_correction_that_makes_no_sense_is_refused(
    service: SearchService, correction: Correction, message: str
) -> None:
    with pytest.raises(InvalidQuery, match=message):
        service.understand(CABLE, correction)


def test_api_search_and_verdict_with_a_correction(service: SearchService, positions: dict[int, ParsedPosition]) -> None:
    cards = MemoryCards(list(positions.values()))
    services = Services(service, CardService(service, cards))
    correction = {"values": {"сечение_жилы": SECTION_1_5}}

    with TestClient(create_app(build=lambda: services), raise_server_exceptions=False) as client:
        body = client.post("/api/search", json={"query": CABLE, "correction": correction}).json()
        first = body["candidates"][0]
        client.post(
            "/api/decisions",
            json={"query": CABLE, "position_id": first["id"], "decision": Decision.SAME, "correction": correction},
        )
        categories = client.get("/api/categories").json()
        bad = client.post("/api/search", json={"query": CABLE, "correction": {"category": "nope"}})

    section = next(f for f in body["facets"] if f["code"] == "сечение_жилы")
    assert (section["value"], section["source"], section["option"]) == ("1,5 мм²", "specialist", SECTION_1_5)
    assert (body["category_code"], body["category_source"]) == ("kabel_kontrolnyy", "rules") and body["group"]
    [card] = cards.cards()
    assert card.key == service.understand("кабель контрольный 7х1,5").key  # the card of the corrected structure
    assert {"code": "kabel_silovoy", "name": "кабель силовой", "group": body["group"]} in categories
    assert bad.status_code == 400
