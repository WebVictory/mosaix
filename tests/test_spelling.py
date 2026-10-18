"""Typing mistakes fixed against the words of the base, and what is offered while a query is typed."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.bootstrap import Services
from domain.spelling import Speller
from repositories.memory import MemoryCards
from services.cards import CardService
from services.search import SearchService

BASE = ["Кабель силовой ВВГнг(A)-LS 3х2,5", "Кабель контрольный", "Огнетушитель порошковая зарядка", "ПВХ"]


@pytest.mark.parametrize(
    ("typed", "meant"),
    [
        ("кабль силовй 3х2,5", "кабель силовой 3х2,5"),  # a letter missed
        ("Кабелъ", "Кабель"),  # replaced; the case kept
        ("кабель контрлоьный", "кабель контрольный"),  # two swapped
        ("rf,tkm", "кабель"),  # the English layout
        ("«кабль»,", "«кабель»,"),  # quotes and a comma around
    ],
)
def test_a_mistyped_word_is_taken_for_the_known_one(typed: str, meant: str) -> None:
    assert Speller(BASE).spell(typed).text == meant


@pytest.mark.parametrize(
    "query",
    [
        "кабель ВВГнг-LS 3х2,5 ПВХ",  # known words, a mark, a size
        "огнетушитель порошковый",  # another form of a known word
        "кабель Legrand",  # a word the base does not know and nothing is near
        "пвз",  # too short to say
    ],
)
def test_what_is_known_or_too_short_to_judge_stays(query: str) -> None:
    assert Speller(BASE).spell(query).fixes == ()


def test_the_word_still_being_typed_is_not_a_mistake() -> None:
    speller = Speller(BASE)

    assert speller.spell("кабель конт", typing=True).fixes == ()
    assert speller.spell("кабль конт", typing=True).text == "кабель конт"


def test_a_mistyped_query_finds_its_category_once_fixed(service: SearchService) -> None:
    typed = "кабль контрольный 7х1,5"

    fixed = service.spell(typed).text

    assert fixed == "кабель контрольный 7х1,5"
    assert service.understand(typed).category is None
    assert service.understand(fixed).category is not None


def test_suggestions_while_typing(service: SearchService) -> None:
    suggested = service.suggest("rf,tkm rjynhjkmysq")

    assert suggested.text == "кабель контрольный"
    assert suggested.names and all("онтрольн" in n for n in suggested.names[:2])
    assert service.suggest("ка").names == []


def test_api_suggests_and_the_search_says_how_to_spell(service: SearchService) -> None:
    services = Services(service, CardService(service, MemoryCards([])))
    with TestClient(create_app(build=lambda: services)) as client:
        suggested = client.get("/api/suggest", params={"q": "трансформатр силовой"}).json()
        spelled = client.post("/api/search", json={"query": "трансформатр силовой"}).json()["spelled"]
        right = client.post("/api/search", json={"query": "трансформатор силовой"}).json()["spelled"]

    assert suggested["fixes"] == [{"typed": "трансформатр", "meant": "трансформатор"}]
    assert suggested["text"] == spelled == "трансформатор силовой" and right is None
