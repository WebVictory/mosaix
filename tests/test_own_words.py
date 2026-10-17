"""Queries in the user's own words: numbers in words, «N на M», the category and soft attributes by the model."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from analog_search.app.config import Settings
from analog_search.domain.knowledge import Category, Knowledge, normalize
from analog_search.domain.positions import ParsedPosition
from analog_search.domain.reading import (
    CATEGORY,
    NONE,
    Choice,
    ChoiceReading,
    Question,
    ReaderError,
    question,
    verify,
)
from analog_search.domain.spoken import spoken_numbers
from analog_search.infrastructure.llm.embeddings import CachedEmbedder, OpenRouterEmbedder
from analog_search.repositories.memory import MemoryStore
from analog_search.services.search import SearchService, Source


class CategoryReader:
    """A stand-in for the model: picks the category whose name contains the word, attributes are left unread."""

    def __init__(self, word: str, confidence: float = 1.0) -> None:
        self.asked: list[str] = []
        self._word, self._confidence = word, confidence

    def choose(self, state: str, questions: list[Question]) -> dict[str, Choice]:
        answers = {}
        for q in questions:
            if q.code == CATEGORY:
                self.asked = [name for option, name in q.options.items() if option != NONE]
                hit = next((o for o, name in q.options.items() if self._word in name.lower()), NONE)
                answers[q.code] = Choice(hit, self._confidence)
            else:
                answers[q.code] = Choice(NONE, 1.0)
        return answers


class MeaningEmbedder:
    """A stand-in for embeddings: every query means the category named `meant`, nothing else is near."""

    def __init__(self, meant: str) -> None:
        self._meant = meant

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 1.0] if ":" in t and not t.startswith(f"{self._meant}:") else [1.0, 0.0] for t in texts]


class DownEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise ReaderError("недоступно")


def of(knowledge: Knowledge, code: str) -> Category:
    return next(c for c in knowledge.categories if c.code == code)


@pytest.mark.parametrize(
    ("text", "numbers"),
    [
        ("семь жил по полтора", [7.0, 1.5]),
        ("три жилы по два с половиной", [3.0, 2.5]),
        ("тридцать ватт", [30.0]),
        ("сто двадцать пять", [125.0]),
        ("автомат двухполюсный", [2.0]),
        ("кабель силовой", []),
    ],
)
def test_numbers_in_words(text: str, numbers: list[float]) -> None:
    assert spoken_numbers(text) == numbers


@pytest.mark.parametrize(
    ("text", "normalized"),
    [
        ("КВВГ 7 на 1.5", "кввг 7x1.5"),
        ("лоток сто на четыреста", "лоток 100x400"),
        ("ВВГ пять на 95", "ввг 5x95"),
        ("кабель на 7 жил", "кабель на 7 жил"),  # «на» between two numbers only
        ("три жилы по два с половиной", "три жилы по два с половиной"),  # a lone number in words: the model's
    ],
)
def test_n_on_m_is_a_size(text: str, normalized: str) -> None:
    assert normalize(text) == normalized


def test_a_number_in_words_is_stated(knowledge: Knowledge) -> None:
    asked = question(
        next(a for a in of(knowledge, "nakonechnik_kabelnyy_gilza").attributes if a.code == "сечение"), None
    )
    assert asked is not None

    assert verify(asked, Choice("v8", 0.99), normalize("наконечник медный на пятьдесят"), 0.9) == 50.0
    assert verify(asked, Choice("v9", 0.99), normalize("наконечник медный на пятьдесят"), 0.9) is None


def test_a_query_is_asked_soft_attributes_a_row_hard_ones_only(knowledge: Knowledge) -> None:
    reading = ChoiceReading(CategoryReader(""), knowledge.categories, 0.9)
    cable = of(knowledge, "kabel_silovoy")

    assert "изоляция" in reading.readable(cable, soft=True)
    assert "изоляция" not in reading.readable(cable)


def test_misspelled_category_is_among_the_nearest(knowledge: Knowledge) -> None:
    reading = ChoiceReading(CategoryReader(""), knowledge.categories, 0.9)

    nearest = [c.code for c in reading.nearest_categories("трансформатр сухой 1250ква")]

    assert "transformator_silovoy" in nearest and len(nearest) == 8


def test_category_named_in_other_words_is_among_the_nearest_by_meaning(knowledge: Knowledge) -> None:
    query = "бесперебойник 12 вольт"
    by_spelling = ChoiceReading(CategoryReader(""), knowledge.categories, 0.9).nearest_categories(query)
    reading = ChoiceReading(
        CategoryReader(""), knowledge.categories, 0.9, embedder=MeaningEmbedder("стабилизатор напряжения")
    )

    nearest = [c.name for c in reading.nearest_categories(query)]

    assert "стабилизатор напряжения" not in [c.name for c in by_spelling]
    assert nearest[:2] == [by_spelling[0].name, "стабилизатор напряжения"]  # in turn: spelling, meaning
    assert len(nearest) == 8 and len(set(nearest)) == 8


def test_without_embeddings_the_nearest_are_by_spelling(knowledge: Knowledge) -> None:
    query = "трансформатр сухой 1250ква"
    by_spelling = ChoiceReading(CategoryReader(""), knowledge.categories, 0.9).nearest_categories(query)

    assert (
        ChoiceReading(CategoryReader(""), knowledge.categories, 0.9, embedder=DownEmbedder()).nearest_categories(query)
        == by_spelling
    )


def test_embeddings_are_kept_and_read_without_a_key(tmp_path: Path) -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        assert request.headers["Authorization"] == "Bearer test-key"
        data = [{"index": i, "embedding": [float(len(t)), 1.0]} for i, t in reversed(list(enumerate(body["input"])))]
        return httpx.Response(200, json={"data": data})

    path = tmp_path / "embeddings.jsonl"
    live = CachedEmbedder(path, "bge", OpenRouterEmbedder("test-key", "bge", transport=httpx.MockTransport(handler)))

    assert live.embed(["ab", "abc"]) == [[2.0, 1.0], [3.0, 1.0]]  # in the order asked, whatever the order answered
    assert live.embed(["abc"]) == [[3.0, 1.0]] and len(requests) == 1
    assert requests[0] == {"model": "bge", "input": ["ab", "abc"]}
    assert CachedEmbedder(path, "bge").embed(["abc", "ab"]) == [[3.0, 1.0], [2.0, 1.0]]
    with pytest.raises(ReaderError):
        CachedEmbedder(path, "another-model").embed(["ab"])


@pytest.mark.parametrize(
    ("word", "confidence", "expected"),
    [
        ("силов", 0.95, "transformator_silovoy"),
        ("силов", 0.6, None),  # not sure: as without the model
        ("нет такого", 1.0, None),  # NONE
    ],
)
def test_category_is_kept_only_when_the_model_is_sure(
    knowledge: Knowledge, word: str, confidence: float, expected: str | None
) -> None:
    reader = CategoryReader(word, confidence)
    category = ChoiceReading(reader, knowledge.categories, 0.9).category("трансформатр сухой 1250ква")

    assert (category.code if category else None) == expected
    assert reader.asked  # the model chooses among the nearest, it never names a category itself


def test_query_no_trigger_word_matched_is_searched_in_the_category_the_model_chose(
    knowledge: Knowledge, positions: dict[int, ParsedPosition], service: SearchService
) -> None:
    query = "трансформатр сухой 1250ква"
    reading = ChoiceReading(CategoryReader("силов"), knowledge.categories, 0.9)
    search = SearchService(knowledge, MemoryStore(knowledge.version, list(positions.values())), service.config, reading)

    by_rules, result = service.search(query), search.search(query)

    assert by_rules.category is None and by_rules.category_source is None
    assert (
        result.category is not None
        and result.category.code == "transformator_silovoy"
        and result.category_source is Source.LLM
    )


@pytest.mark.parametrize(
    ("llm", "key", "mode"),
    [("auto", "k", "live"), ("auto", None, "cache"), ("cache", "k", "cache"), ("off", "k", "off")],
)
def test_auto_asks_the_model_live_only_with_a_key(llm: str, key: str | None, mode: str) -> None:
    settings = Settings(_env_file=None, llm=llm, openrouter_api_key=key)

    assert settings.llm_mode == mode
