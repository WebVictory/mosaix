"""Attributes read by the choice model: what may be asked, what is kept, the cache, the Jev contract."""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx
import pytest

from analog_search.domain.features import Vocabulary
from analog_search.domain.knowledge import Attribute, Category, Knowledge, normalize
from analog_search.domain.matching import Kind
from analog_search.domain.positions import ParsedPosition
from analog_search.domain.reading import NONE, OTHER, Choice, ChoiceReading, Question, ReaderError, question, verify
from analog_search.infrastructure.llm.cache import CachedReader
from analog_search.infrastructure.llm.jev import JevReader
from analog_search.repositories.memory import MemoryStore
from analog_search.services.knowledge import read_hard
from analog_search.services.search import Reason, SearchService, Source


class FirstNumberReader:
    """A stand-in for the model: the first number of the name that is in the series; words are never read.

    unlisted: OTHER when the name has numbers but none of them is in the series.
    """

    def __init__(self, unlisted: bool = False) -> None:
        self.calls = 0
        self._unlisted = unlisted

    def choose(self, state: str, questions: list[Question]) -> dict[str, Choice]:
        self.calls += 1
        numbers = [float(n.replace(",", ".")) for n in re.findall(r"\d+(?:[.,]\d+)?", state.split("«", 1)[1])]
        answers = {}
        for q in questions:
            hits = [o for n in numbers for o, v in q.values.items() if isinstance(v, float) and v == n]
            numeric = any(isinstance(v, float) for v in q.values.values())
            answers[q.code] = Choice(
                hits[0] if hits else OTHER if self._unlisted and numbers and numeric else NONE, 1.0
            )
        return answers


class FailingReader:
    def choose(self, state: str, questions: list[Question]) -> dict[str, Choice]:
        raise ReaderError("недоступен")


def lug(knowledge: Knowledge) -> Category:
    return next(c for c in knowledge.categories if c.code == "nakonechnik_kabelnyy_gilza")


def attribute(category: Category, code: str) -> Attribute:
    return next(a for a in category.attributes if a.code == code)


def test_question_offers_the_series_and_none(knowledge: Knowledge) -> None:
    asked = question(attribute(lug(knowledge), "сечение"), None)

    assert asked is not None
    assert asked.options["v8"] == "50 мм²" and asked.values["v8"] == 50.0
    assert NONE in asked.options and NONE not in asked.values


def test_attribute_an_extractor_reads_in_its_own_words_is_not_asked(knowledge: Knowledge) -> None:
    material = attribute(lug(knowledge), "материал")

    assert question(material, None) is not None  # words: only the model reads it
    own_words = Vocabulary({"медь": ("медн",), "алюминий": ("алюмин",)})
    assert question(material, own_words) is None  # its values are not the series': they would not compare


@pytest.mark.parametrize(
    ("option", "confidence", "text", "expected"),
    [
        ("v8", 0.99, "наконечник тмл 50", 50.0),
        ("v9", 0.99, "наконечник тмл 50", None),  # 70 is not in the name: invented
        ("v8", 0.5, "наконечник тмл 50", None),  # not sure
        (NONE, 1.0, "наконечник тмл 50", None),
        ("v8", 0.99, "наконечник 0.05 тыс", 50.0),  # the same number in another unit
    ],
)
def test_verify_keeps_only_a_sure_choice_stated_in_the_name(
    knowledge: Knowledge, option: str, confidence: float, text: str, expected: float | None
) -> None:
    asked = question(attribute(lug(knowledge), "сечение"), None)
    assert asked is not None

    assert verify(asked, Choice(option, confidence), text, 0.9) == expected


def test_word_value_passes_on_confidence_alone(knowledge: Knowledge) -> None:
    asked = question(attribute(lug(knowledge), "материал"), None)
    assert asked is not None

    assert verify(asked, Choice("v0", 0.95), normalize("Наконечник ТМЛ 50"), 0.9) == "Cu"


def test_number_the_text_confirms_needs_less_confidence_than_a_word(knowledge: Knowledge) -> None:
    section = question(attribute(lug(knowledge), "сечение"), None)
    material = question(attribute(lug(knowledge), "материал"), None)
    assert section is not None and material is not None
    text = normalize("Наконечник ТМЛ 50")

    assert verify(section, Choice("v8", 0.85), text, 0.9, min_stated=0.8) == 50.0
    assert verify(section, Choice("v9", 0.99), text, 0.9, min_stated=0.8) is None  # 70: not in the text
    assert verify(section, Choice("v8", 0.7), text, 0.9, min_stated=0.8) is None
    assert verify(material, Choice("v0", 0.85), text, 0.9, min_stated=0.8) is None  # nothing to check a word by


def test_reading_asks_only_what_the_rules_left_and_may_be_guessed(knowledge: Knowledge) -> None:
    reader = FirstNumberReader()
    reading = ChoiceReading(reader, knowledge.categories, 0.9)
    category = lug(knowledge)

    assert reading.read(category, "Наконечник ТМЛ 50", {}).values == {"сечение": 50.0}
    everything = {"сечение": 50.0, "диаметр_под_болт": 8.0, "материал": "Cu", "тип": "кольцевой"}
    assert reading.read(category, "Наконечник ТМЛ 50", everything).values == {}
    assert reading.read(category, "Наконечник ТМЛ 50", {}, skip=frozenset(reading.readable(category))).values == {}
    assert reader.calls == 1


class ConstantReader:
    def __init__(self, choice: Choice) -> None:
        self._choice = choice

    def choose(self, state: str, questions: list[Question]) -> dict[str, Choice]:
        return {q.code: self._choice for q in questions}


@pytest.mark.parametrize(
    ("choice", "unsure", "unanswered"),
    [
        (Choice(OTHER, 0.95), {"сечение"}, set()),  # stated, not in the series: nothing found is confirmed
        (Choice("v9", 0.95), {"сечение"}, set()),  # 70 is not in the name: stated some other way
        (Choice(NONE, 0.95), set(), set()),  # not stated
        (Choice("v8", 0.6), set(), {"сечение"}),  # a hesitant model adds nothing: as without it
        (Choice(NONE, 0.6), set(), {"сечение"}),
    ],
)
def test_only_a_sure_answer_changes_what_the_rules_left(
    knowledge: Knowledge, choice: Choice, unsure: set[str], unanswered: set[str]
) -> None:
    reading = ChoiceReading(ConstantReader(choice), knowledge.categories, 0.9)
    known = {"диаметр_под_болт": 8.0, "материал": "Cu", "тип": "кольцевой"}

    read = reading.read(lug(knowledge), "Наконечник 50", known)

    assert read.values == {} and read.unsure == unsure and read.unanswered == unanswered


def test_cache_answers_without_the_model_and_keeps_fresh_answers(tmp_path: Path, knowledge: Knowledge) -> None:
    asked = question(attribute(lug(knowledge), "сечение"), None)
    assert asked is not None
    path = tmp_path / "llm" / "readings.jsonl"
    model = FirstNumberReader()

    live = CachedReader(path, "jev", inner=model)
    assert live.choose("Позиция: «ТМЛ 50»", [asked]) == {"сечение": Choice("v8", 1.0)}
    assert live.choose("Позиция: «ТМЛ 50»", [asked]) == {"сечение": Choice("v8", 1.0)}
    assert model.calls == 1 and live.calls == 1

    entry = json.loads(path.read_text(encoding="utf-8"))
    assert entry["answer"] == "50 мм²" and entry["model"] == "jev"
    offline = CachedReader(path, "jev")
    assert offline.choose("Позиция: «ТМЛ 50»", [asked]) == {"сечение": Choice("v8", 1.0)}
    assert offline.choose("Позиция: «ТМЛ 70»", [asked]) == {}  # not kept, no model: unread
    assert CachedReader(path, "another-model").choose("Позиция: «ТМЛ 50»", [asked]) == {}


def jev(handler) -> JevReader:
    return JevReader("test-key", "typesafe/jev-1.13", max_retries=1, transport=httpx.MockTransport(handler))


def test_jev_request_and_answer(knowledge: Knowledge) -> None:
    asked = question(attribute(lug(knowledge), "сечение"), None)
    assert asked is not None
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(200, json={"answers": {"сечение": {"choice": "v8", "probabilities": {"v8": 0.97}}}})

    assert jev(handler).choose("Позиция: «ТМЛ 50»", [asked]) == {"сечение": Choice("v8", 0.97)}
    assert sent[0]["questions"]["сечение"]["type"] == "choice"
    assert sent[0]["questions"]["сечение"]["criteria"] == asked.options


def test_jev_answer_outside_the_question_is_an_error(knowledge: Knowledge) -> None:
    asked = question(attribute(lug(knowledge), "сечение"), None)
    assert asked is not None
    reader = jev(lambda _: httpx.Response(200, json={"answers": {"сечение": {"choice": "v99", "confidence": 1}}}))

    with pytest.raises(ReaderError, match="вне вопроса"):
        reader.choose("Позиция: «ТМЛ 50»", [asked])


def test_jev_retries_then_gives_up(monkeypatch: pytest.MonkeyPatch, knowledge: Knowledge) -> None:
    monkeypatch.setattr("analog_search.infrastructure.llm.jev.time.sleep", lambda _: None)
    asked = question(attribute(lug(knowledge), "сечение"), None)
    assert asked is not None
    statuses = iter([503, 200])

    def handler(_: httpx.Request) -> httpx.Response:
        status = next(statuses, 503)
        return httpx.Response(status, json={"answers": {"сечение": {"choice": NONE, "confidence": 1}}})

    assert jev(handler).choose("s", [asked]) == {"сечение": Choice(NONE, 1)}
    with pytest.raises(ReaderError, match="после 2 попыток"):
        jev(lambda _: httpx.Response(429)).choose("s", [asked])


DUCT = "Воздуховод оцинкованный d=200 толщина 0,5 мм"


def ducts(positions: dict[int, ParsedPosition]) -> list[ParsedPosition]:
    return [p for p in positions.values() if p.category_code == "vozdukhovod"]


def test_duct_thickness_read_by_the_model_is_checked_instead_of_unread(
    knowledge: Knowledge, positions: dict[int, ParsedPosition], service: SearchService
) -> None:
    reading = ChoiceReading(FirstNumberReader(), knowledge.categories, 0.9)
    read, counts = read_hard(knowledge, list(positions.values()), reading, workers=2)
    search = SearchService(knowledge, MemoryStore(knowledge.version, read), service.config, reading, "live")

    by_rules = service.search(DUCT).candidates[0]
    result = search.search(DUCT)

    assert counts["llm_read"] > 0 and counts["llm_failed"] == 0
    assert [u.code for u in by_rules.unreadable] == ["толщина_стали"]  # no extractor reads the thickness
    assert [(f.attribute.code, f.source) for f in result.facets if f.value] == [
        ("сечение", Source.RULES),
        ("толщина_стали", Source.LLM),
    ]
    best = result.candidates[0]
    assert result.found and best.kind is Kind.ANALOG and not best.unreadable
    assert [(c.attribute.code, c.query_by_llm, c.candidate_by_llm) for c in best.checks] == [
        ("сечение", False, False),
        ("толщина_стали", True, True),
    ]
    assert search.stats()["with_llm_readings"] == counts["llm_read"]


def test_model_failure_leaves_the_rules_reading(
    knowledge: Knowledge, positions: dict[int, ParsedPosition], service: SearchService
) -> None:
    reading = ChoiceReading(FailingReader(), knowledge.categories, 0.9)
    read, counts = read_hard(knowledge, ducts(positions), reading, workers=2)
    search = SearchService(knowledge, MemoryStore(knowledge.version, read), service.config, reading, "live")

    assert read == ducts(positions) and counts["llm_failed"] == counts["llm_asked"] > 0
    best = search.search(DUCT).candidates[0]
    assert best.kind is Kind.ANALOG and [u.code for u in best.unreadable] == ["толщина_стали"]


def test_value_outside_the_series_is_not_silence(
    knowledge: Knowledge, positions: dict[int, ParsedPosition], service: SearchService
) -> None:
    reading = ChoiceReading(FirstNumberReader(unlisted=True), knowledge.categories, 0.9)
    read, _ = read_hard(knowledge, ducts(positions), reading, workers=2)
    search = SearchService(knowledge, MemoryStore(knowledge.version, read), service.config, reading, "live")
    query = "Воздуховод оцинкованный d=200 толщина 3 мм"  # 3 мм is not in the series

    duct = next(c for c in knowledge.categories if c.code == "vozdukhovod")
    assert reading.read(duct, query, {"сечение": [200.0]}).unsure == {"толщина_стали"}
    result = search.search(query)

    assert result.candidates and {a.kind for a in result.candidates} == {Kind.UNVERIFIED}
    assert result.not_found is not None and result.not_found.reason is Reason.UNVERIFIED
