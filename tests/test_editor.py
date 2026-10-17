"""A specialist's edits of the knowledge: the draft, the suggestions from golden cards, the impact check,
publishing and re-keying the cards."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from analog_search.app.api import create_app
from analog_search.app.bootstrap import Services
from analog_search.app.config import PROJECT_ROOT
from analog_search.domain.cards import CardKey, Entry
from analog_search.domain.editing import Edit, Field, InvalidEdit, apply, validated
from analog_search.domain.knowledge import Knowledge
from analog_search.domain.matching import Decision
from analog_search.domain.positions import ParsedPosition
from analog_search.infrastructure.knowledge import positions_xlsx
from analog_search.infrastructure.knowledge.yaml_files import YamlKnowledge
from analog_search.repositories.memory import MemoryCards, MemoryKnowledge, MemoryPositions, MemoryStore
from analog_search.services.cards import CardService
from analog_search.services.editor import KnowledgeEditor
from analog_search.services.knowledge import KnowledgeService
from analog_search.services.search import SearchService

SOURCE = "positions.xlsx"
TRANSFORMER = "трансформатор силовой сухой 1000 кВА"  # 1877 is 1250 кВА: the rules allow no step
POWER = Edit(category="transformator_silovoy", attribute="мощность", field=Field.STEP_TOLERANCE, value=1)


class Bench:
    """The editor over the research release and the workbook, everything in memory."""

    def __init__(self, knowledge: Knowledge, positions: dict[int, ParsedPosition], service: SearchService):
        self.config = service.config
        self.cards = MemoryCards(list(positions.values()))
        self.store = MemoryKnowledge(knowledge)
        self.positions = MemoryPositions(SOURCE, list(positions.values()))
        self.editor = KnowledgeEditor(
            self.store,
            KnowledgeService(YamlKnowledge(PROJECT_ROOT / "research"), positions_xlsx.read, self.store, self.positions),
            self.positions,
            self.cards,
            self.config,
            MemoryStore,
            SOURCE,
            PROJECT_ROOT / "evaluation" / "gold_queries.json",
        )

    def services(self) -> Services:
        knowledge = self.store.active()
        store = MemoryStore(knowledge.version, self.positions.stored()[SOURCE])
        search = SearchService(knowledge, store, self.config, cards=self.cards)
        return Services(search, CardService(search, self.cards), self.editor)


@pytest.fixture
def bench(knowledge: Knowledge, positions: dict[int, ParsedPosition], service: SearchService) -> Bench:
    bench = Bench(knowledge, positions, service)
    bench.services().cards.decide(TRANSFORMER, 1877, Decision.ANALOG, note="1250 кВА с запасом")
    return bench


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (
            Edit(category="transformator_silovoy", attribute="мощность", field=Field.LEVEL, value="maybe"),
            "Недопустимое",
        ),
        (POWER.model_copy(update={"value": 9}), "Допуск"),
        (Edit(category="transformator_silovoy", attribute="мощность", field=Field.TRIGGERS, value=["x"]), "нет поля"),
        (Edit(category="vozdukhovod", attribute="сечение", field=Field.COMPARISON, value="step"), "числового ряда"),
        (Edit(category="vozdukhovod", attribute=None, field=Field.TRIGGERS, value=[]), "не останется"),
        (Edit(category="nope", attribute=None, field=Field.TRIGGERS, value=["x"]), "Нет категории"),
    ],
)
def test_an_edit_that_makes_no_sense_is_refused(knowledge: Knowledge, edit: Edit, message: str) -> None:
    with pytest.raises(InvalidEdit, match=message):
        validated(knowledge, edit)


def test_a_release_is_named_by_its_content(knowledge: Knowledge) -> None:
    edited = apply(knowledge, [validated(knowledge, POWER)])
    words = validated(
        knowledge,
        Edit(category="vozdukhovod", attribute=None, field=Field.TRIGGERS, value=[" Воздуховод ", "воздуховод"]),
    )

    assert edited.version.startswith("1-") and edited.version != knowledge.version
    assert edited == apply(knowledge, [validated(knowledge, POWER)])
    assert words.value == ["воздуховод"]
    power = next(
        a for c in edited.categories for a in c.attributes if c.code == POWER.category and a.code == "мощность"
    )
    assert power.rule["step_tolerance"] == 1 and apply(knowledge, []) is knowledge


def test_a_card_the_rules_reject_suggests_the_edit_that_settles_it(bench: Bench) -> None:
    [suggestion] = bench.editor.disagreements()

    assert (suggestion.row, suggestion.decision, suggestion.note) == (1877, Decision.ANALOG, "1250 кВА с запасом")
    assert "мощность" in suggestion.problem and suggestion.edit == POWER
    assert bench.store.edits() == []  # suggested, not applied


def test_the_draft_keeps_one_edit_per_field_and_drops_one_that_restores_the_release(bench: Bench) -> None:
    bench.editor.edit(POWER.model_copy(update={"value": 2}))
    made = bench.editor.edit(POWER.model_copy(update={"author": "инженер"}))

    assert [(e.value, e.previous, e.author) for e in bench.store.edits()] == [(1, 0, "инженер")]
    assert made is not None and bench.editor.draft().knowledge.version != bench.store.active().version
    assert bench.editor.edit(POWER.model_copy(update={"value": 0})) is None and bench.store.edits() == []


def test_the_check_shows_what_the_draft_settles_and_what_it_breaks(bench: Bench) -> None:
    bench.editor.edit(POWER)

    impact = bench.editor.check()

    assert (impact.disagreements_before, impact.disagreements_after, len(impact.resolved)) == (1, 0, 1)
    assert impact.rows == 0  # a tolerance changes no reading, only the comparison
    [query] = [q for q in impact.changed_queries if q.query == TRANSFORMER]
    assert (query.correct_before, query.correct_after) == (True, False)  # the gold query expects «не найдено»
    assert impact.correct_after == impact.correct_before - 1


def test_publishing_makes_the_draft_active_and_parses_the_history_with_it(bench: Bench) -> None:
    bench.editor.edit(POWER)
    draft = bench.editor.draft().knowledge

    published = bench.editor.publish("инженер", "допуск по мощности")

    assert published["version"] == draft.version == bench.store.active().version
    assert bench.positions.versions == {SOURCE: draft.version} and bench.store.edits() == []
    assert bench.editor.disagreements() == []
    with pytest.raises(InvalidEdit, match="пуст"):
        bench.editor.publish()


def test_cards_that_read_the_same_after_a_release_are_merged(positions: dict[int, ParsedPosition]) -> None:
    cards = MemoryCards(list(positions.values()))

    def entry(key: str, row: int, decision: Decision) -> Entry:
        return Entry(key=key, query=f"запрос {key}", category_code="c", values={}, position_id=row, decision=decision)

    cards.decide(entry("a", 1171, Decision.SAME))
    cards.decide(entry("b", 1171, Decision.NOT_ANALOG))
    cards.decide(entry("b", 164, Decision.ANALOG))
    cards.decide(entry("c", 280, Decision.SAME))

    assert cards.rekey(
        {1: CardKey(key="c", category_code="c", values={}), 3: CardKey(key="a", category_code="c", values={})}
    ) == {"rekeyed": 2, "merged": 0}
    assert [(c.id, c.key) for c in cards.cards()] == [(1, "c"), (2, "b"), (3, "a")]  # traded keys
    assert cards.rekey({2: CardKey(key="c", category_code="c", values={})}) == {"rekeyed": 1, "merged": 1}
    [card] = [c for c in cards.cards() if c.key == "c"]
    assert card.id == 1 and [(k.row, k.decision) for k in card.links] == [
        (164, Decision.ANALOG),
        (1171, Decision.NOT_ANALOG),  # the later verdict holds
    ]


def test_a_card_is_rekeyed_when_its_query_reads_otherwise(bench: Bench) -> None:
    [card] = bench.cards.cards()
    bench.editor.edit(Edit(category="transformator_silovoy", attribute=None, field=Field.TRIGGERS, value=["тдцтн"]))
    bench.editor.publish()

    assert bench.services().cards.rekey() == {"rekeyed": 1, "merged": 0}
    [rekeyed] = bench.cards.cards()
    assert rekeyed.id == card.id and rekeyed.key != card.key and rekeyed.links == card.links


def test_api_edits_check_and_publish(bench: Bench) -> None:
    built = []

    def build() -> Services:
        built.append(bench.services())
        return built[-1]

    with TestClient(create_app(build=build), raise_server_exceptions=False) as client:
        body = client.get("/api/knowledge").json()
        transformer = next(c for c in body["categories"] if c["code"] == POWER.category)
        assert next(a for a in transformer["attributes"] if a["code"] == "мощность")["step_tolerance"] == 0
        assert transformer["rows"] == 2 and any("ТСЗ-1250" in name for name in transformer["samples"])
        [suggestion] = client.get("/api/knowledge/suggestions").json()
        assert suggestion["edit"]["field"] == "step_tolerance" and suggestion["edit"]["value"] == 1

        bad = client.post("/api/knowledge/edits", json={**suggestion["edit"], "value": "много"})
        made = client.post("/api/knowledge/edits", json={**suggestion["edit"], "author": "инженер"}).json()
        impact = client.post("/api/knowledge/check").json()
        published = client.post("/api/knowledge/publish", json={"author": "инженер"}).json()

        assert bad.status_code == 400 and "Допуск" in bad.json()["detail"]
        assert (made["previous"], made["value"]) == (0, 1) and impact["disagreements_after"] == 0
        assert published["edits"] == 1 and published["cards"] == {"rekeyed": 0, "merged": 0}
        assert len(built) == 2 and built[-1].search.knowledge.version == published["version"]
        assert client.delete("/api/knowledge/edits/999").status_code == 404
        assert client.get("/knowledge").status_code == 200
