"""Knowledge and positions in PostgreSQL: docker compose up -d db; pytest -m integration."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import pytest
from sqlalchemy import delete, select, update

from analog_search.app.config import PROJECT_ROOT, Settings
from analog_search.domain.cards import CardKey, Entry, Reference, UnknownPosition
from analog_search.domain.editing import Edit, Field, fingerprint
from analog_search.domain.knowledge import Category, Knowledge
from analog_search.domain.matching import Decision
from analog_search.infrastructure.knowledge import yaml_files
from analog_search.repositories.cards import CardRepository
from analog_search.repositories.database import Database
from analog_search.repositories.knowledge import KnowledgeRepository, ParsedPosition, PositionRepository
from analog_search.repositories.orm import (
    DecisionRow,
    GoldenCardRow,
    KnowledgeEditRow,
    KnowledgeReleaseRow,
    PositionRow,
)
from analog_search.repositories.search import SearchRepository

pytestmark = pytest.mark.integration

SOURCE = "pytest.xlsx"
KNOWLEDGE = Knowledge(
    version="pytest-1",
    sha256="x",
    categories=(Category("cables", "кабели", None), Category("cable", "кабель", "кабели", triggers=("кабель",))),
)


@pytest.fixture
def db() -> Iterator[Database]:
    db = Database(Settings().database_url)
    db.init_schema()
    previous = KnowledgeRepository(db).active_version()
    yield db
    with db.session() as session, session.begin():
        session.execute(delete(KnowledgeEditRow).where(KnowledgeEditRow.author == "pytest"))
        session.execute(delete(DecisionRow).where(DecisionRow.query.like("pytest%")))
        session.execute(delete(GoldenCardRow).where(GoldenCardRow.key.like("pytest|%")))
        session.execute(delete(PositionRow).where(PositionRow.source == SOURCE))
        session.execute(delete(KnowledgeReleaseRow).where(KnowledgeReleaseRow.version.like("pytest-%")))
        session.execute(update(KnowledgeReleaseRow).values(is_active=KnowledgeReleaseRow.version == previous))
    db.close()


def _position(row: int, category: str | None) -> ParsedPosition:
    return ParsedPosition(row, f"строка {row}", 10.0, f"строка {row}", category, {}, ())


def test_new_release_becomes_active_and_is_stored_once(db: Database) -> None:
    repository = KnowledgeRepository(db)

    assert repository.save(KNOWLEDGE) == {"categories": 2, "category_attributes": 0}
    assert repository.active_version() == "pytest-1"
    repository.save(replace(KNOWLEDGE, version="pytest-2"))
    assert repository.save(KNOWLEDGE) is None
    assert repository.active_version() == "pytest-1"


def test_positions_upsert_keeps_ids_and_drops_vanished_rows(db: Database) -> None:
    KnowledgeRepository(db).save(KNOWLEDGE)
    repository = PositionRepository(db)

    repository.save(SOURCE, "pytest-1", [_position(2, "cable"), _position(3, None)])
    with db.session() as session:
        first = session.scalar(select(PositionRow.id).where(PositionRow.source == SOURCE, PositionRow.source_row == 2))

    counts = repository.save(SOURCE, "pytest-1", [_position(2, None)])

    assert counts == {"positions": 1, "categorized": 0, "removed": 1, "read_by_llm": 0}
    with db.session() as session:
        assert session.scalars(select(PositionRow.id).where(PositionRow.source == SOURCE)).all() == [first]


def test_model_readings_are_replaced_on_reload_and_reach_the_candidates(db: Database) -> None:
    KnowledgeRepository(db).save(KNOWLEDGE)
    repository = PositionRepository(db)
    read = replace(_position(2, "cable"), attributes={"сечение": 50.0, "марка": "АС"}, read_by_llm=("сечение", "марка"))

    assert repository.save(SOURCE, "pytest-1", [read, _position(3, "cable")])["read_by_llm"] == 2
    repository.save(SOURCE, "pytest-1", [replace(read, read_by_llm=("марка",)), _position(3, "cable")])

    candidates = {
        c.row: c
        for c in SearchRepository(db, "test-model").in_category("pytest-1", "cable", "строка")
        if c.row in (2, 3)
    }
    assert candidates[2].read_by_llm == {"марка"} and candidates[3].read_by_llm == frozenset()


def test_verdicts_link_rows_to_one_card_and_stay_in_the_log(db: Database) -> None:
    KnowledgeRepository(db).save(KNOWLEDGE)
    PositionRepository(db).save(SOURCE, "pytest-1", [_position(2, "cable"), _position(3, "cable")])
    cards = CardRepository(db)
    first, second = (cards.position_id(SOURCE, row) for row in (2, 3))
    assert first is not None and second is not None

    def entry(position_id: int, decision: Decision, note: str | None = None) -> Entry:
        return Entry("pytest|cable", "pytest кабель", "cable", {"число_жил": 3.0}, position_id, decision, "тест", note)

    cards.decide(entry(first, Decision.SAME))
    assert cards.decide(entry(second, Decision.UNCLEAR)) is None
    cards.decide(entry(second, Decision.ANALOG))
    card = cards.decide(entry(first, Decision.NOT_ANALOG, "не та марка"))
    anchored = cards.anchor(card.id, Reference("ksr", "01.1", "Кабель"))

    assert [(k.row, k.decision, k.note) for k in anchored.links] == [
        (2, Decision.NOT_ANALOG, "не та марка"),
        (3, Decision.ANALOG, None),
    ]
    assert cards.card("pytest|cable") == anchored and anchored.reference.code == "01.1"
    with db.session() as session:
        logged = session.scalars(select(DecisionRow.verdict).where(DecisionRow.query == "pytest кабель")).all()
    assert sorted(logged) == ["analog", "not_analog", "same", "unclear"]
    with pytest.raises(UnknownPosition):
        cards.decide(entry(-1, Decision.SAME))


def test_the_active_release_reads_back_as_it_was_stored(db: Database) -> None:
    research = yaml_files.read(PROJECT_ROOT / "research")
    repository = KnowledgeRepository(db)

    repository.save(replace(research, version="pytest-research"))
    active = repository.active()

    assert active is not None and active.version == "pytest-research"
    assert fingerprint(active.categories) == fingerprint(research.categories)


def test_edits_wait_in_the_draft_until_published_with_a_release(db: Database) -> None:
    repository = KnowledgeRepository(db)
    repository.save(KNOWLEDGE)
    edit = Edit("cable", None, Field.TRIGGERS, ["кабель", "провод"], ["кабель"], author="pytest")

    first, second = repository.add(edit), repository.add(replace(edit, value=["провод"]))
    assert repository.discard(second.id) and not repository.discard(second.id)
    assert [(e.id, e.value, e.previous) for e in repository.edits() if e.author == "pytest"] == [
        (first.id, ["кабель", "провод"], ["кабель"])
    ]

    repository.publish(replace(KNOWLEDGE, version="pytest-2"), "pytest")

    assert repository.active_version() == "pytest-2" and repository.edits() == []
    with db.session() as session:
        published = session.get_one(KnowledgeEditRow, first.id)
        assert session.get_one(KnowledgeReleaseRow, published.release_id).version == "pytest-2"


def test_stored_rows_come_back_with_their_category_and_readings(db: Database) -> None:
    KnowledgeRepository(db).save(KNOWLEDGE)
    read = replace(_position(2, "cable"), attributes={"марка": "АС"}, read_by_llm=("марка",))
    PositionRepository(db).save(SOURCE, "pytest-1", [read, _position(3, None)])

    assert PositionRepository(db).stored()[SOURCE] == [read, _position(3, None)]


def test_cards_meeting_on_a_key_are_merged_into_the_older(db: Database) -> None:
    KnowledgeRepository(db).save(KNOWLEDGE)
    PositionRepository(db).save(SOURCE, "pytest-1", [_position(2, "cable"), _position(3, "cable")])
    cards = CardRepository(db)
    first, second = (cards.position_id(SOURCE, row) for row in (2, 3))

    def entry(key: str, position_id: int, decision: Decision) -> Entry:
        return Entry(f"pytest|{key}", f"pytest {key}", "cable", {}, position_id, decision)

    older = cards.decide(entry("a", first, Decision.SAME))
    cards.decide(entry("b", first, Decision.NOT_ANALOG))
    newer = cards.decide(entry("b", second, Decision.ANALOG))
    cards.anchor(older.id, Reference("ksr", "01.1", "Кабель"))

    traded = {older.id: CardKey("pytest|b", "cable", {}), newer.id: CardKey("pytest|a", "cable", {})}
    assert cards.rekey(traded, "pytest-1") == {"rekeyed": 2, "merged": 0}
    assert cards.card("pytest|b").id == older.id and cards.card("pytest|b").reference is None

    assert cards.rekey({newer.id: CardKey("pytest|b", "cable", {})}) == {"rekeyed": 1, "merged": 1}
    merged = cards.card("pytest|b")
    assert merged.id == older.id and cards.card("pytest|a") is None
    assert [(k.row, k.decision) for k in merged.links] == [(2, Decision.NOT_ANALOG), (3, Decision.ANALOG)]
    with db.session() as session:
        logged = session.scalars(select(DecisionRow.card_id).where(DecisionRow.query.like("pytest %"))).all()
    assert set(logged) == {older.id}
