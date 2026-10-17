"""SourceRepository against PostgreSQL: docker compose up -d db; pytest -m integration."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import delete, func, select

from analog_search.app.config import Settings
from analog_search.domain.sources import KtruCharacteristic, KtruPosition, Okpd2Node, Release, SourceKind
from analog_search.repositories.database import Database
from analog_search.repositories.orm import KtruCharacteristicRow, Okpd2NodeRow, SourceReleaseRow
from analog_search.repositories.sources import SourceRepository

pytestmark = pytest.mark.integration

VERSION = "pytest"


@pytest.fixture
def db() -> Iterator[Database]:
    db = Database(Settings().database_url)
    db.init_schema()
    yield db
    with db.session() as session, session.begin():
        session.execute(delete(SourceReleaseRow).where(SourceReleaseRow.version == VERSION))
    db.close()


def _release(source: SourceKind, sha256: str) -> Release:
    return Release(source, VERSION, None, None, ("file",), sha256)


def _count(db: Database, model: type, release_id: int) -> int:
    with db.session() as session:
        return session.scalar(select(func.count()).select_from(model).where(model.release_id == release_id))


def _release_id(db: Database, source: SourceKind) -> int:
    with db.session() as session:
        return session.scalar(
            select(SourceReleaseRow.id).where(
                SourceReleaseRow.source == source.value, SourceReleaseRow.version == VERSION
            )
        )


def test_reload_replaces_rows_of_the_same_version(db: Database) -> None:
    repository = SourceRepository(db)
    nodes = [Okpd2Node("C", None, 0, "ПРОДУКЦИЯ", None), Okpd2Node("27", "C", 1, "Оборудование", None)]

    assert repository.save_okpd2(_release(SourceKind.OKPD2, "a"), nodes) == {"okpd2_nodes": 2}
    assert repository.is_loaded(_release(SourceKind.OKPD2, "a"))
    assert not repository.is_loaded(_release(SourceKind.OKPD2, "b"))

    repository.save_okpd2(_release(SourceKind.OKPD2, "b"), nodes[:1])
    release_id = _release_id(db, SourceKind.OKPD2)
    assert _count(db, Okpd2NodeRow, release_id) == 1
    assert next(r.counts for r in repository.releases() if r.id == release_id) == {"okpd2_nodes": 1}


def test_ktru_characteristics_keep_order(db: Database) -> None:
    characteristic = KtruCharacteristic("Сечение", "quantitative", False, (">=0.5",), ("мм2",))
    position = KtruPosition(
        "27.32.13.143-00000001", 3, "Кабель", "27.32.13.143", "ACTIVE", ("Метр",), (characteristic, characteristic)
    )

    counts = SourceRepository(db).save_ktru(_release(SourceKind.KTRU, "a"), [position])

    assert counts == {"ktru_positions": 1, "ktru_characteristics": 2}
    with db.session() as session:
        ords = session.scalars(
            select(KtruCharacteristicRow.ord).where(
                KtruCharacteristicRow.release_id == _release_id(db, SourceKind.KTRU)
            )
        )
        assert sorted(ords) == [0, 1]
