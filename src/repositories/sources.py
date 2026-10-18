"""External taxonomies in PostgreSQL: one transaction per release.

Reloading a release with the same sha256 is a no-op; with a different one the old rows go
(cascade from source_releases) and the new ones replace them.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import asdict
from itertools import islice
from typing import Any

from sqlalchemy import delete, insert, select
from sqlalchemy.orm import Session

from domain.sources import EtimDump, KsrDump, KtruPosition, Okpd2Node, Release
from repositories.database import Database
from repositories.orm import (
    Base,
    EtimClassFeatureRow,
    EtimClassFeatureValueRow,
    EtimClassRow,
    EtimFeatureRow,
    EtimGroupRow,
    EtimUnitRow,
    EtimValueRow,
    KsrNodeRow,
    KsrPositionRow,
    KsrResourceRow,
    KtruCharacteristicRow,
    KtruPositionRow,
    Okpd2NodeRow,
    SourceReleaseRow,
)

CHUNK = 5000


def _chunks(rows: Iterable[dict[str, Any]]) -> Iterator[list[dict[str, Any]]]:
    iterator = iter(rows)
    while chunk := list(islice(iterator, CHUNK)):
        yield chunk


class SourceRepository:
    def __init__(self, db: Database):
        self._db = db

    def releases(self) -> list[SourceReleaseRow]:
        with self._db.session() as session:
            return list(
                session.scalars(select(SourceReleaseRow).order_by(SourceReleaseRow.source, SourceReleaseRow.loaded_at))
            )

    def is_loaded(self, release: Release) -> bool:
        with self._db.session() as session:
            row = session.scalar(
                select(SourceReleaseRow).where(
                    SourceReleaseRow.source == release.source.value, SourceReleaseRow.version == release.version
                )
            )
            return row is not None and row.sha256 == release.sha256

    def save_ksr(self, release: Release, dump: KsrDump) -> dict[str, int]:
        positions = [asdict(p) | {"resource_code": r.code} for r in dump.resources for p in r.positions]
        return self._save(
            release,
            {
                KsrNodeRow: (asdict(n) for n in dump.nodes),
                KsrResourceRow: ({k: v for k, v in asdict(r).items() if k != "positions"} for r in dump.resources),
                KsrPositionRow: positions,
            },
        )

    def save_okpd2(self, release: Release, nodes: list[Okpd2Node]) -> dict[str, int]:
        return self._save(release, {Okpd2NodeRow: (asdict(n) for n in nodes)})

    def save_ktru(self, release: Release, positions: list[KtruPosition]) -> dict[str, int]:
        characteristics = (
            asdict(c) | {"ktru_code": p.code, "ord": i} for p in positions for i, c in enumerate(p.characteristics)
        )
        return self._save(
            release,
            {
                KtruPositionRow: ({k: v for k, v in asdict(p).items() if k != "characteristics"} for p in positions),
                KtruCharacteristicRow: characteristics,
            },
        )

    def save_etim(self, release: Release, dump: EtimDump) -> dict[str, int]:
        return self._save(
            release,
            {
                EtimGroupRow: (asdict(g) for g in dump.groups),
                EtimClassRow: (asdict(c) for c in dump.classes),
                EtimFeatureRow: (asdict(f) for f in dump.features),
                EtimUnitRow: ({"code": k, "name": v} for k, v in dump.units.items()),
                EtimValueRow: ({"code": k, "name": v} for k, v in dump.values.items()),
                EtimClassFeatureRow: (asdict(f) for f in dump.class_features),
                EtimClassFeatureValueRow: (
                    {"class_feature_id": f, "value_code": v, "sort": s} for f, v, s in dump.class_feature_values
                ),
            },
        )

    def _save(self, release: Release, tables: dict[type[Base], Iterable[dict[str, Any]]]) -> dict[str, int]:
        with self._db.session() as session, session.begin():
            release_id = self._replace_release(session, release)
            counts = {}
            for model, rows in tables.items():
                counts[model.__tablename__] = 0
                for chunk in _chunks(rows):
                    session.execute(insert(model), [r | {"release_id": release_id} for r in chunk])
                    counts[model.__tablename__] += len(chunk)
            session.get_one(SourceReleaseRow, release_id).counts = counts
        return counts

    @staticmethod
    def _replace_release(session: Session, release: Release) -> int:
        session.execute(
            delete(SourceReleaseRow).where(
                SourceReleaseRow.source == release.source.value, SourceReleaseRow.version == release.version
            )
        )
        row = SourceReleaseRow(
            source=release.source.value,
            version=release.version,
            published=release.published,
            url=release.url,
            files=list(release.files),
            sha256=release.sha256,
        )
        session.add(row)
        session.flush()
        return row.id
