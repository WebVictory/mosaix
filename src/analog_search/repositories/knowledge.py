"""Knowledge releases and procurement positions in PostgreSQL."""

from __future__ import annotations

import hashlib
from typing import Any

from sqlalchemy import CursorResult, Result, delete, func, insert, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from analog_search.domain.editing import Edit, Field
from analog_search.domain.knowledge import (
    Attribute,
    AttributeStatus,
    Category,
    Comparison,
    Knowledge,
    Level,
    Provenance,
)
from analog_search.domain.positions import ParsedPosition
from analog_search.repositories.database import Database
from analog_search.repositories.orm import (
    CategoryAttributeRow,
    CategoryRow,
    KnowledgeEditRow,
    KnowledgeReleaseRow,
    PositionEmbeddingRow,
    PositionReadingRow,
    PositionRow,
)


class KnowledgeRepository:
    def __init__(self, db: Database):
        self._db = db

    def active_version(self) -> str | None:
        with self._db.session() as session:
            return session.scalar(select(KnowledgeReleaseRow.version).where(KnowledgeReleaseRow.is_active))

    def active(self) -> Knowledge | None:
        """The release the search works with, as it was stored; None on an empty base."""
        with self._db.session() as session:
            release = session.scalar(select(KnowledgeReleaseRow).where(KnowledgeReleaseRow.is_active))
            if release is None:
                return None
            rows = session.scalars(
                select(CategoryRow).where(CategoryRow.knowledge_release_id == release.id).order_by(CategoryRow.id)
            ).all()
            attributes: dict[int, list[Attribute]] = {}
            for row in session.scalars(
                select(CategoryAttributeRow)
                .where(CategoryAttributeRow.category_id.in_([r.id for r in rows]))
                .order_by(CategoryAttributeRow.id)
            ):
                attributes.setdefault(row.category_id, []).append(_attribute(row))
        names = {r.code: r.name for r in rows}
        categories = tuple(
            Category(
                code=r.code,
                name=r.name,
                group=names[r.parent_code] if r.parent_code else None,
                triggers=tuple(r.triggers),
                anti_triggers=tuple(r.anti_triggers),
                synonyms=tuple(r.synonyms),
                attributes=tuple(attributes.get(r.id, ())),
            )
            for r in rows
        )
        return Knowledge(version=release.version, sha256=release.sha256, categories=categories)

    def save(self, knowledge: Knowledge, note: str | None = None) -> dict[str, int] | None:
        """Store a release and make it active; None — this version is already stored (it is activated)."""
        with self._db.session() as session, session.begin():
            return self._save(session, knowledge, note)[1]

    def edits(self) -> list[Edit]:
        """The draft: edits not yet published, in the order they were made."""
        with self._db.session() as session:
            rows = session.scalars(
                select(KnowledgeEditRow).where(KnowledgeEditRow.release_id.is_(None)).order_by(KnowledgeEditRow.id)
            )
            return [_edit(row) for row in rows]

    def add(self, edit: Edit) -> Edit:
        with self._db.session() as session, session.begin():
            row = KnowledgeEditRow(
                category_code=edit.category,
                attribute_code=edit.attribute,
                field=edit.field,
                value=edit.value,
                previous=edit.previous,
                author=edit.author,
                reason=edit.reason,
            )
            session.add(row)
            session.flush()
            return _edit(row)

    def discard(self, edit_id: int) -> bool:
        """Drop an edit from the draft; False if it is not there (no such edit, or already published)."""
        with self._db.session() as session, session.begin():
            return bool(
                _affected(
                    session.execute(
                        delete(KnowledgeEditRow).where(
                            KnowledgeEditRow.id == edit_id, KnowledgeEditRow.release_id.is_(None)
                        )
                    )
                )
            )

    def publish(self, knowledge: Knowledge, note: str | None = None) -> dict[str, int] | None:
        """Store the draft's release, make it active and close the draft with it, in one transaction."""
        with self._db.session() as session, session.begin():
            release_id, counts = self._save(session, knowledge, note)
            session.execute(
                update(KnowledgeEditRow).where(KnowledgeEditRow.release_id.is_(None)).values(release_id=release_id)
            )
            return counts

    def _save(self, session: Session, knowledge: Knowledge, note: str | None) -> tuple[int, dict[str, int] | None]:
        existing = session.scalar(
            select(KnowledgeReleaseRow.id).where(KnowledgeReleaseRow.version == knowledge.version)
        )
        if existing is not None:
            self._activate(session, existing)
            return existing, None
        release_id = session.execute(
            insert(KnowledgeReleaseRow)
            .values(version=knowledge.version, sha256=knowledge.sha256, note=note)
            .returning(KnowledgeReleaseRow.id)
        ).scalar_one()
        ids = dict(
            session.execute(
                insert(CategoryRow).returning(CategoryRow.code, CategoryRow.id, sort_by_parameter_order=True),
                [_category(c, release_id) for c in knowledge.categories],
            ).all()
        )
        attributes = [
            {"category_id": ids[c.code], **a.model_dump(mode="json")}
            for c in knowledge.categories
            for a in c.attributes
        ]
        if attributes:
            session.execute(insert(CategoryAttributeRow), attributes)
        self._activate(session, release_id)
        return release_id, {"categories": len(ids), "category_attributes": len(attributes)}

    @staticmethod
    def _activate(session: Session, release_id: int) -> None:
        session.execute(update(KnowledgeReleaseRow).values(is_active=KnowledgeReleaseRow.id == release_id))


def _affected(result: Result[Any]) -> int:
    """Rows a DELETE or UPDATE touched: such a statement always gives a cursor result."""
    assert isinstance(result, CursorResult)
    return result.rowcount


def _category(category: Category, release_id: int) -> dict[str, Any]:
    return {
        "knowledge_release_id": release_id,
        "code": category.code,
        "name": category.name,
        "parent_code": category.group_code,
        "triggers": list(category.triggers),
        "anti_triggers": list(category.anti_triggers),
        "synonyms": list(category.synonyms),
    }


def _attribute(row: CategoryAttributeRow) -> Attribute:
    return Attribute(
        code=row.code,
        name=row.name,
        unit=row.unit,
        level=Level(row.level),
        comparison=Comparison(row.comparison),
        provenance=Provenance(row.provenance),
        status=AttributeStatus(row.status),
        source_ref=row.source_ref,
        rule=row.rule,
    )


def _edit(row: KnowledgeEditRow) -> Edit:
    return Edit(
        category=row.category_code,
        attribute=row.attribute_code,
        field=Field(row.field),
        value=row.value,
        previous=row.previous,
        author=row.author,
        reason=row.reason,
        id=row.id,
        created_at=row.created_at,
    )


class PositionRepository:
    def __init__(self, db: Database):
        self._db = db

    def save(self, source: str, knowledge_version: str, positions: list[ParsedPosition]) -> dict[str, int]:
        """Upsert by (source, row): ids stay stable, so decisions and card links survive a re-parse."""
        with self._db.session() as session, session.begin():
            release_id, categories = self._categories(session, knowledge_version)
            rows = [
                {
                    "source": source,
                    "source_row": p.source_row,
                    "name": p.name,
                    "price": p.price,
                    "search_text": p.search_text,
                    "knowledge_release_id": release_id,
                    "category_id": categories.get(p.category_code) if p.category_code else None,
                    "attributes": p.attributes,
                    "flags": list(p.flags),
                }
                for p in positions
            ]
            statement = pg_insert(PositionRow)
            session.execute(
                statement.on_conflict_do_update(
                    index_elements=[PositionRow.source, PositionRow.source_row],
                    set_={k: statement.excluded[k] for k in rows[0] if k not in ("source", "source_row")},
                ),
                rows,
            )
            ids = dict(
                session.execute(
                    select(PositionRow.source_row, PositionRow.id).where(PositionRow.source == source)
                ).all()
            )
            session.execute(delete(PositionReadingRow).where(PositionReadingRow.position_id.in_(ids.values())))
            readings = [
                {"position_id": ids[p.source_row], "attribute_code": code} for p in positions for code in p.read_by_llm
            ]
            if readings:
                session.execute(insert(PositionReadingRow), readings)
            stale = _affected(
                session.execute(
                    delete(PositionRow).where(
                        PositionRow.source == source, PositionRow.source_row.not_in([p.source_row for p in positions])
                    )
                )
            )
            categorized = session.execute(
                select(func.count()).where(PositionRow.source == source, PositionRow.category_id.is_not(None))
            ).scalar_one()
        return {"positions": len(rows), "categorized": categorized, "removed": stale, "read_by_llm": len(readings)}

    def stored(self) -> dict[str, list[ParsedPosition]]:
        """The rows of every source as they were parsed: to parse them again with another release."""
        with self._db.session() as session:
            readings: dict[int, list[str]] = {}
            for position_id, code in session.execute(
                select(PositionReadingRow.position_id, PositionReadingRow.attribute_code)
            ):
                readings.setdefault(position_id, []).append(code)
            rows = session.execute(
                select(PositionRow, CategoryRow.code)
                .outerjoin(CategoryRow, CategoryRow.id == PositionRow.category_id)
                .order_by(PositionRow.source, PositionRow.source_row)
            ).all()
        stored: dict[str, list[ParsedPosition]] = {}
        for row, code in rows:
            stored.setdefault(row.source, []).append(
                ParsedPosition(
                    source_row=row.source_row,
                    name=row.name,
                    price=row.price,
                    search_text=row.search_text,
                    category_code=code,
                    attributes=row.attributes,
                    flags=tuple(row.flags),
                    read_by_llm=tuple(sorted(readings.get(row.id, ()))),
                )
            )
        return stored

    def save_embeddings(self, source: str, model: str, vectors: dict[int, list[float]]) -> int:
        """Vectors of the names by source row, for one model; text_hash says which name each was computed from."""
        with self._db.session() as session, session.begin():
            rows = session.execute(
                select(PositionRow.source_row, PositionRow.id, PositionRow.name).where(PositionRow.source == source)
            ).all()
            values = [
                {
                    "position_id": id_,
                    "model": model,
                    "text_hash": hashlib.sha256(name.encode()).hexdigest()[:24],
                    "embedding": vectors[row],
                }
                for row, id_, name in rows
                if row in vectors
            ]
            session.execute(delete(PositionEmbeddingRow).where(PositionEmbeddingRow.model == model))
            if values:
                session.execute(insert(PositionEmbeddingRow), values)
        return len(values)

    @staticmethod
    def _categories(session: Session, version: str) -> tuple[int, dict[str, int]]:
        release_id = session.scalar(select(KnowledgeReleaseRow.id).where(KnowledgeReleaseRow.version == version))
        if release_id is None:
            raise LookupError(f"knowledge release {version} is not stored")
        codes = session.execute(
            select(CategoryRow.code, CategoryRow.id).where(CategoryRow.knowledge_release_id == release_id)
        )
        return release_id, dict(codes.all())
