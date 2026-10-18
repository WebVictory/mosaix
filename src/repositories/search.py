"""Candidates for a query: rows of one category, and the nearest rows of the whole base by trigrams and by
embeddings (pgvector; no ANN index: a seq scan over a few thousand rows is faster)."""

from __future__ import annotations

from typing import Any

from sqlalchemy import Select, func, literal, not_, select

from domain.knowledge import SERVICE_FLAG
from domain.matching import Candidate
from repositories.database import Database
from repositories.orm import (
    CategoryRow,
    KnowledgeReleaseRow,
    PositionEmbeddingRow,
    PositionReadingRow,
    PositionRow,
)


class SearchRepository:
    def __init__(self, db: Database, embedding_model: str):
        self._db = db
        self._model = embedding_model

    def in_category(
        self, knowledge_version: str, category_code: str, text: str, vector: list[float] | None = None
    ) -> list[Candidate]:
        """Every goods row of the category, most similar first; a category is at most a few hundred rows."""
        statement = self._candidates(text, vector).where(
            CategoryRow.code == category_code, KnowledgeReleaseRow.version == knowledge_version
        )
        return self._fetch(statement)

    def nearest(self, text: str, limit: int, vector: list[float] | None = None) -> list[Candidate]:
        """Second pass: the closest goods rows of the whole base, whatever their category."""
        found = self._fetch(self._candidates(text, vector).limit(limit))
        if vector is not None:
            meaning = self._candidates(text, vector).where(PositionEmbeddingRow.embedding.is_not(None))
            by_meaning = meaning.order_by(None).order_by(
                PositionEmbeddingRow.embedding.cosine_distance(vector), PositionRow.source_row
            )
            found += self._fetch(by_meaning.limit(limit))
        return list({c.id: c for c in found}.values())

    def positions(self, ids: list[int], text: str, vector: list[float] | None = None) -> list[Candidate]:
        return self._fetch(self._candidates(text, vector).where(PositionRow.id.in_(ids)))

    def names(self) -> list[str]:
        with self._db.session() as session:
            return list(
                session.scalars(select(PositionRow.name).where(not_(PositionRow.flags.contains([SERVICE_FLAG]))))
            )

    def stats(self) -> dict[str, Any]:
        with self._db.session() as session:
            positions = session.execute(
                select(
                    func.count(),
                    func.count(PositionRow.category_id),
                    func.count().filter(PositionRow.flags.contains([SERVICE_FLAG])),
                    func.count().filter(PositionRow.attributes != {}),
                )
            ).one()
            read_by_llm = session.scalar(select(func.count(func.distinct(PositionReadingRow.position_id))))
            embedded = session.scalar(
                select(func.count()).select_from(PositionEmbeddingRow).where(PositionEmbeddingRow.model == self._model)
            )
            versions = session.execute(
                select(KnowledgeReleaseRow.version, func.count(PositionRow.id))
                .join(PositionRow, PositionRow.knowledge_release_id == KnowledgeReleaseRow.id)
                .group_by(KnowledgeReleaseRow.version)
            ).all()
        total, categorized, service, with_attributes = positions
        return {
            "positions": total,
            "categorized": categorized,
            "service": service,
            "with_attributes": with_attributes,
            "with_llm_readings": read_by_llm,
            "with_embeddings": embedded,
            "parsed_with": dict(versions),
        }

    def _fetch(self, statement: Select[Any]) -> list[Candidate]:
        with self._db.session() as session:
            return [
                Candidate(
                    id=row.id,
                    row=row.source_row,
                    name=row.name,
                    price=row.price,
                    category_code=row.code,
                    attributes=row.attributes,
                    similarity=round(float(row.similarity), 4),
                    meaning=None if row.meaning is None else round(float(row.meaning), 4),
                    read_by_llm=frozenset(row.read_by_llm or ()),
                )
                for row in session.execute(statement)
            ]

    def _candidates(self, text: str, vector: list[float] | None) -> Select[Any]:
        similarity = func.similarity(PositionRow.search_text, text).label("similarity")
        meaning = (1 - PositionEmbeddingRow.embedding.cosine_distance(vector) if vector else literal(None)).label(
            "meaning"
        )
        read_by_llm = (
            select(func.array_agg(PositionReadingRow.attribute_code))
            .where(PositionReadingRow.position_id == PositionRow.id)
            .scalar_subquery()
            .label("read_by_llm")
        )
        return (
            select(
                PositionRow.id,
                PositionRow.source_row,
                PositionRow.name,
                PositionRow.price,
                PositionRow.attributes,
                CategoryRow.code,
                similarity,
                meaning,
                read_by_llm,
            )
            .outerjoin(CategoryRow, CategoryRow.id == PositionRow.category_id)
            .outerjoin(
                PositionEmbeddingRow,
                (PositionEmbeddingRow.position_id == PositionRow.id) & (PositionEmbeddingRow.model == self._model),
            )
            .outerjoin(KnowledgeReleaseRow, KnowledgeReleaseRow.id == PositionRow.knowledge_release_id)
            .where(not_(PositionRow.flags.contains([SERVICE_FLAG])))
            .order_by(similarity.desc(), PositionRow.source_row)
        )
