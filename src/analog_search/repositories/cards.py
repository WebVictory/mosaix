"""Golden cards, their links to rows of the history and the decision log (domain/cards.py); КСР as the
reference book of ideal positions."""

from __future__ import annotations

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from analog_search.domain.cards import Card, CardKey, Entry, Link, Reference, UnknownPosition
from analog_search.domain.matching import Decision
from analog_search.domain.sources import SourceKind
from analog_search.repositories.database import Database
from analog_search.repositories.orm import (
    CardLinkRow,
    DecisionRow,
    GoldenCardRow,
    KnowledgeReleaseRow,
    KsrResourceRow,
    PositionRow,
    SourceReleaseRow,
)


class CardRepository:
    def __init__(self, db: Database):
        self._db = db

    def card(self, key: str) -> Card | None:
        with self._db.session() as session:
            row = session.scalar(select(GoldenCardRow).where(GoldenCardRow.key == key))
            return self._card(session, row) if row else None

    def decide(self, entry: Entry) -> Card | None:
        with self._db.session() as session, session.begin():
            if session.get(PositionRow, entry.position_id) is None:
                raise UnknownPosition(entry.position_id)
            release_id = session.scalar(
                select(KnowledgeReleaseRow.id).where(KnowledgeReleaseRow.version == entry.knowledge_version)
            )
            card = None if entry.decision is Decision.UNCLEAR else self._card_of(session, entry, release_id)
            decision = DecisionRow(
                query=entry.query,
                query_attributes=entry.values,
                position_id=entry.position_id,
                card_id=card.id if card else None,
                verdict=entry.decision,
                reason=entry.note,
                author=entry.author,
                knowledge_release_id=release_id,
            )
            session.add(decision)
            session.flush()
            if card is None:
                return None
            link = {"card_id": card.id, "position_id": entry.position_id}
            statement = pg_insert(CardLinkRow).values(**link, relation=entry.decision, decision_id=decision.id)
            session.execute(
                statement.on_conflict_do_update(
                    index_elements=list(link),
                    set_={"relation": statement.excluded.relation, "decision_id": statement.excluded.decision_id},
                )
            )
            return self._card(session, card)

    def anchor(self, card_id: int, reference: Reference) -> Card:
        with self._db.session() as session, session.begin():
            row = session.get_one(GoldenCardRow, card_id)
            row.reference_source, row.reference_code, row.reference_name = (
                reference.source,
                reference.code,
                reference.name,
            )
            return self._card(session, row)

    def cards(self) -> list[Card]:
        with self._db.session() as session:
            return [
                self._card(session, row) for row in session.scalars(select(GoldenCardRow).order_by(GoldenCardRow.id))
            ]

    def rekey(self, keys: dict[int, CardKey], knowledge_version: str | None = None) -> dict[str, int]:
        with self._db.session() as session, session.begin():
            rows = session.scalars(select(GoldenCardRow).order_by(GoldenCardRow.id)).all()
            groups: dict[str, list[GoldenCardRow]] = {}
            for row in rows:
                groups.setdefault(keys[row.id].key if row.id in keys else row.key, []).append(row)
            merged = 0
            for survivor, *others in groups.values():
                for other in others:
                    self._merge(session, survivor.id, other.id)
                    merged += 1
            moved = [group[0] for key, group in groups.items() if group[0].key != key]
            # free the keys first: two cards may trade theirs
            for row in moved:
                session.execute(update(GoldenCardRow).where(GoldenCardRow.id == row.id).values(key=f"#{row.id}"))
            release_id = session.scalar(
                select(KnowledgeReleaseRow.id).where(KnowledgeReleaseRow.version == knowledge_version)
            )
            for row in moved:
                new = keys[row.id]
                session.execute(
                    update(GoldenCardRow)
                    .where(GoldenCardRow.id == row.id)
                    .values(
                        key=new.key,
                        category_code=new.category_code,
                        attributes=new.values,
                        knowledge_release_id=release_id,
                        reference_source=None,
                        reference_code=None,
                        reference_name=None,
                    )
                )
        return {"rekeyed": len(keys), "merged": merged}

    def position_id(self, source: str, row: int) -> int | None:
        with self._db.session() as session:
            return session.scalar(
                select(PositionRow.id).where(PositionRow.source == source, PositionRow.source_row == row)
            )

    @staticmethod
    def _merge(session: Session, into: int, merged: int) -> None:
        """Links and log of one card moved to another; a row linked by both keeps the later verdict."""
        ours = dict(
            session.execute(
                select(CardLinkRow.position_id, CardLinkRow.decision_id).where(CardLinkRow.card_id == into)
            ).all()
        )
        theirs = session.execute(
            select(CardLinkRow.position_id, CardLinkRow.relation, CardLinkRow.decision_id).where(
                CardLinkRow.card_id == merged
            )
        ).all()
        for position_id, relation, decision_id in theirs:
            if position_id not in ours:
                session.execute(
                    pg_insert(CardLinkRow).values(
                        card_id=into, position_id=position_id, relation=relation, decision_id=decision_id
                    )
                )
            elif (decision_id or 0) > (ours[position_id] or 0):
                session.execute(
                    update(CardLinkRow)
                    .where(CardLinkRow.card_id == into, CardLinkRow.position_id == position_id)
                    .values(relation=relation, decision_id=decision_id)
                )
        session.execute(update(DecisionRow).where(DecisionRow.card_id == merged).values(card_id=into))
        session.execute(delete(GoldenCardRow).where(GoldenCardRow.id == merged))  # its links go with it

    @staticmethod
    def _card_of(session: Session, entry: Entry, release_id: int | None) -> GoldenCardRow:
        """The card of the key, created on first use; concurrent creators meet on the unique key."""
        existing = session.scalar(select(GoldenCardRow).where(GoldenCardRow.key == entry.key))
        if existing is not None:
            return existing
        statement = pg_insert(GoldenCardRow).values(
            key=entry.key,
            name=entry.query,
            category_code=entry.category_code,
            attributes=entry.values,
            knowledge_release_id=release_id,
        )
        session.execute(statement.on_conflict_do_nothing(index_elements=[GoldenCardRow.key]))
        return session.scalars(select(GoldenCardRow).where(GoldenCardRow.key == entry.key)).one()

    @staticmethod
    def _card(session: Session, row: GoldenCardRow) -> Card:
        links = session.execute(
            select(
                CardLinkRow.position_id,
                PositionRow.source_row,
                PositionRow.name,
                CardLinkRow.relation,
                DecisionRow.reason,
            )
            .join(PositionRow, PositionRow.id == CardLinkRow.position_id)
            .outerjoin(DecisionRow, DecisionRow.id == CardLinkRow.decision_id)
            .where(CardLinkRow.card_id == row.id)
            .order_by(PositionRow.source_row)
        ).all()
        reference = (
            Reference(source=row.reference_source, code=row.reference_code, name=row.reference_name or "")
            if row.reference_source and row.reference_code
            else None
        )
        return Card(
            id=row.id,
            key=row.key,
            name=row.name,
            category_code=row.category_code,
            links=tuple(
                Link(position_id=pid, row=source_row, name=name, decision=Decision(relation), note=note)
                for pid, source_row, name, relation, note in links
            ),
            reference=reference,
        )


class KsrReferences:
    """КСР resources of the latest loaded release, by trigram similarity of the names."""

    def __init__(self, db: Database):
        self._db = db

    def nearest(self, text: str, limit: int) -> list[Reference]:
        with self._db.session() as session:
            release = (
                select(SourceReleaseRow.id)
                .where(SourceReleaseRow.source == SourceKind.KSR)
                .order_by(SourceReleaseRow.id.desc())
                .limit(1)
                .scalar_subquery()
            )
            similarity = func.similarity(KsrResourceRow.name, text)  # pg_trgm ignores case
            rows = session.execute(
                select(KsrResourceRow.code, KsrResourceRow.name)
                .where(KsrResourceRow.release_id == release, KsrResourceRow.name.op("%")(text))
                .order_by(similarity.desc(), KsrResourceRow.code)
                .limit(limit)
            ).all()
        return [Reference(source=SourceKind.KSR, code=code, name=name) for code, name in rows]
