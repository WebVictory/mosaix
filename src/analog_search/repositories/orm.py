"""SQLAlchemy 2.0 ORM models — the single source of the DB schema (ARCHITECTURE.md, «Хранилище»).

Three parts:
- sources: external taxonomies (КСР, ОКПД2, КТРУ, ETIM), each row tied to a `source_releases` row;
- knowledge: our categories, attribute schemas and their mapping to the sources, versioned by
  `knowledge_releases` — a search result names the knowledge version it was made with;
- positions: procurement history with parsed attributes, vectors, golden cards and the decision log.

The schema is created by `Base.metadata.create_all` at startup; no Alembic in the prototype.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    MetaData,
    Numeric,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from analog_search.domain.cards import LINKS
from analog_search.domain.editing import Field
from analog_search.domain.knowledge import AttributeStatus, Comparison, Level, Provenance
from analog_search.domain.matching import Decision
from analog_search.domain.sources import SourceKind

SCHEMA = "analog"
EMBEDDING_DIM = 1024  # baai/bge-m3 (Settings.embedding_model)

SOURCES = tuple(SourceKind)
LEVELS = tuple(Level)
COMPARISONS = tuple(Comparison)
PROVENANCES = tuple(Provenance)
ATTRIBUTE_STATUSES = tuple(AttributeStatus)
MAPPING_RELATIONS = ("exact", "broader", "narrower", "related")
DECISIONS = tuple(Decision)
EDIT_FIELDS = tuple(Field)

TextArray = ARRAY(Text)


def _in(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(str(v)) for v in values)})"


def _trgm(name: str, column: str) -> Index:
    return Index(name, column, postgresql_using="gin", postgresql_ops={column: "gin_trgm_ops"})


class Base(DeclarativeBase):
    metadata = MetaData(schema=SCHEMA)


def _release_fk() -> Mapped[int]:
    return mapped_column(ForeignKey("source_releases.id", ondelete="CASCADE"), primary_key=True)


# --- Sources ---


class SourceReleaseRow(Base):
    """One loaded version of an external source; deleting it removes all its rows."""

    __tablename__ = "source_releases"
    __table_args__ = (
        UniqueConstraint("source", "version"),
        CheckConstraint(_in("source", SOURCES), name="source_releases_source_check"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(Text)
    version: Mapped[str] = mapped_column(Text)
    published: Mapped[date | None] = mapped_column(Date)
    url: Mapped[str | None] = mapped_column(Text)
    files: Mapped[list[str]] = mapped_column(TextArray, server_default="{}")
    sha256: Mapped[str] = mapped_column(Text)
    counts: Mapped[dict[str, int]] = mapped_column(JSONB, server_default="{}")
    loaded_at: Mapped[datetime] = mapped_column(server_default=func.now())


class KsrNodeRow(Base):
    """Book / part / section / group of the КСР tree."""

    __tablename__ = "ksr_nodes"

    release_id: Mapped[int] = _release_fk()
    code: Mapped[str] = mapped_column(Text, primary_key=True)
    parent_code: Mapped[str | None] = mapped_column(Text)
    level: Mapped[str] = mapped_column(Text)
    name: Mapped[str] = mapped_column(Text)
    sheet: Mapped[str] = mapped_column(Text)


class KsrResourceRow(Base):
    __tablename__ = "ksr_resources"
    __table_args__ = (
        Index("idx_ksr_resources_group", "release_id", "group_code"),
        _trgm("idx_ksr_resources_trgm", "name"),
    )

    release_id: Mapped[int] = _release_fk()
    code: Mapped[str] = mapped_column(Text, primary_key=True)
    group_code: Mapped[str | None] = mapped_column(Text)
    name: Mapped[str] = mapped_column(Text)
    unit: Mapped[str | None] = mapped_column(Text)


class KsrPositionRow(Base):
    """Position-level code: the bridge КСР ↔ ОКПД2."""

    __tablename__ = "ksr_positions"
    __table_args__ = (Index("idx_ksr_positions_okpd2", "release_id", "okpd2_code"),)

    release_id: Mapped[int] = _release_fk()
    code: Mapped[str] = mapped_column(Text, primary_key=True)
    resource_code: Mapped[str] = mapped_column(Text)
    okpd2_code: Mapped[str] = mapped_column(Text)
    name: Mapped[str] = mapped_column(Text)
    unit: Mapped[str | None] = mapped_column(Text)


class Okpd2NodeRow(Base):
    __tablename__ = "okpd2_nodes"
    __table_args__ = (Index("idx_okpd2_parent", "release_id", "parent_code"),)

    release_id: Mapped[int] = _release_fk()
    code: Mapped[str] = mapped_column(Text, primary_key=True)
    parent_code: Mapped[str | None] = mapped_column(Text)
    level: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)


class KtruPositionRow(Base):
    __tablename__ = "ktru_positions"
    __table_args__ = (Index("idx_ktru_okpd2", "release_id", "okpd2_code"), _trgm("idx_ktru_name_trgm", "name"))

    release_id: Mapped[int] = _release_fk()
    code: Mapped[str] = mapped_column(Text, primary_key=True)
    version: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(Text)
    okpd2_code: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text)
    units: Mapped[list[str]] = mapped_column(TextArray, server_default="{}")


class KtruCharacteristicRow(Base):
    """Characteristic of a КТРУ position: a draft attribute schema for categories ETIM does not cover."""

    __tablename__ = "ktru_characteristics"
    __table_args__ = (
        ForeignKeyConstraint(
            ["release_id", "ktru_code"],
            ["ktru_positions.release_id", "ktru_positions.code"],
            ondelete="CASCADE",
        ),
    )

    release_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ktru_code: Mapped[str] = mapped_column(Text, primary_key=True)
    ord: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(Text)  # qualitative | quantitative
    required: Mapped[bool] = mapped_column(Boolean)
    values: Mapped[list[str]] = mapped_column(TextArray, server_default="{}")
    units: Mapped[list[str]] = mapped_column(TextArray, server_default="{}")


class EtimGroupRow(Base):
    __tablename__ = "etim_groups"

    release_id: Mapped[int] = _release_fk()
    code: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str] = mapped_column(Text)


class EtimClassRow(Base):
    __tablename__ = "etim_classes"

    release_id: Mapped[int] = _release_fk()
    code: Mapped[str] = mapped_column(Text, primary_key=True)
    group_code: Mapped[str] = mapped_column(Text)
    name: Mapped[str] = mapped_column(Text)
    version: Mapped[int] = mapped_column(Integer)
    synonyms: Mapped[list[str]] = mapped_column(TextArray, server_default="{}")


class EtimFeatureRow(Base):
    __tablename__ = "etim_features"

    release_id: Mapped[int] = _release_fk()
    code: Mapped[str] = mapped_column(Text, primary_key=True)
    group_code: Mapped[str | None] = mapped_column(Text)
    name: Mapped[str] = mapped_column(Text)


class EtimUnitRow(Base):
    __tablename__ = "etim_units"

    release_id: Mapped[int] = _release_fk()
    code: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str] = mapped_column(Text)


class EtimValueRow(Base):
    __tablename__ = "etim_values"

    release_id: Mapped[int] = _release_fk()
    code: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str] = mapped_column(Text)


class EtimClassFeatureRow(Base):
    """Feature of a class; type A alphanumeric (value list), N numeric, R range, L logical."""

    __tablename__ = "etim_class_features"
    __table_args__ = (
        Index("idx_etim_class_features_class", "release_id", "class_code"),
        CheckConstraint(_in("type", ("A", "N", "R", "L")), name="etim_class_features_type_check"),
    )

    release_id: Mapped[int] = _release_fk()
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    class_code: Mapped[str] = mapped_column(Text)
    feature_code: Mapped[str] = mapped_column(Text)
    type: Mapped[str] = mapped_column(Text)
    unit_code: Mapped[str | None] = mapped_column(Text)
    sort: Mapped[int] = mapped_column(Integer)


class EtimClassFeatureValueRow(Base):
    __tablename__ = "etim_class_feature_values"

    release_id: Mapped[int] = _release_fk()
    class_feature_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    value_code: Mapped[str] = mapped_column(Text, primary_key=True)
    sort: Mapped[int] = mapped_column(Integer)


# --- Knowledge: our taxonomy, versioned ---


class KnowledgeReleaseRow(Base):
    """Snapshot of categories / attributes / rules; search results reference the one they used."""

    __tablename__ = "knowledge_releases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[str] = mapped_column(Text, unique=True)
    sha256: Mapped[str] = mapped_column(Text)  # over the source yaml files
    note: Mapped[str | None] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, server_default="false")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class CategoryRow(Base):
    __tablename__ = "categories"
    __table_args__ = (UniqueConstraint("knowledge_release_id", "code"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    knowledge_release_id: Mapped[int] = mapped_column(ForeignKey("knowledge_releases.id", ondelete="CASCADE"))
    code: Mapped[str] = mapped_column(Text)  # cable_control
    name: Mapped[str] = mapped_column(Text)
    parent_code: Mapped[str | None] = mapped_column(Text)
    triggers: Mapped[list[str]] = mapped_column(TextArray, server_default="{}")
    anti_triggers: Mapped[list[str]] = mapped_column(TextArray, server_default="{}")
    synonyms: Mapped[list[str]] = mapped_column(TextArray, server_default="{}")


class CategoryAttributeRow(Base):
    """Attribute of a category's schema with its substitution rule and where the rule came from."""

    __tablename__ = "category_attributes"
    __table_args__ = (
        UniqueConstraint("category_id", "code"),
        CheckConstraint(_in("level", LEVELS), name="category_attributes_level_check"),
        CheckConstraint(_in("comparison", COMPARISONS), name="category_attributes_comparison_check"),
        CheckConstraint(_in("provenance", PROVENANCES), name="category_attributes_provenance_check"),
        CheckConstraint(_in("status", ATTRIBUTE_STATUSES), name="category_attributes_status_check"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    category_id: Mapped[int] = mapped_column(ForeignKey("categories.id", ondelete="CASCADE"))
    code: Mapped[str] = mapped_column(Text)  # cores, cross_section
    name: Mapped[str] = mapped_column(Text)
    quantity: Mapped[str | None] = mapped_column(Text)  # units.yaml quantity
    unit: Mapped[str | None] = mapped_column(Text)
    level: Mapped[str] = mapped_column(Text)
    comparison: Mapped[str] = mapped_column(Text)
    # step size, allowed direction («FRLS вместо LS»), value series
    rule: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default="{}")
    provenance: Mapped[str] = mapped_column(Text)
    source_ref: Mapped[str | None] = mapped_column(Text)  # ГОСТ, ETIM EF…, КТРУ characteristic, URL
    status: Mapped[str] = mapped_column(Text, server_default="proposed")


class CategoryMappingRow(Base):
    """Our category ↔ a code in an external source, against a specific release of it."""

    __tablename__ = "category_mappings"
    __table_args__ = (
        CheckConstraint(_in("relation", MAPPING_RELATIONS), name="category_mappings_relation_check"),
        Index("idx_category_mappings_external", "source_release_id", "external_code"),
    )

    category_id: Mapped[int] = mapped_column(ForeignKey("categories.id", ondelete="CASCADE"), primary_key=True)
    source_release_id: Mapped[int] = mapped_column(
        ForeignKey("source_releases.id", ondelete="CASCADE"), primary_key=True
    )
    external_code: Mapped[str] = mapped_column(Text, primary_key=True)  # 21.1.06, 27.32.13.143, EC000034
    relation: Mapped[str] = mapped_column(Text)
    confirmed_by: Mapped[str | None] = mapped_column(Text)  # None — proposed, not checked
    confirmed_at: Mapped[datetime | None]


class KnowledgeEditRow(Base):
    """A specialist's edit of the knowledge (domain/editing.py). release_id None — in the draft over the active
    release; set — published with that release. Kept after publishing: who changed which rule, when, and why."""

    __tablename__ = "knowledge_edits"
    __table_args__ = (CheckConstraint(_in("field", EDIT_FIELDS), name="knowledge_edits_field_check"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    category_code: Mapped[str] = mapped_column(Text)
    attribute_code: Mapped[str | None] = mapped_column(Text)  # None — a field of the category
    field: Mapped[str] = mapped_column(Text)
    value: Mapped[Any] = mapped_column(JSONB)
    previous: Mapped[Any] = mapped_column(JSONB)
    author: Mapped[str | None] = mapped_column(Text)
    reason: Mapped[str | None] = mapped_column(Text)
    release_id: Mapped[int | None] = mapped_column(ForeignKey("knowledge_releases.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


# --- Positions, cards, decisions ---


class PositionRow(Base):
    """Procurement history row with what the parser made of it."""

    __tablename__ = "positions"
    __table_args__ = (
        UniqueConstraint("source", "source_row"),
        Index("idx_positions_category", "category_id"),
        _trgm("idx_positions_search_trgm", "search_text"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(Text)  # positions.xlsx, a customer upload
    source_row: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(Text)
    price: Mapped[float | None] = mapped_column(Numeric(14, 2, asdecimal=False))
    search_text: Mapped[str] = mapped_column(Text)  # normalized name for pg_trgm
    knowledge_release_id: Mapped[int | None] = mapped_column(ForeignKey("knowledge_releases.id", ondelete="SET NULL"))
    category_id: Mapped[int | None] = mapped_column(ForeignKey("categories.id", ondelete="SET NULL"))
    attributes: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default="{}")
    flags: Mapped[list[str]] = mapped_column(TextArray, server_default="{}")  # conflict, composite, service
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class PositionReadingRow(Base):
    """An attribute of positions.attributes the choice model read, not the rules: the UI marks it «ИИ».

    Which model and how sure: data/llm/readings.jsonl, the answers the reading was made from.
    """

    __tablename__ = "position_readings"

    position_id: Mapped[int] = mapped_column(ForeignKey("positions.id", ondelete="CASCADE"), primary_key=True)
    attribute_code: Mapped[str] = mapped_column(Text, primary_key=True)


class PositionEmbeddingRow(Base):
    """Vector of a position for one model; text_hash says which text it was computed from.

    No ANN index: below ~10k rows a seq scan is faster; beyond that add HNSW.
    """

    __tablename__ = "position_embeddings"

    position_id: Mapped[int] = mapped_column(ForeignKey("positions.id", ondelete="CASCADE"), primary_key=True)
    model: Mapped[str] = mapped_column(Text, primary_key=True)
    text_hash: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))


class GoldenCardRow(Base):
    """A typical position specialists confirmed rows for (domain/cards.py).

    key — category and the values read from a query, «cable_control|{"cores": 7, …}»; a query nothing was read
    from is keyed by its text. It is made with one knowledge release: a new release may read the query
    differently, then the card is re-keyed, and cards that now read the same are merged (docs/steps.md, step 11).
    reference — the ideal from a reference book (КСР resource), if one reads into the same values.
    """

    __tablename__ = "golden_cards"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(Text, unique=True)
    name: Mapped[str] = mapped_column(Text)  # the query that made it
    category_code: Mapped[str | None] = mapped_column(Text)
    attributes: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default="{}")
    knowledge_release_id: Mapped[int | None] = mapped_column(ForeignKey("knowledge_releases.id", ondelete="SET NULL"))
    reference_source: Mapped[str | None] = mapped_column(Text)
    reference_code: Mapped[str | None] = mapped_column(Text)
    reference_name: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class CardLinkRow(Base):
    """The latest verdict on a card and a row: the same position, an analog of it, never to be offered."""

    __tablename__ = "card_links"
    __table_args__ = (
        CheckConstraint(_in("relation", LINKS), name="card_links_relation_check"),
        Index("idx_card_links_position", "position_id"),
    )

    card_id: Mapped[int] = mapped_column(ForeignKey("golden_cards.id", ondelete="CASCADE"), primary_key=True)
    position_id: Mapped[int] = mapped_column(ForeignKey("positions.id", ondelete="CASCADE"), primary_key=True)
    relation: Mapped[str] = mapped_column(Text)
    decision_id: Mapped[int | None] = mapped_column(ForeignKey("decisions.id", ondelete="SET NULL"))


class DecisionRow(Base):
    """A specialist's verdict on a (query, row) pair, append-only: the cards are rebuilt from it, and it is
    the labelled data the rules and the model are measured on."""

    __tablename__ = "decisions"
    __table_args__ = (
        CheckConstraint(_in("verdict", DECISIONS), name="decisions_verdict_check"),
        Index("idx_decisions_position", "position_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    query: Mapped[str] = mapped_column(Text)
    query_attributes: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default="{}")
    position_id: Mapped[int | None] = mapped_column(ForeignKey("positions.id", ondelete="SET NULL"))
    card_id: Mapped[int | None] = mapped_column(ForeignKey("golden_cards.id", ondelete="SET NULL"))
    verdict: Mapped[str] = mapped_column(Text)
    reason: Mapped[str | None] = mapped_column(Text)
    author: Mapped[str | None] = mapped_column(Text)
    knowledge_release_id: Mapped[int | None] = mapped_column(ForeignKey("knowledge_releases.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
