"""PostgreSQL connection: SQLAlchemy 2.0 engine (psycopg 3) and schema bootstrap."""

from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from repositories.orm import SCHEMA, Base

# Arbitrary constant key of the advisory lock that serializes schema creation
SCHEMA_LOCK_KEY = 7312


def sqlalchemy_url(database_url: str) -> str:
    """postgresql://… → postgresql+psycopg://… (psycopg 3 driver)."""
    url = make_url(database_url)
    if url.drivername in {"postgresql", "postgres"}:
        url = url.set(drivername="postgresql+psycopg")
    return url.render_as_string(hide_password=False)


class Database:
    def __init__(self, database_url: str, pool_size: int = 4, connect_timeout_s: int = 10):
        self.engine = create_engine(
            sqlalchemy_url(database_url),
            pool_size=pool_size,
            pool_pre_ping=True,
            connect_args={"connect_timeout": connect_timeout_s},
        )
        self._sessions = sessionmaker(self.engine, expire_on_commit=False)

    def session(self) -> Session:
        return self._sessions()

    def init_schema(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(text(f"SELECT pg_advisory_xact_lock({SCHEMA_LOCK_KEY})"))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"))
            Base.metadata.create_all(conn)

    def close(self) -> None:
        self.engine.dispose()
