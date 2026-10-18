"""The research release and the workbook, parsed once per run: the search tests use the real data."""

from __future__ import annotations

import pytest

from app.config import PROJECT_ROOT
from app.search_config import SEARCH
from domain.knowledge import Knowledge
from domain.positions import ParsedPosition, parse
from infrastructure.knowledge import positions_xlsx, yaml_files
from repositories.memory import MemoryStore
from services.search import SearchService


@pytest.fixture(scope="session")
def knowledge() -> Knowledge:
    return yaml_files.read(PROJECT_ROOT / "research")


@pytest.fixture(scope="session")
def positions(knowledge: Knowledge) -> dict[int, ParsedPosition]:
    records = positions_xlsx.read(PROJECT_ROOT / "data" / "positions.xlsx")
    return {p.source_row: p for p in parse(knowledge, records)}


@pytest.fixture(scope="session")
def service(knowledge: Knowledge, positions: dict[int, ParsedPosition]) -> SearchService:
    return SearchService(knowledge, MemoryStore(knowledge.version, list(positions.values())), SEARCH)
