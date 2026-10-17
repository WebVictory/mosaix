"""The research release and the workbook, parsed once per run: the search tests use the real data."""

from __future__ import annotations

import pytest

from analog_search.app.config import PROJECT_ROOT
from analog_search.domain.knowledge import Knowledge
from analog_search.infrastructure import scoring_yaml
from analog_search.infrastructure.knowledge import positions_xlsx, yaml_files
from analog_search.repositories.knowledge import ParsedPosition
from analog_search.repositories.memory import MemoryStore
from analog_search.services.knowledge import parse
from analog_search.services.search import SearchService


@pytest.fixture(scope="session")
def knowledge() -> Knowledge:
    return yaml_files.read(PROJECT_ROOT / "research")


@pytest.fixture(scope="session")
def positions(knowledge: Knowledge) -> dict[int, ParsedPosition]:
    records = positions_xlsx.read(PROJECT_ROOT / "data" / "positions.xlsx")
    return {p.source_row: p for p in parse(knowledge, records)}


@pytest.fixture(scope="session")
def service(knowledge: Knowledge, positions: dict[int, ParsedPosition]) -> SearchService:
    config = scoring_yaml.read(PROJECT_ROOT / "config" / "search.yaml")
    return SearchService(knowledge, MemoryStore(knowledge.version, list(positions.values())), config)
