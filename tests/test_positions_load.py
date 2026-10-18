"""Loading the history: the vectors already stored are kept when there is nothing to compute new ones with."""

from __future__ import annotations

from app.config import PROJECT_ROOT
from domain.knowledge import Knowledge
from domain.reading import ReaderError
from infrastructure.knowledge import positions_xlsx
from infrastructure.knowledge.yaml_files import YamlKnowledge
from repositories.memory import MemoryKnowledge, MemoryPositions
from services.knowledge import KnowledgeService


class _Keyless:
    """No key and no cache: what a clean machine started from the dump has."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise ReaderError("эмбеддинги: текстов нет в кэше, ключа нет")


class _Recorded(MemoryPositions):
    def __init__(self) -> None:
        super().__init__("positions.xlsx", [])
        self.embedded: list[dict[int, list[float]]] = []

    def save_embeddings(self, source: str, model: str, vectors: dict[int, list[float]]) -> int:
        self.embedded.append(vectors)
        return len(vectors)


def test_a_load_without_vectors_keeps_the_stored_ones(knowledge: Knowledge) -> None:
    positions = _Recorded()
    service = KnowledgeService(
        YamlKnowledge(PROJECT_ROOT / "research"),
        positions_xlsx.read,
        MemoryKnowledge(knowledge),
        positions,
        embedder=_Keyless(),
        embedding_model="m",
    )

    counts = service.load_positions(PROJECT_ROOT / "data" / "positions.xlsx")

    assert positions.embedded == [] and "embedded" not in counts
    assert counts["positions"] > 2000
