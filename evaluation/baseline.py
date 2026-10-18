"""Baseline: plain search by words — the same trigram similarity, no categories, rules or threshold."""

from __future__ import annotations

from dataclasses import dataclass

from domain.knowledge import normalize
from domain.positions import PositionRecord
from domain.trigrams import similarity, trigrams


@dataclass(frozen=True)
class Hit:
    row: int
    name: str
    similarity: float


class Baseline:
    def __init__(self, records: list[PositionRecord]):
        self._rows = [(r, trigrams(normalize(r.name))) for r in records]

    def search(self, query: str, k: int = 5) -> list[Hit]:
        grams = trigrams(normalize(query))
        hits = [Hit(r.row, r.name, similarity(grams, g)) for r, g in self._rows]
        return sorted(hits, key=lambda h: (-h.similarity, h.row))[:k]
