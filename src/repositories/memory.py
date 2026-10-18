"""Candidates, golden cards, knowledge releases and their edits held in memory: the database stores' twins
for evaluation and tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from domain.cards import Card, CardKey, Entry, Link, Reference, UnknownPosition
from domain.editing import Edit
from domain.knowledge import SERVICE_FLAG, Knowledge
from domain.matching import Candidate, Decision
from domain.positions import ParsedPosition
from domain.trigrams import similarity, trigrams
from domain.vectors import cosine


class MemoryStore:
    def __init__(
        self, knowledge_version: str, positions: list[ParsedPosition], vectors: dict[int, list[float]] | None = None
    ):
        """vectors — embeddings of the names by source row; without them, no similarity by meaning."""
        self._version = knowledge_version
        self._positions = [(p, trigrams(p.search_text)) for p in positions if SERVICE_FLAG not in p.flags]
        self._service = len(positions) - len(self._positions)
        self._vectors = vectors or {}

    def in_category(
        self, knowledge_version: str, category_code: str, text: str, vector: list[float] | None = None
    ) -> list[Candidate]:
        if knowledge_version != self._version:
            return []
        return self._ranked(text, vector, lambda p: p.category_code == category_code)

    def nearest(self, text: str, limit: int, vector: list[float] | None = None) -> list[Candidate]:
        ranked = self._ranked(text, vector, lambda _: True)
        by_meaning = sorted((c for c in ranked if c.meaning is not None), key=lambda c: (-(c.meaning or 0.0), c.row))
        return list({c.id: c for c in ranked[:limit] + by_meaning[:limit]}.values())

    def positions(self, ids: list[int], text: str, vector: list[float] | None = None) -> list[Candidate]:
        return self._ranked(text, vector, lambda p: p.source_row in ids)

    def stats(self) -> dict[str, Any]:
        positions = [p for p, _ in self._positions]
        return {
            "positions": len(positions) + self._service,
            "categorized": sum(p.category_code is not None for p in positions) + self._service,
            "service": self._service,
            "with_attributes": sum(bool(p.attributes) for p in positions),
            "with_llm_readings": sum(bool(p.read_by_llm) for p in positions),
            "with_embeddings": sum(p.source_row in self._vectors for p in positions),
            "parsed_with": {self._version: len(positions) + self._service},
        }

    def names(self) -> list[str]:
        return [p.name for p, _ in self._positions]

    def _ranked(
        self, text: str, vector: list[float] | None, accept: Callable[[ParsedPosition], bool]
    ) -> list[Candidate]:
        query = trigrams(text)
        found = [
            Candidate(
                id=p.source_row,
                row=p.source_row,
                name=p.name,
                price=p.price,
                category_code=p.category_code,
                attributes=p.attributes,
                similarity=round(similarity(query, grams), 4),
                meaning=round(cosine(vector, v), 4) if vector and (v := self._vectors.get(p.source_row)) else None,
                read_by_llm=frozenset(p.read_by_llm),
            )
            for p, grams in self._positions
            if accept(p)
        ]
        return sorted(found, key=lambda c: (-c.similarity, c.row))


class MemoryCards:
    """Golden cards over positions held in memory; a position's id is its source row, as in MemoryStore."""

    def __init__(self, positions: list[ParsedPosition]):
        self._names = {p.source_row: p.name for p in positions}
        self._cards: dict[str, Card] = {}
        self._when: dict[tuple[int, int], int] = {}  # (card, row) → the number of its verdict in the log
        self.log: list[Entry] = []

    def card(self, key: str) -> Card | None:
        return self._cards.get(key)

    def decide(self, entry: Entry) -> Card | None:
        if entry.position_id not in self._names:
            raise UnknownPosition(entry.position_id)
        self.log.append(entry)
        if entry.decision is Decision.UNCLEAR:
            return None
        card = self._cards.get(entry.key) or Card(
            id=len(self._cards) + 1, key=entry.key, name=entry.query, category_code=entry.category_code
        )
        link = Link(
            position_id=entry.position_id,
            row=entry.position_id,
            name=self._names[entry.position_id],
            decision=entry.decision,
            note=entry.note,
        )
        links = sorted([*(k for k in card.links if k.position_id != entry.position_id), link], key=lambda k: k.row)
        self._cards[entry.key] = card.model_copy(update={"links": tuple(links)})
        self._when[card.id, entry.position_id] = len(self.log)
        return self._cards[entry.key]

    def anchor(self, card_id: int, reference: Reference) -> Card:
        key = next(key for key, card in self._cards.items() if card.id == card_id)
        self._cards[key] = self._cards[key].model_copy(update={"reference": reference})
        return self._cards[key]

    def cards(self) -> list[Card]:
        return sorted(self._cards.values(), key=lambda c: c.id)

    def rekey(self, keys: dict[int, CardKey], knowledge_version: str | None = None) -> dict[str, int]:
        groups: dict[str, list[Card]] = {}
        for card in self.cards():
            groups.setdefault(keys[card.id].key if card.id in keys else card.key, []).append(card)
        cards, merged = {}, 0
        for key, (survivor, *others) in groups.items():
            latest: dict[int, Link] = {}
            for card in (survivor, *others):
                for link in card.links:
                    kept = latest.get(link.position_id)
                    if (
                        kept is None
                        or self._when[card.id, link.position_id] > self._when[survivor.id, kept.position_id]
                    ):
                        latest[link.position_id] = link
                        self._when[survivor.id, link.position_id] = self._when[card.id, link.position_id]
            merged += len(others)
            if key != survivor.key:
                survivor = survivor.model_copy(
                    update={"key": key, "category_code": keys[survivor.id].category_code, "reference": None}
                )
            cards[key] = survivor.model_copy(update={"links": tuple(sorted(latest.values(), key=lambda k: k.row))})
        self._cards = cards
        return {"rekeyed": len(keys), "merged": merged}

    def position_id(self, source: str, row: int) -> int | None:
        return row if row in self._names else None


class MemoryKnowledge:
    """Knowledge releases and the draft of edits over the active one."""

    def __init__(self, knowledge: Knowledge | None = None):
        self.releases: list[Knowledge] = [knowledge] if knowledge else []
        self._active = knowledge
        self._edits: list[Edit] = []
        self._published: list[Edit] = []

    def active(self) -> Knowledge | None:
        return self._active

    def save(self, knowledge: Knowledge, note: str | None = None) -> dict[str, int] | None:
        stored = next((k for k in self.releases if k.version == knowledge.version), None)
        self._active = stored or knowledge
        if stored is not None:
            return None
        self.releases.append(knowledge)
        return {"categories": len(knowledge.categories)}

    def edits(self) -> list[Edit]:
        return list(self._edits)

    def add(self, edit: Edit) -> Edit:
        added = edit.model_copy(update={"id": len(self._edits) + len(self._published) + 1})
        self._edits.append(added)
        return added

    def discard(self, edit_id: int) -> bool:
        kept = [e for e in self._edits if e.id != edit_id]
        found, self._edits = len(kept) != len(self._edits), kept
        return found

    def publish(self, knowledge: Knowledge, note: str | None = None) -> dict[str, int] | None:
        counts = self.save(knowledge, note)
        self._published += self._edits
        self._edits = []
        return counts


class MemoryPositions:
    """Parsed rows of the history by source, as the database keeps them."""

    def __init__(self, source: str, positions: list[ParsedPosition]):
        self.parsed: dict[str, list[ParsedPosition]] = {source: positions}
        self.versions: dict[str, str] = {}

    def save(self, source: str, knowledge_version: str, positions: list[ParsedPosition]) -> dict[str, int]:
        self.parsed[source], self.versions[source] = positions, knowledge_version
        return {"positions": len(positions), "categorized": sum(p.category_code is not None for p in positions)}

    def save_embeddings(self, source: str, model: str, vectors: dict[int, list[float]]) -> int:
        return len(vectors)

    def stored(self) -> dict[str, list[ParsedPosition]]:
        return dict(self.parsed)
