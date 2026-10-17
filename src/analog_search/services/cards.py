"""A specialist's verdicts on candidates: the decision log, golden cards and their ideal from КСР.

The query is read as for a search, and the verdict goes to the card of what was read: every query read into
the same structure gets it. A new card is anchored to a КСР resource whose name the rules read into the same
category and the same value of every hard attribute — the ideal that names the position, not a row of the
history. A query that leaves a hard attribute unstated or unread gets no ideal: «воздуховод d=200» would
otherwise take any wall thickness.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from analog_search.domain.cards import Card, CardKey, CardStore, Entry, ReferenceBook
from analog_search.domain.features import same
from analog_search.domain.frozen import Frozen
from analog_search.domain.knowledge import Level
from analog_search.domain.matching import Decision
from analog_search.services.search import Correction, SearchService, Understanding

REFERENCE_CANDIDATES = 10  # КСР names closest to the query that are read and compared


class Seed(Frozen):
    """One line of a seed file: what a specialist said about a row (by its number in the source file)."""

    row: int
    decision: Decision
    note: str | None = None


class CardService:
    def __init__(self, search: SearchService, cards: CardStore, references: ReferenceBook | None = None):
        self._search = search
        self._cards = cards
        self._references = references

    def decide(
        self,
        query: str,
        position_id: int,
        decision: Decision,
        author: str | None = None,
        note: str | None = None,
        correction: Correction | None = None,
    ) -> Card | None:
        """The card the verdict went to — that of the query as understood, with the specialist's correction;
        None for «unclear», which is only logged."""
        read = self._search.understand(query, correction)
        card = self._cards.decide(
            Entry(
                key=read.key,
                query=query,
                category_code=read.category.code if read.category else None,
                values=read.values,
                position_id=position_id,
                decision=decision,
                author=author,
                note=note,
                knowledge_version=self._search.knowledge.version,
            )
        )
        if card and card.reference is None and self._references and read.category:
            card = self._anchored(card, read)
        return card

    def cards(self) -> list[Card]:
        return self._cards.cards()

    def load(self, path: Path, author: str, source: str) -> dict[str, int]:
        """Verdicts from a seed file on rows of the source; one already in effect is not repeated, so a reload
        changes nothing."""
        stats = {"decisions": 0, "kept": 0, "unknown_rows": 0}
        for item in json.loads(path.read_text(encoding="utf-8")):
            key = self._search.understand(item["query"]).key
            for verdict in (
                Seed(row=v["row"], decision=Decision(v["decision"]), note=v.get("note")) for v in item["verdicts"]
            ):
                position_id = self._cards.position_id(source, verdict.row)
                if position_id is None:
                    stats["unknown_rows"] += 1
                    continue
                card = self._cards.card(key)
                if card and card.decision(position_id) is verdict.decision:
                    stats["kept"] += 1
                    continue
                self.decide(item["query"], position_id, verdict.decision, author, verdict.note)
                stats["decisions"] += 1
        return stats

    def rekey(self) -> dict[str, int]:
        """After a new knowledge release: every card keyed by what its query reads into now, cards that read
        the same merged, a card whose key changed anchored anew."""
        cards = self._cards.cards()
        read = {card.id: self._search.understand(card.name) for card in cards}
        keys = {
            card.id: CardKey(key=r.key, category_code=r.category.code if r.category else None, values=r.values)
            for card in cards
            if (r := read[card.id]).key != card.key
        }
        if not keys:
            return {"rekeyed": 0, "merged": 0}
        counts = self._cards.rekey(keys, self._search.knowledge.version)
        for card in self._cards.cards():
            if card.reference is None and card.id in keys:
                self._anchored(card, read[card.id])
        return counts

    def export(self) -> list[dict[str, Any]]:
        """The cards as evaluation queries (evaluation/gold_queries.json): what specialists confirmed is what
        the search is measured against next."""
        return [_gold(card) for card in self._cards.cards() if card.links]

    def _anchored(self, card: Card, read: Understanding) -> Card:
        if read.category is None or self._references is None:
            return card
        hard = [a.code for a in read.category.attributes if a.level is Level.HARD]
        if not hard or read.declined or any(code not in read.values for code in hard):
            return card
        for reference in self._references.nearest(read.text, REFERENCE_CANDIDATES):
            category, values = self._search.read_by_rules(reference.name)
            if category == read.category and all(
                code in values and same(read.values[code], values[code]) for code in hard
            ):
                return self._cards.anchor(card.id, reference)
        return card


def _gold(card: Card) -> dict[str, Any]:
    found = [link.row for link in card.links if link.decision in (Decision.SAME, Decision.ANALOG)]
    return {
        "id": f"card{card.id}",
        "group": "specialist",
        "query": card.name,
        "expected_kind": "match" if found else "no_match",
        "expected_ids": found,
        "key": card.key,
        "note": "из карточки специалиста",
    }
