"""Golden cards: what specialists confirmed, kept as typical positions the search answers from first.

A card is a typical position: a category and the attribute values a query states, whatever its words —
«ИБП 12В для улицы» and «бесперебойник 12 вольт уличный» read into the same structure, so they are one card.
A query nothing was read from is keyed by its normalized text.
Its links to rows of the history are the specialists' verdicts:
- same — the row is this position under another name;
- analog — the row may replace it; a link is one-way: FRLS may replace LS, LS may not replace FRLS;
- not_analog — the row is never offered for it, whatever the rules say.
The links make a graph, but a shallow one: «the same» is a cluster (the card), an analog is not
transitive (A for B and B for C does not give A for C), so nothing is walked further than one step:
two tables in PostgreSQL, no graph store. A verdict never merges cards: a wrong «the same» would join two
clusters for good. Only a new knowledge release does, when the queries of two cards now read into the same
structure — the same key (CardService.rekey). A link holds the latest verdict on the pair; every verdict
stays in the decision log.
A card may be anchored to an ideal from a reference book: the КСР resource whose name reads into the
same values. It names the position for everyone, but is not a row of the history.
"""

from __future__ import annotations

import json
from typing import Protocol

from analog_search.domain.features import Value
from analog_search.domain.frozen import Frozen
from analog_search.domain.knowledge import Category
from analog_search.domain.matching import Assessment, Decision, Kind

LINKS = (Decision.SAME, Decision.ANALOG, Decision.NOT_ANALOG)


def card_key(category: Category | None, values: dict[str, Value], text: str) -> str:
    """The same for every query read into the same structure."""
    body = json.dumps(values, ensure_ascii=False, sort_keys=True) if values else text
    return f"{category.code if category else '-'}|{body}"


class Link(Frozen):
    position_id: int
    row: int
    name: str
    decision: Decision
    note: str | None = None  # why, in the specialist's words: «FRLS вместо LS: огнестойкий, не хуже»


class Reference(Frozen):
    """An ideal position from a reference book: КСР resource code and name."""

    source: str
    code: str
    name: str


class Card(Frozen):
    id: int
    key: str
    name: str
    category_code: str | None
    links: tuple[Link, ...] = ()
    reference: Reference | None = None

    def decision(self, position_id: int) -> Decision | None:
        return next((link.decision for link in self.links if link.position_id == position_id), None)


class Entry(Frozen):
    """One verdict of a specialist: the query, how it was read, the row and what they said."""

    key: str
    query: str
    category_code: str | None
    values: dict[str, Value]
    position_id: int
    decision: Decision
    author: str | None = None
    note: str | None = None
    knowledge_version: str | None = None


class CardKey(Frozen):
    """What a card's query reads into with a knowledge release."""

    key: str
    category_code: str | None
    values: dict[str, Value]


class UnknownPosition(LookupError):
    pass


class CardStore(Protocol):
    def card(self, key: str) -> Card | None: ...

    def decide(self, entry: Entry) -> Card | None:
        """Log the verdict; unless it is unclear, link the row to the card of the key (created on first use).
        UnknownPosition if there is no such row."""
        ...

    def anchor(self, card_id: int, reference: Reference) -> Card: ...

    def cards(self) -> list[Card]: ...

    def rekey(self, keys: dict[int, CardKey], knowledge_version: str | None = None) -> dict[str, int]:
        """Give the cards the keys their queries read into now. Cards meeting on one key are merged into the
        lowest id: their links and log join it, a row linked by both keeps the later verdict. A card whose key
        changed loses its ideal: it was anchored by the old reading. Counts of rekeyed and merged cards."""
        ...

    def position_id(self, source: str, row: int) -> int | None:
        """The id of a row of the history by its source file and number in it."""
        ...


class ReferenceBook(Protocol):
    def nearest(self, text: str, limit: int) -> list[Reference]:
        """Ideal positions whose names are the closest to the text."""
        ...


def confirm(assessment: Assessment, decision: Decision | None) -> Assessment:
    """A specialist's verdict wins over the rules: a confirmed row is shown even if a rule rejects it —
    the disagreement is a reason to revise the rule, and the UI shows it."""
    match decision:
        case Decision.SAME:
            return assessment.model_copy(update={"kind": Kind.SAME, "confirmed": decision})
        case Decision.ANALOG:
            return assessment.model_copy(update={"kind": Kind.ANALOG, "confirmed": decision})
        case Decision.NOT_ANALOG:
            return assessment.model_copy(update={"confirmed": decision})
    return assessment
