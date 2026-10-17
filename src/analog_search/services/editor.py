"""A specialist edits the knowledge: a draft over the active release, its impact checked, then published.

The draft is the active release with the specialist's edits (domain/editing.py). Before it is published,
its impact is shown against the active release, by the rules alone and in memory:
- rows of the history the draft parses differently;
- golden-card links the rules disagree with, before and after: a row a specialist confirmed but a hard rule
  rejects, a row they turned down but the rules let through;
- evaluation queries whose first row changes.
Published, the draft becomes a new active release and the stored history is parsed with it; the API then
rebuilds the search and re-keys the golden cards (CardService.rekey).

The disagreements of the golden cards with the active release are also the suggestions shown to a
specialist: each says which attribute is at stake and, where an edit would settle it, which one. A
suggestion is never applied by itself: a verdict on one row is evidence, not a rule.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from analog_search.domain.cards import Card, CardStore, Link
from analog_search.domain.editing import MAX_TOLERANCE, Edit, Field, InvalidEdit, apply, current, validated
from analog_search.domain.frozen import Frozen
from analog_search.domain.knowledge import Attribute, Category, Comparison, Knowledge, Level
from analog_search.domain.matching import Assessment, Decision, Outcome, ScoringConfig
from analog_search.domain.positions import ParsedPosition, carried, reparsed
from analog_search.services.knowledge import KnowledgeService, PositionStore
from analog_search.services.search import CandidateStore, SearchService

EXAMPLES = 20  # changed rows shown in an impact report
SAMPLES = 3  # rows of a category shown next to its rules


# Candidates held in memory: a draft is tried on the history without touching the stored one
InMemory = Callable[[str, list[ParsedPosition]], CandidateStore]


class EditStore(Protocol):
    def edits(self) -> list[Edit]:
        """The draft: edits not yet published, in the order they were made."""
        ...

    def add(self, edit: Edit) -> Edit: ...

    def discard(self, edit_id: int) -> bool: ...

    def publish(self, knowledge: Knowledge, note: str | None = None) -> dict[str, int] | None:
        """Store the release, make it active and close the draft with it."""
        ...


class Draft(Frozen):
    active: Knowledge
    edits: list[Edit]
    knowledge: Knowledge  # the active release with the edits


class Disagreement(Frozen):
    """A golden-card link the rules disagree with, and what might settle it."""

    card_id: int
    query: str
    row: int
    name: str
    decision: Decision
    note: str | None
    problem: str
    advice: str
    edit: Edit | None = None  # the advice as an edit of the draft, if one would settle it

    @property
    def key(self) -> tuple[int, int, str]:
        return self.card_id, self.row, self.problem


class Usage(Frozen):
    """The history rows of a category: what an edit of its rules touches."""

    rows: int
    samples: list[str]


class RowChange(Frozen):
    row: int
    name: str
    before: str
    after: str


class QueryChange(Frozen):
    id: str
    query: str
    expected: str
    before: str
    after: str
    correct_before: bool
    correct_after: bool


class Impact(Frozen):
    version: str
    rows: int  # rows of the history the draft parses differently
    row_examples: list[RowChange]
    links: int
    disagreements_before: int
    disagreements_after: int
    resolved: list[Disagreement]
    new: list[Disagreement]
    queries: int
    correct_before: int
    correct_after: int
    changed_queries: list[QueryChange]


class KnowledgeEditor:
    def __init__(
        self,
        edits: EditStore,
        knowledge: KnowledgeService,
        positions: PositionStore,
        cards: CardStore,
        config: ScoringConfig,
        in_memory: InMemory,
        source: str,
        gold: Path | None = None,
    ):
        """source — the history the cards link to (in memory rows are keyed by their number in one source);
        gold — evaluation queries (evaluation/gold_queries.json), if present."""
        self._edits = edits
        self._knowledge = knowledge
        self._positions = positions
        self._cards = cards
        self._config = config
        self._in_memory = in_memory
        self._source = source
        self._gold = gold

    def draft(self) -> Draft:
        active = self._knowledge.active()
        edits = self._edits.edits()
        return Draft(active=active, edits=edits, knowledge=apply(active, edits))

    def edit(self, edit: Edit) -> Edit | None:
        """Put an edit into the draft, in place of an earlier one of the same field; None if it brings the field
        back to the active release's value, so nothing is left to change. InvalidEdit if it makes no sense."""
        draft = self.draft()
        checked = validated(draft.knowledge, edit)
        previous = current(draft.active, edit.category, edit.attribute, edit.field)
        for earlier in draft.edits:
            if earlier.target == edit.target and earlier.id is not None:
                self._edits.discard(earlier.id)
        if checked.value == previous:
            return None
        return self._edits.add(checked.model_copy(update={"previous": previous}))

    def discard(self, edit_id: int) -> bool:
        return self._edits.discard(edit_id)

    def usage(self) -> dict[str, Usage]:
        """The history rows each category holds under the active release."""
        rows: dict[str, list[str]] = defaultdict(list)
        for position in self._stored():
            if position.category_code is not None:
                rows[position.category_code].append(position.name)
        return {code: Usage(rows=len(names), samples=names[:SAMPLES]) for code, names in rows.items()}

    def disagreements(self) -> list[Disagreement]:
        """Where the golden cards and the active release disagree: the suggestions to a specialist."""
        active = self._knowledge.active()
        return disagreements(self._search(active, self._stored()), self._cards.cards())

    def check(self) -> Impact:
        """What publishing the draft would change, by the rules alone, against the active release."""
        draft = self.draft()
        before = self._stored()
        after = carried(reparsed(draft.knowledge, before), before)
        old, new = self._search(draft.active, before), self._search(draft.knowledge, after)
        cards = self._cards.cards()
        disagreed, agreed = disagreements(old, cards), disagreements(new, cards)
        was, now = {d.key for d in disagreed}, {d.key for d in agreed}
        changed = [_change(earlier, later) for earlier, later in _rows(before, after)]
        queries = self._queries(old, new)
        return Impact(
            version=draft.knowledge.version,
            rows=len(changed),
            row_examples=changed[:EXAMPLES],
            links=sum(len(c.links) for c in cards),
            disagreements_before=len(disagreed),
            disagreements_after=len(agreed),
            resolved=[d for d in disagreed if d.key not in now],
            new=[d for d in agreed if d.key not in was],
            queries=len(queries),
            correct_before=sum(q.correct_before for q in queries),
            correct_after=sum(q.correct_after for q in queries),
            changed_queries=[q for q in queries if q.before != q.after],
        )

    def publish(self, author: str | None = None, note: str | None = None) -> dict[str, Any]:
        """The draft as the new active release, the stored history parsed with it."""
        draft = self.draft()
        if not draft.edits:
            raise InvalidEdit("Черновик пуст: публиковать нечего")
        text = "; ".join(s for s in (author, note) if s) or None
        self._edits.publish(draft.knowledge, text)
        return {
            "version": draft.knowledge.version,
            "edits": len(draft.edits),
            **self._knowledge.reparse(draft.knowledge),
        }

    def _stored(self) -> list[ParsedPosition]:
        return self._positions.stored().get(self._source, [])

    def _search(self, knowledge: Knowledge, positions: list[ParsedPosition]) -> SearchService:
        return SearchService(knowledge, self._in_memory(knowledge.version, positions), self._config)

    def _queries(self, old: SearchService, new: SearchService) -> list[QueryChange]:
        if self._gold is None or not self._gold.exists():
            return []
        changes = []
        for gold in json.loads(self._gold.read_text(encoding="utf-8")):
            before, after = _first(old, gold["query"]), _first(new, gold["query"])
            changes.append(
                QueryChange(
                    id=gold["id"],
                    query=gold["query"],
                    expected=", ".join(map(str, gold["expected_ids"])) or "не найдено",
                    before=_shown(before),
                    after=_shown(after),
                    correct_before=_correct(gold, before),
                    correct_after=_correct(gold, after),
                )
            )
        return changes


def disagreements(search: SearchService, cards: list[Card]) -> list[Disagreement]:
    found = []
    for card in cards:
        category, assessed = search.assess(card.name, [link.row for link in card.links])
        by_row = {a.candidate.row: a for a in assessed}
        for link in card.links:
            if (assessment := by_row.get(link.row)) is not None:
                found += _disagreements(card, link, category, assessment)
    return found


def _disagreements(card: Card, link: Link, category: Category | None, a: Assessment) -> list[Disagreement]:
    def said(problem: str, advice: str, edit: Edit | None = None) -> Disagreement:
        return Disagreement(
            card_id=card.id,
            query=card.name,
            row=link.row,
            name=link.name,
            decision=link.decision,
            note=link.note,
            problem=problem,
            advice=advice,
            edit=edit,
        )

    if link.decision in (Decision.SAME, Decision.ANALOG) and a.rejected:
        return [
            said(f"Правило отвергает: {c.attribute.name} {c.query} → {c.candidate}", *_loosened(category, c.attribute))
            for c in a.checks_with(Outcome.MISMATCH)
            if c.attribute.level is Level.HARD
        ]
    if link.decision is not Decision.NOT_ANALOG or a.rejected or category is None:
        return []
    soft = [
        said(
            f"Различие только снижает балл: {c.attribute.name} {c.query} → {c.candidate}",
            "Сделать жёсткой: строки с другим значением не будут предлагаться",
            Edit(category=category.code, attribute=c.attribute.code, field=Field.LEVEL, value=Level.HARD.value),
        )
        for c in a.checks_with(Outcome.MISMATCH)
        if c.attribute.level is Level.SOFT
    ]
    unread = [
        said(
            f"Не прочитана в названии строки: {attribute.name}",
            "Правило есть, но значение не извлекается: нужен экстрактор (задача разработчику)",
        )
        for attribute in [
            *(c.attribute for c in a.checks_with(Outcome.MISSING) if c.attribute.level is Level.HARD),
            *a.unreadable,
        ]
    ]
    return soft + unread or [
        said(
            "Правила пропускают строку и не видят разницы",
            "Отличие не описано ни одной характеристикой: опишите его в комментарии — нужна новая "
            "характеристика (задача разработчику)",
        )
    ]


def _loosened(category: Category | None, attribute: Attribute) -> tuple[str, Edit | None]:
    if category is None:
        return "Проверить правило", None
    tolerance = attribute.rule.get(Field.STEP_TOLERANCE.value, 1)
    if attribute.comparison is Comparison.STEP and tolerance < MAX_TOLERANCE:
        return (
            f"Расширить допуск по ряду значений до ±{tolerance + 1}",
            Edit(category=category.code, attribute=attribute.code, field=Field.STEP_TOLERANCE, value=tolerance + 1),
        )
    return (
        "Сделать мягкой: различие снизит балл, но не отсеет строку",
        Edit(category=category.code, attribute=attribute.code, field=Field.LEVEL, value=Level.SOFT.value),
    )


def _rows(before: list[ParsedPosition], after: list[ParsedPosition]) -> list[tuple[ParsedPosition, ParsedPosition]]:
    old = {p.source_row: p for p in before}
    return [
        (old[p.source_row], p)
        for p in after
        if p.source_row in old
        and (old[p.source_row].category_code, old[p.source_row].attributes) != (p.category_code, p.attributes)
    ]


def _change(before: ParsedPosition, after: ParsedPosition) -> RowChange:
    """The two readings of a row, by what differs between them."""
    if before.category_code != after.category_code:
        was, now = before.category_code or "—", after.category_code or "—"
    else:
        codes = sorted(
            k for k in {*before.attributes, *after.attributes} if before.attributes.get(k) != after.attributes.get(k)
        )
        was, now = _values(before, codes), _values(after, codes)
    return RowChange(row=after.source_row, name=after.name, before=was, after=now)


def _values(position: ParsedPosition, codes: list[str]) -> str:
    return ", ".join(
        f"{code}={position.attributes[code]}" if code in position.attributes else f"{code}=?" for code in codes
    )


def _first(search: SearchService, query: str) -> Assessment | None:
    result = search.search(query)
    return result.candidates[0] if result.found and result.candidates else None


def _shown(first: Assessment | None) -> str:
    return f"{first.candidate.row}: {first.candidate.name[:60]}" if first else "не найдено"


def _correct(gold: dict[str, Any], first: Assessment | None) -> bool:
    if gold["expected_kind"] == "no_match":
        return first is None
    return first is not None and first.candidate.row in gold["expected_ids"]
