"""The specialist's knowledge as releases in the database, procurement positions parsed with the active one.

research/*.yaml seeds an empty base (and `knowledge load` imports it on purpose); after that the knowledge
is changed by a specialist's edits (services/editor.py), each published as a new release, and the stored
history is parsed again with it.

A row is parsed by rules (classifier, extractors); hard attributes the rules leave unread are then
asked of the choice model, if one is configured (domain/reading.py). The model's failure on a row
leaves that row as the rules read it. With an embedder, the names are embedded for the search by meaning.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from analog_search.domain.features import Extractors
from analog_search.domain.knowledge import SERVICE_FLAG, Category, Classifier, Knowledge, normalize
from analog_search.domain.reading import ChoiceReading, Embedder, ReaderError
from analog_search.infrastructure.knowledge import positions_xlsx, yaml_files
from analog_search.repositories.knowledge import ParsedPosition

log = logging.getLogger(__name__)


class KnowledgeStore(Protocol):
    def active(self) -> Knowledge | None: ...

    def save(self, knowledge: Knowledge, note: str | None = None) -> dict[str, int] | None: ...


class PositionStore(Protocol):
    def save(self, source: str, knowledge_version: str, positions: list[ParsedPosition]) -> dict[str, int]: ...

    def save_embeddings(self, source: str, model: str, vectors: dict[int, list[float]]) -> int: ...

    def stored(self) -> dict[str, list[ParsedPosition]]: ...


Reading = Callable[[Knowledge], ChoiceReading | None]  # the choice model over a release's categories


class KnowledgeService:
    def __init__(
        self,
        knowledge_dir: Path,
        knowledge_store: KnowledgeStore,
        position_store: PositionStore,
        reading: Reading = lambda _: None,
        workers: int = 8,
        embedder: Embedder | None = None,
        embedding_model: str = "",
    ):
        self._dir = knowledge_dir
        self._knowledge = knowledge_store
        self._positions = position_store
        self._reading = reading
        self._workers = workers
        self._embedder = embedder
        self._embedding_model = embedding_model

    def load_knowledge(self) -> tuple[Knowledge, dict[str, int] | None]:
        """Import the yaml state as a release and activate it, over the specialists' releases; counts are None
        if it was stored already."""
        knowledge = yaml_files.read(self._dir)
        return knowledge, self._knowledge.save(knowledge, note=str(self._dir))

    def active(self) -> Knowledge:
        """The release in effect; an empty base is seeded from the yaml files."""
        knowledge = self._knowledge.active()
        return knowledge if knowledge is not None else self.load_knowledge()[0]

    def load_positions(self, path: Path) -> dict[str, int]:
        """Parse the workbook with the active knowledge and upsert the rows."""
        knowledge = self.active()
        positions = parse(knowledge, positions_xlsx.read(path))
        counts = self._save(knowledge, path.name, positions)
        # No vectors to be had (no key, no cache) keeps the stored ones: those of a restored dump
        if self._embedder is not None and (vectors := embed(positions, self._embedder)):
            counts["embedded"] = self._positions.save_embeddings(path.name, self._embedding_model, vectors)
        return counts

    def reparse(self, knowledge: Knowledge) -> dict[str, dict[str, int]]:
        """The stored history parsed again with a new release. The names do not change, so neither do their
        vectors; what the model read before is kept where the category still asks for it."""
        return {
            source: self._save(knowledge, source, carried(reparsed(knowledge, stored), stored))
            for source, stored in self._positions.stored().items()
        }

    def _save(self, knowledge: Knowledge, source: str, positions: list[ParsedPosition]) -> dict[str, int]:
        counts: dict[str, int] = {}
        reading = self._reading(knowledge)
        if reading is not None:
            positions, counts = read_hard(knowledge, positions, reading, self._workers)
        return {**self._positions.save(source, knowledge.version, positions), **counts}


def reparsed(knowledge: Knowledge, stored: list[ParsedPosition]) -> list[ParsedPosition]:
    """Stored rows parsed by the rules of another release, from their names."""
    return parse(knowledge, [positions_xlsx.PositionRecord(p.source_row, p.name, p.price) for p in stored])


def carried(parsed: list[ParsedPosition], stored: list[ParsedPosition]) -> list[ParsedPosition]:
    """What the model read before, added where the rules still leave it unread in the same category."""
    before = {p.source_row: p for p in stored}
    result = []
    for position in parsed:
        old = before.get(position.source_row)
        if old is None or not old.read_by_llm or old.category_code != position.category_code:
            result.append(position)
            continue
        kept = {code: old.attributes[code] for code in old.read_by_llm if code not in position.attributes}
        result.append(
            replace(position, attributes={**position.attributes, **kept}, read_by_llm=tuple(sorted(kept)))
            if kept
            else position
        )
    return result


def parse(knowledge: Knowledge, records: list[positions_xlsx.PositionRecord]) -> list[ParsedPosition]:
    classifier = Classifier(knowledge.categories)
    extractors = Extractors(knowledge.categories)
    parsed = []
    for record in records:
        match = classifier.classify(record.name)
        category = match.category if match else None
        text = normalize(record.name)
        parsed.append(
            ParsedPosition(
                source_row=record.row,
                name=record.name,
                price=record.price,
                search_text=text,
                category_code=category.code if category else None,
                attributes=extractors.extract(category, text) if category else {},
                flags=(SERVICE_FLAG,) if category and category.is_service else (),
            )
        )
    return parsed


def embed(positions: list[ParsedPosition], embedder: Embedder) -> dict[int, list[float]]:
    """Vectors of the goods rows' names by source row; none if the embeddings are unavailable."""
    goods = [p for p in positions if SERVICE_FLAG not in p.flags]
    try:
        vectors = embedder.embed([p.name for p in goods])
    except ReaderError as error:
        log.warning("no search by meaning: %s", error)
        return {}
    return {p.source_row: v for p, v in zip(goods, vectors, strict=True)}


def read_hard(
    knowledge: Knowledge, positions: list[ParsedPosition], reading: ChoiceReading, workers: int = 8
) -> tuple[list[ParsedPosition], dict[str, int]]:
    """Positions with the hard values the model read added; counts of rows asked, read and failed."""
    categories = {c.code: c for c in knowledge.categories}
    extractors = Extractors(knowledge.categories)

    def read(item: tuple[ParsedPosition, Category]) -> ParsedPosition | None:
        position, category = item
        skip = extractors.declined(category, position.search_text)
        try:
            values = reading.read(category, position.name, position.attributes, skip).values
        except ReaderError as error:
            log.warning("row %s: %s", position.source_row, error)
            return None
        return replace(position, attributes={**position.attributes, **values}, read_by_llm=tuple(sorted(values)))

    asked = {
        i: category
        for i, p in enumerate(positions)
        if p.category_code
        and SERVICE_FLAG not in p.flags
        and set(reading.readable(category := categories[p.category_code])) - set(p.attributes)
    }
    with ThreadPoolExecutor(max_workers=workers) as pool:
        answers = list(pool.map(read, [(positions[i], c) for i, c in asked.items()]))
    result = list(positions)
    for i, answer in zip(asked, answers, strict=True):
        if answer is not None:
            result[i] = answer
    return result, {
        "llm_asked": len(asked),
        "llm_read": sum(bool(a and a.read_by_llm) for a in answers),
        "llm_failed": sum(a is None for a in answers),
    }
