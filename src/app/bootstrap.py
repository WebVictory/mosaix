"""Composition root: the search, the specialist's verdicts and the knowledge editor wired to the database
(the active knowledge release), the choice model and the embeddings.

ANALOG_LLM: off — neither; cache — answers and vectors kept before, no key needed; live — what is not kept
is asked (OPENROUTER_API_KEY); auto — live with the key, cache without.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.search_config import SEARCH
from domain.judging import Judge
from domain.knowledge import Knowledge
from domain.reading import ChoiceReader, ChoiceReading, Embedder, ReaderError
from domain.sources import SourceStore
from infrastructure.knowledge import positions_xlsx
from infrastructure.knowledge.yaml_files import YamlKnowledge
from infrastructure.llm.cache import CachedReader
from infrastructure.llm.embeddings import CachedEmbedder, OpenRouterEmbedder
from infrastructure.llm.jev import JevReader
from infrastructure.sources.formats import formats
from infrastructure.sources.store import Downloader, ReleaseFiles
from repositories.cards import CardRepository, KsrReferences
from repositories.database import Database
from repositories.knowledge import KnowledgeRepository, PositionRepository
from repositories.memory import MemoryStore
from repositories.search import SearchRepository
from services.cards import CardService
from services.editor import KnowledgeEditor
from services.knowledge import KnowledgeService
from services.search import SearchService
from services.sources import SourceService


@dataclass(frozen=True)
class Services:
    search: SearchService
    cards: CardService
    editor: KnowledgeEditor | None = None


@dataclass(frozen=True)
class Models:
    reader: ChoiceReader
    embedder: Embedder


def models(settings: Settings, max_retries: int = 2) -> Models | None:
    """The choice model and the embeddings in the mode in effect; None when it is off."""
    if settings.llm_mode == "off":
        return None
    reader, embedder = None, None
    if settings.llm_mode == "live":
        key = settings.openrouter_api_key.get_secret_value() if settings.openrouter_api_key else ""
        if not key:
            raise ReaderError("ANALOG_LLM=live требует OPENROUTER_API_KEY в окружении")
        reader = JevReader(key, settings.llm_model, settings.llm_timeout_s, max_retries)
        embedder = OpenRouterEmbedder(key, settings.embedding_model, settings.llm_timeout_s)
    return Models(
        CachedReader(settings.llm_cache, settings.llm_model, inner=reader),
        CachedEmbedder(settings.embedding_cache, settings.embedding_model, inner=embedder),
    )


def choice_reading(settings: Settings, knowledge: Knowledge, used: Models | None = None) -> ChoiceReading | None:
    """What the rules leave unread in a name, read by the choice model; None when it is off."""
    used = used or models(settings)
    if used is None:
        return None
    return ChoiceReading(
        used.reader,
        knowledge.categories,
        SEARCH.confidence.min,
        SEARCH.confidence.stated,
        embedder=used.embedder,
    )


def knowledge_service(settings: Settings, db: Database, used: Models | None = None) -> KnowledgeService:
    """Releases in the database (seeded from research/*.yaml), the history parsed with them; with the models,
    unread hard attributes are asked of the choice model and the names embedded."""
    return KnowledgeService(
        YamlKnowledge(settings.knowledge_dir),
        positions_xlsx.read,
        KnowledgeRepository(db),
        PositionRepository(db),
        lambda knowledge: choice_reading(settings, knowledge, used) if used else None,
        settings.llm_workers,
        used.embedder if used else None,
        settings.embedding_model,
    )


def search_service(settings: Settings, db: Database) -> SearchService:
    """Over the active knowledge release: the one the positions were parsed with."""
    knowledge = knowledge_service(settings, db).active()
    used = models(settings, max_retries=0)  # a user waits: one attempt, then the rules alone
    return SearchService(
        knowledge,
        SearchRepository(db, settings.embedding_model),
        SEARCH,
        choice_reading(settings, knowledge, used),
        settings.llm_mode,
        embedder=used.embedder if used else None,
        judge=Judge(used.reader, SEARCH.confidence.min, SEARCH.candidates.judge_k) if used else None,
        cards=CardRepository(db),
    )


def services(settings: Settings, db: Database) -> Services:
    search = search_service(settings, db)
    cards = CardRepository(db)
    editor = KnowledgeEditor(
        KnowledgeRepository(db),
        knowledge_service(settings, db, models(settings)),
        PositionRepository(db),
        cards,
        search.rules,
        MemoryStore,
        settings.positions_file.name,
        settings.gold_queries,
    )
    return Services(search, CardService(search, cards, KsrReferences(db)), editor)


def source_service(
    settings: Settings, downloader: Downloader | None = None, store: SourceStore | None = None
) -> SourceService:
    """External taxonomies under settings.sources_dir: a downloader to fetch them, a store to load them."""
    return SourceService(ReleaseFiles(settings.sources_dir), formats(settings.sources_dir, downloader), store)
