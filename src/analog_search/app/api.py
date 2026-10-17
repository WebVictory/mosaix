"""HTTP API and the page.

POST /api/search {"query": "...", "correction"?: {"category"?, "values"?: {attribute: option | null}}}
                                  → what was understood (the category and all its attributes, stated or not,
                                    with what each may be set to), candidates with the score breakdown, or
                                    «не найдено» with the reason (both, when the candidates could not be verified)
POST /api/decisions {"query", "position_id", "decision", "note"?, "author"?, "correction"?}
                                  → a specialist's verdict on a candidate: same | analog | not_analog | unclear;
                                    the golden card it went to
GET  /api/suggest?q=...           → while a query is typed: its typing mistakes fixed, its last word finished,
                                    the categories it names, the names of the history that hold it
GET  /api/categories              → what a query's category may be corrected to
GET  /api/cards                   → golden cards with their rows and КСР ideal
GET  /api/stats                   → what the search works with
GET  /                            → static/index.html
The knowledge editor: app/knowledge_api.py.

uvicorn --factory analog_search.app.api:create_app
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from analog_search.app.bootstrap import Services, services
from analog_search.app.config import PROJECT_ROOT, Settings
from analog_search.app.knowledge_api import router as knowledge_router
from analog_search.domain.cards import Card, UnknownPosition
from analog_search.domain.editing import InvalidEdit
from analog_search.domain.matching import Assessment, Check, Decision, Outcome
from analog_search.repositories.database import Database
from analog_search.services.search import Correction, Facet, InvalidQuery, SearchResult

log = logging.getLogger(__name__)

PAGE = PROJECT_ROOT / "static" / "index.html"


class CorrectionIn(BaseModel):
    """A specialist's correction of what was understood from the query."""

    category: str | None = None  # a category code
    values: dict[str, str | None] = Field(default_factory=dict)  # attribute code → OptionOut.id; null — not stated

    def correction(self) -> Correction:
        return Correction(category=self.category, values=self.values)


class SearchRequest(BaseModel):
    query: str
    correction: CorrectionIn | None = None


class DecisionRequest(BaseModel):
    query: str
    position_id: int  # CandidateOut.id
    decision: Decision
    note: str | None = Field(None, max_length=1000)
    author: str | None = Field(None, max_length=200)
    correction: CorrectionIn | None = None  # the one the search was made with


class OptionOut(BaseModel):
    id: str
    label: str


class FacetOut(BaseModel):
    code: str
    name: str
    level: str
    value: str | None  # null — not stated in the query, or not read
    source: str | None  # rules | llm | specialist
    declined: bool  # stated in a form nobody reads
    options: list[OptionOut]  # what it may be set to; [] — not settable
    option: str | None  # the value among the options


class CategoryOut(BaseModel):
    code: str
    name: str
    group: str | None


class CheckOut(BaseModel):
    code: str
    name: str
    level: str
    query: str
    candidate: str | None
    exact: bool  # False on a match: the candidate is not the same but meets the rule (FRLS for LS)
    query_by_llm: bool  # a value read by the choice model, not the rules
    candidate_by_llm: bool


class CandidateOut(BaseModel):
    id: int  # what a verdict is given on
    row: int
    name: str
    price: float | None
    category: str | None
    score: float
    kind: str | None  # same | analog | unverified | text; None — rejected (shown only as the closest of a miss)
    breakdown: dict[str, float]
    matches: list[CheckOut]
    differences: list[CheckOut]
    missing: list[CheckOut]
    unchecked: list[str]  # hard attributes nobody could read: a buyer checks them by hand
    foreign: bool  # a row of another category, checked by the query's rules read from its name
    verdict_by_llm: str | None  # same | analog | other: the choice model judged what the rules left unsettled
    confirmed: str | None  # same | analog | not_analog: a specialist's verdict, it wins over the rules


class ReferenceOut(BaseModel):
    source: str
    code: str
    name: str


class LinkOut(BaseModel):
    position_id: int
    row: int
    name: str
    decision: str
    note: str | None  # the specialist's reason


class CardOut(BaseModel):
    id: int
    key: str
    name: str
    category: str | None
    reference: ReferenceOut | None  # the ideal from КСР
    links: list[LinkOut]


class NotFoundOut(BaseModel):
    reason: str
    message: str
    closest: CandidateOut | None


class SearchResponse(BaseModel):
    query: str
    found: bool
    knowledge_version: str
    category: str | None
    category_code: str | None
    group: str | None
    category_source: str | None  # rules | llm (no trigger word matched) | specialist
    facets: list[FacetOut]  # every attribute of the category
    candidates: list[CandidateOut]
    explain_not_found: NotFoundOut | None
    card: CardOut | None  # the typical position specialists confirmed rows for
    spelled: str | None = None  # the query with its typing mistakes fixed, if it has any


class FixOut(BaseModel):
    typed: str
    meant: str


class SuggestResponse(BaseModel):
    text: str  # the query with the fixes
    fixes: list[FixOut]
    completions: list[str]  # the query with its last word finished
    categories: list[CategoryOut]  # the categories it names or is the start of
    names: list[str]  # of the history, those starting with the query first


class DecisionResponse(BaseModel):
    card: CardOut | None  # None for «unclear»: only logged


def create_app(build: Callable[[], Services] | None = None) -> FastAPI:
    """build — the services; by default wired to the database from Settings. Called again when a new knowledge
    release is published."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        db = None
        if build is None:
            settings = Settings()
            db = Database(settings.database_url)
            app.state.build = lambda: services(settings, db)
        else:
            app.state.build = build
        app.state.services = app.state.build()
        for warning in app.state.services.search.stats()["warnings"]:
            log.warning(warning)
        yield
        if db is not None:
            db.close()

    app = FastAPI(title="Поиск аналогов", lifespan=lifespan)

    @app.exception_handler(InvalidQuery)
    async def invalid_query(_: Request, error: InvalidQuery) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(error)})

    @app.exception_handler(InvalidEdit)
    async def invalid_edit(_: Request, error: InvalidEdit) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(error)})

    @app.exception_handler(UnknownPosition)
    async def unknown_position(_: Request, error: UnknownPosition) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": f"Нет позиции с id {error}"})

    @app.exception_handler(Exception)
    async def unexpected(_: Request, error: Exception) -> JSONResponse:
        trace_id = uuid.uuid4().hex
        log.exception("trace_id=%s", trace_id, exc_info=error)
        return JSONResponse(status_code=500, content={"detail": "Внутренняя ошибка", "trace_id": trace_id})

    @app.post("/api/search")
    def search(request: SearchRequest) -> SearchResponse:
        correction = request.correction.correction() if request.correction else None
        search = app.state.services.search
        spelled = search.spell(request.query)
        return to_response(search.search(request.query, correction)).model_copy(
            update={"spelled": spelled.text if spelled.fixes else None}
        )

    @app.get("/api/suggest")
    def suggest(q: str = "") -> SuggestResponse:
        found = app.state.services.search.suggest(q)
        return SuggestResponse(
            text=found.text,
            fixes=[FixOut(typed=f.typed, meant=f.meant) for f in found.fixes],
            completions=found.completions,
            categories=[CategoryOut(code=c.code, name=c.name, group=c.group) for c in found.categories],
            names=found.names,
        )

    @app.post("/api/decisions")
    def decide(request: DecisionRequest) -> DecisionResponse:
        card = app.state.services.cards.decide(
            request.query,
            request.position_id,
            request.decision,
            request.author,
            request.note,
            request.correction.correction() if request.correction else None,
        )
        return DecisionResponse(card=_card(card) if card else None)

    @app.get("/api/categories")
    def categories() -> list[CategoryOut]:
        return [CategoryOut(code=c.code, name=c.name, group=c.group) for c in app.state.services.search.categories()]

    @app.get("/api/cards")
    def cards() -> list[CardOut]:
        return [_card(c) for c in app.state.services.cards.cards()]

    @app.get("/api/stats")
    def stats() -> dict[str, Any]:
        return app.state.services.search.stats()

    @app.get("/", include_in_schema=False)
    def page() -> FileResponse:
        return FileResponse(PAGE)

    app.include_router(knowledge_router)
    return app


def to_response(result: SearchResult) -> SearchResponse:
    """The search result as the API returns it."""
    not_found = result.not_found
    return SearchResponse(
        query=result.query,
        found=result.found,
        knowledge_version=result.knowledge_version,
        category=result.category.name if result.category else None,
        category_code=result.category.code if result.category else None,
        group=result.category.group if result.category else None,
        category_source=result.category_source,
        facets=[_facet(f) for f in result.facets],
        candidates=[_candidate(a) for a in result.candidates],
        explain_not_found=NotFoundOut(
            reason=not_found.reason,
            message=not_found.message,
            closest=_candidate(not_found.closest) if not_found.closest else None,
        )
        if not_found
        else None,
        card=_card(result.card) if result.card else None,
    )


def _candidate(assessment: Assessment) -> CandidateOut:
    candidate = assessment.candidate
    return CandidateOut(
        id=candidate.id,
        row=candidate.row,
        name=candidate.name,
        price=candidate.price,
        category=candidate.category_code,
        score=assessment.score,
        kind=assessment.kind,
        breakdown=assessment.breakdown,
        matches=[_check(c) for c in assessment.checks_with(Outcome.MATCH)],
        differences=[_check(c) for c in assessment.checks_with(Outcome.MISMATCH)],
        missing=[_check(c) for c in assessment.checks_with(Outcome.MISSING)],
        unchecked=[a.name for a in assessment.unreadable],
        foreign=assessment.foreign,
        verdict_by_llm=assessment.verdict,
        confirmed=assessment.confirmed,
    )


def _card(card: Card) -> CardOut:
    reference = card.reference
    return CardOut(
        id=card.id,
        key=card.key,
        name=card.name,
        category=card.category_code,
        reference=ReferenceOut(source=reference.source, code=reference.code, name=reference.name)
        if reference
        else None,
        links=[
            LinkOut(position_id=link.position_id, row=link.row, name=link.name, decision=link.decision, note=link.note)
            for link in card.links
        ],
    )


def _facet(facet: Facet) -> FacetOut:
    attribute = facet.attribute
    return FacetOut(
        code=attribute.code,
        name=attribute.name,
        level=attribute.level,
        value=facet.value,
        source=facet.source,
        declined=facet.declined,
        options=[OptionOut(id=o.id, label=o.label) for o in facet.options],
        option=facet.option,
    )


def _check(check: Check) -> CheckOut:
    attribute = check.attribute
    return CheckOut(
        code=attribute.code,
        name=attribute.name,
        level=attribute.level,
        query=check.query,
        candidate=check.candidate,
        exact=check.exact,
        query_by_llm=check.query_by_llm,
        candidate_by_llm=check.candidate_by_llm,
    )
