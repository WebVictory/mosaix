"""HTTP API of the knowledge editor (services/editor.py) and its page.

GET    /api/knowledge              → the draft: categories with the edits applied and their history rows, the
                                     edits, the allowed values
GET    /api/knowledge/suggestions  → where the golden cards disagree with the active release, and what might
                                     settle it
POST   /api/knowledge/edits        {"category", "attribute"?, "field", "value", "author"?, "reason"?}
                                   → the edit as it went into the draft; null if it restores the active value
DELETE /api/knowledge/edits/{id}   → drops an edit from the draft
POST   /api/knowledge/check        → what publishing the draft would change
POST   /api/knowledge/publish      {"author"?, "note"?} → a new active release; the history parsed with it,
                                     the search rebuilt, the golden cards re-keyed
GET    /knowledge                  → static/knowledge.html
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from analog_search.app.config import PROJECT_ROOT
from analog_search.domain import editing
from analog_search.domain.knowledge import Attribute, AttributeStatus, Category, Comparison, Level
from analog_search.services.editor import Disagreement, Impact, KnowledgeEditor, Usage

PAGE = PROJECT_ROOT / "static" / "knowledge.html"

router = APIRouter()


class EditRequest(BaseModel):
    category: str
    attribute: str | None = None
    field: editing.Field
    value: Any = None
    author: str | None = Field(None, max_length=200)
    reason: str | None = Field(None, max_length=1000)


class PublishRequest(BaseModel):
    author: str | None = Field(None, max_length=200)
    note: str | None = Field(None, max_length=1000)


class EditOut(BaseModel):
    id: int | None
    category: str
    attribute: str | None
    field: str
    value: Any
    previous: Any
    author: str | None
    reason: str | None
    created_at: datetime | None


class AttributeOut(BaseModel):
    code: str
    name: str
    unit: str | None
    level: str
    comparison: str
    status: str
    provenance: str
    source_ref: str | None
    step_tolerance: int
    substitution: str | None
    series: list[Any] | None  # the ordered values a step comparison walks


class CategoryOut(BaseModel):
    code: str
    name: str
    group: str | None
    triggers: list[str]
    anti_triggers: list[str]
    synonyms: list[str]
    attributes: list[AttributeOut]
    rows: int  # history rows of the category under the active release
    samples: list[str]


class KnowledgeOut(BaseModel):
    active_version: str
    draft_version: str
    categories: list[CategoryOut]
    edits: list[EditOut]
    options: dict[str, Any]


def _editor(request: Request) -> KnowledgeEditor:
    editor = request.app.state.services.editor
    if editor is None:
        raise HTTPException(status_code=503, detail="Редактор знаний не подключён")
    return editor


@router.get("/api/knowledge")
def knowledge(request: Request) -> KnowledgeOut:
    editor = _editor(request)
    draft, usage = editor.draft(), editor.usage()
    return KnowledgeOut(
        active_version=draft.active.version,
        draft_version=draft.knowledge.version,
        categories=[_category(c, usage.get(c.code, Usage(rows=0, samples=[]))) for c in draft.knowledge.categories],
        edits=[_edit(e) for e in draft.edits],
        options={
            "level": list(Level),
            "comparison": list(Comparison),
            "status": list(AttributeStatus),
            "max_tolerance": editing.MAX_TOLERANCE,
        },
    )


@router.get("/api/knowledge/suggestions")
def suggestions(request: Request) -> list[Disagreement]:
    return _editor(request).disagreements()


@router.post("/api/knowledge/edits")
def edit(request: Request, body: EditRequest) -> EditOut | None:
    made = _editor(request).edit(
        editing.Edit(
            category=body.category,
            attribute=body.attribute,
            field=body.field,
            value=body.value,
            author=body.author,
            reason=body.reason,
        )
    )
    return _edit(made) if made else None


@router.delete("/api/knowledge/edits/{edit_id}", status_code=204)
def discard(request: Request, edit_id: int) -> Response:
    if not _editor(request).discard(edit_id):
        raise HTTPException(status_code=404, detail="Нет такой правки в черновике")
    return Response(status_code=204)


@router.post("/api/knowledge/check")
def check(request: Request) -> Impact:
    return _editor(request).check()


@router.post("/api/knowledge/publish")
def publish(request: Request, body: PublishRequest) -> dict[str, Any]:
    state = request.app.state
    published = _editor(request).publish(body.author, body.note)
    state.services = state.build()  # the search over the new release
    return {**published, "cards": state.services.cards.rekey()}


@router.get("/knowledge", include_in_schema=False)
def page() -> FileResponse:
    return FileResponse(PAGE)


def _category(category: Category, usage: Usage) -> CategoryOut:
    return CategoryOut(
        code=category.code,
        name=category.name,
        group=category.group,
        triggers=list(category.triggers),
        anti_triggers=list(category.anti_triggers),
        synonyms=list(category.synonyms),
        attributes=[_attribute(a) for a in category.attributes],
        rows=usage.rows,
        samples=usage.samples,
    )


def _attribute(attribute: Attribute) -> AttributeOut:
    rule = attribute.rule
    return AttributeOut(
        code=attribute.code,
        name=attribute.name,
        unit=attribute.unit,
        level=attribute.level,
        comparison=attribute.comparison,
        status=attribute.status,
        provenance=attribute.provenance,
        source_ref=attribute.source_ref,
        step_tolerance=rule.get(editing.Field.STEP_TOLERANCE.value, 1),
        substitution=rule.get(editing.Field.SUBSTITUTION.value),
        series=rule.get("series"),
    )


def _edit(edit: editing.Edit) -> EditOut:
    return EditOut(
        id=edit.id,
        category=edit.category,
        attribute=edit.attribute,
        field=edit.field,
        value=edit.value,
        previous=edit.previous,
        author=edit.author,
        reason=edit.reason,
        created_at=edit.created_at,
    )
