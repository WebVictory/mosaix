"""The rules a search goes by, in named sections. The values are set in one place, app/search_config.py;
the code reads them from here by name and keeps no numbers of its own."""

from __future__ import annotations

from domain.frozen import Frozen
from domain.matching import Kind, Scoring


class Candidates(Frozen):
    """How many rows the search takes beyond the query's category, and how many the model checks."""

    nearest_k: int  # the whole base by spelling, and as many by meaning
    judge_k: int  # unresolved candidates the choice model judges


class Gate(Frozen):
    """What is shown as found."""

    threshold: float  # below — «не найдено» with the closest row and the reason
    top_k: int
    order: tuple[Kind, ...]  # how the found are listed: the same position first


class Confidence(Frozen):
    """How sure the choice model must be for its answer to count; a less sure one is no answer."""

    min: float
    stated: float  # a number the text itself confirms needs less


class QueryLimits(Frozen):
    max_length: int
    min_suggested: int  # characters typed before anything is offered
    suggested: int  # names of the history offered while a query is typed
    completed: int  # ways to finish the word being typed, categories offered


class SearchRules(Frozen):
    scoring: Scoring
    candidates: Candidates
    gate: Gate
    confidence: Confidence
    query: QueryLimits
