"""The base of the value objects the layers pass to each other: validated once, never changed.

A changed copy is `model_copy(update=...)`; an unknown field is an error, not silently dropped.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
