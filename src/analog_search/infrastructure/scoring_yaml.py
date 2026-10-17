"""config/search.yaml → ScoringConfig: weights and the decision gate, tuned without touching code."""

from __future__ import annotations

from pathlib import Path

import yaml

from analog_search.domain.matching import ScoringConfig

SECTIONS = ("scoring", "gate")


def read(path: Path) -> ScoringConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    unknown = set(data) - set(SECTIONS)
    if unknown:
        raise ValueError(f"{path}: unknown sections {sorted(unknown)}")
    return ScoringConfig(**{key: value for section in SECTIONS for key, value in (data.get(section) or {}).items()})
