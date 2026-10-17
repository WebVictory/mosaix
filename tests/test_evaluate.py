"""Metrics of the evaluation report: what counts as a hit, a misplaced row and a false «found»."""

from __future__ import annotations

import json
import sys

from analog_search.app.config import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))  # the evaluator is a script, not a package

from evaluate import Answer, Gold, metrics

GOLD = [
    Gold("a", "task", "кабель 7х1,5", "match", [1, 2], key="сечение_жилы"),
    Gold("b", "task", "ворота", "match", [5]),
    Gold("c", "task", "трансформатор 1000", "no_match", []),
    Gold("d", "task", "насос", "no_match", []),
]


def test_metrics() -> None:
    answers = {
        "a": Answer([3, 1], "other section first"),
        "b": Answer([5], "gate"),
        "c": Answer([], "hard_mismatch"),
        "d": Answer([9], "pump-like row"),
    }

    assert metrics(GOLD, answers) == {
        "hit@1": 0.5,
        "hit@5": 1.0,
        "misplace@1": 1.0,
        "no_match_ok": 0.5,
        "false_found": 0.5,
    }


def test_gold_set_is_well_formed() -> None:
    gold = [Gold(**g) for g in json.loads((PROJECT_ROOT / "evaluation" / "gold_queries.json").read_text("utf-8"))]

    assert len({g.id for g in gold}) == len(gold)
    for g in gold:
        assert g.expected_kind in {"match", "no_match"}
        assert bool(g.expected_ids) is (g.expected_kind == "match"), g.id
