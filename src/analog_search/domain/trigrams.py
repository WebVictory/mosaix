"""Trigram similarity as pg_trgm computes it, for search without the database (evaluation, tests).

A word is a run of letters and digits; it is padded with two spaces in front and one behind;
similarity = shared trigrams / all trigrams of both strings.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"[^\W_]+")


def trigrams(text: str) -> frozenset[str]:
    grams = set()
    for word in _WORD.findall(text.lower()):
        padded = f"  {word} "
        grams.update(padded[i : i + 3] for i in range(len(padded) - 2))
    return frozenset(grams)


def similarity(a: frozenset[str], b: frozenset[str]) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 0.0
