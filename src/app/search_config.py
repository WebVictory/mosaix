"""The search's rules, all in one place (see ARCHITECTURE.md, «Шаги»). A change here needs no other code.

score = text_weight · text similarity: by spelling (trigrams) or by meaning (embeddings), the higher
      + features_weight · share of the query's attributes the candidate satisfies
      + category_weight · (same category as the query)
      − soft_mismatch_penalty · soft mismatches − missing_penalty · hard attributes not stated
A hard mismatch excludes the candidate whatever the score.
"""

from domain.matching import Kind, Scoring
from domain.rules import Candidates, Confidence, Gate, QueryLimits, SearchRules

SEARCH = SearchRules(
    scoring=Scoring(
        text_weight=0.35,
        features_weight=0.45,
        category_weight=0.2,
        soft_share=0.5,  # a soft attribute weighs this much of a hard one in the features share
        no_features_credit=0.5,  # features share when the query states no attribute of its category
        inexact_credit=0.75,  # a match with a margin (IP65 for IP54) counts this much of an identical one
        meaning_floor=0.5,  # bge-m3 cosine of unrelated names is ~0.3–0.4: meaning counts from 0.5 up to 1
        soft_mismatch_penalty=0.1,
        missing_penalty=0.05,
    ),
    candidates=Candidates(
        nearest_k=20,  # second pass over the whole base: 20 by spelling and 20 by meaning
        judge_k=10,
    ),
    gate=Gate(
        threshold=0.5,
        top_k=5,
        order=(Kind.SAME, Kind.ANALOG, Kind.TEXT, Kind.UNVERIFIED),
    ),
    confidence=Confidence(
        min=0.9,
        stated=0.8,  # chosen after looking at 105 answers between 0.8 and 0.9
    ),
    query=QueryLimits(
        max_length=500,
        min_suggested=3,
        suggested=6,
        completed=3,
    ),
)
