"""What a query being typed may become: its last word finished, the names of the history it is the start of,
the categories it names.

- The last word is finished with a word of the history that starts with it: of those that follow the previous
  word in the names (when any do), the one that follows it most often, then the most frequent —
  «кабель конт» → «кабель контрольный», «ворота отк» → «ворота откатные» only.
- A name is offered by how much of the query it holds (the share of the query's trigrams found in it, so a
  long name is not penalized for its length, unlike the similarity of the search), and a name that starts
  with the query comes first: «свет» → «Светильник светодиодный…» before «Оповещатель световой».
- A category is offered when the query names it — a trigger word, a synonym or its name, as the
  classifier reads it — or when the word being typed is the start of one.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise

from analog_search.domain.knowledge import Category, normalize
from analog_search.domain.spelling import MIN_LENGTH, cased, folded, words
from analog_search.domain.trigrams import trigrams

MIN_PREFIX = 2  # letters of a word before it is finished
MIN_SHARE = 0.4  # of the query's trigrams a name must hold to be offered


@dataclass(frozen=True)
class _Name:
    name: str
    text: str  # normalized
    grams: frozenset[str]


class Completer:
    def __init__(self, names: Iterable[str], categories: Iterable[Category]):
        """names — of the history rows; categories — those a query may be of."""
        self._names = [_Name(n, text, trigrams(text)) for n in dict.fromkeys(names) if (text := normalize(n))]
        self._counts: Counter[str] = Counter()
        self._follows: Counter[tuple[str, str]] = Counter()
        for name in self._names:
            found = words(name.text)
            self._counts.update(w for w in found if len(w) >= MIN_LENGTH)
            self._follows.update(pairwise(found))
        self._labels = [
            (label, category)
            for category in categories
            for label in dict.fromkeys(folded(s) for s in (category.name, *category.synonyms, *category.triggers))
        ]

    def words(self, query: str, limit: int) -> list[str]:
        """The query with its last word finished, if it is being typed."""
        found = re.search(r"([^\W\d_]+)$", query)
        if found is None or len(found[1]) < MIN_PREFIX:
            return []
        prefix = folded(found[1])
        previous = words(folded(query[: found.start()]))[-1:]
        started = [w for w in self._counts if w.startswith(prefix) and w != prefix]
        following = [w for w in started if previous and self._follows[(previous[0], w)]]
        ranked = sorted(
            following or started,
            key=lambda w: (-self._follows[(previous[0], w)] if previous else 0, -self._counts[w], w),
        )
        return [query[: found.start()] + cased(w, found[1]) for w in ranked[:limit]]

    def names(self, query: str, limit: int) -> list[str]:
        """The names of the history that hold the query, those starting with it first."""
        text = normalize(query)
        grams = trigrams(text)
        if not grams:
            return []
        scored = [
            (not name.text.startswith(text), -share, len(name.name), name.name)
            for name in self._names
            if (share := len(grams & name.grams) / len(grams)) >= MIN_SHARE
        ]
        return [name for *_, name in sorted(scored)[:limit]]

    def categories(self, query: str, limit: int) -> list[Category]:
        """The categories whose name the query is the start of — being typed, so first — then those it names,
        the longest label named first: «кабель конт» → кабель контрольный, then кабель (прочий)."""
        text = folded(normalize(query))
        started, named = [], []
        for label, category in self._labels:
            if label.startswith(text):
                started.append(category)
            elif re.search(r"(?<![а-яa-z0-9])" + re.escape(label), text):
                named.append((-len(label), category))
        ranked = [*started, *(category for _, category in sorted(named, key=lambda n: n[0]))]
        return list(dict.fromkeys(ranked))[:limit]
