"""Typing mistakes in a query, fixed against the words of the base before the search.

The trigger words of a category match from a word start: «кабль силовой» finds no category, and
«rf,tkm» (typed with the English layout on) finds nothing at all. A word unknown to the vocabulary — the
names of the history rows and the words of the knowledge — is replaced by a known word one or two edits
away (Damerau–Levenshtein: a letter missed, added, replaced or two swapped), the most frequent on a tie,
or by itself retyped in the Russian layout. A word is known in any of its forms: «порошковый» is not
mistyped because the base only has «порошковая» — the stems (the word without its ending) are compared.
Words with digits (sizes, marks like «3х2,5», «ВВГнг-LS»),
short words and the word still being typed, when a known word starts with it, are left as they are.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

from domain.frozen import Frozen

MIN_LENGTH = 4  # shorter words are marks and units: «ПВХ», «кВА», «IP»

_TOKEN = re.compile(r"\S+")
_CORE = re.compile(r"(\W*)([^\W\d_]+)(\W*)")  # a word in quotes, before a comma
_LETTERS = re.compile(r"[^\W\d_]+")
_LATIN = "qwertyuiop[]asdfghjkl;'zxcvbnm,."
_CYRILLIC = "йцукенгшщзхъфывапролджэячсмитьбю"
_LAYOUT = str.maketrans(_LATIN, _CYRILLIC)
# Endings of Russian nouns and adjectives, the longest first
_ENDING = re.compile(
    r"(?:ыми|ими|ами|ями|ого|его|ому|ему|ая|яя|ое|ее|ые|ие|ый|ий|ой|ую|юю|ых|их|ым|им|ам|ям|ах|ях|ом|ем|ов|ев|ей"
    r"|а|я|о|е|ы|и|у|ю)$"
)


class Fix(Frozen):
    """A word of the query and what it was taken for."""

    typed: str
    meant: str


class Spelled(Frozen):
    text: str  # the query with the fixes
    fixes: tuple[Fix, ...]


class Speller:
    def __init__(self, texts: Iterable[str]):
        """texts — whatever the base is written in: the names of the rows, the trigger words, the category names."""
        self._counts = Counter(w for text in texts for w in words(folded(text)) if len(w) >= MIN_LENGTH)
        self._by_length: dict[int, list[str]] = {}
        for word in self._counts:
            self._by_length.setdefault(len(word), []).append(word)
        self._stems = {s for word in self._counts if (s := _stem(word))}

    def spell(self, query: str, typing: bool = False) -> Spelled:
        """The query with its unknown words fixed; typing — the last word may be unfinished."""
        tokens = list(_TOKEN.finditer(query))
        fixes, parts, end = [], [], 0
        for i, token in enumerate(tokens):
            parts.append(query[end : token.start()])
            unfinished = typing and i == len(tokens) - 1 and token.end() == len(query)
            fixed = None if unfinished and self._started(token[0]) else self._fixed(token[0])
            if fixed is not None:
                fixes.append(Fix(typed=token[0], meant=fixed))
            parts.append(fixed or token[0])
            end = token.end()
        parts.append(query[end:])
        return Spelled(text="".join(parts), fixes=tuple(fixes))

    def _fixed(self, token: str) -> str | None:
        """The token retyped in the Russian layout, or its word replaced by the nearest known one."""
        retyped = folded(token).translate(_LAYOUT)
        if not _LETTERS.fullmatch(folded(token)) and _LETTERS.fullmatch(retyped) and retyped in self._counts:
            return cased(retyped, token)
        if (core := _CORE.fullmatch(token)) is None:
            return None  # a size, a mark: «3х2,5», «ВВГнг-LS»
        before, word, after = core.groups()
        meant = self._meant(word)
        return f"{before}{cased(meant, word)}{after}" if meant else None

    def _meant(self, word: str) -> str | None:
        plain = folded(word)
        if plain in self._counts or _stem(plain) in self._stems:
            return None
        retyped = plain.translate(_LAYOUT)
        if retyped in self._counts:
            return retyped
        if len(plain) < MIN_LENGTH:
            return None
        limit = 1 if len(plain) < 7 else 2
        near = [
            (distance, -self._counts[known], known)
            for length in range(len(plain) - limit, len(plain) + limit + 1)
            for known in self._by_length.get(length, ())
            if (distance := _distance(plain, known, limit)) <= limit
        ]
        return min(near)[2] if near else None

    def _started(self, word: str) -> bool:
        """A known word starts with it: the word is being typed, not mistyped."""
        plain = folded(word)
        return any(known.startswith(plain) for known in self._counts)


def words(text: str) -> list[str]:
    """The words of a text: runs of letters."""
    return _LETTERS.findall(text)


def folded(text: str) -> str:
    return text.lower().replace("ё", "е")


def _stem(word: str) -> str | None:
    """The word without its ending, if it has one and the rest is long enough to say what the word is."""
    stem = _ENDING.sub("", word)
    return stem if stem != word and len(stem) >= MIN_LENGTH else None


def cased(meant: str, typed: str) -> str:
    if typed.isupper() and len(typed) > 1:
        return meant.upper()
    return meant.capitalize() if typed[:1].isupper() else meant


def _distance(a: str, b: str, limit: int) -> int:
    """Damerau–Levenshtein (optimal string alignment); anything above the limit is limit + 1."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    previous, row = None, list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            current[j] = min(row[j] + 1, current[j - 1] + 1, row[j - 1] + (ca != cb))
            if previous is not None and i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                current[j] = min(current[j], previous[j - 2] + 1)
        if min(current) > limit:
            return limit + 1
        previous, row = row, current
    return row[-1]
