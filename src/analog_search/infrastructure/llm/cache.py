"""The model's answers kept in a JSON Lines file: a re-parse asks only what was not asked yet.

A line is one answer to one question about one name, keyed by everything the answer depends on:
the model, the name with its category, the question with its options. A changed series in
attributes.yaml is a new question, so a stale answer is never reused. Without a model (no key)
the file alone answers: the search runs without external keys on what was read before.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

from analog_search.domain.reading import Choice, ChoiceReader, Question


class CachedReader:
    def __init__(self, path: Path, model: str, inner: ChoiceReader | None = None):
        self._path = path
        self._model = model
        self._inner = inner
        self._lock = threading.Lock()
        self._answers: dict[str, Choice] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self._answers[entry["key"]] = Choice(entry["option"], entry["confidence"])
        self.calls = 0  # requests sent to the model in this run

    def choose(self, state: str, questions: list[Question]) -> dict[str, Choice]:
        keys = {q.code: self._key(state, q) for q in questions}
        with self._lock:
            known = {code: self._answers[key] for code, key in keys.items() if key in self._answers}
        missing = [q for q in questions if q.code not in known]
        if not missing or self._inner is None:
            return known
        fresh = self._inner.choose(state, missing)
        with self._lock:
            self.calls += 1
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as file:
                for question in missing:
                    if (choice := fresh.get(question.code)) is None:
                        continue
                    self._answers[keys[question.code]] = choice
                    entry = {
                        "key": keys[question.code],
                        "model": self._model,
                        "state": state,
                        "attribute": question.code,
                        "option": choice.option,
                        "answer": question.options.get(choice.option),
                        "confidence": choice.confidence,
                    }
                    file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return {**known, **fresh}

    def _key(self, state: str, question: Question) -> str:
        spec = [self._model, state, question.code, question.instructions, sorted(question.options.items())]
        return hashlib.sha256(json.dumps(spec, ensure_ascii=False).encode()).hexdigest()[:24]
