"""Jev Choice via the OpenRouter decisions API: one request per name, one question per attribute.

The key comes from the caller (Settings reads OPENROUTER_API_KEY from the environment) and is never logged.
"""

from __future__ import annotations

import time

import httpx
from pydantic import BaseModel, ValidationError

from domain.reading import Choice, Question, ReaderError

ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
RETRYABLE = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class _Answer(BaseModel):
    choice: str
    probabilities: dict[str, float] | None = None
    confidence: float | None = None


class _Response(BaseModel):
    answers: dict[str, _Answer]


class JevReader:
    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_s: float = 30.0,
        max_retries: int = 2,
        transport: httpx.BaseTransport | None = None,
    ):
        if not api_key:
            raise ReaderError("не задан OPENROUTER_API_KEY")
        self.model = model
        self._max_retries = max_retries
        self._http = httpx.Client(
            timeout=timeout_s, transport=transport, headers={"Authorization": f"Bearer {api_key}"}
        )

    def close(self) -> None:
        self._http.close()

    def choose(self, state: str, questions: list[Question]) -> dict[str, Choice]:
        payload = {
            "model": self.model,
            "state": state,
            "questions": {
                q.code: {"type": "choice", "instructions": q.instructions, "criteria": q.options} for q in questions
            },
        }
        try:
            response = _Response.model_validate(self._post(payload))
        except ValidationError as error:
            raise ReaderError(f"Jev: неожиданный формат ответа: {error.errors()[:1]}") from error
        options = {q.code: q.options for q in questions}
        choices = {}
        for code, answer in response.answers.items():
            if code not in options or answer.choice not in options[code]:
                raise ReaderError(f"Jev: ответ вне вопроса: {code}={answer.choice!r}")
            confidence = answer.confidence
            if confidence is None:
                confidence = (answer.probabilities or {}).get(answer.choice, 0.0)
            choices[code] = Choice(answer.choice, confidence)
        return choices

    def _post(self, payload: dict) -> dict:
        error: str = ""
        for attempt in range(self._max_retries + 1):
            if attempt:
                time.sleep(2**attempt)
            try:
                response = self._http.post(ENDPOINT, json=payload)
            except httpx.TransportError as exc:
                error = type(exc).__name__
                continue
            if response.status_code in RETRYABLE:
                error = f"HTTP {response.status_code}"
                continue
            if response.status_code >= 400:
                raise ReaderError(f"Jev: HTTP {response.status_code}: {response.text[:200]}")
            try:
                return response.json()
            except ValueError as exc:
                raise ReaderError("Jev: ответ не JSON") from exc
        raise ReaderError(f"Jev недоступен после {self._max_retries + 1} попыток: {error}")
