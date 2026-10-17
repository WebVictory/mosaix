"""Text embeddings: via the OpenRouter embeddings API, kept in a JSON Lines file.

The prototype asks OpenRouter so as not to ship a model; in production the same open model (bge-m3)
runs locally next to the service: only OpenRouterEmbedder is replaced. The key comes from the caller
and is never logged. The file spares repeated requests; a text not kept there with no key to ask is
a ReaderError, and the caller does without embeddings.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import httpx
from pydantic import BaseModel, ValidationError

from analog_search.domain.reading import Embedder, ReaderError

ENDPOINT = "https://openrouter.ai/api/v1/embeddings"
BATCH = 64  # texts per request: the whole base is embedded once, at load


class _Item(BaseModel):
    index: int
    embedding: list[float]


class _Response(BaseModel):
    data: list[_Item]


class OpenRouterEmbedder:
    def __init__(self, api_key: str, model: str, timeout_s: float = 30.0, transport: httpx.BaseTransport | None = None):
        if not api_key:
            raise ReaderError("не задан OPENROUTER_API_KEY")
        self._model = model
        self._http = httpx.Client(
            timeout=timeout_s, transport=transport, headers={"Authorization": f"Bearer {api_key}"}
        )

    def close(self) -> None:
        self._http.close()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [vector for i in range(0, len(texts), BATCH) for vector in self._embed(texts[i : i + BATCH])]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self._http.post(ENDPOINT, json={"model": self._model, "input": texts})
        except httpx.TransportError as error:
            raise ReaderError(f"эмбеддинги недоступны: {type(error).__name__}") from error
        if response.status_code >= 400:
            raise ReaderError(f"эмбеддинги: HTTP {response.status_code}: {response.text[:200]}")
        try:
            items = sorted(_Response.model_validate(response.json()).data, key=lambda item: item.index)
        except (ValueError, ValidationError) as error:
            raise ReaderError("эмбеддинги: неожиданный формат ответа") from error
        if len(items) != len(texts):
            raise ReaderError(f"эмбеддинги: {len(items)} векторов на {len(texts)} текстов")
        return [item.embedding for item in items]


class CachedEmbedder:
    """A line is one text's vector, keyed by the model and the text."""

    def __init__(self, path: Path, model: str, inner: Embedder | None = None):
        self._path = path
        self._model = model
        self._inner = inner
        self._lock = threading.Lock()
        self._vectors: dict[str, list[float]] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self._vectors[entry["key"]] = entry["vector"]

    def embed(self, texts: list[str]) -> list[list[float]]:
        keys = [self._key(text) for text in texts]
        with self._lock:
            missing = list(dict.fromkeys(t for t, k in zip(texts, keys, strict=True) if k not in self._vectors))
        if missing:
            if self._inner is None:
                raise ReaderError(f"эмбеддинги: {len(missing)} текстов нет в {self._path.name}, ключа нет")
            fresh = self._inner.embed(missing)
            with self._lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as file:
                    for text, vector in zip(missing, fresh, strict=True):
                        vector = [round(x, 6) for x in vector]
                        self._vectors[self._key(text)] = vector
                        entry = {"key": self._key(text), "model": self._model, "text": text, "vector": vector}
                        file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        with self._lock:
            return [self._vectors[key] for key in keys]

    def _key(self, text: str) -> str:
        return hashlib.sha256(json.dumps([self._model, text], ensure_ascii=False).encode()).hexdigest()[:24]
