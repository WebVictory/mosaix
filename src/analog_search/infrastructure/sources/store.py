"""Release directory: <root>/<source>/<version>/ with the files and manifest.json.

The manifest records where and when the files came from, so a load is reproducible.
"""

from __future__ import annotations

import hashlib
import json
import ssl
import time
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import truststore

from analog_search.domain.sources import Release, SourceKind

MANIFEST = "manifest.json"
USER_AGENT = "analog-search/0.1 (taxonomy loader)"


def release_dir(root: Path, source: SourceKind, version: str) -> Path:
    return root / source.value / version


def sha256_of(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_manifest(directory: Path, release: Release) -> None:
    data = asdict(release) | {"fetched_at": datetime.now(UTC).isoformat(timespec="seconds")}
    data["published"] = release.published.isoformat() if release.published else None
    (directory / MANIFEST).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_manifest(directory: Path) -> Release:
    data = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    return Release(
        source=SourceKind(data["source"]),
        version=data["version"],
        published=date.fromisoformat(data["published"]) if data["published"] else None,
        url=data["url"],
        files=tuple(data["files"]),
        sha256=data["sha256"],
    )


def latest_release(root: Path, source: SourceKind) -> Path | None:
    """Most recently fetched release directory of a source."""
    candidates = [d for d in (root / source.value).glob("*") if (d / MANIFEST).is_file()]
    return max(candidates, key=lambda d: (d / MANIFEST).stat().st_mtime, default=None)


class Downloader:
    """Plain GET with a pause between requests: public state sites, one client per run.

    TLS is verified against the OS trust store: rosstat.gov.ru is signed by the Russian
    Минцифры root CA, which is not in certifi (on Linux install it into the system store).
    """

    def __init__(self, timeout_s: float, delay_s: float):
        self._client = httpx.Client(
            timeout=timeout_s,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
            verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
        )
        self._delay_s = delay_s
        self._last = 0.0

    def get(self, url: str, **params: str) -> httpx.Response:
        wait = self._last + self._delay_s - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        response = self._client.get(url, params=params or None)
        self._last = time.monotonic()
        response.raise_for_status()
        return response

    def save(self, url: str, target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.get(url).content)
        return target

    def close(self) -> None:
        self._client.close()
