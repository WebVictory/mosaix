"""Release directory: <root>/<source>/<version>/ with the files and manifest.json.

The manifest records where and when the files came from, so a load is reproducible.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import ssl
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import truststore

from analog_search.domain.sources import Edition, Release, SourceKind

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
    data = release.model_dump(mode="json") | {"fetched_at": datetime.now(UTC).isoformat(timespec="seconds")}
    (directory / MANIFEST).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_manifest(directory: Path) -> Release:
    data = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    data.pop("fetched_at", None)
    return Release.model_validate(data)


def latest_release(root: Path, source: SourceKind) -> Path | None:
    """Most recently fetched release directory of a source."""
    candidates = [d for d in (root / source.value).glob("*") if (d / MANIFEST).is_file()]
    return max(candidates, key=lambda d: (d / MANIFEST).stat().st_mtime, default=None)


class ReleaseFiles:
    """The release directories under one root."""

    def __init__(self, root: Path):
        self._root = root

    def latest(self, source: SourceKind) -> Path | None:
        return latest_release(self._root, source)

    def release(self, directory: Path) -> Release:
        return read_manifest(directory)

    def register(self, edition: Edition, file: Path) -> Path:
        directory = release_dir(self._root, edition.source, edition.version)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / file.name
        if file.resolve() != target.resolve():
            shutil.copyfile(file, target)
        write_manifest(directory, Release(**edition.model_dump(), files=(target.name,), sha256=sha256_of([target])))
        return directory


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
