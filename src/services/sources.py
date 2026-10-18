"""Fetch external taxonomies to disk and load them into the DB.

Fetch is separate from load: files are kept with a manifest, so a load can be repeated
offline and the DB always says which file version it holds.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from domain.sources import Edition, Release, SourceFormat, SourceKind, SourceStore


class ReleaseArchive(Protocol):
    """Release directories with their manifests."""

    def latest(self, source: SourceKind) -> Path | None:
        """The most recently fetched release of a source."""
        ...

    def release(self, directory: Path) -> Release: ...

    def register(self, edition: Edition, file: Path) -> Path:
        """A file fetched by hand made a release: copied into its directory, a manifest written next to it."""
        ...


class SourceService:
    def __init__(
        self, archive: ReleaseArchive, formats: Mapping[SourceKind, SourceFormat], store: SourceStore | None = None
    ):
        self._archive = archive
        self._formats = formats
        self._store = store

    def fetch(self, source: SourceKind) -> Path:
        """Download the newest freely available release; КТРУ has none (see `register`)."""
        return self._formats[source].fetch()

    def register(self, edition: Edition, file: Path) -> Path:
        """A file fetched by hand (КСР current version behind a captcha, КТРУ export) becomes a release."""
        return self._archive.register(edition, file)

    def load(self, source: SourceKind, directory: Path | None = None, force: bool = False) -> dict[str, int] | None:
        """Parse a release directory (the latest fetched by default) and save it; None — already loaded."""
        if self._store is None:
            raise RuntimeError("load needs a store")
        directory = directory or self._archive.latest(source)
        if directory is None:
            raise FileNotFoundError(f"no fetched release of {source.value}")
        release = self._archive.release(directory)
        if not force and self._store.is_loaded(release):
            return None
        return self._formats[source].load(directory, release, self._store)
