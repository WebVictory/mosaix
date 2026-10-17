"""Fetch external taxonomies to disk and load them into the DB.

Fetch is separate from load: files are kept with a manifest, so a load can be repeated
offline and the DB always says which file version it holds.
"""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path
from typing import Protocol

from analog_search.domain.sources import EtimDump, KsrDump, KtruPosition, Okpd2Node, Release, SourceKind
from analog_search.infrastructure.sources import etim, ksr, ktru, okpd2
from analog_search.infrastructure.sources.store import (
    Downloader,
    latest_release,
    read_manifest,
    release_dir,
    sha256_of,
    write_manifest,
)


class SourceStore(Protocol):
    def is_loaded(self, release: Release) -> bool: ...
    def save_ksr(self, release: Release, dump: KsrDump) -> dict[str, int]: ...
    def save_okpd2(self, release: Release, nodes: list[Okpd2Node]) -> dict[str, int]: ...
    def save_ktru(self, release: Release, positions: list[KtruPosition]) -> dict[str, int]: ...
    def save_etim(self, release: Release, dump: EtimDump) -> dict[str, int]: ...


class SourceService:
    def __init__(self, root: Path, downloader: Downloader | None = None, store: SourceStore | None = None):
        self._root = root
        self._downloader = downloader
        self._store = store

    # --- fetch ---

    def fetch(self, source: SourceKind) -> Path:
        """Download the newest freely available release; КТРУ has none (see `register`)."""
        if self._downloader is None:
            raise RuntimeError("fetch needs a downloader")
        match source:
            case SourceKind.KSR:
                return ksr.fetch_archive(self._downloader, self._root, ksr.list_archives(self._downloader)[0])
            case SourceKind.OKPD2:
                return okpd2.fetch(self._downloader, self._root)
            case SourceKind.ETIM:
                return etim.fetch(self._downloader, self._root)
            case SourceKind.KTRU:
                raise ValueError("КТРУ has no anonymous download: fetch it by hand and use `register`")

    def register(self, source: SourceKind, file: Path, version: str, published: date | None, url: str | None) -> Path:
        """A file fetched by hand (КСР current version behind a captcha, КТРУ export) becomes a release."""
        directory = release_dir(self._root, source, version)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / file.name
        if file.resolve() != target.resolve():
            shutil.copyfile(file, target)
        write_manifest(directory, Release(source, version, published, url, (target.name,), sha256_of([target])))
        return directory

    # --- load ---

    def load(self, source: SourceKind, directory: Path | None = None, force: bool = False) -> dict[str, int] | None:
        """Parse a release directory (the latest fetched by default) and save it; None — already loaded."""
        if self._store is None:
            raise RuntimeError("load needs a store")
        directory = directory or latest_release(self._root, source)
        if directory is None:
            raise FileNotFoundError(f"no fetched release of {source.value} under {self._root}")
        release = read_manifest(directory)
        if not force and self._store.is_loaded(release):
            return None
        match source:
            case SourceKind.KSR:
                return self._store.save_ksr(release, ksr.parse(directory / release.files[0]))
            case SourceKind.OKPD2:
                return self._store.save_okpd2(release, okpd2.parse(directory))
            case SourceKind.KTRU:
                return self._store.save_ktru(release, ktru.parse(directory / release.files[0]))
            case SourceKind.ETIM:
                return self._store.save_etim(release, etim.parse(directory))
