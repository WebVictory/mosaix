"""Each taxonomy behind one interface (domain.sources.SourceFormat): the service never asks which one it holds."""

from __future__ import annotations

from pathlib import Path

from domain.sources import Release, SourceFormat, SourceKind, SourceStore
from infrastructure.sources import etim, ksr, ktru, okpd2
from infrastructure.sources.store import Downloader


class _Fetched:
    def __init__(self, root: Path, downloader: Downloader | None):
        self._root = root
        self._downloader = downloader

    def _client(self) -> Downloader:
        if self._downloader is None:
            raise RuntimeError("fetch needs a downloader")
        return self._downloader


class Ksr(_Fetched):
    def fetch(self) -> Path:
        client = self._client()
        return ksr.fetch_archive(client, self._root, ksr.list_archives(client)[0])

    def load(self, directory: Path, release: Release, store: SourceStore) -> dict[str, int]:
        return store.save_ksr(release, ksr.parse(directory / release.files[0]))


class Okpd2(_Fetched):
    def fetch(self) -> Path:
        return okpd2.fetch(self._client(), self._root)

    def load(self, directory: Path, release: Release, store: SourceStore) -> dict[str, int]:
        return store.save_okpd2(release, okpd2.parse(directory))


class Etim(_Fetched):
    def fetch(self) -> Path:
        return etim.fetch(self._client(), self._root)

    def load(self, directory: Path, release: Release, store: SourceStore) -> dict[str, int]:
        return store.save_etim(release, etim.parse(directory))


class Ktru:
    def fetch(self) -> Path:
        raise ValueError("КТРУ has no anonymous download: fetch it by hand and use `register`")

    def load(self, directory: Path, release: Release, store: SourceStore) -> dict[str, int]:
        return store.save_ktru(release, ktru.parse(directory / release.files[0]))


def formats(root: Path, downloader: Downloader | None = None) -> dict[SourceKind, SourceFormat]:
    return {
        SourceKind.KSR: Ksr(root, downloader),
        SourceKind.OKPD2: Okpd2(root, downloader),
        SourceKind.ETIM: Etim(root, downloader),
        SourceKind.KTRU: Ktru(),
    }
