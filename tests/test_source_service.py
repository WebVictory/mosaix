"""Release bookkeeping: register a hand-downloaded file, load it once, reload on change."""

from __future__ import annotations

from pathlib import Path

from domain.sources import Edition, KtruPosition, Release, SourceKind
from infrastructure.sources.formats import formats
from infrastructure.sources.store import ReleaseFiles, read_manifest
from services.sources import SourceService
from tests.test_sources import KTRU_XML


class FakeStore:
    def __init__(self) -> None:
        self.loaded: dict[tuple[SourceKind, str], str] = {}

    def is_loaded(self, release: Release) -> bool:
        return self.loaded.get((release.source, release.version)) == release.sha256

    def save_ktru(self, release: Release, positions: list[KtruPosition]) -> dict[str, int]:
        self.loaded[release.source, release.version] = release.sha256
        return {"ktru_positions": len(positions)}


def test_register_then_load_once(tmp_path: Path) -> None:
    file = tmp_path / "download" / "ktru.xml"
    file.parent.mkdir()
    file.write_text(KTRU_XML, encoding="utf-8")
    store = FakeStore()
    root = tmp_path / "sources"
    service = SourceService(ReleaseFiles(root), formats(root), store=store)

    edition = Edition(source=SourceKind.KTRU, version="2026-10-01", url="manual")
    directory = service.register(edition, file)
    release = read_manifest(directory)

    assert directory == tmp_path / "sources" / "ktru" / "2026-10-01"
    assert (release.files, release.url) == (("ktru.xml",), "manual")
    assert service.load(SourceKind.KTRU) == {"ktru_positions": 1}
    assert service.load(SourceKind.KTRU) is None
    assert service.load(SourceKind.KTRU, force=True) == {"ktru_positions": 1}

    # Same version, new file content → new sha256 → loaded again
    file.write_text(KTRU_XML.replace("Кабель", "Провод"), encoding="utf-8")
    service.register(edition, file)
    assert service.load(SourceKind.KTRU) == {"ktru_positions": 1}
