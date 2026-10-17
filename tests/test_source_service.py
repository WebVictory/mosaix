"""Release bookkeeping: register a hand-downloaded file, load it once, reload on change."""

from __future__ import annotations

from pathlib import Path

from analog_search.domain.sources import KtruPosition, Release, SourceKind
from analog_search.infrastructure.sources.store import read_manifest
from analog_search.services.sources import SourceService
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
    service = SourceService(tmp_path / "sources", store=store)

    directory = service.register(SourceKind.KTRU, file, "2026-10-01", None, "manual")
    release = read_manifest(directory)

    assert directory == tmp_path / "sources" / "ktru" / "2026-10-01"
    assert (release.files, release.url) == (("ktru.xml",), "manual")
    assert service.load(SourceKind.KTRU) == {"ktru_positions": 1}
    assert service.load(SourceKind.KTRU) is None
    assert service.load(SourceKind.KTRU, force=True) == {"ktru_positions": 1}

    # Same version, new file content → new sha256 → loaded again
    file.write_text(KTRU_XML.replace("Кабель", "Провод"), encoding="utf-8")
    service.register(SourceKind.KTRU, file, "2026-10-01", None, "manual")
    assert service.load(SourceKind.KTRU) == {"ktru_positions": 1}
