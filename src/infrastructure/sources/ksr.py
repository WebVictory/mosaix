"""КСР (классификатор строительных ресурсов), ФГИС ЦС.

Fetch: the archive list `/api/ksr/archives/` gives every superseded version with a ready xlsx
(`filePath`), downloadable without a captcha. The current version is exported only through
«Скачать Классификатор» behind a captcha — download it by hand (MANUAL_URL) and register it.

Workbook: sheets «Материалы, изд, констр и оборуд» and «Машины и механизмы», columns
code / name / unit. Tree rows repeat the title in all three cells («Группа 21.1.06.09: …»);
a resource row «21.1.06.09-0145» is followed by its position-level codes
«27.32.13.143.21.1.06.09-0145-000» — ОКПД2 prefix + resource + suffix.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import openpyxl

from domain.sources import KsrDump, KsrNode, KsrPosition, KsrResource, Release, SourceKind
from infrastructure.sources.store import Downloader, release_dir, sha256_of, write_manifest

BASE_URL = "https://fgiscs.minstroyrf.ru"
ARCHIVES_URL = f"{BASE_URL}/api/ksr/archives/"
FILE_URL = f"{BASE_URL}/api/values/GetFileContent/"
MANUAL_URL = f"{BASE_URL}/ksr"

LEVELS = {"Книга": "book", "Часть": "part", "Раздел": "section", "Группа": "group"}
NODE_RE = re.compile(r"^(Книга|Часть|Раздел|Группа)\s+([\d.]+):\s*(.+)$")
# 01.1.01.01-0002 (materials), 91.01.01-014 (machines)
RESOURCE_RE = re.compile(r"^\d\d(?:\.\d+){2,3}-\d{3,4}$")
POSITION_RE = re.compile(r"^(?P<okpd2>\d\d(?:\.\d+){1,3})\.(?P<resource>\d\d(?:\.\d+){2,3}-\d{3,4})-\d{3}$")


def list_archives(downloader: Downloader) -> list[dict]:
    """Superseded versions, newest first: id, name «Версия №43», publishedDate, filePath."""
    return downloader.get(ARCHIVES_URL).json()["items"]


def fetch_archive(downloader: Downloader, root: Path, archive: dict) -> Path:
    """Download one archived version's xlsx into its release directory."""
    version = archive["name"].removeprefix("Версия №").strip()
    directory = release_dir(root, SourceKind.KSR, version)
    target = downloader.save(FILE_URL + archive["filePath"], directory / f"ksr_v{version}.xlsx")
    published = datetime.strptime(archive["publishedDate"], "%d.%m.%Y").date()
    write_manifest(
        directory,
        Release(
            source=SourceKind.KSR,
            version=version,
            published=published,
            url=FILE_URL + archive["filePath"],
            files=(target.name,),
            sha256=sha256_of([target]),
        ),
    )
    return directory


def _parent(code: str, stack: dict[str, str]) -> str | None:
    """Nearest enclosing node: the longest known code that prefixes this one."""
    parts = code.split(".")
    for size in range(len(parts) - 1, 0, -1):
        prefix = ".".join(parts[:size])
        if prefix in stack:
            return prefix
    return None


def parse(path: Path) -> KsrDump:
    nodes: list[KsrNode] = []
    resources: dict[str, dict] = {}  # code → fields, in file order
    positions: dict[str, list[KsrPosition]] = {}  # resource code → its position-level codes
    workbook = openpyxl.load_workbook(path, read_only=True)
    try:
        for sheet in workbook.worksheets:
            known: dict[str, str] = {}
            group: str | None = None
            for code, name, unit in sheet.iter_rows(min_row=3, max_col=3, values_only=True):
                code, name, unit = str(code or "").strip(), str(name or "").strip(), str(unit or "").strip() or None
                if node := NODE_RE.match(code):
                    word, group, title = node.groups()
                    known[group] = word
                    nodes.append(KsrNode(group, _parent(group, known), LEVELS[word], title.strip(), sheet.title))
                elif RESOURCE_RE.match(code):
                    resources[code] = {"code": code, "group_code": group, "name": name, "unit": unit}
                elif (position := POSITION_RE.match(code)) and position["resource"] in resources:
                    positions.setdefault(position["resource"], []).append(
                        KsrPosition(code, position["okpd2"], name, unit)
                    )
    finally:
        workbook.close()
    return KsrDump(
        nodes=nodes,
        resources=[KsrResource(**r, positions=tuple(positions.get(c, ()))) for c, r in resources.items()],
    )
