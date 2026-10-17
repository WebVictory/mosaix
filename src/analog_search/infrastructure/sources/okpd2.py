"""ОКПД2 (ОК 034-2014), official text from Росстат: OKPD2.rar with two docx («01-35», «36-99»).

The tables hold code / name; the name cell's later paragraphs are «Эта группировка включает…».
RAR is unpacked with bsdtar (libarchive: macOS base system, `libarchive-tools` on Debian).
"""

from __future__ import annotations

import re
import subprocess
import zipfile
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from xml.etree import ElementTree as ET

from analog_search.domain.sources import Okpd2Node, Release, SourceKind
from analog_search.infrastructure.sources.store import Downloader, release_dir, sha256_of, write_manifest

URL = "https://rosstat.gov.ru/storage/mediabank/OKPD2.rar"
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

CODE_RE = re.compile(r"^\d\d(?:\.\d{1,3}){0,4}$")
SECTION_RE = re.compile(r"^РАЗДЕЛ\s+([A-Z])$")


def fetch(downloader: Downloader, root: Path) -> Path:
    """Download the archive; version = its Last-Modified date (Росстат republishes in place)."""
    response = downloader.get(URL)
    modified = response.headers.get("last-modified")
    stamp = parsedate_to_datetime(modified) if modified else datetime.now(UTC)
    directory = release_dir(root, SourceKind.OKPD2, stamp.date().isoformat())
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "OKPD2.rar"
    archive.write_bytes(response.content)
    subprocess.run(["bsdtar", "-xf", str(archive), "-C", str(directory)], check=True)
    files = sorted(p for p in directory.glob("*.docx"))
    write_manifest(
        directory,
        Release(SourceKind.OKPD2, directory.name, stamp.date(), URL, tuple(p.name for p in files), sha256_of(files)),
    )
    return directory


def level(code: str) -> int:
    """Section «C» 0, class «27» 1, subclass «27.3» 2, group «27.32» 3, subgroup «27.32.1» 4,
    kind «27.32.13» 5, category «27.32.13.140» 6, subcategory «27.32.13.143» 7."""
    if not code[0].isdigit():
        return 0
    parts = code.split(".")
    if len(parts) == 4:
        return 6 if parts[3].endswith("0") else 7
    if len(parts) == 1:
        return 1
    # Two-digit last group is the full level, one digit the intermediate one
    return 2 * len(parts) - 1 - (len(parts[-1]) == 1)


def parent_candidates(code: str) -> list[str]:
    """Possible parents, nearest first: «27.32.13.143» → 27.32.13.140, 27.32.13."""
    if "." not in code:
        return []
    head, last = code.rsplit(".", 1)
    if len(last) == 3:
        return ([f"{head}.{last[:2]}0"] if not last.endswith("0") else []) + [head]
    if len(last) == 2:
        return [f"{head}.{last[0]}", head]
    return [head]


def _cell_paragraphs(cell: ET.Element) -> list[str]:
    paragraphs = ("".join(t.text or "" for t in p.iter(f"{W}t")) for p in cell.iter(f"{W}p"))
    return [" ".join(p.split()) for p in paragraphs if p.strip()]


def _rows(docx: Path):
    root = ET.fromstring(zipfile.ZipFile(docx).read("word/document.xml"))
    for row in root.iter(f"{W}tr"):
        cells = [_cell_paragraphs(c) for c in row.findall(f"{W}tc")]
        if len(cells) == 2 and cells[0] and cells[1]:
            yield " ".join(cells[0]).replace("\xa0", " "), cells[1]


def parse(directory: Path) -> list[Okpd2Node]:
    rows: dict[str, tuple[str | None, str, str | None]] = {}  # code → (section, name, description)
    section: str | None = None
    for docx in sorted(directory.glob("*.docx")):
        for code, paragraphs in _rows(docx):
            name, description = paragraphs[0], "\n".join(paragraphs[1:]) or None
            if match := SECTION_RE.match(code):
                section = match[1]
                rows[section] = (None, name, description)
            elif CODE_RE.match(code) and section:
                rows[code] = (section, name, description)
    nodes = []
    for code, (section, name, description) in rows.items():
        parent = next((c for c in parent_candidates(code) if c in rows), section)
        nodes.append(Okpd2Node(code, parent, level(code), name, description))
    return nodes
