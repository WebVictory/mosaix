"""ETIM classification model, international release (ETIM English, free, no registration).

Release zip of semicolon-separated UTF-16 CSV: ETIMARTGROUP, ETIMARTCLASS, ETIMARTCLASSSYNONYMMAP,
ETIMFEATURE, ETIMFEATUREGROUP, ETIMUNIT, ETIMVALUE, ETIMARTCLASSFEATUREMAP,
ETIMARTCLASSFEATUREVALUEMAP. Russian names are not in the free release (RAEC, the Russian ETIM
office, distributes them), so the model gives the structure, not the wording.
"""

from __future__ import annotations

import csv
import io
import zipfile
from collections.abc import Iterator
from datetime import date
from pathlib import Path

from analog_search.domain.sources import (
    EtimClass,
    EtimClassFeature,
    EtimDump,
    EtimFeature,
    EtimGroup,
    Release,
    SourceKind,
)
from analog_search.infrastructure.sources.store import Downloader, release_dir, sha256_of, write_manifest

# ETIM 10.0, all sectors, metric; the downloads page lists the next releases
URL = "https://www.etim-international.com/wp-content/uploads/2024/12/ETIM-10.0-ALL-SECTORS-CSV-METRIC-EI-2024-12-05.zip"
VERSION = "10.0"
PUBLISHED = date(2024, 12, 5)


def fetch(downloader: Downloader, root: Path, url: str = URL, version: str = VERSION) -> Path:
    directory = release_dir(root, SourceKind.ETIM, version)
    target = downloader.save(url, directory / url.rsplit("/", 1)[-1])
    write_manifest(
        directory,
        Release(
            source=SourceKind.ETIM,
            version=version,
            published=PUBLISHED,
            url=url,
            files=(target.name,),
            sha256=sha256_of([target]),
        ),
    )
    return directory


def _table(archive: zipfile.ZipFile, name: str) -> Iterator[dict[str, str]]:
    member = next(n for n in archive.namelist() if n.rsplit("/", 1)[-1].upper() == f"{name}.CSV")
    text = archive.read(member).decode("utf-16")
    for row in csv.DictReader(io.StringIO(text), delimiter=";"):
        yield {k.strip(): (v or "").strip() for k, v in row.items()}


def parse(directory: Path) -> EtimDump:
    zip_path = next(directory.glob("*.zip"))
    dump = EtimDump()
    with zipfile.ZipFile(zip_path) as archive:
        synonyms: dict[str, list[str]] = {}
        for row in _table(archive, "ETIMARTCLASSSYNONYMMAP"):
            synonyms.setdefault(row["ARTCLASSID"], []).append(row["CLASSSYNONYM"])
        dump.groups = [EtimGroup(r["ARTGROUPID"], r["GROUPDESC"]) for r in _table(archive, "ETIMARTGROUP")]
        dump.classes = [
            EtimClass(
                r["ARTCLASSID"],
                r["ARTGROUPID"],
                r["ARTCLASSDESC"],
                int(r["ARTCLASSVERSION"] or 0),
                tuple(s for s in synonyms.get(r["ARTCLASSID"], []) if s != r["ARTCLASSDESC"]),
            )
            for r in _table(archive, "ETIMARTCLASS")
        ]
        dump.features = [
            EtimFeature(r["FEATUREID"], r["FEATUREGROUPID"] or None, r["FEATUREDESC"])
            for r in _table(archive, "ETIMFEATURE")
        ]
        dump.units = {r["UNITOFMEASID"]: r["UNITDESC"] for r in _table(archive, "ETIMUNIT")}
        dump.values = {r["VALUEID"]: r["VALUEDESC"] for r in _table(archive, "ETIMVALUE")}
        dump.class_features = [
            EtimClassFeature(
                int(r["ARTCLASSFEATURENR"]),
                r["ARTCLASSID"],
                r["FEATUREID"],
                r["FEATURETYPE"],
                r["UNITOFMEASID"] or None,
                int(r["SORTNR"] or 0),
            )
            for r in _table(archive, "ETIMARTCLASSFEATUREMAP")
        ]
        dump.class_feature_values = [
            (int(r["ARTCLASSFEATURENR"]), r["VALUEID"], int(r["SORTNR"] or 0))
            for r in _table(archive, "ETIMARTCLASSFEATUREVALUEMAP")
        ]
    return dump
