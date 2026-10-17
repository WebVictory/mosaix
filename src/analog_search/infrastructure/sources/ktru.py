"""КТРУ (каталог товаров, работ, услуг), ЕИС: справочник `nsiKTRUNew`.

No anonymous download: ЕИС closed its FTP on 2025-01-01, the справочник comes from the
`getDocsIP` service (int44.zakupki.gov.ru/eis-integration/services/getDocsIP, request
`getNsiRequest`, nsiCode44=nsiKTRUNew, nsiKind=all) with a personal token from the ЕИС account,
and the site answers only from Russian networks. So the export is fetched by hand and `parse`
reads it: a zip (possibly with nested zips) of `nsiKTRUNew_all_actual_*.xml`, a folder or one xml.

Position: `position/data` — code, version, name, OKPD2/code, status, OKEIs/OKEI/name,
characteristics/characteristic — name, type (1 qualitative, 2 quantitative), isRequired, actual,
values/value — qualityDescription | rangeSet/valueRange (min/max + MathNotation) | valueSet/concreteValue,
OKEI/name of the value.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import IO
from xml.etree import ElementTree as ET

from analog_search.domain.sources import KtruCharacteristic, KtruPosition

SIGNS = {"greaterOrEqual": ">=", "greater": ">", "lessOrEqual": "<=", "less": "<"}


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(element: ET.Element | None, name: str) -> ET.Element | None:
    return next((c for c in element if _local(c.tag) == name), None) if element is not None else None


def _children(element: ET.Element | None, name: str) -> list[ET.Element]:
    return [c for c in element if _local(c.tag) == name] if element is not None else []


def _text(element: ET.Element | None, name: str) -> str:
    child = _child(element, name)
    return " ".join((child.text or "").split()) if child is not None else ""


def _range(value_range: ET.Element) -> str:
    """valueRange → «>=0.5 ... <=1.5»; a bound without a comparison sign is skipped."""
    bounds = []
    for bound in ("min", "max"):
        number, sign = _text(value_range, bound), SIGNS.get(_text(value_range, f"{bound}MathNotation"))
        if number and sign:
            bounds.append(f"{sign}{number}")
    return " ... ".join(bounds)


def _characteristic(element: ET.Element) -> KtruCharacteristic | None:
    name = _text(element, "name")
    if not name or _text(element, "actual") == "false":
        return None
    values: list[str] = []
    units: list[str] = []
    for value in _children(_child(element, "values"), "value"):
        found = [_text(value, "qualityDescription")]
        found += [_range(r) for r in _children(_child(value, "rangeSet"), "valueRange")]
        found += [" ".join((c.text or "").split()) for c in _children(_child(value, "valueSet"), "concreteValue")]
        values += [f for f in found if f and f not in values]
        if (unit := _text(_child(value, "OKEI"), "name")) and unit not in units:
            units.append(unit)
    return KtruCharacteristic(
        name=name,
        kind="quantitative" if _text(element, "type") == "2" else "qualitative",
        required=_text(element, "isRequired") == "true",
        values=tuple(values),
        units=tuple(units),
    )


def parse_position(element: ET.Element) -> KtruPosition | None:
    data = _child(element, "data")
    code = _text(data, "code")
    if not code:
        return None
    version = _text(data, "version")
    characteristics = (_characteristic(c) for c in _children(_child(data, "characteristics"), "characteristic"))
    return KtruPosition(
        code=code,
        version=int(version) if version.isdigit() else 0,
        name=_text(data, "name") or code,
        okpd2_code=_text(_child(data, "OKPD2"), "code") or code.split("-")[0],
        status=_text(data, "status"),
        units=tuple(u for o in _children(_child(data, "OKEIs"), "OKEI") if (u := _text(o, "name"))),
        characteristics=tuple(c for c in characteristics if c is not None),
    )


def iter_positions(stream: IO[bytes]) -> Iterator[KtruPosition]:
    """Positions of one xml, streamed: each parsed element is freed."""
    for _, element in ET.iterparse(stream, events=("end",)):
        if _local(element.tag) == "position":
            if (position := parse_position(element)) is not None:
                yield position
            element.clear()


def _is_actual(name: str) -> bool:
    base = name.rsplit("/", 1)[-1]
    return base.endswith(".xml") and "not-actual" not in base


def _xml_in_zip(archive: zipfile.ZipFile) -> Iterator[bytes]:
    for name in sorted(archive.namelist()):
        if name.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(archive.read(name))) as inner:
                yield from _xml_in_zip(inner)
        elif _is_actual(name):
            yield archive.read(name)


def _xml_payloads(path: Path) -> Iterator[bytes]:
    if path.is_dir():
        for file in sorted(path.rglob("*")):
            if file.suffix in (".xml", ".zip"):
                yield from _xml_payloads(file)
    elif path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            yield from _xml_in_zip(archive)
    elif _is_actual(path.name):
        yield path.read_bytes()


def parse(path: Path) -> list[KtruPosition]:
    """Positions of the export; for a code met several times the highest version wins."""
    best: dict[str, KtruPosition] = {}
    for payload in _xml_payloads(path):
        for position in iter_positions(io.BytesIO(payload)):
            if position.code not in best or position.version > best[position.code].version:
                best[position.code] = position
    return list(best.values())
