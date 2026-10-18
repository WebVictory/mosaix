"""External taxonomies as plain data: КСР, ОКПД2, КТРУ, ETIM.

A source is loaded as a release (source + version); every row belongs to one release, so two
versions can live side by side and mappings point at the version they were made against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from domain.frozen import Frozen


class SourceKind(StrEnum):
    KSR = "ksr"
    OKPD2 = "okpd2"
    KTRU = "ktru"
    ETIM = "etim"


class Edition(Frozen):
    """A version of a source as its publisher names it, and where it came from."""

    source: SourceKind
    version: str
    published: date | None = None
    url: str | None = None


class Release(Edition):
    """One downloaded edition: its files and their checksum."""

    files: tuple[str, ...]  # file names inside the release directory
    sha256: str  # over all files, in order


# --- КСР ---


@dataclass(frozen=True)
class KsrNode:
    """Book / part / section / group of the tree; code without the level word: «21.1.01»."""

    code: str
    parent_code: str | None
    level: str  # book | part | section | group
    name: str
    sheet: str  # «Материалы, изд, констр и оборуд» | «Машины и механизмы»


@dataclass(frozen=True)
class KsrResource:
    """Resource «21.1.06.09-0145» and its position-level codes with ОКПД2 prefix."""

    code: str
    group_code: str | None
    name: str
    unit: str | None
    positions: tuple[KsrPosition, ...] = ()


@dataclass(frozen=True)
class KsrPosition:
    """Position-level code «27.32.13.143.21.1.06.09-0145-000»: ОКПД2 + resource + suffix."""

    code: str
    okpd2_code: str
    name: str
    unit: str | None


@dataclass
class KsrDump:
    nodes: list[KsrNode] = field(default_factory=list)
    resources: list[KsrResource] = field(default_factory=list)


# --- ОКПД2 ---


@dataclass(frozen=True)
class Okpd2Node:
    """Section letter «C» or code «27.32.13.143»; description holds «включает / не включает»."""

    code: str
    parent_code: str | None
    level: int  # 0 section «C», 1 class «27», … 6 category «27.32.13.140», 7 subcategory «27.32.13.143»
    name: str
    description: str | None


# --- КТРУ ---


@dataclass(frozen=True)
class KtruCharacteristic:
    name: str
    kind: str  # qualitative | quantitative
    required: bool
    values: tuple[str, ...]  # «ПВХ», «>=0.5 ... <=1.5», «1.5»
    units: tuple[str, ...]  # ОКЕИ names met in values


@dataclass(frozen=True)
class KtruPosition:
    code: str  # «27.32.13.154-00000012»
    version: int
    name: str
    okpd2_code: str
    status: str  # ACTIVE | TERMINATED | …
    units: tuple[str, ...]
    characteristics: tuple[KtruCharacteristic, ...]


# --- ETIM ---


@dataclass(frozen=True)
class EtimGroup:
    code: str  # EG000001
    name: str


@dataclass(frozen=True)
class EtimClass:
    code: str  # EC000001
    group_code: str
    name: str
    version: int
    synonyms: tuple[str, ...]


@dataclass(frozen=True)
class EtimFeature:
    code: str  # EF000001
    group_code: str | None
    name: str


@dataclass(frozen=True)
class EtimClassFeature:
    """Feature of a class. type: A alphanumeric (value list), N numeric, R range, L logical."""

    id: int
    class_code: str
    feature_code: str
    type: str
    unit_code: str | None
    sort: int


@dataclass
class EtimDump:
    groups: list[EtimGroup] = field(default_factory=list)
    classes: list[EtimClass] = field(default_factory=list)
    features: list[EtimFeature] = field(default_factory=list)
    units: dict[str, str] = field(default_factory=dict)  # EU000002 → hPa
    values: dict[str, str] = field(default_factory=dict)  # EV000005 → Thread holding
    class_features: list[EtimClassFeature] = field(default_factory=list)
    class_feature_values: list[tuple[int, str, int]] = field(default_factory=list)  # (class feature id, value, sort)


class SourceStore(Protocol):
    def is_loaded(self, release: Release) -> bool: ...
    def save_ksr(self, release: Release, dump: KsrDump) -> dict[str, int]: ...
    def save_okpd2(self, release: Release, nodes: list[Okpd2Node]) -> dict[str, int]: ...
    def save_ktru(self, release: Release, positions: list[KtruPosition]) -> dict[str, int]: ...
    def save_etim(self, release: Release, dump: EtimDump) -> dict[str, int]: ...


class SourceFormat(Protocol):
    """How one taxonomy is downloaded and how a release of it is read into the store."""

    def fetch(self) -> Path:
        """Download the newest freely available release; the directory it was written to."""
        ...

    def load(self, directory: Path, release: Release, store: SourceStore) -> dict[str, int]: ...
