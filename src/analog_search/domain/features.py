"""Attribute values from a row's text: deterministic extractors, no LLM.

Context comes from the category: a row is read only for the attributes of its own category,
so «700x400» in a cabinet name is never taken for a cable section, and «2250х6000» counts as
a size only where the category has one. A false value is worse than none: an extractor that is
not sure returns None, and the attribute is reported as missing instead of mismatched.

Values are plain JSON (float, str, sorted list of floats) — they are stored in positions.attributes.
An extractor also knows how to compare its values (`key`) and how to show them (`format`).
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any

from analog_search.domain.knowledge import Attribute, Category, Comparison, normalize

type Value = float | str | list[float]

NUM = r"(\d+(?:[.,]\d+)?)"
X = r"\s*[xх×*]\s*"
START = r"(?<![а-яa-z0-9])"  # word start
END = r"(?![а-яa-z0-9])"  # word end
GLUED = r"(?<=\d)"  # a one-letter unit sticks to its number: «12в» is volts, «2 в составе» is not


def number(text: str) -> float:
    return round(float(text.replace(",", ".")), 6)


def format_number(value: float) -> str:
    return f"{value:g}".replace(".", ",")


class FeatureExtractor(ABC):
    # A value the choice model picks from the attribute's series compares with this extractor's values when
    numeric = False  # values are plain numbers and the series is numeric, or
    series_items = False  # values are the series' own strings

    @abstractmethod
    def extract(self, text: str) -> Value | None:
        """Value from a normalized text, None when not sure."""

    def declines(self, text: str) -> bool:
        """The text states the value in a form this extractor does not read: the query is not to be taken as silent."""
        return False

    def key(self, value: Value) -> Any:
        """Comparable form: a number, a position on an ordered scale, a token set."""
        return value

    def accepts(self, item: Value) -> bool:
        """An item of the attribute's series is one of this extractor's values: chosen, it compares like a read
        one. The choice model is let fill in only for extractors of the series' own items; a specialist sets any
        accepted item."""
        return self.numeric if isinstance(item, float) else self.series_items

    def format(self, value: Value, unit: str | None) -> str:
        if isinstance(value, float):
            return f"{format_number(value)} {unit}" if unit else format_number(value)
        return str(value)


# --- Numbers ---

# Spellings of a quantity (regex fragments) → factor to its base unit: V, A, VA, W, var, mm, mm², …
SPELLINGS: dict[str, dict[str, float]] = {
    "voltage": {r"кв(?!\.?\s*м)": 1e3, "kv": 1e3, GLUED + "в": 1, GLUED + "v": 1},
    "current": {"ка": 1e3, "ka": 1e3, GLUED + "а": 1, GLUED + "a": 1},
    "apparent_power": {"мва": 1e6, "mva": 1e6, "ква": 1e3, "kva": 1e3, "ва": 1, "va": 1},
    "active_power": {"мвт": 1e6, "mw": 1e6, "квт": 1e3, "kw": 1e3, "вт": 1, GLUED + "w": 1},
    "reactive_power": {"мвар": 1e6, "квар": 1e3, "kvar": 1e3},
    "section": {"мм2": 1, "мм²": 1, r"кв\.?\s?мм": 1, r"мм\.?\s?кв": 1, "mm2": 1},
    "length": {"км": 1e6, GLUED + "м": 1e3, "см": 10, "мм": 1, "mm": 1},
    "resolution": {"мегапиксел[а-я]*": 1, "мпикс": 1, "мп": 1, "mp": 1},
    "mass": {GLUED + "т": 1e3, "кг": 1, "kg": 1},
    "volume": {"м3": 1e3, "м³": 1e3, GLUED + "л": 1, GLUED + "l": 1},
    "force": {"кн": 1, "kn": 1},
    "pressure": {"мпа": 1, "mpa": 1, "бар": 0.1, "bar": 0.1},
    "colour_temperature": {GLUED + "k": 1, GLUED + "к": 1},
    "duration": {"мин": 1},
    "charge": {"а·ч": 1, "ач": 1, "а/ч": 1, "ah": 1},
}
# Unit of an attribute (attributes.yaml «единица») → quantity and the unit's factor to the base.
# Lengths are not here on purpose: any «N мм» in a name may be a length, width or thickness,
# so a length is read only by an extractor with an explicit marker (d=, Ду, «2250х6000»).
UNITS: dict[str, tuple[str, float]] = {
    "кВ": ("voltage", 1e3), "В": ("voltage", 1),
    "кА": ("current", 1e3), "А": ("current", 1),
    "МВА": ("apparent_power", 1e6), "кВА": ("apparent_power", 1e3), "ВА": ("apparent_power", 1),
    "МВт": ("active_power", 1e6), "кВт": ("active_power", 1e3), "Вт": ("active_power", 1),
    "квар": ("reactive_power", 1e3),
    "мм²": ("section", 1),
    "Мп": ("resolution", 1),
    "кг": ("mass", 1), "т": ("mass", 1e3),
    "л": ("volume", 1),
    "кН": ("force", 1),
    "МПа": ("pressure", 1),
    "K": ("colour_temperature", 1),
    "мин": ("duration", 1),
    "А·ч": ("charge", 1),
}  # fmt: skip


class Quantity(FeatureExtractor):
    """«2,5мм2», «1250 кВА», «12В», «0,4 кВ» → number in the attribute's unit.

    A number right after «х» is the tail of a formula («3х2,5+1х1,5 мм2»), not a quantity on its own.
    """

    numeric = True

    def __init__(self, quantity: str, unit_factor: float):
        self._factors = list(SPELLINGS[quantity].values())
        alternatives = "|".join(f"({s})" for s in SPELLINGS[quantity])
        self._pattern = re.compile(rf"(?<![\d.,xх×*])(?<![xх×*]\s){NUM}\s*(?:{alternatives}){END}")
        self._unit_factor = unit_factor

    def extract(self, text: str) -> float | None:
        if found := self._pattern.search(text):
            spelling = next(i for i, group in enumerate(found.groups()[1:]) if group is not None)
            return round(number(found[1]) * self._factors[spelling] / self._unit_factor, 6)
        return None


def quantity(unit: str | None) -> Quantity | None:
    """Generic reader of an attribute by its unit; None for a unit without a safe spelling."""
    if unit not in UNITS:
        return None
    name, factor = UNITS[unit]
    return Quantity(name, factor)


class Pattern(FeatureExtractor):
    """A marked number: «SDR 11», «ТСЗ-1250/11», «РИП-12» → the first matched group, times a factor."""

    numeric = True

    def __init__(self, pattern: str, factor: float = 1.0):
        self._pattern = re.compile(pattern)
        self._factor = factor

    def extract(self, text: str) -> float | None:
        if found := self._pattern.search(text):
            return round(number(next(g for g in found.groups() if g)) * self._factor, 6)
        return None


class Fraction(FeatureExtractor):
    """Two numbers through a slash, kept as written: «АС 240/39» → «240/39» (aluminium / steel, мм²).

    One wire, not a ratio: 240/32 and 240/39 are different wires of the same total-ish section.
    """

    series_items = True

    def __init__(self) -> None:
        self._pattern = re.compile(r"(?<![\d.,/])(\d{2,3})\s*/\s*(\d{1,2}(?:[.,]\d)?)(?![\d.,])")

    def extract(self, text: str) -> str | None:
        if found := self._pattern.search(text):
            return f"{found[1]}/{found[2].replace('.', ',')}"
        return None


class Marker(FeatureExtractor):
    """A word that stands for a number: «HDTV 1080» → 2 Мп."""

    numeric = True

    def __init__(self, pattern: str, value: float):
        self._pattern = re.compile(pattern)
        self._value = value

    def extract(self, text: str) -> float | None:
        return self._value if self._pattern.search(text) else None


class FirstOf(FeatureExtractor):
    """Several spellings of one attribute, the most explicit first; they share the value form."""

    def __init__(self, *extractors: FeatureExtractor):
        self._extractors = extractors
        self.numeric = all(e.numeric for e in extractors)

    def extract(self, text: str) -> Value | None:
        return next((v for e in self._extractors if (v := e.extract(text)) is not None), None)

    def declines(self, text: str) -> bool:
        return self.extract(text) is None and any(e.declines(text) for e in self._extractors)

    def key(self, value: Value) -> Any:
        return self._extractors[0].key(value)

    def format(self, value: Value, unit: str | None) -> str:
        return self._extractors[0].format(value, unit)


# --- Shapes ---


class CoresSection(FeatureExtractor):
    """«5х70», «7*1,5», «3х95/50» → cores or section.

    Not read: «4х2х0,52» (pairs), «700x400», «3х2,5+1х1,5» (cores of two sections: neither 3 nor 4 says it).
    """

    numeric = True

    _pattern = re.compile(rf"(?<![\d.,xх×*/+])(?<!\+\s)(\d{{1,2}}){X}{NUM}(?!\d|[.,]\d|{X}\d|\s*\+)")

    _composite = re.compile(rf"\d{X}{NUM}\s*\+\s*\d{X}{NUM}")

    def __init__(self, part: str):
        self._group = {"cores": 1, "section": 2}[part]

    def declines(self, text: str) -> bool:
        return bool(self._composite.search(text))

    def extract(self, text: str) -> float | None:
        for found in self._pattern.finditer(text):
            if 1 <= int(found[1]) <= 61 and 0.2 <= number(found[2]) <= 1200:
                return number(found[self._group])
        return None


class PairsSection(FeatureExtractor):
    """«2х2х0,5», «1x2х,1.0» → pairs or core section of a twisted-pair cable: N pairs × 2 cores × section."""

    numeric = True

    _pattern = re.compile(rf"(?<![\d.,])(\d{{1,2}}){X}2{X},?{NUM}(?![\d.,]\d)")

    def __init__(self, part: str):
        self._group = {"pairs": 1, "section": 2}[part]

    def extract(self, text: str) -> float | None:
        if found := self._pattern.search(text):
            return number(found[self._group])
        return None


class Dimensions(FeatureExtractor):
    """«2250х6000 мм», «6000x2250» → [2250.0, 6000.0]: sides compared as a sorted multiset."""

    _pattern = re.compile(rf"(?<![\d.,]){NUM}((?:{X}\d+(?:[.,]\d+)?)+)")

    def extract(self, text: str) -> list[float] | None:
        if found := self._pattern.search(text):
            sides = [found[1], *re.findall(r"\d+(?:[.,]\d+)?", found[2])]
            return sorted(number(s) for s in sides)
        return None

    def format(self, value: Value, unit: str | None) -> str:
        assert isinstance(value, list)
        return "×".join(format_number(v) for v in value) + (f" {unit}" if unit else "")


class Side(FeatureExtractor):
    """One side of a two-side size, whatever the writing order: «LPS 100х400» and «400x100» are 400 wide, 100 high.

    For a cable tray the width is the larger side; «SL 200 h-100» states them by marks.
    """

    numeric = True

    def __init__(self, larger: bool, marked: str | None = None):
        self._larger = larger
        self._marked = re.compile(marked) if marked else None

    def extract(self, text: str) -> float | None:
        if self._marked and (found := self._marked.search(text)):
            return number(found[1])
        sides = Dimensions().extract(text)
        if sides is None or len(sides) != 2:
            return None
        return sides[-1] if self._larger else sides[0]


class Diameter(FeatureExtractor):
    """«d=25 мм», «д.50мм», «Ду 110», «диаметр: 20 мм», «номинальный внутренний диаметр 32».

    bare=True also takes a lone «25 мм» — only where the category has no other length.
    """

    numeric = True

    _marked = re.compile(
        rf"{START}(?:d|д|ду|dn|ø|диаметр[а-я]*|номинальн[а-я]* (?:[а-я]+ )?диаметр[а-я]*)\s*[=.:]?\s*{NUM}"
    )
    _bare = re.compile(rf"(?<![\d.,xх×*]){NUM}\s*мм{END}")

    def __init__(self, bare: bool = False):
        self._bare_allowed = bare

    def extract(self, text: str) -> float | None:
        if found := self._marked.search(text):
            return number(found[1])
        if self._bare_allowed and len(bare := self._bare.findall(text)) == 1:
            return number(bare[0])
        return None


class DuctSection(FeatureExtractor):
    """«(500х500 мм)», «d=160», «диаметр 125 мм» → sides of a duct, [500.0, 500.0] or [160.0].

    «диаметр до 500 мм (d=200 мм)» is 200: the size range of a price line is not the duct's size.
    """

    _sides = Dimensions()
    _diameter = Diameter()

    def extract(self, text: str) -> list[float] | None:
        if (sides := self._sides.extract(text)) is not None and len(sides) == 2:
            return sides
        diameter = self._diameter.extract(text)
        return None if diameter is None else [diameter]

    def format(self, value: Value, unit: str | None) -> str:
        assert isinstance(value, list), "a section is read as a list: two sides or one diameter"
        return self._sides.format(value, unit) if len(value) == 2 else f"d={format_number(value[0])} мм"


class IPRating(FeatureExtractor):
    """«IP66», «ip 54» → "IP66"; compared digit by digit (solids, water)."""

    _pattern = re.compile(rf"{START}ip\s?([0-6])([0-9]){END}")

    def extract(self, text: str) -> str | None:
        found = self._pattern.search(text)
        return f"IP{found[1]}{found[2]}" if found else None

    def key(self, value: Value) -> tuple[int, ...]:
        return tuple(int(c) for c in str(value)[2:])


# --- Words ---


class Vocabulary(FeatureExtractor):
    """Canonical value by its spellings (regex fragments read from a word start).

    The earliest spelling in the text wins, a longer one breaks a tie. by_priority=True: the
    first listed value found anywhere wins instead — an explicit «с экраном» beats a guess from
    the mark. scale: values in order (indoor < outdoor, fixed < varifocal < motorized), on which
    «not worse» / «includes» compare positions.
    """

    def __init__(
        self,
        values: Mapping[str, tuple[str, ...]],
        scale: tuple[str, ...] | None = None,
        by_priority: bool = False,
    ):
        self._scale = scale
        self._by_priority = by_priority
        self._patterns = [
            (value, re.compile(rf"{START}(?:{'|'.join(spellings)})")) for value, spellings in values.items()
        ]

    def extract(self, text: str) -> str | None:
        hits = []
        for rank, (value, pattern) in enumerate(self._patterns):
            if found := pattern.search(text):
                hits.append(((rank,) if self._by_priority else (found.start(), -len(found[0])), value))
        return min(hits)[1] if hits else None

    def key(self, value: Value) -> Any:
        return self._scale.index(str(value)) if self._scale and value in self._scale else value

    def accepts(self, item: Value) -> bool:
        return any(item == value for value, _ in self._patterns)


class SeriesWords(Vocabulary):
    """A vocabulary whose values are the attribute's series: the model may read what it misses."""

    series_items = True


def series_vocabulary(attribute: Attribute) -> Vocabulary | None:
    """Fallback for a qualitative attribute: its value series spelled as is (3+ letters to be safe).

    A numeric series («100», «500x500») is not a vocabulary: «диаметр до 500 мм (d=200)» would read 500.
    """
    series = attribute.rule.get("series") or []
    words = [s for s in series if isinstance(s, str) and len(s) >= 3 and re.search(r"[а-яa-z]", normalize(s))]
    if not words or len(words) != len(series):
        return None
    ordered = attribute.comparison in (Comparison.NOT_WORSE, Comparison.INCLUDES, Comparison.STEP)
    return SeriesWords({w: (re.escape(normalize(w)) + END,) for w in words}, scale=tuple(words) if ordered else None)


class FireRating(FeatureExtractor):
    """«нг(А)-FRLS», «нг-LS», «нг(A)» → canonical spelling; «includes» compares token sets.

    FRLS covers LS: a fire-resistant low-smoke cable replaces a low-smoke one, not the other way.
    """

    _pattern = re.compile(r"нг\s*(\(\s*[аa](?:\s*-?\s*f)?\s*\))?(?:\s*-\s*(frls|ls|frhf|hf|frltx|ltx))?" + END)

    def extract(self, text: str) -> str | None:
        found = self._pattern.search(text)
        if not found:
            return None
        return "нг" + ("(А)" if found[1] else "") + (f"-{found[2].upper()}" if found[2] else "")

    def accepts(self, item: Value) -> bool:
        return isinstance(item, str) and self.extract(normalize(item)) == item  # a canonical spelling

    def key(self, value: Value) -> frozenset[str]:
        mark, _, suffix = str(value).partition("-")
        tokens = {"нг"} | ({"А"} if "(А)" in mark else set())
        if suffix.startswith("FR"):
            tokens.add("FR")
            suffix = suffix[2:]
        return frozenset(tokens | ({suffix} if suffix else set()))


# --- Registry: which extractor reads which attribute ---

type Factory = Callable[[Attribute], FeatureExtractor | None]

# Suffix of a cable mark after the section: «3х1,5-660», «5х70мс(N, РЕ)-1000» (volts), «-0,66» (kV)
_AFTER_SECTION = rf"\d{X}\d+(?:[.,]\d+)?[^-\s(]*(?:\s*\([^)]*\))?\s*-\s*"
CABLE_VOLTAGE_V = Pattern(_AFTER_SECTION + r"(660|1000|380|3000|6000|10000)(?![\d.,])", factor=1e-3)
CABLE_VOLTAGE_KV = Pattern(_AFTER_SECTION + r"(0[.,]66|1[.,]0|1|6|10|20|35)(?![\d.,])")
# …or right after the mark: «ВБШвнг(А)-LS-0,66 4х2,5», «АПвВнг(А)-LS-10 3х95/50», «КГВВ-0,66 5х25»
CABLE_MARK_VOLTAGE = Pattern(r"(?:[а-я]|ls|hf)-(0[.,]66|1[.,]0|6|10|20|35)(?![\d.,])")
FIRE_RESISTANCE = Pattern(rf"{START}(?:[eе][il])\s*-?\s*(\d+)|дпм-\d+/(\d+)")
OUTDOOR = Vocabulary(
    {
        "помещение": ("для помещ", "в помещени", "внутренн", "indoor"),
        "улица": ("уличн", "для улиц", "наружн", "outdoor", r"исп\.?\s*5м"),
    },
    scale=("помещение", "улица"),
)


def _voltage(attribute: Attribute) -> FeatureExtractor | None:
    generic = quantity(attribute.unit)
    if attribute.unit == "кВ" and generic is not None:
        return FirstOf(generic, CABLE_VOLTAGE_V, CABLE_VOLTAGE_KV, CABLE_MARK_VOLTAGE)
    return generic


BY_CODE: dict[str, Factory] = {
    "число_жил": lambda a: FirstOf(CoresSection("cores"), Pattern(rf"{START}(\d{{1,2}})[ -]?жил")),
    "сечение_жилы": lambda a: FirstOf(CoresSection("section"), Quantity("section", 1)),
    "номинальное_напряжение": _voltage,
    "напряжение": _voltage,
    "класс_напряжения": _voltage,
    "материал_жилы": lambda a: Vocabulary(
        {
            "Al": ("алюмин", r"а(?:ввг|пвп|пвбп|абл|пвв|сбл|бшв)"),
            "Cu": ("медн", r"к?ввг", "вбшв", r"пв[вп]", "пвбп"),
        }
    ),
    "исполнение_пожарное": lambda a: FireRating(),
    "сечение_экрана": lambda a: Pattern(rf"\d{X}\d+(?:[.,]\d+)?[а-я]*/(\d+)"),
    "экран": lambda a: Vocabulary(  # «Э» in the mark or the word; an unscreened mark only when neither
        {"экран": (r"[а-я]*э(?:нг|кр)", "экран", "кпсэ", "кипэв"), "без экрана": (r"к?ввг", "вбшв")},
        scale=("без экрана", "экран"),
        by_priority=True,
    ),
    "степень_защиты": lambda a: IPRating(),
    "размер": lambda a: Dimensions(),
    "диаметр": lambda a: Diameter(),
    "наружный_диаметр": lambda a: Diameter(),
    "внутренний_диаметр": lambda a: Diameter(),
    "исполнение_уличное": lambda a: OUTDOOR,
    "разрешение": lambda a: FirstOf(
        Quantity("resolution", 1), Marker(r"1080p|hdtv\s*1080|full\s*hd|1920\s*[xх×]\s*1080", 2.0)
    ),
    "объектив": lambda a: Vocabulary(
        {
            "фиксированный": ("фиксирован",),
            "вариофокальный": ("вариофокал", r"2[.,]8\s*[-–]\s*12"),
            "моторизованный": ("моторизир", "моторизов", r"\d+-кратн[а-я]* оптическ"),
        },
        scale=("фиксированный", "вариофокальный", "моторизованный"),
    ),
    "тип_сигнала": lambda a: Vocabulary({"IP": (r"ip(?![0-9a-z])", "сетев"), "HD-аналог": ("ahd", "tvi", "cvi")}),
    "тип_корпуса": lambda a: Vocabulary(
        {"PTZ": ("ptz", "поворотн", "speed dome"), "купольная": ("купольн", "dome"), "корпусная": ("корпусн", "bullet")}
    ),
    "ик_подсветка": lambda a: Vocabulary({"есть": (r"ик[ -]?подсвет",)}),
    "тип_изоляции": lambda a: Vocabulary(
        {"сухой": ("сух", r"тс[злг]", "литой"), "масляный": ("масл", r"тм[гнз]?-", "тдтн", "трдн", "тдцтн")}
    ),
    "род_тока": lambda a: Vocabulary({"DC": (r"\d*v?dc", "постоянн"), "AC": (r"\d*v?ac(?![a-z])", "переменн")}),
    "выходное_напряжение": lambda a: FirstOf(
        Quantity("voltage", 1), Pattern(r"рип-(\d+)"), Pattern(r"(?<![\d,])(\d+)\s*v?dc(?![a-z])")
    ),
    "тип_отв": lambda a: Vocabulary(
        {
            "ОП": (r"оп-?\d", "порошков"),
            "ОУ": (r"оу-?\d", "углекисл"),
            "ОВП": ("овп", "воздушно-пенн"),
            "ОВЭ": ("овэ", "эмульс"),
            "ОВ": (r"ов-?\d", "водн"),
        }
    ),
    "масса_заряда": lambda a: FirstOf(Pattern(r"(?<![а-я])о(?:п|у|вп|вэ|в)-?(\d+)"), Quantity("mass", 1)),
    "протяжка": lambda a: Vocabulary({"есть": ("с протяжк", "с зонд")}),
    "sdr": lambda a: Pattern(rf"sdr\s*-?\s*{NUM}"),
    "кольцевая_жесткость": lambda a: Pattern(rf"{START}sn\s*-?\s*(\d+)"),
    # only stated in кА: «6А» is the rated current, not a breaking capacity of 0,006 кА
    "отключающая_способность": lambda a: Pattern(rf"{NUM}\s*(?:ка|ka){END}"),
    "огнестойкость": lambda a: FIRE_RESISTANCE,
    "предел_огнестойкости": lambda a: FIRE_RESISTANCE,
}
# The same code means different things in different categories: «тип» of a gate is not «тип» of a lug
GOFRA = "truba_gofrirovannaya_elektromontazhnaya"
SIGNAL = "kabel_svyazi_i_signalizatsii"
BY_CATEGORY: dict[tuple[str, str], Factory] = {
    ("dveri_okna_vorota", "тип"): lambda a: Vocabulary(
        {
            "ворота откатные": ("откатн",),
            "ворота распашные": ("распашн",),
            "роллворота": ("роллворот", "рулонн"),
            "окно": ("окн",),
            "дверь": ("двер", "блок дверн"),
        }
    ),
    # «ТМЛ 35-12-10», «НШП 50-20», «ГМЛ 35»: the size is the first number after the mark, the bolt the second
    ("nakonechnik_kabelnyy_gilza", "сечение"): lambda a: FirstOf(
        *filter(None, (quantity(a.unit), Pattern(rf"{START}(?:тмл|там?|нш[пв]?и?|гмл|гм|га)\s*-?\s*(\d+(?:\.\d+)?)")))
    ),
    ("nakonechnik_kabelnyy_gilza", "диаметр_под_болт"): lambda a: FirstOf(
        Pattern(rf"{START}(?:тмл|там?)\s*-?\s*\d+(?:\.\d+)?\s*[-–]\s*(\d+)"),
        Pattern(rf"{START}под\s*болт\s*м?\s*(\d+)"),
    ),
    ("nakonechnik_kabelnyy_gilza", "тип"): lambda a: SeriesWords(
        {
            "кольцевой": (r"тмл", r"там?\s*-?\s*\d", r"кольцев"),
            "штифтовой": (r"нш", r"штифт"),
            "гильза": (r"гм", r"гильз", r"га\s*-?\s*\d"),
        }
    ),
    ("vozdukhovod", "сечение"): lambda a: DuctSection(),
    ("provod_neizolirovannyy_as_m_a", "сечение_ал_ст"): lambda a: Fraction(),
    (SIGNAL, "число_пар_или_жил"): lambda a: PairsSection("pairs"),
    (SIGNAL, "сечение_или_диаметр_жилы"): lambda a: PairsSection("section"),
    ("lotok_kabelnyy", "ширина"): lambda a: Side(larger=True, marked=rf"{START}sl\s*-?\s*(\d+)"),
    ("lotok_kabelnyy", "высота_борта"): lambda a: Side(larger=False, marked=rf"{START}h\s*-\s*(\d+)"),
    (GOFRA, "диаметр"): lambda a: Diameter(bare=True),
    (GOFRA, "материал"): lambda a: Vocabulary(
        {"ПВХ": ("пвх",), "ПНД": ("пнд",), "ПП": ("полипропилен", r"пп(?![а-я])"), "полиамид": ("полиамид",)}
    ),
    (GOFRA, "конструкция"): lambda a: Vocabulary({"двустенная": ("двустен",), "армированная": ("армирован",)}),
    ("transformator_silovoy", "мощность"): lambda a: FirstOf(
        Quantity("apparent_power", 1e3), Pattern(r"(?<![а-я])т[а-я]{1,4}-(\d+)/")
    ),
}


class Extractors:
    """Extractors of every attribute of every category, resolved once per knowledge release.

    Order: a category-specific extractor, one by attribute code, a generic one by unit,
    the attribute's value series as a vocabulary. An attribute with none of these is not read.
    """

    def __init__(self, categories: tuple[Category, ...]):
        self._by_category: dict[str, dict[str, FeatureExtractor]] = {}
        self._total = 0
        for category in categories:
            resolved = {}
            for attribute in category.attributes:
                factory = BY_CATEGORY.get((category.code, attribute.code)) or BY_CODE.get(attribute.code)
                extractor = factory(attribute) if factory else quantity(attribute.unit) or series_vocabulary(attribute)
                if extractor is not None:
                    resolved[attribute.code] = extractor
            self._by_category[category.code] = resolved
            self._total += len(category.attributes)

    def of(self, category: Category) -> dict[str, FeatureExtractor]:
        return self._by_category.get(category.code, {})

    def extract(self, category: Category, text: str) -> dict[str, Value]:
        """Values of the category's attributes found in a normalized text."""
        found = {}
        for code, extractor in self.of(category).items():
            if (value := extractor.extract(text)) is not None:
                found[code] = value
        return found

    def declined(self, category: Category, text: str) -> frozenset[str]:
        """Attributes the text states in a form their extractor does not read."""
        return frozenset(code for code, extractor in self.of(category).items() if extractor.declines(text))

    def coverage(self) -> tuple[int, int]:
        """Attributes that have an extractor, attributes in total."""
        return sum(len(e) for e in self._by_category.values()), self._total


def same(a: Value, b: Value) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-6)
    return a == b
