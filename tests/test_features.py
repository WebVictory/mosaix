"""Attribute values read from real history rows and queries, and the text normalization under them."""

from __future__ import annotations

import pytest

from analog_search.domain.features import Extractors
from analog_search.domain.knowledge import Classifier, Knowledge, normalize
from analog_search.domain.positions import ParsedPosition
from analog_search.domain.trigrams import similarity, trigrams


def read(knowledge: Knowledge, text: str) -> tuple[str | None, dict]:
    match = Classifier(knowledge.categories).classify(text)
    if match is None:
        return None, {}
    return match.category.code, Extractors(knowledge.categories).extract(match.category, normalize(text))


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (1171, {"число_жил": 7.0, "сечение_жилы": 1.5, "экран": "экран", "исполнение_пожарное": "нг(А)-LS"}),
        (159, {"тип": "ворота откатные", "размер": [2250.0, 6000.0]}),
        (374, {"ширина": 400.0, "высота_борта": 100.0}),  # «LPS 100х400»: the larger side is the width
        (779, {"номинальный_ток": 6.0}),  # «6А» is the rated current, not a breaking capacity in кА
        (1541, {"выходное_напряжение": 12.0, "род_тока": "DC", "исполнение_уличное": "улица"}),
        (815, {"сечение": [200.0]}),  # «диаметр до 500 мм (d=200 мм)»: 500 is a price-list column, not a size
        (29, {"сечение": [500.0, 500.0]}),
        (1373, {"число_пар_или_жил": 2.0, "сечение_или_диаметр_жилы": 0.5}),  # «2x2x0,5»: pairs, not 2 cores
        (662, {"число_пар_или_жил": 1.0, "сечение_или_диаметр_жилы": 1.0}),  # «1x2х,1.0» as written
    ],
)
def test_history_row_attributes(positions: dict[int, ParsedPosition], row: int, expected: dict) -> None:
    attributes = positions[row].attributes
    assert {code: attributes.get(code) for code in expected} == expected


@pytest.mark.parametrize(
    ("query", "category", "expected"),
    [
        ("кабель КВВГнг-LS 7*1.5", "kabel_kontrolnyy", {"число_жил": 7.0, "сечение_жилы": 1.5}),
        ("кабель контрольный экранированный 7 жил 2,5 мм2", "kabel_kontrolnyy", {"число_жил": 7.0, "экран": "экран"}),
        ("труба гофра ПВХ 25 мм", "truba_gofrirovannaya_elektromontazhnaya", {"диаметр": 25.0}),
        ("выключатель автоматический Legrand 2Р 6А", "avtomaticheskiy_vyklyuchatel", {"отключающая_способность": None}),
        ("лоток 100х400", "lotok_kabelnyy", {"ширина": 400.0, "высота_борта": 100.0}),
        # cores of two sections: neither 3 nor 4 cores, neither 2,5 nor 1,5 says it
        ("Кабель ВВГнг(А)-LS 3х2,5+1х1,5", "kabel_silovoy", {"число_жил": None, "сечение_жилы": None}),
    ],
)
def test_query_attributes(knowledge: Knowledge, query: str, category: str, expected: dict) -> None:
    code, values = read(knowledge, query)
    assert code == category
    assert {k: values.get(k) for k in expected} == expected


def test_normalize_spells_sizes_one_way() -> None:
    assert normalize("КВВГ 7Х1,5") == normalize("кввг 7 x 1.5") == normalize("кввг 7*1.5") == "кввг 7x1.5"


@pytest.mark.parametrize(
    ("a", "b", "pg_trgm"),  # SELECT similarity(a, b) on PostgreSQL 17
    [
        ("кабель кввгнг-ls 7x1.5", "кабель контрольный кввгэнг(a)-ls 7x1.5", 0.5263158),
        ("труба гофра пвх 25 мм", "гофрированная труба из пвх (серия 9) с протяжкой d=25 мм", 0.36363637),
    ],
)
def test_trigram_similarity_equals_pg_trgm(a: str, b: str, pg_trgm: float) -> None:
    assert similarity(trigrams(a), trigrams(b)) == pytest.approx(pg_trgm, abs=1e-6)


def test_formula_the_extractor_does_not_read_is_declined_not_silent(knowledge: Knowledge) -> None:
    extractors = Extractors(knowledge.categories)
    cable = next(c for c in knowledge.categories if c.code == "kabel_silovoy")

    assert extractors.declined(cable, normalize("ВВГнг(А)-LS 3х2,5+1х1,5")) == {"число_жил", "сечение_жилы"}
    assert extractors.declined(cable, normalize("ВВГнг(А)-LS 3х2,5")) == set()
