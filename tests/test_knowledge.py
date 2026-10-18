"""Knowledge from the specialist's yaml and the category rule applied to history rows."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import PROJECT_ROOT
from domain.knowledge import (
    SERVICE_FLAG,
    Category,
    Classifier,
    Comparison,
    Level,
    Provenance,
    normalize,
    slug,
)
from domain.positions import parse
from infrastructure.knowledge import positions_xlsx, yaml_files

RESEARCH = PROJECT_ROOT / "research"

CABLE = Category(code="kabel_silovoy", name="кабель силовой", group="кабели и провода", triggers=("ввг", "кабель силовой"),
                 anti_triggers=("кабель-канал",))  # fmt: skip
CHANNEL = Category(code="kabel_kanal", name="кабель-канал", group="кабеленесущие системы", triggers=("кабель-канал",))
PIPE_OTHER = Category(code="truba_prochaya", name="труба (прочая)", group="трубы", triggers=("труба",))
PIPE_STEEL = Category(code="truba_stalnaya", name="труба стальная", group="трубы", triggers=("стальная",))


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Кабель силовой ВВГнг(A)-LS 5х70", "kabel_silovoy"),  # earliest match
        ("1.2) Кабель-канал 40х25 для ВВГ", "kabel_kanal"),  # numbering stripped; anti-trigger
        ("Труба стальная 57х3,5", "truba_stalnaya"),  # catch-all yields to a sibling
        ("Трубопровод", None),  # «труба» is not at a word start of «трубопровод»
        ("ПВВГ 3х2,5", None),  # trigger must start a word
    ],
)
def test_classifier(name: str, expected: str | None) -> None:
    match = Classifier((CABLE, CHANNEL, PIPE_OTHER, PIPE_STEEL)).classify(name)
    assert (match.category.code if match else None) == expected


def test_normalize_and_slug() -> None:
    assert normalize("  3.1. Ёмкость   ДЛЯ   воды ") == "емкость для воды"
    assert normalize("BВГнг(А)-LS") == normalize("ВВГнг(А)-LS") == "ввгнг(а)-ls"  # Latin B in a Russian mark
    assert normalize("ХNV-6081RЕ") == normalize("XNV-6081RE") == "xnv-6081re"  # Russian Х, Е in a model code
    assert normalize("ip65 ип") == "ip65 ип"  # a word in one alphabet stays as it is
    assert slug("СКУД / турникет / шлагбаум") == "skud_turniket_shlagbaum"


def test_research_yaml_reads_into_one_consistent_release() -> None:
    knowledge = yaml_files.read(RESEARCH)

    by_name = {c.name: c for c in knowledge.categories}
    cable = by_name["кабель силовой"]
    assert cable.group_code == by_name["кабели и провода"].code
    section = next(a for a in cable.attributes if a.code == "сечение_жилы")
    assert (section.level, section.comparison, section.provenance, section.unit) == (
        Level.HARD, Comparison.EQUAL, Provenance.SOURCE, "мм²",
    )  # fmt: skip
    assert 70 in section.rule["series"]
    assert knowledge.version.startswith("1-")
    assert knowledge == yaml_files.read(RESEARCH)  # same files, same version


def test_yaml_with_unknown_category_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "categories.yaml").write_text(
        "версия: 1\nкатегории:\n  - {имя: кабель, группа: кабели, триггеры: [], анти_триггеры: [], синонимы: []}\n",
        encoding="utf-8",
    )
    (tmp_path / "attributes.yaml").write_text("категории:\n  - {категория: провод, характеристики: []}\n", "utf-8")

    with pytest.raises(ValueError, match="провод"):
        yaml_files.read(tmp_path)


def test_history_rows_get_category_and_service_flag() -> None:
    knowledge = yaml_files.read(RESEARCH)
    records = positions_xlsx.read(PROJECT_ROOT / "data" / "positions.xlsx")

    parsed = {p.source_row: p for p in parse(knowledge, records)}

    assert len(parsed) == 2381
    assert parsed[42].category_code == slug("кабель силовой")
    assert parsed[4].flags == (SERVICE_FLAG,)  # «Перевозка длинномерных грузов…»
    assert sum(p.category_code is None for p in parsed.values()) == 117  # research/categories.yaml, итого


def test_catch_all_yields_to_a_category_named_by_its_head_noun() -> None:
    gofra = Category(code="truba_gofrirovannaya", name="труба гофрированная", group="электромонтаж", triggers=("гофр",))

    match = Classifier((PIPE_OTHER, gofra)).classify("Труба гофра ПВХ 25 мм")

    assert match is not None and match.category is gofra  # another group, same head noun «труба»
