"""Parsers of external taxonomies on small files built to the real layouts."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import openpyxl
import pytest

from analog_search.infrastructure.sources import etim, ksr, ktru, okpd2

# --- КСР ---


def _ksr_workbook(path: Path) -> Path:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Материалы, изд, констр и оборуд"
    rows = [
        ("Классификатор строительных ресурсов", None, None),
        ("Код ресурса", "Наименование", "Ед.изм."),
        *[
            (t, t, t)
            for t in (
                "Книга 21: Кабельная продукция",
                "Часть 21.1: Кабели",
                "Раздел 21.1.08: Кабели контрольные",
                "Группа 21.1.08.03: Кабели контрольные с ПВХ",
            )
        ],
        ("21.1.08.03-0049", "Кабель контрольный КВВГнг(A)-FRLS 7х1,5", "1000 м"),
        ("27.32.13.143.21.1.08.03-0049-000", "Кабель контрольный КВВГнг(A)-FRLS 7х1,5", "1000 м"),
        ("21.1.08.03-0079", "Кабель контрольный КВВГЭнг(A)-FRLS 7х1,5", "1000 м"),
        (None, None, None),
    ]
    for row in rows:
        sheet.append(row)
    machines = workbook.create_sheet("Машины и механизмы")
    for row in [
        ("Классификатор строительных ресурсов", None, None),
        ("Код ресурса", "Наименование", "Ед.изм."),
        *[
            (t, t, t)
            for t in (
                "Книга 91: Строительные машины",
                "Раздел 91.01: Машины для земляных работ",
                "Группа 91.01.01: Бульдозеры",
            )
        ],
        ("91.01.01-014", "Бульдозеры-рыхлители, мощность 79 кВт", "маш.-ч"),
        ("28.92.21.110.91.01.01-014-000", "Бульдозеры-рыхлители, мощность 79 кВт", "шт"),
    ]:
        machines.append(row)
    workbook.save(path)
    return path


def test_ksr_tree_resources_and_okpd2_bridge(tmp_path: Path) -> None:
    dump = ksr.parse(_ksr_workbook(tmp_path / "ksr.xlsx"))

    assert [(n.code, n.parent_code, n.level) for n in dump.nodes[:4]] == [
        ("21", None, "book"),
        ("21.1", "21", "part"),
        ("21.1.08", "21.1", "section"),
        ("21.1.08.03", "21.1.08", "group"),
    ]
    # Machines have no «Часть»: the section hangs straight off the book
    assert ("91.01", "91") in [(n.code, n.parent_code) for n in dump.nodes]

    by_code = {r.code: r for r in dump.resources}
    cable = by_code["21.1.08.03-0049"]
    assert (cable.group_code, cable.unit) == ("21.1.08.03", "1000 м")
    assert [(p.okpd2_code, p.code) for p in cable.positions] == [("27.32.13.143", "27.32.13.143.21.1.08.03-0049-000")]
    assert by_code["21.1.08.03-0079"].positions == ()
    assert by_code["91.01.01-014"].positions[0].okpd2_code == "28.92.21.110"


# --- ОКПД2 ---


@pytest.mark.parametrize(
    ("code", "level"),
    [
        ("C", 0),
        ("27", 1),
        ("27.3", 2),
        ("27.32", 3),
        ("27.32.1", 4),
        ("27.32.13", 5),
        ("27.32.13.140", 6),
        ("27.32.13.143", 7),
    ],
)
def test_okpd2_level(code: str, level: int) -> None:
    assert okpd2.level(code) == level


W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'


def _docx(path: Path, rows: list[tuple[str, list[str]]]) -> None:
    def cell(paragraphs: list[str]) -> str:
        return "<w:tc>" + "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs) + "</w:tc>"

    body = "".join(f"<w:tr>{cell([code])}{cell(text)}</w:tr>" for code, text in rows)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", f"<w:document {W}><w:body><w:tbl>{body}</w:tbl></w:body></w:document>")


def test_okpd2_parents_by_existing_codes(tmp_path: Path) -> None:
    _docx(
        tmp_path / "OKPD2 01-35.docx",
        [
            ("ХХ.ХХ", ["группа"]),  # legend table of the standard: skipped
            ("РАЗДЕЛ\xa0C", ["ПРОДУКЦИЯ ОБРАБАТЫВАЮЩИХ ПРОИЗВОДСТВ"]),
            ("27", ["Оборудование электрическое"]),
            ("27.3", ["Кабели и арматура кабельная"]),
            ("27.32", ["Провода и кабели электронные и электрические прочие"]),
            ("27.32.1", ["Провода и кабели электронные и электрические прочие"]),
            (
                "27.32.13",
                [
                    "Проводники электрические прочие на напряжение не более 1 кВ",
                    "Эта группировка включает:",
                    "- кабели контрольные",
                ],
            ),
            ("27.32.13.140", ["Кабели контрольные и прочие"]),
            ("27.32.13.143", ["Кабели контрольные"]),
            ("27.32.13.190", ["Проводники прочие"]),
        ],
    )
    nodes = {n.code: n for n in okpd2.parse(tmp_path)}

    assert "ХХ.ХХ" not in nodes
    assert nodes["27"].parent_code == "C"
    assert nodes["27.32.13.143"].parent_code == "27.32.13.140"
    assert nodes["27.32.13.190"].parent_code == "27.32.13"
    assert nodes["27.32.13"].parent_code == "27.32.1"
    assert nodes["27.32.13"].description == "Эта группировка включает:\n- кабели контрольные"


# --- КТРУ ---

KTRU_XML = """<?xml version="1.0" encoding="UTF-8"?>
<export xmlns="http://zakupki.gov.ru/oos/export/1" xmlns:oos="http://zakupki.gov.ru/oos/types/1">
 <nsiKTRUs>
  <oos:position><oos:data>
   <oos:code>27.32.13.143-00000001</oos:code><oos:version>3</oos:version>
   <oos:name>Кабель  контрольный</oos:name>
   <oos:OKPD2><oos:code>27.32.13.143</oos:code></oos:OKPD2>
   <oos:status>ACTIVE</oos:status>
   <oos:OKEIs><oos:OKEI><oos:code>006</oos:code><oos:name>Метр</oos:name></oos:OKEI></oos:OKEIs>
   <oos:characteristics>
    <oos:characteristic>
     <oos:name>Материал изоляции</oos:name><oos:type>1</oos:type><oos:isRequired>true</oos:isRequired>
     <oos:values><oos:value><oos:qualityDescription>ПВХ</oos:qualityDescription></oos:value>
      <oos:value><oos:qualityDescription>Полиэтилен</oos:qualityDescription></oos:value></oos:values>
    </oos:characteristic>
    <oos:characteristic>
     <oos:name>Сечение жилы</oos:name><oos:type>2</oos:type><oos:isRequired>false</oos:isRequired>
     <oos:values><oos:value>
      <oos:OKEI><oos:code>050</oos:code><oos:name>Квадратный миллиметр</oos:name></oos:OKEI>
      <oos:rangeSet><oos:valueRange><oos:min>0.5</oos:min><oos:minMathNotation>greaterOrEqual</oos:minMathNotation>
       <oos:max>1.5</oos:max><oos:maxMathNotation>lessOrEqual</oos:maxMathNotation></oos:valueRange></oos:rangeSet>
     </oos:value></oos:values>
    </oos:characteristic>
    <oos:characteristic><oos:name>Устаревшая</oos:name><oos:actual>false</oos:actual></oos:characteristic>
   </oos:characteristics>
  </oos:data></oos:position>
  <oos:position><oos:data>
   <oos:code>27.32.13.143-00000001</oos:code><oos:version>2</oos:version><oos:name>Старая версия</oos:name>
  </oos:data></oos:position>
 </nsiKTRUs>
</export>"""


def test_ktru_nested_zip_latest_version_and_characteristics(tmp_path: Path) -> None:
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as archive:
        archive.writestr("nsiKTRUNew_all_actual_20261001.xml", KTRU_XML)
        archive.writestr("nsiKTRUNew_all_not-actual_20261001.xml", KTRU_XML.replace("ACTIVE", "TERMINATED"))
    with zipfile.ZipFile(tmp_path / "export.zip", "w") as archive:
        archive.writestr("part1.zip", inner.getvalue())

    [position] = ktru.parse(tmp_path / "export.zip")

    assert (position.code, position.version, position.name, position.okpd2_code, position.status) == (
        "27.32.13.143-00000001",
        3,
        "Кабель контрольный",
        "27.32.13.143",
        "ACTIVE",
    )
    assert position.units == ("Метр",)
    insulation, section = position.characteristics
    assert (insulation.kind, insulation.required, insulation.values) == ("qualitative", True, ("ПВХ", "Полиэтилен"))
    assert (section.kind, section.values, section.units) == (
        "quantitative",
        (">=0.5 ... <=1.5",),
        ("Квадратный миллиметр",),
    )


# --- ETIM ---


def test_etim_utf16_csv(tmp_path: Path) -> None:
    tables = {
        "ETIMARTGROUP": "ARTGROUPID;GROUPDESC\nEG000001;Cables and wires",
        "ETIMARTCLASS": "ARTCLASSID;ARTGROUPID;ARTCLASSDESC;ARTCLASSVERSION;ARTCLASSVERSIONDATE\n"
        "EC000034;EG000001;Fibre optic cable;10;",
        "ETIMARTCLASSSYNONYMMAP": "ARTCLASSID;CLASSSYNONYM\nEC000034;Fibre optic cable\nEC000034;Optical fibre cable",
        "ETIMFEATURE": "FEATUREID;FEATUREGROUPID;FEATUREDESC\nEF000001;EFG00007;Number of fibres",
        "ETIMUNIT": "UNITOFMEASID;UNITDESC\nEU570448;mm²",
        "ETIMVALUE": "VALUEID;VALUEDESC\nEV000005;PVC",
        "ETIMARTCLASSFEATUREMAP": "ARTCLASSFEATURENR;ARTCLASSID;FEATUREID;FEATURETYPE;UNITOFMEASID;SORTNR\n"
        "1;EC000034;EF000001;N;;1",
        "ETIMARTCLASSFEATUREVALUEMAP": "ARTCLASSFEATUREVALUENR;ARTCLASSFEATURENR;VALUEID;SORTNR\n1;1;EV000005;1",
    }
    with zipfile.ZipFile(tmp_path / "etim.zip", "w") as archive:
        for name, text in tables.items():
            archive.writestr(f"{name}.csv", text.replace("\n", "\r\n").encode("utf-16"))

    dump = etim.parse(tmp_path)

    [cls] = dump.classes
    assert (cls.code, cls.group_code, cls.synonyms) == ("EC000034", "EG000001", ("Optical fibre cable",))
    assert dump.class_features[0].type == "N"
    assert dump.class_features[0].unit_code is None
    assert dump.units == {"EU570448": "mm²"}
    assert dump.class_feature_values == [(1, "EV000005", 1)]
