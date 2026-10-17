"""Procurement history workbook: «Позиция (наименование из сметы)», «Цена за ед., руб.»."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import openpyxl

SHEET = "База позиций"


@dataclass(frozen=True)
class PositionRecord:
    row: int  # Excel row number: the id the customer sees
    name: str
    price: float | None


def read(path: Path) -> list[PositionRecord]:
    workbook = openpyxl.load_workbook(path, read_only=True)
    try:
        sheet = workbook[SHEET]
        return [
            PositionRecord(row, str(name).strip(), float(price) if isinstance(price, int | float) else None)
            for row, (name, price, *_) in enumerate(sheet.iter_rows(min_row=2, values_only=True), start=2)
            if name is not None and str(name).strip()
        ]
    finally:
        workbook.close()
