"""Export data/positions.xlsx to rows.json (row number, name, price) for classify.py."""

import json
from pathlib import Path

import openpyxl

HERE = Path(__file__).parent
XLSX = HERE.parent.parent / "data" / "positions.xlsx"

ws = openpyxl.load_workbook(XLSX, read_only=True)["База позиций"]
rows = [
    {"row": i, "name": str(name), "price": price}
    for i, (name, price, *_) in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2)
    if name
]
(HERE / "rows.json").write_text(json.dumps(rows, ensure_ascii=False))
print(len(rows), "rows")
