"""Small helpers for writing experiment result rows."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


def append_csv_row(path: str | Path, fieldnames: Iterable[str], row: dict[str, object]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames)
    should_write_header = not output_path.exists()

    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if should_write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})
