"""Every cell of a ``data_table`` is checked, not only that the table has rows (F345).

F345 (night 10b): the finalisation guard held for an empty chip and for a table with
no rows, and nothing else. Three price lists went out with empty prices and invoices
with a blank Total cell: a row that lacked a column's key, or held "" or spaces
there, rendered as an empty cell and counted as filled.

Now each row is read against the table's columns, the way the renderers read it
(``cell_text``: an object by key, a plain list by position). A missing key or a
blank cell is unresolved and named by row (counted from 1, as a reader counts) and
column, e.g. ``data.line_items[row 3].total``, so the owner and the agent know
exactly what to fill. A column the author marks ``optional`` may stay empty.

F369 (night 10c): Auto's invoice turned "12 kg at £22.00 a kg" into "Qty 1 · £264.00"
(description "12 kg Harbour Blend"): the line items had no place for a unit, so the
model folded the quantity into the description. A row may now carry ``unit``
("kg", "hours"): it prints after the row's ``quantity`` ("12 kg"), and the unit price
stays the price of one unit. :func:`cell_text` is the one reading both renderers
(PDF and Word) print a cell with.

Pure: no IO.
"""
from __future__ import annotations

from typing import Any, List

from ..amounts import field_text
from ..variables.catalog import is_blank
from .schema import DataTableBlock

# A long table with one empty column would name hundreds of cells: name the first
# few, then say how many more there are.
MAX_NAMED_CELLS = 20
MORE_CELLS = "{path}: {count} more blank cells"
MISSING = object()
# F369: a row's unit prints after its quantity ("12 kg").
QUANTITY_KEY, UNIT_KEY = "quantity", "unit"
UNIT_NOTE = "the unit a line is counted in (kg, hours, days); printed after its quantity, as in 12 kg"


def cell_path(path: str, row_number: int, key: str) -> str:
    """How a blank cell is named: ``data.line_items[row 3].total``."""
    return f"{path}[row {row_number}].{key}"


def _cell(row: Any, key: str, index: int) -> Any:
    """A row's value for one column, read as the renderers read it."""
    if isinstance(row, dict):
        return row.get(key, MISSING)
    if isinstance(row, (list, tuple)):
        return row[index] if index < len(row) else MISSING
    return row if index == 0 else MISSING


def _unit(row: Any) -> str:
    unit = row.get(UNIT_KEY) if isinstance(row, dict) else None
    return unit.strip() if isinstance(unit, str) else ""


def cell_text(row: Any, key: str, index: int, currency: str = "") -> str:
    """A cell as the renderers print it: an amount with two decimals in the kit's currency (F347,
    PRD-255 FR-7), and a quantity followed by the row's unit when it has one (F369: "12 kg")."""
    value = _cell(row, key, index)
    text = field_text(key, "" if value is MISSING else value, currency)
    unit = _unit(row) if key == QUANTITY_KEY else ""
    if not unit or not text.strip() or text.casefold().endswith(unit.casefold()):
        return text  # no unit, no quantity, or the quantity already says it ("12 kg")
    return f"{text} {unit}"


def unfilled_cells(block: DataTableBlock, rows: List[Any]) -> List[str]:
    """The cells of ``rows`` a required column of ``block`` leaves missing or blank. Pure."""
    found: List[str] = []
    for number, row in enumerate(rows, start=1):
        for index, column in enumerate(block.columns):
            if column.optional:
                continue
            value = _cell(row, column.key, index)
            if value is MISSING or is_blank(value):
                found.append(cell_path(block.path, number, column.key))
    if len(found) <= MAX_NAMED_CELLS:
        return found
    rest = len(found) - MAX_NAMED_CELLS
    return [*found[:MAX_NAMED_CELLS], MORE_CELLS.format(path=block.path, count=rest)]


__all__ = ["MAX_NAMED_CELLS", "QUANTITY_KEY", "UNIT_KEY", "UNIT_NOTE", "cell_path", "cell_text", "unfilled_cells"]
