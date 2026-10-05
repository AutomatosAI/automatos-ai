"""Every cell of a ``data_table`` is checked, not only that the table has rows (F345).

F345 (night 10b): the finalisation guard held for an empty chip and for a table with
no rows, and nothing else. Three price lists went out with empty prices and invoices
with a blank Total cell: a row that lacked a column's key, or held "" or spaces
there, rendered as an empty cell and counted as filled.

Now each row is read against the table's columns, the way the renderers read it
(``_cell_value``: an object by key, a plain list by position). A missing key or a
blank cell is unresolved and named by row (counted from 1, as a reader counts) and
column, e.g. ``data.line_items[row 3].total``, so the owner and the agent know
exactly what to fill. A column the author marks ``optional`` may stay empty.

Pure: no IO.
"""
from __future__ import annotations

from typing import Any, List

from ..variables.catalog import is_blank
from .schema import DataTableBlock

# A long table with one empty column would name hundreds of cells: name the first
# few, then say how many more there are.
MAX_NAMED_CELLS = 20
MORE_CELLS = "{path}: {count} more blank cells"
MISSING = object()


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


__all__ = ["MAX_NAMED_CELLS", "cell_path", "unfilled_cells"]
