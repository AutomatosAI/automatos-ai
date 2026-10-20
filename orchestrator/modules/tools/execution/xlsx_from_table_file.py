"""create_xlsx without pandas (PRD-262 US-025): a CSV or JSON file in the workspace, as a spreadsheet.

The tool read its source with pandas (``read_csv`` / ``read_json``) and wrote it with
``DataFrame.to_excel``. pandas left the image, so the source is read with the standard
library and written by the platform's spreadsheet writer,
``modules.documents.xlsx_render.write_xlsx`` (xlsxwriter), with no brand kit:

* CSV: the first row names the columns. A column whose every filled cell is a number
  is written as numbers, as pandas typed a numeric column; any other column stays text.
  An empty cell is a blank cell, and a row longer than the header is refused.
* JSON: a list of objects, one row each, with the columns in first-seen key order; or
  an object of columns, each a list or an object keyed by row, as ``read_json`` read
  them. A nested value is written as its JSON text.

``execute_document_tool`` is over AGENTS.md's function length, so this module serves
the tool from a decorator above it (:func:`an_xlsx_is_made_from_csv_or_json`) instead of
an edit inside it. A call whose paths that function refuses, and every other tool, goes
to it unchanged, so its refusals stay the only ones.
"""
from __future__ import annotations

import asyncio
import csv
import functools
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

Async = Callable[..., Awaitable[Dict[str, Any]]]
Table = Tuple[List[str], List[List[Any]]]

TOOL_NAME = "create_xlsx"
DEFAULT_SHEET_NAME = "Sheet1"
SOURCE_ERROR = "Source file must be CSV or JSON for XLSX creation"
JSON_SHAPE_ERROR = "JSON must be a list of objects (one per row) or an object of columns"
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", re.ASCII)
_INTEGER = re.compile(r"[+-]?\d+", re.ASCII)


def _number(text: str) -> Any:
    stripped = text.strip()
    return int(stripped) if _INTEGER.fullmatch(stripped) else float(stripped)


def _typed_column(cells: Sequence[str]) -> List[Any]:
    """A CSV column's cells: numbers when every filled cell is one, else text; an empty cell is None."""
    filled = [cell for cell in cells if cell != ""]
    numeric = bool(filled) and all(_NUMBER.fullmatch(cell.strip()) for cell in filled)
    return [None if cell == "" else (_number(cell) if numeric else cell) for cell in cells]


def read_csv_table(path: str) -> Table:
    """The columns and typed rows of the CSV file at ``path``. Empty lines are skipped, as pandas skipped them."""
    with open(path, encoding="utf-8-sig", newline="") as handle:
        records = [row for row in csv.reader(handle) if row]
    if not records:
        raise ValueError("No columns to parse from file")
    columns, body = records[0], records[1:]
    for number, row in enumerate(body, start=1):
        if len(row) > len(columns):
            raise ValueError(f"Data row {number} has {len(row)} fields; the header has {len(columns)}")
    padded = [row + [""] * (len(columns) - len(row)) for row in body]
    typed = [_typed_column([row[col] for row in padded]) for col in range(len(columns))]
    return columns, [list(row) for row in zip(*typed)]


def _cell(value: Any) -> Any:
    """A JSON value as a cell: a nested value as its JSON text, NaN or infinity as a blank."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _records_table(items: Sequence[Mapping[str, Any]]) -> Table:
    columns = list(dict.fromkeys(key for item in items for key in item))
    return columns, [[_cell(item.get(column)) for column in columns] for item in items]


def _columns_table(data: Mapping[str, Any]) -> Table:
    keyed = {name: dict(enumerate(values)) if isinstance(values, list) else values for name, values in data.items()}
    index = list(dict.fromkeys(key for values in keyed.values() for key in values))
    return list(keyed), [[_cell(values.get(key)) for values in keyed.values()] for key in index]


def read_json_table(path: str) -> Table:
    """The columns and rows of the JSON file at ``path``: records, or an object of columns."""
    with open(path, encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        return _records_table(data)
    if isinstance(data, dict) and all(isinstance(values, (list, dict)) for values in data.values()):
        return _columns_table(data)
    raise ValueError(JSON_SHAPE_ERROR)


_READERS: Mapping[str, Callable[[str], Table]] = {".csv": read_csv_table, ".json": read_json_table}


def create_xlsx(source_file: str, output_file: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """The tool's result for ``source_file`` (CSV or JSON) written to ``output_file`` as one sheet."""
    from xlsxwriter.exceptions import XlsxWriterException

    from modules.documents.xlsx_render import write_xlsx

    reader = next((read for suffix, read in _READERS.items() if source_file.endswith(suffix)), None)
    if reader is None:
        return {"success": False, "error": SOURCE_ERROR, "tool": TOOL_NAME}
    sheet = str(parameters.get("sheet_name") or DEFAULT_SHEET_NAME)
    try:
        columns, rows = reader(source_file)
        write_xlsx(output_file, sheet, {"columns": columns, "rows": rows}, None)
    # A JSON or encoding error is a ValueError; xlsxwriter wraps a file it cannot write
    # (a missing folder, say) in its own FileCreateError.
    except (OSError, ValueError, csv.Error, XlsxWriterException) as exc:
        logger.warning("create_xlsx could not convert %s: %s", source_file, exc)
        return {"success": False, "error": f"Document creation failed: {exc}", "tool": TOOL_NAME}
    return {
        "success": True,
        "action": TOOL_NAME,
        "params": parameters,
        "result": f"Excel file created successfully: {output_file}",
        "output_file": output_file,
    }


def xlsx_paths(executor: Any, tool_name: str, parameters: Mapping[str, Any]) -> Optional[Tuple[str, str]]:
    """A create_xlsx call's source and output, resolved inside the executor's workspace.

    None for another tool, a missing path or one outside the workspace: the same check
    ``execute_document_tool`` makes, which then answers those calls itself.
    """
    if tool_name != TOOL_NAME:
        return None
    source, output = parameters.get("source_file"), parameters.get("output_file")
    if not (isinstance(source, str) and source and isinstance(output, str) and output):
        return None
    workspace = Path(executor.workspace_dir).resolve()
    resolved = (workspace / source).resolve(), (workspace / output).resolve()
    if not all(path.is_relative_to(workspace) for path in resolved):
        return None
    return str(resolved[0]), str(resolved[1])


def an_xlsx_is_made_from_csv_or_json(execute: Async) -> Async:
    """Wrap ``exec_document.execute_document_tool``: create_xlsx is served here, off the event loop."""
    @functools.wraps(execute)
    async def wrapped(executor: Any, tool_name: str, parameters: Dict[str, Any], agent_id: int) -> Dict[str, Any]:
        paths = xlsx_paths(executor, tool_name, parameters)
        if paths is None:
            return await execute(executor, tool_name, parameters, agent_id)
        return await asyncio.to_thread(create_xlsx, paths[0], paths[1], parameters)
    return wrapped


__all__ = ["an_xlsx_is_made_from_csv_or_json", "create_xlsx", "read_csv_table", "read_json_table", "xlsx_paths"]
