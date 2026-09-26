"""F181 (night 6): a spreadsheet the owner uploads is copied into the workspace,
where an agent counts and totals it with code.

Ticket #1115 asked how many bags the club ordered and whether the roastery was
short. It answered 47.06 kg from the first pages of the export. The file (440
rows, 380 of them on 5 October) says 341 bags, about 100.3 kg of green coffee,
and about 40.3 kg short. The file lived only in the knowledge base, read a page
of chunks at a time, and python3 in the workspace had nothing to open ("File
not found").

Now the upload writes documents/<name> into the workspace: a CSV as it came, an
Excel file as one CSV per sheet (the worker writes text only). read_document
names that copy with a row count made in code. A missing copy (a file uploaded
before this, or a copy that failed) is made from the stored upload. A copy
that cannot be made never fails the upload or the read.
"""
from __future__ import annotations

import asyncio
import csv
import io
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DOCUMENTS_DIR = "documents"
# documents.file_type values that are tables (api.documents.UPLOAD_FILE_TYPES).
SPREADSHEET_TYPES = frozenset({"csv", "spreadsheet"})
# A workspace file name: letters, digits, dot, dash and underscore, so a shell
# command can use the path as it is.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
COUNT_WITH_CODE = ("This is a spreadsheet. Count or total it with code on workspace_path (python3's csv "
                   "module, or awk), never from these pages: a page is a slice of its rows.")


def _safe(name: str) -> str:
    return _UNSAFE.sub("_", name).strip("._") or "sheet"


def _csv_text(content: bytes) -> str:
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return content.decode("latin-1")


def _sheets_as_csv(content: bytes) -> List[Tuple[str, str]]:
    """(sheet name, CSV text) for each sheet of an Excel workbook, values as the
    workbook last computed them."""
    import openpyxl

    workbook = openpyxl.load_workbook(io.BytesIO(content), read_only=True, data_only=True)
    try:
        sheets = []
        for sheet in workbook.worksheets:
            out = io.StringIO()
            writer = csv.writer(out, lineterminator="\n")
            for row in sheet.iter_rows(values_only=True):
                writer.writerow(["" if value is None else value for value in row])
            sheets.append((sheet.title, out.getvalue()))
        return sheets
    finally:
        workbook.close()


def spreadsheet_tables(filename: str, content: bytes) -> List[Tuple[str, str]]:
    """(workspace path, CSV text) for each table in the file: one for a CSV,
    one per sheet for an Excel workbook (the one sheet of a single-sheet book
    keeps the file's name)."""
    stem, suffix = Path(filename).stem, Path(filename).suffix.lower()
    if suffix == ".csv":
        return [(f"{DOCUMENTS_DIR}/{_safe(stem)}.csv", _csv_text(content))]
    if suffix in (".xlsx", ".xlsm"):
        sheets = _sheets_as_csv(content)
        if len(sheets) == 1:
            return [(f"{DOCUMENTS_DIR}/{_safe(stem)}.csv", sheets[0][1])]
        return [(f"{DOCUMENTS_DIR}/{_safe(stem)}.{_safe(title)}.csv", text) for title, text in sheets]
    return []


def row_count(text: str) -> int:
    """Data rows in a CSV text, counted in code: every non-blank row after the header."""
    rows = [row for row in csv.reader(io.StringIO(text)) if any(cell.strip() for cell in row)]
    return max(0, len(rows) - 1)


def _counted_tables(filename: str, content: bytes) -> List[Tuple[str, str, int]]:
    """(workspace path, CSV text, row count) for each table: the parsing and
    counting, done off the event loop by the callers (F105)."""
    return [(path, text, row_count(text)) for path, text in spreadsheet_tables(filename, content)]


async def copy_to_workspace(workspace_id: Any, filename: str, content: bytes,
                            only_missing: bool = False) -> List[Dict[str, Any]]:
    """Write the file's tables into the workspace; ``[{workspace_path, row_count}]``
    for each one there. ``only_missing`` writes just the tables not there yet.
    Never raises: a table that cannot be written is left out."""
    try:
        tables = await asyncio.to_thread(_counted_tables, filename, content)
    except Exception:  # noqa: BLE001 -- an unreadable workbook: nothing to copy
        logger.warning("[F181] could not read %s as a spreadsheet", filename, exc_info=True)
        return []
    if not tables:
        return []
    from core.workspace_client import WorkspaceClient

    client = WorkspaceClient(str(workspace_id))
    present = await _present(client) if only_missing else set()
    copied = []
    for path, text, rows in tables:
        if path not in present:
            written = await client.write_file(path, text)
            if written.get("success") is False:
                logger.warning("[F181] could not copy %s to %s: %s", filename, path, written.get("error"))
                continue
        copied.append({"workspace_path": path, "row_count": rows})
    return copied


async def _present(client: Any) -> set:
    listing = await client.list_dir(DOCUMENTS_DIR)
    entries = listing.get("entries") if isinstance(listing, dict) else None
    return {f"{DOCUMENTS_DIR}/{entry.get('name')}" for entry in entries or [] if isinstance(entry, dict)}


async def workspace_copies(workspace_id: Any, document: Any) -> Optional[List[Dict[str, Any]]]:
    """``[{workspace_path, row_count}]`` for a spreadsheet document, its copies
    made from the stored upload where missing; None when it is not a
    spreadsheet or its upload is gone."""
    if getattr(document, "file_type", None) not in SPREADSHEET_TYPES:
        return None
    stored = Path(str(getattr(document, "file_path", None) or ""))
    if not stored.is_file():
        return None
    name = getattr(document, "original_filename", None) or getattr(document, "filename", None) or stored.name
    content = await asyncio.to_thread(stored.read_bytes)
    return await copy_to_workspace(workspace_id, name, content, only_missing=True)
