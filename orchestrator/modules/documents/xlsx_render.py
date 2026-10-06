"""A generated spreadsheet in the brand's colours, font and logo (brand kit at generation, night 10 prep).

The PDF, Word and social renders read the workspace brand kit; the spreadsheet did
not: its header row was hard-coded "#1a1a2e" with white text, whatever the kit said,
and it had no logo. :func:`write_xlsx` writes the same sheet (headers, rows typed by
value, columns sized to their text) with the kit's look when there is one
(``services.brand_rules.brand_assets``):

* PRD-255 (US-005): the header row on the kit's ``surface_2`` with ``heading``
  text, or on the ``accent`` (its text white where white reads) only when the kit's
  ``accent_use`` is ``bold``: no longer the primary on every sheet;
* every cell in the kit's body font and body size (``type_scale.body``), an amount
  column in the kit's currency (``xlsx_letterhead``);
* the logo above the table at the kit's letterhead height, when the kit has an
  uploaded logo or a public one;
* F356: under the logo the title and a company line, zebra rows and hairlines,
  a frozen, filterable header, and a printed footer (``xlsx_letterhead``).

Without a kit (``brand`` None) the sheet is exactly what it was.

F370 (night 10c): the sheet's title (its name, the letterhead's title line and the
printed header) was the request's title, never the data's: "Wholesale price list
XLSX" over a list whose ``data.title`` said otherwise. It is now ``data.title`` when
the data gives one (the request's title fills it in when it does not), and the
sheet's name is cleaned of the characters Excel refuses in one.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .amounts import is_amount_key
from .blocks.design_tokens import Design
from .xlsx_letterhead import (
    body_format, branded_formats, cell_font, logo_px, logo_rows, set_print_layout, sheet_design, write_letterhead,
)

DEFAULT_HEADER_FILL = "#1a1a2e"
DEFAULT_HEADER_TEXT = "white"
SHEET_NAME_MAX_CHARS = 31  # Excel's limit
# The characters Excel refuses in a sheet's name; a name may not start or end with an apostrophe.
SHEET_NAME_REFUSED = re.compile(r"[\[\]:*?/\\]")
SHEET_NAME_FALLBACK = "Sheet1"
COLUMN_PADDING = 2
COLUMN_MAX_WIDTH = 50
DATE_FORMAT = "yyyy-mm-dd"
LOGO_FILE_NAME = "logo"


def header_colours(design: Optional[Design]) -> Tuple[str, str]:
    """(fill, text) for the header row: the kit's table header roles (``surface_2`` and
    ``heading``, the accent only under ``bold``), or the plain sheet's without a kit."""
    if design is None:
        return DEFAULT_HEADER_FILL, DEFAULT_HEADER_TEXT
    return design.palette.header_fill, design.palette.header_text


def logo_image(brand: Optional[Mapping[str, Any]], design: Design) -> Optional[Tuple[Any, float]]:
    """(the logo's bytes, the scale that makes it the kit's letterhead height), or None."""
    from modules.documents.blocks.docx_renderer import _safe_image_bytes
    from modules.documents.brand_logo import image_dimensions

    source = (brand or {}).get("logo") or ""
    data = _safe_image_bytes(source) if source else None
    size = image_dimensions(data.getvalue()) if data is not None else None
    if not size or not size[1]:
        return None
    return data, logo_px(design) / size[1]


def _write_cell(worksheet: Any, row: int, col: int, value: Any, formats: Mapping[str, Any], cell: Any = None) -> None:
    """One body cell; ``cell`` is its format when the sheet is branded (F356), else the plain one."""
    if isinstance(value, (int, float)):
        worksheet.write_number(row, col, value, cell or formats["body"])
    elif isinstance(value, datetime):
        worksheet.write_datetime(row, col, value, formats["date"])
    else:
        worksheet.write_string(row, col, str(value) if value is not None else "", cell or formats["body"])


def _widths(columns: Sequence[Any], rows: Sequence[Sequence[Any]]) -> List[int]:
    widths = []
    for col, name in enumerate(columns):
        values = [str(r[col]) if col < len(r) and r[col] is not None else "" for r in rows]
        longest = max(len(str(name)), max((len(v) for v in values), default=0))
        widths.append(min(longest + COLUMN_PADDING, COLUMN_MAX_WIDTH))
    return widths


def _formats(workbook: Any, brand: Optional[Mapping[str, Any]], design: Optional[Design]) -> Dict[str, Any]:
    fill, text = header_colours(design)
    font = cell_font(brand, design) if design is not None else {}
    return {
        "header": workbook.add_format({"bold": True, "bg_color": fill, "font_color": text, "border": 1,
                                       "text_wrap": True, **font}),
        "body": workbook.add_format(dict(font)),
        "date": workbook.add_format({"num_format": DATE_FORMAT, **font}),
    }


def _place_logo(worksheet: Any, brand: Optional[Mapping[str, Any]], design: Design) -> int:
    """Put the logo above the table; the row the table starts on."""
    logo = logo_image(brand, design)
    if logo is None:
        return 0
    data, scale = logo
    worksheet.insert_image(0, 0, LOGO_FILE_NAME, {"image_data": data, "x_scale": scale, "y_scale": scale})
    return logo_rows(design)


def _write_rows(worksheet: Any, columns: Sequence[Any], rows: Sequence[Sequence[Any]], top: int,
                formats: Mapping[str, Any], branded: Optional[Mapping[str, Any]]) -> None:
    """The body rows under the header at ``top``; zebra, hairlines and amounts in the kit's
    currency when ``branded`` formats are given."""
    amounts = [is_amount_key(name) for name in columns]
    for number, row in enumerate(rows):
        for col_idx, value in enumerate(row[:len(columns)]):
            cell = body_format(branded, number, value, amounts[col_idx]) if branded else None
            _write_cell(worksheet, top + 1 + number, col_idx, value, formats, cell)


def sheet_title(title: str, data: Mapping[str, Any]) -> str:
    """The sheet's title: ``data.title`` when the data gives one, else the request's ``title`` (F370). Pure."""
    given = data.get("title")
    return given.strip() if isinstance(given, str) and given.strip() else title


def sheet_name(title: str) -> str:
    """``title`` as Excel takes a sheet's name: no square brackets, colon, star, question mark or slash, no edge apostrophe, 31 characters. Pure."""
    name = SHEET_NAME_REFUSED.sub(" ", title)[:SHEET_NAME_MAX_CHARS].strip().strip("'").strip()
    return name or SHEET_NAME_FALLBACK


def write_xlsx(path: str, title: str, data: Mapping[str, Any], brand: Optional[Mapping[str, Any]]) -> None:
    """Write ``data``'s ``columns`` and ``rows`` to ``path`` as one sheet, titled ``data.title`` (else ``title``)."""
    import xlsxwriter

    columns, rows = data.get("columns") or [], data.get("rows") or []
    if not columns:
        raise ValueError("XLSX generation requires 'columns' in data.")
    title = sheet_title(title, data)
    workbook = xlsxwriter.Workbook(path)
    worksheet = workbook.add_worksheet(sheet_name(title))
    design = sheet_design(brand) if brand else None
    formats = _formats(workbook, brand, design)
    top, branded = 0, None
    if brand:
        branded = branded_formats(workbook, brand, design)
        top = write_letterhead(worksheet, _place_logo(worksheet, brand, design), title, brand, branded)
    for col, name in enumerate(columns):
        worksheet.write(top, col, name, formats["header"])
    _write_rows(worksheet, columns, rows, top, formats, branded)
    for col, width in enumerate(_widths(columns, rows)):
        worksheet.set_column(col, col, width)
    if brand:
        set_print_layout(worksheet, top, len(columns), title, brand)
    workbook.close()


__all__ = ["header_colours", "logo_image", "sheet_name", "sheet_title", "write_xlsx"]
