"""A generated spreadsheet in the brand's colours, font and logo (brand kit at generation, night 10 prep).

The PDF, Word and social renders read the workspace brand kit; the spreadsheet did
not: its header row was hard-coded "#1a1a2e" with white text, whatever the kit said,
and it had no logo. :func:`write_xlsx` writes the same sheet (headers, rows typed by
value, columns sized to their text) with the kit's look when there is one
(``services.brand_rules.brand_assets``):

* the header row filled with the primary colour, its text white or the kit's text
  colour, whichever reads on it;
* every cell in the kit's body font;
* the logo above the table, when the kit has an uploaded logo or a public one.

Without a kit (``brand`` None) the sheet is exactly what it was.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from core.brand_palette import WHITE, contrast, parse_hex

DEFAULT_HEADER_FILL = "#1a1a2e"
DEFAULT_HEADER_TEXT = "white"
HEADER_TEXT_MIN_CONTRAST = 4.5
SHEET_NAME_MAX_CHARS = 31  # Excel's limit
COLUMN_PADDING = 2
COLUMN_MAX_WIDTH = 50
DATE_FORMAT = "yyyy-mm-dd"
LOGO_HEIGHT_PX = 48
LOGO_ROWS = 3  # the rows above the table that hold the logo
LOGO_FILE_NAME = "logo"


def header_colours(brand: Optional[Mapping[str, Any]]) -> Tuple[str, str]:
    """(fill, text) for the header row: the kit's primary, and text that reads on it."""
    colours = (brand or {}).get("colours") or {}
    fill = parse_hex(colours.get("primary"))
    if fill is None:
        return DEFAULT_HEADER_FILL, DEFAULT_HEADER_TEXT
    text = colours.get("text")
    if contrast(fill, WHITE) >= HEADER_TEXT_MIN_CONTRAST or parse_hex(text) is None:
        return colours["primary"], DEFAULT_HEADER_TEXT
    return colours["primary"], text


def _font(brand: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    body = ((brand or {}).get("fonts") or {}).get("body")
    return {"font_name": body} if body else {}


def logo_image(brand: Optional[Mapping[str, Any]]) -> Optional[Tuple[Any, float]]:
    """(the logo's bytes, the scale that makes it LOGO_HEIGHT_PX tall), or None."""
    from modules.documents.blocks.docx_renderer import _safe_image_bytes
    from modules.documents.brand_logo import image_dimensions

    source = (brand or {}).get("logo") or ""
    data = _safe_image_bytes(source) if source else None
    size = image_dimensions(data.getvalue()) if data is not None else None
    if not size or not size[1]:
        return None
    return data, LOGO_HEIGHT_PX / size[1]


def _write_cell(worksheet: Any, row: int, col: int, value: Any, formats: Mapping[str, Any]) -> None:
    if isinstance(value, (int, float)):
        worksheet.write_number(row, col, value, formats["body"])
    elif isinstance(value, datetime):
        worksheet.write_datetime(row, col, value, formats["date"])
    else:
        worksheet.write_string(row, col, str(value) if value is not None else "", formats["body"])


def _widths(columns: Sequence[Any], rows: Sequence[Sequence[Any]]) -> List[int]:
    widths = []
    for col, name in enumerate(columns):
        values = [str(r[col]) if col < len(r) and r[col] is not None else "" for r in rows]
        longest = max(len(str(name)), max((len(v) for v in values), default=0))
        widths.append(min(longest + COLUMN_PADDING, COLUMN_MAX_WIDTH))
    return widths


def _formats(workbook: Any, brand: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    fill, text = header_colours(brand)
    font = _font(brand)
    return {
        "header": workbook.add_format({"bold": True, "bg_color": fill, "font_color": text, "border": 1,
                                       "text_wrap": True, **font}),
        "body": workbook.add_format(dict(font)),
        "date": workbook.add_format({"num_format": DATE_FORMAT, **font}),
    }


def _place_logo(worksheet: Any, brand: Optional[Mapping[str, Any]]) -> int:
    """Put the logo above the table; the row the table starts on."""
    logo = logo_image(brand)
    if logo is None:
        return 0
    data, scale = logo
    worksheet.insert_image(0, 0, LOGO_FILE_NAME, {"image_data": data, "x_scale": scale, "y_scale": scale})
    return LOGO_ROWS


def write_xlsx(path: str, title: str, data: Mapping[str, Any], brand: Optional[Mapping[str, Any]]) -> None:
    """Write ``data``'s ``columns`` and ``rows`` to ``path`` as one sheet named ``title``."""
    import xlsxwriter

    columns, rows = data.get("columns") or [], data.get("rows") or []
    if not columns:
        raise ValueError("XLSX generation requires 'columns' in data.")
    workbook = xlsxwriter.Workbook(path)
    worksheet = workbook.add_worksheet(title[:SHEET_NAME_MAX_CHARS])
    formats = _formats(workbook, brand)
    top = _place_logo(worksheet, brand)
    for col, name in enumerate(columns):
        worksheet.write(top, col, name, formats["header"])
    for row_idx, row in enumerate(rows, top + 1):
        for col_idx, value in enumerate(row[:len(columns)]):
            _write_cell(worksheet, row_idx, col_idx, value, formats)
    for col, width in enumerate(_widths(columns, rows)):
        worksheet.set_column(col, col, width)
    workbook.close()


__all__ = ["header_colours", "logo_image", "write_xlsx"]
