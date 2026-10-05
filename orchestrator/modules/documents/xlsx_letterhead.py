"""A branded spreadsheet's letterhead, table look and printed page (F356, 5 Oct).

F356: the seeded "Data Export" starter is a spreadsheet, and next to the Branded
documents it was a logo over a bare grid. With a brand kit, the sheet now has
what the documents have, from the same design system (``blocks.design_tokens``):

* under the logo, the title in the kit's ``heading`` colour (PRD-255: not the
  primary) and a line with the company and the date (in the kit's date style);
* zebra rows on the kit's ``surface``, ``rule`` hairlines between rows, numbers
  with thousands separators and two decimals when they carry decimals;
* the header row frozen and filterable;
* printed: landscape, one page wide, the header row repeated on every page, and
  a footer with the company, the title and "Page X of Y".

Without a kit the sheet is exactly what it was (``xlsx_render``).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Mapping, Optional

from .blocks import design_tokens as tokens
from .locale_text import date_style_of, long_date

DECIMAL_FORMAT = "#,##0.00"
HAIR_BORDER = 7  # XlsxWriter's hairline border style
FOOTER_TEXT_PT = 8
LETTERHEAD_ROWS = 3  # the title, the company line, a blank row


def kit_of(brand: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """``brand`` (``brand_rules.brand_assets``) as the kit the design tokens read."""
    colours = (brand or {}).get("colours") or {}
    return {f"{role}_color": colours.get(role) for role in ("primary", "secondary", "accent", "text")}


def branded_formats(workbook: Any, brand: Mapping[str, Any], font: Mapping[str, str]) -> Dict[str, Any]:
    """The letterhead's and the body rows' formats for a kit."""
    design = tokens.design(kit_of(brand))
    roles, steps = design.palette, design.type
    row = {"bottom": HAIR_BORDER, "bottom_color": roles.rule, **font}
    return {
        "title": workbook.add_format({"bold": steps["h2"].bold, "font_size": steps["h2"].size_pt,
                                      "font_color": roles.heading, **font}),
        "company": workbook.add_format({"font_size": steps["small"].size_pt, "font_color": roles.muted, **font}),
        "body": workbook.add_format(row),
        "zebra": workbook.add_format({**row, "bg_color": roles.surface}),
        "decimal": workbook.add_format({**row, "num_format": DECIMAL_FORMAT}),
        "decimal_zebra": workbook.add_format({**row, "num_format": DECIMAL_FORMAT, "bg_color": roles.surface}),
    }


def write_letterhead(worksheet: Any, top: int, title: str, brand: Mapping[str, Any], formats: Mapping[str, Any]) -> int:
    """The title and the company line from row ``top``; the row the table's header goes on."""
    today = datetime.utcnow()
    company = str(brand.get("name") or "").strip()
    dated = long_date(today, date_style_of(brand))
    worksheet.write_string(top, 0, title, formats["title"])
    worksheet.write_string(top + 1, 0, f"{company}  ·  {dated}" if company else dated, formats["company"])
    return top + LETTERHEAD_ROWS


def body_format(formats: Mapping[str, Any], row_number: int, value: Any) -> Any:
    """A body cell's format: zebra on every second row, two decimals for a number that has them."""
    zebra = row_number % 2 == 1
    if isinstance(value, float) and not value.is_integer():
        return formats["decimal_zebra" if zebra else "decimal"]
    return formats["zebra" if zebra else "body"]


def _footer_text(text: str) -> str:
    return text.replace("&", "&&")


def set_print_layout(worksheet: Any, header_row: int, columns: int, title: str, brand: Mapping[str, Any]) -> None:
    """Freeze and filter the header; print landscape, one page wide, with a footer."""
    worksheet.freeze_panes(header_row + 1, 0)
    worksheet.autofilter(header_row, 0, header_row, max(columns - 1, 0))
    worksheet.set_landscape()
    worksheet.fit_to_pages(1, 0)
    worksheet.repeat_rows(header_row)
    company = _footer_text(str(brand.get("name") or ""))
    size = f"&{FOOTER_TEXT_PT}"
    worksheet.set_footer(f"&L{size}{company}&C{size}{_footer_text(title)}&R{size}Page &P of &N")


__all__ = ["body_format", "branded_formats", "kit_of", "set_print_layout", "write_letterhead"]
