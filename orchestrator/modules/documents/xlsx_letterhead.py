"""A branded spreadsheet's letterhead, table look and printed page (F356, 5 Oct).

F356: the seeded "Data Export" starter is a spreadsheet, and next to the Branded
documents it was a logo over a bare grid. With a brand kit, the sheet now has
what the documents have, from the same design system (``blocks.design_tokens``):

* under the logo, the title in the kit's ``heading`` colour (PRD-255: not the
  primary) and a line with the company and the date (in the kit's date style);
* zebra rows on the kit's ``surface``, ``rule`` hairlines between rows, numbers
  with thousands separators and two decimals when they carry decimals;
* PRD-255 (US-005): every cell in the kit's body size (``type_scale.body``), an
  amount column (its header's last word, ``amounts.is_amount_key``) in the kit's
  currency with two decimals ("£#,##0.00"; a kit without one keeps the plain
  number), and the logo at the kit's letterhead height (``logo_rules``);
* the header row frozen and filterable;
* printed: landscape, one page wide, the header row repeated on every page, and
  a footer with the company, the title and "Page X of Y".

Without a kit the sheet is exactly what it was (``xlsx_render``).
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, Mapping, Optional

from .blocks import design_tokens as tokens
from .locale_text import currency_of, currency_prefix, date_style_of, long_date

DECIMAL_FORMAT = "#,##0.00"
HAIR_BORDER = 7  # XlsxWriter's hairline border style
FOOTER_TEXT_PT = 8
LETTERHEAD_ROWS = 3  # the title, the company line, a blank row
KIT_COLOURS = ("primary", "secondary", "accent", "text")
# The design fields ``brand_rules.brand_assets`` carries as the kit stores them.
KIT_DESIGN_KEYS = ("palette", "accent_use", "type_scale", "logo_rules")
# Excel draws an image at 96 px to the inch, and a default row is 15 pt (20 px) tall.
PX_PER_MM = 96 / 25.4
DEFAULT_ROW_PX = 20


def kit_of(brand: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """``brand`` (``brand_rules.brand_assets``) as the kit the design tokens read."""
    assets = brand or {}
    colours = assets.get("colours") or {}
    design = {key: assets[key] for key in KIT_DESIGN_KEYS if assets.get(key) is not None}
    return {**{f"{role}_color": colours.get(role) for role in KIT_COLOURS}, **design}


def sheet_design(brand: Optional[Mapping[str, Any]]) -> tokens.Design:
    """The design system a branded sheet prints with (``blocks.design_tokens``)."""
    return tokens.design(kit_of(brand))


def cell_font(brand: Optional[Mapping[str, Any]], design: tokens.Design) -> Dict[str, Any]:
    """Every cell's font: the kit's body family and the type scale's body size."""
    family = ((brand or {}).get("fonts") or {}).get("body")
    return {**({"font_name": family} if family else {}), "font_size": design.type["body"].size_pt}


def money_format(brand: Optional[Mapping[str, Any]]) -> Optional[str]:
    """An amount's number format in the kit's currency ('"£"#,##0.00'); None for a kit without one."""
    prefix = currency_prefix(currency_of(brand))
    return f'"{prefix}"{DECIMAL_FORMAT}' if prefix else None


def logo_px(design: tokens.Design) -> float:
    """The logo's height on the sheet, in pixels: the kit's letterhead height."""
    return design.logo_mm * PX_PER_MM


def logo_rows(design: tokens.Design) -> int:
    """The rows above the letterhead that hold the logo and its clear space."""
    return math.ceil((design.logo_mm + design.logo_clear_mm) * PX_PER_MM / DEFAULT_ROW_PX)


def _number_formats(workbook: Any, row: Mapping[str, Any], surface: str, money: Optional[str]) -> Dict[str, Any]:
    """The decimal formats, and the currency ones when the kit has a currency, plain and zebra."""
    kinds = {"decimal": DECIMAL_FORMAT, **({"money": money} if money else {})}
    formats = {}
    for kind, number in kinds.items():
        formats[kind] = workbook.add_format({**row, "num_format": number})
        formats[f"{kind}_zebra"] = workbook.add_format({**row, "num_format": number, "bg_color": surface})
    return formats


def branded_formats(workbook: Any, brand: Mapping[str, Any], design: tokens.Design) -> Dict[str, Any]:
    """The letterhead's and the body rows' formats for a kit."""
    roles, steps, font = design.palette, design.type, cell_font(brand, design)
    family = {key: value for key, value in font.items() if key == "font_name"}
    row = {"bottom": HAIR_BORDER, "bottom_color": roles.rule, **font}
    return {
        "title": workbook.add_format({"bold": steps["h2"].bold, "font_size": steps["h2"].size_pt,
                                      "font_color": roles.heading, **family}),
        "company": workbook.add_format({"font_size": steps["small"].size_pt, "font_color": roles.muted, **family}),
        "body": workbook.add_format(row),
        "zebra": workbook.add_format({**row, "bg_color": roles.surface}),
        **_number_formats(workbook, row, roles.surface, money_format(brand)),
    }


def write_letterhead(worksheet: Any, top: int, title: str, brand: Mapping[str, Any], formats: Mapping[str, Any]) -> int:
    """The title and the company line from row ``top``; the row the table's header goes on."""
    today = datetime.utcnow()
    company = str(brand.get("name") or "").strip()
    dated = long_date(today, date_style_of(brand))
    worksheet.write_string(top, 0, title, formats["title"])
    worksheet.write_string(top + 1, 0, f"{company}  ·  {dated}" if company else dated, formats["company"])
    return top + LETTERHEAD_ROWS


def body_format(formats: Mapping[str, Any], row_number: int, value: Any, amount: bool = False) -> Any:
    """A body cell's format: zebra on every second row, the kit's currency for a number in an
    ``amount`` column (when the kit has one), two decimals for a number that has them."""
    suffix = "_zebra" if row_number % 2 == 1 else ""
    number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if number and amount and "money" in formats:
        return formats[f"money{suffix}"]
    if isinstance(value, float) and not value.is_integer():
        return formats[f"decimal{suffix}"]
    return formats["zebra" if suffix else "body"]


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


__all__ = [
    "body_format", "branded_formats", "cell_font", "kit_of", "logo_px", "logo_rows", "money_format",
    "set_print_layout", "sheet_design", "write_letterhead",
]
