"""Word tables that look like the PDF's, and blocks kept together (F356, 5 Oct).

F356: a block document's Word tables wore Word's "Light Grid Accent 1" (theme
blue grid lines), whatever the kit said. They now follow the PDF's table rules
(``page_style``), from the same ``design_tokens``:

* a header row repeated on every page;
* body rows with a hairline under each and zebra tints, no row split over pages;
* a data table's columns aligned as the template says (numbers right);
* an invoice's totals and a proposal's total: no fill, the last row bold over a
  rule; an agreement's signatures: a plain header over a rule;
* a report's KPIs (``data.kpis``) as one row of tiles, as in the PDF.

PRD-255 (US-004): the colours are the kit's roles (``design_tokens``): the header
row on ``surface_2`` in ``heading`` (on the accent, its text white where white
reads, only when ``accent_use`` is ``bold``), zebra rows on ``surface``, hairlines
in ``rule``, strong rules in ``heading``; KPI tiles on ``surface`` with the figure
in the accent; sizes from the kit's type scale.

:func:`keep_together` keeps a run of blocks on one page (the Agreement's last
clause with its signatures).
"""
from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence

from . import design_tokens as t
from .docx_style import cell_edges, full_width, rgb, text_width

HAIRLINE_EIGHTHS = 4
RULE_EIGHTHS = 8
TOTALS_IDS = frozenset({"totals", "pricing-total"})
SIGNATURES_ID = "signatures"
KPIS_ID = "kpis"
NO_EDGE = ("#ffffff", 0)


def _set_on(parent: Any, tag: str, attrs: Optional[Mapping[str, str]] = None) -> None:
    """Add a ``tag`` child to ``parent`` once (a row is never marked twice)."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    if not attrs and parent.find(qn(tag)) is not None:
        return
    element = OxmlElement(tag)
    for key, value in (attrs or {}).items():
        element.set(qn(key), value)
    parent.append(element)


def shade(cell: Any, fill: str) -> None:
    """Fill a cell (after its borders, as Word's schema orders them), in upper-case hex as Word writes it."""
    _set_on(cell._tc.get_or_add_tcPr(), "w:shd",
            {"w:val": "clear", "w:color": "auto", "w:fill": fill.lstrip("#").upper()})


def _row_props(row: Any, header: bool) -> None:
    """A row that never splits over a page; a header row repeats on every page."""
    tr_pr = row._tr.get_or_add_trPr()
    _set_on(tr_pr, "w:cantSplit")
    if header:
        _set_on(tr_pr, "w:tblHeader")


def _runs(cell: Any, *, bold: Optional[bool] = None, colour: Optional[str] = None, size: Optional[float] = None) -> None:
    from docx.shared import Pt

    for paragraph in cell.paragraphs:
        paragraph.paragraph_format.space_after = Pt(0)
        for run in paragraph.runs:
            if bold is not None:
                run.bold = bold
            if colour:
                run.font.color.rgb = rgb(colour)
            if size:
                run.font.size = Pt(size)


def _align(cell: Any, align: str) -> None:
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    where = {"right": WD_ALIGN_PARAGRAPH.RIGHT, "center": WD_ALIGN_PARAGRAPH.CENTER}.get(align, WD_ALIGN_PARAGRAPH.LEFT)
    for paragraph in cell.paragraphs:
        paragraph.alignment = where


def _header_row(row: Any, design: t.Design) -> None:
    roles = design.palette
    for cell in row.cells:
        cell_edges(cell, {"bottom": NO_EDGE})
        shade(cell, roles.header_fill)
        _runs(cell, bold=True, colour=roles.header_text, size=design.type["small"].size_pt)
    _row_props(row, header=True)


def _body_row(row: Any, roles: t.Palette, number: int) -> None:
    for cell in row.cells:
        cell_edges(cell, {"bottom": (roles.rule, HAIRLINE_EIGHTHS)})
        if number % 2 == 1:
            shade(cell, roles.surface)
        _runs(cell)
    _row_props(row, header=False)


def style_table(table: Any, kit: Mapping[str, Any], header: bool, aligns: Sequence[str] = ()) -> None:
    """A table in the PDF's look: a filled header row, hairlines, zebra rows, columns aligned."""
    design = t.design(kit)
    full_width(table, text_width(design))
    rows = list(table.rows)
    for index, row in enumerate(rows):
        if header and index == 0:
            _header_row(row, design)
        else:
            _body_row(row, design.palette, index - (1 if header else 0))
        for cell, align in zip(row.cells, aligns):
            _align(cell, align)


def style_totals(table: Any, kit: Mapping[str, Any]) -> None:
    """Totals: amounts right-aligned, no fill, the last row bold over a rule in ``heading``."""
    design = t.design(kit)
    roles = design.palette
    full_width(table, text_width(design))
    rows = list(table.rows)
    for index, row in enumerate(rows):
        last = index == len(rows) - 1
        for cell in row.cells:
            edges = {"top": (roles.heading, RULE_EIGHTHS)} if last else {"bottom": (roles.rule, HAIRLINE_EIGHTHS)}
            cell_edges(cell, edges)
            _runs(cell, bold=True if last else None)
        _align(row.cells[-1], "right")
        _row_props(row, header=False)


def style_signatures(table: Any, kit: Mapping[str, Any]) -> None:
    """Signatures: a plain header row in ``heading`` over a rule; the lines below open."""
    design = t.design(kit)
    roles = design.palette
    full_width(table, text_width(design))
    for index, row in enumerate(table.rows):
        for cell in row.cells:
            cell_edges(cell, {"bottom": (roles.heading, RULE_EIGHTHS)} if index == 0 else {"bottom": NO_EDGE})
            _runs(cell, bold=True if index == 0 else None, colour=roles.heading if index == 0 else None)
        _row_props(row, header=index == 0)


def kpi_tiles(document: Any, tiles: List[Sequence[str]], kit: Mapping[str, Any], font: Optional[str]) -> Any:
    """``tiles`` (label, value, change) as one row of tiles: a cell on ``surface`` each, the figure in the accent."""
    from docx.shared import Pt

    design = t.design(kit)
    roles, steps = design.palette, design.type
    table = document.add_table(rows=1, cols=len(tiles))
    full_width(table, text_width(design))
    sizes = ((steps["small"].size_pt, False, roles.muted), (steps["h1"].size_pt, steps["h1"].bold, roles.accent),
             (steps["caption"].size_pt, False, roles.muted))
    for cell, tile in zip(table.rows[0].cells, tiles):
        cell_edges(cell, {"top": NO_EDGE})
        shade(cell, roles.surface)
        cell.paragraphs[0]._p.getparent().remove(cell.paragraphs[0]._p)
        for text, (size, bold, colour) in zip(tile, sizes):
            run = cell.add_paragraph().add_run(text)
            run.font.size, run.bold, run.font.color.rgb = Pt(size), bold, rgb(colour)
            if font:
                run.font.name = font
        _runs(cell)
    return table


def keep_together(document: Any, start: int) -> None:
    """Keep every block written after body element ``start`` on one page with the next."""
    from docx.oxml.ns import qn
    from docx.text.paragraph import Paragraph

    for element in list(document.element.body)[start:]:
        for row in element.iter(qn("w:tr")):
            _set_on(row.get_or_add_trPr(), "w:cantSplit")
        for paragraph in element.iter(qn("w:p")):
            Paragraph(paragraph, None).paragraph_format.keep_with_next = True


__all__ = ["KPIS_ID", "SIGNATURES_ID", "TOTALS_IDS", "keep_together", "kpi_tiles", "shade", "style_signatures",
           "style_table", "style_totals"]
