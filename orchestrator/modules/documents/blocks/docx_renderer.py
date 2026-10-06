"""Block document → DOCX renderer (PRD-167 S2, Q71).

Compiles a :class:`BlockDocument` directly into a ``python-docx`` Document — the *same
block tree* that drives the PDF path. This removes the previous requirement that DOCX
generation needs a pre-uploaded ``.docx`` template file (Q71): a block template renders
to both PDF and DOCX from one source.

python-docx is pure-Python (no system libraries), so this path needs no Docker / native
deps to build or unit-test.

Brand kit (PRD-167 S4): the body font from ``brand.font_family`` — no hardcoded
Automatos styling.

F347: a text block keeps its paragraphs, line breaks, lists and bold/italic, read the
same way as for the PDF (``text_body``).

F356: the Word file matches the PDF. Its styles, page, letterhead (the first page's
header) and footer with page numbers are ``docx_style``; its tables, KPI tiles and
the Agreement's kept-together sign-off are ``docx_tables``; an optional part with
nothing to print is left out, as in the PDF (``optional_parts``).
"""

from __future__ import annotations

import base64
import binascii
import logging
import os
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..bundled_fonts import families
from ..locale_text import currency_of
from ..variables.catalog import walk_dynamic
from . import design_tokens as tokens
from .brand_board_docx import add_brand_part
from .docx_style import add_footer, add_letterhead, apply_styles, rgb
from .docx_tables import KPIS_ID, SIGNATURES_ID, TOTALS_IDS, keep_together, kpi_tiles, style_signatures, style_table, style_totals
from .letterhead_run import company_of, logo_of, split_letterhead
from .optional_parts import block_is_blank, row_is_blank
from .page_style import footer_name
from .schema import BlockDocument
from .table_cells import cell_text, unfilled_cells
from .text_body import BULLETED, MISSING, PARAGRAPH, Group, block_groups, kept_runs

logger = logging.getLogger(__name__)

_MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB cap on fetched images
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
# F347: a bulleted item takes Word's own bullet style; a numbered one is indented and
# carries its number as text (Word's numbered style counts on across separate lists).
DOCX_BULLET_STYLE = "List Bullet"
DOCX_LIST_INDENT_MM = 6.35
# F356: the Agreement's last clause and signatures, kept on one page.
KEEP_TOGETHER_IDS = frozenset({"sign-off"})
# F356: the lines of an address block sit tight, as in the PDF (page_starters).
TIGHT_IDS = frozenset({"bill-to-label", "bill-to", "bill-to-address", "to-name", "to-company", "sig-name"})
RIGHT_ALIGNED_IDS = frozenset({"date"})  # a letter's date, set right as in the PDF
DEFAULT_IMAGE_MM = 60


@dataclass
class RenderedDocx:
    document: object  # docx.document.Document (typed loosely to avoid import at module load)
    unresolved: List[str] = field(default_factory=list)


def _safe_local_image(src: str) -> Optional[BytesIO]:
    """Read a local/upload image, confined under the workspace document-storage root.

    Rejects path traversal (``../``, absolute paths escaping the root) and non-image
    extensions, so a template can't coerce an arbitrary-file read (e.g. ``/etc/passwd``)."""
    if os.path.splitext(src)[1].lower() not in _IMAGE_EXTS:
        return None
    try:
        from config import config

        root = Path(config.DOCUMENT_STORAGE_DIR).resolve()
    except Exception:  # noqa: BLE001 — config unavailable
        return None
    candidate = (root / src.lstrip("/\\")).resolve()
    if root != candidate and root not in candidate.parents:
        logger.warning("[DocxRender] refusing image path outside storage root: %r", src)
        return None
    try:
        with open(candidate, "rb") as fh:
            return BytesIO(fh.read(_MAX_IMAGE_BYTES + 1))
    except OSError:
        return None


def _inline_image_bytes(src: str) -> Optional[BytesIO]:
    """Decode a base64 ``data:image/…`` URI (PRD-242 S3 — the inlined brand logo).

    Same size cap as fetched images; anything that is not a base64 image URI
    (or does not decode) yields ``None`` so the caller falls back to alt text."""
    header, sep, payload = src.partition(",")
    if not sep or not header.startswith("data:image/") or ";base64" not in header:
        return None
    if len(payload) > _MAX_IMAGE_BYTES * 4 // 3 + 4:
        return None
    try:
        return BytesIO(base64.b64decode(payload, validate=True))
    except (ValueError, binascii.Error):
        return None


def _safe_image_bytes(src: str) -> Optional[BytesIO]:
    """Best-effort, SSRF-guarded image fetch for DOCX embedding.

    ``data:`` URIs (the inlined brand logo, PRD-242 S3) are decoded in-process;
    local/upload paths are read only from within the document-storage root
    (:func:`_safe_local_image`); http(s) URLs are fetched from the public address that
    was checked, each redirect hop checked again (F375, ``pinned_fetch``). Any failure
    returns ``None`` (the caller falls back to alt text)."""
    if not src:
        return None
    if src.startswith("data:"):
        return _inline_image_bytes(src)
    if not src.startswith(("http://", "https://")):
        return _safe_local_image(src)
    from modules.documents.pinned_fetch import FetchRefused, fetch_public

    try:
        return BytesIO(fetch_public(src, max_bytes=_MAX_IMAGE_BYTES).data)
    except FetchRefused as refused:  # F375: checked, pinned, every redirect hop checked again
        logger.warning("[DocxRender] image not fetched: %s", refused)
        return None


def _resolve_var(path: str, fallback, values: Dict[str, str], unresolved: List[str]) -> str:
    if path in values:
        return values[path]
    if fallback is not None:
        return fallback
    unresolved.append(path)
    return f"[[{path}]]"


def _add_inline(paragraph, content: list, values: Dict[str, str], unresolved: List[str], font: Optional[str]):
    for run_spec in kept_runs(content, values):
        if run_spec.type == "text":
            run = paragraph.add_run(run_spec.text)
            # F356: an unmarked run inherits its style (a heading's bold), never forced off.
            run.bold = True if "bold" in run_spec.marks else None
            run.italic = True if "italic" in run_spec.marks else None
            run.underline = True if "underline" in run_spec.marks else None
            if font:
                run.font.name = font
        elif run_spec.type == "variable":
            text = _resolve_var(run_spec.path, run_spec.fallback, values, unresolved)
            run = paragraph.add_run(text)
            if font:
                run.font.name = font


def _add_data_table(doc, block, data: Optional[Dict[str, Any]], unresolved: List[str], font: Optional[str], kit=None):
    """Rows from the per-generation ``data.*`` list (PRD-243); mirrors the HTML renderer's
    empty policy (unresolved unless ``empty_text`` is set; F356: an ``empty_text`` of ""
    prints nothing). F356: styled like the PDF's tables; a report's KPIs as tiles."""
    rows = walk_dynamic(data or {}, block.path)
    if not isinstance(rows, list) or not rows:
        if block.empty_text is not None:
            if block.empty_text:
                doc.add_paragraph(block.empty_text)
            return
        unresolved.append(block.path)
        doc.add_paragraph(f"[[{block.path}]]")
        return
    unresolved.extend(unfilled_cells(block, rows))  # F345: every row fills every required column
    currency = currency_of(kit)
    if block.id == KPIS_ID:
        tiles = [[cell_text(row, col.key, i, currency) for i, col in enumerate(block.columns)] for row in rows]
        kpi_tiles(doc, tiles, kit or {}, font)
        return
    table = doc.add_table(rows=len(rows) + 1, cols=len(block.columns))
    _fill_data_table(table, block, rows, font, currency)
    style_table(table, kit or {}, header=True, aligns=[col.align for col in block.columns])


def _fill_data_table(table, block, rows: List[Any], font: Optional[str], currency: str = "") -> None:
    for c_idx, col in enumerate(block.columns):
        para = table.cell(0, c_idx).paragraphs[0]
        run = para.add_run(col.label or col.key)
        run.bold = True
        if font:
            run.font.name = font
    for r_idx, row in enumerate(rows, start=1):
        for c_idx, col in enumerate(block.columns):
            run = table.cell(r_idx, c_idx).paragraphs[0].add_run(cell_text(row, col.key, c_idx, currency))
            if font:
                run.font.name = font


def _styled_run(paragraph, text: str, marks, font: Optional[str]):
    run = paragraph.add_run(text)
    run.bold = "bold" in marks
    run.italic = "italic" in marks
    run.underline = "underline" in marks
    if font:
        run.font.name = font
    return run


def _add_segs(paragraph, line, font: Optional[str]) -> None:
    for seg in line:
        text = f"[[{seg.text}]]" if seg.kind == MISSING else seg.text
        _styled_run(paragraph, text, seg.marks, font)


def _add_list(doc, group: Group, font: Optional[str]) -> None:
    """One paragraph per item: Word's bullet style, or the item's number written before it."""
    from docx.shared import Mm

    for index, line in enumerate(group.lines):
        if group.kind == BULLETED:
            paragraph = doc.add_paragraph(style=DOCX_BULLET_STYLE)
        else:
            paragraph = doc.add_paragraph()
            paragraph.paragraph_format.left_indent = Mm(DOCX_LIST_INDENT_MM)
            _styled_run(paragraph, f"{group.start + index}. ", (), font)
        _add_segs(paragraph, line, font)


def _add_text_body(doc, groups: List[Group], font: Optional[str]) -> None:
    """F347: a text block's paragraphs (their line breaks kept), lists and bold/italic."""
    for group in groups or [Group(kind=PARAGRAPH)]:
        if group.kind != PARAGRAPH:
            _add_list(doc, group, font)
            continue
        paragraph = doc.add_paragraph()
        for index, line in enumerate(group.lines):
            if index:
                paragraph.add_run().add_break()
            _add_segs(paragraph, line, font)


def _add_heading(doc, block, values, unresolved) -> None:
    """A heading in its level's style (PRD-255: the kit's heading colour and type scale; the title over an accent rule).
    F360: its runs name no font, so they take the style's: the kit's heading font when it sets one."""
    p = doc.add_heading(level=min(block.level, 9))
    p.clear()
    _add_inline(p, block.content, values, unresolved, None)


def _add_table(doc, block, values, unresolved, font, kit=None) -> None:
    """A table; F356: an optional row whose chips are all empty is left out, and the
    table is styled like the PDF's (totals and signatures by their block ids)."""
    kept = [(index, row) for index, row in enumerate(block.rows) if not row_is_blank(row, values)]
    n_cols = max((len(r) for r in block.rows), default=0)
    if not (kept and n_cols):
        return
    table = doc.add_table(rows=len(kept), cols=n_cols)
    for r_idx, (_, row) in enumerate(kept):
        for c_idx, cell in enumerate(row):
            _add_inline(table.cell(r_idx, c_idx).paragraphs[0], cell, values, unresolved, font)
    header = block.header and kept[0][0] == 0
    if block.id in TOTALS_IDS:
        style_totals(table, kit or {})
    elif block.id == SIGNATURES_ID:
        style_signatures(table, kit or {})
    else:
        style_table(table, kit or {}, header=header)


def _add_image(doc, block, brand_kit, unresolved) -> None:
    from docx.shared import Mm

    src = (brand_kit or {}).get("logo_url", "") if block.source == "brand_logo" else (block.src or "")
    if block.source == "brand_logo" and not src:
        unresolved.append("brand.logo_url")
        return
    stream = _safe_image_bytes(src)
    if stream is None:
        doc.add_paragraph(block.alt or "")
        return
    try:
        width = Mm(block.width_mm) if block.width_mm else Mm(DEFAULT_IMAGE_MM)
        doc.add_picture(stream, width=width)
    except Exception:  # noqa: BLE001 — unreadable image format: the alt text instead
        logger.warning("[DocxRender] image block %s could not be embedded; its alt text is printed", block.id, exc_info=True)
        doc.add_paragraph(block.alt or "")


def _add_section(doc, block, values, brand_kit, unresolved, *, font, data=None) -> None:
    """A titled group; F356: left out when it has nothing to print, kept on one page
    when it is the Agreement's sign-off."""
    if block_is_blank(block, values, data):
        return
    start = len(doc.element.body) - 1  # the body's last element is its section properties
    if block.title:
        doc.add_heading(block.title, level=2)
    for child in block.children:
        _add_block(doc, child, values, brand_kit, unresolved, font=font, data=data)
    if block.id in KEEP_TOGETHER_IDS:
        keep_together(doc, start)


def _add_text_block(doc, block, values, unresolved, font) -> None:
    """A text block; F356: an empty optional line adds no empty paragraph."""
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt

    groups = block_groups(block, values, unresolved)
    if groups or not block_is_blank(block, values):
        _add_text_body(doc, groups, font)
        if block.id in TIGHT_IDS:
            doc.paragraphs[-1].paragraph_format.space_after = Pt(0)
        if block.id in RIGHT_ALIGNED_IDS:
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.RIGHT


def _add_block(doc, block, values, brand_kit, unresolved, *, font, data=None):
    kind = block.type
    if kind == "heading":
        return _add_heading(doc, block, values, unresolved)
    if kind in ("text", "variable"):
        return _add_text_block(doc, block, values, unresolved, font)
    if kind == "table":
        return _add_table(doc, block, values, unresolved, font, brand_kit)
    if kind == "image":
        return _add_image(doc, block, brand_kit, unresolved)
    if kind == "data_table":
        return _add_data_table(doc, block, data, unresolved, font, brand_kit)
    if kind == "page_break":
        return doc.add_page_break()
    if kind == "section":
        return _add_section(doc, block, values, brand_kit, unresolved, font=font, data=data)
    if kind == "brand":  # PRD-255 US-009: a part of the brand board, drawn from the kit
        return add_brand_part(doc, block, brand_kit, font, _safe_image_bytes)
    return None


def _plain_text(content: list, values: Dict[str, str]) -> str:
    """Inline content as plain text (a chip with no value prints nothing)."""
    return "".join(run.text if run.type == "text" else values.get(run.path, run.fallback or "") for run in content)


def _document_title(blocks, values: Dict[str, str]) -> str:
    """The first level-1 heading's text: what the footer names the document by, as the PDF's does."""
    for block in blocks:
        if block.type == "heading" and block.level == 1:
            return _plain_text(block.content, values)
        if block.type == "section":
            found = _document_title(block.children, values)
            if found:
                return found
    return ""


def _letterhead_logo(logo, kit: Dict, unresolved: List[str]):
    """Writes the letterhead's logo into a header cell, ``logo_rules.letterhead_mm`` high (PRD-255)."""
    def write(cell) -> None:
        from docx.shared import Mm

        src = (kit or {}).get("logo_url", "") if logo is not None else ""
        stream = _safe_image_bytes(src) if src else None
        if logo is not None and not src:
            unresolved.append("brand.logo_url")
        if stream is None:
            return
        try:
            cell.paragraphs[0].add_run().add_picture(stream, height=Mm(tokens.design(kit or {}).logo_mm))
        except Exception:  # noqa: BLE001 — unreadable image format: the letterhead goes without it
            logger.warning("[DocxRender] the letterhead logo could not be embedded; left out", exc_info=True)
    return write


def _letterhead_company(blocks, values: Dict[str, str], kit: Dict, unresolved: List[str], font: Optional[str]):
    """Writes the letterhead's company block into a header cell: the name, then the muted lines."""
    def write(cell) -> None:
        from docx.shared import Pt

        design = tokens.design(kit or {})
        roles = design.palette
        for index, block in enumerate(blocks):
            paragraph = cell.paragraphs[0] if index == 0 else cell.add_paragraph()
            paragraph.paragraph_format.space_after = Pt(0)
            _add_inline(paragraph, block.content, values, unresolved, font)
            step = design.type["h3" if block.type == "heading" else "small"]
            for run in paragraph.runs:
                run.bold = block.type == "heading" and step.bold
                run.font.size = Pt(step.size_pt)
                run.font.color.rgb = rgb(roles.heading if block.type == "heading" else roles.muted)
    return write


def render_document_docx(
    doc_model: BlockDocument, values: Dict[str, str], brand_kit: Dict, data: Optional[Dict[str, Any]] = None
) -> RenderedDocx:
    """Render a block document to a python-docx Document + unresolved-path list.
    ``data`` is the raw per-generation object read by ``data_table`` blocks."""
    from docx import Document

    document = Document()
    bk = brand_kit or {}
    # font_family may be a CSS stack ("Inter, 'Segoe UI', ..."); take the first family.
    font_stack = next(iter(families(bk.get("font_family"))), None)
    # F360: the headings in the kit's heading font, as in the PDF (Word used the body font for them).
    heading_font = next(iter(families(bk.get("heading_font"))), None)
    apply_styles(document, bk, font_stack, heading_font)  # F356: the PDF's type, spacing and colours

    unresolved: List[str] = []
    head, rest = split_letterhead(doc_model.blocks)
    if head:  # F356: the letterhead is the first page's header
        logo = _letterhead_logo(logo_of(head), bk, unresolved)
        add_letterhead(document, bk, logo, _letterhead_company(company_of(head), values, bk, unresolved, font_stack))
    for block in rest:
        _add_block(document, block, values, bk, unresolved, font=font_stack, data=data)
    add_footer(document, bk, footer_name(bk), _document_title(doc_model.blocks, values))

    seen: Dict[str, None] = {}
    for path in unresolved:
        seen.setdefault(path, None)
    return RenderedDocx(document=document, unresolved=list(seen))


__all__ = ["RenderedDocx", "render_document_docx"]
