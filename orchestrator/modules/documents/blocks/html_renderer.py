"""Block document → HTML renderer (PRD-167 S2).

Produces a complete, brand-styled HTML document from a :class:`BlockDocument` plus a
resolved-variable map and the workspace brand kit. The output is handed to the existing
WeasyPrint path in ``generation_service.generate_pdf`` (which keeps its PRD-156 S4
SSRF-safe URL fetcher).

Security: unlike the legacy Jinja templates, blocks are *not* a template language — we
build HTML directly here and **HTML-escape every text run, resolved value, attribute
and brand string**. There is no user-controlled markup surface, so the SSTI class from
PRD-156 does not apply to block templates. A text block's paragraphs, lists and emphasis
are read by ``text_body`` and escaped as they are written (F347); the kit's uploaded
font files and heading font are ``page_fonts`` (F347), the rest of the sheet ``page_style``.

Variable policy (PRD-167 S3): a variable with no resolved value and no explicit
``fallback`` is recorded in ``unresolved`` and emitted as a *visible* marker
(``[[path]]``) rather than a silent blank, so the caller can refuse to finalise.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..amounts import field_text
from ..variables.catalog import walk_dynamic
from .letterhead_run import company_of, logo_of, split_letterhead
from .optional_parts import block_is_blank, row_is_blank
from .page_fonts import font_css
from .page_style import KEEP_CLASS, KEEP_TOGETHER_MAX_HTML_CHARS, build_styles
from .schema import BlockDocument
from .table_cells import unfilled_cells
from .text_body import MARK_TAGS, block_groups, body_html, unresolved_html


@dataclass
class RenderedHtml:
    html: str
    unresolved: List[str] = field(default_factory=list)


def _esc(value: str) -> str:
    return html.escape(value, quote=True)


def _tag(block) -> str:
    """The block's id as a ``data-block`` attribute: the starters' styles key on it (F350)."""
    return f' data-block="{_esc(block.id)}"'


def _resolve_var(path: str, fallback, values: Dict[str, str], unresolved: List[str]) -> str:
    if path in values:
        return _esc(values[path])
    if fallback is not None:
        return _esc(fallback)
    unresolved.append(path)
    return unresolved_html(path)


def _render_inline(content: list, values: Dict[str, str], unresolved: List[str]) -> str:
    parts: List[str] = []
    for run in content:
        if run.type == "text":
            text = _esc(run.text)
            for mark in run.marks:
                open_tag, close_tag = MARK_TAGS.get(mark, ("", ""))
                text = f"{open_tag}{text}{close_tag}"
            parts.append(text)
        elif run.type == "variable":
            parts.append(_resolve_var(run.path, run.fallback, values, unresolved))
    return "".join(parts)


def _render_image(block, brand_kit: Dict, unresolved: List[str]) -> str:
    if block.source == "brand_logo":
        src = (brand_kit or {}).get("logo_url") or ""
        if not src:
            unresolved.append("brand.logo_url")
            return ""
    else:
        src = block.src or ""
    style = f"width:{block.width_mm}mm;" if block.width_mm else "max-width:100%;"
    alt = _esc(block.alt or "")
    return f'<img class="doc-image"{_tag(block)} src="{_esc(src)}" alt="{alt}" style="{style}" />'


def _render_table(block, values: Dict[str, str], unresolved: List[str]) -> str:
    rows_html: List[str] = []
    for r_idx, row in enumerate(block.rows):
        if row_is_blank(row, values):  # F356: an optional row (a total not sent) is left out
            continue
        cell_tag = "th" if (block.header and r_idx == 0) else "td"
        cells = "".join(
            f"<{cell_tag}>{_render_inline(cell, values, unresolved)}</{cell_tag}>" for cell in row
        )
        rows_html.append(f"<tr>{cells}</tr>")
    if block.rows and not rows_html:
        return ""  # every row was an empty optional one: no table at all
    return f'<table class="doc-table"{_tag(block)}>{"".join(rows_html)}</table>'


def _cell_value(row: Any, key: str, index: int) -> str:
    if isinstance(row, dict):
        value = row.get(key, "")
    elif isinstance(row, (list, tuple)):
        value = row[index] if index < len(row) else ""
    else:
        value = row if index == 0 else ""
    return field_text(key, value)  # F347: a bare amount with two decimals, never a currency added


def _render_data_table(block, data: Optional[Dict[str, Any]], unresolved: List[str]) -> str:
    """Rows from the per-generation ``data.*`` list (PRD-243). Empty/missing is
    unresolved (a blocked document) unless the author allowed ``empty_text``."""
    rows = walk_dynamic(data or {}, block.path)
    if not isinstance(rows, list) or not rows:
        if block.empty_text is not None:
            return f'<p class="doc-empty"{_tag(block)}>{_esc(block.empty_text)}</p>'
        unresolved.append(block.path)
        return (
            f'<p><span class="unresolved-var" data-path="{_esc(block.path)}">'
            f"[[{_esc(block.path)}]]</span></p>"
        )
    unresolved.extend(unfilled_cells(block, rows))  # F345: every row fills every required column
    head = "".join(
        f'<th style="text-align:{c.align}">{_esc(c.label or c.key)}</th>' for c in block.columns
    )
    body: List[str] = []
    for row in rows:
        cells = "".join(
            f'<td style="text-align:{c.align}">{_esc(_cell_value(row, c.key, i))}</td>'
            for i, c in enumerate(block.columns)
        )
        body.append(f"<tr>{cells}</tr>")
    return f'<table class="doc-table"{_tag(block)}><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def _render_block(
    block, values: Dict[str, str], brand_kit: Dict, unresolved: List[str], data: Optional[Dict[str, Any]] = None
) -> str:
    kind = block.type
    if kind == "heading":
        inner = _render_inline(block.content, values, unresolved)
        return f"<h{block.level}{_tag(block)}>{inner}</h{block.level}>"
    if kind in ("text", "variable"):  # F347: paragraphs, line breaks, lists and emphasis kept
        return body_html(block_groups(block, values, unresolved), _tag(block))
    if kind == "table":
        return _render_table(block, values, unresolved)
    if kind == "image":
        return _render_image(block, brand_kit, unresolved)
    if kind == "data_table":
        return _render_data_table(block, data, unresolved)
    if kind == "page_break":
        return '<div class="page-break"></div>'
    if kind == "section":
        return _render_section(block, values, brand_kit, unresolved, data)
    return ""


def _render_section(block, values: Dict[str, str], brand_kit: Dict, unresolved: List[str], data) -> str:
    """A titled group; a short one (F350) is kept on one page rather than split over two.
    F356: one with nothing to print (every child an empty optional part) is left out."""
    if block_is_blank(block, values, data):
        return ""
    title = f"<h2>{_esc(block.title)}</h2>" if block.title else ""
    inner = title + "".join(_render_block(child, values, brand_kit, unresolved, data) for child in block.children)
    classes = "doc-section" if len(inner) > KEEP_TOGETHER_MAX_HTML_CHARS else f"doc-section {KEEP_CLASS}"
    return f'<section class="{classes}"{_tag(block)}>{inner}</section>'


def _render_letterhead(head, values: Dict[str, str], brand_kit: Dict, unresolved: List[str]) -> str:
    """F356: the letterhead as one row: the logo, and the company block beside it."""
    if not head:
        return ""
    logo = logo_of(head)
    mark = _render_image(logo, brand_kit, unresolved) if logo is not None else ""
    company = "".join(_render_block(block, values, brand_kit, unresolved) for block in company_of(head))
    return f'<div class="letterhead"><div class="lh-mark">{mark}</div><div class="lh-company">{company}</div></div>'


def render_document_html(
    doc: BlockDocument,
    values: Dict[str, str],
    brand_kit: Dict,
    *,
    title: str = "",
    data: Optional[Dict[str, Any]] = None,
) -> RenderedHtml:
    """Render a block document to a full HTML page. Returns the HTML and the list of
    unresolved variable paths encountered during rendering.

    ``data`` is the raw per-generation object; ``data_table`` blocks read their rows
    from it (scalar chips still come pre-resolved in ``values``)."""
    unresolved: List[str] = []
    head, rest = split_letterhead(doc.blocks)
    body = _render_letterhead(head, values, brand_kit, unresolved) + "".join(
        _render_block(b, values, brand_kit, unresolved, data) for b in rest
    )
    page = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>{_esc(title)}</title>
<style>{build_styles(brand_kit)}{font_css(brand_kit or {})}</style>
</head>
<body>
{body}
</body>
</html>"""
    # De-duplicate while preserving order.
    seen: Dict[str, None] = {}
    for path in unresolved:
        seen.setdefault(path, None)
    return RenderedHtml(html=page, unresolved=list(seen))


__all__ = ["RenderedHtml", "render_document_html"]
