"""Block document → HTML renderer (PRD-167 S2).

Produces a complete, brand-styled HTML document from a :class:`BlockDocument` plus a
resolved-variable map and the workspace brand kit. The output is handed to the existing
WeasyPrint path in ``generation_service.generate_pdf`` (which keeps its PRD-156 S4
SSRF-safe URL fetcher).

Security: unlike the legacy Jinja templates, blocks are *not* a template language — we
build HTML directly here and **HTML-escape every text run, resolved value, attribute
and brand string**. There is no user-controlled markup surface, so the SSTI class from
PRD-156 does not apply to block templates.

Variable policy (PRD-167 S3): a variable with no resolved value and no explicit
``fallback`` is recorded in ``unresolved`` and emitted as a *visible* marker
(``[[path]]``) rather than a silent blank, so the caller can refuse to finalise.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from typing import Dict, List

from .schema import BlockDocument

_MARK_TAGS = {
    "bold": ("<strong>", "</strong>"),
    "italic": ("<em>", "</em>"),
    "underline": ("<u>", "</u>"),
    "strike": ("<s>", "</s>"),
    "code": ("<code>", "</code>"),
}


@dataclass
class RenderedHtml:
    html: str
    unresolved: List[str] = field(default_factory=list)


def _esc(value: str) -> str:
    return html.escape(value, quote=True)


def _resolve_var(path: str, fallback, values: Dict[str, str], unresolved: List[str]) -> str:
    if path in values:
        return _esc(values[path])
    if fallback is not None:
        return _esc(fallback)
    unresolved.append(path)
    return f'<span class="unresolved-var" data-path="{_esc(path)}">[[{_esc(path)}]]</span>'


def _render_inline(content: list, values: Dict[str, str], unresolved: List[str]) -> str:
    parts: List[str] = []
    for run in content:
        if run.type == "text":
            text = _esc(run.text)
            for mark in run.marks:
                open_tag, close_tag = _MARK_TAGS.get(mark, ("", ""))
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
    return f'<img class="doc-image" src="{_esc(src)}" alt="{alt}" style="{style}" />'


def _render_table(block, values: Dict[str, str], unresolved: List[str]) -> str:
    rows_html: List[str] = []
    for r_idx, row in enumerate(block.rows):
        cell_tag = "th" if (block.header and r_idx == 0) else "td"
        cells = "".join(
            f"<{cell_tag}>{_render_inline(cell, values, unresolved)}</{cell_tag}>" for cell in row
        )
        rows_html.append(f"<tr>{cells}</tr>")
    return f'<table class="doc-table">{"".join(rows_html)}</table>'


def _render_block(block, values: Dict[str, str], brand_kit: Dict, unresolved: List[str]) -> str:
    kind = block.type
    if kind == "heading":
        inner = _render_inline(block.content, values, unresolved)
        return f"<h{block.level}>{inner}</h{block.level}>"
    if kind == "text":
        return f"<p>{_render_inline(block.content, values, unresolved)}</p>"
    if kind == "table":
        return _render_table(block, values, unresolved)
    if kind == "image":
        return _render_image(block, brand_kit, unresolved)
    if kind == "variable":
        return f"<p>{_resolve_var(block.path, block.fallback, values, unresolved)}</p>"
    if kind == "page_break":
        return '<div class="page-break"></div>'
    if kind == "section":
        parts: List[str] = ['<section class="doc-section">']
        if block.title:
            parts.append(f"<h2>{_esc(block.title)}</h2>")
        for child in block.children:
            parts.append(_render_block(child, values, brand_kit, unresolved))
        parts.append("</section>")
        return "".join(parts)
    return ""


def _build_styles(brand_kit: Dict) -> str:
    bk = brand_kit or {}
    primary = bk.get("primary_color") or "#1a1a2e"
    secondary = bk.get("secondary_color") or "#16213e"
    accent = bk.get("accent_color") or "#0f3460"
    text = bk.get("text_color") or "#1a1a2e"
    font = bk.get("font_family") or "Inter, 'Segoe UI', system-ui, sans-serif"
    # Brand strings are validated hex / font names; still escape defensively since they
    # land inside a <style> block.
    return f"""
  @page {{ size: A4; margin: 2cm; }}
  body {{ font-family: {_esc(font)}; color: {_esc(text)}; line-height: 1.6; font-size: 11pt; }}
  h1, h2, h3, h4, h5, h6 {{ color: {_esc(primary)}; }}
  h1 {{ font-size: 26pt; margin: 0 0 0.5rem 0; border-bottom: 3px solid {_esc(accent)}; padding-bottom: 0.5rem; }}
  h2 {{ font-size: 18pt; border-bottom: 1px solid {_esc(secondary)}33; padding-bottom: 0.3rem; margin-top: 1.5rem; }}
  p {{ margin: 0.5rem 0; }}
  .doc-section {{ margin-bottom: 1.5rem; }}
  .doc-image {{ display: block; margin: 1rem 0; }}
  .doc-table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  .doc-table th {{ background: {_esc(primary)}; color: #fff; text-align: left; padding: 0.5rem 0.75rem; }}
  .doc-table td {{ border: 1px solid {_esc(secondary)}33; padding: 0.5rem 0.75rem; }}
  .page-break {{ page-break-after: always; }}
  .unresolved-var {{ color: #b00020; background: #fde7ea; padding: 0 2px; border-radius: 2px; }}
"""


def render_document_html(
    doc: BlockDocument,
    values: Dict[str, str],
    brand_kit: Dict,
    *,
    title: str = "",
) -> RenderedHtml:
    """Render a block document to a full HTML page. Returns the HTML and the list of
    unresolved variable paths encountered during rendering."""
    unresolved: List[str] = []
    body = "".join(_render_block(b, values, brand_kit, unresolved) for b in doc.blocks)
    page = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>{_esc(title)}</title>
<style>{_build_styles(brand_kit)}</style>
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
