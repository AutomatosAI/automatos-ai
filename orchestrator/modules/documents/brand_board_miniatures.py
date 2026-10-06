"""The brand board's miniature applications: the Branded Invoice and Letter, printed with the kit (PRD-255 US-009).

The board shows three applications of the kit. Two are the starters themselves:
each is rendered with the kit and its own sample data, printed by WeasyPrint and
its first page drawn by the F353 renderer (``thumbnails.render.pdf_first_page_png``)
as a PNG the board embeds as a ``data:`` URI. The third, the social card, is drawn
in the board from the kit's own colours (``blocks/brand_board.social_colours``):
a document render never calls media-render.

A miniature reads the kit only (FR-10). The starters' chips resolve from the kit
(``variables.resolver.build_context`` with no person and no business profile); a
chip the kit has nothing for prints nothing on a miniature, never a red marker.
The print fetches only ``data:`` URIs, which is how the render-ready kit carries
its uploaded logo and fonts: nothing reaches the network.

A miniature that cannot be drawn is logged and left out; the board shows its frame
and its name instead.
"""
from __future__ import annotations

import base64
import logging
from datetime import datetime
from typing import Any, Dict, Mapping
from urllib.parse import urlparse

from modules.documents.blocks import collect_variable_paths, render_document_html, validate_blocks
from modules.documents.blocks.brand_board import APPLICATION_INVOICE, APPLICATION_LETTER
from modules.documents.presets import INVOICE, LETTER
from modules.documents.template_preview import preview_data
from modules.documents.thumbnails.render import pdf_first_page_png
from modules.documents.variables.resolver import build_context, resolve_paths

logger = logging.getLogger(__name__)

PNG_DATA_URI = "data:image/png;base64,"
# The starters each miniature is printed from.
MINIATURE_STARTERS = ((APPLICATION_INVOICE, INVOICE), (APPLICATION_LETTER, LETTER))


def only_data_uris(url: str, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """WeasyPrint's fetcher for the board and its miniatures: a ``data:`` URI (the inlined logo and fonts), nothing else."""
    from weasyprint import default_url_fetcher

    if (urlparse(url).scheme or "").lower() != "data":
        raise ValueError(f"a brand board fetches nothing ({url[:80]})")
    return default_url_fetcher(url, *args, **kwargs)


def starter_page(preset: Mapping[str, Any], kit: Mapping[str, Any], now: datetime) -> str:
    """The starter ``preset`` rendered with ``kit`` and its own sample, as a full HTML page."""
    doc = validate_blocks(preset["blocks"])
    data = preview_data(None, preset.get("sample_data"))
    resolved = resolve_paths(build_context(None, None, dict(kit), now, data), collect_variable_paths(doc))
    values = {**{path: "" for path in resolved.unresolved}, **resolved.values}
    return render_document_html(doc, values, dict(kit), title=preset["name"], data=data).html


def print_page(page: str) -> bytes:
    """``page`` printed to PDF by WeasyPrint, fetching only ``data:`` URIs."""
    from weasyprint import HTML

    return HTML(string=page, url_fetcher=only_data_uris).write_pdf()


def page_png_uri(page: str) -> str:
    """``page`` printed to PDF and its first page drawn, as a PNG ``data:`` URI."""
    pdf = print_page(page)
    return PNG_DATA_URI + base64.b64encode(pdf_first_page_png(pdf)).decode("ascii")


def starter_miniatures(kit: Mapping[str, Any]) -> Dict[str, str]:
    """``{application: PNG data URI}`` for the invoice and the letter; one that cannot be drawn is left out."""
    now = datetime.utcnow()
    drawn: Dict[str, str] = {}
    for key, preset in MINIATURE_STARTERS:
        try:
            drawn[key] = page_png_uri(starter_page(preset, kit, now))
        except Exception:
            logger.exception("[BrandBoard] the %s miniature could not be drawn; the board shows its frame", key)
    return drawn


__all__ = ["MINIATURE_STARTERS", "only_data_uris", "page_png_uri", "print_page", "starter_miniatures", "starter_page"]
