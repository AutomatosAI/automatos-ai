"""A block template's page drawn as a PNG from a kit, so an agent can look at it (PRD-255 US-012).

The Brand designer renders what it made and looks at the result (``render_preview``).
A template is drawn the way the brand board draws its miniature starters
(``brand_board_miniatures.starter_page``): its blocks rendered with the kit and its
own sample data, chips resolved from the kit alone, printed by WeasyPrint fetching
only ``data:`` URIs (the render-ready kit carries its logo and fonts inlined), and
the page drawn by the F353 renderer, whole and :data:`PREVIEW_WIDTH_PX` wide.

The kit is whatever the caller passes: the stored one, or a proposal laid over it
for this render only. Nothing here reads or writes the database.

The print is CPU-bound, so the caller runs it in a child process
(:func:`render_template_page_isolated`: ``python -m
modules.documents.template_page_render <page>``, the template and the kit as JSON
on stdin, the PNG on stdout) under a time limit, the F353 way.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from typing import Any, Mapping

from modules.documents.brand_board_miniatures import print_page, starter_page
from modules.documents.thumbnails.render import RENDER_TIMEOUT_S, ThumbnailError, pdf_first_page_png, run_isolated

# Wide enough to read a page's small print; the brand board's own PNG width.
PREVIEW_WIDTH_PX = 1240
PREVIEW_RENDER_TIMEOUT_S = RENDER_TIMEOUT_S
TEMPLATE_PAGE_MODULE = "modules.documents.template_page_render"
# What of a template the child needs: never its row, its id or its workspace.
TEMPLATE_FIELDS = ("name", "blocks", "sample_data")


def render_template_page(template: Mapping[str, Any], kit: Mapping[str, Any], page_number: int) -> bytes:
    """Page ``page_number`` of ``template`` (its blocks and sample) rendered with ``kit``, as a PNG, in this process."""
    page = starter_page(template, kit, datetime.utcnow())
    return pdf_first_page_png(print_page(page), width_px=PREVIEW_WIDTH_PX, max_height_px=None, page_number=page_number)


def render_template_page_isolated(
    template: Mapping[str, Any], kit: Mapping[str, Any], page_number: int,
    timeout_s: float = PREVIEW_RENDER_TIMEOUT_S,
) -> bytes:
    """``render_template_page`` in a child process; ThumbnailError (with the child's reason) when it fails."""
    payload = {"template": {k: template.get(k) for k in TEMPLATE_FIELDS}, "kit": dict(kit)}
    data = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    return run_isolated(TEMPLATE_PAGE_MODULE, [str(page_number)], data, timeout_s)


def _payload(raw: bytes) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """The template and the kit the parent sent; ThumbnailError when they are not objects."""
    payload = json.loads(raw.decode("utf-8"))
    template = payload.get("template") if isinstance(payload, dict) else None
    kit = payload.get("kit") if isinstance(payload, dict) else None
    if not isinstance(template, dict) or not isinstance(kit, dict):
        raise ThumbnailError("the render needs a template and a kit, as JSON objects")
    return template, kit


def main(argv: list[str]) -> int:
    """Child-process entry: ``{"template", "kit"}`` as JSON on stdin, page ``argv[0]``'s PNG on stdout."""
    if len(argv) != 1 or not argv[0].isdigit():
        sys.stderr.write(f"usage: python -m {TEMPLATE_PAGE_MODULE} <page>\n")
        return 2
    try:
        template, kit = _payload(sys.stdin.buffer.read())
        png = render_template_page(template, kit, int(argv[0]))
    except (ThumbnailError, ValueError) as e:
        sys.stderr.write(f"{e}\n")
        return 1
    sys.stdout.buffer.write(png)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


__all__ = ["PREVIEW_WIDTH_PX", "render_template_page", "render_template_page_isolated"]
