"""The brand board printed from a workspace's kit, as a PDF or a PNG (PRD-255 US-010).

The Brand kit page shows the board (``presets.BRAND_BOARD``) and offers it as a
PDF and a PNG. Both are printed from the kit alone (FR-10): the board's blocks
are rendered with the render-ready kit (its uploaded logo, variants and fonts
inlined as ``data:`` URIs, ``brand_fonts.brand_kit_for_media_render``) and no
data, printed by WeasyPrint fetching only ``data:`` URIs, and the PNG is page 1
of that PDF drawn by the F353 renderer at :data:`BOARD_PNG_WIDTH_PX`, uncut.

The print is CPU-bound (the board and its two miniature starters), so the API
never runs it itself: :func:`render_board_isolated` runs it in a child process
(``python -m modules.documents.brand_board_render <format>``, the kit as JSON on
stdin, the file on stdout) under a time limit, the F353 way
(``thumbnails.render.run_isolated``).
"""
from __future__ import annotations

import json
import sys
from typing import Any, Mapping

from modules.documents.blocks import render_document_html, validate_blocks
from modules.documents.brand_board_miniatures import print_page
from modules.documents.presets import BRAND_BOARD
from modules.documents.thumbnails.render import RENDER_TIMEOUT_S, ThumbnailError, pdf_first_page_png, run_isolated

BOARD_PDF, BOARD_PNG = "pdf", "png"
BOARD_MEDIA_TYPES = {BOARD_PDF: "application/pdf", BOARD_PNG: "image/png"}
# A4 at 150 dpi: sharp on the page and legible when downloaded.
BOARD_PNG_WIDTH_PX = 1240
BOARD_RENDER_TIMEOUT_S = RENDER_TIMEOUT_S
BOARD_RENDER_MODULE = "modules.documents.brand_board_render"


def _known_format(fmt: str) -> None:
    """ThumbnailError unless ``fmt`` is a format the board is printed as."""
    if fmt not in BOARD_MEDIA_TYPES:
        raise ThumbnailError(f"the brand board is printed as {' or '.join(BOARD_MEDIA_TYPES)}, not {fmt!r}")


def board_page(kit: Mapping[str, Any]) -> str:
    """The Brand Board starter rendered with ``kit`` (render-ready) and no data, as a full HTML page."""
    doc = validate_blocks(BRAND_BOARD["blocks"])
    return render_document_html(doc, {}, dict(kit), title=BRAND_BOARD["name"], data={}).html


def render_board(kit: Mapping[str, Any], fmt: str) -> bytes:
    """The board printed from ``kit`` as ``fmt`` ("pdf" or "png"), in this process."""
    _known_format(fmt)
    pdf = print_page(board_page(kit))
    if fmt == BOARD_PDF:
        return pdf
    return pdf_first_page_png(pdf, width_px=BOARD_PNG_WIDTH_PX, max_height_px=None)


def render_board_isolated(kit: Mapping[str, Any], fmt: str, timeout_s: float = BOARD_RENDER_TIMEOUT_S) -> bytes:
    """``render_board`` in a child process; ThumbnailError (with the child's reason) when it fails."""
    _known_format(fmt)
    data = json.dumps(dict(kit), ensure_ascii=False).encode("utf-8")
    return run_isolated(BOARD_RENDER_MODULE, [fmt], data, timeout_s)


def main(argv: list[str]) -> int:
    """Child-process entry: the kit as JSON on stdin, the board as ``argv[0]`` on stdout."""
    if len(argv) != 1:
        sys.stderr.write(f"usage: python -m {BOARD_RENDER_MODULE} <{'|'.join(BOARD_MEDIA_TYPES)}>\n")
        return 2
    try:
        kit = json.loads(sys.stdin.buffer.read().decode("utf-8"))
        if not isinstance(kit, dict):
            raise ThumbnailError("the kit on stdin is not a JSON object")
        out = render_board(kit, argv[0])
    except (ThumbnailError, ValueError) as e:
        sys.stderr.write(f"{e}\n")
        return 1
    sys.stdout.buffer.write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


__all__ = [
    "BOARD_MEDIA_TYPES",
    "BOARD_PDF",
    "BOARD_PNG",
    "BOARD_PNG_WIDTH_PX",
    "board_page",
    "render_board",
    "render_board_isolated",
]
