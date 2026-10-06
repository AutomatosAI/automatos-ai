"""F353 (issue #947): a document's first page, drawn as a small PNG.

PRD-255 US-012: any page can be drawn (``page_number``, page 1 by default), so an
agent's ``render_preview`` can look at page 2 of what it made.

* A PDF's page 1 is drawn by pypdfium2.
* A Word document, a sheet, a CSV or markdown is first laid out as a page
  (``html_sources``) and printed to PDF by WeasyPrint, which fetches nothing:
  every URL in the file (an image, a stylesheet) is refused, so a file can never
  make the server call out. Then page 1 is drawn the same way.

``render_png_isolated`` (what the job uses) runs the render in a child Python
process (``python -m modules.documents.thumbnails.render <ext>``, the file on
stdin, the PNG on stdout), under a time limit. A file that crashes PDFium or
stalls the layout costs that child, never the API process, and the CPU-bound
layout never holds the server's GIL. ``run_isolated`` is that child-process run,
for any module with the same stdin-to-stdout entry (the brand board's render,
PRD-255 US-010, is one).
"""
from __future__ import annotations

import io
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional, Sequence

from modules.documents.thumbnails.html_sources import BODY_BUILDERS, document_html

THUMBNAIL_WIDTH_PX = 480
# A very tall page is cut at this height: the card shows the top of the page.
MAX_THUMBNAIL_HEIGHT_PX = 960
RENDER_TIMEOUT_S = 60
FIRST_PAGE = 1
MAX_REASON_CHARS = 300

PDF_EXTENSIONS = frozenset({".pdf"})
SUPPORTED_EXTENSIONS = PDF_EXTENSIONS | frozenset(BODY_BUILDERS)

ORCHESTRATOR_ROOT = Path(__file__).resolve().parents[3]

# PDFium is not thread-safe: one render at a time in a process.
_pdfium_lock = threading.Lock()


class ThumbnailError(Exception):
    """The first page could not be drawn; the message says why (it is logged)."""


def _refuse_fetch(url: str, *args: object, **kwargs: object) -> dict:
    raise ValueError(f"a document preview fetches nothing ({url[:80]})")


def html_to_pdf(page: str) -> bytes:
    """Print ``page`` to PDF with WeasyPrint, fetching no URL."""
    from weasyprint import HTML

    return HTML(string=page, url_fetcher=_refuse_fetch).write_pdf()


def _crop_bottom(height_pt: float, scale: float, max_height_px: Optional[int]) -> float:
    """How much of the page's bottom to leave out so the picture stays short enough (none without a cap)."""
    if max_height_px is None:
        return 0.0
    return max(0.0, height_pt - max_height_px / scale)


def _page_index(count: int, page: int) -> int:
    """``page`` (1 = the first) as an index into a PDF of ``count`` pages; ThumbnailError when it has none."""
    if count == 0:
        raise ThumbnailError("the PDF has no pages")
    if page < 1 or page > count:
        raise ThumbnailError(f"there is no page {page}: the document has {count} page{'s' if count != 1 else ''}")
    return page - 1


def pdf_first_page_png(
    pdf_bytes: bytes, width_px: int = THUMBNAIL_WIDTH_PX, max_height_px: Optional[int] = MAX_THUMBNAIL_HEIGHT_PX,
    page_number: int = FIRST_PAGE,
) -> bytes:
    """Page ``page_number`` (page 1 by default) of a PDF as a PNG ``width_px`` wide, cut at
    ``max_height_px`` (``None``: the whole page)."""
    import pypdfium2 as pdfium

    with _pdfium_lock:
        try:
            pdf = pdfium.PdfDocument(pdf_bytes)
        except pdfium.PdfiumError as e:
            raise ThumbnailError(f"not a readable PDF: {e}") from e
        try:
            page = pdf[_page_index(len(pdf), page_number)]
            width, height = page.get_size()
            scale = width_px / max(width, 1.0)
            crop = (0, _crop_bottom(height, scale, max_height_px), 0, 0)
            image = page.render(scale=scale, crop=crop).to_pil()
        finally:
            pdf.close()
    out = io.BytesIO()
    image.save(out, format="PNG", optimize=True)
    return out.getvalue()


def render_png(data: bytes, ext: str, page_number: int = FIRST_PAGE, whole_width_px: Optional[int] = None) -> bytes:
    """Page ``page_number`` (the first by default) of a file of type ``ext`` (".pdf", ".docx", …) as a PNG,
    in this process: a card's thumbnail, or with ``whole_width_px`` the whole page that wide."""
    ext = ext.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ThumbnailError(f"no preview is drawn for {ext or 'extensionless'} files")
    if not data:
        raise ThumbnailError("the file is empty")
    pdf_bytes = data if ext in PDF_EXTENSIONS else html_to_pdf(document_html(data, ext))
    if whole_width_px is None:
        return pdf_first_page_png(pdf_bytes, page_number=page_number)
    return pdf_first_page_png(pdf_bytes, width_px=whole_width_px, max_height_px=None, page_number=page_number)


def _reason(stderr: bytes) -> str:
    lines = [line for line in stderr.decode("utf-8", errors="replace").splitlines() if line.strip()]
    return (lines[-1] if lines else "no error output")[:MAX_REASON_CHARS]


def run_isolated(module: str, args: Sequence[str], data: bytes, timeout_s: float = RENDER_TIMEOUT_S) -> bytes:
    """``python -m <module> <args>`` in a child process, ``data`` on its stdin; its stdout.

    ThumbnailError (with the child's reason) when it fails, exits non-zero, writes
    nothing or runs past ``timeout_s``.
    """
    command = [sys.executable, "-m", module, *args]
    try:
        done = subprocess.run(
            command, input=data, capture_output=True, timeout=timeout_s, cwd=ORCHESTRATOR_ROOT, check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise ThumbnailError(f"the render took longer than {timeout_s:g}s") from e
    if done.returncode != 0 or not done.stdout:
        raise ThumbnailError(f"the render failed (exit {done.returncode}): {_reason(done.stderr)}")
    return done.stdout


def render_png_isolated(
    data: bytes, ext: str, timeout_s: float = RENDER_TIMEOUT_S, page_number: int = FIRST_PAGE,
    whole_width_px: Optional[int] = None,
) -> bytes:
    """``render_png`` in a child process; ThumbnailError (with the child's reason) when it fails."""
    args = [ext]
    if page_number != FIRST_PAGE or whole_width_px is not None:
        args.append(str(page_number))
    if whole_width_px is not None:
        args.append(str(whole_width_px))
    return run_isolated("modules.documents.thumbnails.render", args, data, timeout_s)


def main(argv: list[str]) -> int:
    """Child-process entry: the file on stdin, a page's PNG on stdout (``<ext> [page [whole page width]]``)."""
    if not 1 <= len(argv) <= 3 or not all(arg.isdigit() for arg in argv[1:]):
        sys.stderr.write("usage: python -m modules.documents.thumbnails.render <ext> [page [width]]\n")
        return 2
    numbers = [int(arg) for arg in argv[1:]]
    try:
        png = render_png(sys.stdin.buffer.read(), argv[0], *numbers)
    except ThumbnailError as e:
        sys.stderr.write(f"{e}\n")
        return 1
    sys.stdout.buffer.write(png)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
