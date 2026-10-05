"""F353 (issue #947): a document's first page, drawn as a small PNG.

* A PDF's page 1 is drawn by pypdfium2.
* A Word document, a sheet, a CSV or markdown is first laid out as a page
  (``html_sources``) and printed to PDF by WeasyPrint, which fetches nothing:
  every URL in the file (an image, a stylesheet) is refused, so a file can never
  make the server call out. Then page 1 is drawn the same way.

``render_png_isolated`` (what the job uses) runs the render in a child Python
process (``python -m modules.documents.thumbnails.render <ext>``, the file on
stdin, the PNG on stdout), under a time limit. A file that crashes PDFium or
stalls the layout costs that child, never the API process, and the CPU-bound
layout never holds the server's GIL.
"""
from __future__ import annotations

import io
import subprocess
import sys
import threading
from pathlib import Path

from modules.documents.thumbnails.html_sources import BODY_BUILDERS, document_html

THUMBNAIL_WIDTH_PX = 480
# A very tall page is cut at this height: the card shows the top of the page.
MAX_THUMBNAIL_HEIGHT_PX = 960
RENDER_TIMEOUT_S = 60
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


def _crop_bottom(height_pt: float, scale: float) -> float:
    """How much of the page's bottom to leave out so the picture stays short enough."""
    return max(0.0, height_pt - MAX_THUMBNAIL_HEIGHT_PX / scale)


def pdf_first_page_png(pdf_bytes: bytes) -> bytes:
    """Page 1 of a PDF as a PNG ``THUMBNAIL_WIDTH_PX`` wide."""
    import pypdfium2 as pdfium

    with _pdfium_lock:
        try:
            pdf = pdfium.PdfDocument(pdf_bytes)
        except pdfium.PdfiumError as e:
            raise ThumbnailError(f"not a readable PDF: {e}") from e
        try:
            if len(pdf) == 0:
                raise ThumbnailError("the PDF has no pages")
            page = pdf[0]
            width, height = page.get_size()
            scale = THUMBNAIL_WIDTH_PX / max(width, 1.0)
            crop = (0, _crop_bottom(height, scale), 0, 0)
            image = page.render(scale=scale, crop=crop).to_pil()
        finally:
            pdf.close()
    out = io.BytesIO()
    image.save(out, format="PNG", optimize=True)
    return out.getvalue()


def render_png(data: bytes, ext: str) -> bytes:
    """The first page of a file of type ``ext`` (".pdf", ".docx", …) as a PNG, in this process."""
    ext = ext.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ThumbnailError(f"no preview is drawn for {ext or 'extensionless'} files")
    if not data:
        raise ThumbnailError("the file is empty")
    pdf_bytes = data if ext in PDF_EXTENSIONS else html_to_pdf(document_html(data, ext))
    return pdf_first_page_png(pdf_bytes)


def _reason(stderr: bytes) -> str:
    lines = [line for line in stderr.decode("utf-8", errors="replace").splitlines() if line.strip()]
    return (lines[-1] if lines else "no error output")[:MAX_REASON_CHARS]


def render_png_isolated(data: bytes, ext: str, timeout_s: float = RENDER_TIMEOUT_S) -> bytes:
    """``render_png`` in a child process; ThumbnailError (with the child's reason) when it fails."""
    command = [sys.executable, "-m", "modules.documents.thumbnails.render", ext]
    try:
        done = subprocess.run(
            command, input=data, capture_output=True, timeout=timeout_s, cwd=ORCHESTRATOR_ROOT, check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise ThumbnailError(f"the render took longer than {timeout_s:g}s") from e
    if done.returncode != 0 or not done.stdout:
        raise ThumbnailError(f"the render failed (exit {done.returncode}): {_reason(done.stderr)}")
    return done.stdout


def main(argv: list[str]) -> int:
    """Child-process entry: the file on stdin, its first page's PNG on stdout."""
    if len(argv) != 1:
        sys.stderr.write("usage: python -m modules.documents.thumbnails.render <ext>\n")
        return 2
    try:
        png = render_png(sys.stdin.buffer.read(), argv[0])
    except ThumbnailError as e:
        sys.stderr.write(f"{e}\n")
        return 1
    sys.stdout.buffer.write(png)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
