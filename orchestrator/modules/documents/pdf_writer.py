"""A rendered page written as a PDF by the brand kit's name (F331, night 10).

F331 (5 Oct): the invoices' PDF metadata named the wrong author ("Automatos", or
another brand's name). The file's author is now the workspace's brand kit:
its company name, else the brand's name (``services.brand_rules.document_author``).
A workspace that never set a kit keeps WeasyPrint's default.

``services`` is imported when a PDF is written, never when this module loads:
the document modules load before the services.
"""
from __future__ import annotations

from typing import Any


async def write_pdf(page: Any, output_path: str, db: Any, workspace_id: Any) -> None:
    """Render ``page`` (a WeasyPrint ``HTML``) to ``output_path``; RuntimeError when it cannot."""
    from services.brand_rules import document_author, kit_off_loop

    author = document_author(await kit_off_loop(db, workspace_id))
    try:
        document = page.render()
        if author:
            document.metadata.authors = [author]
        document.write_pdf(output_path)
    except Exception as e:
        raise RuntimeError(f"PDF generation failed: {e}") from e


__all__ = ["write_pdf"]
