"""Document template block system (PRD-167 S2).

Canonical, editor-independent block schema + renderers (HTML→PDF and DOCX) and the
legacy-data → blocks mapper.
"""

from .docx_renderer import RenderedDocx, render_document_docx
from .html_renderer import RenderedHtml, render_document_html
from .legacy_mapper import blocks_from_legacy
from .schema import SCHEMA_VERSION, BlockDocument
from .validation import BlockValidationError, collect_variable_paths, validate_blocks

__all__ = [
    "SCHEMA_VERSION",
    "BlockDocument",
    "BlockValidationError",
    "validate_blocks",
    "collect_variable_paths",
    "render_document_html",
    "RenderedHtml",
    "render_document_docx",
    "RenderedDocx",
    "blocks_from_legacy",
]
