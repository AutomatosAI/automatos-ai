"""F311 (night 9): Markdown and text documents with headings are chunked by section.

``DocumentProcessor.chunk_document`` ran every document through the topic-coherence
chunker, which cut wholesale-terms-2026.md between "## Delivery" and "Carriage is
charged per drop …" and glued the carriage lines to "## Payment" (document 1526,
chunks 4 and 5). Auto, asked for the delivery charge, was handed neither and said the
terms have none (ledger L1, L93). ``sections_first`` puts the section chunker of
:mod:`modules.rag.chunking.markdown_sections` in front of it for a Markdown or text
document that has headings; every other document is chunked as before.

Documents already stored keep their old chunks until they are processed again.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Dict, List, Optional

from config import config
from modules.rag.chunking.markdown_sections import Piece, section_chunks

logger = logging.getLogger(__name__)

# DocumentType values (modules.rag.ingestion.manager) whose text may carry Markdown headings.
SECTIONED_TYPES = ("md", "txt")


def _pieces_for(text: str, file_type: Any) -> List[Piece]:
    """The document's section pieces, or [] when it is not chunked by section."""
    if not config.RAG_SECTION_CHUNKING_ENABLED or getattr(file_type, "value", None) not in SECTIONED_TYPES:
        return []
    return section_chunks(text, config.RAG_SECTION_MIN_CHARS, config.RAG_SECTION_MAX_CHARS)


def sections_first(chunk_document: Callable[..., List[Any]]) -> Callable[..., List[Any]]:
    """Chunk a Markdown or text document with headings by its sections (F311);
    any other document goes to ``chunk_document`` unchanged."""

    @functools.wraps(chunk_document)
    def chunk(self: Any, text: str, file_type: Any, metadata: Optional[Dict] = None) -> List[Any]:
        pieces = _pieces_for(text, file_type)
        if not pieces:
            return chunk_document(self, text, file_type, metadata)
        # Imported here: the manager imports this module to decorate its processor.
        from modules.rag.ingestion.manager import DocumentChunk

        logger.info("[F311] chunked by section: %d chunks", len(pieces))
        return [
            DocumentChunk(
                document_id=0, chunk_index=index, content=piece.content,
                metadata={"file_type": file_type.value, "chunk_size": len(piece.content), "sections": True,
                          **(metadata or {})},
                parent_content=None, headers={"section": piece.heading} if piece.heading else {},
            )
            for index, piece in enumerate(pieces)
        ]

    return chunk
