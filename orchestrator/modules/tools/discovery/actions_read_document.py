"""The tool that reads a document in full, page by page (PRD-157 S2).

F327 (night 9b, chat ca9d92d2): Auto meant club-box-october-2026.md (document
1518), sent its filename as ``document_id`` (refused: not an integer), then
``document_name`` (refused), then 1523 and 1522, and was handed roast-rules.md
and margin-sheet-sep-2026.csv. The tool now takes the filename too, and an id
and a filename that name two different documents read nothing
(read_document_lookup.py). It left actions_documents.py, whose one register
function is past the length rule.
"""

from __future__ import annotations

import functools
from typing import Callable

from .action_registry import ActionDefinition, ActionRegistry

Register = Callable[[ActionRegistry], None]


def register_read_document_action(registry: ActionRegistry) -> None:
    """Register platform_read_document: read-only, workspace and team scoped by its handler."""
    registry.register(ActionDefinition(
        name="platform_read_document",
        description=(
            "Read the full text of a knowledge-base document, one page at a time. "
            "Use this to read PAST the short snippet returned by search — pass the "
            "document_id from a search result, then request successive pages to read "
            "the whole document. Each page is a token-budgeted slice; the response "
            "reports total_pages, has_more and next_page so you can keep reading. "
            "Name the document by its document_id, its filename, or both: with both, "
            "they must be the same document or nothing is read."
        ),
        category="documents",
        parameters={
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "integer",
                    "description": "ID of the document to read (from a search result or list_documents).",
                },
                "filename": {
                    "type": "string",
                    "description": "The document's filename, exactly as list_documents shows it.",
                },
                "document_name": {
                    "type": "string",
                    "description": "The same as filename (night 9b: Auto sent the filename under this name).",
                },
                "page": {
                    "type": "integer",
                    "description": "Zero-based page number to read. Defaults to 0 (the first page).",
                },
                "offset": {
                    "type": "integer",
                    "description": "Optional chunk-index to start from; the page containing it is returned.",
                },
            },
            "required": [],  # its document_id or its filename; the handler refuses a call naming neither
        },
        permission_level="read",
        tags=["documents", "read", "knowledge", "rag"],
        examples=[
            "read the rest of that document",
            "show me page 2 of document 12",
            "read document 7 in full",
        ],
    ))


def with_read_document(register: Register) -> Register:
    """``register``, then platform_read_document, so the documents actions stay one call."""

    @functools.wraps(register)
    def registers_both(registry: ActionRegistry) -> None:
        register(registry)
        register_read_document_action(registry)

    return registers_both
