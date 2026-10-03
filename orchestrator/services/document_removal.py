"""Removing one Document — the Documents page's delete, shared.

F235 removes an earlier filed round of an approved ticket the same way the page
does, so a removed report leaves no file, chunk or vector behind either.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def remove_document(document: Any, workspace_id: str) -> None:
    """Remove ``document``: its stored file, then its chunks, row and vectors through
    the ingestion manager, which owns the §H contract "delete removes the vector" (a
    bare row delete left the vectors in the index, and the document kept surfacing
    in search). The caller has checked that the document is in ``workspace_id``:
    the manager deletes by bare id and must never be reachable without that check."""
    from api.documents import get_document_manager

    if document.file_path and os.path.exists(document.file_path):
        try:
            os.remove(document.file_path)
        except OSError:
            logger.warning("Could not delete file %s", document.file_path, exc_info=True)
    get_document_manager(str(workspace_id)).delete_document(document.id)


__all__ = ["remove_document"]
