"""Every generated document lands in the workspace's ``documents/`` folder too (F370, night 10c).

F370 (6 Oct): a generated PDF, Word file or spreadsheet lived only behind
``/api/documents/generated/<file>`` and object storage; the owner's deliverables
folder (locally ``~/Development/deliverables``, the workspace root that compose
mounts from ``AUTOMATOS_WORKSPACE_DIR``, #722) never held it, while a session's
files and an approved post's files do (``sessions/<ticket>/``, ``socials/``).

When a generated document is registered as a Deliverable
(``DocumentGenerationService.register_as_deliverable``: an agent's
generate_document, the API, a playbook step, a mission), the file the platform just
wrote is copied into ``documents/<file>`` through the workspace worker
(``WorkspaceClient.write_binary``, as the Socials and session copies are). The copy
is not registered again: the Deliverable already is, and a second row would show
every document twice.

* Documents only (PDF, Word, spreadsheet); a social render is copied when its post
  is approved (``modules/socials/workspace_copies.py``).
* Never outside that folder: the file name is a bare name (no ``/``, ``\\``, ``..``)
  and the source must sit in the workspace's own generated folder.
* A copy is a convenience: it runs in the background, and a worker that cannot
  write, or a file that is not there, is logged and skipped. The generation and its
  Deliverable stand.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path, PurePosixPath
from typing import Any, AsyncIterator, Optional

from core.workspace_client import BINARY_PIECE_BYTES, WorkspaceClient

logger = logging.getLogger(__name__)

DOCUMENTS_FOLDER = "documents"
# Where DocumentGenerationService writes a workspace's documents, under GENERATED_DIR/<workspace>.
GENERATED_SUBDIR = "generated"
COPIED_FORMATS = frozenset({"pdf", "docx", "xlsx"})
NOT_A_NAME = frozenset({"", ".", ".."})


def bare_name(filename: Any) -> Optional[str]:
    """``filename`` when it is a bare file name (no folder, no ``..``), else ``None``. Pure."""
    name = filename if isinstance(filename, str) else ""
    if name in NOT_A_NAME or "\\" in name or PurePosixPath(name).name != name:
        return None
    return name


def documents_path(filename: Any) -> Optional[str]:
    """``documents/<filename>``, or ``None`` when ``filename`` is not a bare file name. Pure."""
    name = bare_name(filename)
    return f"{DOCUMENTS_FOLDER}/{name}" if name else None


def generated_file(workspace_id: Any, filename: str) -> Optional[Path]:
    """The document the platform wrote for ``workspace_id``, or ``None`` when it is not there."""
    from modules.documents import generation_service

    base = (Path(generation_service.GENERATED_DIR) / str(workspace_id) / GENERATED_SUBDIR).resolve()
    source = (base / filename).resolve()
    return source if source.is_relative_to(base) and source.is_file() else None


async def file_pieces(source: Path) -> AsyncIterator[bytes]:
    """The file's bytes a piece at a time, each read off the event loop."""
    handle = await asyncio.to_thread(source.open, "rb")
    try:
        while True:
            piece = await asyncio.to_thread(handle.read, BINARY_PIECE_BYTES)
            if not piece:
                return
            yield piece
    finally:
        await asyncio.to_thread(handle.close)


async def copy_document(workspace_id: Any, filename: Any) -> Optional[str]:
    """Write the generated ``filename`` into the workspace's ``documents/`` folder; the path, or ``None``."""
    target = documents_path(filename)
    source = generated_file(workspace_id, filename) if target else None
    if source is None:
        logger.warning("[DocGen] %r is not a document of workspace %s: not copied to documents/", filename, workspace_id)
        return None
    try:
        written = await WorkspaceClient(str(workspace_id)).write_binary(target, file_pieces(source))
    except Exception:
        logger.exception("[DocGen] %s could not be copied into workspace %s", target, workspace_id)
        return None
    if not written.get("success"):
        logger.warning("[DocGen] workspace %s could not take %s: %s", workspace_id, target, written.get("error"))
        return None
    return target


def copy_when_registered(workspace_id: Any, filename: Any, fmt: Any) -> bool:
    """Start copying a registered document into ``documents/``; True when the copy was started.
    Never raises: with no event loop (a sync caller) the copy is skipped and logged."""
    from core.utils.background_tasks import launch_guarded

    if str(fmt or "").lower() not in COPIED_FORMATS or not workspace_id or documents_path(filename) is None:
        return False
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        logger.info("[DocGen] no event loop: %r was not copied to documents/", filename)
        return False
    launch_guarded(copy_document(workspace_id, filename), subsystem="documents", operation="workspace_copy",
                   workspace_id=workspace_id)
    return True


__all__ = [
    "COPIED_FORMATS", "DOCUMENTS_FOLDER", "bare_name", "copy_document", "copy_when_registered", "documents_path",
    "file_pieces", "generated_file",
]
