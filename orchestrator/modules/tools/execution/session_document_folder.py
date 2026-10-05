"""A document a session makes is copied into its ticket's folder (F349, night 10b).

F349 (5 Oct): a Claude Code session working a ticket called ``generate_document``,
and the PDF lived only behind ``/api/documents/generated/<file>`` and an object
storage link (``localhost:9000`` locally). The session's sandbox reaches neither
("the platform's file storage refused the connection"), so the agent never opened
the document it made to check it, and the ticket's own folder never held it.

When the call comes from a session's ticket (``session_task_id`` in the
server-built caller context, F341), the file the platform just wrote is now also
written into that ticket's folder, ``sessions/<ticket>/<file>``, through the
workspace worker (``WorkspaceClient.write_binary``, as the Socials copies are), and
the tool's answer says where: "A copy is in your folder: sessions/<ticket>/<file>".

* Only a session's ticket: a board run's card (``board_task_id``) and a chat call
  are left as they were.
* Never outside that folder: the ticket is a positive integer from the server's
  context, the file name a bare name (no ``/``, ``\\``, ``..``), and the source must
  sit in the workspace's own generated folder.
* A copy is a convenience: a folder the worker cannot write, or a file that is not
  there, is logged and skipped, and the answer stays what it was.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path, PurePosixPath
from typing import Any, AsyncIterator, Dict, Optional

from core.workspace_client import BINARY_PIECE_BYTES, WorkspaceClient
from modules.tools.execution.generate_document_tool import SESSION_CARD_KEY, card_of

logger = logging.getLogger(__name__)

# The ticket's folder, by the host's own rule (cli_host_service: a session with no
# working directory runs in sessions/<ticket>).
SESSIONS_FOLDER = "sessions"
# Where DocumentGenerationService writes a workspace's documents, under GENERATED_DIR/<workspace>.
GENERATED_SUBDIR = "generated"
SESSION_COPY_KEY = "session_copy"
SESSION_COPY_NOTE_KEY = "session_copy_note"
COPY_NOTE = "A copy is in your folder: {path}. Open it there to check the document before you report."
NOT_A_NAME = frozenset({"", ".", ".."})


def session_ticket(caller_context: Any) -> Optional[int]:
    """The ticket a SESSION's call works (``session_task_id``), else ``None``; never a board card."""
    raw = caller_context.get(SESSION_CARD_KEY) if isinstance(caller_context, dict) else None
    ticket = card_of({SESSION_CARD_KEY: raw}) if raw is not None else None
    return ticket if ticket is not None and ticket > 0 else None


def session_copy_path(ticket: int, filename: Any) -> Optional[str]:
    """``sessions/<ticket>/<filename>``, or ``None`` when ``filename`` is not a bare file name."""
    name = filename if isinstance(filename, str) else ""
    if name in NOT_A_NAME or "\\" in name or PurePosixPath(name).name != name:
        return None
    return f"{SESSIONS_FOLDER}/{ticket}/{name}"


def generated_file(workspace_id: Any, filename: str) -> Optional[Path]:
    """The document the platform wrote for ``workspace_id``, or ``None`` when it is not there."""
    from modules.documents import generation_service

    base = (Path(generation_service.GENERATED_DIR) / str(workspace_id) / GENERATED_SUBDIR).resolve()
    source = (base / filename).resolve()
    return source if source.is_relative_to(base) and source.is_file() else None


async def _file_pieces(source: Path) -> AsyncIterator[bytes]:
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


async def copy_to_session(workspace_id: Any, ticket: int, filename: Any) -> Optional[str]:
    """Write the generated ``filename`` into the ticket's folder; the path written, or ``None``."""
    target = session_copy_path(ticket, filename)
    source = generated_file(workspace_id, filename) if target else None
    if source is None:
        logger.warning("[generate_document] %r is not a document of workspace %s: no copy for ticket %s",
                       filename, workspace_id, ticket)
        return None
    try:
        written = await WorkspaceClient(str(workspace_id)).write_binary(target, _file_pieces(source))
    except Exception:
        logger.exception("[generate_document] the copy for ticket %s could not be written to %s", ticket, target)
        return None
    if not written.get("success"):
        logger.warning("[generate_document] ticket %s's folder is not writable (%s): %s",
                       ticket, target, written.get("error"))
        return None
    return target


async def with_session_copy(result: Any, caller_context: Any, workspace_id: Any) -> Any:
    """``result`` with the copy's path on its first row when a session's ticket made it; a new dict."""
    ticket = session_ticket(caller_context)
    rows = result.get("results") if isinstance(result, dict) and result.get("success") else None
    if ticket is None or not workspace_id or not rows or not isinstance(rows[0], dict):
        return result
    path = await copy_to_session(workspace_id, ticket, rows[0].get("filename"))
    if path is None:
        return result
    first: Dict[str, Any] = {**rows[0], SESSION_COPY_KEY: path, SESSION_COPY_NOTE_KEY: COPY_NOTE.format(path=path)}
    return {**result, "results": [first, *rows[1:]]}


__all__ = ["COPY_NOTE", "SESSION_COPY_KEY", "copy_to_session", "session_copy_path", "session_ticket",
           "with_session_copy"]
