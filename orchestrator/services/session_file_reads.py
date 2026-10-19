"""The session tools that read a file whole: ``read_step_file`` (F161) and ``read_document`` (F448).

``read_step_file`` reads a file the ticket lists by its id (``services/step_files.py``): an
earlier step of its mission saved it, or it is on a card the owner's words name (F448).

``read_document`` (P256-FIX-N17-9, night 17, F448): WRITER's brief named
harbourline-green-coffee-price-list.csv and its session had only ``search_documents``,
which timed out twice; Auto had read the same CSV with ``platform_read_document``. A session
now reads one of the owner's knowledge-base documents whole, a page at a time, by its
filename or id, through the same action and the same PRD-157 team scoping (the executor
mints ``_agent_id`` from the ticket's agent). A spreadsheet's ``workspace_path`` lies outside
the session's folder, so its copy is put inside it, and the answer names that path.

Both run through the API agents' own executor (``session_tools.call_tool`` hands them the
ticket's context), never a second execution path. ``SessionToolRefused`` and the dispatcher
are imported lazily: ``session_tools`` imports this module to build its list.
"""
from __future__ import annotations

import asyncio
import logging
import posixpath
from typing import Any, Dict, Optional
from uuid import UUID

logger = logging.getLogger(__name__)

# What read_document forwards: the action's own names. ``document_name`` is the filename.
READ_DOCUMENT_FIELDS = {"document_id": "document_id", "filename": "filename", "document_name": "filename",
                        "page": "page"}
DOC_COPY_NAME = "doc-{document_id}-{name}"
WORKSPACE_STORAGE = "workspace"
NO_SPREADSHEET_COPY = ("This is a spreadsheet and no copy could be put in your folder: read every page (next_page), "
                       "say how many of its row_count rows you read, and call no figure exact.")


def _refused(message: str) -> Exception:
    from services.session_tools import SessionToolRefused

    return SessionToolRefused(message)


def scope_read_step_file(params: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    """An id from the ticket's list and nothing else: no path, no other field."""
    raw = str(params.get("file_id") or "").strip()
    try:
        return {"file_id": str(UUID(raw))}
    except ValueError:
        raise _refused(
            "read_step_file takes the id your ticket lists beside the file "
            "(e.g. 125a3625-1351-435e-a128-6eca909f17b6), never a path or a name."
        ) from None


async def run_read_step_file(db: Any, params: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    """F161: read one file the ticket lists: another step of its mission saved it, or it is
    on a card its owner named (F448). The read itself is the executor's own
    ``platform_get_deliverable``."""
    from config import config
    from modules.tools.execution.unified_executor import UnifiedToolExecutor
    from services.session_tools import MAX_TOOL_RESULT_CHARS, PLATFORM_DISPATCHER, STEP_FILE_FRAME_CHARS
    from services.step_files import NOT_A_MISSION_STEP, files_for_ticket, not_listed, step_file_answer

    files = files_for_ticket(db, ticket_id=ctx.task_id, workspace_id=ctx.workspace_id)
    if files is None:
        raise _refused(NOT_A_MISSION_STEP)
    chosen = next((f for f in files if f.deliverable_id == params["file_id"]), None)
    if chosen is None:
        raise _refused(not_listed(files))
    result = await UnifiedToolExecutor(db).execute_tool(
        tool_name=PLATFORM_DISPATCHER,
        parameters={
            "action": "platform_get_deliverable",
            "params": {"deliverable_id": chosen.deliverable_id, "include_content": True},
        },
        agent_id=int(ctx.agent_id or 0),
        workspace_id=ctx.workspace_id,
        trace_id=f"session:{ctx.task_id}:read_step_file",
        caller_context=None,
    )
    configured = int(getattr(config, "SESSION_STEP_FILE_MAX_CHARS", 0) or 0)
    ceiling = MAX_TOOL_RESULT_CHARS - STEP_FILE_FRAME_CHARS
    return await step_file_answer(chosen, result, min(configured, ceiling) if configured > 0 else ceiling, ctx)


def scope_read_document(params: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    """The document by its id or its filename, and the page: nothing else is forwarded."""
    out: Dict[str, Any] = {}
    for key, value in params.items():
        target = READ_DOCUMENT_FIELDS.get(key)
        if target and value is not None and str(value).strip() and target not in out:
            out[target] = value
    if "document_id" not in out and "filename" not in out:
        raise _refused("read_document needs the document's filename (as your brief names it) or its document_id.")
    return out


async def run_read_document(db: Any, params: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    """``platform_read_document`` as the ticket's agent (no user context: the executor mints
    ``_agent_id``, so the agent's team scoping applies), a spreadsheet's copy put in the folder."""
    from modules.tools.execution.unified_executor import UnifiedToolExecutor
    from services.session_tools import PLATFORM_DISPATCHER

    result = await UnifiedToolExecutor(db).execute_tool(
        tool_name=PLATFORM_DISPATCHER,
        parameters={"action": "platform_read_document", "params": params},
        agent_id=int(ctx.agent_id or 0),
        workspace_id=ctx.workspace_id,
        trace_id=f"session:{ctx.task_id}:read_document",
        caller_context=None,
    )
    if not isinstance(result, dict) or not result.get("success") or not result.get("workspace_path"):
        return result
    return await _with_session_copy(result, ctx)


async def _with_session_copy(result: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    """A spreadsheet's answer with its ``workspace_path`` replaced by a copy in the ticket's
    folder, so counting it with code works there; without a copy, no path and the honest note."""
    copied = await _session_copy(ctx, result.get("document_id"), str(result["workspace_path"]))
    if copied is not None:
        return {**result, "workspace_path": copied}
    kept = {k: v for k, v in result.items() if k not in ("workspace_path", "count_with_code", "sheets")}
    return {**kept, "note": NO_SPREADSHEET_COPY}


async def _session_copy(ctx: Any, document_id: Any, workspace_path: str) -> Optional[str]:
    from modules.documents.thumbnails.sources import read_source
    from modules.tools.execution.session_document_folder import write_to_session

    data = await asyncio.to_thread(read_source, ctx.workspace_id, WORKSPACE_STORAGE, workspace_path)
    if data is None:
        logger.warning("[F448] ticket %s: %s could not be read for its folder", ctx.task_id, workspace_path)
        return None
    name = DOC_COPY_NAME.format(document_id=document_id, name=posixpath.basename(workspace_path))
    return await write_to_session(ctx.workspace_id, int(ctx.task_id), name, data)
