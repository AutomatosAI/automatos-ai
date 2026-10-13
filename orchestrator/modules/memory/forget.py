"""P256-FIX-N14-4 (night 14, F418): forget a memory by the id the Memory page shows.

Night 14: the owner asked twice to delete a standing rule and both deletes answered 'Memory not
found or not owned by this workspace'. Auto's delete looked for the id among the first 500 rows
of the workspace's own namespace (scroll order, so a workspace past 500 rows lost every newer
one), and the Memory page's looked only in that one namespace, while the page also lists the
agents' and the daily log's. Both now go through ``forget``: the store reads the point itself
and deletes it when its own namespace is this workspace's (``DurableMemoryStore
.delete_in_workspace``); a private memory with an owner is forgotten by that owner alone.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

NOT_FOUND_ERROR = "Memory not found or not owned by this workspace"
NOT_THEIRS_ERROR = "That memory is another member's private note: only they can delete it"


def subject_of(user_pk: Any) -> Optional[str]:
    """The ``user:{users.id}`` subject a memory's owner is stored as, or None for no person."""
    if user_pk is None or isinstance(user_pk, bool) or not str(user_pk).isdigit():
        return None
    return f"user:{int(str(user_pk))}"


async def forget(memory_id: str, workspace_id: Any, deleter_subject_id: Optional[str]) -> Dict[str, Any]:
    """Delete the memory ``memory_id`` from this workspace for ``deleter_subject_id``; the answer
    says plainly what happened. A store that is down or refuses raises (the caller logs it)."""
    from modules.memory.durable_store import DELETED, NOT_THEIRS
    from modules.memory.unified_memory_service import get_unified_memory_service

    service = get_unified_memory_service()
    outcome = await service._durable.delete_in_workspace(str(memory_id), str(workspace_id), deleter_subject_id)
    if outcome == DELETED:
        logger.info("Memory %s deleted in workspace %s", memory_id, workspace_id)
        return {"success": True, "message": f"Memory {memory_id} deleted"}
    return {"success": False, "error": NOT_THEIRS_ERROR if outcome == NOT_THEIRS else NOT_FOUND_ERROR}


__all__ = ["NOT_FOUND_ERROR", "NOT_THEIRS_ERROR", "forget", "subject_of"]
