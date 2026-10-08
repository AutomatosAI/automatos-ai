"""PRD-256 FX-015 (night 12, F1, F2): who owns a memory written on a chat turn.

The memory tools record their owner from the server's caller context as ``_user_id``
(``platform_executor``, strip-then-inject). That was the Clerk id alone, which the local
edition has none of: every ``store_memory`` there was written with no owner, and a
``scope: private`` row with no owner is visible to nobody (``injection_filter``). On the
local edition the owner is the driving person's ``users.id`` the chat threads as
``driving_user_id``; the handler reads a digit string as that id (``user:{id}``). SaaS
keeps the Clerk id, as before. A turn made for nobody (a widget visitor, a board ticket,
a heartbeat) has neither, so its memories have no owner.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

from core.security.driving_user import driving_user_id


def memory_owner_id(caller_context: Optional[Mapping[str, Any]]) -> Optional[str]:
    """The Clerk id the chat threads, else the driving person's ``users.id``, else None."""
    if not isinstance(caller_context, Mapping):
        return None
    clerk = caller_context.get("user_id")
    if clerk:
        return str(clerk)
    person = driving_user_id(caller_context)
    return str(person) if person is not None else None


__all__ = ["memory_owner_id"]
