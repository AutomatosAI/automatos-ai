"""PRD-256 FX-015 (night 12, F1, F2): who owns a memory written on a chat turn.

The memory tools read their owner (``store_memory``) or viewer (``resume_context``) from
``_user_id``, which the executor injects from the caller context's Clerk id alone
(strip-then-inject). The local edition has no Clerk id: every ``store_memory`` there was
written with no owner, and a ``scope: private`` row with no owner is visible to nobody
(``injection_filter``). Where no Clerk id came, the owner is the driving person's
``users.id``: ``_driving_user_id``, which the executor injects for these actions from the
server context only (``_DRIVER_AWARE_ACTIONS``, both keys stripped from the model's params
first). The handler reads a digit string as that id (``user:{id}``). SaaS keeps the Clerk
id; a turn made for nobody (a widget visitor, a board ticket, a heartbeat) has neither.
"""
from __future__ import annotations

import functools
from typing import Any, Awaitable, Callable, Dict, Mapping

from sqlalchemy.orm import Session

Handler = Callable[[Session, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]

OWNER = "_user_id"
DRIVER = "_driving_user_id"


def with_the_drivers_id(params: Mapping[str, Any]) -> Dict[str, Any]:
    """``params`` with ``_user_id`` the Clerk id it carries, else the driving person's id.
    A new dict, never the caller's."""
    driver = params.get(DRIVER)
    if params.get(OWNER) or driver is None or isinstance(driver, bool) or not str(driver).isdigit():
        return dict(params)
    return {**params, OWNER: str(driver)}


def owned_by_the_driver(handler: Handler) -> Handler:
    """Wrap a memory handler: with no Clerk id, the person driving the turn is the owner."""
    @functools.wraps(handler)
    async def wrapped(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        return await handler(db, workspace_id, with_the_drivers_id(params or {}))
    return wrapped


__all__ = ["owned_by_the_driver", "with_the_drivers_id"]
