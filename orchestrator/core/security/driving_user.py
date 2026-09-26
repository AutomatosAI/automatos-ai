"""F133 — who an agent's call is made for, and whether that person may act as a
workspace owner or admin.

Both answers read ONLY the server-built caller context (the chat's
``build_tool_caller_context``), never a tool call's params, which the model
writes. A lane that acts for nobody (board dispatch, missions, heartbeats) has
no driving user, and so no admin.

- ``driving_user_id``: the ``users.id`` the chat threads as ``driving_user_id``.
- ``driver_is_workspace_admin``: the literal ``system_role: super_admin`` the
  chat writes for a super admin, or the driving user's ACTIVE owner/admin row in
  ``workspace_members``, read fresh at the check. Fails closed on any error.

A public widget turn (``core.security.surface``) is made for nobody, whatever
its caller context names (F155).
"""
from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

from sqlalchemy import text

from core.security.surface import widget_turn

logger = logging.getLogger(__name__)

ADMIN_ROLES = ("owner", "admin")
SUPER_ADMIN = "super_admin"


def driving_user_id(caller_context: Optional[Mapping[str, Any]]) -> Optional[int]:
    """The ``users.id`` the call is made for, or None when it is made for nobody."""
    if widget_turn() or not isinstance(caller_context, Mapping):
        return None
    value = str(caller_context.get("driving_user_id") or "").strip()
    return int(value) if value.isdigit() else None


def driver_is_workspace_admin(db: Any, workspace_id: Any, caller_context: Optional[Mapping[str, Any]]) -> bool:
    """The call is made for a super admin, or for an active owner/admin of this
    workspace. No caller context, or no driving user, is never an admin."""
    if widget_turn() or not isinstance(caller_context, Mapping):
        return False
    if caller_context.get("system_role") == SUPER_ADMIN:
        return True
    user = driving_user_id(caller_context)
    if user is None or workspace_id is None:
        return False
    try:
        with db.begin_nested():
            row = db.execute(
                text("SELECT 1 FROM workspace_members WHERE workspace_id = CAST(:ws AS uuid) "
                     "AND user_id = :user AND role IN ('owner', 'admin') AND is_active IS TRUE LIMIT 1"),
                {"ws": str(workspace_id), "user": user},
            ).first()
    except Exception:  # noqa: BLE001 — an unreadable membership is not an admin
        logger.warning("[driving-user] membership lookup failed for user %s in %s", user, workspace_id,
                       exc_info=True)
        return False
    return row is not None
