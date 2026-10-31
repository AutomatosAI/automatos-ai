"""Who a mission action is made for (F166), moved out of platform_executor.py (PRD-268 US-C03b, K3).

The executor injects it as ``_created_by`` for the mission actions; its body is unchanged.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

__all__ = ["person_for_mission"]


def person_for_mission(db, caller_context: Optional[Dict[str, Any]]) -> Optional[str]:
    """F166: who a mission action is made for, as the mission records it — the
    chat's Clerk id, or on the local edition (no Clerk id) the person's email, read
    from the server-threaded ``driving_user_id``. The email is what the local REST
    API records (``created_by = ctx.user.id``). Never the bare ``users.id``: a digit
    string in ``created_by`` is already an agent id wherever no person drove the
    action. A widget turn is made for nobody (F155); an unreadable user is nobody."""
    from core.security.driving_user import driving_user_id
    from core.security.surface import widget_turn

    if widget_turn():
        return None
    ctx = caller_context if isinstance(caller_context, dict) else {}
    if ctx.get("user_id"):
        return str(ctx["user_id"])
    user = driving_user_id(ctx)
    if user is None:
        return None
    try:
        from core.models.core import User

        with db.begin_nested():  # a failed lookup never poisons the caller's transaction
            row = db.query(User.email).filter(User.id == user).first()
    except Exception:  # noqa: BLE001 — fails closed: nobody, so F036 still leaves it to the owner
        logger.warning("[PlatformExecutor] no email for driving user %s — the call is made for nobody", user,
                       exc_info=True)
        return None
    email = str(row[0] or "").strip() if row else ""
    return email or None
