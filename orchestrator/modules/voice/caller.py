"""
PRD-207 / F149: who is calling, and which agent answers
========================================================

Two strict resolutions the live-voice lane makes before anything is
attributed: the caller's integer ``users.id`` at mint (no fallback principal
ever mints a call), and the agent a call names (only the call's own
workspace's agent, or a platform system agent). Moved out of
``api/voice_retell.py`` by PRD-258 US-001 (D6: that file never grows).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from sqlalchemy.orm import Session

from core.auth.principal import claimed_user_pk

logger = logging.getLogger(__name__)


def resolve_caller_int_id(db: Session, ctx: Any) -> Optional[int]:
    """STRICT #513 resolution: the caller's integer ``users.id`` or None.

    Deliberately not ``api.chat.get_user_id`` — that helper falls back to a
    default user for principal-less paths, and a fallback principal must
    never mint a call attributed to somebody. No user → no mint.

    The local operator and a paired device (PRD-266) carry their PK on ``raw_claims``;
    it is read by ``core.auth.principal.resolve_user_pk``'s own claim rule
    (``claimed_user_pk``), so the anonymous local operator mints as itself (403 before).
    """
    user = getattr(ctx, "user", None)
    if user is None:
        return None
    uid = getattr(user, "id", None)
    if isinstance(uid, int):
        return uid
    claimed = claimed_user_pk(ctx)
    if claimed is not None:
        return claimed
    clerk_uid = getattr(user, "clerk_user_id", None) or (uid if isinstance(uid, str) else None)
    if not clerk_uid:
        return None
    from core.models.core import User

    row = db.query(User.id).filter(User.clerk_user_id == str(clerk_uid)).first()
    return int(row[0]) if row else None


def named_call_agent(db: Session, workspace_id: Any, raw_agent_id: Any) -> Optional[int]:
    """The agent a call names, when it is one of the call's workspace's own or a
    platform system agent (F149); otherwise None, and the default answers."""
    from core.security.workspace_scope import agent_in_workspace

    try:
        agent_id = int(raw_agent_id) if raw_agent_id else None
    except (TypeError, ValueError):
        return None
    if agent_id is not None and not agent_in_workspace(db, agent_id, workspace_id):
        logger.warning("[voice] agent %s is not in workspace %s — the default answers", agent_id, workspace_id)
        return None
    return agent_id
