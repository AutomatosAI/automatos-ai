"""A chat's agent-switch history (``chats.agent_switches``).

Moved out of ``api/chat.py`` for #935, when the switch record's timestamp gained its
UTC offset: the route was over the function-length rule and the module over the
file-size rule, so the record lives here.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

DEFAULT_SWITCH_REASON = "User requested switch"


def record_agent_switch(
    db: Session, chat: Any, from_agent_id: int, to_agent_id: int, reason: Optional[str]
) -> None:
    """Append one switch to the chat's history, stamped in UTC, and write it.

    Builds a new list rather than appending to the chat's own, and reads a history
    stored as JSON text as well as a list. The caller commits."""
    existing = getattr(chat, "agent_switches", None) or []
    if isinstance(existing, str):
        existing = json.loads(existing)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "from_agent_id": from_agent_id,
        "to_agent_id": to_agent_id,
        "reason": reason or DEFAULT_SWITCH_REASON,
    }
    db.execute(
        text("UPDATE chats SET agent_switches = :switches WHERE id = :chat_id"),
        {"switches": json.dumps([*existing, record]), "chat_id": chat.id},
    )
