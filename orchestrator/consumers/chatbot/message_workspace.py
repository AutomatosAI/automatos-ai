"""The workspace a chat message is saved in (``ChatService.save_message``).

A message needs a workspace that exists. The local edition's development workspace (the fixed id
below) is made on first use; any other unknown id is refused. Moved out of ``save_message``
unchanged, so the function stays within its size.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Union

from core.models import Workspace

DEV_WORKSPACE_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")


def _dev_workspace() -> Workspace:
    """The local edition's development workspace, as ``save_message`` has always made it."""
    now = datetime.utcnow()
    return Workspace(
        id=DEV_WORKSPACE_ID,
        name="Dev Workspace",
        slug="dev",
        plan="basic",  # PRD-222 W2·S1: entry tier (renamed from 'starter')
        plan_limits={},
        settings={},
        is_personal=True,
        is_active=True,
        created_at=now,
        updated_at=now,
    )


def message_workspace(db: Any, workspace_id: Union[str, uuid.UUID, None]) -> uuid.UUID:
    """The workspace's id as a UUID once it exists: the development workspace is made on first use.

    Raises ``ValueError`` for a missing, malformed or unknown id, as ``save_message`` always has.
    """
    if not workspace_id:
        raise ValueError("workspace_id is required to save messages")
    try:
        wanted = workspace_id if isinstance(workspace_id, uuid.UUID) else uuid.UUID(str(workspace_id))
    except ValueError:
        raise ValueError("Invalid workspace_id format") from None
    if db.query(Workspace).filter(Workspace.id == wanted).first() is None:
        if wanted != DEV_WORKSPACE_ID:
            raise ValueError(f"workspace_id does not exist: {wanted}")
        db.add(_dev_workspace())
        db.commit()
    return wanted


__all__ = ["DEV_WORKSPACE_ID", "message_workspace"]
