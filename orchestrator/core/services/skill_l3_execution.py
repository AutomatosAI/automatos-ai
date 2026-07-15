"""
L3 script-execution enablement (PRD-202 S4)
===========================================

The Agent Skills open standard has no governance. Automatos keeps import/read of
a skill (L1 + L2) always-allowed once scanned, but gates **running** its bundled
scripts (L3) behind an explicit, per-workspace, workspace-admin enablement —
**import != executable**.

Enablement is stored on ``workspace.settings["skills_l3_enabled"]`` (a list of
skill ids) — no new table, mirroring the ``auto_autonomy`` settings pattern —
and every enable/disable is written to ``SkillAuditLog``.

Pure DB helpers (no worker, no network): the caller owns the transaction.
"""

from __future__ import annotations

import logging
from typing import Any, List
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

_SETTINGS_KEY = "skills_l3_enabled"


def _enabled_ids(settings: dict) -> List[int]:
    raw = (settings or {}).get(_SETTINGS_KEY) or []
    out: List[int] = []
    for v in raw:
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            continue
    return out


def is_l3_execution_enabled(db: Session, workspace_id: UUID | str, skill_id: int) -> bool:
    """True iff this workspace has explicitly enabled L3 execution for the skill."""
    from core.models.workspaces import Workspace

    ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if ws is None:
        return False
    return int(skill_id) in _enabled_ids(ws.settings or {})


def set_l3_execution_enabled(
    db: Session,
    workspace_id: UUID | str,
    skill_id: int,
    enabled: bool,
    *,
    actor: str = "admin",
) -> List[int]:
    """Enable/disable L3 execution for a skill in this workspace, audited.

    Caller owns the transaction — this stages the settings change, writes a
    ``SkillAuditLog`` row, and flushes. Returns the updated enabled-id list.
    Raises ValueError if the workspace is missing.
    """
    from core.models.core import SkillAuditLog
    from core.models.workspaces import Workspace

    ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if ws is None:
        raise ValueError(f"workspace {workspace_id} not found")

    settings = dict(ws.settings or {})
    ids = set(_enabled_ids(settings))
    skill_id = int(skill_id)

    if enabled:
        ids.add(skill_id)
    else:
        ids.discard(skill_id)

    settings[_SETTINGS_KEY] = sorted(ids)
    # Reassign the whole dict so SQLAlchemy detects the JSON mutation.
    ws.settings = settings

    db.add(SkillAuditLog(
        skill_id=skill_id,
        action="l3_enable" if enabled else "l3_disable",
        action_details={
            "workspace_id": str(workspace_id),
            "enabled": bool(enabled),
            "actor": actor,
        },
        status="success",
        user_id=actor,
    ))
    db.flush()

    logger.info(
        "[skill-l3] workspace=%s skill=%s L3 execution %s by %s",
        workspace_id, skill_id, "ENABLED" if enabled else "DISABLED", actor,
    )
    return sorted(ids)
