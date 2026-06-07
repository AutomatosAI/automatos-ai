"""Autonomy-level handlers.

Thin wrappers over ``core.services.auto_autonomy`` — the canonical reader/writer
for ``workspace.settings.autonomy``. The executor owns permission gating; these
just read/write and commit.
"""

from __future__ import annotations

import logging
from typing import Any, Dict
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

_LEVEL_MEANING: Dict[str, str] = {
    "standard": (
        "Supervised. Admin-only tools need an admin; actions flagged "
        "requires_confirmation stop and ask before running."
    ),
    "full": (
        "Unsupervised. Auto runs as admin and the confirmation gate is skipped — "
        "writes and the destructive deletes execute without asking. Rate limits and "
        "the agent-hierarchy check still apply."
    ),
}


async def get_autonomy_level(
    db: Session, workspace_id: UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Return the workspace's current autonomy level and what it means."""
    from core.services.auto_autonomy import get_autonomy_level as _get

    try:
        level = _get(db, workspace_id)
        return {
            "success": True,
            "data": {"level": level, "meaning": _LEVEL_MEANING.get(level, "")},
        }
    except Exception as exc:
        logger.error("[autonomy] get level failed: %s", exc, exc_info=True)
        return {"success": False, "error": str(exc)}


async def set_autonomy_level(
    db: Session, workspace_id: UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Persist a new autonomy ``level`` to ``workspace.settings.autonomy``."""
    from core.services.auto_autonomy import set_autonomy_level as _set

    level = params.get("level")
    if not level:
        return {"success": False, "error": "level is required (standard | full)"}

    try:
        result = _set(db, workspace_id, level)
        db.commit()
        return {
            "success": True,
            "data": {**result, "meaning": _LEVEL_MEANING.get(result["level"], "")},
        }
    except ValueError as exc:
        return {"success": False, "error": str(exc)}
    except Exception as exc:
        logger.error("[autonomy] set level failed: %s", exc, exc_info=True)
        db.rollback()
        return {"success": False, "error": str(exc)}
