"""Power-mode handler for PlatformActionExecutor (PRD-142 Wave 4, W4-S5).

Sets the workspace's default power mode (``workspace.settings['power_mode']``) —
the persistent per-workspace knob a Mission run inherits when its run_config
doesn't pin one (resolved in ``coordinator_service._workspace_power_mode_default``).
This is the platform action HARNESS's ``power_mode_upgrade`` / ``power_mode_downgrade``
prescription applies. The ``workspace_id`` comes from the executor context, never
the params, so the knob can only ever be set for the caller's own workspace.
"""

import logging
from typing import Any, Dict
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Canonical compute/quality tiers. The source of truth (with each mode's caps) is
# ``coordinator_service._POWER_MODE_DEFAULTS``; kept as a local literal here to
# avoid importing that heavy module into a handler. Keep the two in sync.
_VALID_POWER_MODES = ("light", "standard", "max")


async def set_power_mode(
    db: Session, workspace_id: UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Set the workspace default power mode.

    Validates the mode against the canonical tier set before any write — an
    unknown mode is refused (fail-closed) rather than silently stored, so a
    Mission run never inherits a garbage tier. Writes ``settings['power_mode']``
    via a fresh dict (SQLAlchemy mutation detection on the JSONB column).
    """
    mode = params.get("power_mode")
    if mode not in _VALID_POWER_MODES:
        return {
            "success": False,
            "error": f"power_mode must be one of {list(_VALID_POWER_MODES)}, got {mode!r}",
        }

    try:
        from core.models.workspaces import Workspace

        ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
        if ws is None:
            return {"success": False, "error": "Workspace not found"}

        previous = (ws.settings or {}).get("power_mode")
        ws.settings = {**(ws.settings or {}), "power_mode": mode}
        db.commit()

        return {
            "success": True,
            "data": {
                "workspace_id": str(workspace_id),
                "power_mode": mode,
                "previous_power_mode": previous,
            },
        }
    except Exception as exc:
        db.rollback()
        logger.error("[power] set_power_mode failed: %s", exc, exc_info=True)
        return {"success": False, "error": str(exc)}
