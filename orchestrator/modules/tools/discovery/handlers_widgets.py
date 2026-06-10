"""Widget-config handlers for PlatformActionExecutor (PRD-143 S10).

Reads/writes the public widget-config slice of ``workspace.settings`` — the
exact keys ``api.widgets.config.PUBLIC_WIDGET_CONFIG_KEYS`` whitelists for
browser widgets (proactive engagement, cart-idle, callback form). The
whitelist is imported, never duplicated, so the tool can never write a
settings key the widget layer doesn't expose. ``workspace_id`` comes from
the executor context, never the params.
"""

import logging
from typing import Any, Dict
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


async def get_widget_config(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Read the workspace's public widget config (the slice browser widgets see)."""
    try:
        from api.widgets.config import PUBLIC_WIDGET_CONFIG_KEYS, _project_public_keys
        from core.models.workspaces import Workspace

        ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
        if ws is None:
            return {"success": False, "error": "Workspace not found"}

        return {
            "success": True,
            "widget_config": _project_public_keys(ws.settings) or {},
            "configurable_keys": list(PUBLIC_WIDGET_CONFIG_KEYS),
        }
    except Exception as exc:
        logger.error("[widgets] get_widget_config failed: %s", exc, exc_info=True)
        return {"success": False, "error": str(exc)}


async def update_widget_config(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Set one public widget-config key. Fail-closed: non-whitelisted keys are refused."""
    key = params.get("key")
    value = params.get("config")

    try:
        from api.widgets.config import PUBLIC_WIDGET_CONFIG_KEYS
        from core.models.workspaces import Workspace

        if key not in PUBLIC_WIDGET_CONFIG_KEYS:
            return {
                "success": False,
                "error": f"key must be one of {list(PUBLIC_WIDGET_CONFIG_KEYS)}, got {key!r}",
            }
        if not isinstance(value, dict):
            return {"success": False, "error": "config must be an object"}

        ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
        if ws is None:
            return {"success": False, "error": "Workspace not found"}

        previous = (ws.settings or {}).get(key)
        ws.settings = {**(ws.settings or {}), key: value}
        db.commit()

        return {"success": True, "key": key, "config": value, "previous": previous}
    except Exception as exc:
        db.rollback()
        logger.error("[widgets] update_widget_config failed: %s", exc, exc_info=True)
        return {"success": False, "error": str(exc)}
