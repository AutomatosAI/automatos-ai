"""SDK API-key handlers for PlatformActionExecutor (PRD-143 S11).

List and revoke the workspace's SDK API keys (PRD-37) by delegating to
``core.services.api_key_service.ApiKeyService`` — the exact service layer
``api/api_keys.py`` uses, so masking and workspace scoping cannot drift
between the dashboard and Auto. List returns masked prefixes only. There is
no create handler (F151): a key's full value exists only in its create
response, so keys are created in Settings, never through the LLM context.

BYOK provider keys (api/user_api_keys.py) are deliberately NOT exposed as
tools: adding one requires pasting a raw provider secret into the
conversation, and secrets must never transit the LLM context.
``workspace_id`` comes from the executor context, never the params.
"""

import logging
from typing import Any, Dict
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


async def list_api_keys(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """List this workspace's SDK API keys (masked prefixes only)."""
    try:
        from core.services.api_key_service import ApiKeyService

        keys = ApiKeyService.list_api_keys(db=db, workspace_id=workspace_id)
        return {"success": True, "keys": keys, "count": len(keys)}
    except Exception as exc:
        logger.error("[api_keys] list_api_keys failed: %s", exc, exc_info=True)
        return {"success": False, "error": str(exc)}


async def revoke_api_key(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Revoke (deactivate) an SDK API key. Workspace-scoped by the service."""
    key_id = params.get("key_id")
    if not key_id:
        return {"success": False, "error": "key_id is required"}

    try:
        from uuid import UUID as _UUID

        try:
            key_uuid = _UUID(str(key_id))
        except (TypeError, ValueError):
            return {"success": False, "error": f"key_id must be a UUID, got {key_id!r}"}

        from core.services.api_key_service import ApiKeyService

        revoked = ApiKeyService.revoke_api_key(db=db, key_id=key_uuid, workspace_id=workspace_id)
        if not revoked:
            return {"success": False, "error": "API key not found in this workspace"}

        return {"success": True, "key_id": str(key_uuid), "message": "API key revoked."}
    except Exception as exc:
        db.rollback()
        logger.error("[api_keys] revoke_api_key failed: %s", exc, exc_info=True)
        return {"success": False, "error": str(exc)}
