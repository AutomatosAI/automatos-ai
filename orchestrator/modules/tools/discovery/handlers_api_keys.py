"""SDK API-key handlers for PlatformActionExecutor (PRD-143 S11).

List/create/revoke the workspace's SDK API keys (PRD-37) by delegating to
``core.services.api_key_service.ApiKeyService`` — the exact service layer
``api/api_keys.py`` uses, so key generation, masking and workspace scoping
cannot drift between the dashboard and Auto. The full key appears exactly
once, in the create result, straight from the service (same contract as the
REST route); list returns masked prefixes only.

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


async def create_api_key(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Create an SDK API key. Mirrors the router's validation: key_type in
    {public, server}, public keys need allowed_domains, permission scopes
    come from api.api_keys.VALID_PERMISSIONS (imported, never duplicated)."""
    name = (params.get("name") or "").strip()
    key_type = params.get("key_type") or "server"
    permissions = params.get("permissions") or []
    allowed_domains = params.get("allowed_domains")

    if not name:
        return {"success": False, "error": "name is required"}
    if key_type not in ("public", "server"):
        return {"success": False, "error": f"key_type must be 'public' or 'server', got {key_type!r}"}
    if key_type == "public" and not allowed_domains:
        return {"success": False, "error": "Public keys require a non-empty allowed_domains list"}

    try:
        from api.api_keys import VALID_PERMISSIONS

        invalid = [p for p in permissions if p not in VALID_PERMISSIONS]
        if invalid:
            return {
                "success": False,
                "error": f"Invalid permissions: {invalid}. Valid: {VALID_PERMISSIONS}",
            }

        from core.services.api_key_service import ApiKeyService

        result = ApiKeyService.create_api_key(
            db=db,
            workspace_id=workspace_id,
            name=name,
            key_type=key_type,
            permissions=list(permissions),
            allowed_domains=allowed_domains,
        )
        return {
            "success": True,
            "key": result,
            "message": (
                "API key created. The full key is shown here exactly once — "
                "store it now; only the masked prefix is retrievable later."
            ),
        }
    except Exception as exc:
        db.rollback()
        logger.error("[api_keys] create_api_key failed: %s", exc, exc_info=True)
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
