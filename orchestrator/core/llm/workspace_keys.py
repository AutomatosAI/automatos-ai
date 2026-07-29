"""Resolve provider API keys from the operator workspace's own key store.

Background workers (embeddings, system LLM) and the pilot chat fallback
historically read only the platform credential store (``credentials`` table,
e.g. the ``openrouter_api`` row). When the operator's real key lives in their
workspace key store (``user_api_keys``), that credential-store copy drifts —
it held a key deleted on the provider side for months while every memory
write/search 401'd (2026-07-30 incident).

This helper points those paths at the workspace key store directly, so the
key is read from where it actually lives instead of being duplicated into a
second slot. Enabled only when ``PLATFORM_KEY_WORKSPACE_ID`` is set; returns
``None`` otherwise so other deployments keep the legacy behaviour unchanged.

Row selection mirrors ``AgentFactory._resolve_api_key`` BYOK ordering
(``last_used_at DESC NULLS LAST``) so these paths use the exact key already
proven live by chat.
"""

import logging
from typing import Optional

from config import config

logger = logging.getLogger(__name__)


def get_platform_workspace_key(provider: str) -> Optional[str]:
    """Return the operator workspace's active key for ``provider``, or None.

    Never raises: any lookup/decrypt failure logs a warning and returns None
    so callers fall through to the credential store / env resolution tiers.
    """
    workspace_id = (config.PLATFORM_KEY_WORKSPACE_ID or "").strip()
    if not workspace_id:
        return None

    try:
        from core.database.database import SessionLocal
        from core.models.core import UserApiKey
        from core.credentials.encryption import get_encryption_service

        db = SessionLocal()
        try:
            row = (
                db.query(UserApiKey)
                .filter(
                    UserApiKey.workspace_id == workspace_id,
                    UserApiKey.provider == provider,
                    UserApiKey.is_active.is_(True),
                )
                .order_by(UserApiKey.last_used_at.desc().nullslast())
                .first()
            )
            if not row:
                return None
            key = get_encryption_service().decrypt(row.encrypted_key)
            logger.info(
                "Resolved '%s' key from operator workspace key store "
                "(PLATFORM_KEY_WORKSPACE_ID)",
                provider,
            )
            return key
        finally:
            db.close()
    except Exception as e:  # noqa: BLE001 — resolution must never break callers
        logger.warning("Workspace-key resolution failed for '%s': %s", provider, e)
        return None
