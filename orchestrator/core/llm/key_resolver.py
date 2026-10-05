"""One answer to "which key pays for this provider's calls" (F244, night 7).

The Analytics credit card looked for a narrower set of keys than the calls use: a
workspace BYOK row, then the env var. Night 7's calls ran on the operator workspace's
key (PLATFORM_KEY_WORKSPACE_ID). So the card said "No OpenRouter API key configured"
beside $22.65 of OpenRouter spend, and the owner never saw the balance run down.

The agent factory and the card now both use this resolver, in the calls' own order:
1. the workspace's BYOK key, when its override is on;
2. the operator workspace's key;
3. the credential store;
4. the config's env key.

A provider that is BYO-key only in this edition never resolves from 2-4 (PRD-236).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ResolvedKey:
    """Result of API key resolution with source metadata."""
    api_key: str
    source: str  # "byok", "platform_workspace", "platform", "env"
    is_byok: bool
    provider: str = ""
    # #873: a BYOK key's own endpoint (Azure); None keeps the client's own default
    base_url: Optional[str] = None


def resolve_provider_key(db: Any, provider_name: str, *, workspace_id: Any = None,
                         agent_name: str = "") -> Optional[ResolvedKey]:
    """The key that pays for ``provider_name``'s calls in ``workspace_id``, or None."""
    if workspace_id:
        byok = _byok_key(db, provider_name, workspace_id)
        if byok is not None:
            return byok
    from core.llm.providers import platform_key_allowed

    if not platform_key_allowed(provider_name):
        logger.info("No platform key lane for '%s' in this edition (BYO key only) - %s", provider_name, agent_name)
        return None
    return _platform_key(provider_name, agent_name)


def _byok_key(db: Any, provider_name: str, workspace_id: Any) -> Optional[ResolvedKey]:
    """The workspace's own key, when it has switched the override on for this provider."""
    try:
        from core.credentials.encryption import get_encryption_service
        from core.models.core import UserApiKey
        from core.models.workspaces import Workspace

        workspace = db.query(Workspace).get(workspace_id)
        if not ((workspace.settings or {}).get("byok_overrides", {}) if workspace else {}).get(provider_name, False):
            return None
        row = (db.query(UserApiKey)
               .filter(UserApiKey.workspace_id == workspace_id, UserApiKey.provider == provider_name,
                       UserApiKey.is_active.is_(True))
               .order_by(UserApiKey.last_used_at.desc().nullslast()).first())
        if row is None:
            logger.info("BYOK enabled but no active key for '%s', falling through", provider_name)
            return None
        logger.info("Resolved BYOK API key for '%s' workspace=%s", provider_name, workspace_id)
        return ResolvedKey(api_key=get_encryption_service().decrypt(row.encrypted_key), source="byok", is_byok=True,
                           provider=provider_name, base_url=getattr(row, "base_url", None) or None)
    except Exception:
        logger.exception("BYOK key lookup failed for %s; trying the platform's keys", provider_name)
        return None


def byok_endpoint(provider_name: str, workspace_id: Any) -> Optional[str]:
    """The endpoint saved with the workspace's BYOK key for ``provider_name`` (#873), or None.

    The same row ``resolve_provider_key`` picks, read in a session of its own: an
    ``LLMManager`` built from a key alone asks for it (``byok_endpoint.with_key_endpoint``).
    """
    from core.database.database import SessionLocal

    db = SessionLocal()
    try:
        resolved = _byok_key(db, provider_name, workspace_id)
    finally:
        db.close()
    return resolved.base_url if resolved else None


def _platform_key(provider_name: str, agent_name: str) -> Optional[ResolvedKey]:
    """The operator workspace's key, then the credential store's, then the env's."""
    from core.credentials.resolver import get_credential_resolver
    from core.llm.providers import env_api_key
    from core.llm.workspace_keys import get_platform_workspace_key

    ws_key = get_platform_workspace_key(provider_name)
    if ws_key:
        logger.info("Resolved platform key from operator workspace store for '%s' (%s)", provider_name, agent_name)
        return ResolvedKey(api_key=ws_key, source="platform_workspace", is_byok=False, provider=provider_name)
    resolver = get_credential_resolver()
    for name in (f"development_{provider_name}_api", f"development_{provider_name}", f"{provider_name}_api",
                 provider_name):
        key = _credential(resolver, name)
        if key:
            logger.info("Resolved platform API key from credential '%s' for %s", name, agent_name)
            return ResolvedKey(api_key=key, source="platform", is_byok=False, provider=provider_name)
    key = env_api_key(provider_name)
    if key:
        logger.info("Using config API key for %s for %s", provider_name, agent_name)
        return ResolvedKey(api_key=key, source="env", is_byok=False, provider=provider_name)
    return None


def _credential(resolver: Any, name: str) -> Optional[str]:
    try:
        return resolver.get_credential_field(name, "api_key") or resolver.get_credential_field(name, "api_token")
    except Exception:  # noqa: BLE001 - a missing credential is the next name's turn, as the factory always did
        return None


__all__ = ["ResolvedKey", "byok_endpoint", "resolve_provider_key"]
