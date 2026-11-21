"""One answer to "which key pays for this provider's calls" (F244, night 7).

The Analytics credit card looked for a narrower set of keys than the calls use: a
workspace BYOK row, then the env var. Night 7's calls ran on the operator workspace's
key (PLATFORM_KEY_WORKSPACE_ID). So the card said "No OpenRouter API key configured"
beside $22.65 of OpenRouter spend, and the owner never saw the balance run down.

The agent factory and the card now both use this resolver, in the calls' own order:
1. the workspace's BYOK key, when its override is on;
2. the operator workspace's key;
3. the credential store;
4. the config's env key;
5. the pod's own cloud role, for a provider whose client can sign with it (Bedrock),
   when ``AWS_USE_WORKLOAD_IDENTITY`` is on (PRD-278 US-103). It resolves with no key
   (``api_key`` empty, ``source`` ``workload_identity``): ``carries_calls`` says it can
   still pay for a call. It never answers after the workspace's own key could not be read
   (``OwnKeyLookupFailed``): a workspace's choice is not replaced by the install's identity.

A provider that is BYO-key only in this edition never resolves from 2-5 (PRD-236).
``own_keys_only`` (PRD-265) stops at 1: a workspace whose credits can't pay for the
platform's keys runs on its own keys or not at all.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

# PRD-278 US-103: the call is signed by the pod's own role; there is no key to send.
SOURCE_WORKLOAD_IDENTITY = "workload_identity"


@dataclass
class ResolvedKey:
    """Result of API key resolution with source metadata."""
    api_key: str
    source: str  # "byok", "platform_workspace", "platform", "env"
    is_byok: bool
    provider: str = ""
    # #873: a BYOK key's own endpoint (Azure); None keeps the client's own default
    base_url: Optional[str] = None


@dataclass(frozen=True)
class OwnKeyLookupFailed:
    """The workspace's own key could not be read (a database or decrypt error, logged).

    The platform's stored and env keys are still tried, as before; the install's own role
    is not (PRD-278 US-103).
    """
    provider: str


def resolve_provider_key(db: Any, provider_name: str, *, workspace_id: Any = None,
                         agent_name: str = "", own_keys_only: bool = False) -> Optional[ResolvedKey]:
    """The key that pays for ``provider_name``'s calls in ``workspace_id``, or None."""
    own_key_lookup_failed = False
    if workspace_id:
        byok = _own_key(db, provider_name, workspace_id)
        own_key_lookup_failed = isinstance(byok, OwnKeyLookupFailed)
        if byok is not None and not own_key_lookup_failed:
            return byok
    if own_keys_only:
        return None
    from core.llm.providers import platform_key_allowed

    if not platform_key_allowed(provider_name):
        logger.info("No platform key lane for '%s' in this edition (BYO key only) - %s", provider_name, agent_name)
        return None
    platform = _platform_key(provider_name, agent_name)
    if platform is not None:
        return platform
    return _role_key(provider_name, agent_name, own_key_lookup_failed=own_key_lookup_failed)


def _own_key_names(provider_name: str) -> tuple:
    """The names the workspace's own key is looked up under: the stored name; with
    ``AWS_USE_WORKLOAD_IDENTITY`` on and a provider the install's role signs for (Bedrock),
    its canonical slug first, the one a workspace's key is saved under (PRD-278 US-105)."""
    from core.llm.providers import normalize_slug, runs_on_workload_identity

    if not runs_on_workload_identity(provider_name):
        return (provider_name,)
    slug = normalize_slug(provider_name)
    return (slug,) if slug == provider_name else (slug, provider_name)


def _own_key(db: Any, provider_name: str, workspace_id: Any) -> Union[ResolvedKey, OwnKeyLookupFailed, None]:
    """The workspace's own key under any of ``_own_key_names``: the first found wins; when
    none is found and any lookup failed, ``OwnKeyLookupFailed``."""
    failed: Optional[OwnKeyLookupFailed] = None
    for name in _own_key_names(provider_name):
        found = _byok_key(db, name, workspace_id)
        if isinstance(found, ResolvedKey):
            return found
        failed = failed or (found if isinstance(found, OwnKeyLookupFailed) else None)
    return failed


def _role_key(provider_name: str, agent_name: str, *, own_key_lookup_failed: bool) -> Optional[ResolvedKey]:
    """The install's own role, the last tier (PRD-278 US-103): only for a provider that can
    use it, with ``AWS_USE_WORKLOAD_IDENTITY`` on, and never after the workspace's own key
    could not be read."""
    from core.llm.providers import runs_on_workload_identity

    if not runs_on_workload_identity(provider_name):
        return None
    if own_key_lookup_failed:
        logger.warning("Not using the install's own role for %s (%s): the workspace's own key could not be read",
                       provider_name, agent_name)
        return None
    logger.info("Using the pod's own role for %s for %s (AWS_USE_WORKLOAD_IDENTITY)", provider_name, agent_name)
    return ResolvedKey(api_key="", source=SOURCE_WORKLOAD_IDENTITY, is_byok=False, provider=provider_name)


def is_workload_identity(resolved: Optional[ResolvedKey]) -> bool:
    """Did the resolver answer with the install's own role (PRD-278 US-103)?

    The one signal that lets a Bedrock client use the SDK's default chain: it rides on
    ``LLMConfig.workload_identity`` to ``bedrock_client.runtime_client``.
    """
    return resolved is not None and getattr(resolved, "source", None) == SOURCE_WORKLOAD_IDENTITY


def carries_calls(resolved: Optional[ResolvedKey]) -> bool:
    """Can ``resolved`` pay for a call: a key to send, or the pod's own role (PRD-278 US-103)?"""
    if resolved is None:
        return False
    return bool(resolved.api_key) or is_workload_identity(resolved)


def with_workload_identity(llm_config: Any) -> Any:
    """A settings-tier config for a provider the install's role can sign for (Bedrock), with
    ``AWS_USE_WORKLOAD_IDENTITY`` on. With no key it is marked to run on that role, in the
    install's region (``AWS_REGION``): the settings path's twin of ``_platform_key``'s last
    tier. With a key it keeps the region its credential names, else ``AWS_REGION`` (PRD-278
    US-105, ``providers.workload_identity_region``). Any other config comes back unchanged."""
    from dataclasses import replace

    from core.llm.providers import runs_on_workload_identity, workload_identity_region

    provider = llm_config.provider.value
    if not runs_on_workload_identity(provider):
        return llm_config
    on_role = llm_config.workload_identity or not llm_config.api_key
    return replace(llm_config, workload_identity=on_role,
                   base_url=workload_identity_region(provider, llm_config.base_url, on_role=on_role))


def _byok_key(db: Any, provider_name: str, workspace_id: Any) -> Union[ResolvedKey, OwnKeyLookupFailed, None]:
    """The workspace's own key, when it has switched the override on for this provider;
    ``OwnKeyLookupFailed`` when it could not be read."""
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
        return OwnKeyLookupFailed(provider=provider_name)


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
    return resolved.base_url if isinstance(resolved, ResolvedKey) else None


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


__all__ = [
    "SOURCE_WORKLOAD_IDENTITY", "OwnKeyLookupFailed", "ResolvedKey", "byok_endpoint", "carries_calls", "is_workload_identity",
    "resolve_provider_key", "with_workload_identity",
]
