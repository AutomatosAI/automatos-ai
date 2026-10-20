"""The AI providers a workspace's data goes to (PRD-266 US-017, D23, K11).

The app's one-time AI consent sheet names them, and shows itself again when the
list changes. ``workspace_ai_providers`` answers a new list of ``{id, name}``,
sorted by id with each id once, built from one small function per source:

* (a) **the workspace's own keys**: an active ``user_api_keys`` row of this
  workspace whose ``byok_overrides`` switch is on, the lane
  ``core.llm.key_resolver`` tries first; chat providers only;
* (b) **platform routes**: every chat provider this edition allows a platform
  key for (``platform_key_allowed``: NVIDIA is the user's own key on saas) with
  a platform key present: an active row in the operator workspace
  (``PLATFORM_KEY_WORKSPACE_ID``), the credential store under the resolver's
  four names, or the config's env key.

Ids are the provider registry's slugs (``core/llm/providers.py``, stable),
names its labels. A key-only provider (Cohere has no chat client) is never a
chat route, so neither source lists it.

It reads key PRESENCE only. It selects the ``provider`` and ``name`` columns,
never a key column; nothing is decrypted, and no key value is returned, logged
or cached. The resolvers the calls themselves use decrypt and log a line per
call, so they are not called here. Each store is read with one query, whatever
the number of providers.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from sqlalchemy import or_
from sqlalchemy.orm import Session

from config import config
from core.llm import providers as registry
from core.models.core import UserApiKey
from core.models.credentials import Credential
from core.models.workspaces import Workspace

logger = logging.getLogger(__name__)

# The workspace setting that switches its own key on, per provider slug
# (api/user_api_keys._enable_proven_key writes it; key_resolver._byok_key reads it).
BYOK_OVERRIDES_KEY = "byok_overrides"

# The names the credential store holds a platform key under, in the resolver's
# order (key_resolver._platform_key, api/user_api_keys.get_platform_key_status).
CREDENTIAL_NAME_FORMS: Tuple[str, ...] = (
    "development_{slug}_api",
    "development_{slug}",
    "{slug}_api",
    "{slug}",
)

ProviderNames = Dict[str, str]
ProviderSource = Callable[[Session, uuid.UUID], ProviderNames]


def workspace_ai_providers(db: Session, workspace_id: Any) -> List[Dict[str, str]]:
    """The AI providers this workspace's data goes to, as a new list of ``{id, name}``.

    Sorted by id, each id once. Raises on a read failure: the caller decides what
    an unknown list means (GET /api/workspaces/current answers ``null``).
    """
    workspace_uuid = uuid.UUID(str(workspace_id))
    found: ProviderNames = {}
    for source in _SOURCES:
        found = {**found, **source(db, workspace_uuid)}
    return [{"id": provider_id, "name": found[provider_id]} for provider_id in sorted(found)]


def _registry_names(slugs: Iterable[str]) -> ProviderNames:
    """Each slug's registry label, keyed by its canonical slug (an unknown slug names itself)."""
    names: ProviderNames = {}
    for slug in slugs:
        spec = registry.get_spec(slug)
        names = {**names, **({spec.slug: spec.label} if spec else {slug: slug})}
    return names


def _chat_slug(stored: Optional[str]) -> Optional[str]:
    """The canonical slug of a stored provider name when it has a chat client, else None."""
    spec = registry.get_spec(stored)
    return spec.slug if spec is not None and spec.chat else None


def _chat_slugs(stored: Iterable[Optional[str]]) -> List[str]:
    return [slug for slug in (_chat_slug(name) for name in stored) if slug]


# ---------------------------------------------------------------------------
# (a) the workspace's own keys
# ---------------------------------------------------------------------------


def _own_keys(db: Session, workspace_id: uuid.UUID) -> ProviderNames:
    """(a) Chat providers whose own key this workspace has switched on and holds active."""
    switched_on = _overrides_on(db.get(Workspace, workspace_id))
    if not switched_on:
        return {}
    rows = (
        db.query(UserApiKey.provider)
        .filter(
            UserApiKey.workspace_id == workspace_id,
            UserApiKey.provider.in_(switched_on),
            UserApiKey.is_active.is_(True),
        )
        .distinct()
        .all()
    )
    return _registry_names(_chat_slugs(row.provider for row in rows))


def _overrides_on(workspace: Optional[Workspace]) -> List[str]:
    """The provider names whose ``byok_overrides`` switch is on (the resolver reads it as truthy)."""
    overrides = ((workspace.settings or {}) if workspace is not None else {}).get(BYOK_OVERRIDES_KEY)
    if not isinstance(overrides, dict):
        return []
    return sorted(str(name) for name, enabled in overrides.items() if enabled)


# ---------------------------------------------------------------------------
# (b) platform routes
# ---------------------------------------------------------------------------


def _platform_routes(db: Session, workspace_id: uuid.UUID) -> ProviderNames:
    """(b) Chat providers this edition allows a platform key for, with a platform key present.

    ``workspace_id`` is unused: a platform key serves every workspace.
    """
    allowed = [slug for slug in registry.chat_slugs() if registry.platform_key_allowed(slug)]
    present = (
        set(_operator_workspace_slugs(db))
        | set(_credential_store_slugs(db, allowed))
        | {slug for slug in allowed if registry.env_api_key(slug)}
    )
    return _registry_names(slug for slug in allowed if slug in present)


def _operator_workspace_slugs(db: Session) -> List[str]:
    """Chat providers with an active key in the operator workspace (PLATFORM_KEY_WORKSPACE_ID)."""
    operator_workspace = _operator_workspace_id()
    if operator_workspace is None:
        return []
    rows = (
        db.query(UserApiKey.provider)
        .filter(UserApiKey.workspace_id == operator_workspace, UserApiKey.is_active.is_(True))
        .distinct()
        .all()
    )
    return _chat_slugs(row.provider for row in rows)


def _operator_workspace_id() -> Optional[uuid.UUID]:
    """The operator workspace, or None when unset or not a UUID (the query would abort the transaction)."""
    raw = (config.PLATFORM_KEY_WORKSPACE_ID or "").strip()
    if not raw:
        return None
    try:
        return uuid.UUID(raw)
    except ValueError:
        logger.warning("PLATFORM_KEY_WORKSPACE_ID is not a UUID; the operator workspace's keys are not counted")
        return None


def _credential_store_slugs(db: Session, slugs: List[str]) -> List[str]:
    """Those of ``slugs`` with an active, unexpired credential under one of the resolver's names.

    Read as the resolver reads it: by name, in any workspace (the platform lane of
    ``CredentialStore.get_credential_by_name``). The resolver also needs an
    ``api_key`` or ``api_token`` field in the row, which only decrypting shows, so
    a row under one of the names counts: the list errs towards naming a provider.
    """
    slug_by_name = {form.format(slug=slug): slug for slug in slugs for form in CREDENTIAL_NAME_FORMS}
    if not slug_by_name:
        return []
    now = datetime.now(timezone.utc).replace(tzinfo=None)  # the column is naive UTC
    rows = (
        db.query(Credential.name)
        .filter(
            Credential.name.in_(sorted(slug_by_name)),
            Credential.is_active.is_(True),
            or_(Credential.expires_at.is_(None), Credential.expires_at > now),
        )
        .distinct()
        .all()
    )
    return sorted({slug_by_name[row.name] for row in rows})


_SOURCES: Tuple[ProviderSource, ...] = (_own_keys, _platform_routes)

__all__ = ["workspace_ai_providers"]
