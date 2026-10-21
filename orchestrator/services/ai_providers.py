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
  four names, or the config's env key;
* (c) **the embeddings provider**: the ``embeddings.provider`` system setting,
  else ``config.EMBEDDING_PROVIDER``, the read ``EmbeddingManager`` makes
  (``disabled`` and ``huggingface_local`` send nothing out);
* (d) **the RAG reranker** (Cohere), which receives the user's query, when
  reranking is on and the reranker has a key;
* (e) **session-mode runtimes** (local edition with ``CLI_RUNTIME_ENABLED``):
  the CLI each session agent of this workspace runs on, named as the analytics
  page names its spend (Claude Code, Codex, GitHub Copilot);
* (f) **Retell**, when this workspace can use Auto Live voice: Auto Live sends
  the user's audio to Retell (Apple 5.1.2(i)).

Ids are the provider registry's slugs (``core/llm/providers.py``, stable),
names its labels. A key-only provider (Cohere has no chat client) is never a
chat route, so neither (a) nor (b) lists it; (c) and (d) can.

It reads key PRESENCE only. It selects the ``provider`` and ``name`` columns,
never a key column; nothing is decrypted, and no key value is returned, logged
or cached. The resolvers the calls themselves use decrypt and log a line per
call, so they are not called here. Each store is read with one query, whatever
the number of providers.

A key store that cannot be read raises. The settings behind (c), (d) and (f)
are read through their runtime readers, which answer their defaults when the
read fails: the list then says what the runtime itself does in that moment
(the embeddings manager falls back to the config, the reranker to no key, the
voice mint refuses).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from sqlalchemy import or_
from sqlalchemy.orm import Session

from config import config
from core.cli_presets import USAGE_PROVIDER_LABELS, usage_provider_slug
from core.cli_runtime import is_cli_agent, session_cli
from core.llm import embedding_manager, rerank_manager
from core.llm import providers as registry
from core.models.core import Agent, UserApiKey
from core.models.credentials import Credential
from core.models.workspaces import Workspace
from modules.voice import live_settings
from services import plan_tiers

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

# (c) Embeddings provider values that send nothing out: the deterministic
# fallback and the model on this machine (EmbeddingManager._load_provider).
EMBEDDINGS_STAY_HERE = frozenset({"disabled", "huggingface_local"})
# Embeddings provider values the registry knows under another slug.
EMBEDDINGS_REGISTRY_SLUGS = {"huggingface_api": "huggingface"}

# (d) The reranker's provider (core/llm/rerank_manager.py calls Cohere's API).
RERANK_PROVIDER_SLUG = "cohere"

# (f) Retell, the voice vendor Auto Live streams the user's audio to.
RETELL_PROVIDER_ID = "retell"
RETELL_PROVIDER_NAME = "Retell AI"
# The capability family a plan switches voice on with (PLAN_TIERS[*]["families"]),
# and the plan a workspace row without one is on (Workspace.plan's default).
VOICE_FAMILY = "voice"
DEFAULT_PLAN = "basic"

ProviderNames = Dict[str, str]
ProviderSource = Callable[[Session, uuid.UUID], ProviderNames]


def workspace_ai_providers(db: Session, workspace_id: Any) -> List[Dict[str, str]]:
    """The AI providers this workspace's data goes to, as a new list of ``{id, name}``.

    Sorted by id, each id once. Raises when a key store cannot be read: the caller
    decides what an unknown list means (GET /api/workspaces/current answers ``null``).
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


# ---------------------------------------------------------------------------
# (c) the embeddings provider, (d) the reranker
# ---------------------------------------------------------------------------


def _embeddings(_db: Session, _workspace_id: uuid.UUID) -> ProviderNames:
    """(c) The embeddings provider, read as ``EmbeddingManager._load_provider`` reads it.

    The ``embeddings.provider`` system setting, else ``config.EMBEDDING_PROVIDER``.
    Unset, ``disabled`` and ``huggingface_local`` add none; a slug the registry
    knows takes its label, so a provider that is also a chat route appears once.
    """
    provider = embedding_manager.get_system_setting("provider") or config.EMBEDDING_PROVIDER
    if not provider or provider in EMBEDDINGS_STAY_HERE:
        return {}
    return _registry_names([EMBEDDINGS_REGISTRY_SLUGS.get(provider, provider)])


def _reranker(_db: Session, _workspace_id: uuid.UUID) -> ProviderNames:
    """(d) Cohere, when RAG reranking is on and the reranker has a key (presence: ``is_available``)."""
    if not (config.RAG_RERANK_ENABLED and rerank_manager.get_rerank_manager().is_available()):
        return {}
    return _registry_names([RERANK_PROVIDER_SLUG])


# ---------------------------------------------------------------------------
# (e) session-mode runtimes
# ---------------------------------------------------------------------------


def _session_runtimes(db: Session, workspace_id: uuid.UUID) -> ProviderNames:
    """(e) Local edition with session mode on: the CLI every session agent of this workspace runs on.

    Id the CLI's usage slug, name its label (``core/cli_presets``). Every agent of
    the workspace counts, an inactive one too: the CLI host's claim runs it. A
    CLI with no preset adds none: no host runs it, and its text is never shown.
    Nothing on saas, where session mode cannot be switched on.
    """
    if not (config.IS_LOCAL_EDITION and config.CLI_RUNTIME_ENABLED):
        return {}
    rows = db.query(Agent.configuration).filter(Agent.workspace_id == workspace_id).all()
    slugs = {
        usage_provider_slug(session_cli(row.configuration))
        for row in rows
        if is_cli_agent(row.configuration)
    }
    return {slug: USAGE_PROVIDER_LABELS[slug] for slug in sorted(slugs) if slug in USAGE_PROVIDER_LABELS}


# ---------------------------------------------------------------------------
# (f) Retell, for Auto Live voice
# ---------------------------------------------------------------------------


def _retell(db: Session, workspace_id: uuid.UUID) -> ProviderNames:
    """(f) Retell, when this workspace can use Auto Live voice.

    The mint route's own gates, read through the same helpers, minus the monthly
    cap (a capped workspace is still a Retell user this month): the workspace's
    switch, the plan (saas only), the platform switch, and Retell armed. The
    credentials are read last, and only ``armed`` is kept: no key leaves here.
    """
    workspace = db.get(Workspace, workspace_id)
    if workspace is None or not live_settings.parse_workspace_voice_live(workspace.settings).enabled:
        return {}
    if not _plan_allows_voice(workspace.plan):
        return {}
    if not (live_settings.voice_live_enabled() and live_settings.retell_credentials().armed):
        return {}
    return {RETELL_PROVIDER_ID: RETELL_PROVIDER_NAME}


def _plan_allows_voice(plan: Optional[str]) -> bool:
    """D34: on saas the plan's families switch voice on, or it is a known tier that declares none.

    A tier restricts only by declaring families (``enterprise`` declares none);
    an unknown plan resolves to the entry tier, which does. A local install is
    never plan-gated.
    """
    if not config.IS_SAAS_EDITION:
        return True
    plan_name = plan or DEFAULT_PLAN
    if bool(plan_tiers.exposure_for_plan(plan_name)["families"].get(VOICE_FAMILY)):
        return True
    tier = plan_tiers.get_tier(plan_name)
    return isinstance(tier, dict) and not isinstance(tier.get("families"), dict)


_SOURCES: Tuple[ProviderSource, ...] = (
    _own_keys,
    _platform_routes,
    _embeddings,
    _reranker,
    _session_runtimes,
    _retell,
)

__all__ = ["workspace_ai_providers"]
