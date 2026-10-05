"""
Anthropic catalogue sync (#829)
===============================

Issue #829: ``POST /api/marketplace/llm/sync/anthropic`` answered "'anthropic' has
no catalogue sync. Syncable: openrouter, nvidia". A workspace with its own
Anthropic key could not refresh the direct Anthropic models, and one local install
still offered claude-sonnet-4-5-20250929 as its newest: the row seeded in February.

Anthropic is now syncable. The sync reads Anthropic's Models API
(``GET https://api.anthropic.com/v1/models``), every page: while ``has_more``,
the next request carries ``after_id`` = the previous page's ``last_id``. It uses
the key that pays for the workspace's Anthropic calls
(``core.llm.key_resolver.resolve_provider_key``, the agent factory's resolver:
the workspace's BYOK key, then the platform's), and upserts one route per listed
id, served by Anthropic:

- the name, the context window (``max_input_tokens``), the output cap
  (``max_tokens``) and vision support (``capabilities.image_input``) come from
  the API and refresh existing rows; a field the API leaves null keeps the row's
  value;
- the API publishes no prices. An existing row keeps its price, description,
  tags and flags. A new row borrows price, description and tool support from the
  OpenRouter cache row for the same model (``anthropic/claude-opus-4.8`` for
  ``claude-opus-4-8``, ``core.llm.anthropic_ids``), as the NVIDIA sync borrows its metadata; with no such row
  it starts unpriced;
- ids Anthropic no longer lists are marked ``deprecated`` (installs keep their
  row), as the OpenRouter and NVIDIA syncs do. An empty answer retires nothing.
  An alias Anthropic still answers but does not list is kept: the API lists
  ``claude-sonnet-4-5-20250929``, not ``claude-sonnet-4-5`` or
  ``claude-3-5-sonnet-latest`` (``is_alias_of_listed``), and a deprecated route
  stops routing every agent on it (``model_refusals._route_is_retired``).

The key is sent in the ``x-api-key`` header only; it is never logged or put in a
message. Failures raise ``AnthropicCatalogError`` with a message for the owner,
which the sync endpoint returns as it does for the other syncs.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from sqlalchemy.orm import Session

from core.llm.anthropic_ids import DATE_SUFFIX, openrouter_twin_ids
from core.models.core import LLMModel
from core.models.openrouter_cache import OpenRouterModelCache, OpenRouterSyncJob

logger = logging.getLogger(__name__)

PROVIDER = "anthropic"
JOB_TYPE = "anthropic_sync"
ANTHROPIC_MODELS_URL = "https://api.anthropic.com/v1/models"
# The version header the credential tester sends to the same API.
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_PAGE_LIMIT = 1000  # the API's maximum page size
ANTHROPIC_MAX_PAGES = 50  # a cursor that never ends is an error, not a hang
ANTHROPIC_FETCH_TIMEOUT_S = 30
SOURCING_DIRECT = "direct"
MODEL_FAMILY = "claude"
ANTHROPIC_TAG = "anthropic"

NO_KEY_MESSAGE = (
    "No Anthropic API key is available to this workspace. Add your Anthropic key "
    "in Settings > API Keys and switch it on, then sync again."
)
REJECTED_KEY_MESSAGE = "Anthropic refused the API key (HTTP {status}). Check the Anthropic key in Settings > API Keys."
HTTP_ERROR_MESSAGE = "Anthropic's Models API answered HTTP {status}."
UNREACHABLE_MESSAGE = "Could not reach Anthropic's Models API ({error})."
ENDLESS_PAGES_MESSAGE = "Anthropic's Models API kept paging past {pages} pages; nothing was synced."
NEW_ROW_DESCRIPTION = "{name}, served directly by Anthropic with your Anthropic key."

_LATEST_SUFFIX = "-latest"
_REJECTED_STATUSES = (401, 403)


class AnthropicCatalogError(RuntimeError):
    """An Anthropic sync failure, worded for the owner (no key, refused key, no answer)."""


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def run_anthropic_sync(catalog: Any, workspace_id: Any = None) -> Dict[str, Any]:
    """Sync Anthropic's model list into ``llm_models`` and record the job.

    ``catalog`` is the ``ProviderCatalogSync`` whose session and route upsert
    are used. Raises ``AnthropicCatalogError`` (recorded as a failed job).
    """
    db: Session = catalog.db
    started = datetime.utcnow()
    job = OpenRouterSyncJob(job_type=JOB_TYPE, status="running", started_at=started)
    db.add(job)
    db.commit()
    try:
        result = _sync_listed(catalog, workspace_key(db, workspace_id))
    except Exception as exc:
        db.rollback()
        _record_failure(db, job, exc)
        logger.exception("[CatalogSync] Anthropic sync failed")
        raise
    _record_success(db, job, started, result)
    logger.info(
        "[CatalogSync] Anthropic: %d routes from %d listed (%d deprecated)",
        result["models_synced"], result["listed"], result["deprecated"],
    )
    return result


def workspace_key(db: Session, workspace_id: Any) -> str:
    """The key that pays for this workspace's Anthropic calls, or ``AnthropicCatalogError``."""
    from core.llm.key_resolver import resolve_provider_key

    resolved = resolve_provider_key(db, PROVIDER, workspace_id=workspace_id, agent_name="anthropic catalogue sync")
    if resolved is None or not resolved.api_key:
        raise AnthropicCatalogError(NO_KEY_MESSAGE)
    return resolved.api_key


def _sync_listed(catalog: Any, api_key: str) -> Dict[str, Any]:
    models = fetch_anthropic_models(api_key)
    kept: List[str] = []
    for model in models:
        model_id = model["id"]
        catalog._upsert_route(
            PROVIDER, model_id, synced_values(model),
            insert_only=new_row_defaults(model, openrouter_twin(catalog.db, model_id)),
        )
        kept.append(model_id)
    deprecated = _deprecate_unlisted(catalog.db, kept)
    return {
        "provider": PROVIDER, "status": "completed", "models_synced": len(kept),
        "listed": len(models), "deprecated": deprecated,
    }


def _deprecate_unlisted(db: Session, kept: List[str]) -> int:
    """Active Anthropic routes the API no longer lists stop being offered,
    unless the route is an alias of a listed id (Anthropic still answers it)."""
    if not kept:
        return 0
    unlisted = (
        db.query(LLMModel.model_id)
        .filter(
            LLMModel.serving_provider == PROVIDER,
            LLMModel.status == "active",
            ~LLMModel.model_id.in_(kept),
        )
        .all()
    )
    retired = [model_id for (model_id,) in unlisted if not is_alias_of_listed(model_id, kept)]
    if not retired:
        return 0
    deprecated = (
        db.query(LLMModel)
        .filter(LLMModel.serving_provider == PROVIDER, LLMModel.model_id.in_(retired))
        .update({"status": "deprecated"}, synchronize_session=False)
    )
    return int(deprecated or 0)


def is_alias_of_listed(model_id: str, listed: List[str]) -> bool:
    """True when ``model_id`` is an undated alias of an id the Models API listed.

    ``claude-sonnet-4-5`` is an alias of ``claude-sonnet-4-5-20250929`` (the id
    plus ``-`` and 8 digits); ``claude-3-5-sonnet-latest`` is an alias of any
    listed ``claude-3-5-sonnet-…``.
    """
    if model_id.endswith(_LATEST_SUFFIX):
        stem = model_id[: -len(_LATEST_SUFFIX)] + "-"
        return any(listed_id.startswith(stem) for listed_id in listed)
    return any(
        listed_id.startswith(model_id + "-") and DATE_SUFFIX.fullmatch(listed_id[len(model_id):])
        for listed_id in listed
    )


def _record_success(db: Session, job: OpenRouterSyncJob, started: datetime, result: Dict[str, Any]) -> None:
    finished = datetime.utcnow()
    job.status = "completed"
    job.models_synced = result["models_synced"]
    job.models_updated = result["models_synced"]
    job.completed_at = finished
    job.duration_ms = int((finished - started).total_seconds() * 1000)
    job.job_metadata = {"listed": result["listed"], "deprecated": result["deprecated"]}
    db.commit()


def _record_failure(db: Session, job: OpenRouterSyncJob, exc: Exception) -> None:
    job.status = "failed"
    job.errors_count = 1
    job.error_details = {"error": str(exc)[:500]}
    job.completed_at = datetime.utcnow()
    db.add(job)
    db.commit()


# --------------------------------------------------------------------------- #
# Models API
# --------------------------------------------------------------------------- #


def fetch_anthropic_models(api_key: str) -> List[Dict[str, Any]]:
    """Every model the key can see, across all pages, in the API's order."""
    headers = {"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION}
    models: List[Dict[str, Any]] = []
    after_id: Optional[str] = None
    with httpx.Client(timeout=ANTHROPIC_FETCH_TIMEOUT_S) as client:
        for _ in range(ANTHROPIC_MAX_PAGES):
            page = _fetch_page(client, headers, after_id)
            models.extend(m for m in page.get("data") or [] if isinstance(m, dict) and m.get("id"))
            after_id = page.get("last_id")
            if not page.get("has_more") or not after_id:
                return models
    raise AnthropicCatalogError(ENDLESS_PAGES_MESSAGE.format(pages=ANTHROPIC_MAX_PAGES))


def _fetch_page(client: httpx.Client, headers: Dict[str, str], after_id: Optional[str]) -> Dict[str, Any]:
    params: Dict[str, Any] = {"limit": ANTHROPIC_PAGE_LIMIT}
    if after_id:
        params["after_id"] = after_id
    try:
        resp = client.get(ANTHROPIC_MODELS_URL, headers=headers, params=params)
    except httpx.HTTPError as exc:
        raise AnthropicCatalogError(UNREACHABLE_MESSAGE.format(error=type(exc).__name__)) from exc
    if resp.status_code in _REJECTED_STATUSES:
        raise AnthropicCatalogError(REJECTED_KEY_MESSAGE.format(status=resp.status_code))
    if resp.status_code >= 400:
        raise AnthropicCatalogError(HTTP_ERROR_MESSAGE.format(status=resp.status_code))
    body = resp.json()
    return body if isinstance(body, dict) else {}


# --------------------------------------------------------------------------- #
# Row values
# --------------------------------------------------------------------------- #


def synced_values(model: Dict[str, Any]) -> Dict[str, Any]:
    """What the API says about a model: written on insert AND on every re-sync.

    A null ``max_input_tokens`` / ``max_tokens`` / ``capabilities`` leaves the
    existing row's value alone (the key is simply absent from the update).
    """
    values: Dict[str, Any] = dict(
        provider=PROVIDER,
        display_name=model.get("display_name") or model["id"],
        status="active",
        sourcing=SOURCING_DIRECT,
        external_id=model["id"],
    )
    context = _positive_int(model.get("max_input_tokens"))
    if context:
        values["context_window"] = context
    output = _positive_int(model.get("max_tokens"))
    if output:
        values["max_output_tokens"] = output
    vision = _supported(model.get("capabilities"), "image_input")
    if vision is not None:
        values["supports_vision"] = vision
    return values


def new_row_defaults(model: Dict[str, Any], twin: Optional[OpenRouterModelCache]) -> Dict[str, Any]:
    """Fields only a NEW row gets; an existing row keeps its own (price, description, tags…)."""
    name = model.get("display_name") or model["id"]
    defaults: Dict[str, Any] = dict(
        description=NEW_ROW_DESCRIPTION.format(name=name),
        model_family=MODEL_FAMILY,
        context_window=0,
        max_output_tokens=0,
        input_cost_per_1k_tokens=None,
        output_cost_per_1k_tokens=None,
        # Every model the Messages API serves takes tools; the Models API has no flag for it.
        supports_functions=True,
        supports_vision=False,
        supports_streaming=True,
        category=None,
        tags=[ANTHROPIC_TAG],
        capabilities={},
        recommended_for=[],
    )
    if twin is None:
        return defaults
    return dict(
        defaults,
        description=twin.description or defaults["description"],
        context_window=int(twin.context_length or 0),
        max_output_tokens=int(twin.max_completion_tokens or 0),
        input_cost_per_1k_tokens=float(twin.prompt_cost or 0) * 1000,
        output_cost_per_1k_tokens=float(twin.completion_cost or 0) * 1000,
        supports_functions=bool(twin.supports_tools),
        supports_vision=bool(twin.supports_vision),
        category=twin.category,
        tags=sorted(set(list(twin.tags or []) + [ANTHROPIC_TAG])),
        pricing_updated_at=datetime.utcnow(),
    )


def openrouter_twin(db: Session, model_id: str) -> Optional[OpenRouterModelCache]:
    """The OpenRouter cache row for the same model (``openrouter_twin_ids``), if the cache has one."""
    q = db.query(OpenRouterModelCache)
    for candidate in openrouter_twin_ids(model_id):
        row = q.filter(OpenRouterModelCache.model_id == candidate).first()
        if row is not None:
            return row
    return None


def _positive_int(value: Any) -> Optional[int]:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _supported(capabilities: Any, name: str) -> Optional[bool]:
    if not isinstance(capabilities, dict):
        return None
    entry = capabilities.get(name)
    if not isinstance(entry, dict) or "supported" not in entry:
        return None
    return bool(entry["supported"])


__all__ = [
    "AnthropicCatalogError", "fetch_anthropic_models", "is_alias_of_listed", "new_row_defaults", "openrouter_twin",
    "run_anthropic_sync", "synced_values", "workspace_key",
]
