"""
LLM Usage Tracker (PRD-54 · PRD-236 W1 · analytics cost tracking 2026-09-09)
============================================================================

Records every LLM, embedding, rerank and Claude Code session call in
``llm_usage`` — metered, free and subscription alike — so the Analytics page
can say what each lane, agent, model and PROVIDER cost, and what cost nothing.

Rules that used to be violated:

* a call is never silently booked at $0 on a metered route — the price comes
  from the route row, then any row for the id, then the OpenRouter catalogue
  cache, then the static estimate map; a free route (NVIDIA) books $0 by the
  registry multiplier, a subscription session by its billing;
* a provider-reported cost (OpenRouter returns the exact credits charged)
  beats every estimate;
* ``tier`` is the registry KIND of the serving provider (``direct`` /
  ``aggregator`` / ``hosted_open``) or ``subscription`` — one vocabulary;
* no workspace context does not mean no row: the request ContextVar and, in
  the local edition, the single default workspace stand in for it.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
from uuid import UUID

logger = logging.getLogger(__name__)

TIER_SUBSCRIPTION = "subscription"
TIER_UNKNOWN = "unknown"
LANE_SESSION = "session"
STATUS_SUCCESS = "success"
STATUS_ERROR = "error"
STATUS_CANCELLED = "cancelled"

# Vendor pricing facts for prompt-cache tokens, as a multiple of the input
# price: Anthropic bills cache reads at 10% and cache writes at 125% of input;
# OpenAI bills cached prompt tokens at 50%. A provider not listed keeps the
# full input price (conservative — never under-reports).
CACHE_READ_PRICE_MULTIPLIER: Dict[str, float] = {"anthropic": 0.10, "bedrock": 0.10, "openai": 0.50, "azure": 0.50}
CACHE_WRITE_PRICE_MULTIPLIER: Dict[str, float] = {"anthropic": 1.25, "bedrock": 1.25}


def resolve_workspace_id(workspace_id: Any = None) -> Optional[Any]:
    """Explicit → the request's ContextVar → the local edition's one workspace."""
    if workspace_id:
        return workspace_id
    try:
        from core.monitoring.automatos_logging import workspace_id_var

        scoped = workspace_id_var.get() or None
        if scoped:
            return scoped
    except Exception:
        pass
    try:
        from config import config

        if (config.AUTH_EDITION or "").lower() == "local" and config.DEFAULT_WORKSPACE_ID:
            return config.DEFAULT_WORKSPACE_ID
    except Exception:
        pass
    return None


def _as_uuid(value: Any) -> Optional[UUID]:
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (TypeError, ValueError):
        return None


def _rate_from_static_map(model_id: str) -> Optional[Tuple[float, float]]:
    """The manager's estimate map, only when a key actually matches the model
    (its flat default would price an embedding like a chat model)."""
    try:
        from core.llm.manager import MODEL_COST_MAP
    except Exception:
        return None
    lower = (model_id or "").lower()
    for key, (inp, out) in MODEL_COST_MAP.items():
        if key in lower:
            return float(inp), float(out)
    return None


def _rate_from_openrouter_cache(db, model_id: str) -> Optional[Tuple[float, float]]:
    """The OpenRouter catalogue cache prices per TOKEN; per 1k here."""
    try:
        from core.models.openrouter_cache import OpenRouterModelCache

        row = db.query(OpenRouterModelCache).filter(OpenRouterModelCache.model_id == model_id).first()
    except Exception:
        return None
    if row is None:
        return None
    return float(row.prompt_cost or 0) * 1000.0, float(row.completion_cost or 0) * 1000.0


def resolve_price(db, model_id: str, provider: Optional[str]) -> Dict[str, Any]:
    """``{input_per_1k, output_per_1k, tier, source, multiplier}`` for the route
    that served ``model_id``. ``source`` names where the price came from so a
    test (or a curious operator) can tell an estimate from a catalogue price."""
    from core.llm.providers import get_spec, normalize_slug, price_multiplier_for
    from core.models.core import LLMModel

    route = normalize_slug(provider)
    spec = get_spec(route) if route else None
    kind = spec.kind if spec else TIER_UNKNOWN
    multiplier = price_multiplier_for(provider)

    row = None
    if route:
        row = (
            db.query(LLMModel)
            .filter(LLMModel.model_id == model_id, LLMModel.serving_provider == route)
            .first()
        )
    if row is not None:
        return {
            "input_per_1k": float(row.input_cost_per_1k_tokens or 0),
            "output_per_1k": float(row.output_cost_per_1k_tokens or 0),
            "tier": row.sourcing or kind,
            "source": "route",
            "multiplier": 1.0,
        }
    any_row = db.query(LLMModel).filter(LLMModel.model_id == model_id).first()
    if any_row is not None:
        return {
            "input_per_1k": float(any_row.input_cost_per_1k_tokens or 0),
            "output_per_1k": float(any_row.output_cost_per_1k_tokens or 0),
            "tier": kind,
            "source": "model",
            "multiplier": multiplier,
        }
    cached = _rate_from_openrouter_cache(db, model_id)
    if cached is not None:
        return {"input_per_1k": cached[0], "output_per_1k": cached[1], "tier": kind, "source": "catalogue", "multiplier": multiplier}
    static = _rate_from_static_map(model_id)
    if static is not None:
        return {"input_per_1k": static[0], "output_per_1k": static[1], "tier": kind, "source": "estimate", "multiplier": multiplier}
    return {"input_per_1k": 0.0, "output_per_1k": 0.0, "tier": kind, "source": "none", "multiplier": multiplier}


def price_call(
    price: Dict[str, Any],
    *,
    provider: Optional[str],
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> Tuple[float, float]:
    """(input_cost, output_cost) in USD. ``input_tokens`` is the full prompt; the
    cached and written parts are re-priced at the vendor's cache multipliers."""
    from core.llm.providers import normalize_slug

    slug = normalize_slug(provider) or ""
    read_mult = CACHE_READ_PRICE_MULTIPLIER.get(slug, 1.0)
    write_mult = CACHE_WRITE_PRICE_MULTIPLIER.get(slug, 1.0)
    cache_read = max(0, min(int(cache_read_tokens or 0), int(input_tokens or 0)))
    cache_write = max(0, min(int(cache_write_tokens or 0), int(input_tokens or 0) - cache_read))
    fresh = max(0, int(input_tokens or 0) - cache_read - cache_write)
    per_1k_in = float(price.get("input_per_1k") or 0) * float(price.get("multiplier", 1.0))
    per_1k_out = float(price.get("output_per_1k") or 0) * float(price.get("multiplier", 1.0))
    input_cost = (fresh + cache_read * read_mult + cache_write * write_mult) / 1000.0 * per_1k_in
    output_cost = int(output_tokens or 0) / 1000.0 * per_1k_out
    return input_cost, output_cost


def bump_agent_usage_stats(db, agent_id: Optional[int], *, total_tokens: int, total_cost: float) -> None:
    """``Agent.model_usage_stats`` — the cumulative counters the agent cards read."""
    if not agent_id:
        return
    try:
        from core.models.core import Agent
        from sqlalchemy.orm.attributes import flag_modified

        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            return
        stats = dict(agent.model_usage_stats or {})
        stats["total_tokens"] = int(stats.get("total_tokens", 0) or 0) + int(total_tokens)
        stats["total_cost"] = round(float(stats.get("total_cost", 0.0) or 0.0) + float(total_cost), 6)
        stats["total_requests"] = int(stats.get("total_requests", 0) or 0) + 1
        stats["avg_tokens_per_request"] = (
            int(stats["total_tokens"] / stats["total_requests"]) if stats["total_requests"] > 0 else 0
        )
        stats["last_used_at"] = datetime.utcnow().isoformat()
        agent.model_usage_stats = stats
        flag_modified(agent, "model_usage_stats")
    except Exception as agent_err:  # noqa: BLE001 — stats are a cache, never the record
        logger.debug(f"Agent stats update skipped: {agent_err}")


class UsageTracker:
    """Tracks usage per request for analytics and billing."""

    @staticmethod
    def track(
        workspace_id: Any,
        model_id: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        agent_id: Optional[int] = None,
        execution_id: Optional[str] = None,
        request_type: str = "chat",
        latency_ms: Optional[int] = None,
        status: str = STATUS_SUCCESS,
        is_byok: bool = False,
        error_message: Optional[str] = None,
        tier: Optional[str] = None,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        reported_cost: Optional[float] = None,
        cost_override: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Record one call. Runs in its own DB session so a failure here never
        touches the caller's transaction, and never raises.

        ``reported_cost`` is the provider's own figure for the call (OpenRouter's
        ``usage.cost``) and wins over any estimate; ``cost_override`` is an
        explicit ``(input_cost, output_cost)`` for calls priced elsewhere
        (a subscription session books ``(0, 0)``; a rerank prices its search
        units). ``tier`` given explicitly wins over the route's kind.
        """
        try:
            ws = _as_uuid(resolve_workspace_id(workspace_id))
            if ws is None:
                logger.debug("usage not recorded: no workspace for %s/%s (%s)", provider, model_id, request_type)
                return
            from core.database.database import SessionLocal
            from core.models.core import LLMUsage

            db = SessionLocal()
            try:
                model_id = model_id or "unknown"
                input_tokens = int(input_tokens or 0)
                output_tokens = int(output_tokens or 0)
                cache_read_tokens = int(cache_read_tokens or 0)
                cache_write_tokens = int(cache_write_tokens or 0)

                if cost_override is not None:
                    input_cost, output_cost = float(cost_override[0]), float(cost_override[1])
                    row_tier = tier or TIER_UNKNOWN
                else:
                    price = resolve_price(db, model_id, provider)
                    row_tier = tier or price["tier"] or TIER_UNKNOWN
                    input_cost, output_cost = price_call(
                        price,
                        provider=provider,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cache_read_tokens=cache_read_tokens,
                        cache_write_tokens=cache_write_tokens,
                    )
                    if reported_cost is not None and float(reported_cost) > 0:
                        # The provider's own number for THIS call: keep the
                        # input/output split proportional, make the total exact.
                        estimated = input_cost + output_cost
                        share = (input_cost / estimated) if estimated > 0 else 0.5
                        input_cost = float(reported_cost) * share
                        output_cost = float(reported_cost) - input_cost
                    elif price["source"] == "none" and (input_tokens or output_tokens) and price["multiplier"] > 0:
                        logger.info(
                            "usage priced at $0: no price known for %s via %s (request_type=%s)",
                            model_id, provider, request_type,
                        )

                total_cost = input_cost + output_cost
                total_tokens = input_tokens + output_tokens

                row = LLMUsage(
                    workspace_id=ws,
                    model_id=model_id,
                    provider=provider or "unknown",
                    tier=row_tier,
                    agent_id=agent_id,
                    execution_id=execution_id,
                    request_type=request_type,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    cache_read_tokens=cache_read_tokens,
                    cache_write_tokens=cache_write_tokens,
                    input_cost=round(input_cost, 8),
                    output_cost=round(output_cost, 8),
                    total_cost=round(total_cost, 8),
                    is_byok=is_byok,
                    latency_ms=latency_ms,
                    status=status,
                    error_message=error_message,
                )
                db.add(row)
                if agent_id and status == STATUS_SUCCESS:
                    bump_agent_usage_stats(db, agent_id, total_tokens=total_tokens, total_cost=total_cost)
                db.commit()
            finally:
                db.close()
        except Exception as e:  # noqa: BLE001 — usage tracking never breaks the main flow
            logger.warning(f"Failed to track LLM usage: {e}")

    @staticmethod
    def track_embedding(
        *,
        provider: str,
        model_id: str,
        prompt_tokens: int,
        latency_ms: Optional[int] = None,
        status: str = STATUS_SUCCESS,
        error_message: Optional[str] = None,
        inputs: int = 1,
    ) -> None:
        """One embeddings request (a batch of ``inputs`` texts is one call).

        Workspace comes from the request ContextVar / the local default — the
        embedding singleton has no workspace of its own; the lane and execution
        come from the enclosing ``usage_scope`` (a chat turn's tool-routing
        embed books to that turn) or default to ``embedding``.
        """
        from core.llm.usage_context import LANE_EMBEDDING, current_usage_scope

        scope = current_usage_scope()
        UsageTracker.track(
            workspace_id=scope.get("workspace_id"),
            model_id=model_id,
            provider=provider,
            input_tokens=int(prompt_tokens or 0),
            output_tokens=0,
            agent_id=scope.get("agent_id"),
            execution_id=scope.get("execution_id"),
            request_type=LANE_EMBEDDING,
            latency_ms=latency_ms,
            status=status,
            error_message=error_message,
        )

    @staticmethod
    def track_rerank(
        *,
        provider: str,
        model_id: str,
        search_units: int,
        usd_per_1k_units: float,
        latency_ms: Optional[int] = None,
        status: str = STATUS_SUCCESS,
        error_message: Optional[str] = None,
    ) -> None:
        """One rerank request. Cohere bills SEARCH UNITS (one query over up to
        100 documents), not tokens: the units are recorded as ``input_tokens``
        and priced at the configured rate — never a token price."""
        from core.llm.usage_context import LANE_RERANK, current_usage_scope

        scope = current_usage_scope()
        units = int(search_units or 0)
        UsageTracker.track(
            workspace_id=scope.get("workspace_id"),
            model_id=model_id,
            provider=provider,
            input_tokens=units,
            output_tokens=0,
            agent_id=scope.get("agent_id"),
            execution_id=scope.get("execution_id"),
            request_type=LANE_RERANK,
            latency_ms=latency_ms,
            status=status,
            error_message=error_message,
            tier="direct",
            cost_override=(units / 1000.0 * float(usd_per_1k_units or 0), 0.0),
        )

    @staticmethod
    def track_session(
        workspace_id: Any,
        *,
        cli_provider: str,
        usage: Optional[Dict[str, Any]],
        agent_id: Optional[int],
        execution_id: str,
        request_type: str = LANE_SESSION,
        status: str = STATUS_SUCCESS,
        latency_ms: Optional[int] = None,
        error_message: Optional[str] = None,
        fallback_model: Optional[str] = None,
    ) -> int:
        """Book a Claude Code (or Codex) session on the user's own subscription.

        The host reports the transcript's token totals per model (input, output,
        cache read, cache write — Anthropic's accounting, where ``input_tokens``
        EXCLUDES the cached parts). One row per model, priced at $0 with
        ``tier='subscription'``: the plan pays, and there is no dollar figure to
        invent. Returns the number of rows written.
        """
        from core.cli_runtime import usage_provider_slug

        provider = usage_provider_slug(cli_provider)
        usage = usage if isinstance(usage, dict) else {}
        per_model = usage.get("per_model") if isinstance(usage.get("per_model"), dict) else {}
        buckets: Dict[str, Dict[str, Any]] = {}
        for model, counts in per_model.items():
            if isinstance(counts, dict):
                buckets[str(model)] = counts
        if not buckets:
            buckets[str(usage.get("model") or fallback_model or "default")] = usage

        written = 0
        for model, counts in buckets.items():
            fresh = int(counts.get("input_tokens") or 0)
            cache_read = int(counts.get("cache_read_input_tokens") or counts.get("cache_read_tokens") or 0)
            cache_write = int(counts.get("cache_creation_input_tokens") or counts.get("cache_write_tokens") or 0)
            output = int(counts.get("output_tokens") or 0)
            UsageTracker.track(
                workspace_id=workspace_id,
                model_id=model,
                provider=provider,
                input_tokens=fresh + cache_read + cache_write,
                output_tokens=output,
                agent_id=agent_id,
                execution_id=execution_id,
                request_type=request_type,
                latency_ms=latency_ms,
                status=status,
                is_byok=True,
                error_message=error_message,
                tier=TIER_SUBSCRIPTION,
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
                cost_override=(0.0, 0.0),
            )
            written += 1
        return written
