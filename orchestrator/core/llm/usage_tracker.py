"""
LLM Usage Tracker (PRD-54)
===========================

Records every LLM call for analytics and billing.
Calculates cost from model registry and inserts into llm_usage table.
"""

import logging
from typing import Optional
from uuid import UUID
from datetime import datetime

logger = logging.getLogger(__name__)


class UsageTracker:
    """Tracks LLM usage per-request for analytics and billing."""

    @staticmethod
    def track(
        workspace_id: UUID,
        model_id: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        agent_id: Optional[int] = None,
        execution_id: Optional[str] = None,
        request_type: str = "chat",
        latency_ms: Optional[int] = None,
        status: str = "success",
        is_byok: bool = False,
        error_message: Optional[str] = None,
        tier: str = "direct",
    ) -> None:
        """
        Record a single LLM usage event.

        Runs in a separate DB session to avoid interfering with
        the caller's transaction.
        """
        try:
            from core.database.database import SessionLocal
            from core.models.core import LLMUsage, LLMModel

            db = SessionLocal()
            try:
                # PRD-236 W1: price the ROUTE that served the call — the row for
                # (serving_provider, model_id). A vendor id may exist once per
                # provider with different prices (Kimi K3: OpenRouter $3/M, NVIDIA 0).
                from core.llm.providers import normalize_slug, price_multiplier_for
                route = normalize_slug(provider)
                model_row = None
                if route:
                    model_row = (
                        db.query(LLMModel)
                        .filter(LLMModel.model_id == model_id, LLMModel.serving_provider == route)
                        .first()
                    )
                if model_row is None:
                    # No route row yet (catalogue not synced for this provider):
                    # fall back to any row for the id, corrected by the registry's
                    # multiplier so a free route still books zero.
                    model_row = db.query(LLMModel).filter(LLMModel.model_id == model_id).first()
                    multiplier = price_multiplier_for(provider)
                else:
                    multiplier = 1.0
                if model_row:
                    input_cost = (input_tokens / 1000) * float(model_row.input_cost_per_1k_tokens or 0) * multiplier
                    output_cost = (output_tokens / 1000) * float(model_row.output_cost_per_1k_tokens or 0) * multiplier
                    tier = model_row.sourcing or tier
                else:
                    input_cost = 0.0
                    output_cost = 0.0

                total_cost = input_cost + output_cost

                total_tokens = input_tokens + output_tokens

                row = LLMUsage(
                    workspace_id=workspace_id,
                    model_id=model_id,
                    provider=provider,
                    tier=tier,
                    agent_id=agent_id,
                    execution_id=execution_id,
                    request_type=request_type,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    input_cost=round(input_cost, 8),
                    output_cost=round(output_cost, 8),
                    total_cost=round(total_cost, 8),
                    is_byok=is_byok,
                    latency_ms=latency_ms,
                    status=status,
                    error_message=error_message,
                )
                db.add(row)

                # Also update Agent.model_usage_stats so the agents API
                # returns cumulative token/cost data without a separate query.
                if agent_id and status == "success":
                    try:
                        from core.models.core import Agent
                        from sqlalchemy.orm.attributes import flag_modified

                        agent = db.query(Agent).filter(Agent.id == agent_id).first()
                        if agent:
                            stats = agent.model_usage_stats or {
                                "total_tokens": 0, "total_cost": 0.0,
                                "total_requests": 0, "avg_tokens_per_request": 0,
                                "last_used_at": None,
                            }
                            stats["total_tokens"] = stats.get("total_tokens", 0) + total_tokens
                            stats["total_cost"] = round(stats.get("total_cost", 0.0) + total_cost, 6)
                            stats["total_requests"] = stats.get("total_requests", 0) + 1
                            stats["avg_tokens_per_request"] = int(
                                stats["total_tokens"] / stats["total_requests"]
                            ) if stats["total_requests"] > 0 else 0
                            stats["last_used_at"] = datetime.utcnow().isoformat()
                            agent.model_usage_stats = stats
                            flag_modified(agent, "model_usage_stats")
                    except Exception as agent_err:
                        logger.debug(f"Agent stats update skipped: {agent_err}")

                db.commit()
            finally:
                db.close()

        except Exception as e:
            # Usage tracking should never break the main flow
            logger.warning(f"Failed to track LLM usage: {e}")
