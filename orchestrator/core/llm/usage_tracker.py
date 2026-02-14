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
                # Look up cost from model registry
                model_row = db.query(LLMModel).filter(LLMModel.model_id == model_id).first()
                if model_row:
                    input_cost = (input_tokens / 1000) * float(model_row.input_cost_per_1k_tokens or 0)
                    output_cost = (output_tokens / 1000) * float(model_row.output_cost_per_1k_tokens or 0)
                    tier = model_row.tier or tier
                else:
                    input_cost = 0.0
                    output_cost = 0.0

                total_cost = input_cost + output_cost

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
                    total_tokens=input_tokens + output_tokens,
                    input_cost=round(input_cost, 8),
                    output_cost=round(output_cost, 8),
                    total_cost=round(total_cost, 8),
                    is_byok=is_byok,
                    latency_ms=latency_ms,
                    status=status,
                    error_message=error_message,
                )
                db.add(row)
                db.commit()
            finally:
                db.close()

        except Exception as e:
            # Usage tracking should never break the main flow
            logger.warning(f"Failed to track LLM usage: {e}")
