"""Policy plane — model-aware pricing from the DB registry (PRD-174 F059).

One source of truth for "what does a call on model X cost". Reads the DB price
registry (``llm_models`` via :class:`core.llm.model_registry.ModelRegistry`) —
the model-aware table that already exists but sat unused while four hardcoded
price maps (``manager._MODEL_COST_MAP``, ``logging_utils.TOKEN_COSTS``,
``model_usage_tracker`` fallbacks, seed data) costed the same call four
different, model-blind ways. The budget admission gate (F086) prices against
this so it approves/blocks on the *right* dollars.

Kept dependency-light: SQLAlchemy is imported lazily inside the function so this
module loads in the stdlib-only unit-test env. Pass a live ``db`` to get real
prices; without one (or on any lookup miss) it returns ``None`` and the caller
decides how to fail — the gate fails *open on price-unavailability* for reads
and *closed* for spend, never silently mis-prices.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def price_per_1k(db: Any, model_id: str, provider: Optional[str] = None) -> Optional["ModelPrice"]:
    """Return the (input, output) $/1k-token price for ``model_id`` from the DB.

    ``None`` when the model is unknown to the registry or the registry can't be
    read — never a guessed price. Callers must handle ``None`` explicitly.
    """
    if not model_id or db is None:
        return None
    try:
        from core.llm.model_registry import get_model_registry

        info = get_model_registry(db).get_model(model_id, provider=provider)
        if info is None:
            return None
        return ModelPrice(
            model_id=model_id,
            input_per_1k=float(info.input_cost_per_1k or 0.0),
            output_per_1k=float(info.output_cost_per_1k or 0.0),
        )
    except Exception:
        logger.warning(
            "[policy.pricing] registry price lookup failed for model=%s", model_id,
            exc_info=True,
        )
        return None


def estimate_cost_usd(
    db: Any, model_id: str, input_tokens: int, output_tokens: int
) -> Optional[float]:
    """Model-aware cost estimate for a token count, or ``None`` if unpriceable.

    Thin wrapper over :func:`price_per_1k` + :meth:`ModelPrice.cost_for` so
    callers get one call. ``None`` propagates the "can't price this" signal.
    """
    price = price_per_1k(db, model_id)
    if price is None:
        return None
    return price.cost_for(input_tokens, output_tokens)


def flat_rate_per_1k() -> float:
    """The documented LAST-RESORT flat $/1k-token rate (PRD-192 S3, F059).

    ``COORDINATOR_COST_PER_1K_TOKENS`` demoted to a registry-miss fallback:
    this module is its ONLY consumer (source-grep-guarded) — every dollar
    figure in the platform routes through pricing, and the flat rate only
    applies when the model registry has no price for the call. Fail-safe: if
    config can't be read, the historical default (0.003) stands.
    """
    try:
        from config import config

        return float(config.COORDINATOR_COST_PER_1K_TOKENS)
    except Exception:
        logger.warning("[policy.pricing] config read failed — flat rate default", exc_info=True)
        return 0.003


def price_total_tokens_usd(db: Any, model_id: Optional[str], total_tokens: int) -> float:
    """Price an UNDIFFERENTIATED total-token estimate (PRD-192 S3, F059 finish).

    The mission/playbook ceilings and cost read-outs carry a single total-token
    number with no input/output split (and often span several models). One
    documented convention, one source:

    - registry hit ⇒ the model's blended per-1k rate (mean of input+output —
      an admission/read-out ESTIMATE, not billing; ``llm_usage`` stays the
      ledger of record);
    - no model / registry miss / no db ⇒ :func:`flat_rate_per_1k` (the demoted
      ``COORDINATOR_COST_PER_1K_TOKENS`` last resort).
    """
    tokens = max(0, int(total_tokens or 0))
    if model_id:
        price = price_per_1k(db, model_id)
        if price is not None:
            blended = (price.input_per_1k + price.output_per_1k) / 2.0
            return round((tokens / 1000.0) * blended, 6)
    return round((tokens / 1000.0) * flat_rate_per_1k(), 6)


class ModelPrice:
    """A model's per-1k-token prices, with the token→dollars arithmetic in one place."""

    __slots__ = ("model_id", "input_per_1k", "output_per_1k")

    def __init__(self, model_id: str, input_per_1k: float, output_per_1k: float) -> None:
        self.model_id = model_id
        self.input_per_1k = float(input_per_1k)
        self.output_per_1k = float(output_per_1k)

    def cost_for(self, input_tokens: int, output_tokens: int) -> float:
        """USD cost for the given token counts, rounded to 6dp (matches ModelInfo)."""
        cost = (max(0, int(input_tokens)) / 1000.0) * self.input_per_1k
        cost += (max(0, int(output_tokens)) / 1000.0) * self.output_per_1k
        return round(cost, 6)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"ModelPrice({self.model_id!r}, in={self.input_per_1k}, "
            f"out={self.output_per_1k})"
        )
