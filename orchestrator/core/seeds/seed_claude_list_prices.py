"""The Claude 5.5 routes carry their list prices (P256-FIX-T2, F399).

Anthropic's Models API publishes no prices, so the catalogue sync added
claude-opus-5-5, claude-sonnet-5-5 and claude-haiku-5-5 unpriced (#829) and
the model pickers showed them as free. This seed gives each direct Anthropic
route its list price from ``core.llm.list_prices``:

- a missing route is inserted with the sync's own new-row values
  (``anthropic_catalog_sync.new_row_defaults`` / ``synced_values``), priced;
- an existing route gets a price only in a column that is NULL: an operator's
  figure, or an explicit 0, is never overwritten; its status is left alone (a
  route the sync deprecated stays deprecated).

Haiku 5.5's row holds its first tier ($0.10/$0.50); the over-100k tier is
applied per call by the usage tracker from the same table.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Tuple

from sqlalchemy.orm import Session

from core.llm.list_prices import list_price
from core.models.core import LLMModel

logger = logging.getLogger(__name__)

PROVIDER = "anthropic"
_ONE_M = 1_000_000
_OUTPUT_128K = 128_000

# (model id, display name, context window, output cap); 0 = not published, the
# Anthropic sync fills it from the Models API.
SEEDED_ROUTES: Tuple[Tuple[str, str, int, int], ...] = (
    ("claude-opus-5-5", "Claude Opus 5.5", _ONE_M, _OUTPUT_128K),
    ("claude-sonnet-5-5", "Claude Sonnet 5.5", _ONE_M, _OUTPUT_128K),
    ("claude-haiku-5-5", "Claude Haiku 5.5", 0, 0),
)


def seed_claude_list_prices(db: Session) -> Dict[str, int]:
    """Insert the missing 5.5 routes and price the unpriced ones; ``{"inserted", "priced"}``.
    The caller commits."""
    counts = {"inserted": 0, "priced": 0}
    for model_id, name, context, output in SEEDED_ROUTES:
        price = list_price(model_id)
        row = (
            db.query(LLMModel)
            .filter(LLMModel.serving_provider == PROVIDER, LLMModel.model_id == model_id)
            .first()
        )
        if row is None:
            db.add(LLMModel(**_new_route(model_id, name, context, output, price)))
            counts["inserted"] += 1
        elif _price_if_null(row, price):
            counts["priced"] += 1
    db.flush()
    logger.info("Claude list prices: %(inserted)d routes added, %(priced)d priced", counts)
    return counts


def _new_route(model_id: str, name: str, context: int, output: int, price: Any) -> Dict[str, Any]:
    from core.services.anthropic_catalog_sync import new_row_defaults, synced_values

    listed = {"id": model_id, "display_name": name, "max_input_tokens": context, "max_tokens": output}
    return dict(
        new_row_defaults(listed, None),
        **synced_values(listed),
        serving_provider=PROVIDER,
        model_id=model_id,
        input_cost_per_1k_tokens=price.input_per_1k,
        output_cost_per_1k_tokens=price.output_per_1k,
        pricing_updated_at=datetime.utcnow(),
    )


def _price_if_null(row: LLMModel, price: Any) -> bool:
    """Fill a NULL input or output price; True when either was filled."""
    filled = False
    if row.input_cost_per_1k_tokens is None:
        row.input_cost_per_1k_tokens = price.input_per_1k
        filled = True
    if row.output_cost_per_1k_tokens is None:
        row.output_cost_per_1k_tokens = price.output_per_1k
        filled = True
    if filled:
        row.pricing_updated_at = datetime.utcnow()
    return filled


__all__ = ["SEEDED_ROUTES", "seed_claude_list_prices"]
