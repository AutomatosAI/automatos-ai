"""PRD-270 US-270-07b (K10) — the public read of the Marketplace's LLM catalogue.

Active ``llm_models`` rows, one per route (a serving provider and a model id), in the member LLMs tab's
default order (``api/llm_marketplace.py`` ``browse_catalog``: most installed, featured, then by name),
with the id as the last tie. The member answer also says whether THIS workspace installed a route and
whether it holds a key for its provider (``_get_installed_ids``, ``_get_available_providers``): both are
a workspace's, so neither is called, and no workspace model, API key or credential is read. The query
names its columns, so a route's temperatures, plan, external id, sourcing and recommendations are never
read either.

The public fields are what the catalogue says about the route itself, through the member route's own
helpers: its name, ``provider`` (the serving provider's slug, the catalogue's ``provider`` filter),
``vendor``, context window, price tier (``_model_to_out``'s rule: from the input price, or the model's
list price, else ``unknown``) and capabilities, each only its name and rating (a rating that isn't text
is left out). The key is the route key, ``<serving provider>:<model id>``: one model id can be served
by several providers, and §5.7 wants one item per link. The publisher is the serving provider's label.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from api.llm_marketplace import PRICE_TIER_UNKNOWN, price_tier_for, route_capabilities, route_key, route_price
from core.llm import providers as registry
from core.llm.list_prices import list_rates
from core.models.core import LLMModel
from services.marketplace_public.fields import (
    LLM_CAPABILITY_FIELDS, LLM_FIELDS, allowed_item, count_of, display_item, text_or_none,
)
from services.marketplace_public.publishers import display_name

ACTIVE_ROUTE = "active"


def _price_tier(row: Any) -> str:
    """The route's price tier as the member catalogue shows it: ``unknown`` when no price is known."""
    price_known = (
        row.input_cost_per_1k_tokens is not None or row.output_cost_per_1k_tokens is not None
        or list_rates(row.model_id) is not None
    )
    if not price_known:
        return PRICE_TIER_UNKNOWN
    input_cost, _output_cost, _free = route_price(row)
    return price_tier_for(input_cost)


def _capabilities(row: Any) -> List[Dict[str, Any]]:
    """The route's capability ratings as ``{name, rating}``, text only, in the row's order."""
    rated = ((text_or_none(name), text_or_none(rating)) for name, rating in route_capabilities(row).items())
    return [allowed_item(LLM_CAPABILITY_FIELDS, name=name, rating=rating) for name, rating in rated if name and rating]


def _provider_label(serving_provider: Optional[str]) -> str:
    spec = registry.get_spec(serving_provider)
    return display_name(spec.label if spec else (serving_provider or "").title())


def _item(row: Any) -> Dict[str, Any]:
    return display_item(
        LLM_FIELDS, key=route_key(row.serving_provider, row.model_id), name=row.display_name,
        description=row.description, category=row.category, tags=row.tags,
        publisher=_provider_label(row.serving_provider), provider=text_or_none(row.serving_provider),
        vendor=text_or_none(row.provider), contextWindow=count_of(row.context_window),
        priceTier=_price_tier(row), capabilities=_capabilities(row),
    )


def read_llms(db: Session, limit: int, offset: int) -> List[Dict[str, Any]]:
    """Up to ``limit`` active LLM routes from ``offset``, each as its public item."""
    rows = (
        db.query(
            LLMModel.id, LLMModel.provider, LLMModel.serving_provider, LLMModel.model_id, LLMModel.display_name,
            LLMModel.description, LLMModel.category, LLMModel.tags, LLMModel.context_window,
            LLMModel.input_cost_per_1k_tokens, LLMModel.output_cost_per_1k_tokens, LLMModel.capabilities,
        )
        .filter(LLMModel.status == ACTIVE_ROUTE)
        .order_by(
            LLMModel.install_count.desc().nullslast(), LLMModel.is_featured.desc().nullslast(),
            LLMModel.display_name, LLMModel.id,
        )
        .offset(offset)
        .limit(limit)
        .all()
    )
    return [_item(row) for row in rows]


__all__ = ["read_llms"]
