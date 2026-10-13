"""P256-FIX-N14-4 (night 14, F418): the Memory page lists the standing rules Auto reads.

Night 14: Auto pointed the owner to the Memory page to delete a standing rule, and the page did
not list it: /browse shows the first rows of each namespace in scroll order, and a rule past them
never shows. The page now lists the viewer's standing rules first, from the read the prompt's block
makes (``modules.context.sections.memory.owners_rule_rows``: same query, same scope), each with
the id the page's delete takes (``modules.memory.forget``).
"""
from __future__ import annotations

import functools
import inspect
import logging
from typing import Any, Awaitable, Callable, Dict, List

logger = logging.getLogger(__name__)

Endpoint = Callable[..., Awaitable[Dict[str, Any]]]
# The /browse filters under which the page shows the durable store's (L3) rows.
L3_TIERS = (None, "all", "l3")


def rule_item(row: Dict[str, Any]) -> Dict[str, Any]:
    """A standing rule as a /browse row (the workspace namespace's, so the global tier)."""
    return {"id": str(row.get("id", "")), "content": row.get("memory") or row.get("content", ""),
            "score": None, "metadata": row.get("metadata"), "created_at": row.get("created_at"),
            "updated_at": None, "tier": "global", "layer": "l3", "content_type": None, "standing_rule": True}


def with_rules_first(page: Dict[str, Any], rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The page with ``rules`` first and no row twice (a new dict)."""
    items = [rule_item(row) for row in rules]
    ids = {item["id"] for item in items}
    rest = [m for m in page.get("memories") or [] if str(m.get("id")) not in ids]
    shown = len(page.get("memories") or []) - len(rest)
    return {**page, "memories": [*items, *rest], "total": int(page.get("total") or 0) + len(items) - shown}


def _shows_the_rules(arguments: Dict[str, Any], page: Any) -> bool:
    """An unfiltered look at the durable rows that answered."""
    tier = (arguments.get("tier") or "all").lower()
    return (isinstance(page, dict) and page.get("success") is True and not arguments.get("query")
            and not arguments.get("content_type") and tier in L3_TIERS)


async def _viewers_rules(arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The standing rules of the person viewing the page, in this workspace."""
    from core.auth.principal import resolve_user_pk
    from modules.context.sections.memory import owners_rule_rows
    from modules.memory.forget import subject_of

    ctx, db = arguments["ctx"], arguments["db"]
    return await owners_rule_rows(ctx.workspace_id, subject_of(resolve_user_pk(db, ctx)))


def lists_the_viewers_rules(endpoint: Endpoint) -> Endpoint:
    """Wrap /browse: an unfiltered page lists the viewer's standing rules first. A failed read
    is logged and the page answers without them, saying so (``standing_rules_unavailable``)."""
    signature = inspect.signature(endpoint)

    @functools.wraps(endpoint)
    async def wrapped(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        page = await endpoint(*args, **kwargs)
        if not _shows_the_rules(bound.arguments, page):
            return page
        try:
            rules = await _viewers_rules(bound.arguments)
        except Exception:  # noqa: BLE001 — logged; the page still answers, and says the rules are missing
            logger.exception("[N14-4] standing rules not listed on the Memory page")
            return {**page, "standing_rules_unavailable": True}
        return with_rules_first(page, rules)
    return wrapped


__all__ = ["lists_the_viewers_rules", "rule_item", "with_rules_first"]
