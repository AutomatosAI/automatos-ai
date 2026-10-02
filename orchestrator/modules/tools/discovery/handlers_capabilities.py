"""Capability-discovery handlers for PlatformActionExecutor (PR-B).

``find_tools`` searches the action registry itself — semantic ranking first,
keyword fallback when the ranker can't answer (embed timeout / empty index) —
so discovery NEVER comes back empty-handed just because an upstream embed was
slow. Results are fail-closed: admin/su-gated actions are never advertised
here regardless of caller (execution-time gates in PlatformActionExecutor
remain the enforcement point; privileged surfaces reach privileged callers
through the enum/include_super_admin path instead).
"""

import logging
from typing import Any, Dict, List, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

_DEFAULT_LIMIT = 8
_MAX_LIMIT = 25


def _compact_params(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """The parameter schema exactly as an LLM needs it to make a call."""
    props = (parameters or {}).get("properties") or {}
    return {
        "required": list((parameters or {}).get("required") or []),
        "properties": {
            name: {
                "type": spec.get("type", "string"),
                "description": spec.get("description", ""),
            }
            for name, spec in props.items()
            if isinstance(spec, dict)
        },
    }


def _keyword_matches(actions: List[Any], query: str, limit: int) -> List[Any]:
    """Ranker-less fallback: token overlap over name/description/tags."""
    tokens = [t for t in query.lower().split() if len(t) > 2]
    if not tokens:
        return []
    scored: List[tuple] = []
    for action in actions:
        haystack = " ".join(
            [action.name, action.description or "", " ".join(action.tags or [])]
        ).lower()
        hits = sum(1 for t in tokens if t in haystack)
        if hits:
            scored.append((hits, action))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [a for _, a in scored[:limit]]


def _discoverable(hidden: Any) -> List[Any]:
    """Fail-closed advertisement: never surface admin/su actions via discovery, nor a
    hidden category (PRD-251B US-B106), nor what cannot run here (F078)."""
    from modules.tools.discovery.action_registry import action_is_available, get_action_registry
    from modules.tools.discovery.hidden_categories import without_hidden

    eligible = [
        a for a in get_action_registry().get_all()
        if not getattr(a, "admin_only", False)
        and not getattr(a, "super_admin_only", False)
        and action_is_available(a)
    ]
    return without_hidden(eligible, hidden)


def _find_limit(params: Dict[str, Any]) -> int:
    """How many matches to return: ``limit``, bounded to 1.._MAX_LIMIT, else the default."""
    try:
        limit = min(int(params.get("limit", _DEFAULT_LIMIT)), _MAX_LIMIT)
    except (TypeError, ValueError):
        limit = _DEFAULT_LIMIT
    return max(1, limit)


def _for_this_turn(eligible: List[Any]) -> List[Any]:
    """F155: a widget turn discovers only what its key's scopes grant."""
    from core.security.surface import widget_scopes, widget_turn

    if not widget_turn():
        return eligible
    from core.security.widget_scopes import allowed_tools

    granted = allowed_tools(widget_scopes())
    return [a for a in eligible if a.name in granted]


async def _ranked_matches(query: str, limit: int, eligible: List[Any], hidden: Any) -> Tuple[List[Any], str]:
    """Semantic ranking first; the keyword fallback when the ranker cannot answer (an embed
    time-out, an empty index), so discovery never comes back empty-handed for that."""
    from modules.tools.discovery.hidden_categories import exclude_kwargs

    by_name = {a.name: a for a in eligible}
    matched: List[Any] = []
    try:
        from modules.tools.discovery.action_semantic_index import get_action_semantic_index

        ranked = await get_action_semantic_index().rank_actions(
            query=query,
            top_k=limit,
            exclude_admin=True,
            exclude_promoted=False,  # discovery spans the WHOLE catalog
            include_super_admin=False,
            **exclude_kwargs(hidden),
        )
        matched = [by_name[n] for n, _ in ranked if n in by_name]
    except Exception:
        logger.warning("find_tools: semantic ranking failed — keyword fallback", exc_info=True)
    if matched:
        return matched, "semantic"
    return _keyword_matches(eligible, query, limit), "keyword"


def _match_row(action: Any, include_params: bool) -> Dict[str, Any]:
    """One match as find_tools answers it: how to call it, and its parameters when asked."""
    row: Dict[str, Any] = {
        "action": action.name,
        "description": action.description,
        "category": action.category,
        "permission_level": getattr(action, "permission_level", "read"),
        "call_with": f"platform_execute(action='{action.name}', params={{...}})",
    }
    if include_params:
        row["params"] = _compact_params(getattr(action, "parameters", {}) or {})
    return row


async def find_tools(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Search the full platform action catalog by natural-language intent."""
    query = str(params.get("query") or "").strip()
    if not query:
        return {"success": False, "error": "query parameter is required"}
    limit = _find_limit(params)
    include_params = params.get("include_params", True) is not False

    from modules.tools.discovery.hidden_categories import hidden_categories_for_workspace

    # PRD-251B US-B106 (B3): a category the workspace is not shown (Socials while it is
    # off for the workspace) is not discoverable either; the handlers' own refusal stays.
    hidden = hidden_categories_for_workspace(workspace_id, db)
    eligible = _for_this_turn(_discoverable(hidden))
    matched, ranker = await _ranked_matches(query, limit, eligible, hidden)
    return {
        "success": True,
        "query": query,
        "ranker": ranker,
        "matches": [_match_row(action, include_params) for action in matched],
        "catalog_size": len(eligible),
        "note": (
            "Call any match via platform_execute with its 'action' name and "
            "required params. Nothing relevant? Rephrase the query — the "
            "catalog is searched by meaning."
        ),
    }
