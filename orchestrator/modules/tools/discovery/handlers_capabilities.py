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
from typing import Any, Dict, List
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


async def find_tools(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Search the full platform action catalog by natural-language intent."""
    query = str(params.get("query") or "").strip()
    if not query:
        return {"success": False, "error": "query parameter is required"}
    try:
        limit = min(int(params.get("limit", _DEFAULT_LIMIT)), _MAX_LIMIT)
    except (TypeError, ValueError):
        limit = _DEFAULT_LIMIT
    limit = max(1, limit)
    include_params = params.get("include_params", True) is not False

    from modules.tools.discovery.action_registry import get_action_registry

    registry = get_action_registry()
    # Fail-closed advertisement: never surface admin/su actions via discovery.
    eligible = [
        a for a in registry.get_all()
        if not getattr(a, "admin_only", False)
        and not getattr(a, "super_admin_only", False)
    ]
    by_name = {a.name: a for a in eligible}

    matched: List[Any] = []
    ranker = "semantic"
    try:
        from modules.tools.discovery.action_semantic_index import (
            get_action_semantic_index,
        )

        ranked = await get_action_semantic_index().rank_actions(
            query=query,
            top_k=limit,
            exclude_admin=True,
            exclude_promoted=False,  # discovery spans the WHOLE catalog
            include_super_admin=False,
        )
        matched = [by_name[n] for n, _ in ranked if n in by_name]
    except Exception:
        logger.warning("find_tools: semantic ranking failed — keyword fallback", exc_info=True)

    if not matched:
        ranker = "keyword"
        matched = _keyword_matches(eligible, query, limit)

    results = []
    for action in matched:
        row: Dict[str, Any] = {
            "action": action.name,
            "description": action.description,
            "category": action.category,
            "permission_level": getattr(action, "permission_level", "read"),
            "call_with": f"platform_execute(action='{action.name}', params={{...}})",
        }
        if include_params:
            row["params"] = _compact_params(getattr(action, "parameters", {}) or {})
        results.append(row)

    return {
        "success": True,
        "query": query,
        "ranker": ranker,
        "matches": results,
        "catalog_size": len(eligible),
        "note": (
            "Call any match via platform_execute with its 'action' name and "
            "required params. Nothing relevant? Rephrase the query — the "
            "catalog is searched by meaning."
        ),
    }
