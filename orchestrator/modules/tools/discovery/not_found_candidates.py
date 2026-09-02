"""Not-found errors that name the closest REAL marketplace entries.

Live tests 2026-08-29 → 2026-09-02: Auto invented marketplace names
('shopify-store-manager', 'shopify-tools', 'customer-support-agent-skill',
'Restaurant Customer Service Agent'), got a bare "not found", and either
guessed again or stalled the build. A dead-end error is the wrong shape for
an LLM caller: the honest reply names what DOES exist so the next call can be
right. Nothing here installs or chooses — it only tells the truth about the
catalogue, and a failing lookup never turns a not-found into a crash.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

_GENERIC = frozenset({
    "agent", "agents", "tool", "tools", "plugin", "plugins", "skill", "skills",
    "package", "packages", "service", "services", "manager", "management",
    "assistant", "custom", "automatos", "the", "and", "for", "with", "app", "apps",
})


def candidate_terms(requested: str, *, max_terms: int = 4) -> List[str]:
    """Distinctive words of a requested name, in order (generic words dropped)."""
    tokens = [t for t in re.split(r"[^a-z0-9]+", (requested or "").lower()) if len(t) >= 3]
    distinct = [t for t in tokens if t not in _GENERIC]
    out: List[str] = []
    for t in distinct or tokens:
        if t not in out:
            out.append(t)
    return out[:max_terms]


async def find_candidates(browse, db: Any, workspace_id: Any, requested: str, *,
                          list_key: str, limit: int = 3) -> List[Dict[str, Any]]:
    """Up to ``limit`` real entries from ``browse`` that share a word with the request."""
    seen: Dict[str, Dict[str, Any]] = {}
    for term in candidate_terms(requested):
        try:
            result = await browse(db, workspace_id, {"search": term, "limit": 5})
        except Exception:  # noqa: BLE001 — candidates are a courtesy, never a crash
            continue
        for item in (result or {}).get(list_key) or []:
            key = str(item.get("slug") or item.get("name") or item.get("id") or "")
            if key and key not in seen:
                seen[key] = {k: item.get(k) for k in ("id", "slug", "name") if item.get(k) is not None}
        if len(seen) >= limit:
            break
    return list(seen.values())[:limit]


def not_found_error(kind: str, requested: str, candidates: List[Dict[str, Any]], *,
                    search_tool: str) -> Dict[str, Any]:
    """The tool result for a miss: what was asked, what exists, what to do."""
    if candidates:
        named = ", ".join(
            f"'{c.get('slug') or c.get('name')}'" + (f" ({c['name']})" if c.get("slug") and c.get("name") else "")
            for c in candidates
        )
        error = (
            f"{kind} not found: '{requested}'. Closest in the marketplace: {named}. "
            f"Use one of those exactly, or search with {search_tool} first — never guess a name."
        )
    else:
        error = (
            f"{kind} not found: '{requested}', and nothing in the marketplace resembles it. "
            f"Search with {search_tool} first — never guess a name."
        )
    return {"success": False, "error": error, "requested": requested, "candidates": list(candidates)}
