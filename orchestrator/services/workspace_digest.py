"""PRD-221 S8 — workspace digest snapshot for Auto's Read.

Reduces the live activity feed + stats into a compact, plain-language snapshot
(names and one-line reasons, never raw event dumps) plus a stable ``state_hash``.
The hash is the cache key for the digest endpoint (S9): identical underlying
state — regardless of feed ordering — yields the same hash, so the digest LLM
is invoked at most once per real state change, not per pageview.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List

from services.activity_service import ActivityService

# Item statuses (produced by ActivityService fetchers) grouped for the digest.
_ATTENTION_STATUSES = {"failed", "error"}
_ACTIVE_STATUSES = {"running", "pending"}
_DONE_STATUSES = {"completed"}

_FEED_SCAN_LIMIT = 50
_MAX_LISTED = 8


def _attention_reason(item: Dict[str, Any]) -> str:
    """One-line plain-English reason an item needs attention."""
    msg = (item.get("error_message") or "").strip()
    if msg:
        return msg[:160]
    return "Stopped and needs a look."


def _name(item: Dict[str, Any]) -> str:
    return (item.get("name") or "").strip() or "(untitled)"


def _state_hash(items: List[Dict[str, Any]]) -> str:
    """Stable digest of the workspace's item states.

    Canonical projection = sorted ``type:id:status`` triples. Independent of
    feed order and free of any generation timestamp, so equal state → equal
    hash; a status flip or a new/removed item changes it.
    """
    projection = sorted(
        f"{it.get('type')}:{it.get('id')}:{it.get('status')}" for it in items
    )
    blob = json.dumps(projection, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_digest_snapshot(
    db, workspace_id, period: str = "1d"
) -> Dict[str, Any]:
    """Build the plain-language snapshot + state_hash for a workspace.

    Pure aggregation over ActivityService — no LLM, no cache. The endpoint
    (S9) turns this into prose and caches it by ``state_hash``.
    """
    svc = ActivityService(db, workspace_id)
    stats = svc.get_stats(period=period)
    feed = svc.get_feed(period=period, limit=_FEED_SCAN_LIMIT)
    items = feed.get("items", []) or []

    needs_attention: List[Dict[str, str]] = []
    active: List[Dict[str, str]] = []
    recent_completions: List[Dict[str, str]] = []

    for item in items:
        status = item.get("status")
        if status in _ATTENTION_STATUSES:
            needs_attention.append({"name": _name(item), "reason": _attention_reason(item)})
        elif status in _ACTIVE_STATUSES:
            active.append({"name": _name(item), "type": item.get("type") or "work"})
        elif status in _DONE_STATUSES:
            recent_completions.append({"name": _name(item), "type": item.get("type") or "work"})

    snapshot = {
        "period": period,
        "counts": {
            "working_now": stats.get("working_now", 0),
            "completed": stats.get("completed_today", 0),
            "needs_attention": stats.get("needs_attention", 0),
            "channels_live": stats.get("channels_live", 0),
        },
        "needs_attention": needs_attention[:_MAX_LISTED],
        "active": active[:_MAX_LISTED],
        "recent_completions": recent_completions[:_MAX_LISTED],
        "needs_attention_count": len(needs_attention),
        "state_hash": _state_hash(items),
    }
    return snapshot


def render_fallback_digest(snapshot: Dict[str, Any]) -> str:
    """Deterministic plain-English digest from the snapshot — the never-500
    fallback when the LLM is unavailable (S9)."""
    counts = snapshot.get("counts", {})
    attention = snapshot.get("needs_attention", [])
    parts: List[str] = []

    working = counts.get("working_now", 0)
    done = counts.get("completed", 0)
    if working or done:
        parts.append(
            f"{working} item{'s' if working != 1 else ''} working now, "
            f"{done} completed."
        )
    else:
        parts.append("Nothing is running right now.")

    if attention:
        first = attention[0]
        parts.append(
            f"{len(attention)} need{'s' if len(attention) != 1 else ''} attention — "
            f"e.g. \"{first['name']}\": {first['reason']}"
        )
    else:
        parts.append("Nothing needs your attention.")

    return " ".join(parts)
