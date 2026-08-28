"""Fleet-status handler for PlatformActionExecutor — PRD-228 US-003.

``platform_fleet_status`` renders the US-001 read-model
(:func:`services.fleet_state.get_fleet_state`) into a compact, token-cheap form
Auto can read in one call: one line per agent (name — current work or idle —
queue depth — 24h cost) plus an ANOMALIES section (stalled, over-budget watches,
blocked-with-open-ask). It writes nothing — it reads the floor and summarises it.

The rendering + anomaly detection live in the pure :func:`_render_fleet` so they
are unit-testable against a fixture fleet with an injected clock and threshold.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from config import config
from services.fleet_state import get_fleet_state

logger = logging.getLogger(__name__)


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _format_cost(entry: Dict[str, Any]) -> str:
    cost = entry.get("cost_24h")
    if cost is None:
        return "cost n/a"
    tokens = int(cost.get("tokens", 0) or 0)
    usd = float(cost.get("usd", 0.0) or 0.0)
    return f"{tokens:,} tok / ${usd:,.2f}"


def _work_phrase(entry: Dict[str, Any]) -> str:
    """The agent's live line: blocked > working > idle."""
    if entry["blocked"]["open_asks"]:
        return "blocked: awaiting answer"
    current = entry.get("current")
    if current:
        return f"working: {current['title']}"
    return "idle"


def _render_fleet(
    state: Dict[str, Any],
    *,
    now: datetime,
    stall_seconds: int,
) -> Dict[str, Any]:
    """Compact rendering + anomaly detection over a fleet-state dict (pure).

    Anomalies:
      * **stalled** — an agent shown as working whose last activity is older
        than ``stall_seconds``.
      * **over_budget_watches** — an agent with one or more watches that hit
        their action budget (``watches.needs_attention``).
      * **blocked_with_open_ask** — an agent with a pending question ask.
    """
    lines: List[str] = []
    stalled: List[Dict[str, Any]] = []
    over_budget: List[Dict[str, Any]] = []
    blocked_with_ask: List[Dict[str, Any]] = []

    for entry in state.get("agents", []):
        name = entry["name"]
        queue = entry["queue_depth"]
        lines.append(
            f"{name} — {_work_phrase(entry)} — queue {queue} — {_format_cost(entry)}"
        )

        # Stalled: claims to be working, but no recent activity.
        if entry.get("current") is not None:
            last = _parse_iso(entry.get("last_activity_at"))
            if last is not None and (now - last).total_seconds() > stall_seconds:
                stalled.append({
                    "agent_id": entry["agent_id"],
                    "agent": name,
                    "last_activity_at": entry.get("last_activity_at"),
                })

        na = entry["watches"].get("needs_attention", 0)
        if na:
            over_budget.append({
                "agent_id": entry["agent_id"], "agent": name, "watches": na,
            })

        open_asks = entry["blocked"]["open_asks"]
        if open_asks:
            blocked_with_ask.append({
                "agent_id": entry["agent_id"], "agent": name, "open_asks": open_asks,
            })

    anomalies = {
        "stalled": stalled,
        "over_budget_watches": over_budget,
        "blocked_with_open_ask": blocked_with_ask,
    }
    return {
        "as_of": state.get("generated_at"),
        "window": "last 24h",
        "cost_available": state.get("cost_available", False),
        "agent_count": len(state.get("agents", [])),
        "lines": lines,
        "anomalies": anomalies,
        "text": _as_text(lines, anomalies),
    }


def _as_text(lines: List[str], anomalies: Dict[str, List[Dict[str, Any]]]) -> str:
    """The single-string compact rendering (agent lines + ANOMALIES section)."""
    body = "\n".join(lines) if lines else "(no agents)"
    flagged: List[str] = []
    for agent in anomalies["stalled"]:
        flagged.append(f"STALLED: {agent['agent']} (no activity since {agent['last_activity_at']})")
    for agent in anomalies["over_budget_watches"]:
        flagged.append(f"OVER-BUDGET WATCH: {agent['agent']} ({agent['watches']} need attention)")
    for agent in anomalies["blocked_with_open_ask"]:
        flagged.append(f"BLOCKED (awaiting answer): {agent['agent']} — asks {agent['open_asks']}")
    anomaly_block = "\n".join(flagged) if flagged else "none"
    return f"{body}\n\nANOMALIES:\n{anomaly_block}"


async def fleet_status(
    db: Session, workspace_id: UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """platform_fleet_status — a compact, one-call read of live floor state."""
    try:
        state = get_fleet_state(db, workspace_id)
    except Exception as e:  # noqa: BLE001 — a read tool degrades to an honest error
        logger.error("[Fleet] fleet_status failed: %s", e, exc_info=True)
        return {"success": False, "error": f"Failed to read fleet state: {str(e)[:300]}"}

    rendered = _render_fleet(
        state,
        now=datetime.now(timezone.utc),
        stall_seconds=config.FLEET_STALL_SECONDS,
    )
    return {"success": True, **rendered}
