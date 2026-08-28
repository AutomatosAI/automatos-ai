"""PRD-228 US-003 — platform_fleet_status tool.

Pure tests over the 3-file registration, the compact rendering / anomaly
detection, and the heartbeat-orchestrator visibility. No DB / network: the
handler's read-model call is monkeypatched to a fixture fleet, and the
rendering logic is exercised through the pure ``_render_fleet``.
"""
from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import modules.tools.discovery.handlers_fleet as hf  # noqa: E402
from modules.tools.discovery.handlers_fleet import _render_fleet, fleet_status  # noqa: E402

NOW = datetime(2026, 8, 28, 12, 0, 0, tzinfo=timezone.utc)
_UNSET = object()


def _entry(agent_id, name, *, current=None, queue=0, open_asks=None,
           needs_attention=0, cost=_UNSET, last_activity=None):
    entry = {
        "agent_id": agent_id,
        "name": name,
        "current": current,
        "queue_depth": queue,
        "blocked": {"count": len(open_asks or []), "open_asks": list(open_asks or [])},
        "watches": {"active": max(needs_attention, 0), "needs_attention": needs_attention},
        "last_activity_at": last_activity,
    }
    if cost is not _UNSET:
        entry["cost_24h"] = cost
    return entry


def _current(title="Ship it", kind="board_task"):
    return {"kind": kind, "id": 1, "title": title, "since": (NOW - timedelta(minutes=5)).isoformat()}


def _state(entries, *, cost_available=True):
    return {
        "version": 1,
        "generated_at": NOW.isoformat(),
        "window_hours": 24,
        "cost_available": cost_available,
        "cost_source": "llm_usage" if cost_available else None,
        "agents": entries,
    }


# ===========================================================================
# 1. Registration — the 3-file pattern
# ===========================================================================

def test_registered_in_action_registry():
    from modules.tools.discovery.action_registry import ActionRegistry
    from modules.tools.discovery.actions_fleet import register_fleet_actions

    reg = ActionRegistry()
    register_fleet_actions(reg)
    action = reg.get("platform_fleet_status")
    assert action is not None
    assert action.permission_level == "read"  # reachable from normal chat
    assert action.admin_only is False
    # Takes no arguments — no field the handler cannot default.
    assert action.parameters["required"] == []


def test_handler_is_async():
    assert asyncio.iscoroutinefunction(fleet_status)


def test_wired_into_executor_map():
    """The executor routes the tool name to the handler (3rd file)."""
    from modules.tools.discovery import platform_executor as pe

    src = Path(pe.__file__).read_text(encoding="utf-8")
    assert '"platform_fleet_status": fleet_status' in src


def test_registration_wired_into_platform_actions():
    from modules.tools.discovery import platform_actions as pa

    src = Path(pa.__file__).read_text(encoding="utf-8")
    assert "register_fleet_actions(registry)" in src


# ===========================================================================
# 2. Compact rendering from a fixture fleet
# ===========================================================================

@pytest.mark.asyncio
async def test_handler_returns_compact_form(monkeypatch):
    fleet = _state([
        _entry(1, "Builder", current=_current("Ship the widget"), queue=2,
               cost={"tokens": 1200, "usd": 0.34}, last_activity=(NOW - timedelta(minutes=1)).isoformat()),
        _entry(2, "Bench", cost={"tokens": 0, "usd": 0.0}),
    ])
    monkeypatch.setattr(hf, "get_fleet_state", lambda db, ws: fleet)

    res = await fleet_status(db=None, workspace_id=uuid4(), params={})
    assert res["success"] is True
    assert res["agent_count"] == 2
    assert res["window"] == "last 24h"
    assert len(res["lines"]) == 2
    assert any("working: Ship the widget" in ln and "queue 2" in ln for ln in res["lines"])
    assert any("idle" in ln for ln in res["lines"])
    assert "1,200 tok / $0.34" in res["lines"][0]
    assert "ANOMALIES" in res["text"]


@pytest.mark.asyncio
async def test_handler_failsoft_on_read_error(monkeypatch):
    def _boom(db, ws):
        raise RuntimeError("db down")

    monkeypatch.setattr(hf, "get_fleet_state", _boom)
    res = await fleet_status(db=None, workspace_id=uuid4(), params={})
    assert res["success"] is False
    assert "Failed to read fleet state" in res["error"]


def test_cost_unavailable_renders_na():
    fleet = _state([_entry(1, "Solo", current=_current(), cost=_UNSET)], cost_available=False)
    out = _render_fleet(fleet, now=NOW, stall_seconds=1800)
    assert "cost n/a" in out["lines"][0]


# ===========================================================================
# 3. Anomaly detection (stalled + over-budget + blocked-with-ask)
# ===========================================================================

def test_anomalies_flag_stalled_and_blocked_and_over_budget():
    fleet = _state([
        # Stalled: shown working, but last activity 2h ago (> 30min threshold).
        _entry(1, "Staller", current=_current("Long job"),
               last_activity=(NOW - timedelta(hours=2)).isoformat()),
        # Healthy: working with recent activity — NOT stalled.
        _entry(2, "Fresh", current=_current("Active job"),
               last_activity=(NOW - timedelta(minutes=1)).isoformat()),
        # Blocked on an open ask.
        _entry(3, "Stuck", open_asks=[99]),
        # A watch that hit its action budget.
        _entry(4, "Overseer", current=_current("Watched job"),
               last_activity=(NOW - timedelta(minutes=1)).isoformat(), needs_attention=1),
    ])
    out = _render_fleet(fleet, now=NOW, stall_seconds=1800)
    an = out["anomalies"]

    assert [a["agent"] for a in an["stalled"]] == ["Staller"]
    assert [a["agent"] for a in an["blocked_with_open_ask"]] == ["Stuck"]
    assert an["blocked_with_open_ask"][0]["open_asks"] == [99]
    assert [a["agent"] for a in an["over_budget_watches"]] == ["Overseer"]

    assert "STALLED: Staller" in out["text"]
    assert "BLOCKED (awaiting answer): Stuck" in out["text"]
    assert "OVER-BUDGET WATCH: Overseer" in out["text"]


def test_no_anomalies_says_none():
    fleet = _state([
        _entry(1, "Fresh", current=_current(), last_activity=(NOW - timedelta(minutes=1)).isoformat()),
        _entry(2, "Bench"),
    ])
    out = _render_fleet(fleet, now=NOW, stall_seconds=1800)
    assert out["anomalies"] == {"stalled": [], "over_budget_watches": [], "blocked_with_open_ask": []}
    assert out["text"].rstrip().endswith("none")


def test_idle_agent_is_never_stalled():
    # An idle agent with old activity is idle, not stalled (no current work).
    fleet = _state([_entry(1, "Bench", current=None,
                           last_activity=(NOW - timedelta(days=3)).isoformat())])
    out = _render_fleet(fleet, now=NOW, stall_seconds=1800)
    assert out["anomalies"]["stalled"] == []


def test_stall_threshold_constant_in_config():
    from config import config

    assert isinstance(config.FLEET_STALL_SECONDS, int)
    assert config.FLEET_STALL_SECONDS > 0
    # The handler reads the threshold from config (no os.getenv at the call site).
    src = Path(hf.__file__).read_text(encoding="utf-8")
    assert "config.FLEET_STALL_SECONDS" in src
    assert "os.getenv" not in src


# ===========================================================================
# 4. Heartbeat-orchestrator visibility
# ===========================================================================

def test_heartbeat_orchestrator_mode_is_dispatcher_only():
    from modules.context.modes import MODE_CONFIGS, ContextMode

    assert MODE_CONFIGS[ContextMode.HEARTBEAT_ORCHESTRATOR].tool_loading == "dispatcher_only"


def test_visible_in_heartbeat_dispatcher_surface():
    """The tool appears in the open-full dispatcher enum the heartbeat loads
    (``_load_dispatcher_only`` → to_dispatcher_schema(exclude_admin=True))."""
    from modules.tools.discovery.action_registry import get_action_registry

    registry = get_action_registry()
    # allowed_names=None mirrors the default open-full fallback the heartbeat
    # tick uses when no query narrows the enum.
    schema = registry.to_dispatcher_schema(exclude_admin=True, allowed_names=None)
    enum = schema["function"]["parameters"]["properties"]["action"]["enum"]
    assert "platform_fleet_status" in enum
