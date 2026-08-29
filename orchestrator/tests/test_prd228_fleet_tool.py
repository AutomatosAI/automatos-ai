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
           blocked_count=None, needs_attention=0, cost=_UNSET, last_activity=None):
    # blocked_count defaults to len(open_asks); pass it explicitly to model an
    # approval-/manually-blocked task (blocked_at set) that raised NO question —
    # count > 0 with open_asks == [] (P228-RVW-5).
    asks = list(open_asks or [])
    entry = {
        "agent_id": agent_id,
        "name": name,
        "current": current,
        "queue_depth": queue,
        "blocked": {
            "count": blocked_count if blocked_count is not None else len(asks),
            "open_asks": asks,
        },
        "watches": {"active": max(needs_attention, 0), "needs_attention": needs_attention},
        "last_activity_at": last_activity,
    }
    if cost is not _UNSET:
        entry["cost_24h"] = cost
    return entry


def _current(title="Ship it", kind="board_task"):
    return {"kind": kind, "id": 1, "title": title, "since": (NOW - timedelta(minutes=5)).isoformat()}


def _state(entries, *, cost_available=True, watches_available=True, asks_available=True):
    return {
        "version": 1,
        "generated_at": NOW.isoformat(),
        "window_hours": 24,
        "cost_available": cost_available,
        "cost_source": "llm_usage" if cost_available else None,
        "watches_available": watches_available,
        "asks_available": asks_available,
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
    assert out["anomalies"] == {
        "stalled": [], "over_budget_watches": [],
        "blocked_with_open_ask": [], "blocked_no_ask": [],
    }
    assert out["text"].rstrip().endswith("none")


def test_idle_agent_is_never_stalled():
    # An idle agent with old activity is idle, not stalled (no current work).
    fleet = _state([_entry(1, "Bench", current=None,
                           last_activity=(NOW - timedelta(days=3)).isoformat())])
    out = _render_fleet(fleet, now=NOW, stall_seconds=1800)
    assert out["anomalies"]["stalled"] == []


def test_blocked_without_open_ask_is_blocked_not_idle():
    """P228-RVW-5: an approval-/manually-blocked agent (blocked.count>0, open_asks
    empty, no current work) renders a 'blocked' live line — NOT 'idle' — and is
    flagged in the anomalies section, so 'is anyone stuck?' catches it. The idle
    agent alongside it stays idle (blocked.count==0)."""
    fleet = _state([
        # Approval-/manually-blocked: blocked_at set, no KIND_QUESTION grant.
        _entry(1, "Approval", current=None, open_asks=[], blocked_count=1,
               cost={"tokens": 0, "usd": 0.0}),
        # Genuinely idle: nothing blocked, nothing current.
        _entry(2, "Bench", current=None, cost={"tokens": 0, "usd": 0.0}),
    ])
    out = _render_fleet(fleet, now=NOW, stall_seconds=1800)

    approval_line = next(ln for ln in out["lines"] if ln.startswith("Approval"))
    assert " — blocked — " in approval_line          # not idle, not working
    assert "idle" not in approval_line
    bench_line = next(ln for ln in out["lines"] if ln.startswith("Bench"))
    assert " — idle — " in bench_line                # the real idle stays idle

    an = out["anomalies"]
    # Disjoint from the awaiting-answer bucket; caught here instead.
    assert [a["agent"] for a in an["blocked_no_ask"]] == ["Approval"]
    assert an["blocked_no_ask"][0]["count"] == 1
    assert an["blocked_with_open_ask"] == []
    assert "BLOCKED (no open ask): Approval (1 task(s) blocked)" in out["text"]


def test_blocked_count_supersedes_working_line():
    """P228-RVW-5: blocked outranks working — an agent with a flagged-blocked task
    reads as 'blocked' even if it also holds a running task (consistent with the
    existing awaiting-answer-wins-over-working precedence)."""
    fleet = _state([
        _entry(1, "Juggler", current=_current("Another job"), open_asks=[],
               blocked_count=1, cost={"tokens": 0, "usd": 0.0}),
    ])
    out = _render_fleet(fleet, now=NOW, stall_seconds=1800)
    line = out["lines"][0]
    assert " — blocked — " in line
    assert "working:" not in line
    assert [a["agent"] for a in out["anomalies"]["blocked_no_ask"]] == ["Juggler"]


def test_stall_threshold_constant_in_config():
    from config import config

    assert isinstance(config.FLEET_STALL_SECONDS, int)
    assert config.FLEET_STALL_SECONDS > 0
    # The handler reads the threshold from config, never a direct env read.
    src = Path(hf.__file__).read_text(encoding="utf-8")
    assert "config.FLEET_STALL_SECONDS" in src
    assert "getenv" not in src


# ===========================================================================
# 3b. Source-degradation observability (P228-RVW-6)
# ===========================================================================

def test_degraded_source_surfaces_notice_not_clean_none():
    """A watches/asks source failure defaults its anomaly bucket to empty; the
    render must state the source was down (SOURCES DEGRADED) so the empty
    ANOMALIES block is not misread as a clean 'no anomalies'."""
    fleet = _state(
        [_entry(1, "Solo", current=_current("Active job"),
                last_activity=(NOW - timedelta(minutes=1)).isoformat(),
                cost={"tokens": 0, "usd": 0.0})],
        watches_available=False, asks_available=False,
    )
    out = _render_fleet(fleet, now=NOW, stall_seconds=1800)

    # No real anomalies, but the sources are down — must NOT read as clean.
    assert out["anomalies"]["over_budget_watches"] == []
    assert out["anomalies"]["blocked_with_open_ask"] == []
    # The flags propagate to the rendered dict (mirroring cost_available)...
    assert out["watches_available"] is False
    assert out["asks_available"] is False
    # ...and the text warns instead of implying a clean bill of health.
    assert "SOURCES DEGRADED" in out["text"]
    assert "watches: source unavailable" in out["text"]
    assert "asks: source unavailable" in out["text"]


def test_only_the_failed_source_is_flagged_degraded():
    """Watches down but asks healthy → only the watches notice appears."""
    fleet = _state(
        [_entry(1, "Solo", cost={"tokens": 0, "usd": 0.0})],
        watches_available=False, asks_available=True,
    )
    out = _render_fleet(fleet, now=NOW, stall_seconds=1800)
    assert out["watches_available"] is False
    assert out["asks_available"] is True
    assert "watches: source unavailable" in out["text"]
    assert "asks: source unavailable" not in out["text"]


def test_healthy_sources_have_no_degraded_notice():
    """A healthy fleet renders no SOURCES DEGRADED section (no false alarm)."""
    fleet = _state([
        _entry(1, "Fresh", current=_current(),
               last_activity=(NOW - timedelta(minutes=1)).isoformat(),
               cost={"tokens": 0, "usd": 0.0}),
    ])
    out = _render_fleet(fleet, now=NOW, stall_seconds=1800)
    assert out["watches_available"] is True
    assert out["asks_available"] is True
    assert "SOURCES DEGRADED" not in out["text"]


# ===========================================================================
# 4. Heartbeat-orchestrator visibility
# ===========================================================================

def test_heartbeat_orchestrator_mode_is_dispatcher_only():
    from modules.context.modes import MODE_CONFIGS, ContextMode

    assert MODE_CONFIGS[ContextMode.HEARTBEAT_ORCHESTRATOR].tool_loading == "dispatcher_only"


def test_visible_in_heartbeat_dispatcher_surface_openfull_fallback():
    """FALLBACK path only: the tool is in the open-full dispatcher enum the
    heartbeat loads when narrowing CAN'T decide (flag off / no query / rank
    failure → allowed_names=None). This does NOT prove production visibility —
    a real tick supplies a query and ships the ranked semantic top-K, where the
    tool has no reserved slot. That production path is proven by
    ``test_heartbeat_dispatcher_always_includes_fleet_under_semantic_narrowing``
    (the always-include union) — the guarantee P228-RVW-4 requires.
    """
    from modules.tools.discovery.action_registry import get_action_registry

    registry = get_action_registry()
    # allowed_names=None is ONLY the open-full fallback (no query / flag off /
    # rank failure) — NOT the narrowed production tick.
    schema = registry.to_dispatcher_schema(exclude_admin=True, allowed_names=None)
    enum = schema["function"]["parameters"]["properties"]["action"]["enum"]
    assert "platform_fleet_status" in enum


@pytest.mark.asyncio
async def test_heartbeat_dispatcher_always_includes_fleet_under_semantic_narrowing(monkeypatch):
    """P228-RVW-4 (production path): on a real heartbeat tick the dispatcher enum
    is the ranked semantic top-K (SEMANTIC_TOOL_ROUTING on, a non-empty query),
    where platform_fleet_status has NO reserved slot. Drive the SAME loader the
    heartbeat runs (``ToolsSection().load_tools(strategy=DISPATCHER_ONLY,
    query=...)``) with ranking that returns a TOP-K WITHOUT the fleet tool (as if
    it ranked #16+), and assert it is STILL in the enum — reachable regardless of
    the ranking outcome, via the always-include union rather than the
    allowed_names=None open-full path.
    """
    from modules.context.sections.tools import ToolLoadingStrategy, ToolsSection
    import modules.tools.tool_router as tr

    # A decided, NARROWED ranked list that does NOT contain the fleet tool.
    ranked_without_fleet = [
        "platform_list_agents", "platform_list_tasks", "platform_list_missions",
        "platform_list_watches", "platform_search_memory",
    ]
    assert "platform_fleet_status" not in ranked_without_fleet

    async def _fake_narrow(query, is_admin, is_super_admin):
        # (allowed_names, reason, from_pins) — a decided, narrowed surface.
        assert query  # the heartbeat always supplies a non-empty query
        return list(ranked_without_fleet), "ranked", False

    monkeypatch.setattr(tr, "_narrow_dispatcher_actions_async", _fake_narrow)

    # A representative non-empty heartbeat query (mirrors heartbeat_service.py).
    query = ("Perform a scheduled health check for this workspace. "
             "Analyze your workspace using the tools provided.")
    tools, tool_choice = await ToolsSection().load_tools(
        agent_id=None, workspace_id="ws-hb",
        strategy=ToolLoadingStrategy.DISPATCHER_ONLY, query=query,
    )

    assert tool_choice == "auto"
    enum = tools[0]["function"]["parameters"]["properties"]["action"]["enum"]
    # The always-include re-added the fleet tool despite it not being ranked...
    assert "platform_fleet_status" in enum
    # ...and the union is ADDITIVE — non-promoted ranked entries are preserved
    # (promoted names like platform_list_agents are excluded from the dispatcher
    # enum by exclude_promoted, which is orthogonal to this fix).
    assert "platform_list_tasks" in enum
    assert len(enum) >= 2


def test_apply_dispatcher_always_include_unions_onto_narrowed_but_not_openfull():
    """P228-RVW-4 unit: the helper appends configured, gate-cleared pins onto a
    NARROWED list (dedup, order-stable) and leaves an open-full (None) surface
    untouched — the full enum already exposes everything."""
    from modules.tools.tool_router import _apply_dispatcher_always_include

    # open-full surface: unchanged.
    assert _apply_dispatcher_always_include(None) is None

    # narrowed surface without the fleet tool → fleet appended (registered +
    # non-admin, so it clears the gate); the ranked head stays first.
    out = _apply_dispatcher_always_include(["platform_list_agents"])
    assert out is not None
    assert out[0] == "platform_list_agents"
    assert "platform_fleet_status" in out

    # idempotent: already present → not duplicated.
    already = _apply_dispatcher_always_include(["platform_fleet_status", "platform_list_agents"])
    assert already.count("platform_fleet_status") == 1


def test_apply_dispatcher_always_include_drops_unknown_or_gated_pin(monkeypatch):
    """P228-RVW-4: a configured pin that is unregistered (or role-gated) is never
    forced into the enum — the same fail-closed gate page-prior uses."""
    import modules.tools.tool_router as tr

    monkeypatch.setattr(
        tr, "_dispatcher_always_include",
        lambda: ["platform_fleet_status", "definitely_not_a_real_action_xyz"],
    )
    out = tr._apply_dispatcher_always_include(["platform_list_agents"])
    assert "platform_fleet_status" in out                  # registered → admitted
    assert "definitely_not_a_real_action_xyz" not in out   # unknown → dropped


def test_dispatcher_always_include_defaults_to_fleet_status():
    """The config default pins platform_fleet_status so the heartbeat guarantee
    holds out of the box (no env override required)."""
    from modules.tools.tool_router import _dispatcher_always_include

    assert "platform_fleet_status" in _dispatcher_always_include()
