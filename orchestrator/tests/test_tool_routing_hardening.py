"""PRD-142 Wave 4 (W4-S9): harden the tool-routing learning loop to the §H DoD.

The loop (signal_recorder -> tool_routing_edges/affinities -> edge_builder ->
graph_router, PRD-138/139) is already wired; this hardens it:

- **Tenant isolation (A4):** signals for different workspaces aggregate into
  DISTINCT rows — never bleed across tenants. Proven at the aggregation layer
  (the upsert SQL keys on `workspace_id IS NOT DISTINCT FROM`).
- **Failure-path / restart-safety (E1):** `record()` is best-effort and NEVER
  raises or blocks a tool call — disabled, no event loop, or a full queue all
  drop silently. Authoritative learning survives a restart because the nightly
  edge_builder recomputes from the durable tool_execution_logs (so the in-memory
  queue is intra-day freshness only, not the system of record).
- **Observable:** `stats()` exposes the counters the self-learning tile reads.

`signal_recorder` is leaf-loadable (stdlib-only top imports), so these are pure
unit tests — no DB, no apscheduler, no env stubbing.
"""
from modules.tools.discovery.signal_recorder import ToolSignal, ToolSignalRecorder


# --- tenant + agent isolation (the aggregation never merges scopes) --------

def test_aggregate_separates_workspaces():
    _edges, affs = ToolSignalRecorder._aggregate([
        ToolSignal("send_email", True, workspace_id="ws-1"),
        ToolSignal("send_email", True, workspace_id="ws-2"),
    ])
    # Same action + affinity_type, different workspace -> two distinct rows.
    # affinity key = (action_name, affinity_type, agent_id, workspace_id)
    workspaces = {key[3] for key in affs}
    assert workspaces == {"ws-1", "ws-2"}
    assert len(affs) == 2  # no cross-tenant bleed


def test_aggregate_separates_agents():
    _edges, affs = ToolSignalRecorder._aggregate([
        ToolSignal("send_email", True, agent_id=1, workspace_id="ws-1"),
        ToolSignal("send_email", True, agent_id=2, workspace_id="ws-1"),
    ])
    assert len(affs) == 2


def test_same_scope_aggregates_into_one_row():
    _edges, affs = ToolSignalRecorder._aggregate([
        ToolSignal("send_email", True, agent_id=1, workspace_id="ws-1"),
        ToolSignal("send_email", True, agent_id=1, workspace_id="ws-1"),
    ])
    assert len(affs) == 1
    assert list(affs.values()) == [2]  # incremented, not duplicated


# --- the negative-signal path (what HARNESS reads in W4-S10) ----------------

def test_failure_records_fails_for_intent_and_failed_after():
    edges, affs = ToolSignalRecorder._aggregate([
        ToolSignal("flaky_tool", False, workspace_id="ws-1", prior_action="search"),
    ])
    assert "fails_for_intent" in {key[1] for key in affs}
    assert "failed_after" in {key[2] for key in edges}


def test_success_records_agent_prefers_and_used_after():
    edges, affs = ToolSignalRecorder._aggregate([
        ToolSignal("send_email", True, workspace_id="ws-1", prior_action="draft"),
    ])
    assert "agent_prefers" in {key[1] for key in affs}
    assert "used_after" in {key[2] for key in edges}


def test_self_transition_produces_no_edge():
    edges, _affs = ToolSignalRecorder._aggregate([
        ToolSignal("send_email", True, workspace_id="ws-1", prior_action="send_email"),
    ])
    assert edges == {}  # an action can't be "used after" itself


# --- failure-safety: telemetry never breaks a tool call --------------------

def test_record_is_safe_when_disabled(monkeypatch):
    r = ToolSignalRecorder()
    monkeypatch.setattr(ToolSignalRecorder, "_enabled", staticmethod(lambda: False))
    r.record(ToolSignal("x", True))  # must not raise
    assert r.stats()["recorded"] == 0


def test_record_is_safe_without_event_loop(monkeypatch):
    r = ToolSignalRecorder()
    monkeypatch.setattr(ToolSignalRecorder, "_enabled", staticmethod(lambda: True))
    # Called synchronously (no running loop): drops, never raises.
    r.record(ToolSignal("x", True))
    s = r.stats()
    assert s["recorded"] == 0
    assert s["dropped"] == 1


# --- observability ----------------------------------------------------------

def test_stats_exposes_tile_counters():
    s = ToolSignalRecorder().stats()
    assert set(s) >= {"recorded", "dropped", "flushes", "flush_errors", "queue_depth"}
    assert all(v == 0 for v in s.values())  # fresh recorder starts clean
