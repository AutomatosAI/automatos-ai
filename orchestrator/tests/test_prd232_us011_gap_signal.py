"""PRD-232 US-011: the gap signal + shown-vs-used decay + resolution joins.

C7 found the learning loop structurally blind to the two failures that mattered
most: a capability the model needed but wasn't given (a GAP), and an action
surfaced turn after turn but never used (SHOWN-not-used). This suite proves the
three halves US-011 adds, all on the EXISTING tool_execution_logs + affinity
tables (no new table):

  (a) tool_gap events — write a synthetic ``__tool_gap__`` row when the model
      hunts for a capability (platform_find_tools) or a tool-warranted turn runs
      no tool at all. Tested at the writer (write_tool_gap) with the 2026-08-28
      VECTOR query.
  (b) record_selection persists the surfaced set durably, batched through the
      SAME recorder (one DB session per flush), as ``__tool_shown__`` rows; the
      nightly computes shown-vs-used and DECAYS never-used affinities (floored).
  (c) the nightly gap→resolution join: a gap answered later in the same
      conversation (≤24h) by a successful action becomes a succeeds_for_intent for
      the action that served it.

Pure / fixture-based — no DB, no Redis, no embeddings, no network. The nightly
helpers are exercised directly (the edge_builder test idiom) plus one faked
``build_edges`` unit run; the recorder is leaf-loaded with a capturing fake DB.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

# Nightly helpers import cleanly (real ToolRouting* models + numpy) — the
# edge_builder test idiom (test_prd139_edge_builder.py).
from core.services.edge_builder import (  # noqa: E402
    _TOOL_GAP_ACTION,
    _TOOL_SHOWN_ACTION,
    _apply_shown_not_used_decay,
    _compute_gap_resolution_affinities,
    _merge_affinities,
    wilson_lower_bound,
)


# ---------------------------------------------------------------------------
# Leaf-load telemetry + signal_recorder (bypass the heavy modules/tools/__init__)
# ---------------------------------------------------------------------------
def _leaf_load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # Register BEFORE exec so dataclass processing can resolve cls.__module__.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_telemetry = _leaf_load(
    "telemetry_us011", _ORCH / "modules" / "tools" / "execution" / "telemetry.py"
)
_signal_recorder = _leaf_load(
    "signal_recorder_us011", _ORCH / "modules" / "tools" / "discovery" / "signal_recorder.py"
)
write_tool_gap = _telemetry.write_tool_gap
TOOL_GAP_ACTION = _telemetry.TOOL_GAP_ACTION
ToolSignalRecorder = _signal_recorder.ToolSignalRecorder
ToolSignal = _signal_recorder.ToolSignal
SelectionSignal = _signal_recorder.SelectionSignal


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------
_VECTOR_QUERY = "close all the blocked tickets from vector"


def _log(action, status, conv, offset_s, ws="ws1", query=None, shown=None):
    return {
        "id": offset_s,
        "action_name": action,
        "agent_id": 1,
        "workspace_id": ws,
        "status": status,
        "user_query": query or f"do {action}",
        "turn_id": None,
        "conversation_id": conv,
        "executed_at": datetime(2026, 1, 1) + timedelta(seconds=offset_s),
        "shown_actions": shown or [],
    }


def _succeeds_aff(action, cluster, weight, ws="ws1"):
    return {
        "action_name": action,
        "affinity_type": "succeeds_for_intent",
        "workspace_id": ws,
        "agent_id": None,
        "intent_cluster_id": cluster,
        "weight": weight,
        "confidence": wilson_lower_bound(int(weight), int(weight)),
        "sample_count": int(weight),
    }


# ===========================================================================
# (a) tool_gap events — the VECTOR replay produces a gap row
# ===========================================================================
class _FakeSession:
    def __init__(self):
        self.added = []
        self.committed = False
        self.closed = False

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.committed = True

    def rollback(self):
        pass

    def close(self):
        self.closed = True

    def query(self, *a, **k):  # pragma: no cover - no user lookup on these paths
        raise AssertionError("no user lookup expected for a null user_id gap")


def test_vector_replay_produces_a_tool_gap_row():
    """Replay of the 2026-08-28 VECTOR turn: the model hunts for the board-write
    capability (platform_find_tools) → write_tool_gap lands one __tool_gap__ row
    carrying the real intent query, gap_source, and turn identity."""
    sess = _FakeSession()
    asyncio.run(write_tool_gap(
        query=_VECTOR_QUERY,
        workspace_id=None,
        agent_id=7,
        gap_source="find_tools",
        caller_context={"conversation_id": "conv-1", "turn_id": "turn-1"},
        session_factory=lambda: sess,
    ))
    assert sess.committed and sess.closed
    assert len(sess.added) == 1
    row = sess.added[0]
    assert row.action_name == TOOL_GAP_ACTION == "__tool_gap__"
    assert row.status == "gap"
    assert row.user_query == _VECTOR_QUERY
    assert row.agent_id == 7
    assert row.router_decision["tool_gap"] is True
    assert row.router_decision["gap_source"] == "find_tools"
    assert row.router_decision["conversation_id"] == "conv-1"
    assert row.router_decision["turn_id"] == "turn-1"
    # A gap is a signal, NOT a tool execution — it must never land in the
    # 'production' bucket the success-rate SLO / silence canary read.
    assert row.telemetry_source == "synthetic_signal"
    assert row.telemetry_source != "production"


def test_gap_write_never_raises_and_rolls_back_on_error():
    """A failing gap write must roll back + close its own session and never
    propagate — telemetry must never fail the turn that recorded the gap."""
    class _Boom(_FakeSession):
        def commit(self):
            raise RuntimeError("db down")

    sess = _Boom()
    # Must not raise.
    asyncio.run(write_tool_gap(
        query=_VECTOR_QUERY, workspace_id=None, gap_source="no_tool_call",
        session_factory=lambda: sess,
    ))
    assert sess.closed  # cleaned up despite the failure


def test_no_tool_call_gap_source_shape():
    sess = _FakeSession()
    asyncio.run(write_tool_gap(
        query="do something", workspace_id=None, gap_source="no_tool_call",
        session_factory=lambda: sess,
    ))
    row = sess.added[0]
    assert row.router_decision["gap_source"] == "no_tool_call"
    assert row.status == "gap"


# ===========================================================================
# (c) the nightly gap→resolution join
# ===========================================================================
def test_gap_resolution_produces_cluster_action_affinity():
    """A gap, then a successful real action in the SAME conversation, becomes a
    succeeds_for_intent(resolving_action, gap's_cluster)."""
    logs = [
        _log(_TOOL_GAP_ACTION, "gap", conv="c1", offset_s=0, query=_VECTOR_QUERY),
        _log("platform_update_task_status", "success", conv="c1", offset_s=60),
    ]
    cluster_map = {0: 42}  # the gap row's query clusters to 42
    affs = _compute_gap_resolution_affinities(logs, cluster_map)
    assert len(affs) == 1
    a = affs[0]
    assert a["action_name"] == "platform_update_task_status"
    assert a["affinity_type"] == "succeeds_for_intent"
    assert a["intent_cluster_id"] == 42
    assert a["sample_count"] == 1
    assert a["confidence"] == pytest.approx(wilson_lower_bound(1, 1))


def test_gap_resolution_requires_same_conversation():
    """A success in a DIFFERENT conversation does not resolve the gap."""
    logs = [
        _log(_TOOL_GAP_ACTION, "gap", conv="c1", offset_s=0),
        _log("platform_update_task_status", "success", conv="c2", offset_s=60),
    ]
    assert _compute_gap_resolution_affinities(logs, {0: 42}) == []


def test_gap_resolution_respects_the_window():
    """A success beyond the window (25h later) does not resolve the gap."""
    logs = [
        _log(_TOOL_GAP_ACTION, "gap", conv="c1", offset_s=0),
        _log("platform_update_task_status", "success", conv="c1", offset_s=25 * 3600),
    ]
    assert _compute_gap_resolution_affinities(logs, {0: 42}, window=timedelta(hours=24)) == []
    # Widen the window and it resolves.
    assert len(_compute_gap_resolution_affinities(logs, {0: 42}, window=timedelta(hours=26))) == 1


def test_gap_resolution_credits_only_the_first_serving_action():
    """The FIRST successful action after the gap is the one that served it."""
    logs = [
        _log(_TOOL_GAP_ACTION, "gap", conv="c1", offset_s=0),
        _log("platform_list_tasks", "success", conv="c1", offset_s=30),
        _log("platform_update_task_status", "success", conv="c1", offset_s=60),
    ]
    affs = _compute_gap_resolution_affinities(logs, {0: 7})
    assert len(affs) == 1
    assert affs[0]["action_name"] == "platform_list_tasks"


def test_gap_with_no_cluster_is_skipped():
    logs = [
        _log(_TOOL_GAP_ACTION, "gap", conv="c1", offset_s=0),
        _log("platform_update_task_status", "success", conv="c1", offset_s=60),
    ]
    assert _compute_gap_resolution_affinities(logs, {}) == []  # gap not clustered


# ===========================================================================
# (b) shown-not-used decay
# ===========================================================================
def test_shown_not_used_decay_erodes_weight_but_holds_the_floor():
    """A succeeds_for_intent shown far more than it is used loses weight, but the
    decay never drops it below the floor."""
    aff = _succeeds_aff("A", cluster=5, weight=5.0)
    logs = []
    # A used twice in cluster 5
    for i in range(2):
        logs.append(_log("A", "success", conv="c", offset_s=i))
    # A shown 20 times in cluster 5 (excess 18)
    for i in range(20):
        logs.append(_log(_TOOL_SHOWN_ACTION, "shown", conv="c", offset_s=100 + i, shown=["A"]))
    cluster_map = {i: 5 for i in range(len(logs))}

    out, n = _apply_shown_not_used_decay([aff], logs, cluster_map, decay_factor=0.9, floor=0.5)
    assert n == 1
    assert out[0]["weight"] < 5.0                 # decayed
    assert out[0]["weight"] >= 0.5                # floored
    assert out[0]["weight"] == pytest.approx(max(0.5, 5.0 * 0.9 ** 18))
    # input not mutated
    assert aff["weight"] == 5.0


def test_decay_clamps_at_the_floor_under_heavy_excess():
    aff = _succeeds_aff("A", cluster=5, weight=5.0)
    logs = [_log(_TOOL_SHOWN_ACTION, "shown", conv="c", offset_s=i, shown=["A"]) for i in range(200)]
    cluster_map = {i: 5 for i in range(len(logs))}
    out, _n = _apply_shown_not_used_decay([aff], logs, cluster_map, decay_factor=0.9, floor=0.5)
    assert out[0]["weight"] == pytest.approx(0.5)  # 5 * 0.9**200 → clamped


def test_no_decay_when_used_at_least_as_often_as_shown():
    aff = _succeeds_aff("A", cluster=5, weight=5.0)
    logs = [_log("A", "success", conv="c", offset_s=i) for i in range(10)]
    logs += [_log(_TOOL_SHOWN_ACTION, "shown", conv="c", offset_s=100 + i, shown=["A"]) for i in range(5)]
    cluster_map = {i: 5 for i in range(len(logs))}
    out, n = _apply_shown_not_used_decay([aff], logs, cluster_map)
    assert n == 0
    assert out[0] is aff  # untouched (same object)


def test_decay_leaves_non_succeeds_affinities_alone():
    fails = {
        "action_name": "A", "affinity_type": "fails_for_intent", "workspace_id": "ws1",
        "agent_id": None, "intent_cluster_id": 5, "weight": 3.0, "confidence": 0.4, "sample_count": 3,
    }
    logs = [_log(_TOOL_SHOWN_ACTION, "shown", conv="c", offset_s=i, shown=["A"]) for i in range(10)]
    out, n = _apply_shown_not_used_decay([fails], logs, {i: 5 for i in range(len(logs))})
    assert n == 0 and out[0] is fails


# ===========================================================================
# merge — gap resolution reinforces, never clobbers, an organic affinity
# ===========================================================================
def test_merge_reinforces_matching_intent_affinity():
    base = [_succeeds_aff("A", cluster=5, weight=3.0)]
    extra = [_succeeds_aff("A", cluster=5, weight=1.0)]  # a gap resolution
    merged = _merge_affinities(base, extra)
    assert len(merged) == 1
    assert merged[0]["sample_count"] == 4
    assert merged[0]["weight"] == 4.0
    assert merged[0]["confidence"] == pytest.approx(wilson_lower_bound(4, 4))


def test_merge_keeps_distinct_cluster_keys_separate():
    merged = _merge_affinities(
        [_succeeds_aff("A", cluster=5, weight=3.0)],
        [_succeeds_aff("A", cluster=6, weight=1.0)],  # different intent cluster
    )
    keys = {(m["action_name"], m["intent_cluster_id"]) for m in merged}
    assert keys == {("A", 5), ("A", 6)}


# ===========================================================================
# build_edges wiring — AC2 "in a build_edges unit run"
# ===========================================================================
def test_build_edges_wires_gap_resolution(monkeypatch):
    """A faked build_edges run (no DB, no embeddings) proves the gap→resolution
    affinity actually reaches _upsert_affinities and is counted."""
    import core.services.edge_builder as eb

    logs = [
        _log(_TOOL_GAP_ACTION, "gap", conv="c1", offset_s=0, query=_VECTOR_QUERY),
        _log("platform_update_task_status", "success", conv="c1", offset_s=60),
    ]

    @contextmanager
    def _fake_session():
        yield object()

    async def _fake_clusters(db, logs_):
        return {0: 42}  # the gap row (index 0) → cluster 42

    captured = {}
    monkeypatch.setattr(eb, "get_db_session", _fake_session)
    monkeypatch.setattr(eb, "_load_logs", lambda db, cutoff, workspace_id=None: logs)
    monkeypatch.setattr(eb, "_upsert_edges", lambda db, d: 0)
    monkeypatch.setattr(eb, "_upsert_failed_after_edges", lambda db, d: 0)
    monkeypatch.setattr(eb, "_compute_and_upsert_clusters", _fake_clusters)
    monkeypatch.setattr(eb, "_upsert_affinities",
                        lambda db, affs: captured.setdefault("affs", affs) or len(affs))

    summary = asyncio.run(eb.build_edges())

    assert summary.gap_resolutions_built == 1
    keys = {(a["action_name"], a["intent_cluster_id"]) for a in captured["affs"]}
    assert ("platform_update_task_status", 42) in keys


# ===========================================================================
# (b) recorder — shown-set persists batched, one session per flush
# ===========================================================================
class _Result:
    def __init__(self, rowcount):
        self.rowcount = rowcount


class _CapturingDB:
    def __init__(self, update_rowcount=0):
        self.executed = []
        self._u = update_rowcount

    def execute(self, stmt, params=None):
        sql = str(stmt)
        self.executed.append((sql, params or {}))
        return _Result(self._u if sql.strip().upper().startswith("UPDATE") else 1)

    def flush(self):
        pass


class _SessionFactory:
    def __init__(self, update_rowcount=0):
        self.db = _CapturingDB(update_rowcount)
        self.enter_count = 0

    @contextmanager
    def session(self):
        self.enter_count += 1
        yield self.db


def _wilson_real(s, t, z=1.96):
    import math
    if t == 0:
        return 0.0
    p = s / t
    denom = 1 + z ** 2 / t
    centre = p + z ** 2 / (2 * t)
    spread = z * math.sqrt((p * (1 - p) + z ** 2 / (4 * t)) / t)
    return (centre - spread) / denom


def _recorder_with_fake_db(monkeypatch, update_rowcount=0):
    factory = _SessionFactory(update_rowcount)
    fake_db = types.ModuleType("core.database.database")
    fake_db.get_db_session = factory.session
    monkeypatch.setitem(sys.modules, "core.database.database", fake_db)
    fake_eb = types.ModuleType("core.services.edge_builder")
    fake_eb.wilson_lower_bound = _wilson_real
    monkeypatch.setitem(sys.modules, "core.services.edge_builder", fake_eb)
    return ToolSignalRecorder(), factory


def _shown_stmts(db):
    return [(s, p) for s, p in db.executed if "tool_execution_logs" in s]


def test_record_selection_persists_shown_row_batched(monkeypatch):
    """A SelectionSignal flushes to exactly one __tool_shown__ row carrying the
    query + the surfaced candidate set — in ONE session."""
    recorder, factory = _recorder_with_fake_db(monkeypatch)
    asyncio.run(recorder._flush([
        SelectionSignal(
            query=_VECTOR_QUERY,
            shown_actions=("platform_update_task_status", "platform_list_tasks"),
            workspace_id="ws1", agent_id=3,
        )
    ]))
    assert factory.enter_count == 1
    shown = _shown_stmts(factory.db)
    assert len(shown) == 1
    _sql, params = shown[0]
    assert params["action"] == _TOOL_SHOWN_ACTION == "__tool_shown__"
    assert params["user_query"] == _VECTOR_QUERY
    assert "platform_update_task_status" in json.loads(params["router"])["candidates"]
    # NOT 'production' — a shown row must not pollute the success-rate SLO / canary.
    assert params["source"] == "synthetic_signal"
    assert params["source"] != "production"


def test_mixed_batch_uses_one_session(monkeypatch):
    """ToolSignals (edges/affinities) and SelectionSignals (shown rows) in one
    batch share a SINGLE DB session — the PRD-141 US-019 one-session contract."""
    recorder, factory = _recorder_with_fake_db(monkeypatch)
    asyncio.run(recorder._flush([
        ToolSignal("b", True, agent_id=1, workspace_id="ws", prior_action="a"),
        SelectionSignal(query="q", shown_actions=("x",), workspace_id="ws", agent_id=1),
    ]))
    assert factory.enter_count == 1
    sqls = [s for s, _ in factory.db.executed]
    assert any("tool_execution_logs" in s for s in sqls)   # shown row
    assert any("tool_routing_edges" in s for s in sqls)    # edge from the ToolSignal


def test_record_selection_shown_enqueue_requires_query_and_narrowed_set(monkeypatch):
    """No query or no narrowed set → nothing durable is enqueued (only the
    in-memory stash), so the full non-narrowed catalog never floods 'shown'."""
    monkeypatch.setitem(sys.modules, "config", types.ModuleType("config"))
    sys.modules["config"].config = types.SimpleNamespace(
        TOOL_SIGNAL_RECORDER_ENABLED=True, TOOL_SELECTION_STASH_MAXSIZE=512,
    )
    recorder = ToolSignalRecorder()

    async def _drive():
        # narrowed but NO query → no durable enqueue
        recorder.record_selection(workspace_id="ws", agent_id=1, narrowed=True,
                                  allowed_names=["a", "b"], query=None)
        # query but NOT narrowed (allowed_names None) → no durable enqueue
        recorder.record_selection(workspace_id="ws", agent_id=1, narrowed=False,
                                  allowed_names=None, query="q")
        return recorder._queue

    q = asyncio.run(_drive())
    assert q is None or q.qsize() == 0


def test_record_selection_enqueues_when_narrowed_with_query(monkeypatch):
    monkeypatch.setitem(sys.modules, "config", types.ModuleType("config"))
    sys.modules["config"].config = types.SimpleNamespace(
        TOOL_SIGNAL_RECORDER_ENABLED=True, TOOL_SELECTION_STASH_MAXSIZE=512,
        TOOL_SIGNAL_QUEUE_MAXSIZE=10000,
    )
    recorder = ToolSignalRecorder()

    async def _drive():
        recorder.record_selection(workspace_id="ws", agent_id=1, narrowed=True,
                                  allowed_names=["platform_update_task_status"], query="close tickets")
        return recorder._queue.qsize()

    assert asyncio.run(_drive()) == 1


# ===========================================================================
# markers agree across writer + reader; no new table
# ===========================================================================
def test_gap_shown_markers_agree_across_modules():
    """The wire-protocol marker strings must match between the writers
    (telemetry gap, recorder shown) and the reader (edge_builder), and both
    writers must tag the SAME distinct (non-production) telemetry_source."""
    assert _telemetry.TOOL_GAP_ACTION == _TOOL_GAP_ACTION == "__tool_gap__"
    assert _signal_recorder._TOOL_SHOWN_ACTION == _TOOL_SHOWN_ACTION == "__tool_shown__"
    # The synthetic-signal source is a wire protocol shared by both writers and is
    # NOT 'production' (SLO/canary) nor 'synthetic' (seed_telemetry deletes those).
    assert _telemetry.SYNTHETIC_SIGNAL_SOURCE == _signal_recorder._SYNTHETIC_SIGNAL_SOURCE
    assert _telemetry.SYNTHETIC_SIGNAL_SOURCE not in ("production", "synthetic")


def test_find_tools_gap_covers_the_platform_execute_shape():
    """Trigger 1 fires whether find_tools is called directly OR via the
    platform_execute meta-dispatcher (the closed-pins fallback routing shape)."""
    src = (_ORCH / "modules" / "tools" / "execution" / "unified_executor.py").read_text()
    assert "_is_find_tools" in src
    assert 'tool_name == "platform_find_tools"' in src           # direct promoted call
    assert '.get("action") == "platform_find_tools"' in src      # via platform_execute enum


def test_no_new_table_created():
    """US-011 uses only the EXISTING tool_execution_logs + tool_routing_affinities
    tables — the gap/shown writers must never CREATE a table."""
    src = (
        (_ORCH / "modules" / "tools" / "execution" / "telemetry.py").read_text()
        + (_ORCH / "modules" / "tools" / "discovery" / "signal_recorder.py").read_text()
        + (_ORCH / "core" / "services" / "edge_builder.py").read_text()
    )
    assert "CREATE TABLE" not in src.upper()
    # the gap/shown rows land on the existing lane
    assert "tool_execution_logs" in src
