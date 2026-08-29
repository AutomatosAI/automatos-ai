"""PRD-228 US-001 — the fleet read-model service (pure tests).

These run with no database: the derivation/shape logic lives in the pure
``_assemble_fleet`` and is exercised with plain row objects, and the bounded
query set is proven with a counting session that records ``.query()`` calls
without a live DB. End-to-end behaviour against real Postgres lives in
``test_prd228_fleet_state_realdb.py`` (``@integration``, skips cleanly locally).

Covers the US-001 acceptance criteria:
  * parametrized fixture shapes (leased board task / running mission task /
    idle / blocked-with-open-ask),
  * query-count assertion (no per-agent N+1),
  * fail-soft cost (omitted, response still 200-shaped),
  * read-only service (grep for session mutation calls),
  * cost-source pin recorded in the service.
"""
from __future__ import annotations

import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
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

import services.fleet_state as fs  # noqa: E402
from services.fleet_state import get_fleet_state  # noqa: E402

NOW = datetime(2026, 8, 28, 12, 0, 0, tzinfo=timezone.utc)


def _mins(n: int) -> datetime:
    return NOW - timedelta(minutes=n)


# ---------------------------------------------------------------------------
# Row factories — mimic only the attributes the assembler reads.
# ---------------------------------------------------------------------------

def _agent(agent_id: int, name: str = "Agent"):
    return SimpleNamespace(id=agent_id, name=name)


def _board(
    task_id, agent_id, *, status="in_progress", title="Board task",
    started_at=None, updated_at=None, completed_at=None, blocked_at=None,
):
    return SimpleNamespace(
        id=task_id, assigned_agent_id=agent_id, status=status, title=title,
        started_at=started_at, updated_at=updated_at,
        completed_at=completed_at, blocked_at=blocked_at,
    )


def _orch(
    task_id, agent_id, *, state="running", title="Mission task",
    started_at=None, updated_at=None, completed_at=None,
):
    return SimpleNamespace(
        id=task_id, assigned_agent_id=agent_id, state=state, title=title,
        started_at=started_at, updated_at=updated_at, completed_at=completed_at,
    )


def _watch(watch_id, *, owner_agent_id=None, target_type="board_task",
           target_id="0", status="watching"):
    return SimpleNamespace(
        id=watch_id, owner_agent_id=owner_agent_id,
        target_type=target_type, target_id=target_id, status=status,
    )


def _grant(
    grant_id, *, agent_id=None, asked_by_agent_id=None,
    subject_type="board_task", subject_id="0",
):
    return SimpleNamespace(
        id=grant_id, agent_id=agent_id, asked_by_agent_id=asked_by_agent_id,
        subject_type=subject_type, subject_id=subject_id,
    )


_UNSET = object()


def _assemble(agents, board=None, orch=None, watches=None, asks=None, costs=_UNSET):
    """Call the pure assembler with sensible empty defaults.

    ``costs`` defaults to ``{}`` (source available, zero usage) so ``cost_24h``
    is present; pass ``None`` explicitly to model an unavailable cost source.
    """
    if costs is _UNSET:
        costs = {}
    return fs._assemble_fleet(
        agents, board or [], orch or [], watches or [], asks or [],
        costs, generated_at=NOW,
    )


def _assert_documented_shape(entry, *, cost_expected=True):
    assert {
        "agent_id", "name", "current", "queue_depth",
        "blocked", "watches", "last_activity_at",
    }.issubset(entry)
    assert set(entry["blocked"]) == {"count", "open_asks"}
    assert isinstance(entry["blocked"]["open_asks"], list)
    assert set(entry["watches"]) == {"active", "needs_attention"}
    if entry["current"] is not None:
        assert set(entry["current"]) == {"kind", "id", "title", "since"}
        assert entry["current"]["kind"] in {"board_task", "mission_task"}
    if cost_expected:
        assert set(entry["cost_24h"]) == {"tokens", "usd"}
    else:
        assert "cost_24h" not in entry


# ===========================================================================
# 1. Parametrized fixture shapes (the four documented states)
# ===========================================================================

def _scenario_leased_board_task():
    agents = [_agent(1, "Builder")]
    board = [_board(7, 1, status="in_progress", title="Ship the widget",
                    started_at=_mins(5), updated_at=_mins(1))]
    out = _assemble(agents, board=board)
    entry = out["agents"][0]
    assert entry["current"]["kind"] == "board_task"
    assert entry["current"]["id"] == 7
    assert entry["current"]["title"] == "Ship the widget"
    assert entry["current"]["since"] == _mins(5).isoformat()
    assert entry["queue_depth"] == 0
    assert entry["blocked"] == {"count": 0, "open_asks": []}
    return entry


def _scenario_running_mission_task():
    agents = [_agent(1, "Researcher")]
    orch = [_orch("uuid-abc", 1, state="running", title="Draft the brief",
                  started_at=_mins(3), updated_at=_mins(1))]
    out = _assemble(agents, orch=orch)
    entry = out["agents"][0]
    assert entry["current"]["kind"] == "mission_task"
    assert entry["current"]["id"] == "uuid-abc"
    assert entry["current"]["title"] == "Draft the brief"
    return entry


def _scenario_idle():
    out = _assemble([_agent(1, "Bench")])
    entry = out["agents"][0]
    assert entry["current"] is None
    assert entry["queue_depth"] == 0
    assert entry["blocked"] == {"count": 0, "open_asks": []}
    assert entry["watches"] == {"active": 0, "needs_attention": 0}
    assert entry["last_activity_at"] is None
    return entry


def _scenario_blocked_with_open_ask():
    agents = [_agent(1, "Stuck")]
    board = [_board(7, 1, status="in_progress", title="Blocked one",
                    started_at=_mins(9), updated_at=_mins(2), blocked_at=_mins(1))]
    asks = [_grant(99, subject_type="board_task", subject_id="7")]
    out = _assemble(agents, board=board, asks=asks)
    entry = out["agents"][0]
    assert entry["blocked"]["count"] == 1
    assert entry["blocked"]["open_asks"] == [99]
    return entry


@pytest.mark.parametrize(
    "scenario",
    [
        _scenario_leased_board_task,
        _scenario_running_mission_task,
        _scenario_idle,
        _scenario_blocked_with_open_ask,
    ],
    ids=["leased_board_task", "running_mission_task", "idle", "blocked_with_open_ask"],
)
def test_documented_shape_per_state(scenario):
    entry = scenario()
    _assert_documented_shape(entry, cost_expected=True)


# ===========================================================================
# 2. Derivation details
# ===========================================================================

def test_board_task_wins_over_running_mission():
    # Both present: board task (leased/in_progress) is the current work.
    agents = [_agent(1)]
    board = [_board(7, 1, status="in_progress", title="Board wins", started_at=_mins(4))]
    orch = [_orch("m1", 1, state="running", title="Mission", started_at=_mins(2))]
    entry = _assemble(agents, board=board, orch=orch)["agents"][0]
    assert entry["current"]["kind"] == "board_task"
    assert entry["current"]["title"] == "Board wins"


def test_queue_depth_counts_assigned_not_started():
    agents = [_agent(1)]
    board = [
        _board(1, 1, status="in_progress", started_at=_mins(3)),
        _board(2, 1, status="assigned"),
        _board(3, 1, status="assigned"),
        _board(4, 1, status="review"),   # not "assigned" → not queued
    ]
    entry = _assemble(agents, board=board)["agents"][0]
    assert entry["queue_depth"] == 2
    assert entry["current"]["kind"] == "board_task"


def test_watches_attributed_by_owner_and_target_deduped():
    agents = [_agent(1)]
    board = [_board(7, 1, status="in_progress", started_at=_mins(3))]
    watches = [
        _watch("w-owned", owner_agent_id=1, target_type="mission", target_id="x"),
        _watch("w-target", owner_agent_id=None, target_type="board_task", target_id="7"),
        # Owned AND targets the same board task → must count once, not twice.
        _watch("w-both", owner_agent_id=1, target_type="board_task", target_id="7"),
        _watch("w-other", owner_agent_id=2, target_type="board_task", target_id="99"),
    ]
    entry = _assemble(agents, board=board, watches=watches)["agents"][0]
    assert entry["watches"] == {"active": 3, "needs_attention": 0}


def test_watches_needs_attention_counted():
    # A watch that hit its action budget (needs_attention) is the over-budget signal.
    agents = [_agent(1)]
    board = [_board(7, 1, status="in_progress", started_at=_mins(3))]
    watches = [
        _watch("w-ok", owner_agent_id=1, status="watching"),
        _watch("w-stuck", owner_agent_id=1, status="needs_attention"),
    ]
    entry = _assemble(agents, board=board, watches=watches)["agents"][0]
    assert entry["watches"] == {"active": 2, "needs_attention": 1}


def test_open_asks_attributed_by_agent_and_by_subject():
    agents = [_agent(1)]
    board = [_board(7, 1, status="in_progress", started_at=_mins(3))]
    asks = [
        _grant(10, agent_id=1, subject_type="tool_call", subject_id="abc"),
        _grant(11, subject_type="board_task", subject_id="7"),
        _grant(12, asked_by_agent_id=1, subject_type="playbook_run", subject_id="p"),
        _grant(13, agent_id=2, subject_type="board_task", subject_id="99"),  # other agent
    ]
    entry = _assemble(agents, board=board, asks=asks)["agents"][0]
    assert sorted(entry["blocked"]["open_asks"]) == [10, 11, 12]


def test_last_activity_is_latest_timestamp():
    agents = [_agent(1)]
    board = [_board(1, 1, status="review", started_at=_mins(30),
                    updated_at=_mins(10), completed_at=None)]
    orch = [_orch("m1", 1, state="assigned", started_at=_mins(20), updated_at=_mins(2))]
    entry = _assemble(agents, board=board, orch=orch)["agents"][0]
    assert entry["last_activity_at"] == _mins(2).isoformat()


def test_multi_agent_no_cross_attribution():
    agents = [_agent(1, "One"), _agent(2, "Two")]
    board = [
        _board(1, 1, status="in_progress", title="A1", started_at=_mins(3)),
        _board(2, 2, status="assigned"),
    ]
    out = _assemble(agents, board=board)
    by_id = {e["agent_id"]: e for e in out["agents"]}
    assert by_id[1]["current"]["title"] == "A1"
    assert by_id[1]["queue_depth"] == 0
    assert by_id[2]["current"] is None
    assert by_id[2]["queue_depth"] == 1


# ===========================================================================
# 3. Cost fields + fail-soft
# ===========================================================================

def test_cost_attached_when_source_available():
    agents = [_agent(1), _agent(2)]
    costs = {1: {"tokens": 1200, "usd": 0.34}}
    out = _assemble(agents, costs=costs)
    by_id = {e["agent_id"]: e for e in out["agents"]}
    assert by_id[1]["cost_24h"] == {"tokens": 1200, "usd": 0.34}
    # Agent with no usage still gets a zeroed cost when the source is available.
    assert by_id[2]["cost_24h"] == {"tokens": 0, "usd": 0.0}
    assert out["cost_available"] is True
    assert out["cost_source"] == "llm_usage"


def test_cost_omitted_when_source_unavailable():
    agents = [_agent(1)]
    out = fs._assemble_fleet(agents, [], [], [], [], None, generated_at=NOW)
    assert "cost_24h" not in out["agents"][0]
    assert out["cost_available"] is False
    assert out["cost_source"] is None
    _assert_documented_shape(out["agents"][0], cost_expected=False)


def test_safe_cost_returns_none_on_failure():
    class _Boom:
        def query(self, *a, **k):
            raise RuntimeError("cost lane down")

    assert fs._safe_cost(_Boom(), uuid4(), [1, 2], NOW) is None


# ===========================================================================
# 4. Bounded query set — no per-agent N+1
# ===========================================================================

class _CountingQuery:
    def __init__(self, session, is_agent):
        self._session = session
        self._is_agent = is_agent

    def filter(self, *a, **k):
        return self

    def order_by(self, *a, **k):
        return self

    def group_by(self, *a, **k):
        return self

    def distinct(self, *a, **k):
        return self

    def all(self):
        return list(self._session.agents) if self._is_agent else []

    def first(self):
        return None

    def one(self):
        return None


class _CountingSession:
    """Records how many ``.query()`` calls a full fleet build issues.

    Only the ``Agent`` query returns rows (the roster); every other source
    returns empty. The count must not grow with the agent roster.
    """

    def __init__(self, agents):
        self.agents = agents
        self.query_count = 0

    def query(self, *entities):
        from core.models.core import Agent

        self.query_count += 1
        is_agent = bool(entities) and entities[0] is Agent
        return _CountingQuery(self, is_agent)


def test_query_count_bounded_no_n_plus_1():
    ws = uuid4()

    def run(n):
        agents = [_agent(i, f"A{i}") for i in range(1, n + 1)]
        sess = _CountingSession(agents)
        result = get_fleet_state(sess, ws)
        return sess.query_count, result

    count_1, res_1 = run(1)
    count_25, res_25 = run(25)

    assert count_1 == count_25, (
        f"query count scales with agents ({count_1} vs {count_25}) — N+1 regression"
    )
    # One query per source: agents, board, mission, watches, asks, cost.
    assert count_25 <= 8, f"expected a bounded query set, got {count_25}"
    assert len(res_25["agents"]) == 25


def test_get_fleet_state_failsoft_cost_still_200_shaped(monkeypatch):
    """A cost-source failure omits cost fields; the response is still whole."""
    ws = uuid4()
    agents = [_agent(1, "A1"), _agent(2, "A2")]
    sess = _CountingSession(agents)

    def _boom(*a, **k):
        raise RuntimeError("llm_usage unreachable")

    monkeypatch.setattr(fs, "_cost_by_agent", _boom)
    out = get_fleet_state(sess, ws)

    assert out["cost_available"] is False
    assert out["cost_source"] is None
    assert len(out["agents"]) == 2
    for entry in out["agents"]:
        _assert_documented_shape(entry, cost_expected=False)


# ===========================================================================
# 4b. Fail-soft on the omittable enrichment sources (P228-RVW-2)
# ===========================================================================

def test_get_fleet_state_failsoft_watches_still_200_shaped(monkeypatch):
    """A watches-source failure defaults the watch block; response stays whole.

    Every agent is present with cost + current + blocked intact — only the
    watches field degrades to its zeroed default (kept, not dropped).
    """
    ws = uuid4()
    agents = [_agent(1, "A1"), _agent(2, "A2")]
    sess = _CountingSession(agents)

    def _boom(*a, **k):
        raise RuntimeError("watches table locked")

    monkeypatch.setattr(fs, "_watches_source", _boom)
    out = get_fleet_state(sess, ws)

    assert len(out["agents"]) == 2
    assert out["cost_available"] is True
    for entry in out["agents"]:
        assert entry["watches"] == {"active": 0, "needs_attention": 0}
        _assert_documented_shape(entry, cost_expected=True)


def test_get_fleet_state_failsoft_asks_still_200_shaped(monkeypatch):
    """An asks-source failure defaults open_asks to []; response stays whole."""
    ws = uuid4()
    agents = [_agent(1, "A1"), _agent(2, "A2")]
    sess = _CountingSession(agents)

    def _boom(*a, **k):
        raise RuntimeError("approval_grants unavailable")

    monkeypatch.setattr(fs, "_asks_source", _boom)
    out = get_fleet_state(sess, ws)

    assert len(out["agents"]) == 2
    assert out["cost_available"] is True
    for entry in out["agents"]:
        assert entry["blocked"]["open_asks"] == []
        _assert_documented_shape(entry, cost_expected=True)


def test_safe_watches_and_safe_asks_return_none_on_failure():
    class _Boom:
        def query(self, *a, **k):
            raise RuntimeError("source down")

    assert fs._safe_watches(_Boom(), uuid4()) is None
    assert fs._safe_asks(_Boom(), uuid4()) is None


def test_assembler_defaults_when_watches_or_asks_unavailable():
    """watches=None → zeroed block; asks=None → open_asks [] (count from board)."""
    agents = [_agent(1)]
    board = [_board(7, 1, status="in_progress", started_at=_mins(3),
                    blocked_at=_mins(1))]
    out = fs._assemble_fleet(agents, board, [], None, None, {}, generated_at=NOW)
    entry = out["agents"][0]
    assert entry["watches"] == {"active": 0, "needs_attention": 0}
    # blocked.count still reflects the board source; only open_asks defaults.
    assert entry["blocked"] == {"count": 1, "open_asks": []}
    _assert_documented_shape(entry, cost_expected=True)


def test_empty_workspace_returns_empty_fleet():
    sess = _CountingSession([])
    out = get_fleet_state(sess, uuid4())
    assert out["agents"] == []
    assert out["version"] == fs.FLEET_STATE_VERSION


# ===========================================================================
# 4c. Source-availability flags — degraded ≠ genuine zero (P228-RVW-6)
# ===========================================================================

def test_availability_flags_true_when_sources_healthy():
    """P228-RVW-6: healthy watches/asks sources → both availability flags True
    (mirroring cost_available)."""
    out = fs._assemble_fleet([_agent(1)], [], [], [], [], {}, generated_at=NOW)
    assert out["watches_available"] is True
    assert out["asks_available"] is True


def test_watches_available_false_when_source_fails(monkeypatch):
    """P228-RVW-6: a watches-source failure sets watches_available False — a
    defaulted zero distinguishable from a real zero — while the response stays
    whole (every agent present, cost + asks intact)."""
    ws = uuid4()
    agents = [_agent(1, "A1"), _agent(2, "A2")]
    sess = _CountingSession(agents)

    def _boom(*a, **k):
        raise RuntimeError("watches table locked")

    monkeypatch.setattr(fs, "_watches_source", _boom)
    out = get_fleet_state(sess, ws)

    assert out["watches_available"] is False   # degradation is observable
    assert out["asks_available"] is True       # the other source is unaffected
    assert out["cost_available"] is True
    assert len(out["agents"]) == 2
    for entry in out["agents"]:
        assert entry["watches"] == {"active": 0, "needs_attention": 0}
        _assert_documented_shape(entry, cost_expected=True)


def test_asks_available_false_when_source_fails(monkeypatch):
    """P228-RVW-6: an asks-source failure sets asks_available False while the
    response stays whole (every agent present, cost + watches intact)."""
    ws = uuid4()
    agents = [_agent(1, "A1"), _agent(2, "A2")]
    sess = _CountingSession(agents)

    def _boom(*a, **k):
        raise RuntimeError("approval_grants unavailable")

    monkeypatch.setattr(fs, "_asks_source", _boom)
    out = get_fleet_state(sess, ws)

    assert out["asks_available"] is False
    assert out["watches_available"] is True
    assert out["cost_available"] is True
    assert len(out["agents"]) == 2
    for entry in out["agents"]:
        assert entry["blocked"]["open_asks"] == []
        _assert_documented_shape(entry, cost_expected=True)


# ===========================================================================
# 5. Structural invariants (read-only + cost-source pin)
# ===========================================================================

_SVC_SRC = Path(fs.__file__).read_text(encoding="utf-8")


def test_service_performs_no_writes():
    """The read-model must contain zero session mutation calls."""
    hits = re.findall(r"\.(?:add|delete|commit|flush)\(", _SVC_SRC)
    assert not hits, f"read-only service has write-like calls: {hits}"


def test_cost_source_is_pinned_in_source():
    lowered = _SVC_SRC.lower()
    assert "cost source" in lowered or "canonical" in lowered
    assert "llm_usage" in lowered


def test_agent_population_mirrors_roster_exclusions():
    """P228-RVW-1: the agent query hides the two categories the canonical roster
    hides from a per-workspace view — Mission Zero ephemeral clones and the
    per-workspace system agent (Auto). Structural guard against silent removal;
    behavioural proof is in the realdb suite (``test_roster_hidden_agents_excluded``).
    (Admin/super_admin parity is a separate matter — see P228-RVW-3.)
    """
    # Ephemeral clones excluded (mirrors api/agents.py:538).
    assert re.search(r'agent_type\s*!=\s*"ephemeral"', _SVC_SRC)
    # Per-workspace system agent (Auto) excluded (mirrors api/agents.py:545-547).
    assert "~and_(" in _SVC_SRC
    assert "Agent.is_system_agent.is_(True)" in _SVC_SRC
    assert "Agent.workspace_id.isnot(None)" in _SVC_SRC


def test_agent_population_docstring_is_honest_about_admin_global_system_agents():
    """P228-RVW-3: the fleet is the workspace's OWN floor, not an exact copy of
    every roster viewer's tab. For admin/super_admin callers api/agents.py ALSO
    surfaces GLOBAL system agents (workspace_id=None, e.g. Auto CTO) via its
    workspace-OR-system scope; the fleet omits those (they own no per-workspace
    floor state). The docs must NOT claim exact / same-set parity, and must
    record the intentional divergence so a future reader reads it as deliberate,
    not an oversight. Behavioural proof is in the realdb suite
    (``test_global_system_agent_absent_from_fleet``).
    """
    lowered = _SVC_SRC.lower()
    # The overstated claims are gone.
    assert "mirrors the canonical roster exactly" not in lowered
    assert "show the same set" not in lowered
    # The real scope + the intentional global-system-agent divergence are documented.
    assert "workspace_id=none" in lowered          # the global persona shape
    assert "auto cto" in lowered                    # the concrete live example
    assert "per-workspace floor state" in lowered   # why it is omitted
    assert "admin" in lowered                       # names who sees the divergence


def test_reuses_canonical_busy_derivation_no_rival():
    # The service imports the canonical busy states rather than defining its own,
    # and it is the SAME constant the dispatcher's matcher consumes.
    assert "from core.models.orchestration_enums import BUSY_TASK_STATES" in _SVC_SRC
    from core.models.orchestration_enums import BUSY_TASK_STATES as ENUM_BUSY
    from modules.coordination.agent_matcher import BUSY_TASK_STATES as MATCHER_BUSY

    assert fs.BUSY_TASK_STATES is ENUM_BUSY is MATCHER_BUSY
    assert fs.BUSY_TASK_STATES == ("assigned", "running")
