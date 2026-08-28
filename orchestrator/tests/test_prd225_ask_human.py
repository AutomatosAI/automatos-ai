"""PRD-225 US-002 — platform_ask_human: park, notify, return.

Pure tests over a fake session (the test_p2w2_ask_notification pattern): the
3-file registration, parking across all three subject types, the
``question_pending`` dispatch with ``link_type='question'``, the ≥3-cascade
urgent bypass (cycle-safe), server-minted asker id, and the no-wait guarantee.
"""
from __future__ import annotations

import os
import sys
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


# ---------------------------------------------------------------------------
# Fake session — parses equality filters off SQLAlchemy binary expressions.
# ---------------------------------------------------------------------------

class _Query:
    def __init__(self, rows):
        self._rows = list(rows)

    def filter(self, *conds):
        rows = self._rows
        for cond in conds:
            key = cond.left.key
            op = getattr(getattr(cond, "operator", None), "__name__", "")
            value = getattr(cond.right, "value", None)
            if op == "in_op":  # BoardTask.id.in_([...]) — the batched cascade load
                allowed = {str(v) for v in (value or [])}
                rows = [r for r in rows if str(getattr(r, key, None)) in allowed]
            else:
                rows = [r for r in rows if str(getattr(r, key, None)) == str(value)]
        return _Query(rows)

    def order_by(self, *a):
        return _Query(list(reversed(self._rows)))

    def first(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return list(self._rows)


class _FakeSession:
    def __init__(self):
        self.rows = []
        self.commits = 0

    def add(self, obj):
        if getattr(obj, "id", None) is None:
            obj.id = len(self.rows) + 1
        self.rows.append(obj)

    def flush(self):
        pass

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass

    def query(self, model):
        return _Query([r for r in self.rows if isinstance(r, model)])


@pytest.fixture()
def dispatched(monkeypatch):
    """Record every question_pending dispatch without touching the DB/network."""
    calls = []

    async def _record(self, **kwargs):
        calls.append(kwargs)
        return {"dispatched_to": ["in_app"]}

    monkeypatch.setattr(
        "core.services.notification_dispatcher.NotificationDispatcher.dispatch",
        _record,
    )
    return calls


def _board_task(ws, task_id, *, status="in_progress", parent=None):
    from core.models.core import BoardTask

    t = BoardTask(
        id=task_id, workspace_id=ws, title=f"T{task_id}",
        status=status, parent_task_id=parent,
    )
    return t


# ===========================================================================
# 1. Registration — the 3-file pattern
# ===========================================================================

def test_ask_human_registered_in_actions():
    from modules.tools.discovery.action_registry import ActionRegistry
    from modules.tools.discovery.actions_asks import register_asks_actions

    reg = ActionRegistry()
    register_asks_actions(reg)
    action = reg.get("platform_ask_human")
    assert action is not None
    assert action.permission_level == "write"
    props = action.parameters["properties"]
    assert set(action.parameters["required"]) == {"subject_type", "subject_id", "question"}
    # The asker id is server-minted, never a declared tool parameter.
    assert "_agent_id" not in props
    assert "asked_by_agent_id" not in props


def test_ask_human_wired_into_executor_map():
    """The executor routes the tool name to the handler (3rd file)."""
    from modules.tools.discovery.handlers_asks import ask_human
    from modules.tools.discovery import platform_executor as pe

    src = Path(pe.__file__).read_text()
    assert '"platform_ask_human": ask_human' in src


# ===========================================================================
# 2. Only board_task questions resume on answer — tool_call / playbook_run are
# refused up front (their answer would no-op the resume, P225-RVW-11).
# ===========================================================================

@pytest.mark.asyncio
async def test_tool_call_ask_resumes_or_is_refused(dispatched):
    """A tool_call question's answer cannot resume the parked call (no stored,
    re-dispatchable action), so platform_ask_human refuses it up front rather
    than reporting parked:true for work its answer cannot resume — the way a
    terminal board task is refused (P225-RVW-11 / P225-RVW-8). No grant row is
    staged and nothing is dispatched."""
    from modules.tools.discovery.handlers_asks import ask_human
    from core.models.approval_grants import ApprovalGrant

    db = _FakeSession()
    ws = uuid4()
    res = await ask_human(db, ws, {
        "subject_type": "tool_call", "subject_id": "call-xyz",
        "question": "Which vendor?", "_agent_id": 7,
    })
    assert res["success"] is False and res.get("parked") is False
    assert "tool_call" in res["error"]
    assert not any(isinstance(r, ApprovalGrant) for r in db.rows)
    assert dispatched == []


@pytest.mark.asyncio
async def test_playbook_run_ask_resumes_or_is_refused(dispatched):
    """Same for a playbook_run question — refused up front, no row, no dispatch
    (P225-RVW-11)."""
    from modules.tools.discovery.handlers_asks import ask_human
    from core.models.approval_grants import ApprovalGrant

    db = _FakeSession()
    ws = uuid4()
    res = await ask_human(db, ws, {
        "subject_type": "playbook_run", "subject_id": "run-abc",
        "question": "Which vendor?", "_agent_id": 7,
    })
    assert res["success"] is False and res.get("parked") is False
    assert "playbook_run" in res["error"]
    assert not any(isinstance(r, ApprovalGrant) for r in db.rows)
    assert dispatched == []


@pytest.mark.asyncio
async def test_board_task_parks_with_exact_reason(dispatched):
    from modules.tools.discovery.handlers_asks import ask_human
    from core.models.approval_grants import ApprovalGrant, KIND_QUESTION

    db = _FakeSession()
    ws = uuid4()
    task = _board_task(ws, 42, status="in_progress")
    db.add(task)

    res = await ask_human(db, ws, {
        "subject_type": "board_task", "subject_id": "42",
        "question": "Ship **A** or **B**?", "options": ["A", "B"], "_agent_id": 9,
    })
    assert res["success"] is True
    assert task.status == "blocked"
    assert task.blocked_reason == f"Awaiting human answer (ask #{res['ask_id']})"
    assert task.blocked_at is not None
    grant = next(r for r in db.rows if isinstance(r, ApprovalGrant))
    assert grant.kind == KIND_QUESTION and grant.options == ["A", "B"]
    assert db.commits >= 1  # durably parked


@pytest.mark.asyncio
async def test_board_task_not_found_is_rejected(dispatched):
    from modules.tools.discovery.handlers_asks import ask_human

    db = _FakeSession()
    res = await ask_human(db, uuid4(), {
        "subject_type": "board_task", "subject_id": "999",
        "question": "?", "_agent_id": 1,
    })
    assert res["success"] is False
    assert "not found" in res["error"]
    assert dispatched == []


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["done", "failed", "cancelled"])
async def test_terminal_board_task_is_rejected_not_parked(dispatched, terminal):
    """Asking against a finished task is refused honestly — no grant row, no
    'Parked' confirmation, no dispatch, status untouched. Answering later would
    no-op the resume yet claim it 'resumed', so the tool refuses up front
    (P225-RVW-8)."""
    from modules.tools.discovery.handlers_asks import ask_human
    from core.models.approval_grants import ApprovalGrant

    db = _FakeSession()
    ws = uuid4()
    task = _board_task(ws, 77, status=terminal)
    db.add(task)

    res = await ask_human(db, ws, {
        "subject_type": "board_task", "subject_id": "77",
        "question": "Ship it?", "_agent_id": 5,
    })

    assert res["success"] is False
    assert res.get("parked") is False
    assert terminal in res["error"]          # honest: names the finished state
    assert "Parked" not in res.get("error", "")
    assert "message" not in res              # no fabricated "Parked ..." message
    assert not any(isinstance(r, ApprovalGrant) for r in db.rows)  # no resumable row
    assert task.status == terminal           # not flipped to blocked
    assert dispatched == []                  # no question_pending emitted


@pytest.mark.asyncio
async def test_missing_question_rejected(dispatched):
    from modules.tools.discovery.handlers_asks import ask_human

    res = await ask_human(_FakeSession(), uuid4(), {
        "subject_type": "tool_call", "subject_id": "c1", "question": "  ",
    })
    assert res["success"] is False and "question" in res["error"]


# ===========================================================================
# 3. question_pending dispatch — link_type + urgency
# ===========================================================================

@pytest.mark.asyncio
async def test_dispatch_fires_with_question_link(dispatched):
    from modules.tools.discovery.handlers_asks import ask_human

    db = _FakeSession()
    ws = uuid4()
    db.add(_board_task(ws, 1, status="in_progress"))  # lone task ⇒ no cascade
    res = await ask_human(db, ws, {
        "subject_type": "board_task", "subject_id": "1",
        "question": "Proceed?", "_agent_id": 3, "_agent_name": "Scout",
    })
    assert len(dispatched) == 1
    call = dispatched[0]
    assert call["event_type"] == "question_pending"
    assert call["link_type"] == "question"
    assert call["link_id"] == str(res["ask_id"])
    assert call["severity"] is None  # no cascade ⇒ respects quiet hours


@pytest.mark.asyncio
async def test_large_cascade_marks_urgent(dispatched):
    """A board task with ≥3 transitive dependents bypasses quiet hours."""
    from modules.tools.discovery.handlers_asks import ask_human

    db = _FakeSession()
    ws = uuid4()
    db.add(_board_task(ws, 1, status="in_progress"))
    for cid in (2, 3, 4):
        db.add(_board_task(ws, cid, status="assigned", parent=1))

    res = await ask_human(db, ws, {
        "subject_type": "board_task", "subject_id": "1",
        "question": "Blocking the whole batch — go?", "_agent_id": 5,
    })
    assert res["downstream_blocked"] == 3
    assert dispatched[0]["severity"] == "urgent"


@pytest.mark.asyncio
async def test_small_cascade_is_not_urgent(dispatched):
    from modules.tools.discovery.handlers_asks import ask_human

    db = _FakeSession()
    ws = uuid4()
    db.add(_board_task(ws, 1, status="in_progress"))
    db.add(_board_task(ws, 2, status="assigned", parent=1))

    res = await ask_human(db, ws, {
        "subject_type": "board_task", "subject_id": "1",
        "question": "One dependent only", "_agent_id": 5,
    })
    assert res["downstream_blocked"] == 1
    assert dispatched[0]["severity"] is None


# ===========================================================================
# 4. Cascade is cycle-safe (pure)
# ===========================================================================

def test_reachable_from_is_cycle_safe():
    from services.ask_cascade import reachable_from, is_urgent_cascade

    adj = {"root": ["a"], "a": ["b"], "b": ["a", "c"], "c": []}
    order = reachable_from(adj, "root")
    assert order == ["a", "b", "c"]  # terminates, each node once
    assert is_urgent_cascade(len(order)) is True
    assert is_urgent_cascade(2) is False
    assert is_urgent_cascade(None) is False


# ===========================================================================
# 4b. P225-RVW-3 — the cascade follows the mission DAG, not just parent_task_id
# ===========================================================================

def _dag_mission(db, ws):
    """Steps 5..9 as FLAT parent_task_id siblings of one mission board task, with
    6,7,8,9 depending on step 5 ONLY through OrchestrationTaskDependency (not
    parent_task_id) — the flat-sibling shape orchestration_board_bridge produces."""
    from core.models.core import BoardTask
    from core.models.orchestration import OrchestrationTaskDependency

    ot = {n: uuid4() for n in (5, 6, 7, 8, 9)}
    mission_parent = 100
    db.add(BoardTask(id=5, workspace_id=ws, title="step 5", status="in_progress",
                     parent_task_id=mission_parent, orchestration_task_id=ot[5]))
    for n in (6, 7, 8, 9):
        db.add(BoardTask(id=n, workspace_id=ws, title=f"step {n}", status="assigned",
                         parent_task_id=mission_parent, orchestration_task_id=ot[n]))
        # OrchestrationTaskDependency(task_id=downstream, depends_on_task_id=upstream):
        # step n depends on step 5, so parking step 5 blocks step n.
        db.add(OrchestrationTaskDependency(task_id=ot[n], depends_on_task_id=ot[5]))
    return ot


def test_mission_dag_cascade_counts_dependents():
    """count_downstream_blocked / board_task_cascade_detail follow the DAG: step 5
    with steps 6-9 depending on it (NOT parent_task_id children) counts 4."""
    from services.ask_cascade import (
        count_downstream_blocked, is_urgent_cascade, board_task_cascade_detail,
    )

    db = _FakeSession()
    ws = uuid4()
    _dag_mission(db, ws)

    count = count_downstream_blocked(db, ws, "board_task", "5")
    assert count == 4  # 0 via parent_task_id, 4 via the DAG
    assert is_urgent_cascade(count) is True

    detail = board_task_cascade_detail(db, ws, "5")
    assert detail["total"] == 4
    assert {t["id"] for t in detail["tasks"]} == {6, 7, 8, 9}


@pytest.mark.asyncio
async def test_mission_dag_ask_marks_urgent(dispatched):
    """The real handler on a mission-blocking ask fires the urgent bypass — the
    baked >=3 rule now works for a genuine mission ask (was 0 → never urgent)."""
    from modules.tools.discovery.handlers_asks import ask_human

    db = _FakeSession()
    ws = uuid4()
    _dag_mission(db, ws)

    res = await ask_human(db, ws, {
        "subject_type": "board_task", "subject_id": "5",
        "question": "Blocking steps 6-9 — proceed?", "_agent_id": 5,
    })
    assert res["downstream_blocked"] == 4
    assert dispatched[0]["severity"] == "urgent"


def test_mission_dag_cascade_is_cycle_safe():
    """A DAG cycle (5→6→5 via dependency edges) terminates and each node counts
    once — cycle-safety holds across the merged adjacency."""
    from services.ask_cascade import board_task_cascade
    from core.models.core import BoardTask
    from core.models.orchestration import OrchestrationTaskDependency

    db = _FakeSession()
    ws = uuid4()
    ot5, ot6 = uuid4(), uuid4()
    db.add(BoardTask(id=5, workspace_id=ws, title="s5", status="in_progress",
                     parent_task_id=None, orchestration_task_id=ot5))
    db.add(BoardTask(id=6, workspace_id=ws, title="s6", status="assigned",
                     parent_task_id=None, orchestration_task_id=ot6))
    # 6 depends on 5 AND 5 depends on 6 — a corrupt cycle.
    db.add(OrchestrationTaskDependency(task_id=ot6, depends_on_task_id=ot5))
    db.add(OrchestrationTaskDependency(task_id=ot5, depends_on_task_id=ot6))

    order = board_task_cascade(db, ws, 5)
    assert order == ["6"]  # terminates; 5 is the root, 6 counted once


# ===========================================================================
# 5. Asker is server-minted (never a tool param) + no waiting
# ===========================================================================

@pytest.mark.asyncio
async def test_asked_by_agent_id_is_server_minted(dispatched):
    from modules.tools.discovery.handlers_asks import ask_human
    from core.models.approval_grants import ApprovalGrant

    db = _FakeSession()
    ws = uuid4()
    db.add(_board_task(ws, 1, status="in_progress"))
    res = await ask_human(db, ws, {
        "subject_type": "board_task", "subject_id": "1", "question": "Q",
        "_agent_id": 42,
        # A spoofed asker in the tool args must be ignored — only _agent_id counts.
        "asked_by_agent_id": 999,
    })
    assert res["success"] is True
    grant = next(r for r in db.rows if isinstance(r, ApprovalGrant))
    assert grant.asked_by_agent_id == 42


def test_handler_never_waits():
    """No sleep / polling loop in the handler — park-and-return is mechanical."""
    from modules.tools.discovery import handlers_asks

    src = Path(handlers_asks.__file__).read_text()
    assert "sleep(" not in src
    assert "while True" not in src
    assert ".poll(" not in src
