"""PRD-225 US-004 (backend half) — the Questions list reuses the grants route.

Pure tests: the ``kind=question`` filter on the EXISTING list route, and the
cascade enrichment (``board_task_cascade_detail``) — capped at 6 with an honest
total, cycle-safe, and only attached to question-kind rows.
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from core.models.approval_grants import ApprovalGrant, GrantStatus, KIND_QUESTION
from core.models.core import BoardTask


class _Query:
    def __init__(self, rows, stats=None):
        self._rows = list(rows)
        self._stats = stats if stats is not None else {}

    def filter(self, *conds):
        rows = self._rows
        for cond in conds:
            key = cond.left.key
            op = getattr(getattr(cond, "operator", None), "__name__", "")
            value = getattr(cond.right, "value", None)
            if op == "in_op":
                # Batched id.in_([...]) — record it so a test can prove the
                # display loads in ONE query, not a per-id loop (P225-RVW-4).
                self._stats["in_filters"] = self._stats.get("in_filters", 0) + 1
                allowed = {str(v) for v in (value or [])}
                rows = [r for r in rows if str(getattr(r, key, None)) in allowed]
            else:
                rows = [r for r in rows if str(getattr(r, key, None)) == str(value)]
        return _Query(rows, self._stats)

    def order_by(self, *a):
        return _Query(list(reversed(self._rows)), self._stats)

    def limit(self, *a):
        return self

    def first(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return list(self._rows)


class _FakeSession:
    def __init__(self):
        self.rows = []
        self.stats = {}

    def add(self, obj):
        if getattr(obj, "id", None) is None:
            obj.id = len(self.rows) + 1
        self.rows.append(obj)

    def query(self, model):
        return _Query([r for r in self.rows if isinstance(r, model)], self.stats)


def _task(ws, tid, *, parent=None, status="assigned", title=None):
    return BoardTask(
        id=tid, workspace_id=ws, title=title or f"T{tid}",
        status=status, parent_task_id=parent,
    )


# ===========================================================================
# board_task_cascade_detail — cap + total + shape
# ===========================================================================

def test_cascade_detail_caps_at_six_with_total():
    from services.ask_cascade import board_task_cascade_detail

    db = _FakeSession()
    ws = uuid.uuid4()
    db.add(_task(ws, 1))
    for cid in range(2, 10):  # 8 direct children of task 1
        db.add(_task(ws, cid, parent=1, title=f"child {cid}"))

    detail = board_task_cascade_detail(db, ws, 1)
    assert detail["total"] == 8
    assert len(detail["tasks"]) == 6  # capped
    assert detail["tasks"][0]["title"].startswith("child")
    assert set(detail["tasks"][0]) == {"id", "title", "status"}


def test_cascade_detail_empty_when_no_descendants():
    from services.ask_cascade import board_task_cascade_detail

    db = _FakeSession()
    ws = uuid.uuid4()
    db.add(_task(ws, 1))
    detail = board_task_cascade_detail(db, ws, 1)
    assert detail == {"total": 0, "tasks": []}


def test_cascade_detail_is_cycle_safe_and_transitive():
    from services.ask_cascade import board_task_cascade_detail

    db = _FakeSession()
    ws = uuid.uuid4()
    db.add(_task(ws, 1))
    db.add(_task(ws, 2, parent=1))
    db.add(_task(ws, 3, parent=2))  # transitive grandchild
    detail = board_task_cascade_detail(db, ws, 1)
    assert detail["total"] == 2  # both 2 and 3, no loop


# ===========================================================================
# P225-RVW-4 — terminal descendants excluded; the display is ONE batched query
# ===========================================================================

def test_terminal_children_excluded_from_cascade():
    """A mission-level ask over 7 done + 1 blocked flat sibling steps counts the
    ONE live task, not 8 — so it does NOT trip the urgent bypass (P225-RVW-4)."""
    from services.ask_cascade import (
        board_task_cascade,
        board_task_cascade_detail,
        count_downstream_blocked,
        is_urgent_cascade,
    )

    db = _FakeSession()
    ws = uuid.uuid4()
    db.add(_task(ws, 100, status="in_progress"))  # the mission board task (asked)
    for tid in range(2, 9):  # 7 finished siblings under the mission
        db.add(_task(ws, tid, parent=100, status="done"))
    db.add(_task(ws, 9, parent=100, status="blocked"))  # the one live one

    assert board_task_cascade(db, ws, 100) == ["9"]

    count = count_downstream_blocked(db, ws, "board_task", 100)
    assert count == 1  # not 8
    assert is_urgent_cascade(count) is False  # 8 would have tripped it

    detail = board_task_cascade_detail(db, ws, 100)
    assert detail["total"] == 1
    assert [t["id"] for t in detail["tasks"]] == [9]
    assert detail["tasks"][0]["status"] == "blocked"


def test_cascade_detail_loads_shown_tasks_in_one_query():
    """The shown tasks load via a single ``id.in_([...])`` batch, not a per-id
    loop — this runs per question row on the 30s-polled grants list (P225-RVW-4).
    The traversal uses equality filters, so exactly ONE ``in_`` query fires."""
    from services.ask_cascade import board_task_cascade_detail

    db = _FakeSession()
    ws = uuid.uuid4()
    db.add(_task(ws, 1))
    for cid in (2, 3, 4):
        db.add(_task(ws, cid, parent=1))

    detail = board_task_cascade_detail(db, ws, 1)
    assert [t["id"] for t in detail["tasks"]] == [2, 3, 4]  # BFS order preserved
    assert db.stats["in_filters"] == 1  # one batched load, not three point reads


def test_cascade_traversal_batches_dependent_lookups():
    """The DAG dependent board tasks load in ONE batched ``in_()`` query per
    level, not a per-edge ``.first()`` — so a fan-out of M dependency edges costs
    O(1) dep-board queries, independent of M (P225-RVW-13). This traversal reruns
    per question row on the 30s-polled grants list."""
    from services.ask_cascade import board_task_cascade
    from core.models.orchestration import OrchestrationTaskDependency

    def _run(fanout):
        db = _FakeSession()
        ws = uuid.uuid4()
        # Root board task 1 → orchestration task 100.
        db.add(BoardTask(id=1, workspace_id=ws, title="root", status="in_progress",
                         orchestration_task_id=100))
        # `fanout` dependents, each blocked on OT 100 via the DAG ONLY (flat
        # siblings, no parent_task_id — the primary mission shape, P225-RVW-3).
        for k in range(fanout):
            db.add(BoardTask(id=2 + k, workspace_id=ws, title=f"d{k}", status="assigned",
                             parent_task_id=None, orchestration_task_id=200 + k))
            db.add(OrchestrationTaskDependency(task_id=200 + k, depends_on_task_id=100))
        cascade = board_task_cascade(db, ws, 1)
        return len(cascade), db.stats.get("in_filters", 0)

    small_n, small_in = _run(3)
    large_n, large_in = _run(30)
    assert (small_n, large_n) == (3, 30)   # every dependent is found either way
    # ONE batched dep-board load at the root level, edge-count-independent (the
    # per-edge .first() loop would have fired 0 in_ filters and M point reads).
    assert small_in == large_in == 1


# ===========================================================================
# list route — kind filter + question-only cascade enrichment
# ===========================================================================

@pytest.mark.asyncio
async def test_list_filters_by_kind_and_enriches_cascade():
    from api.approval_grants import list_grants

    db = _FakeSession()
    ws = uuid.uuid4()
    ctx = SimpleNamespace(workspace_id=ws, user_id=1)

    # A board-task question with two downstream tasks.
    db.add(_task(ws, 5))
    db.add(_task(ws, 6, parent=5))
    db.add(_task(ws, 7, parent=5))
    db.add(ApprovalGrant(
        workspace_id=ws, subject_type="board_task", subject_id="5",
        kind=KIND_QUESTION, question_md="Go?", status=GrantStatus.PENDING.value,
    ))
    # An approval-kind grant that must be filtered out by kind=question.
    db.add(ApprovalGrant(
        workspace_id=ws, subject_type="board_task", subject_id="9",
        status=GrantStatus.PENDING.value,  # kind defaults to 'approval'
    ))

    res = await list_grants(status=None, kind="question", ctx=ctx, db=db)
    grants = res["grants"]
    assert len(grants) == 1
    q = grants[0]
    assert q["kind"] == KIND_QUESTION
    assert q["cascade"]["total"] == 2
    assert {t["id"] for t in q["cascade"]["tasks"]} == {6, 7}


@pytest.mark.asyncio
async def test_list_without_kind_does_not_enrich_approvals():
    """An approval-kind grant carries no cascade key (only questions do)."""
    from api.approval_grants import list_grants

    db = _FakeSession()
    ws = uuid.uuid4()
    ctx = SimpleNamespace(workspace_id=ws, user_id=1)
    db.add(ApprovalGrant(
        workspace_id=ws, subject_type="board_task", subject_id="1",
        status=GrantStatus.PENDING.value,
    ))
    res = await list_grants(status=None, kind=None, ctx=ctx, db=db)
    assert "cascade" not in res["grants"][0]
