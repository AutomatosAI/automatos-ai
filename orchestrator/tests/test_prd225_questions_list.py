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
    def __init__(self, rows):
        self._rows = list(rows)

    def filter(self, *conds):
        rows = self._rows
        for cond in conds:
            key = cond.left.key
            value = getattr(cond.right, "value", None)
            rows = [r for r in rows if str(getattr(r, key, None)) == str(value)]
        return _Query(rows)

    def order_by(self, *a):
        return _Query(list(reversed(self._rows)))

    def limit(self, *a):
        return self

    def first(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return list(self._rows)


class _FakeSession:
    def __init__(self):
        self.rows = []

    def add(self, obj):
        if getattr(obj, "id", None) is None:
            obj.id = len(self.rows) + 1
        self.rows.append(obj)

    def query(self, model):
        return _Query([r for r in self.rows if isinstance(r, model)])


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
