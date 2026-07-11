"""PRD-193 S5 (P2-12) — surface the ask beyond the open chat + loop hygiene.

An ask that fires on a board/heartbeat/scheduled run with nobody watching must
not be silent — the exact failure class P2-02 closed for playbooks. The fresh
pending grant dispatches an ``approval_pending`` notification on non-chat
lanes (chat sees the S3 card live — no double-notify).

EVENT-VOCABULARY NOTE (deliberate deviation from the PRD text): the PRD names
``approval_requested``; PR #531 already shipped ``approval_pending`` for
exactly this concept (pending grant → tell the workspace's humans). One event
type per concept — this PRD REUSES ``approval_pending`` rather than minting a
rival (reuse-don't-fork; deviation stated in the PR body).

Loop hygiene pinned here too:
  - the ask outcome is typed as an ASK in memory, not failure-spam
    (``tool_outcome_capture`` — the noise class PRD-187 S2 cleaned up);
  - same-turn duplicate suppression stands (correct: ask-spam guard), while a
    fresh tracker per loop run means the post-grant retry is NOT dedup-blocked
    — pinned so a future shared-tracker refactor can't silently resurrect the
    dossier's dedup dead-end.
"""
from __future__ import annotations

import asyncio
import importlib.util as _ilu
import os
import sys as _sys
import uuid

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")


def _camelot_unlocatable() -> bool:  # pragma: no cover - env-dependent
    try:
        return _ilu.find_spec("camelot") is None
    except ValueError:
        return False


if _camelot_unlocatable():  # pragma: no cover - env-dependent
    import types as _types

    _sys.modules.setdefault("camelot", _types.ModuleType("camelot"))

from modules.tools.execution import tool_grants

pytestmark = pytest.mark.asyncio


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

    def order_by(self, *args):
        return _Query(list(reversed(self._rows)))

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

    def flush(self):
        pass

    def query(self, model):
        return _Query([r for r in self.rows if isinstance(r, model)])


@pytest.fixture()
def dispatched(monkeypatch):
    """Record every scheduled approval_pending dispatch (no real session/IO)."""
    calls = []

    async def _record(workspace_id, **kwargs):
        calls.append({"workspace_id": workspace_id, **kwargs})

    monkeypatch.setattr(tool_grants, "_dispatch_approval_pending", _record)
    return calls


def _issue(db, ws, caller_context):
    return tool_grants.issue_tool_grant(
        db, ws,
        action="platform_delete_document",
        params={"document_id": 7},
        permission_level="destructive",
        description="Delete a document permanently.",
        caller_context=caller_context,
    )


# ===========================================================================
# 1. The event is registered — reusing #531's approval_pending (see docstring)
# ===========================================================================

async def test_approval_requested_event_registered():
    """Mirror of test_prd163_async_planning's registration pin — the dispatcher
    must accept the approval event so a silent-lane ask actually delivers.
    (Named per the PRD's test list; asserts the REUSED ``approval_pending``.)"""
    from core.services.notification_dispatcher import VALID_EVENT_TYPES

    assert "approval_pending" in VALID_EVENT_TYPES


# ===========================================================================
# 2. Non-chat lanes notify; chat does not double-notify
# ===========================================================================

async def test_agent_lane_ask_notifies(dispatched):
    db = _FakeSession()
    ws = uuid.uuid4()
    grant = _issue(db, ws, caller_context=None)  # heartbeat/agent lane
    await asyncio.sleep(0)  # let the fire-and-forget task run

    assert len(dispatched) == 1
    call = dispatched[0]
    assert call["workspace_id"] == str(ws)
    assert call["grant_id"] == grant.id
    assert call["tool_name"] == "platform_delete_document"


async def test_board_lane_ask_notifies(dispatched):
    db = _FakeSession()
    grant = _issue(db, uuid.uuid4(), caller_context={"board_task_id": 77})
    await asyncio.sleep(0)

    assert len(dispatched) == 1
    assert dispatched[0]["grant_id"] == grant.id


async def test_chat_lane_ask_does_not_double_notify(dispatched):
    db = _FakeSession()
    _issue(db, uuid.uuid4(), caller_context={"conversation_id": "c-1", "turn_id": "t-1"})
    await asyncio.sleep(0)

    assert dispatched == []  # the S3 card is live in the conversation


async def test_reused_pending_grant_does_not_renotify(dispatched):
    """The idempotent re-ask reuses the pending row — announced once, on
    creation, exactly like the board_approval reuse branch."""
    db = _FakeSession()
    ws = uuid.uuid4()
    first = _issue(db, ws, caller_context=None)
    second = _issue(db, ws, caller_context=None)
    await asyncio.sleep(0)

    assert first.id == second.id
    assert len(dispatched) == 1


# ===========================================================================
# 3. The ask is typed as an ASK in memory — not failure-spam
# ===========================================================================

async def test_ask_not_captured_as_failure():
    from modules.memory.tool_outcome_capture import build_tool_outcome

    rec = build_tool_outcome(
        tool_name="platform_delete_document",
        parameters={},
        result={
            "success": False,
            "requires_confirmation": True,
            "grant_id": 42,
            "message": "This action (destructive) requires confirmation.",
        },
        workspace_id="ws-1",
    )
    assert rec is not None, "the ask IS a notable outcome — typed, not dropped"
    assert rec["metadata"]["outcome"] == "ask"
    assert rec["metadata"]["error_class"] == ""
    assert "failed" not in rec["fact"].lower()
    assert "awaiting human approval" in rec["fact"]

    # A genuine failure still classifies as a failure (unchanged behaviour).
    fail = build_tool_outcome(
        tool_name="platform_delete_document",
        parameters={},
        result={"success": False, "error": "boom"},
        workspace_id="ws-1",
    )
    assert "failed" in fail["fact"]
    assert fail["metadata"].get("outcome") != "ask"


# ===========================================================================
# 4. Loop hygiene: same-turn dedup stands; the post-grant retry is not blocked
# ===========================================================================

async def test_granted_retry_not_dedup_blocked():
    from modules.tools.execution.tool_execution_tracker import ToolExecutionTracker
    from modules.tools.execution.tool_loop import ToolLoopExecutor

    args = {"document_id": 7}

    # Same-turn duplicate suppression is CORRECT (the ask-spam guard) …
    turn1 = ToolExecutionTracker()
    assert turn1.should_skip_execution("platform_delete_document", args)[0] is False
    turn1.record_execution("platform_delete_document", args)
    assert turn1.should_skip_execution("platform_delete_document", args)[0] is True

    # … but the NEXT run's tracker is fresh, so the granted retry executes.
    turn2 = ToolExecutionTracker()
    assert turn2.should_skip_execution("platform_delete_document", args)[0] is False

    # Pin the loop executor's fresh-tracker-per-instance default so a future
    # shared-tracker refactor can't silently resurrect the dedup dead-end.
    async def _cb(*a, **k):  # pragma: no cover - never invoked
        return {}

    loop_a = ToolLoopExecutor(llm_callback=_cb, tool_callback=_cb)
    loop_b = ToolLoopExecutor(llm_callback=_cb, tool_callback=_cb)
    assert loop_a.tracker is not loop_b.tracker
    loop_a.tracker.record_execution("platform_delete_document", args)
    assert loop_b.tracker.should_skip_execution("platform_delete_document", args)[0] is False
