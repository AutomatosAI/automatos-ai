"""PRD-193 S4 (P2-12) — approving must complete the work, not just flip a row.

The grant API's subject dispatch — board-only until now — gains the
``tool_call`` branch: on grant, re-dispatch the stored call through the spine
(``UnifiedToolExecutor.execute_tool`` — telemetry, the policy seam, and
outcome capture all fire on the resumed execution); on deny, end it honestly
(the DENIED status is the record; nothing executes). Board-originated asks
(``details.board_task_id``) resume through the EXISTING board re-queue — the
re-run then meets the now-active grant (S2) and completes (locked decision 4:
lean board linkage, no J2 mid-run pause).

Guarantees pinned here:
  1. ``test_grant_resumes_tool_call``       — re-dispatch carries the stored
     params + workspace + caller context + server-minted agent identity; the
     outcome summary lands on ``details.executed_result``.
  2. ``test_deny_does_not_execute``         — deny leaves the executor untouched
     and the grant DENIED.
  3. ``test_resume_failure_is_honest``      — a failing (or raising) re-dispatch
     surfaces as a failure on the grant — never a fake success (dossier C.3).
  4. ``test_board_linked_ask_requeues_task``— a grant carrying board_task_id
     re-queues the blocked task (mirror of the existing board branch) and does
     NOT double-dispatch.
  5. Board-linked but not blocked ⇒ direct re-dispatch still completes the work.
  6. ``to_dict`` exposes ``details`` so the card can render the executed state.

Pure: fake session, real ApprovalGrant/BoardTask model objects, executor and
board notifier mocked at their boundaries. No DB / network.
"""
from __future__ import annotations

import importlib.util as _ilu
import os
import sys as _sys
import uuid
from unittest.mock import AsyncMock, patch

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

from core.models.approval_grants import ApprovalGrant, GrantStatus, SUBJECT_TOOL_CALL
from core.models.core import BoardTask
from core.services.approval_grants import deny_grant, grant_grant

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fake session — .query(Model).get(pk) plus the list shape the service uses.
# ---------------------------------------------------------------------------

class _Query:
    def __init__(self, rows):
        self._rows = list(rows)

    def get(self, pk):
        for r in self._rows:
            if getattr(r, "id", None) == pk:
                return r
        return None

    def filter(self, *conds):
        return self

    def order_by(self, *a):
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

    def flush(self):
        pass

    def query(self, model):
        return _Query([r for r in self.rows if isinstance(r, model)])


def _tool_call_grant(
    db: _FakeSession,
    *,
    workspace_id=None,
    action: str = "platform_delete_document",
    params: dict | None = None,
    board_task_id=None,
    agent_id: int = 9,
) -> ApprovalGrant:
    params = {"document_id": 7} if params is None else params
    details = {
        "action": action,
        "params": params,
        "params_hash": "h",
        "lane": "chat" if board_task_id is None else "board",
        "caller_context": {"user_id": "user_clerk_1", "conversation_id": "c-1"},
    }
    if board_task_id is not None:
        details["board_task_id"] = board_task_id
    grant = ApprovalGrant(
        workspace_id=workspace_id or uuid.uuid4(),
        subject_type=SUBJECT_TOOL_CALL,
        subject_id=f"{action}:deadbeef",
        tool_name=action,
        risk_tier="destructive",
        agent_id=agent_id,
        status=GrantStatus.PENDING.value,
        details=details,
    )
    db.add(grant)
    grant_grant(grant, granted_by="user:1")
    return grant


def _blocked_task(db: _FakeSession, task_id: int = 77) -> BoardTask:
    task = BoardTask(
        title="probe task",
        status="blocked",
        workspace_id=uuid.uuid4(),
    )
    task.id = task_id
    db.rows.append(task)
    return task


# ===========================================================================
# 1. Grant ⇒ re-dispatch through the spine with the stored call
# ===========================================================================

async def test_grant_resumes_tool_call():
    from api.approval_grants import _requeue_subject

    db = _FakeSession()
    grant = _tool_call_grant(db)

    executor = AsyncMock()
    executor.execute_tool = AsyncMock(return_value={"success": True, "deleted": True})

    with patch(
        "modules.tools.execution.unified_executor.UnifiedToolExecutor",
        return_value=executor,
    ) as executor_cls:
        await _requeue_subject(db, grant)

    executor_cls.assert_called_once_with(db)
    kwargs = executor.execute_tool.await_args.kwargs
    assert kwargs["tool_name"] == "platform_delete_document"
    assert kwargs["parameters"] == {"document_id": 7}
    assert kwargs["workspace_id"] == grant.workspace_id
    assert kwargs["agent_id"] == 9  # the server-minted identity, re-threaded
    assert kwargs["caller_context"] == {
        "user_id": "user_clerk_1",
        "conversation_id": "c-1",
    }

    executed = (grant.details or {}).get("executed_result")
    assert executed is not None, "the outcome summary must land on the grant"
    assert executed["success"] is True
    assert executed.get("error") is None
    # No board side effects for a chat-lane grant.
    assert all(not isinstance(r, BoardTask) for r in db.rows)


# ===========================================================================
# 2. Deny is honest: nothing executes, the DENIED row is the record
# ===========================================================================

async def test_deny_does_not_execute():
    from api.approval_grants import _fail_subject

    db = _FakeSession()
    grant = _tool_call_grant(db)
    deny_grant(grant, revoked_by="user:1")

    with patch(
        "modules.tools.execution.unified_executor.UnifiedToolExecutor"
    ) as executor_cls:
        _fail_subject(db, grant)

    executor_cls.assert_not_called()
    assert grant.status == GrantStatus.DENIED.value
    assert "executed_result" not in (grant.details or {})


# ===========================================================================
# 3. A failed resume surfaces as a failure — never a fake success
# ===========================================================================

async def test_resume_failure_is_honest():
    from api.approval_grants import _requeue_subject

    # (a) executor returns errors-as-data
    db = _FakeSession()
    grant = _tool_call_grant(db)
    executor = AsyncMock()
    executor.execute_tool = AsyncMock(return_value={"success": False, "error": "boom"})
    with patch(
        "modules.tools.execution.unified_executor.UnifiedToolExecutor",
        return_value=executor,
    ):
        await _requeue_subject(db, grant)
    executed = grant.details["executed_result"]
    assert executed["success"] is False
    assert "boom" in (executed.get("error") or "")

    # (b) executor raises — still an honest failure on the grant, no exception
    db2 = _FakeSession()
    grant2 = _tool_call_grant(db2)
    executor2 = AsyncMock()
    executor2.execute_tool = AsyncMock(side_effect=RuntimeError("spine down"))
    with patch(
        "modules.tools.execution.unified_executor.UnifiedToolExecutor",
        return_value=executor2,
    ):
        await _requeue_subject(db2, grant2)
    executed2 = grant2.details["executed_result"]
    assert executed2["success"] is False
    assert "spine down" in (executed2.get("error") or "")


async def test_resume_reask_is_reported_not_faked():
    """If the re-dispatch hits the gate again (e.g. consumption raced), the
    summary says so — the card must not claim the action ran."""
    from api.approval_grants import _requeue_subject

    db = _FakeSession()
    grant = _tool_call_grant(db)
    executor = AsyncMock()
    executor.execute_tool = AsyncMock(
        return_value={"success": False, "requires_confirmation": True, "grant_id": 99}
    )
    with patch(
        "modules.tools.execution.unified_executor.UnifiedToolExecutor",
        return_value=executor,
    ):
        await _requeue_subject(db, grant)
    executed = grant.details["executed_result"]
    assert executed["success"] is False
    assert executed["requires_confirmation"] is True


# ===========================================================================
# 4. Board linkage — the EXISTING re-queue resumes board-originated asks
# ===========================================================================

async def test_board_linked_ask_requeues_task():
    from api.approval_grants import _requeue_subject

    db = _FakeSession()
    task = _blocked_task(db, task_id=77)
    grant = _tool_call_grant(db, board_task_id=77)

    executor = AsyncMock()
    with patch(
        "modules.tools.execution.unified_executor.UnifiedToolExecutor",
        return_value=executor,
    ) as executor_cls, patch(
        "services.board_dispatcher.notify_task_available"
    ) as notify:
        await _requeue_subject(db, grant)

    # Mirror of the existing board branch: blocked → assigned + re-notified.
    assert task.status == "assigned"
    assert task.blocked_at is None and task.blocked_reason is None
    notify.assert_called_once()
    # The re-run executes into the now-active grant — no direct double-dispatch.
    executor_cls.assert_not_called()
    executed = grant.details["executed_result"]
    assert executed["resumed_via"] == "board_task_requeue"
    assert executed["board_task_id"] == 77


async def test_board_linked_but_unblocked_falls_through_to_dispatch():
    """The task already finished (the ask ended its run) — the human's yes
    must still complete the WORK: direct re-dispatch through the spine."""
    from api.approval_grants import _requeue_subject

    db = _FakeSession()
    task = _blocked_task(db, task_id=77)
    task.status = "done"
    grant = _tool_call_grant(db, board_task_id=77)

    executor = AsyncMock()
    executor.execute_tool = AsyncMock(return_value={"success": True})
    with patch(
        "modules.tools.execution.unified_executor.UnifiedToolExecutor",
        return_value=executor,
    ):
        await _requeue_subject(db, grant)

    assert task.status == "done"  # untouched
    executor.execute_tool.assert_awaited_once()
    assert grant.details["executed_result"]["success"] is True


# ===========================================================================
# 5. Existing board-task subject behaviour is unchanged (regression pin)
# ===========================================================================

async def test_board_task_subject_branch_unchanged():
    from api.approval_grants import _requeue_subject
    from core.models.approval_grants import SUBJECT_BOARD_TASK

    db = _FakeSession()
    task = _blocked_task(db, task_id=55)
    grant = ApprovalGrant(
        workspace_id=uuid.uuid4(),
        subject_type=SUBJECT_BOARD_TASK,
        subject_id="55",
        status=GrantStatus.PENDING.value,
    )
    db.add(grant)
    grant_grant(grant, granted_by="user:1")

    with patch("services.board_dispatcher.notify_task_available") as notify:
        await _requeue_subject(db, grant)

    assert task.status == "assigned"
    notify.assert_called_once()


# ===========================================================================
# 6. The card can see the executed state (to_dict exposes details)
# ===========================================================================

async def test_grant_to_dict_exposes_details():
    db = _FakeSession()
    grant = _tool_call_grant(db)
    grant.details = {**grant.details, "executed_result": {"success": True}}
    payload = grant.to_dict()
    assert payload["details"]["executed_result"]["success"] is True
