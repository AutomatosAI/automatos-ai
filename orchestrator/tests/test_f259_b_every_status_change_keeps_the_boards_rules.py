"""F259 (b), with F272, F275 and F278 — every status change keeps the board's rules.

PRD-252 R6 made a drag do what the matching button does. Two other doors kept their
own paths:

1. The general PATCH /{task_id} (an API client, not the board): it filed an
   untouched ticket as Done, approved a Review by moving it, moved a running ticket
   out from under its run, and launched the bare brief without the owner's consent.
2. Auto's platform_update_task_status: the same holes, and night 7b's #0181 — "send
   it back" moved the card to Assigned, it re-ran the OLD brief, and its first draft
   vanished (no history, no note). Asked to approve #0177 "with this note", Auto
   reached for status 'approved', which the tool refused, and had nowhere to put the
   note (F278).

And on the board itself, night 7b:
- F275: the board's Assign made the owner approve their own click (#0192 waited on
  grant #1220 under always_ask), while Auto's assign on their word did not.
- F272: Run now on #0177, the Analyst's, answered "nothing can start it yet:
  Waiting for a CLI host" and ran 8 s later: the line was night 7's, from when a CLI
  agent held it, and nothing ever took it off.
- The HARNESS approves a change by marking its board task done, then writing the
  actuation's result onto it: under the rule above, that move would be refused.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pytest
from fastapi import HTTPException

from tests import test_1094_a_ticket_with_no_agent_is_never_in_progress as f1094
from tests import test_prd252_drags_match_buttons as r6

board = r6.board        # PRD-252 R6's fakes over a seeded ticket, recording what each move set off
shop = f1094.board      # #1094's Postgres: a workspace, its Content Creator, Auto's real tool
CTX, _ticket = r6.CTX, r6._ticket
OWNER = "owner@cafe.test"


def _patch(task, body):
    """PATCH /{task_id}, the general route, over the R6 fakes."""
    from api import board_tasks as bt
    from tests.test_board_task_handlers import _FakeSession

    return asyncio.run(bt.update_task(task.id, r6._Req(body), ctx=CTX, db=_FakeSession(agent=NS(id=5), task=task)))


def _untouched(**over):
    return _ticket("inbox", **{"workspace_seq": 44, "assigned_agent_id": None, "result": None,
                               "completed_at": None, **over})


def _running(**over):
    lease = datetime.now(timezone.utc) + timedelta(minutes=10)   # the dispatch lease a live run renews
    return _ticket("in_progress", **{"workspace_seq": 94, "lease_until": lease, "result": None,
                                     "completed_at": None, **over})


# --- the general PATCH: the board's rules, judged on the ticket as the PATCH leaves it ----------


@pytest.mark.parametrize("to", ["done", "review"])
def test_a_patch_cannot_file_an_untouched_ticket_and_stores_nothing_it_carried(board, to):
    task = _untouched()

    with pytest.raises(HTTPException) as refused:
        _patch(task, {"status": to, "title": "Renamed"})

    assert refused.value.status_code == 409 and "give it to an agent, which starts it" in refused.value.detail  # F294
    assert (task.status, task.title, task.completed_at) == ("inbox", "Spring newsletter", None)
    assert board.completed == []


@pytest.mark.parametrize("to, button", [("done", "Use Approve"), ("assigned", "Use Reject")])
def test_a_patch_never_decides_a_review(board, to, button):
    task = _ticket("review", planning_data={"approval_action": {"type": "publish_blog", "post_id": "p1"}})

    with pytest.raises(HTTPException) as refused:
        _patch(task, {"status": to})

    assert refused.value.detail.startswith(button) and task.status == "review" and board.completed == []


def test_a_patch_leaves_a_running_ticket_to_its_run(board):
    task = _running()
    lease = task.lease_until

    with pytest.raises(HTTPException) as refused:
        _patch(task, {"status": "review"})

    assert "use Cancel to stop it first" in refused.value.detail
    assert (task.status, task.lease_until) == ("in_progress", lease)


def test_a_patch_that_brings_the_work_may_file_it(board):
    task = _untouched()

    reply = _patch(task, {"status": "done", "result": "Filed by hand: the June price list."})

    assert (reply["status"], task.result) == ("done", "Filed by hand: the June price list.")
    assert task.completed_at is not None and board.completed == [task.id]   # PRD-128's task_complete
    assert reply["number"] == "#0044"                                      # F276: the answer names it as the board does


def test_a_patch_to_in_progress_is_run_now_and_says_it_gave_the_approval(board, monkeypatch):
    from api import board_tasks as bt

    monkeypatch.setattr(bt, "record_operator_consent", lambda *a, **kw: "granted")
    task = _ticket("blocked", workspace_seq=59, result=None,
                   blocked_reason="Awaiting human approval (grant #1142): board task requires approval",
                   blocked_at=datetime(2026, 10, 3, 2, 3, tzinfo=timezone.utc))

    reply = _patch(task, {"status": "in_progress"})

    assert reply["message"] == "Ticket #0059 started. It was waiting for your approval; starting it gave it."
    assert (task.status, task.blocked_reason) == ("assigned", None) and board.launched == []
    assert board.available == [task.id]                                   # the dispatch loop runs it


def test_a_patch_to_cancelled_is_cancel(board, monkeypatch):
    import services.run_cancel as run_cancel

    stopped = []

    def _cancel(db, task, *, by, may):
        stopped.append(by)
        task.status = "cancelled"

    monkeypatch.setattr(run_cancel, "cancel_ticket", _cancel)
    task = _running()

    reply = _patch(task, {"status": "cancelled"})

    assert reply["status"] == "cancelled" and stopped == ["user:1"]      # F245: Cancel's stop, the run included


def test_a_patch_that_repeats_the_status_moves_nothing_and_refuses_nothing(board):
    task = _ticket("done", result=None)

    reply = _patch(task, {"status": "done", "title": "Spring newsletter (final)"})

    assert (reply["status"], task.title) == ("done", "Spring newsletter (final)")


# --- Auto's status tool, on Postgres: the ticket's state afterwards ---------------------------


def _tool(shop, **params):
    from modules.tools.discovery.handlers_board_tasks import update_board_task_status

    return asyncio.run(update_board_task_status(shop.db, shop.ws, params))


def _row(shop, task):
    shop.db.refresh(task)
    return task


def test_auto_cannot_file_an_untouched_ticket(shop):
    from services.ticket_numbers import ticket_label

    task = f1094._ticket(shop)

    out = _tool(shop, task_id=task.id, status="done")

    assert out["success"] is False
    assert out["error"] == (f"No one has worked on {ticket_label(task)} yet, so there is nothing to approve: "
                            "give it to an agent, which starts it, or Cancel it if it isn't needed.")  # F294
    assert (_row(shop, task).status, task.completed_at) == ("inbox", None)


def test_auto_leaves_a_running_ticket_to_its_run(shop):
    lease = datetime.now(timezone.utc) + timedelta(minutes=10)
    task = f1094._ticket(shop, status="in_progress", assigned_agent_id=shop.agent, lease_until=lease)

    out = _tool(shop, task_id=task.id, status="review")

    assert out["success"] is False and "use Cancel to stop it first" in out["error"]
    assert _row(shop, task).status == "in_progress" and task.lease_until is not None


def test_a_bulk_move_refuses_only_the_tickets_the_board_would(shop):
    bare = f1094._ticket(shop)
    drafted = f1094._ticket(shop, status="review", assigned_agent_id=shop.agent, result="Twelve descriptions.",
                            completed_at=datetime.now(timezone.utc))

    out = _tool(shop, task_ids=[bare.id, drafted.id], status="done")

    assert (out["updated"], [f["task_id"] for f in out["failed"]], out["partial"]) == ([drafted.id], [bare.id], True)
    assert (_row(shop, drafted).status, _row(shop, bare).status) == ("done", "inbox")


@pytest.mark.parametrize("word", ["assigned", "send back"])
def test_autos_send_back_is_the_boards_reject(shop, word):
    """F278 (#0181): the draft goes on record, the owner's note is the correction."""
    from services.ticket_redo import SENT_BACK

    task = f1094._ticket(shop, status="review", assigned_agent_id=shop.agent, result="Draft 1: the old prices.",
                         completed_at=datetime.now(timezone.utc))

    out = _tool(shop, task_id=task.id, status=word, note="Use the June prices.", _user_id=OWNER)

    row = _row(shop, task)
    kept = (row.planning_data or {}).get("previous_runs", [{}])[-1]
    assert out["success"] is True and (row.status, row.result) == ("assigned", None)
    assert (kept.get("result"), kept.get("why"), kept.get("by")) == ("Draft 1: the old prices.", SENT_BACK,
                                                                       f"user:{OWNER}")
    assert row.planning_data["owner_corrections"][-1]["note"] == "Use the June prices."
    assert row.review_feedback == "Use the June prices."                  # what the redo works from


def test_autos_approval_keeps_the_owners_note(shop):
    """#0177: "Approve it with this note" — 'approved' is Done, and the note is theirs."""
    task = f1094._ticket(shop, status="review", assigned_agent_id=shop.agent, result="35 and 22 subscriptions.",
                         completed_at=datetime.now(timezone.utc))

    out = _tool(shop, task_id=task.id, status="approved", note="Going with Kestrel's 250-box run.", _user_id=OWNER)

    row = _row(shop, task)
    notes = [(n.get("by"), n.get("note")) for n in (row.runtime_ref or {}).get("session_notes") or []]
    assert out["success"] is True and row.status == "done"
    assert ("you", "Approved: Going with Kestrel's 250-box run.") in notes


# --- the board: Assign is the owner's approval (F275); no stale host line (F272) ----------------


def test_the_boards_assign_is_the_owners_approval(shop):
    import api.board_tasks as bt
    from core.models.approval_grants import SUBJECT_BOARD_TASK
    from core.services.approval_grants import find_active_grant
    from services.board_consent import WHY_ASSIGNED_BY_HAND

    task = f1094._ticket(shop)

    reply = asyncio.run(bt.update_task(task.id, f1094._Request({"assigned_agent_id": shop.agent}),
                                       ctx=shop.ctx, db=shop.db))

    grant = find_active_grant(shop.db, shop.ws, subject_type=SUBJECT_BOARD_TASK, subject_id=str(task.id))
    assert reply["status"] == "assigned" and grant is not None and grant.reason == WHY_ASSIGNED_BY_HAND


def test_a_ticket_no_cli_agent_holds_loses_the_wait_for_a_host_line(shop):
    import api.board_tasks as bt
    from services.cli_ticket_lane import NO_HOST_REASON

    task = f1094._ticket(shop, blocked_reason=NO_HOST_REASON)            # night 7's, from the CLI agent

    asyncio.run(bt.update_task(task.id, f1094._Request({"assigned_agent_id": shop.agent}),
                               ctx=shop.ctx, db=shop.db))

    assert _row(shop, task).blocked_reason is None


def test_run_now_never_says_a_host_must_start_an_api_agents_ticket(shop):
    import api.board_tasks as bt
    from services.cli_ticket_lane import NO_HOST_REASON

    task = f1094._ticket(shop, status="assigned", assigned_agent_id=shop.agent, blocked_reason=NO_HOST_REASON)

    reply = asyncio.run(bt.run_task_now(task.id, ctx=shop.ctx, db=shop.db))

    assert "nothing can start it" not in reply["message"] and reply["message"].endswith("started.")
    assert _row(shop, task).blocked_reason is None


# --- the HARNESS: the result first, then Done --------------------------------------------------


def test_the_harness_files_its_task_done_with_its_result(shop):
    from api import harness_commands as hc
    from modules.tools.discovery.handlers_board_tasks import update_board_task_status as tool  # the status move

    task = f1094._ticket(shop, title="[HARNESS] heartbeat for Analyst", tags=["harness", "rx:rx-7"])
    executor = NS(execute=lambda name, params: tool(shop.db, shop.ws, params))

    asyncio.run(hc._done_with_its_result(shop.db, executor, shop.ws, task.id, {"success": True, "data": {"ok": 1}},
                                         NS(reason="the owner approved it")))

    row = _row(shop, task)
    assert row.status == "done" and row.result and row.completed_at                # F048: done, result != null
