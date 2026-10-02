"""PRD-252 R3 — every Review or Blocked ticket says why it is there.

Review had six causes and Blocked seven, and the card said neither. The board's
API now serves ``review_reason`` and ``blocked_code`` (core/services/
ticket_reasons.py), and a spend hold, which said the ticket "comes back on its
own once the ceiling is raised", now does (release_spend_holds).
"""
from __future__ import annotations

import inspect
import json
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace as NS

import pytest
from sqlalchemy import create_engine, text

from core.services import ticket_reasons as tr

NOW = datetime(2026, 10, 2, 9, 30, tzinfo=timezone.utc)


def _ticket(status="review", **over):
    fields = dict(status=status, source_type="user", review_mode="auto", result="Draft: welcome email.",
                  review_feedback=None, planning_data={}, runtime_ref={}, blocked_reason=None, completed_at=NOW)
    fields.update(over)
    return NS(**fields)


@pytest.mark.parametrize("over, code", [
    (dict(source_type="orchestration_task"), tr.MISSION_CHECKING),
    (dict(result=f"Done.\n\n{tr.FILE_MISSING_NOTE_PREFIX} `pack.md` — the result names it, ..."), tr.FILE_MISSING),
    (dict(runtime_ref={"denials": 2}), tr.HELD_COMMAND),
    (dict(review_feedback="Finished, worker never reported — its deliverables are on the ticket."), tr.RETRIES_USED_UP),
    (dict(review_feedback="Stopped after 6 attempts — answered twice. Nothing was re-queued."), tr.RETRIES_USED_UP),
    (dict(planning_data={"approval_action": {"type": "publish_blog"}}), tr.APPROVAL_ACTION),
    (dict(source_type="recipe"), tr.STOPPED_WITH_WORK),
    (dict(review_mode="human"), tr.ASKED),
    (dict(review_mode="llm"), tr.ASKED),            # llm has no reviewer: it is a person's review
    (dict(result="  "), tr.NOTHING_DONE),
    (dict(), tr.UNEXPLAINED),
])
def test_a_review_says_why(over, code):
    assert tr.review_reason(_ticket(**over)) == code


def test_the_nothing_done_note_is_read_as_its_reason():
    from services.result_substance import nothing_done_note

    note = nothing_done_note("platform_send_email: Skipped: no recipient")
    assert tr.review_reason(_ticket(result=f"Here.\n\n{note}")) == tr.NOTHING_DONE


def test_a_recorded_reason_speaks_only_for_its_own_review():
    moved = _ticket(runtime_ref=tr.with_review_reason({}, tr.MOVED_BY_YOU, NOW), review_mode="human")
    assert tr.review_reason(moved) == tr.MOVED_BY_YOU
    later = _ticket(runtime_ref=moved.runtime_ref, review_mode="human",
                    completed_at=datetime(2026, 10, 3, 8, 0, tzinfo=timezone.utc))
    assert tr.review_reason(later) == tr.ASKED          # a later review is not "you moved it"


@pytest.mark.parametrize("over, code", [
    (dict(runtime_ref={"operator_stop": {"status": "blocked", "by": "operator"}}), tr.STOPPED_BY_YOU),
    (dict(runtime_ref={tr.SPEND_HOLD_KEY: NOW.isoformat()}, blocked_reason="Spend this window is $21.40, over"),
     tr.SPEND_CEILING),
    (dict(blocked_reason="Waiting on your answer to the agent's question (ask #41)"), tr.QUESTION),
    (dict(blocked_reason="Awaiting human approval (grant #11): board task requires approval"), tr.APPROVAL),
    (dict(source_type="orchestration", blocked_reason="Mission paused"), tr.MISSION_PAUSED),
    (dict(source_type="orchestration_task", blocked_reason="tool error"), tr.STEP_FAILED),
    (dict(blocked_reason="waiting on the café"), tr.WAITING),
])
def test_a_block_says_what_it_waits_for(over, code):
    assert tr.blocked_code(_ticket("blocked", **over)) == code


def test_only_review_and_blocked_tickets_carry_a_code():
    assert tr.review_reason(_ticket("done")) is None and tr.blocked_code(_ticket("review")) is None


def test_the_writers_still_use_the_lines_the_codes_read():
    """The notes and lines are written where they always were; this keeps them matching."""
    from services import board_dispatcher, cli_host_service, result_files, result_substance

    assert result_substance.NOTHING_DONE_NOTE.startswith(tr.NOTHING_DONE_NOTE_PREFIX)
    assert "FILE_MISSING_NOTE_PREFIX" in inspect.getsource(result_files.check_named_files)
    assert f"'{tr.NEVER_REPORTED_PREFIX}" in inspect.getsource(board_dispatcher.requeue_expired_leases)
    assert f'f"{tr.ATTEMPTS_STOPPED_PREFIX} {{' in inspect.getsource(cli_host_service.park_exhausted)


def test_the_board_api_serves_both_codes():
    from core.models.core import BoardTask
    from services.board_task_view import board_dict

    task = BoardTask(id=7, status="blocked", source_type="user", review_mode="auto", tags=[], attempts=0,
                     blocked_reason="Waiting on your answer to the agent's question (ask #41)")
    served = board_dict(task)
    assert (served["blocked_code"], served["review_reason"]) == (tr.QUESTION, None)


# ── The spend hold comes back on its own, as it said it would ───────────────

@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the spend-hold tests need a reachable Postgres: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def held(engine, new_session, monkeypatch):
    import services.board_dispatcher as dispatcher

    ws = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'prd252-spend')"), {"id": ws})
    task = s.execute(text(
        "INSERT INTO board_tasks (workspace_id, title, status, blocked_reason, runtime_ref) "
        "VALUES (CAST(:w AS uuid), 'Weekly numbers', 'blocked', 'Spend this window is $21.40, over the $20 ceiling', "
        "CAST(:ref AS jsonb)) RETURNING id"),
        {"w": ws, "ref": json.dumps({tr.SPEND_HOLD_KEY: NOW.isoformat(), "session_id": "s-1"})}).scalar()
    s.commit()
    woken = []
    monkeypatch.setattr(dispatcher, "notify_task_available", lambda db, **kw: woken.append(kw["task_id"]))
    yield NS(id=task, ws=ws, new=new_session, woken=woken)
    s = new_session.sweep()
    s.execute(text("DELETE FROM board_tasks WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:w AS uuid)"), {"w": ws})
    s.commit()


def _row(held):
    return held.new().execute(text(
        "SELECT status, blocked_reason, runtime_ref FROM board_tasks WHERE id = :i"), {"i": held.id}).first()


def _over(monkeypatch, over):
    import services.daily_spend_guard as guard

    monkeypatch.setattr(guard, "spend_state", lambda db, ws: NS(over=over))


def test_a_held_ticket_goes_back_to_assigned_once_the_ceiling_allows(held, monkeypatch):
    from services.daily_spend_guard import release_spend_holds

    _over(monkeypatch, False)
    released = release_spend_holds(held.new())

    status, reason, ref = _row(held)
    assert held.id in released and held.id in held.woken
    assert (status, reason) == ("assigned", None)
    assert tr.SPEND_HOLD_KEY not in ref and ref["session_id"] == "s-1"


def test_a_ticket_stays_held_while_the_workspace_is_over(held, monkeypatch):
    from services.daily_spend_guard import release_spend_holds

    _over(monkeypatch, True)
    assert held.id not in release_spend_holds(held.new())
    assert _row(held)[0] == "blocked"


def test_autos_ticket_tools_offer_no_llm_review():
    """Hidden until a model reviewer exists: the tools told Auto "'llm' has a model
    review it", and nothing did (it behaved as 'human')."""
    from modules.tools.discovery import actions_board_tasks

    source = inspect.getsource(actions_board_tasks)
    assert '"enum": ["human", "llm", "auto"]' not in source
    assert "has a model review it" not in source
