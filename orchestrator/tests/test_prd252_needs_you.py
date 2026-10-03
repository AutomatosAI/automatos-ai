"""PRD-252 R5 — one "Needs you" number, the same on every counter.

Five counters had five definitions: the Board tab badge counted every open
ticket, ATTENTION counted a grant-blocked ticket twice and a mission step
waiting on its own mission as the owner's, the Needs you widget left out
failures, and Auto's pill read a super-admin-only endpoint. Now the endpoint
serves the number and its rows (services/needs_you.py), and ATTENTION is that
number. Acceptance: for one period every counter shows it, and it equals the
rows the widget lists.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, text

NOW = datetime.now(timezone.utc)


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the Needs-you tests need a reachable Postgres: {exc}")
    yield eng
    eng.dispose()


def _ticket(s, ws, title, status, *, source="user", done=None, run=None):
    return s.execute(text(
        "INSERT INTO board_tasks (workspace_id, title, status, source_type, completed_at, orchestration_run_id) "
        "VALUES (CAST(:w AS uuid), :t, :s, :src, :done, CAST(:run AS uuid)) RETURNING id"),
        {"w": ws, "t": title, "s": status, "src": source, "done": done, "run": run}).scalar()


def _grant(s, ws, kind, status, subject_id, *, question=None, reason=None, details=None, expires=None):
    s.execute(text(
        "INSERT INTO approval_grants (workspace_id, subject_type, subject_id, status, kind, question_md, reason, "
        "details, requested_at, expires_at) VALUES (CAST(:w AS uuid), 'board_task', :sid, :st, :k, :q, :r, "
        "CAST(:d AS jsonb), :at, :exp)"),
        {"w": ws, "sid": str(subject_id), "st": status, "k": kind, "q": question, "r": reason,
         "d": json.dumps(details or {}), "at": NOW, "exp": expires})


@pytest.fixture
def floor(engine, new_session):
    """One café's floor: what needs the owner, and what looks like it but does not."""
    ws = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'prd252-needs-you')"), {"id": ws})
    run = s.execute(text(
        "INSERT INTO orchestration_runs (workspace_id, goal, created_by, state, state_type) "
        "VALUES (CAST(:w AS uuid), 'Launch the autumn blend', 'owner', 'awaiting_approval', 'blocked') RETURNING id"),
        {"w": ws}).scalar()
    lost = s.execute(text(
        "INSERT INTO orchestration_runs (workspace_id, goal, created_by, state, state_type) "
        "VALUES (CAST(:w AS uuid), 'Winter menu', 'owner', 'failed', 'terminal') RETURNING id"),
        {"w": ws}).scalar()
    hour_ago, days_ago = NOW - timedelta(hours=1), NOW - timedelta(days=3)
    t = NS(
        review=_ticket(s, ws, "Price list for the cafés", "review", done=hour_ago),
        step=_ticket(s, ws, "Draft the launch email", "review", source="orchestration_task", done=hour_ago, run=run),
        card=_ticket(s, ws, "Launch the autumn blend", "review", source="orchestration", run=run),
        failed=_ticket(s, ws, "Weekly numbers", "failed", done=hour_ago),
        failed_old=_ticket(s, ws, "Supplier check", "failed", done=days_ago),
        failed_step=_ticket(s, ws, "Book the photographer", "failed", source="orchestration_task", done=hour_ago),
        blocked=_ticket(s, ws, "Rota for October", "blocked"),
        # A step a Claude Code session ran carries its run, and it is on the board: it opens itself.
        session_step=_ticket(s, ws, "Mission: write the menu post", "review", source="mission", done=hour_ago, run=run),
        failed_card=_ticket(s, ws, "Winter menu", "failed", source="orchestration", done=hour_ago, run=lost),
        run=str(run), lost=str(lost), ws=ws,
    )
    _grant(s, ws, "question", "pending", t.blocked, question="Which café opens first on Sundays?")
    # Past its expiry but never answered: the Questions tab still lists it, so it still counts.
    _grant(s, ws, "question", "pending", t.blocked, question="Is Ana on the rota?", expires=NOW - timedelta(hours=2))
    _grant(s, ws, "question", "granted", t.blocked, question="Answered already")
    _grant(s, ws, "approval", "pending", t.review, reason="Send the price list to 40 cafés")
    s.commit()
    yield t
    s = new_session.sweep()
    for table in ("approval_grants", "board_tasks", "orchestration_runs"):
        s.execute(text(f"DELETE FROM {table} WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:w AS uuid)"), {"w": ws})
    s.commit()


def test_needs_you_counts_each_thing_once(floor, new_session):
    from services.needs_you import needs_you

    out = needs_you(new_session(), UUID(floor.ws))

    # review: not the step (its mission checks it), not the card (the mission's approval);
    # a session-run step on the board is. approval: the grant and the mission, once.
    # failed: every failed ticket and mission card, not the step (F246: whatever its age).
    assert out["counts"] == {"review": 2, "question": 2, "approval": 2, "stuck": 0, "failed": 3}
    assert out["total"] == 9
    assert sum(len(rows) for rows in out["rows"].values()) == out["total"]     # the widget lists every one


def test_each_row_opens_the_thing_itself(floor, new_session):
    from services.needs_you import needs_you

    rows = needs_you(new_session(), UUID(floor.ws))["rows"]

    assert {(r["ticket_id"], r["mission_id"]) for r in rows["review"]} == {
        (floor.review, None), (floor.session_step, None)}                      # F225 review: never the mission page
    assert {r["ticket_id"] for r in rows["question"]} == {floor.blocked}       # opens inside its ticket
    assert {(r["source"], r["ticket_id"]) for r in rows["approval"]} == {("grant", floor.review), ("mission", floor.card)}  # F246: its card
    assert [r["id"] for r in rows["approval"] if r["source"] == "mission"] == [floor.run]
    assert {(r["ticket_id"], r["mission_id"]) for r in rows["failed"]} == {
        (floor.failed, None), (floor.failed_old, None), (floor.failed_card, floor.lost)}  # a mission's card opens its mission


def test_a_failure_counts_whatever_its_age(floor, new_session):
    """F246: a '1d' window dropped a failure nobody had dealt with (#0003, #0004, #0050, #0052)."""
    from services.needs_you import needs_you

    out = needs_you(new_session(), UUID(floor.ws))

    assert floor.failed_old in {r["ticket_id"] for r in out["rows"]["failed"]}
    assert "period" not in out


def test_a_member_who_cannot_answer_is_not_counted_what_they_cannot_open(floor, new_session):
    from services.needs_you import needs_you

    out = needs_you(new_session(), UUID(floor.ws), may_answer=False)

    assert out["counts"] == {"review": 2, "question": 0, "approval": 1, "stuck": 0, "failed": 3}
    assert out["rows"]["question"] == [] and [r["source"] for r in out["rows"]["approval"]] == ["mission"]
    assert sum(len(rows) for rows in out["rows"].values()) == out["total"] == 6


@pytest.mark.parametrize("may_answer", [True, False])
def test_attention_is_the_needs_you_number(floor, new_session, monkeypatch, may_answer):
    """Night 1: ATTENTION counted the blocked ticket and its question twice, and the step."""
    from services.activity_service import ActivityService
    from services.needs_you import needs_you

    for counter in ("_count_working_now", "_count_channels_live", "_count_agents_active", "_count_tasks_in_queue"):
        monkeypatch.setattr(ActivityService, counter, lambda self: 0)
    monkeypatch.setattr(ActivityService, "_count_completed", lambda self, since: 0)
    for period in ("1d", "7d"):
        stats = ActivityService(new_session(), UUID(floor.ws)).get_stats(period=period, may_answer=may_answer)
        assert stats["needs_attention"] == needs_you(new_session(), UUID(floor.ws), may_answer=may_answer)["total"]


def test_the_endpoint_serves_the_viewers_number(floor, new_session, monkeypatch):
    import api.activity as activity

    monkeypatch.setattr(activity, "may_see_own_workspace_health", lambda db, ctx: False)
    out = activity.get_needs_you(db=new_session(), ctx=NS(workspace_id=UUID(floor.ws)))

    assert out["total"] == 6 and out["rows"]["question"] == []


def test_a_failed_read_is_never_nothing_needs_you(monkeypatch):
    """F207: a failure is an error the widget shows, not a zero."""
    import api.activity as activity

    def broken(*a, **k):
        raise RuntimeError("relation does not exist")

    monkeypatch.setattr(activity, "may_see_own_workspace_health", lambda db, ctx: True)
    monkeypatch.setattr(activity, "needs_you", broken)
    with pytest.raises(HTTPException) as failed:
        activity.get_needs_you(db=NS(), ctx=NS(workspace_id=uuid.uuid4()))

    assert failed.value.status_code == 503 and failed.value.detail == activity.NEEDS_YOU_NOT_LOADED


def test_the_feed_names_a_tickets_stage_in_the_boards_words():
    """The feed's mapped status ('failed' for Blocked, 'pending' for Review) stays
    for its filters; the board's own status rides along for the label."""
    from core.models.core import BoardTask
    from services.activity_service import ActivityService

    svc = ActivityService(NS(), uuid.uuid4())
    for status, mapped in (("blocked", "failed"), ("review", "pending"), ("in_progress", "running"), ("done", "completed")):
        item = svc._board_feed_item(BoardTask(id=9, title="Rota", status=status, source_type="user"), {})
        assert (item["status"], item["board_status"]) == (mapped, status)
