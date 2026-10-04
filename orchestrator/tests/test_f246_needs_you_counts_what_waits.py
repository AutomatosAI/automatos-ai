"""F246 — Needs you counts whatever waits for the owner, not only plain Review and Failed.

Night 7, against the persona's own count, Needs you missed:
1. stuck cards: waiting for a CLI host (#0016, #0161, #0177), Assigned to no
   agent (#0067), a rejected or re-briefed playbook card in Assigned (#0070,
   #0149), a mission's steps left open after it failed or was cancelled
   (#0119.3, #0176.9-.12);
2. failures more than a day old (#0003, #0004, #0050, #0052);
3. approvals that lapsed (#999, #1000) or whose card was cancelled (#1138 for
   #0093, #0144, #0166);
4. card numbers on mission plans (#0031, #0126, #0176) and on a mission step's
   question (#1178 from #0139.1).
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from sqlalchemy import text

from tests.test_prd252_needs_you import engine  # noqa: F401  (the module's Postgres)

NOW = datetime.now(timezone.utc)


def _card(s, ws, seq, title, status, *, source="user", agent=None, source_id=None, why=None,
          parent=None, run=None, step_of=None, done=None):
    return s.execute(text(
        "INSERT INTO board_tasks (workspace_id, workspace_seq, title, status, source_type, source_id, "
        "assigned_agent_id, blocked_reason, parent_task_id, orchestration_run_id, orchestration_task_id, "
        "completed_at) VALUES (CAST(:w AS uuid), :seq, :t, :s, :src, :sid, :agent, :why, :parent, "
        "CAST(:run AS uuid), CAST(:step AS uuid), :done) RETURNING id"),
        {"w": ws, "seq": seq, "t": title, "s": status, "src": source, "sid": source_id, "agent": agent,
         "why": why, "parent": parent, "run": run, "step": step_of, "done": done}).scalar()


def _run(s, ws, goal, state, state_type):
    return str(s.execute(text(
        "INSERT INTO orchestration_runs (workspace_id, goal, created_by, state, state_type) "
        "VALUES (CAST(:w AS uuid), :g, 'owner', :st, :stt) RETURNING id"),
        {"w": ws, "g": goal, "st": state, "stt": state_type}).scalar())


def _task(s, run, title, n, state):
    return str(s.execute(text(
        "INSERT INTO orchestration_tasks (run_id, title, sequence_number, state) "
        "VALUES (CAST(:r AS uuid), :t, :n, :st) RETURNING id"), {"r": run, "t": title, "n": n, "st": state}).scalar())


def _grant(s, ws, kind, subject_type, subject_id, *, title, expires=None):
    return s.execute(text(
        "INSERT INTO approval_grants (workspace_id, subject_type, subject_id, status, kind, question_md, reason, "
        "details, requested_at, expires_at) VALUES (CAST(:w AS uuid), :st, :sid, 'pending', :k, :q, :r, "
        "CAST('{}' AS jsonb), :at, :exp) RETURNING id"),
        {"w": ws, "st": subject_type, "sid": str(subject_id), "k": kind, "q": title if kind == "question" else None,
         "r": title if kind == "approval" else None, "at": NOW, "exp": expires or NOW + timedelta(hours=20)}).scalar()


def _agent(s, ws, name):
    return s.execute(text(
        "INSERT INTO agents (name, agent_type, workspace_id, status, configuration, owner_type) "
        "VALUES (:n, 'custom', CAST(:w AS uuid), 'active', CAST(:c AS json), 'workspace') RETURNING id"),
        {"n": name, "w": ws, "c": json.dumps({})}).scalar()


def _missions(s, ws, writer):
    """#0176 failed with steps .2 and .3 left in the Inbox; a running mission's queued
    step; #0031's plan waiting for approval."""
    ended = _run(s, ws, "Prepare the cafés for the price change", "failed", "terminal")
    card = _card(s, ws, 176, "Mission: prepare the cafés", "failed", source="orchestration", run=ended, done=NOW)
    done_task, open_task, unassigned_task = (_task(s, ended, f"Step {n}", n, st)
                                             for n, st in ((1, "verified"), (2, "queued"), (3, "pending")))
    _card(s, ws, None, "Extract the cafés' balances", "done", source="orchestration_task", parent=card, step_of=done_task)
    stranded = _card(s, ws, None, "Send the café letters", "inbox", source="orchestration_task", agent=writer,
                     parent=card, step_of=open_task)
    _card(s, ws, None, "Chase the late payers", "inbox", source="orchestration_task", parent=card, step_of=unassigned_task)
    live = _run(s, ws, "Autumn menu", "running", "active")
    live_card = _card(s, ws, 180, "Mission: autumn menu", "in_progress", source="orchestration", run=live)
    live_step = _card(s, ws, None, "Draft the menu", "inbox", source="orchestration_task", agent=writer,
                      parent=live_card, step_of=_task(s, live, "Draft the menu", 1, "queued"))
    plan = _run(s, ws, "Winter blend launch", "awaiting_approval", "blocked")
    plan_card = _card(s, ws, 31, "Mission: winter blend launch", "review", source="orchestration", run=plan)
    return NS(card=card, stranded=stranded, done_task=done_task, plan=plan, plan_card=plan_card,
              live=live, live_step=live_step, ended=ended)


def _cards(s, ws, writer, mac):
    from services.cli_ticket_lane import NO_CLI_HOST_REASON, NO_HOST_REASON

    days_ago = NOW - timedelta(days=7)
    return NS(
        orphan=_card(s, ws, 67, "Order oat milk", "assigned"),
        playbook=_card(s, ws, 149, "Weekly posts", "assigned", source="recipe", agent=writer, source_id="exec-5f1e2a"),
        session_step=_card(s, ws, 150, "Weekly posts: step 2", "assigned", source="recipe", agent=mac,
                           source_id="recipe:exec-5f1e2a:2"),
        no_host=_card(s, ws, 161, "Cash-up check", "assigned", agent=mac, why=NO_HOST_REASON),
        no_cli=_card(s, ws, 177, "Break-even", "assigned", agent=mac,
                     why=NO_CLI_HOST_REASON.format(cli="codex", served="claude")),
        queued=_card(s, ws, 170, "Rota for October", "assigned", agent=writer),
        old_failure=_card(s, ws, 3, "Supplier check", "failed", agent=writer, done=days_ago),
        new_failure=_card(s, ws, 171, "Price list", "failed", agent=writer, done=NOW),
        review=_card(s, ws, 172, "Menu post", "review", agent=writer, done=NOW),
        cancelled=_card(s, ws, 93, "Order the cups", "cancelled", agent=writer),
    )


def _grants(s, ws, cards, missions):
    lapsed = NOW - timedelta(days=6)
    return NS(
        live=_grant(s, ws, "approval", "board_task", cards.queued, title="Spend £40 on cups"),
        lapsed=_grant(s, ws, "approval", "tool_call", "platform_delete_playbook:cef4b570", title="Delete the playbook",
                      expires=lapsed),
        on_cancelled=_grant(s, ws, "approval", "board_task", cards.cancelled, title="Order 500 cups"),
        asked_on_cancelled=_grant(s, ws, "question", "board_task", cards.cancelled, title="Which supplier?"),
        step_question=_grant(s, ws, "question", "tool_call", missions.done_task, title="Where is the invoice file?"),
        lapsed_question=_grant(s, ws, "question", "board_task", cards.no_host, title="Which till?", expires=lapsed),
    )


@pytest.fixture
def night(engine, new_session):  # noqa: F811
    ws = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'f246-needs-you')"), {"id": ws})
    writer, mac = _agent(s, ws, "Content Creator"), _agent(s, ws, "Numbers (on my Mac)")
    missions = _missions(s, ws, writer)
    cards = _cards(s, ws, writer, mac)
    grants = _grants(s, ws, cards, missions)
    s.commit()
    yield NS(ws=ws, cards=cards, missions=missions, grants=grants)
    s = new_session.sweep()
    for table in ("approval_grants", "board_tasks"):
        s.execute(text(f"DELETE FROM {table} WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws})
    s.execute(text("DELETE FROM orchestration_tasks WHERE run_id IN "
                   "(SELECT id FROM orchestration_runs WHERE workspace_id = CAST(:w AS uuid))"), {"w": ws})
    for table in ("orchestration_runs", "agents"):
        s.execute(text(f"DELETE FROM {table} WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:w AS uuid)"), {"w": ws})
    s.commit()


def test_needs_you_counts_what_the_persona_counted(night, new_session):
    from services.needs_you import needs_you

    out = needs_you(new_session(), UUID(night.ws))

    # stuck: #0067, #0149, #0161, #0177, #0176.2, #0176.3. failed: #0003 (a week old), #0171,
    # the mission's card. approval: the live grant and #0031's plan. question: the step's and
    # the lapsed one (a question can still be answered).
    assert out["counts"] == {"review": 1, "question": 2, "approval": 2, "stuck": 6, "failed": 3}
    assert out["total"] == 14
    assert sum(len(rows) for rows in out["rows"].values()) == out["total"]


def test_each_stuck_ticket_says_why(night, new_session):
    from services.needs_you import needs_you

    stuck = needs_you(new_session(), UUID(night.ws))["rows"]["stuck"]

    assert {(r["number"], r["why"]) for r in stuck} == {
        ("#0067", "no_agent"), ("#0149", "not_picked_up"), ("#0161", "no_host"), ("#0177", "no_host"),
        ("#0176.2", "mission_ended"), ("#0176.3", "mission_ended")}
    assert next(r for r in stuck if r["number"] == "#0176.2")["ticket_id"] == night.missions.stranded
    # F274: a step whose mission ended opens its mission, where it is resumed; the rest open themselves.
    assert {r["number"]: r["mission_id"] for r in stuck if r["why"] == "mission_ended"} == {
        "#0176.2": night.missions.ended, "#0176.3": night.missions.ended}
    assert all(r["mission_id"] is None for r in stuck if r["why"] != "mission_ended")


def test_a_failure_counts_until_it_is_dealt_with(night, new_session):
    from services.needs_you import needs_you

    failed = needs_you(new_session(), UUID(night.ws))["rows"]["failed"]
    assert night.cards.old_failure in {r["ticket_id"] for r in failed}        # night: gone after a day

    s = new_session()
    s.execute(text("UPDATE board_tasks SET status = 'cancelled' WHERE id = :id"), {"id": night.cards.old_failure})
    s.commit()
    assert needs_you(new_session(), UUID(night.ws))["counts"]["failed"] == 2


def test_only_an_approval_that_can_still_be_given_is_listed(night, new_session):
    from services.needs_you import needs_you

    rows = needs_you(new_session(), UUID(night.ws))["rows"]

    grants = {r["id"] for r in rows["approval"] if r["source"] == "grant"}
    assert grants == {str(night.grants.live)}                       # not the lapsed one, not #0093's
    assert str(night.grants.asked_on_cancelled) not in {r["id"] for r in rows["question"]}


def test_a_mission_plan_and_a_steps_question_carry_their_card_numbers(night, new_session):
    from services.needs_you import needs_you

    rows = needs_you(new_session(), UUID(night.ws))["rows"]

    plan = next(r for r in rows["approval"] if r["source"] == "mission")
    assert (plan["id"], plan["ticket_id"], plan["number"]) == (       # F274: `number`, as on every row
        night.missions.plan, night.missions.plan_card, "#0031")
    asked = next(r for r in rows["question"] if r["id"] == str(night.grants.step_question))
    assert asked["number"] == "#0176.1"                                # night: question #1178 had none


def test_a_mission_step_held_for_the_owners_check_is_in_review(night, new_session):
    """F242: a step the owner asked to check waits for them in Review while its
    mission waits; one whose mission ended is stuck, never both."""
    from services.needs_you import needs_you

    s = new_session()
    s.execute(text("UPDATE board_tasks SET status = 'review', review_mode = 'human' WHERE id IN (:held, :ended)"),
              {"held": night.missions.live_step, "ended": night.missions.stranded})
    s.execute(text("UPDATE orchestration_runs SET state = 'paused' WHERE id = CAST(:r AS uuid)"),
              {"r": night.missions.live})
    s.commit()
    out = needs_you(new_session(), UUID(night.ws))

    assert (out["counts"]["review"], out["counts"]["stuck"]) == (2, 6)
    assert {r["number"] for r in out["rows"]["review"]} == {"#0172", "#0180.1"}


def test_a_member_who_cannot_answer_still_sees_what_is_stuck(night, new_session):
    from services.needs_you import needs_you, needs_you_counts

    out = needs_you(new_session(), UUID(night.ws), may_answer=False)

    assert out["counts"] == {"review": 1, "question": 0, "approval": 1, "stuck": 6, "failed": 3}
    assert needs_you_counts(new_session(), UUID(night.ws), may_answer=False)["total"] == out["total"] == 11
