"""F293 (night 8, the Needs-you check) — Needs you counts what waits on the owner, and only that.

Against the persona's own count:
- not counted: missions paused at their budget (#0356, #0383, #0400: "Paused: spent
  $0.16 of the $0.14 budget… raise the budget or resume") or when the AI credit ran
  out (#0458), and steps that failed their mission's check while the mission ran on
  with nothing moving (#0352.2, #0433's two steps);
- counted though nobody had worked on them: cards Auto made straight into Review
  with no run and no answer (#0251, #0386);
- a mission step's Review and question rows read ``mission_id`` null (#0214.2,
  #0237.1, #0324.1, #0428.1).
"""
from __future__ import annotations

from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from sqlalchemy import text

from tests import test_f246_needs_you_counts_what_waits as f246

engine = f246.engine
night = f246.night

BUDGET_LINE = ("Paused: spent $0.16 of the $0.14 budget (the plan's 45,000-token estimate) "
               "— raise the budget or resume")
CREDIT_LINE = "Paused: the AI provider's account ran out of credit. Nothing is lost: top up, then press Resume."
CHECK_LINE = "Waiting for your check of ticket #0360.1"


def _paused(s, ws, seq, goal, *, stop_reason, detail):
    run = f246._run(s, ws, goal, "paused", "blocked")
    s.execute(text("UPDATE orchestration_runs SET stop_reason = :r, stop_detail = :d WHERE id = CAST(:id AS uuid)"),
              {"r": stop_reason, "d": detail, "id": run})
    card = f246._card(s, ws, seq, f"Mission: {goal}", "blocked", source="orchestration", run=run, why=detail)
    return NS(run=run, card=card)


@pytest.fixture
def waits(night, new_session):
    """Night 8's cases beside F246's night: two missions paused for the owner, one
    paused for the owner's check of a step, one the owner paused; a step that failed
    its running mission's check; a step held for the owner's check, with a question
    on another step of that mission; and two cards nobody worked on."""
    s = new_session()
    ws, writer = night.ws, f246._agent(s, night.ws, "Analyst")
    over_budget = _paused(s, ws, 356, "January club box", stop_reason="budget_exhausted", detail=BUDGET_LINE)
    no_credit = _paused(s, ws, 458, "Christmas price list", stop_reason="out_of_credit", detail=CREDIT_LINE)
    checking = _paused(s, ws, 360, "Roast plan", stop_reason=None, detail=CHECK_LINE)
    paused_by_you = _paused(s, ws, 361, "Shop photos", stop_reason=None, detail=None)
    running = f246._run(s, ws, "Welcome a new cafe", "running", "active")
    card = f246._card(s, ws, 352, "Mission: welcome a new cafe", "in_progress", source="orchestration", run=running)
    held = f246._card(s, ws, None, "Margin on the Rwanda", "review", source="orchestration_task", agent=writer,
                      parent=card, step_of=f246._task(s, running, "Margin", 1, "completed"), done=f246.NOW)
    s.execute(text("UPDATE board_tasks SET review_mode = 'human', started_at = :at, result = '£2.95 a bag' "
                   "WHERE id = :id"), {"at": f246.NOW, "id": held})
    failed_step = f246._card(s, ws, None, "Welcome email", "blocked", source="orchestration_task", agent=writer,
                             parent=card, step_of=f246._task(s, running, "Welcome email", 2, "failed"),
                             why="It still has placeholders where its content belongs: [Cafe Name].")
    asked_task = f246._task(s, running, "Stall card", 3, "running")
    asked = f246._card(s, ws, None, "Stall card", "blocked", source="orchestration_task", agent=writer, parent=card,
                       step_of=asked_task)
    question = f246._grant(s, ws, "question", "tool_call", asked_task, title="Which price for the stall?")
    empty = f246._card(s, ws, 251, "Draft reply to Hannah", "review", agent=writer)
    approval = f246._card(s, ws, 252, "Publish the Guji post", "review")
    s.execute(text("UPDATE board_tasks SET planning_data = CAST(:p AS json) WHERE id = :id"),
              {"p": '{"approval_action": {"type": "publish_blog", "post_id": "guji"}}', "id": approval})
    s.commit()
    return NS(over_budget=over_budget, no_credit=no_credit, checking=checking, paused_by_you=paused_by_you,
              running=running, failed_step=failed_step, held=held, asked=asked, question=question, empty=empty,
              approval=approval)


def _needs_you(night, new_session):
    from services.needs_you import needs_you

    return needs_you(new_session(), UUID(night.ws))


def test_a_mission_paused_for_its_owner_is_stuck_and_opens_the_mission(night, waits, new_session):
    stuck = {r["ticket_id"]: r for r in _needs_you(night, new_session)["rows"]["stuck"]}

    budget, credit = stuck[waits.over_budget.card], stuck[waits.no_credit.card]   # night: neither counted
    assert (budget["why"], budget["opens"], budget["mission_id"]) == ("over_budget", "mission", waits.over_budget.run)
    assert (credit["why"], credit["opens"], credit["mission_id"]) == ("out_of_credit", "mission", waits.no_credit.run)
    assert (budget["number"], budget["mission_number"], budget["mission_title"]) == (
        "#0356", "#0356", "Mission: January club box")


def test_a_mission_waiting_on_a_steps_check_or_paused_by_you_is_not_stuck(night, waits, new_session):
    stuck = {r["ticket_id"] for r in _needs_you(night, new_session)["rows"]["stuck"]}

    assert waits.checking.card not in stuck          # its step in Review is what is counted
    assert waits.paused_by_you.card not in stuck     # the owner paused it, as a ticket blocked by hand


def test_a_step_that_failed_its_running_missions_check_is_stuck(night, waits, new_session):
    row = next(r for r in _needs_you(night, new_session)["rows"]["stuck"] if r["ticket_id"] == waits.failed_step)

    # night: #0352.2 sat blocked for 20 minutes while its mission said "running", counted nowhere
    assert (row["why"], row["opens"], row["mission_id"]) == ("step_failed", "mission", waits.running)
    assert (row["number"], row["mission_number"], row["mission_title"]) == (
        "#0352.2", "#0352", "Mission: welcome a new cafe")


def test_a_card_nobody_worked_on_is_not_a_review(night, waits, new_session):
    review = {r["ticket_id"] for r in _needs_you(night, new_session)["rows"]["review"]}

    assert waits.empty not in review                 # night: #0251 and #0386 counted
    assert waits.approval in review                  # what it asks is the approval of its action


def test_a_steps_rows_name_its_mission_and_a_held_step_opens_itself(night, waits, new_session):
    rows = _needs_you(night, new_session)["rows"]
    held = next(r for r in rows["review"] if r["ticket_id"] == waits.held)
    asked = next(r for r in rows["question"] if r["id"] == str(waits.question))

    assert (held["mission_id"], held["mission_number"], held["mission_title"]) == (
        waits.running, "#0352", "Mission: welcome a new cafe")                   # night: mission_id null
    assert held["opens"] == "ticket"                 # the owner approves or sends it back on the board
    assert (asked["ticket_id"], asked["mission_id"], asked["mission_number"]) == (waits.asked, waits.running, "#0352")


def test_the_number_is_the_rows(night, waits, new_session):
    out = _needs_you(night, new_session)

    # F246's night (14) and tonight's: two paused missions and a failed step (stuck), the
    # held step and the approval card (review), the step's question.
    assert out["counts"] == {"review": 3, "question": 3, "approval": 2, "stuck": 9, "failed": 3}
    assert out["total"] == sum(len(rows) for rows in out["rows"].values()) == 20
