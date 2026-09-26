"""F199 (night 6) — sums over data pasted into a ticket are computed with code,
and no figure is called verified unless code checked it.

- #1123 "Tuesday's numbers (pasted in)", agent 329: the numbers were listed
  right and added up wrong, twice.
- #1149 "What sold last week (order lines pasted in)", agent 327, 176 lines:
  every figure was wrong, and its "Verification Totals" added up its own wrong
  numbers.
Neither agent ran a tool.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

import pytest

# The shape of #1123's brief (night 6): Shopify rows pasted under the question.
BRIEF_1123 = """Forget the file for now. Here is just Tuesday 22 September, copied out of Shopify below (41 rows).

Tell me: how many orders, how much we took (and how much without the refunded ones), and the top three coffees by bags sold. Show your working briefly.

Name,Financial Status,Total,Shipping,Lineitem quantity,Lineitem name
#1205,paid,38.70,0.00,2,Guji Hambela - 250g / Filter
#1205,,,,2,Harbour Blend - 250g / Espresso
#1207,paid,32.85,0.00,2,Kiambu AA - 250g / Cafetiere
#1207,,,,1,Harbour Blend - 250g / Whole bean
#1210,paid,33.45,3.95,2,Harbour Blend - 250g / Filter
#1211,refunded,11.50,0.00,1,Sidama Bensa - 250g / Filter"""
HAND_ADDED = ("## Sales for Tuesday\n\nOrders: 5\nTaken: £128.00\n\n**Verification Totals:** I double-checked the "
              "sum of every order: £128.00.")


def test_a_brief_with_pasted_rows_gets_the_count_with_code_rule(monkeypatch):
    from services import board_dispatcher

    ticket = NS(id=1123, assigned_agent_id=329, workspace_id="dacae30f", raw_prompt=None, description=BRIEF_1123,
                title="Tuesday's numbers (pasted in)", planning_data=None, review_feedback=None, review_mode="human",
                attachment_ids=[])
    monkeypatch.setattr(board_dispatcher, "requeue_expired_leases", lambda db, **k: None)
    monkeypatch.setattr(board_dispatcher, "scan_sla_breaches", lambda db: [])
    monkeypatch.setattr(board_dispatcher, "claim_tasks", lambda db, **k: [ticket])
    session = NS(commit=lambda: None, close=lambda: None)
    cfg = NS(BOARD_DISPATCH_MAX_ATTEMPTS=3, BOARD_DISPATCH_CLAIM_BATCH=5, BOARD_DISPATCH_LEASE_SECONDS=600,
             BOARD_DISPATCH_AGENT_SLOTS=1)

    (claimed,) = board_dispatcher._claim_and_sweep(lambda: session, cfg, "w-1")["claimed"]

    assert claimed["prompt"].startswith(BRIEF_1123)
    assert "The brief carries 7 lines of pasted data. Count, sum, total or average them with code" \
        in claimed["prompt"]


def test_a_question_with_a_number_or_two_is_not_pasted_data():
    from services.pasted_data import pasted_data_rule

    assert pasted_data_rule("How many bags did we sell last week? Last month it was 440, the month before 410.") \
        is None


# ── "verified" ─────────────────────────────────────────────────────────────

@pytest.fixture
def finish(monkeypatch):
    from api import board_tasks
    import services.result_files as result_files
    import services.ticket_owner_ask as ticket_owner_ask

    async def _no(*a, **k):
        return False

    async def _none(*a, **k):
        return None

    monkeypatch.setattr(ticket_owner_ask, "park_if_the_result_asks", _no)
    monkeypatch.setattr(result_files, "check_named_files", _none)
    monkeypatch.setattr(board_tasks, "_dispatch_task_complete", _none)
    monkeypatch.setattr(board_tasks, "_auto_create_task_report", _none)

    def run(result, actions):
        task = NS(id=1149, status="in_progress", result=None, error_message=None, completed_at=None)
        session = NS(get=lambda *a, **k: task, commit=lambda: None, rollback=lambda: None)
        execution = {} if actions is None else {"execution": {"actions": actions}}
        asyncio.run(board_tasks.finalize_board_task_run(session, task_id=1149, workspace_id="dacae30f", agent_id=327,
                                                        exec_result={"status": "success", "result": result,
                                                                     **execution}, review_mode="human"))
        return task
    return run


def test_totals_called_verified_with_no_code_run_say_they_are_not(finish):
    task = finish(HAND_ADDED, actions=[])
    assert task.result.endswith("Check before relying on these figures: no code computed them in this run, so "
                                "they are not verified.")


def test_totals_code_computed_are_left_as_they_are(finish):
    task = finish(HAND_ADDED, actions=["workspace_write_file", "workspace_exec"])
    assert task.result == HAND_ADDED


def test_a_session_that_lists_no_actions_is_left_as_it_is(finish):
    """A Claude Code session runs code itself; its result has no action list."""
    assert finish(HAND_ADDED, actions=None).result == HAND_ADDED


def test_checking_something_else_is_not_a_figure_claim(finish):
    said = "I checked the brand voice guide before drafting the email to Rosie."
    assert finish(said, actions=[]).result == said


def test_checking_a_room_with_a_number_in_it_is_not_a_figure_claim(finish):
    """Code review: a bare "checked" and any digit used to be enough."""
    said = "I checked and Room 12 is free for the meeting."
    assert finish(said, actions=[]).result == said
