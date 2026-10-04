"""F304 (night 9): an answer cites only what a tool returned in its own run.

#1858 ("September retail takings", Business Analyst) failed twice; its third run, sent
back through Auto, answered "£1,455.00 … comes from the harbourline_shop database's
retail orders table, which I accessed in a previous query that successfully retrieved
it". No earlier query had worked: the figure was Auto's chat answer (report L81).

Every agent run is told to cite only what a tool returned in that run; a board card's
answer that cites an earlier run's query, or gives the database as its source when no
database query worked in its run, gets a line saying so.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

import pytest

ANSWER_1858 = ("September 2026 retail takings: **£1,455.00**.\n\nThis figure comes from the harbourline_shop "
               "database's retail orders table, which I accessed in a previous query that successfully retrieved "
               "the September total.")
FROM_THE_DATABASE = ("63 Harvest Club boxes go out on Monday 5 October.\n\nSource: the harbourline_shop database, "
                     "subscription_orders table.")
SAID_IT_FAILED = ("I couldn't read the retail orders: my previous query failed on the order_date column, so I "
                  "can't give the September total.")
FROM_THE_BRIEF = "£1,455.00 for September. That figure comes from the brief: Auto's answer in chat, not a query."
EARLIER_RUN = "nothing from an earlier run reaches this one, so no tool read that source for this answer."
NO_QUERY = "it gives the database as its source, but no database query worked in this run."


def test_every_agent_run_is_told_to_cite_only_what_its_tools_returned():
    from services.step_lessons import ON_THE_CARD

    assert "Cite only what a tool returned in this run, by the name the tool gave it" in ON_THE_CARD
    assert "never write \"a previous query\"" in ON_THE_CARD
    assert "A figure from the brief, the owner's notes or another card is cited as coming from there." in ON_THE_CARD


@pytest.fixture
def finish(monkeypatch):
    """A board card's run ending through ``finalize_board_task_run``."""
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
        task = NS(id=1858, status="in_progress", result=None, error_message=None, completed_at=None,
                  title="September retail takings", description="What did the shop take in September?")
        session = NS(get=lambda *a, **k: task, commit=lambda: None, rollback=lambda: None)
        execution = {} if actions is None else {"execution": {"actions": actions}}
        asyncio.run(board_tasks.finalize_board_task_run(
            session, task_id=1858, workspace_id="febae41b-374b-4580-a5ef-f698bdd382e4", agent_id=343,
            exec_result={"status": "success", "result": result, **execution}, review_mode="human"))
        return task
    return run


def test_1858s_previous_query_says_no_tool_read_it(finish):
    task = finish(ANSWER_1858, actions=["platform_load_skill"])

    assert task.result.startswith(ANSWER_1858)
    assert task.result.endswith(f"it cites \"a previous query\", but {EARLIER_RUN}")


def test_a_previous_query_is_flagged_even_when_this_run_queried_too(finish):
    """What this run's query returned is not what an earlier one did."""
    assert finish(ANSWER_1858, actions=["query_database"]).result.endswith(EARLIER_RUN)


def test_the_database_as_a_source_with_no_query_that_worked_says_so(finish):
    assert finish(FROM_THE_DATABASE, actions=[]).result.endswith(NO_QUERY)


def test_the_database_as_a_source_after_a_query_that_worked_is_left_as_it_is(finish):
    assert finish(FROM_THE_DATABASE, actions=["platform_query_data"]).result == FROM_THE_DATABASE
    assert finish(FROM_THE_DATABASE, actions=["smart_query_database"]).result == FROM_THE_DATABASE


def test_saying_plainly_that_a_query_failed_is_not_a_source_claim(finish):
    assert finish(SAID_IT_FAILED, actions=[]).result == SAID_IT_FAILED


def test_a_figure_cited_to_the_brief_is_left_as_it_is(finish):
    assert finish(FROM_THE_BRIEF, actions=[]).result == FROM_THE_BRIEF


def test_a_session_that_lists_no_actions_is_left_as_it_is(finish):
    """A Claude Code session's result has no action list to check against."""
    assert finish(ANSWER_1858, actions=None).result == ANSWER_1858


def test_both_checks_add_their_lines():
    """F199's line (figures called verified with no code run) and F304's both show."""
    from services.pasted_data import NOT_VERIFIED_NOTE, unverified_figures_note

    said = f"{ANSWER_1858} I double-checked the total: £1,455.00."
    note = unverified_figures_note(said, [])
    assert note.startswith(NOT_VERIFIED_NOTE) and note.endswith(EARLIER_RUN)
