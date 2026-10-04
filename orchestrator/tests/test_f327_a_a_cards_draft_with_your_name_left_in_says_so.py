"""F327 (night 9b): a board card's answer that still holds "[Your name]", or a line
about a skill that is not available, says so on the card.

#1971's reorder email to the importer was signed "[Your name]" for three rounds, and
#0107's newsletter draft opened "It appears the 'Harbourline voice' skill is not
available". A plain card had no check; a mission step's (F248) already fails.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

import pytest

SIGNED_YOUR_NAME = ("Hi Maya,\n\nWe'd like to order two 60 kg sacks of Kirinyaga AA at £9.75/kg on our usual "
                    "60-day terms.\n\nBest regards,\n[Your name]")
SKILL_NOT_AVAILABLE = ("It appears the 'Harbourline voice' skill is not available, so I followed brand-voice.md.\n\n"
                       "This month's box: Guji Shakiso and Nariño Buesaco, roasted Tuesday.")
READY_TO_SEND = ("Hi Rosa,\n\nDelivery on a 10 kg order is £8.50; it's free from 12 kg.\n\n"
                 "Gerard, Harbourline Coffee Roasters")


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
        task = NS(id=1971, status="in_progress", result=None, error_message=None, completed_at=None,
                  title="Reorder Kirinyaga from Tidewater", description="Draft the reorder email to Maya.")
        session = NS(get=lambda *a, **k: task, commit=lambda: None, rollback=lambda: None)
        execution = {} if actions is None else {"execution": {"actions": actions}}
        asyncio.run(board_tasks.finalize_board_task_run(
            session, task_id=1971, workspace_id="febae41b-374b-4580-a5ef-f698bdd382e4", agent_id=341,
            exec_result={"status": "success", "result": result, **execution}, review_mode="human"))
        return task
    return run


def test_1971s_your_name_says_so_on_the_card(finish):
    task = finish(SIGNED_YOUR_NAME, actions=["platform_search_documents"])

    assert task.result.startswith(SIGNED_YOUR_NAME)
    assert task.result.endswith("it still has placeholders where its content belongs: [Your name].")


def test_0107s_skill_line_says_so_on_the_card(finish):
    task = finish(SKILL_NOT_AVAILABLE, actions=["platform_search_documents"])

    assert task.result.endswith(
        "it has a line about a tool or a skill, not the work: \"It appears the 'Harbourline voice' skill is not "
        "available, so I followed brand-voice.md.\".")


def test_a_draft_ready_to_send_is_left_as_it_is(finish):
    assert finish(READY_TO_SEND, actions=["platform_search_documents"]).result == READY_TO_SEND


def test_a_session_that_lists_no_actions_is_left_as_it_is(finish):
    """A Claude Code session's result is not checked here (its runs list no actions)."""
    assert finish(SIGNED_YOUR_NAME, actions=None).result == SIGNED_YOUR_NAME


def test_the_other_checks_lines_still_come_first_and_last():
    """F199's line, then F327's, then F304's."""
    from services.answer_sources import EARLIER_RUN_NOTE
    from services.pasted_data import NOT_VERIFIED_NOTE, unverified_figures_note

    said = (f"{SIGNED_YOUR_NAME}\n\nI double-checked the total: £1,170.00. The price comes from the "
            "harbourline_shop database, which I accessed in a previous query.")
    note = unverified_figures_note(said, [])

    assert note.startswith(NOT_VERIFIED_NOTE)
    assert "placeholders where its content belongs: [Your name]." in note
    assert note.endswith(EARLIER_RUN_NOTE.format(cited="a previous query"))
