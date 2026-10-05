"""F334 (night 10): a card whose own answer says it isn't done waits for the owner.

With review mode auto, #2084 answered "I couldn't make the invoice you asked for. … I
won't call it done." and #2093 "Invoice HL-2026-0142 isn't done." Both closed done.
Such a card now goes to review, with the agent's own sentence quoted on it and
"Says not done" as its reason. A negative sentence about the business ("the
delivery isn't done on Sundays"), a step after the work ("isn't done until you
approve"), a finding ("I couldn't find a 14-day term") and the owner's words quoted
back ("You said the last invoice isn't done") are not a verdict on the work, and the
card still closes done.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

import pytest

from core.services import ticket_reasons as tr
from services.said_not_done import said_not_done_note

WS = "febae41b-374b-4580-a5ef-f698bdd382e4"
ANSWER_2084 = (
    "I couldn't make the invoice you asked for. The PDF I generated is still not checked, and it doesn't use "
    "your layout or logo.\n\n"
    "- **PDF not opened:** the generator saved it to Deliverables, but I can't download it from this session, so "
    "I haven't opened it. The generator's text preview shows every line, including the £269.00 total, but I "
    "can't tell whether the page is blank again. I won't call it done.\n"
    "- **Colours:** navy and copper are not applied.")
ANSWER_2093 = (
    "Invoice HL-2026-0142 isn’t done. I generated a PDF, but I couldn't open it, and it has no logo.\n\n"
    "**What's on the PDF**\n"
    "- 12 kg Harbour Blend at £22.00/kg is £264.00.\n"
    "- Carriage is £5.00, the rate for drops of 12 kg and over. The total is £269.00 with no VAT.")
ANSWER_2069 = ("I generated the quote PDF, but I haven't opened it, and it probably isn't on your brand kit yet. "
               "So this card isn't done.\n\n**What the PDF says**\n- The total is £200.50.")
FINISHED = "Invoice HL-2026-0142 for Lantern Kitchen is in Deliverables: 12 kg Harbour Blend, £269.00, due 4 Nov."
SUNDAYS = "The delivery isn't done on Sundays, so Lantern Kitchen's 12 kg goes on Monday's Bath drop (£5.00)."
UNTIL_YOU_APPROVE = ("Here is the reorder email to Maya.\n\nThe card isn't done until you approve the wording: "
                     "two 60 kg sacks of Kirinyaga AA at £9.75/kg.")
A_FINDING = "I couldn't find a 14-day term for Lantern Kitchen, so the invoice uses your standard 30 days."
YOUR_WORDS = "You said the last invoice isn't done, so I rebuilt it: £269.00, due 4 Nov, with the table filled."
A_CUSTOMER_DRAFT = ("Hi Rosa,\n\nI'm afraid your order isn't done yet: the roaster is down until Thursday.\n\n"
                    "Cheers, Gerard (Harbourline Coffee Roasters)")


@pytest.fixture
def finish(monkeypatch):
    """A board card's run, review mode auto, ending through ``finalize_board_task_run``."""
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

    def run(answer):
        task = NS(id=2084, status="in_progress", result=None, error_message=None, completed_at=None,
                  runtime_ref=None, review_mode="auto", source_type="user",
                  title="Invoice HL-2026-0142 — Lantern Kitchen", description="Make the invoice on our layout.")
        session = NS(get=lambda *a, **k: task, commit=lambda: None, rollback=lambda: None)
        asyncio.run(board_tasks.finalize_board_task_run(
            session, task_id=2084, workspace_id=WS, agent_id=341,
            exec_result={"status": "success", "result": answer}, review_mode="auto"))
        return task
    return run


@pytest.mark.parametrize("answer, quoted", [
    (ANSWER_2084, "I couldn't make the invoice you asked for."),
    (ANSWER_2093, "Invoice HL-2026-0142 isn't done."),
    (ANSWER_2069, "this card isn't done"),
])
def test_a_card_that_says_it_isnt_done_waits_for_you_saying_why(finish, answer, quoted):
    task = finish(answer)

    assert task.status == "review"                                       # night 10: done
    assert task.result.startswith(answer)                                # nothing the agent wrote is lost
    assert task.result.endswith(f'{tr.SAYS_NOT_DONE_NOTE_PREFIX} "{quoted}" Sent to review instead of done.')
    assert tr.review_reason(task) == tr.SAYS_NOT_DONE


def test_an_agent_that_wont_call_it_done_is_heard_anywhere_in_its_answer():
    assert said_not_done_note("Here is the draft.\n\nI won't call it done until I've opened the PDF.")
    assert said_not_done_note("Why I can’t call it done\n- The page may be blank.")
    assert said_not_done_note("**Not done:** the generator returned a blank page.")


@pytest.mark.parametrize("answer", [FINISHED, SUNDAYS, UNTIL_YOU_APPROVE, A_FINDING, YOUR_WORDS, A_CUSTOMER_DRAFT])
def test_a_finished_answer_still_closes_done(finish, answer):
    task = finish(answer)

    assert said_not_done_note(answer) is None
    assert (task.status, task.result) == ("done", answer)
