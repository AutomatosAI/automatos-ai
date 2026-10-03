"""PRD-252 R4 (step 3c) — every text names a ticket by its number, and an id never
follows a '#'.

Auto's ticket tools take "#0042" as a ticket's number. A text that still put an
id behind a '#' ("ticket #612", "board task #12" for a schedule) told Auto a
number that names a different ticket, or none. The CLI lane, the scheduler, the
session host, step files, the mission wait, the reconciler, the watcher, a gated
call's card and the question cards now say "ticket #0042", or "ticket 612" for a
ticket with no number.
"""
from __future__ import annotations

from types import SimpleNamespace as NS

import pytest


def _ticket(**over):
    return NS(**{"id": 612, "workspace_seq": 42, "source_type": "user", "title": "Weekly numbers", **over})


# ── the CLI lane's wait ─────────────────────────────────────────────────────

def test_the_lane_gives_up_in_the_tickets_number(monkeypatch):
    from services import cli_ticket_lane as lane

    monkeypatch.setattr(lane, "_ticket_is_alive", lambda current: False)
    out = lane._still_running(612, "ticket #0042", NS(status="in_progress"), 61.9)

    assert out["error"].startswith("ticket #0042 is still running after 61 s")
    assert (out["task_id"], out["timed_out"], out["waited_s"]) == (612, True, 61)


def test_the_lane_waits_longer_only_for_a_live_ticket_under_its_ceiling(monkeypatch):
    from services import cli_ticket_lane as lane

    alive = {"yes": True}
    monkeypatch.setattr(lane, "_ticket_is_alive", lambda current: alive["yes"])
    assert lane._past_deadline(None, 10, None, None) == lane.WAITING            # no deadline
    assert lane._past_deadline(None, 10, 60, 600) == lane.WAITING               # before the soft one
    assert lane._past_deadline(None, 90, 60, 600) == lane.WAIT_LONGER           # alive, under the hard one
    assert lane._past_deadline(None, 600, 60, 600) == lane.GIVE_UP              # at the hard one
    alive["yes"] = False
    assert lane._past_deadline(None, 90, 60, 600) == lane.GIVE_UP               # not alive: no extension


# ── the scheduler ───────────────────────────────────────────────────────────

def test_a_schedule_is_never_named_with_a_ticket_number():
    from services.scheduled_task_steps import DELIVER_BOARD_TASK, DELIVER_CHAT, scheduled_message

    board = scheduled_message(12, "one_shot", DELIVER_BOARD_TASK, {"title": "Weekly numbers"}, None)
    chat = scheduled_message(12, "recurring", DELIVER_CHAT, {}, "Numbers")

    assert board == "Scheduled one_shot board task (schedule 12) 'Weekly numbers' — filed in the Inbox when it fires"
    assert chat == "Scheduled recurring task (schedule 12) for agent 'Numbers'" and "#" not in board + chat


def test_the_chat_hears_the_filed_ticket_by_its_number(monkeypatch):
    import services.chat_messenger as messenger
    from services.scheduled_task_steps import tell_the_chat

    said = []
    monkeypatch.setattr(messenger, "deliver_background_message", lambda db, **kw: said.append(kw))
    tell_the_chat(None, NS(workspace_id="ws", origin_chat_id="chat-1"), _ticket())

    assert said[0]["text"] == "Filed board ticket #0042: Weekly numbers" and said[0]["link_id"] == "612"


def test_a_request_is_refused_in_the_order_it_always_was():
    from services.scheduled_task_steps import DELIVER_BOARD_TASK, DELIVER_CHAT, request_error, schedule_error

    assert request_error("weekly", "x", 1, None, 2, {}) == "task_type must be 'one_shot' or 'recurring'"
    assert request_error("one_shot", DELIVER_CHAT, None, None, None, {}) == "A creator is required (an agent or a user)"
    assert request_error("one_shot", DELIVER_CHAT, 1, None, None, {}) == "Chat delivery needs a target agent"
    assert request_error("one_shot", DELIVER_BOARD_TASK, None, "user-1", None, {"title": " "}) == "A board task needs a title"
    assert request_error("recurring", DELIVER_BOARD_TASK, None, "user-1", None, {"title": "Rota"}) is None
    assert schedule_error("one_shot", "2020-01-01T09:00:00Z") == "Schedule datetime must be in the future"
    assert schedule_error("recurring", "not cron").startswith("Invalid cron expression")
    assert schedule_error("recurring", "0 9 * * 1") is None


# ── the session host and step files ─────────────────────────────────────────

def test_a_held_command_and_a_step_file_name_their_ticket():
    from services.cli_host_service import session_hold_question
    from services.step_files import StepFile

    assert session_hold_question(612, {"subject": "pip --version"}, ticket="ticket #0042").startswith(
        "**Allow this command in ticket #0042?**")
    assert session_hold_question(612, {"subject": "pip --version"}).startswith("**Allow this command in ticket 612?**")
    assert StepFile("d1", "offer.md", 612, "Draft the offer", "ticket #0051.2").ticket_name == "ticket #0051.2"
    assert StepFile("d1", "offer.md", 612, "Draft the offer").ticket_name == "ticket 612"     # never "#612"


@pytest.mark.parametrize("ticket_number, expected", [("#0042", "#0042 step 1 done"), (None, "ticket 612 step 1 done")])
def test_a_stalled_run_names_its_step_tickets(ticket_number, expected):
    from services.task_reconciler import _stalled_error

    error = _stalled_error(300, "running", [{"id": 612, "number": ticket_number, "step": 1, "status": "done"}])

    assert error.startswith("Stalled: no progress for 300s (status was 'running')") and expected in error
