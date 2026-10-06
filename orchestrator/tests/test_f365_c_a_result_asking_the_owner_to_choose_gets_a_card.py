"""F365 (c) (night 10c) — a result that asks the owner to choose gets a card to choose on.

Mission #2142's step #2145 (Brand Designer, "Present palette options") wrote a review
asking the owner to "reply A, B, C or D" and ended its result "a call to action: the
owner replies A, B, C or D, then gets an approval card … Still open: the owner's
choice". No card was raised (F140's test looks for a last-line question or a request
for missing information), the mission closed done, and the owner had nowhere to
reply. A result that asks the owner to pick one of a list of options is now a
question to the owner: a mission step waits parked behind it (F163), and a ticket
waits behind a card whose answers are the options (F183).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import UUID

import pytest

import api.board_tasks as bt
from core.models.orchestration_enums import TaskState
from services.clarification_ladder import pending_ask_id
from services.playbook_owner_ask import owner_question
from tests import test_f163_a_step_that_asks_waits_for_the_owner as f163
from tests import test_f183_a_ticket_that_only_asks_waits_for_the_owner as f183

quiet = f163.quiet          # the step's bells, narration, field and Telegram, faked

# #2145's result, abridged, verbatim.
RESULT_2145 = (
    "The review page for the owner is written and filed on the ticket. Nothing has been saved to the brand kit.\n\n"
    "**What's in it:** `warm-palette-review.md`, with the pictures it shows in `previews/`. It has:\n"
    "- the four palettes (A Clay & Sand, B Copper & Parchment, C Brick & Linen, D Amber & Cream), each with hex "
    "and RGB values, a readability figure and a mood line;\n"
    "- a recommendation: A, with a warm sand band behind the invoice's table headings;\n"
    "- a call to action: the owner replies A, B, C or D, then gets an approval card. Nothing is saved until they "
    "press Approve on it.\n\n"
    "**Still open:** the owner's choice. After that I file the approval card for that option, with a Brand Board "
    "drawn from its colours.")
BRIEF_2145 = "Present palette options and previews for review\nShow the owner the four warm palettes."
# #2144's result: the work, with the options named but no choice handed to the owner.
RESULT_2144 = ("All five letter renders, today's and A to D, have the same file hash. **Recommendation:** Option A "
               "with the warm table header. Choose B if you want a clearly warmer, more traditional look.")


@pytest.mark.parametrize("text, options", [
    (RESULT_2145, ["A", "B", "C", "D"]),
    ("Four palettes are in the review. Reply A–D and I'll propose that one.", ["A", "B", "C", "D"]),
    ("Three subject lines are below.\n\nPick 1, 2 or 3 and I'll send it.", ["1", "2", "3"]),
])
def test_a_result_that_asks_the_owner_to_choose_asks_with_the_options(text, options):
    assert owner_question(text, {}, BRIEF_2145) == {"question": text, "options": options}


@pytest.mark.parametrize("text, brief", [
    (RESULT_2144, "Generate previews of the proposed palettes on the brand documents"),
    ("Hi Maya,\n\nReply A or B to choose your delivery slot.\n\nCheers, Gerard", "Send Maya her delivery slots"),
    ("Here are the two slots.\n\nReply A or B to choose yours.", "Draft an email to Maya offering two delivery slots"),
])
def test_named_options_a_draft_or_a_message_to_someone_else_ask_nothing(text, brief):
    assert owner_question(text, {}, brief) is None


# ── a mission step (F163's ladder, on the real schema) ──────────────────────

def test_2145s_step_waits_parked_behind_a_question_to_the_owner(db_session, seed_workspace, quiet):
    ws = UUID(seed_workspace())
    run, task, agent = f163._step(db_session, ws)
    task.title, _, task.description = BRIEF_2145.partition("\n")       # #2145's step, not F163's emails
    db_session.flush()
    f163._record(db_session, run, task, agent, RESULT_2145)

    assert task.state == TaskState.QUEUED.value          # held: the mission does not close on it
    (question,) = f163._questions(db_session, ws)
    assert pending_ask_id(task) == question.id
    assert question.status == "pending" and "replies A, B, C or D" in question.question_md


# ── a ticket (F183's park), with the options on the card ────────────────────

@pytest.fixture
def staged(monkeypatch):
    seen = []

    async def _noop(*_a, **_k):
        return None

    async def _stage(_db, workspace_id, *, subject_type, subject_id, question, park=None, options=None, **_kw):
        seen.append({"subject": (subject_type, subject_id), "question": question, "options": options})
        park.status, park.blocked_reason = "blocked", "Awaiting human answer (ask #51)"
        return {"success": True, "ask_id": 51, "parked": True}

    for name in ("_dispatch_task_complete", "_dispatch_task_failed", "_auto_create_task_report"):
        monkeypatch.setattr(bt, name, _noop)
    monkeypatch.setattr("services.result_files.check_named_files", _noop)
    monkeypatch.setattr("services.board_events.notify_board_event", lambda *_a, **_k: None)
    monkeypatch.setattr("modules.tools.discovery.handlers_asks.stage_question", _stage)
    return seen


def _finish(task, text):
    return asyncio.run(bt.finalize_board_task_run(
        f183._Session(task), task_id=task.id, workspace_id=f183.WS, agent_id=347,
        exec_result={"status": "success", "result": text}))


def test_a_ticket_that_asks_the_owner_to_choose_waits_behind_a_card_with_the_options(staged):
    title, _, brief = BRIEF_2145.partition("\n")
    task = f183._ticket(id=2145, title=title, description=brief)

    assert _finish(task, RESULT_2145) == "blocked"
    assert staged == [{"subject": ("board_task", "2145"), "question": RESULT_2145,
                       "options": ["A", "B", "C", "D"]}]
    assert task.completed_at is None


def test_a_ticket_that_only_names_options_still_closes(staged):
    task = f183._ticket(id=2144, title="Generate previews", description="Draw the palettes on the documents.")
    assert _finish(task, RESULT_2144) == "done" and staged == []


def test_the_question_is_the_whole_result_so_the_owner_sees_what_each_option_is():
    asked = owner_question(RESULT_2145, {}, BRIEF_2145)
    assert "A Clay & Sand" in asked["question"] and "D Amber & Cream" in asked["question"]
    assert SimpleNamespace(**asked).options == ["A", "B", "C", "D"]
