"""F183 (night 6) — a ticket whose result only asks the owner waits for the owner.

#1097 (agent #324, 7 s) closed done on four questions to the owner and no
newsletter, and Questions stayed empty all night. Its result now parks the
ticket behind its question, the way platform_ask_human parks one (Questions and
Telegram, blocked). The owner's answer reaches the run it re-queues, whether an
API agent or a Claude Code session picks it up. Tickets a mission or a playbook
runs keep their own asks (F163, F140).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import api.board_tasks as bt
from services.playbook_owner_ask import owner_question

WS = "00000000-0000-0000-0000-0000000000c1"
TITLE_1097 = "Draft fortnightly club newsletter for review"
BRIEF_1097 = ("Draft the next fortnightly newsletter for the coffee club (approx. 400 people). Focus on engaging "
              "content for club members. The draft must be submitted for Gerard's approval before being finalized.")
RESULT_1097 = (
    "In order to draft an engaging newsletter for the coffee club, I need to know more about what kind of content "
    "Gerard would like to include. For example:\n\n"
    "*   Are there any new coffee origins or special roasts we should highlight?\n"
    "*   Are there any upcoming events or promotions for club members?\n"
    "*   Should I include any brewing tips or recipes?\n"
    "*   Is there a particular theme or message Gerard wants to convey in this newsletter?\n\n"
    "Without this information, I can only create a very generic draft.\n\n"
    "Once I have a better understanding of the desired content, I will draft the newsletter and submit it for "
    "Gerard's approval using the `platform_submit_report` tool."
)
# #1111, abridged: a finished draft handed over for review is the ticket's work.
RESULT_1111 = ("Perfect! I've drafted the email for Maya following the Harbourline brand voice guide.\n\n"
               "**Subject:** Wholesale prices for Lamplight Café\n\n**Body:**\nHi Maya,\n\n"
               "Thanks for your interest in our wholesale coffee. Minimum order is 3kg.\n\nCheers,\n"
               "Gerard & the Harbourline crew\n\n---\n\nThe email is ready for Gerard's review before sending.")
ANSWER = "Autumn roasts, and the tasting on the 12th. Keep it short."


def test_1097s_result_asks_the_owner():
    assert owner_question(RESULT_1097, {}, f"{TITLE_1097}\n{BRIEF_1097}") == {"question": RESULT_1097,
                                                                                 "options": None}


def test_a_finished_draft_for_review_is_the_tickets_work():
    assert owner_question(RESULT_1111, {}, "Draft the wholesale price email to Maya.") is None


# ── the completion writer ───────────────────────────────────────────────────

def _ticket(**overrides):
    fields = dict(id=1097, status="in_progress", result=None, error_message=None, completed_at=None,
                  lease_until=None, runtime_ref=None, source_type="user", source_id=None, title=TITLE_1097,
                  description=BRIEF_1097, planning_data=None, blocked_at=None, blocked_reason=None,
                  workspace_id=WS)
    return SimpleNamespace(**{**fields, **overrides})


class _Session:
    def __init__(self, task):
        self.task = task

    def query(self, *_a, **_k):
        return self

    def get(self, *_a, **_k):
        return self.task

    def filter(self, *_a, **_k):
        return self

    def first(self):
        return SimpleNamespace(name="NEWSROOM")

    def commit(self):
        pass


@pytest.fixture
def finalize(monkeypatch):
    staged, completed = [], []

    async def _noop(*_a, **_k):
        return None

    async def _complete(_db, _ws, task):
        completed.append(task.id)

    async def _stage(_db, workspace_id, *, subject_type, subject_id, question, park=None, **kwargs):
        staged.append({"subject": (subject_type, subject_id), "question": question,
                       "agent": (kwargs.get("asked_by_agent_id"), kwargs.get("agent_name"))})
        park.status, park.blocked_reason = "blocked", "Awaiting human answer (ask #41)"
        return {"success": True, "ask_id": 41, "parked": True}

    monkeypatch.setattr(bt, "_dispatch_task_complete", _complete)
    for name in ("_dispatch_task_failed", "_auto_create_task_report"):
        monkeypatch.setattr(bt, name, _noop)
    monkeypatch.setattr("services.result_files.check_named_files", _noop)
    monkeypatch.setattr("services.board_events.notify_board_event", lambda *_a, **_k: None)
    monkeypatch.setattr("modules.tools.discovery.handlers_asks.stage_question", _stage)

    def run(task, text):
        return asyncio.run(bt.finalize_board_task_run(
            _Session(task), task_id=task.id, workspace_id=WS, agent_id=324,
            exec_result={"status": "success", "result": text}))
    return SimpleNamespace(run=run, staged=staged, completed=completed)


def test_1097_waits_for_the_owner_behind_its_question(finalize):
    task = _ticket()

    assert finalize.run(task, RESULT_1097) == "blocked"
    assert finalize.staged == [{"subject": ("board_task", "1097"), "question": RESULT_1097,
                                "agent": (324, "NEWSROOM")}]
    assert task.blocked_reason == "Awaiting human answer (ask #41)"
    assert (task.result, task.completed_at, finalize.completed) == (RESULT_1097, None, [])


def test_real_work_still_closes_done(finalize):
    task = _ticket()

    assert finalize.run(task, RESULT_1111) == "done"
    assert finalize.staged == []


@pytest.mark.parametrize("owner", [dict(source_type="mission", source_id="mission:7:12"),
                                   dict(source_type="recipe", source_id="exec-120")])
def test_a_missions_or_a_playbooks_ticket_keeps_its_own_ask(finalize, owner):
    task = _ticket(**owner)

    assert finalize.run(task, RESULT_1097) == "done"
    assert finalize.staged == []


# ── the answer reaches the run it re-queues ─────────────────────────────────

def _answered_ticket():
    return SimpleNamespace(
        id=1097, raw_prompt=None, description=BRIEF_1097, title=TITLE_1097, review_feedback=None,
        runtime_ref=None, assigned_agent_id=324, workspace_id=WS, review_mode="auto", attachment_ids=[],
        planning_data={"human_qa": [{"q": RESULT_1097, "a": ANSWER, "answered_by": "user:1",
                                     "at": "2026-09-26T03:00:00+00:00"}]})


def test_an_api_agent_reruns_with_the_owners_answer(monkeypatch):
    import services.board_dispatcher as dispatcher

    ticket = _answered_ticket()
    monkeypatch.setattr(dispatcher, "requeue_expired_leases", lambda *_a, **_k: {})
    monkeypatch.setattr(dispatcher, "scan_sla_breaches", lambda *_a, **_k: [])
    monkeypatch.setattr(dispatcher, "claim_tasks", lambda *_a, **_k: [ticket])
    session = SimpleNamespace(commit=lambda: None, close=lambda: None, rollback=lambda: None)
    cfg = SimpleNamespace(BOARD_DISPATCH_MAX_ATTEMPTS=3, BOARD_DISPATCH_CLAIM_BATCH=5,
                          BOARD_DISPATCH_LEASE_SECONDS=300, BOARD_DISPATCH_AGENT_SLOTS=1)

    (claimed,) = dispatcher._claim_and_sweep(lambda: session, cfg, "worker-f183")["claimed"]

    assert claimed["prompt"].startswith(BRIEF_1097)
    assert f"**The owner answered:** {ANSWER}" in claimed["prompt"]


def test_a_session_reruns_with_the_owners_answer():
    from services.cli_host_service import _ticket_prompt

    prompt = _ticket_prompt(_answered_ticket())

    assert prompt.startswith(BRIEF_1097)
    assert f"**The owner answered:** {ANSWER}" in prompt
