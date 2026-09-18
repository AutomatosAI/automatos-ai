"""PRD-245 W2 — a session asks, its ticket parks, the answer resumes it.

The loop WRITER needed on 2026-09-17: told to summarise notes that did not exist,
its question tool denied, the question buried in a report nobody was waiting for.
Now it asks, the question reaches the Questions tab and Telegram through PRD-225's
own path, the ticket parks instead of finishing, and the answer re-queues it —
resuming the SAME Claude Code session with the answer folded into its prompt.

The rule this file exists to pin: the ask is filed UNPARKED, and the TURN'S END
parks it. Parking a ticket whose session is still mid-turn would make that
session's own result undeliverable (``apply_result`` writes only a run that is
still ``in_progress``), losing its text, its usage and its deliverables.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

import pytest

from services import cli_host_service as svc
from services import session_tools as st

CTX = st.SessionContext(task_id=117, agent_id=58, agent_name="WRITER", workspace_id="ws-c1")


def _task(status="in_progress", ref=None, task_id=117):
    return NS(id=task_id, workspace_id="ws-c1", status=status, assigned_agent_id=58,
              runtime_ref=dict(ref or {}), blocked_at=None, blocked_reason=None,
              lease_until="soon", raw_prompt=None, description="turn the notes into a paragraph",
              title="WRITER", review_feedback=None)


class _Db:
    """Enough session for the bookkeeping: one task, counted commits."""

    def __init__(self, task=None):
        self.task = task
        self.commits = 0

    def query(self, _model):
        return self

    def filter(self, *_conds):
        return self

    def first(self):
        return self.task

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass


def _grant(grant_id=42, task_id=117, answer="They are in deliverables/sessions/116.", marker=True):
    details = {svc.SESSION_ASK_MARKER: {"task_id": task_id}} if marker else {}
    return NS(id=grant_id, workspace_id="ws-c1", details=details, answer_text=answer,
              answered_by="user:1", kind="question")


# ── the ask ─────────────────────────────────────────────────────────────────

def test_the_question_is_the_tickets_own_and_is_kept_short():
    tool = st.get_tool("ask_human")
    scoped = st.resolve_parameters(tool, {"question": "Which notes should I use?",
                                          "options": ["the brief", "the reports", "", "wait"]}, CTX)
    assert scoped == {"subject_type": "board_task", "subject_id": "117",
                      "question": "Which notes should I use?",
                      "options": ["the brief", "the reports", "wait"]}
    for refused in ({}, {"question": "   "}, {"question": "x" * (st.MAX_QUESTION_CHARS + 1)}):
        with pytest.raises(st.SessionToolRefused):
            st.resolve_parameters(tool, refused, CTX)


def test_the_ask_tool_brings_its_own_runner_so_the_ticket_is_not_parked_mid_turn():
    tool = st.get_tool("ask_human")
    assert tool.runner is not None, "the generic dispatcher would park the ticket while the session runs"
    assert tool.action == "platform_ask_human"       # the shared internals, still named
    assert tool.reads_only is False


def test_filing_an_ask_records_it_on_the_ticket_and_tells_the_session_what_happens(monkeypatch):
    task = _task()
    db = _Db(task)
    staged = {}

    async def fake_stage(_db, workspace_id, **kwargs):
        staged.update(kwargs)
        return {"success": True, "ask_id": 42, "parked": False}

    import modules.tools.discovery.handlers_asks as handlers
    monkeypatch.setattr(handlers, "stage_question", fake_stage)

    out = asyncio.run(svc.raise_session_ask(
        db, task_id=117, workspace_id="ws-c1", agent_id=58, agent_name="WRITER",
        question="Which notes?", options=["a"],
    ))
    assert out["success"] is True and out["result"]["ask_id"] == 42
    message = out["result"]["message"]
    assert "parks" in message and "end your turn" in message and "not wait" in message
    # filed UNPARKED, through the shared internals, with the marker the answer reads
    assert staged["park"] is None
    assert staged["details"] == {svc.SESSION_ASK_MARKER: {"task_id": 117}}
    assert staged["subject_type"] == "board_task" and staged["subject_id"] == "117"
    # …and remembered on the ticket
    asks = svc.session_asks(task.runtime_ref)
    assert [a["grant_id"] for a in asks] == [42]
    assert asks[0]["question"] == "Which notes?" and not asks[0].get("answered_at")
    assert svc.open_session_asks(task.runtime_ref) == asks


def test_a_failure_to_file_is_told_to_the_session_not_raised(monkeypatch):
    async def boom(*_a, **_k):
        raise RuntimeError("the grants table is unhappy")

    import modules.tools.discovery.handlers_asks as handlers
    monkeypatch.setattr(handlers, "stage_question", boom)
    out = asyncio.run(svc.raise_session_ask(_Db(_task()), task_id=117, workspace_id="ws-c1",
                                            agent_id=58, agent_name="WRITER", question="q"))
    assert out["success"] is False and "final message" in out["error"]


def test_an_ask_for_a_ticket_of_another_workspace_is_refused():
    out = asyncio.run(svc.raise_session_ask(_Db(None), task_id=117, workspace_id="ws-other",
                                            agent_id=58, agent_name="WRITER", question="q"))
    assert out["success"] is False and "not in this workspace" in out["error"]


# ── the park ────────────────────────────────────────────────────────────────

def test_a_turn_that_ends_with_a_question_open_parks_the_ticket_and_keeps_its_session():
    ref = svc.record_session_ask({"session_id": "sid-1", "cli_session_id": "claude-1", "host_id": "h1"},
                                 grant_id=42, question="Which notes?")
    task = _task(ref=ref)
    assert svc._park_for_answer(_Db(task), task, dict(task.runtime_ref)) == "blocked"
    assert task.blocked_reason and "42" in task.blocked_reason
    assert task.lease_until is None                       # no longer a running session
    # the next claim continues the SAME Claude Code session, on the host that ran it
    assert task.runtime_ref["resume_session_id"] == "claude-1"
    assert task.runtime_ref["resume_host_id"] == "h1"


def test_a_question_answered_while_the_turn_ran_sends_the_ticket_straight_back_to_work():
    ref = svc.record_session_ask({"cli_session_id": "claude-1", "host_id": "h1"}, grant_id=42, question="q")
    ref = svc.record_session_answer(ref, grant_id=42, answer="use the brief")
    task = _task(ref=ref)
    assert svc._park_for_answer(_Db(task), task, dict(task.runtime_ref)) == "assigned"
    assert task.blocked_reason is None and task.runtime_ref["resume_session_id"] == "claude-1"


def test_a_turn_with_no_question_finishes_the_normal_way():
    task = _task()
    assert svc._park_for_answer(_Db(task), task, dict(task.runtime_ref)) is None
    assert task.status == "in_progress"                   # untouched; the completion writer decides


# ── the answer ──────────────────────────────────────────────────────────────

def test_the_answer_resumes_a_parked_ticket():
    ref = svc.record_session_ask({"cli_session_id": "claude-1", "host_id": "h1"},
                                 grant_id=42, question="Which notes?")
    task = _task(status="blocked", ref=ref)
    task.blocked_reason = svc.PARKED_FOR_ANSWER_REASON.format(grant_id=42)
    assert svc.answer_session_ask(_Db(task), _grant()) is True
    assert task.status == "assigned" and task.blocked_reason is None
    answered = svc.session_asks(task.runtime_ref)[0]
    assert answered["answer"].startswith("They are in deliverables") and answered["answered_at"]
    assert task.runtime_ref["resume_session_id"] == "claude-1"
    assert svc.open_session_asks(task.runtime_ref) == []


def test_an_answer_that_lands_while_the_session_still_runs_is_only_recorded():
    """A running session must never be claimed twice: its own turn end picks the
    answer up and sends the ticket straight back to work."""
    ref = svc.record_session_ask({"cli_session_id": "claude-1"}, grant_id=42, question="q")
    task = _task(status="in_progress", ref=ref)
    assert svc.answer_session_ask(_Db(task), _grant()) is False
    assert task.status == "in_progress"
    assert svc.session_asks(task.runtime_ref)[0]["answer"]      # …but the answer is safe on the ticket


def test_an_answer_to_a_ticket_that_moved_on_changes_nothing():
    for status in ("done", "review", "cancelled", "failed"):
        ref = svc.record_session_ask({}, grant_id=42, question="q")
        task = _task(status=status, ref=ref)
        assert svc.answer_session_ask(_Db(task), _grant()) is False
        assert task.status == status


def test_only_a_session_ask_takes_this_path():
    assert svc.session_ask_marker(_grant(marker=False)) is None
    assert svc.session_ask_marker(NS(details={"cli_ask": {}})) is None        # no task_id
    assert svc.session_ask_marker(NS(details=None)) is None
    assert svc.session_ask_marker(_grant())["task_id"] == 117
    # a HELD COMMAND (W0) is a different marker and a different path
    hold = NS(details={svc.SESSION_HOLD_MARKER: {"request_id": "r", "task_id": 117}})
    assert svc.session_ask_marker(hold) is None and svc.session_hold_marker(hold) is not None


def test_an_answer_is_written_once():
    ref = svc.record_session_ask({}, grant_id=42, question="q")
    once = svc.record_session_answer(ref, grant_id=42, answer="first")
    twice = svc.record_session_answer(once, grant_id=42, answer="second")
    assert svc.session_asks(twice)[0]["answer"] == "first"
    assert svc.record_session_answer(ref, grant_id=999, answer="x") is ref     # unknown ask, no change


# ── the resumed session reads the answer ────────────────────────────────────

def test_the_answer_is_folded_into_the_prompt_of_the_session_that_carries_on():
    ref = svc.record_session_ask({}, grant_id=42, question="Which notes should I use?")
    ref = svc.record_session_answer(ref, grant_id=42, answer="deliverables/sessions/116/workspace-testing-brief.md")
    task = _task(ref=ref)
    prompt = svc._ticket_prompt(task)
    assert "turn the notes into a paragraph" in prompt            # the ticket itself
    assert "## Answers to your questions" in prompt
    assert "Which notes should I use?" in prompt
    assert "workspace-testing-brief.md" in prompt
    # an unanswered ask says nothing — the session is being asked to wait, not guess
    open_only = _task(ref=svc.record_session_ask({}, grant_id=7, question="unanswered"))
    assert "## Answers to your questions" not in svc._ticket_prompt(open_only)


def test_the_prompt_still_carries_reviewer_feedback_alongside_an_answer():
    ref = svc.record_session_answer(svc.record_session_ask({}, grant_id=42, question="q"),
                                    grant_id=42, answer="a")
    task = _task(ref=ref)
    task.review_feedback = "tighten the second paragraph"
    prompt = svc._ticket_prompt(task)
    assert "## Answers to your questions" in prompt and "tighten the second paragraph" in prompt
    assert task.review_feedback is None        # consumed for this attempt, as before


# ── the answer has to REACH the resumed session (round-two review) ───────────

def test_the_claim_folds_the_answer_in_and_keeps_the_ask_ledger(monkeypatch):
    """The bug every other test in this file walked past.

    ``claim_for_host`` built a brand-new ``runtime_ref`` and assigned it to the
    task BEFORE rendering the prompt, so ``_answers_fold_in`` read the new empty
    dict: no answer ever reached a resumed session. And because ``session_asks``
    was dropped on every claim, ``MAX_ASKS_PER_TICKET`` reset to zero each time —
    a ticket could ask, park, resume and ask again without end. Every other test
    here calls ``_ticket_prompt`` on a hand-built task and never goes through a
    claim, which is exactly why it went unseen.
    """
    ref = svc.record_session_answer(
        svc.record_session_ask({"cli_session_id": "claude-1", "host_id": "h1"},
                               grant_id=42, question="Which notes should I use?"),
        grant_id=42, answer="deliverables/sessions/116/workspace-testing-brief.md")
    task = _task(status="assigned", ref=ref)
    # the columns the claim reads that the file's fixture leaves out
    task.attempts = 1                      # the claim stamps the attempt it won
    task.review_mode = "auto"
    task.attachment_ids = []
    host = NS(id="h1", workspace_id="ws-c1")

    monkeypatch.setattr(svc, "claim_tasks", lambda db, **kw: [task])
    monkeypatch.setattr(svc, "_blocked_pending_approval", lambda db, t: False)
    monkeypatch.setattr(svc, "served_providers_of", lambda h: None)
    monkeypatch.setattr(svc, "default_session_folder", lambda db, ws: "/tmp/projects")
    monkeypatch.setattr(svc, "explorer_root_for", lambda *a, **k: None)
    monkeypatch.setattr(svc, "_session_system_prompt", lambda agent: "")
    monkeypatch.setattr(svc, "session_tool_names", lambda: ("board_summary",))

    claimed = svc.claim_for_host(_Db(None), host, limit=1)["tasks"][0]

    assert "## Answers to your questions" in claimed["prompt"]
    assert "Which notes should I use?" in claimed["prompt"]
    assert "workspace-testing-brief.md" in claimed["prompt"]
    # the ledger survives the claim, so the per-ticket ceiling still counts
    kept = svc.session_asks(task.runtime_ref)
    assert [a["grant_id"] for a in kept] == [42]
    # …and the answer is stamped, so a LATER resume does not show it again
    assert kept[0].get("folded_at")
    assert svc._answers_fold_in(task) == ""


def test_a_mid_turn_answer_is_folded_in_before_the_park_decides():
    """The operator answers while the turn is still running. ``apply_result``
    reads the row once at the top and writes it whole at the end, so that answer
    was overwritten and the ticket parked ``blocked`` on a question already
    answered — where nothing would ever re-queue it."""
    ref = svc.record_session_ask({"cli_session_id": "claude-1", "host_id": "h1"},
                                 grant_id=42, question="Which notes?")
    task = _task(ref=ref)

    answered = svc.session_asks(svc.record_session_answer(dict(ref), grant_id=42, answer="Use the brief."))

    class _RowDb(_Db):
        def execute(self, *_a, **_k):        # what a concurrent request committed
            return NS(first=lambda: ({svc.SESSION_ASKS_KEY: answered},))

    db = _RowDb(task)
    merged = svc._merge_fresh_session_asks(db, task, dict(ref))
    assert svc.session_asks(merged)[0]["answer"] == "Use the brief."
    # so the park sends it back to the queue instead of blocking on it
    assert svc._park_for_answer(db, task, merged) == "assigned"
    assert task.blocked_reason is None


def test_a_merge_without_a_fresh_answer_changes_nothing():
    ref = svc.record_session_ask({}, grant_id=42, question="q")
    task = _task(ref=ref)

    class _EmptyDb(_Db):
        def execute(self, *_a, **_k):
            return NS(first=lambda: ({},))

    same = svc._merge_fresh_session_asks(_EmptyDb(task), task, dict(ref))
    assert svc.session_asks(same)[0].get("answered_at") is None

    class _BrokenDb(_Db):
        def execute(self, *_a, **_k):
            raise RuntimeError("the row is not readable right now")

    assert svc._merge_fresh_session_asks(_BrokenDb(task), task, dict(ref)) == dict(ref)
