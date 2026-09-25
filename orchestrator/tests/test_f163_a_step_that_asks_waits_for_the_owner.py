"""F163 (night 5, persona B11/B42/B43) — a mission step whose answer only asks
the owner waits for the owner.

COUNTER CLERK's step ba7cacb5 (mission e0633a0d, 15:50:19Z) answered "I need this
specific information to draft the confirmation emails. Could you please provide
…?". No question card appeared, and the step passed verification. F140's test
for such an answer (services/playbook_owner_ask) now sends the question through
PRD-229's ladder, the channel a step's own ask_orchestrator uses: it lands in the
owner's Questions, the task waits parked, and the answer re-runs it with the
question and the answer in its prompt. On the real schema.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from core.models import Agent
from core.models.approval_grants import ApprovalGrant
from core.models.orchestration import OrchestrationRun, OrchestrationTask
from core.models.orchestration_enums import RunState, TaskState
from services import coordinator_service as cs
from services.clarification_ladder import pending_ask_id, render_resume_block

# Night 5, step ba7cacb5, verbatim.
BA7CACB5 = ("The information about the five biggest Bristol cafes for tasting visits, their contacts, and emails "
            "was found in the `wholesale-accounts.md` document. However, the exact days and times for Gerard's "
            "visits, and who he's hoping to see at each café, are not explicitly stated in the provided documents. "
            "I need this specific information to draft the confirmation emails. Could you please provide the "
            "confirmed day, time, and the person Gerard hopes to see for each of the five cafés?")
# Step bf3b62ad's shape: the work is done, and the answer ends on a note, not a question.
BF3B62AD = ("Here is the list of outstanding payments for the cafes Gerard is visiting:\n\n"
            + "".join(f"*   {cafe}: {owed}\n" for cafe, owed in (
                ("Quayward (Harbourside)", "No outstanding payments."), ("Tidemark Espresso", "£181.20 outstanding."),
                ("Wren & Ladle", "£189.00 outstanding."), ("Little Fathom", "£498.60 outstanding."),
                ("Lantern Yard Cafe", "No outstanding payments."))) * 4
            + "I need to check the \"wholesale-accounts.md\" document for the five biggest cafes in Bristol, as the "
              "previous search only returned partial information.")
ANSWER = "Tuesday 29 Sep: Quayward at 10:00 (Priya), Tidemark at 11:30 (Sam), Wren & Ladle at 14:00 (Ade)."


@pytest.fixture
def quiet(monkeypatch):
    """The step's side paths: bells (captured), chat narration, the field, Telegram."""
    from core.services.notification_dispatcher import NotificationDispatcher
    from modules.tools.discovery import handlers_asks

    sent = []

    async def _capture(self, event_type, title, message=None, **kwargs):
        sent.append(event_type)
        return {"dispatched_to": ["in_app"]}

    async def _no_telegram(*args, **kwargs):
        return None

    monkeypatch.setattr(NotificationDispatcher, "dispatch", _capture)
    monkeypatch.setattr(cs, "_narrate_mission", lambda *args, **kwargs: None)
    monkeypatch.setattr(cs.CoordinatorService, "_inject_task_output_into_field", AsyncMock())
    monkeypatch.setattr(handlers_asks, "_capture_question_telegram", _no_telegram)
    return sent


def _step(db, ws, *, config=None):
    agent = Agent(name="COUNTER CLERK", agent_type="chatbot", description="", status="active", configuration={},
                  model_config=None, workspace_id=ws, created_by="test", owner_type="workspace", owner_id=str(ws))
    db.add(agent)
    db.flush()
    run = OrchestrationRun(workspace_id=ws, goal="Visit the five biggest Bristol cafés", state=RunState.RUNNING.value,
                           created_by="user_test", config=config or {})
    db.add(run)
    db.flush()
    task = OrchestrationTask(run_id=run.id, title="Draft confirmation emails for café visits",
                             description="Draft one confirmation email per café visit.", sequence_number=1,
                             state=TaskState.RUNNING.value, state_type="active", assigned_agent_id=agent.id)
    db.add(task)
    db.flush()
    return run, task, agent


def _record(db, run, task, agent, output):
    result = {"status": "success", "result": output, "execution": {"tokens_used": 900}}
    asyncio.run(cs.CoordinatorService()._record_task_result(db, run, task, agent.id, result))


def _questions(db, ws):
    return db.query(ApprovalGrant).filter(ApprovalGrant.workspace_id == ws, ApprovalGrant.kind == "question").all()


def test_a_step_that_asks_waits_parked_and_the_owner_is_asked(db_session, seed_workspace, quiet):
    ws = UUID(seed_workspace())
    run, task, agent = _step(db_session, ws)
    _record(db_session, run, task, agent, BA7CACB5)

    assert task.state == TaskState.QUEUED.value          # held: never completed, never verified
    (question,) = _questions(db_session, ws)
    assert pending_ask_id(task) == question.id
    assert (question.status, question.subject_type, question.subject_id) == ("pending", "tool_call", str(task.id))
    assert question.question_md == BA7CACB5
    assert "question_pending" in quiet


def test_the_answer_re_runs_the_step_with_the_question_and_the_answer(db_session, seed_workspace, quiet):
    from api.approval_grants import apply_question_answer

    ws = UUID(seed_workspace())
    run, task, agent = _step(db_session, ws)
    _record(db_session, run, task, agent, BA7CACB5)
    (question,) = _questions(db_session, ws)

    outcome = asyncio.run(apply_question_answer(db_session, question, answer_text=ANSWER, answered_by="user:owner"))

    assert outcome.resumed and pending_ask_id(task) is None
    block = render_resume_block(task)
    assert f"The human answered: {ANSWER}" in block and BA7CACB5 in block


def test_a_step_that_did_its_work_is_recorded_as_it_is(db_session, seed_workspace, quiet):
    ws = UUID(seed_workspace())
    run, task, agent = _step(db_session, ws)
    _record(db_session, run, task, agent, BF3B62AD)
    assert task.state == TaskState.COMPLETED.value and pending_ask_id(task) is None
    assert _questions(db_session, ws) == []


def test_a_mission_a_website_visitor_started_never_asks_the_owner(db_session, seed_workspace, quiet):
    ws = UUID(seed_workspace())
    run, task, agent = _step(db_session, ws, config={"origin_surface": "widget"})
    _record(db_session, run, task, agent, BA7CACB5)
    assert task.state == TaskState.COMPLETED.value and _questions(db_session, ws) == []
