"""F171 (night 5, B34) — why the owner turned a plan down reaches the next plan.

Mission e0633a0d ("Monday 2 November checklist") took four plans. The owner
turned down 6de1b9b4 and cd562b66 in chat and e2d4006c on the card, each with
a reason, and none of the changes they asked for appeared in the next plan. The
reason was kept on the run (its RUN_REJECTED event) and never reached the
planner: the next mission was new, made from Auto's one-line summary, and the
planner's chat context (the last five messages, 500 characters each) no longer
held the owner's list. Through CoordinatorService.create_mission and
reject_plan on the real schema; the planner's model is stood in for, and its
prompt is what the test reads.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from core.models import Agent
from modules.coordination import planner as planner_module
from services import coordinator_service as cs

GOAL = "Prepare what I need for my café tasting visits by Monday 2 November."
# e2d4006c's reason, as the owner wrote it on the card (night 5).
REASON = ("Wrong job and wrong people again. I asked for four parts: COUNTER CLERK drafts a confirmation "
          "email to each café I am visiting with the day and time.")
PLAN = {"tasks": [
    {"temp_id": "task_1", "title": "Draft the café visit emails", "description": "Draft them.",
     "agent_role": "writer", "sequence_number": 1, "task_type": "llm_generation", "complexity": "simple",
     "dependencies": []},
    {"temp_id": "task_2", "title": "Check the visit times", "description": "Check them.",
     "agent_role": "researcher", "sequence_number": 2, "task_type": "review", "complexity": "simple",
     "dependencies": ["task_1"]},
]}


@pytest.fixture
def prompts(monkeypatch):
    """The planner's model, answering every plan with PLAN; the prompts it was given, in order."""
    from core.services.notification_dispatcher import NotificationDispatcher

    seen = []

    async def _respond(messages):
        seen.append(messages[-1]["content"])
        return MagicMock(content=json.dumps(PLAN))

    model = MagicMock()
    model.generate_response = _respond
    monkeypatch.setattr(planner_module, "create_llm_manager", lambda **kwargs: model)
    monkeypatch.setattr(planner_module, "match_template", lambda goal: None)
    monkeypatch.setattr(planner_module, "_build_planning_context", AsyncMock(return_value=None))
    monkeypatch.setattr(cs.AgentMatcher, "compute_signals_for_tasks", AsyncMock(return_value={}))
    monkeypatch.setattr(NotificationDispatcher, "dispatch", AsyncMock(return_value={"dispatched_to": []}))
    monkeypatch.setattr(cs, "_narrate_mission", lambda *args, **kwargs: None)
    return seen


def _roster(db, ws):
    for role in ("writer", "researcher"):
        db.add(Agent(name=role, agent_type="chatbot", description=f"The {role}.", status="active",
                     configuration={}, model_config=None, workspace_id=ws, created_by="test",
                     owner_type="workspace", owner_id=str(ws)))
    db.flush()


def _plan_in_chat(db, ws, chat_id):
    return asyncio.run(cs.CoordinatorService().create_mission(
        db, ws, GOAL, "user_test", config={"source": "chat", "origin_chat_id": chat_id}))


def test_the_owners_reason_for_turning_a_plan_down_reaches_the_next_plan(db_session, seed_workspace, prompts):
    ws = UUID(seed_workspace())
    _roster(db_session, ws)
    chat = str(uuid4())
    first = _plan_in_chat(db_session, ws, chat)
    cs.CoordinatorService().reject_plan(db_session, first.id, "user_test", reason=REASON)

    _plan_in_chat(db_session, ws, chat)

    first_prompt, next_prompt = prompts
    assert "Plans the owner turned down" not in first_prompt
    assert REASON in next_prompt                                      # old: nowhere in the prompt
    assert "Draft the café visit emails (" in next_prompt              # and what the turned-down plan had
    assert f"plan {str(first.id)[:8]}" in next_prompt


def test_another_conversations_turned_down_plan_is_not_read(db_session, seed_workspace, prompts):
    ws = UUID(seed_workspace())
    _roster(db_session, ws)
    first = _plan_in_chat(db_session, ws, str(uuid4()))
    cs.CoordinatorService().reject_plan(db_session, first.id, "user_test", reason=REASON)

    _plan_in_chat(db_session, ws, str(uuid4()))

    assert REASON not in prompts[-1]


def test_a_plan_the_owner_kept_ends_the_turned_down_run(db_session, seed_workspace, prompts):
    ws = UUID(seed_workspace())
    _roster(db_session, ws)
    chat = str(uuid4())
    first = _plan_in_chat(db_session, ws, chat)
    cs.CoordinatorService().reject_plan(db_session, first.id, "user_test", reason=REASON)
    kept = _plan_in_chat(db_session, ws, chat)                         # this one was not turned down
    # One test transaction gives every run the same now(); in the app they are minutes apart.
    now = datetime.now(timezone.utc)
    first.created_at, kept.created_at = now - timedelta(minutes=20), now - timedelta(minutes=10)
    db_session.flush()

    _plan_in_chat(db_session, ws, chat)

    assert REASON in prompts[1] and REASON not in prompts[2]
