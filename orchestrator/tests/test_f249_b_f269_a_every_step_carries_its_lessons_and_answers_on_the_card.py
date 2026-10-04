"""F249 (partly fixed) and F269 (night 7b): lessons reach every run of an agent's work,
its answer goes on the card, and a draft is never guided by an agent's old drafts.

- F249: the Analyst carried its note, but the Content Creator slipped on #0199 and the
  newsletter helper on #0200. Lessons written in an Approve note ("next time count")
  never reached the next card, and no lesson reached a mission or a playbook step.
- F269: #0188.2's redo took a cut-off date and a handwritten card from last night's
  draft in Documents, given to it as a "guide"; answers landed in a PDF, a file and a
  delivery report instead of on the card.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

LINE_BEFORE = "Just the caption, no line before it."
COUNT = "Fine, I will add a line myself. Still short of 80 words, so next time count."


@pytest.fixture
def studio(db_session, seed_workspace):
    """The Content Creator, with a card sent back and a card approved with a lesson."""
    from core.models import Agent
    from core.models.core import BoardTask

    ws = UUID(seed_workspace())

    def agent(name, configuration=None):
        made = Agent(name=name, agent_type="custom", description="", status="active",
                     configuration=configuration or {}, workspace_id=ws, created_by="test", owner_type="workspace",
                     owner_id=str(ws))
        db_session.add(made)
        db_session.flush()
        return made

    creator, helper = agent("Content Creator"), agent("Club newsletter helper")
    sent_back = BoardTask(workspace_id=ws, title="Instagram caption for the Guji", status="done",
                          assigned_agent_id=creator.id,
                          planning_data={"owner_corrections": [{"note": LINE_BEFORE, "by": "user:o",
                                                                "at": "2026-10-03T19:20:00+00:00"}]})
    approved = BoardTask(workspace_id=ws, title="Newsletter opening", status="done", assigned_agent_id=helper.id,
                         runtime_ref={"session_notes": [
                             {"note": f"Approved: {COUNT}", "by": "you", "at": "2026-10-03T19:30:00+00:00"},
                             {"note": "Approved: Spot on. That is how I talk to Mel.", "by": "you",
                              "at": "2026-10-03T19:31:00+00:00"}]})
    db_session.add_all([sent_back, approved])
    db_session.flush()
    return NS(db=db_session, ws=ws, creator=creator, helper=helper, agent=agent)


def test_an_approve_note_that_says_what_to_do_next_time_is_a_lesson(studio):
    from services.ticket_redo import agent_lessons

    assert agent_lessons(studio.db, studio.ws, studio.helper.id) == [COUNT]      # praise alone teaches nothing
    assert agent_lessons(studio.db, studio.ws, studio.creator.id) == [LINE_BEFORE]


def _mission_step(studio, agent):
    from core.models.orchestration import OrchestrationRun, OrchestrationTask

    run = OrchestrationRun(workspace_id=studio.ws, goal="Christmas gift subscription", state="running",
                           created_by="user_test", config={})
    studio.db.add(run)
    studio.db.flush()
    step = OrchestrationTask(run_id=run.id, title="Draft the shop words", description="Plain and warm.",
                             sequence_number=1, state="assigned", state_type="active", assigned_agent_id=agent.id)
    studio.db.add(step)
    studio.db.flush()
    return step


def test_a_mission_step_carries_its_agents_lessons_and_where_its_answer_goes(studio):
    from modules.coordination.dispatcher import MissionDispatcher
    from services.step_lessons import ON_THE_CARD
    from services.ticket_redo import STANDING_HEADING

    prompt = MissionDispatcher.build_task_prompt(_mission_step(studio, studio.creator))

    assert prompt.startswith("# Task: Draft the shop words")
    assert STANDING_HEADING in prompt and f"- {LINE_BEFORE}" in prompt and COUNT not in prompt
    assert prompt.endswith(ON_THE_CARD)


def test_a_session_agents_step_is_left_to_its_ticket(studio):
    from modules.coordination.dispatcher import MissionDispatcher
    from services.step_lessons import ON_THE_CARD

    session = studio.agent("Numbers (on my Mac)", {"runtime": "cli"})
    prompt = MissionDispatcher.build_task_prompt(_mission_step(studio, session))
    assert ON_THE_CARD not in prompt


def test_a_playbook_step_carries_its_agents_lessons_and_where_its_answer_goes(studio):
    from services.step_lessons import ON_THE_CARD, a_playbook_step_carries_its_lessons

    sent = {}

    async def execute(**kwargs):
        sent.update(kwargs)
        return {"status": "success"}

    asyncio.run(a_playbook_step_carries_its_lessons(execute)(
        db=studio.db, agent=studio.helper, clean_prompt="Write the newsletter opening.", workspace_id=studio.ws))

    assert sent["clean_prompt"].startswith("Write the newsletter opening.")
    assert f"- {COUNT}" in sent["clean_prompt"] and sent["clean_prompt"].endswith(ON_THE_CARD)


def test_the_playbook_runner_and_the_dispatcher_run_through_them():
    import api.recipe_executor as runner
    from modules.coordination.dispatcher import MissionDispatcher

    assert runner._execute_step.__wrapped__.__name__ == "_execute_step"
    assert hasattr(MissionDispatcher.build_task_prompt, "__wrapped__")


def test_a_draft_is_never_guided_by_a_document_an_agent_wrote(studio):
    from core.models.core import Document
    from services.draft_guides import owners_own

    guide = Document(filename="Harbourline voice.md", workspace_id=studio.ws, source_type="upload")
    old_draft = Document(filename="gift-box-33.md", workspace_id=studio.ws, source_type="agent_output")
    studio.db.add_all([guide, old_draft])
    studio.db.flush()
    result = {"raw_result": {"results": [
        {"document_id": guide.id, "similarity": 0.8, "content": "Plain and warm. Never promise a date."},
        {"document_id": old_draft.id, "similarity": 0.9, "content": "Orders close 14 December. £33."}]},
        "frontend_data": None}

    kept = owners_own(studio.db, result)["raw_result"]["results"]
    assert [r["document_id"] for r in kept] == [guide.id]
