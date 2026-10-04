"""F321 (night 9b): a playbook's last agent step is told its answer is the run's.

The run's card shows the last step's answer. Runs #0085 and #0102 put two of eight
coffees on the card, "saved to the scratchpad": the last step summed up its own step.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

FIND = "List every coffee on the October card with its tasting notes."
WRITE = "Write the card copy for every coffee."


@pytest.fixture
def card_run(db_session, seed_workspace):
    """A two-agent-step playbook with a fixed document step after them, and a run of it."""
    from core.models import Agent
    from core.models.core import PLAYBOOK_DOCUMENT_STEP, RecipeExecution, WorkflowTemplate

    ws = UUID(seed_workspace())
    writer = Agent(name="Content Creator", agent_type="custom", description="", status="active", configuration={},
                   workspace_id=ws, created_by="test", owner_type="workspace", owner_id=str(ws))
    db_session.add(writer)
    db_session.flush()
    playbook = WorkflowTemplate(template_id=f"f321-{uuid.uuid4().hex[:8]}", name="Coffee card", description="d",
                                workspace_id=ws, template_definition={"steps": []}, created_by="f321",
                                steps=[{"order": 2, "agent_id": writer.id, "prompt_template": WRITE},
                                       {"order": 1, "agent_id": writer.id, "prompt_template": FIND},
                                       {"order": 3, "type": PLAYBOOK_DOCUMENT_STEP}])
    db_session.add(playbook)
    db_session.flush()
    execution_id = f"exec-{uuid.uuid4().hex[:12]}"
    db_session.add(RecipeExecution(execution_id=execution_id, recipe_id=playbook.id, workspace_id=ws,
                                   status="running", input_data={}, triggered_by="platform_action"))
    db_session.flush()
    return NS(db=db_session, ws=ws, writer=writer, run=execution_id)


def _sent(place, clean_prompt, step_order, total_steps=3, run=None):
    from services.step_lessons import a_playbook_step_carries_its_lessons

    sent = {}

    async def execute(**kwargs):
        sent.update(kwargs)
        return {"status": "success"}

    asyncio.run(a_playbook_step_carries_its_lessons(execute)(
        db=place.db, agent=place.writer, clean_prompt=clean_prompt, workspace_id=place.ws,
        recipe_execution_id=run or place.run, step_order=step_order, total_steps=total_steps))
    return sent["clean_prompt"]


def test_the_last_agent_step_is_told_its_answer_is_the_whole_run(card_run):
    from services.step_lessons import ON_THE_CARD
    from services.playbook_last_step import LAST_STEP_RULE

    prompt = _sent(card_run, WRITE, step_order=2)

    assert prompt.startswith(WRITE) and prompt.endswith(ON_THE_CARD)
    assert LAST_STEP_RULE in prompt and prompt.index(LAST_STEP_RULE) < prompt.index(ON_THE_CARD)
    assert "carrying forward what the earlier steps produced" in prompt


def test_an_earlier_step_is_not(card_run):
    from services.playbook_last_step import LAST_STEP_HEADING

    assert LAST_STEP_HEADING not in _sent(card_run, FIND, step_order=1)


def test_without_its_playbook_the_step_count_says_which_is_last(card_run):
    from services.playbook_last_step import LAST_STEP_HEADING

    assert LAST_STEP_HEADING in _sent(card_run, WRITE, step_order=2, total_steps=2, run="exec-unknown")
    assert LAST_STEP_HEADING not in _sent(card_run, FIND, step_order=1, total_steps=2, run="exec-unknown")
