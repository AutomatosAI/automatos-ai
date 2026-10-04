"""F300 and F313 (night 9): an agent reads the shop database's own tables and columns
and answers a "now" question from it, never from a dated list or the owner's memory.

- F300: the Analyst asked the owner for table and column names three times (#1856,
  asks #1458, #1459, #1461); #1886 wanted "the exact plan_code" and #1891 "the
  subscription_orders schema". The owner: "I don't know table names, I run a roastery."
- F313: the Inventory Watchdog answered "how much Guji right now" with the 1 September
  list's 140 kg (#1855; the shop system said 118), and #1864, #1865 and #1869 used
  September stock with the database in reach.

Every API run of an agent's work gets the rules: a plain card, a mission step and a
playbook step.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

WS = "febae41b-374b-4580-a5ef-f698bdd382e4"
GUJI_NOW = "How much Guji Shakiso green have we got right now?"
TOP_CAFES = "Top three cafés by kg of coffee ordered, June to August."

NEVER_ASK_FOR_SCHEMA = "Never ask the owner for table, column or schema names"
SCHEMA_THROUGH_THE_TOOL = "learn its tables and columns through the tool before your first question"
LIVE_FIRST = "A question about now (\"now\", \"current\", \"today\", \"left\") is answered from the live system first"
DATED_LIST = "A dated document (a list \"as of 1 Sep\") is not today's figure"
SAY_ITS_DATE = "and give its date"


def _reads_the_live_system_first(prompt: str) -> None:
    assert NEVER_ASK_FOR_SCHEMA in prompt and SCHEMA_THROUGH_THE_TOOL in prompt          # F300
    assert LIVE_FIRST in prompt and DATED_LIST in prompt and SAY_ITS_DATE in prompt     # F313


def test_a_plain_cards_run_reads_the_schema_and_the_live_stock_first():
    from services.step_lessons import a_cards_answer_goes_on_the_card

    launched = {}
    a_cards_answer_goes_on_the_card(lambda **kwargs: launched.update(kwargs))(
        task_id=1855, agent_id=342, workspace_id=WS, prompt=GUJI_NOW, review_mode="human")

    assert launched["prompt"].startswith(GUJI_NOW)
    _reads_the_live_system_first(launched["prompt"])


@pytest.fixture
def roastery(db_session, seed_workspace):
    """The Analyst and the Inventory Watchdog, API agents."""
    from core.models import Agent

    ws = UUID(seed_workspace())

    def agent(name):
        made = Agent(name=name, agent_type="custom", description="", status="active", configuration={},
                     workspace_id=ws, created_by="test", owner_type="workspace", owner_id=str(ws))
        db_session.add(made)
        db_session.flush()
        return made

    return NS(db=db_session, ws=ws, analyst=agent("Analyst"), watchdog=agent("Inventory Watchdog"))


def test_a_mission_step_reads_the_schema_and_the_live_stock_first(roastery):
    from core.models.orchestration import OrchestrationRun, OrchestrationTask
    from modules.coordination.dispatcher import MissionDispatcher

    run = OrchestrationRun(workspace_id=roastery.ws, goal="Weekly cafe report", state="running",
                           created_by="user_test", config={})
    roastery.db.add(run)
    roastery.db.flush()
    step = OrchestrationTask(run_id=run.id, title="Top cafés by kg", description=TOP_CAFES, sequence_number=1,
                             state="assigned", state_type="active", assigned_agent_id=roastery.analyst.id)
    roastery.db.add(step)
    roastery.db.flush()

    prompt = MissionDispatcher.build_task_prompt(step)

    assert prompt.startswith("# Task: Top cafés by kg")
    _reads_the_live_system_first(prompt)


def test_a_playbook_step_reads_the_schema_and_the_live_stock_first(roastery):
    """The Monday Stock Check (playbook 114): 'Report green coffee stock'."""
    from services.step_lessons import a_playbook_step_carries_its_lessons

    sent = {}

    async def execute(**kwargs):
        sent.update(kwargs)
        return {"status": "success"}

    asyncio.run(a_playbook_step_carries_its_lessons(execute)(
        db=roastery.db, agent=roastery.watchdog, clean_prompt="Report green coffee stock; flag anything under 50 kg.",
        workspace_id=roastery.ws))

    assert sent["clean_prompt"].startswith("Report green coffee stock")
    _reads_the_live_system_first(sent["clean_prompt"])


def test_the_rules_never_reach_a_drafts_guide_search():
    """A draft's guide search reads the brief alone (F297): the rules stay out of it."""
    from services.draft_guides import _brief_only
    from services.step_lessons import ON_THE_CARD

    assert _brief_only(f"{GUJI_NOW}\n\n{ON_THE_CARD}") == GUJI_NOW
