"""F321 (night 9b): a playbook Auto builds has an agent on every step, and Auto's run of
one no agent can do is refused before a run or a card is made.

Chat 5247a359: platform_add_playbook_step took two steps with ``agent_id: null`` for
"Monday green stock" (playbook 115), and platform_execute_playbook made run #0068 and
its card and answered success, so Auto said it had started it. The run failed at once:
"steps 1 and 2 have no agent". These go through the handlers Auto's tools run
(platform_executor.PLATFORM_HANDLERS).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID, uuid4

import pytest

STEP = "Report every coffee's green stock from the shop system and flag anything under 50 kg."


@pytest.fixture
def roastery(db_session, seed_workspace, monkeypatch):
    """Harbourline: three agents, and nothing a run would start really starts."""
    import modules.tools.discovery.handlers_watches as watches
    import services.concurrency_guard as guard
    import services.playbook_engine as engine
    from core.models import Agent

    async def allowed(workspace_id, db):
        return NS(allowed=True, reason="")

    launched = []
    monkeypatch.setattr(guard, "check_concurrency", allowed)
    monkeypatch.setattr(engine, "get_playbook_engine", lambda: NS(launch=lambda **kw: launched.append(kw)))
    monkeypatch.setattr(watches, "auto_create_watch", lambda *a, **k: None)
    ws = UUID(seed_workspace())

    def agent(name, job_title=None):
        made = Agent(name=name, job_title=job_title, agent_type="custom", description="", status="active",
                     configuration={}, workspace_id=ws, created_by="test", owner_type="workspace", owner_id=str(ws))
        db_session.add(made)
        db_session.flush()
        return made

    return NS(db=db_session, ws=ws, launched=launched, watchdog=agent("Shopify Inventory Watchdog"),
              analyst=agent("Analyst"), ba=agent("Shopify Business Analyst", "Business Analyst"))


def _playbook(roastery, steps):
    from core.models.core import WorkflowTemplate

    made = WorkflowTemplate(template_id=f"f321-{uuid4().hex[:8]}", name="Monday green stock", workspace_id=roastery.ws,
                            description="Every coffee's green stock.", template_definition={"steps": steps},
                            steps=steps, created_by="platform")
    roastery.db.add(made)
    roastery.db.flush()
    return made


def _tool(action, roastery, params):
    from modules.tools.discovery.platform_executor import PLATFORM_HANDLERS

    return asyncio.run(PLATFORM_HANDLERS[action](roastery.db, roastery.ws, params))


def _runs(roastery, playbook):
    from core.models.core import BoardTask, RecipeExecution

    runs = roastery.db.query(RecipeExecution).filter(RecipeExecution.recipe_id == playbook.id).count()
    cards = roastery.db.query(BoardTask).filter(BoardTask.workspace_id == roastery.ws).count()
    return runs, cards


def test_a_step_with_no_agent_is_not_added_and_auto_is_told_who_can_do_it(roastery):
    playbook = _playbook(roastery, [])

    out = _tool("platform_add_playbook_step", roastery,
                {"playbook_id": playbook.id, "prompt_template": STEP, "agent_id": None, "order": 0})

    assert out["success"] is False
    assert "a step with none can't run, so no step was added" in out["error"]
    assert f"{roastery.watchdog.id}=Shopify Inventory Watchdog" in out["error"] and "ask them" in out["error"]
    roastery.db.refresh(playbook)
    assert not playbook.steps


def test_the_owners_words_for_an_agent_put_that_agent_on_the_step(roastery):
    playbook = _playbook(roastery, [])

    out = _tool("platform_add_playbook_step", roastery,
                {"playbook_id": playbook.id, "prompt_template": STEP, "agent_name": "the Inventory Watchdog"})
    by_title = _tool("platform_add_playbook_step", roastery,
                     {"playbook_id": playbook.id, "prompt_template": "Write it up.", "agent_name": "business analyst"})

    assert out["success"] is True and by_title["success"] is True
    roastery.db.refresh(playbook)
    assert [s["agent_id"] for s in playbook.steps] == [roastery.watchdog.id, roastery.ba.id]


def test_a_name_several_agents_answer_to_adds_nothing(roastery):
    playbook = _playbook(roastery, [])

    out = _tool("platform_add_playbook_step", roastery,
                {"playbook_id": playbook.id, "prompt_template": STEP, "agent_name": "Shopify"})
    exact = _tool("platform_add_playbook_step", roastery,
                  {"playbook_id": playbook.id, "prompt_template": STEP, "agent_name": "Analyst"})

    assert out["success"] is False and "several do" in out["error"] and "no step was added" in out["error"]
    assert exact["success"] is True                                     # "Analyst" is one agent's whole name
    roastery.db.refresh(playbook)
    assert [s["agent_id"] for s in playbook.steps] == [roastery.analyst.id]


def test_auto_cannot_start_a_playbook_whose_steps_have_no_agent(roastery):
    playbook = _playbook(roastery, [{"prompt_template": STEP, "agent_id": None, "order": 1},
                                    {"prompt_template": "Submit the report.", "agent_id": None, "order": 2}])

    out = _tool("platform_execute_playbook", roastery, {"playbook_name": "Monday green stock"})

    assert out["success"] is False and out["run_started"] is False
    assert out["error"] == (f'Playbook {playbook.id} "Monday green stock" can\'t run: steps 1 and 2 have no agent. '
                            "Give each of them an agent, then run it again.")
    assert _runs(roastery, playbook) == (0, 0) and roastery.launched == []      # no run, no card


def test_a_playbook_with_no_steps_is_not_started_either(roastery):
    playbook = _playbook(roastery, [])

    out = _tool("platform_execute_playbook", roastery, {"playbook_id": playbook.id})

    assert out["success"] is False and "has no steps, so no run was started" in out["error"]
    assert _runs(roastery, playbook) == (0, 0) and roastery.launched == []


def test_a_playbook_with_an_agent_on_every_step_still_starts(roastery):
    playbook = _playbook(roastery, [{"prompt_template": STEP, "agent_id": roastery.watchdog.id, "order": 1}])

    out = _tool("platform_execute_playbook", roastery, {"playbook_id": playbook.id})

    assert out["success"] is True and out["execution_id"]
    assert _runs(roastery, playbook)[0] == 1 and len(roastery.launched) == 1


# ── what Auto may say after it ──────────────────────────────────────────────

CHAT_5247A359 = [  # the turn's calls, as Auto made them, with the run refused as it is now
    ("platform_create_playbook", {"name": "Monday green stock"}, {"success": True}),
    ("platform_add_playbook_step", {"playbook_id": 115, "prompt_template": STEP, "agent_name": "Inventory Watchdog"},
     {"success": True}),
    ("platform_schedule_playbook", {"cron_expression": "0 8 * * 1", "playbook_name": "Monday green stock"},
     {"success": True}),
    ("platform_execute_playbook", {"playbook_name": "Monday green stock"},
     {"success": False, "error": "Playbook 115 \"Monday green stock\" can't run: step 2 has no agent."}),
]


@pytest.mark.parametrize("said", [
    "I've also started it for you right now. You can track its progress on card #0068.",
    "I've initiated the \"Monday green stock\" playbook.",
    "Your \"Monday green stock\" playbook is now running.",
])
def test_a_timer_set_never_backs_a_run_said_to_have_started(said):
    from modules.tools.execution.action_claims import claimed_action_not_done
    from modules.tools.execution.tool_execution_tracker import ToolExecutionTracker

    tracker = ToolExecutionTracker()
    for action, params, result in CHAT_5247A359:
        tracker.record_outcome(action, params, result)

    assert "platform_schedule_playbook" in tracker.succeeded
    assert claimed_action_not_done(said, tracker.succeeded) == "started"
    tracker.record_outcome("platform_execute_playbook", {"playbook_id": 115}, {"success": True})
    assert claimed_action_not_done(said, tracker.succeeded) is None             # a run that started backs it


def test_the_tool_no_longer_tells_auto_a_step_may_go_without_an_agent():
    from modules.tools.discovery import get_action_registry

    tool = get_action_registry().get("platform_add_playbook_step")
    agent_id = tool.parameters["properties"]["agent_id"]["description"]

    assert "default agent if not set" not in agent_id and "no default agent" in agent_id
    assert tool.parameters["properties"]["agent_name"]["type"] == "string"
