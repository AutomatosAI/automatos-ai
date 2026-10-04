"""Brand kit at generation (A, night 9b → night 10): every run that drafts work is told the brand's rules.

Night 9b: the newsletter helper cited the brand voice paper and still missed the
sign-off (#0107, #1964); #1971 and #0095 were signed "[Your name]"; #1982 and #1986 said
"delightful", which the voice bans. Following the brand was left to the agent finding a
document. Now the platform gives the rules from the workspace's brand kit to a board
card's run (on both claim paths: the dispatcher's and a Claude Code session's), a
mission step and a playbook step: who signs, the tone, the banned words. A workspace
without a kit is told nothing.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from sqlalchemy import text

from tests import test_f249_a_rejects_lesson_reaches_the_agents_next_card as f249a

engine = f249a.engine
shop = f249a.shop

SIGN_OFF = "Gerard, Harbourline Coffee Roasters"
KIT = {
    "name": "Harbourline Coffee Roasters",
    "voice": {"tone": ["warm", "plain", "local"], "banned_phrases": ["delightful", "exquisite"], "sign_off": SIGN_OFF},
}


@pytest.fixture(autouse=True)
def fresh_kits():
    from services.brand_rules import forget_cached_kits

    forget_cached_kits()
    yield
    forget_cached_kits()


@pytest.fixture
def roastery(db_session, seed_workspace):
    """A workspace whose brand kit is set, another without one, and an agent in each."""
    from core.models import Agent
    from core.models.workspaces import Workspace

    def workspace(kit):
        ws = UUID(seed_workspace())
        if kit is not None:
            db_session.get(Workspace, ws).settings = {"brand_kit": kit}
        agent = Agent(name="Club newsletter helper", agent_type="custom", description="", status="active",
                      configuration={}, workspace_id=ws, created_by="test", owner_type="workspace", owner_id=str(ws))
        db_session.add(agent)
        db_session.flush()
        return NS(ws=ws, agent=agent)

    return NS(db=db_session, branded=workspace(KIT), plain=workspace(None))


def _mission_step(db, shop):
    from core.models.orchestration import OrchestrationRun, OrchestrationTask

    run = OrchestrationRun(workspace_id=shop.ws, goal="October club box", state="running", created_by="user_test",
                           config={})
    db.add(run)
    db.flush()
    step = OrchestrationTask(run_id=run.id, title="Write the box note", description="85 to 95 words.",
                             sequence_number=1, state="assigned", state_type="active", assigned_agent_id=shop.agent.id)
    db.add(step)
    db.flush()
    return step


def _says_the_rules(prompt):
    assert "## The brand's rules" in prompt
    assert f'Sign it "{SIGN_OFF}"' in prompt and "[Your name]" in prompt
    assert "Tone: warm, plain, local." in prompt
    assert 'Never use these words or phrases: "delightful", "exquisite".' in prompt


def _with_kit(shop, kit):
    s = shop.new()
    s.execute(text("UPDATE workspaces SET settings = CAST(:s AS jsonb) WHERE id = CAST(:w AS uuid)"),
              {"s": json.dumps({"brand_kit": kit}), "w": shop.ws})
    s.commit()


def test_a_board_cards_run_is_told_the_rules_on_both_claim_paths(shop):
    _with_kit(shop, KIT)
    card = f249a._ticket(shop, "Content Creator", "October club newsletter")
    for prompt in (f249a._session_prompt(shop, card), f249a._dispatch_prompt(shop, card)):
        assert prompt.startswith("October club newsletter")
        _says_the_rules(prompt)
        assert prompt.count("## The brand's rules") == 1


def test_a_board_card_in_a_workspace_without_a_kit_is_told_nothing(shop):
    card = f249a._ticket(shop, "Content Creator", "October club newsletter")
    for prompt in (f249a._session_prompt(shop, card), f249a._dispatch_prompt(shop, card)):
        assert "## The brand's rules" not in prompt


def test_a_mission_step_is_told_the_rules(roastery):
    from modules.coordination.dispatcher import MissionDispatcher

    prompt = MissionDispatcher.build_task_prompt(_mission_step(roastery.db, roastery.branded))
    assert prompt.startswith("# Task: Write the box note")
    _says_the_rules(prompt)


def test_a_playbook_step_is_told_the_rules_once(roastery):
    import api.recipe_executor as runner
    from services.brand_hooks import a_playbook_step_is_on_brand

    sent = {}

    async def execute(**kwargs):
        sent.update(kwargs)
        return {"status": "success", "result": "Hello club,\n\nThe Guji is in.\n\nBest,\n[Your name]"}

    step = a_playbook_step_is_on_brand(execute)
    out = asyncio.run(step(db=roastery.db, agent=roastery.branded.agent, clean_prompt="Write the club note.",
                           workspace_id=roastery.branded.ws))
    assert sent["clean_prompt"].startswith("Write the club note.")
    _says_the_rules(sent["clean_prompt"])
    assert out["result"].endswith(f"Best,\n{SIGN_OFF}")

    again = asyncio.run(step(db=roastery.db, agent=roastery.branded.agent, clean_prompt=sent["clean_prompt"],
                             workspace_id=roastery.branded.ws))
    assert again and sent["clean_prompt"].count("## The brand's rules") == 1       # a redo's words carry them once
    # The playbook runner runs every step through it.
    chain, fn = [], runner._execute_step
    while fn is not None:
        chain.append(fn.__code__.co_qualname)
        fn = getattr(fn, "__wrapped__", None)
    assert "a_playbook_step_is_on_brand.<locals>.wrapped" in chain


def test_a_step_in_a_workspace_without_a_kit_is_told_nothing(roastery):
    from modules.coordination.dispatcher import MissionDispatcher

    assert "## The brand's rules" not in MissionDispatcher.build_task_prompt(_mission_step(roastery.db, roastery.plain))


def test_the_rules_say_only_what_the_kit_says():
    from services.brand_rules import rules_for_kit

    assert rules_for_kit(None) is None
    assert rules_for_kit({"name": "", "voice": {"tone": [], "banned_phrases": []}, "company": {}}) is None
    only_company = rules_for_kit({"name": "", "voice": {}, "company": {"name": "Harbourline Coffee Roasters"}})
    assert 'Sign it "Harbourline Coffee Roasters"' in only_company and "Tone" not in only_company
