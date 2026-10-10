"""PRD-256 P256-FIX-RVW-17 (FX-008, FX-009, FX-010): a skill, plugin or ticket card shows
the subject and the change its click runs.

The second fix-wave review: {skill_id: 5, skill_name: 'menu-writer'} showed 'menu-writer'
and gave skill 5; skill_name 'sourcing' showed 'sourcing' and gave 'Sourcing Advanced'; a
plugin card read plugin_slug while the handler read plugin_id; and platform_update_task
closing a card with an agent_id said 'status: review → done' while the click also gave the
card to an agent, by a name read again at the click. Now the ask binds the skill, the
plugin and the agent to the row the handler acts on, the card reads that row, and the
click runs on its id; none, or a partial name several skills carry, is refused before any card.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID, uuid4

import pytest

from modules.tools.discovery import owner_only

ASSIGN_SKILL, UNASSIGN_SKILL = "platform_assign_skill_to_agent", "platform_unassign_skill_from_agent"
ASSIGN_PLUGIN, UPDATE_TASK = "platform_assign_plugin_to_agent", "platform_update_task"


def _owners_chat():
    return {"driving_user_id": "7", "user_id": "user_owner", "conversation_id": str(uuid4()), "turn_id": "t-1"}


def _agent(db, ws, name):
    from core.models import Agent

    agent = Agent(name=name, agent_type="worker", description="", status="active", configuration={},
                  workspace_id=ws, created_by="test", owner_type="workspace", owner_id=str(ws))
    db.add(agent)
    db.flush()
    return agent


def _skill(db, ws, name, *, enabled_here=None):
    """A skill of this workspace, or (``enabled_here``) a marketplace skill enabled for it."""
    from core.models.core import Skill
    from core.models.marketplace_plugins import WorkspaceEnabledSkill

    skill = Skill(name=name, skill_type="technical", workspace_id=None if enabled_here else ws, is_active=True)
    db.add(skill)
    db.flush()
    if enabled_here:
        db.add(WorkspaceEnabledSkill(workspace_id=enabled_here, skill_id=skill.id))
        db.flush()
    return skill


def _plugin(db, ws, slug):
    from core.models.marketplace_plugins import MarketplacePlugin, WorkspaceEnabledPlugin

    plugin = MarketplacePlugin(slug=f"{slug}-{uuid4().hex[:8]}", name=slug, version="1.0.0")
    db.add(plugin)
    db.flush()
    db.add(WorkspaceEnabledPlugin(workspace_id=ws, plugin_id=plugin.id))
    db.flush()
    return plugin


@pytest.fixture
def cafe(db_session, seed_workspace):
    """GREEN BUYER holding menu-writer; installed: menu-writer, a marketplace 'Sourcing
    Advanced' and 'Sourcing Lite' of this workspace; two plugins enabled here; ROASTER
    holding ticket 'Cup the Guji' in Review; OPS, the agent to give it to."""
    from core.models.core import BoardTask, agent_skills

    db = db_session
    ws = UUID(seed_workspace())
    tag = uuid4().hex[:6]
    buyer, roaster, ops = _agent(db, ws, "GREEN BUYER"), _agent(db, ws, "ROASTER"), _agent(db, ws, f"OPS-{tag}")
    menu = _skill(db, ws, f"menu-writer-{tag}")
    advanced = _skill(db, ws, f"Sourcing Advanced {tag}", enabled_here=ws)
    lite = _skill(db, ws, f"Sourcing Lite {tag}")
    db.execute(agent_skills.insert().values(agent_id=buyer.id, skill_id=menu.id))
    orders, inventory = _plugin(db, ws, "shopify-orders"), _plugin(db, ws, "shopify-inventory")
    card = BoardTask(workspace_id=ws, title="Cup the Guji", description="Score it.", status="review",
                     source_type="user", assigned_agent_id=roaster.id)
    db.add(card)
    db.flush()
    return NS(db=db, ws=ws, tag=tag, buyer=buyer, roaster=roaster, ops=ops, menu=menu, advanced=advanced,
              lite=lite, orders=orders, inventory=inventory, card=card, number=f"#{card.workspace_seq:04d}")


def _ask(cafe, action, params):
    return owner_only.platform_ask(cafe.db, cafe.ws, action, params, _owners_chat())


def _grants(cafe):
    from core.models.approval_grants import ApprovalGrant

    return cafe.db.query(ApprovalGrant).filter(ApprovalGrant.workspace_id == cafe.ws).count()


def _granted_params(cafe, ask):
    from core.models.approval_grants import ApprovalGrant

    return cafe.db.get(ApprovalGrant, ask["grant_id"]).details["params"]


# ── A skill: the card and the click name the same one ───────────────────────────────

def test_a_skill_id_and_another_skills_name_card_and_click_name_the_id(cafe):
    """{skill_id: A, skill_name: <B's name>}: the handler takes the id first, so does the card."""
    from core.models.core import agent_skills
    from modules.tools.discovery.handlers_assignments import assign_skill_to_agent

    ask = _ask(cafe, ASSIGN_SKILL, {"agent_id": cafe.buyer.id, "skill_id": cafe.advanced.id,
                                    "skill_name": cafe.menu.name})
    assert ask["requires_confirmation"] is True
    after = ", ".join(sorted([cafe.menu.name, cafe.advanced.name]))
    assert f"- skills: {cafe.menu.name} → {after}" in ask["question_md"]              # the card names A
    assert "skill_name" not in ask["params"]
    clicked = _granted_params(cafe, ask)
    assert clicked["skill_id"] == cafe.advanced.id and "skill_name" not in clicked

    done = asyncio.run(assign_skill_to_agent(cafe.db, cafe.ws, clicked))           # the click runs on A
    assert done["success"] is True and done["skill"]["id"] == cafe.advanced.id
    held = cafe.db.execute(agent_skills.select().where(agent_skills.c.agent_id == cafe.buyer.id)).fetchall()
    assert {row.skill_id for row in held} == {cafe.menu.id, cafe.advanced.id}


def test_a_partial_name_one_installed_skill_contains_is_bound_to_it(cafe):
    ask = _ask(cafe, ASSIGN_SKILL, {"agent_id": cafe.buyer.id, "skill_name": f"advanced {cafe.tag}"})
    assert ask["params"]["skill_id"] == cafe.advanced.id and "skill_name" not in ask["params"]
    assert f"- skills: {cafe.menu.name} → {', '.join(sorted([cafe.menu.name, cafe.advanced.name]))}" in ask["question_md"]


def test_a_partial_name_two_installed_skills_contain_is_refused_with_no_grant(cafe):
    reply = _ask(cafe, ASSIGN_SKILL, {"agent_id": cafe.buyer.id, "skill_name": "sourcing"})
    assert reply["success"] is False and "requires_confirmation" not in reply
    assert cafe.advanced.name in reply["error"] and cafe.lite.name in reply["error"]
    assert _grants(cafe) == 0


def test_a_skill_not_installed_here_is_refused_before_any_card(cafe, seed_workspace):
    theirs = _skill(cafe.db, UUID(seed_workspace()), f"their-skill-{cafe.tag}")
    for params in ({"skill_name": "no-such-skill"}, {"skill_id": theirs.id}):
        reply = _ask(cafe, ASSIGN_SKILL, {"agent_id": cafe.buyer.id, **params})
        assert reply["success"] is False and "requires_confirmation" not in reply, params
    assert _grants(cafe) == 0


def test_a_skill_taken_is_one_the_agent_holds(cafe):
    ask = _ask(cafe, UNASSIGN_SKILL, {"agent_id": cafe.buyer.id, "skill_name": "menu-writer"})
    assert ask["params"]["skill_id"] == cafe.menu.id
    assert f"- skills: {cafe.menu.name} → (empty)" in ask["question_md"]
    reply = _ask(cafe, UNASSIGN_SKILL, {"agent_id": cafe.buyer.id, "skill_name": "sourcing"})
    assert reply["success"] is False and cafe.menu.name in reply["error"] and _grants(cafe) == 1


# ── A plugin: the card and the click name the same one ──────────────────────────────

def test_a_plugin_with_both_keys_card_and_click_name_the_id(cafe):
    from core.models.marketplace_plugins import AgentAssignedPlugin
    from modules.tools.discovery.handlers_assignments import assign_plugin_to_agent

    ask = _ask(cafe, ASSIGN_PLUGIN, {"agent_id": cafe.buyer.id, "plugin_id": str(cafe.orders.id),
                                     "plugin_slug": cafe.inventory.slug})
    assert ask["requires_confirmation"] is True
    assert f"- plugins: (empty) → {cafe.orders.slug}" in ask["question_md"]
    assert cafe.inventory.slug not in ask["question_md"]
    clicked = _granted_params(cafe, ask)
    assert clicked["plugin_id"] == str(cafe.orders.id) and "plugin_slug" not in clicked

    done = asyncio.run(assign_plugin_to_agent(cafe.db, cafe.ws, clicked))
    assert done["success"] is True and done["plugin"]["id"] == str(cafe.orders.id)
    given = cafe.db.query(AgentAssignedPlugin.plugin_id).filter(AgentAssignedPlugin.agent_id == cafe.buyer.id).all()
    assert [row.plugin_id for row in given] == [cafe.orders.id]


def test_a_plugin_by_slug_is_bound_and_one_not_enabled_here_is_refused(cafe):
    from core.models.marketplace_plugins import MarketplacePlugin

    ask = _ask(cafe, ASSIGN_PLUGIN, {"agent_id": cafe.buyer.id, "plugin_slug": cafe.inventory.slug})
    assert ask["params"]["plugin_id"] == str(cafe.inventory.id) and "plugin_slug" not in ask["params"]
    idle = MarketplacePlugin(slug=f"idle-{cafe.tag}", name="Not enabled here", version="1.0.0")
    cafe.db.add(idle)
    cafe.db.flush()
    reply = _ask(cafe, ASSIGN_PLUGIN, {"agent_id": cafe.buyer.id, "plugin_id": str(idle.id)})
    assert reply["success"] is False and "requires_confirmation" not in reply and _grants(cafe) == 1


# ── A ticket closed and given to an agent: the card says so, the click gives that id ─

def test_an_update_that_closes_and_gives_the_card_names_the_agent_change(cafe):
    ask = _ask(cafe, UPDATE_TASK, {"task_id": cafe.number, "status": "done", "agent_id": cafe.ops.id})
    assert ask["requires_confirmation"] is True
    assert "- status: review → done" in ask["question_md"]
    assert (f"- agent: ROASTER (agent #{cafe.roaster.id}) → {cafe.ops.name} (agent #{cafe.ops.id})"
            in ask["question_md"])
    assert _granted_params(cafe, ask)["agent_id"] == cafe.ops.id


def test_the_click_gives_the_card_to_the_id_the_card_showed(cafe, monkeypatch):
    """The resumed call carries the bound id alone: the click never reads a name again."""
    from modules.tools.discovery import handlers_board_task_assign
    from modules.tools.discovery.agent_refs import board_agent, gives_the_card_on_update

    ask = _ask(cafe, UPDATE_TASK, {"task_id": cafe.number, "status": "done", "agent_name": cafe.ops.name.lower()})
    clicked = _granted_params(cafe, ask)
    assert clicked["agent_id"] == cafe.ops.id and "agent_name" not in clicked

    given = {}

    async def assign(db, workspace_id, params):
        given.update(params)
        return {"success": True, "assigned_agent": cafe.ops.name, "status": "review"}

    async def update(db, workspace_id, params):
        return {"success": True, "status": params.get("status")}

    monkeypatch.setattr(handlers_board_task_assign, "assign_board_task", assign)
    done = asyncio.run(gives_the_card_on_update(update)(cafe.db, cafe.ws, clicked))
    assert done["success"] is True and given["agent_id"] == cafe.ops.id and "agent_name" not in given
    assert board_agent(cafe.db, cafe.ws, given)[0].id == cafe.ops.id                # the handler's own read


def test_an_agent_name_two_active_agents_carry_is_refused_before_the_card(cafe):
    twin = _agent(cafe.db, cafe.ws, cafe.ops.name)
    reply = _ask(cafe, UPDATE_TASK, {"task_id": cafe.number, "status": "done", "agent_name": cafe.ops.name})
    assert reply["success"] is False and "requires_confirmation" not in reply
    assert str(cafe.ops.id) in reply["error"] and str(twin.id) in reply["error"]
    assert _grants(cafe) == 0


def test_an_update_naming_its_agent_alone_has_no_click_until_bound():
    from modules.tools.discovery.agent_binding import names_the_agent_alone

    assert names_the_agent_alone(UPDATE_TASK, {"task_id": 1, "status": "done", "agent_name": "OPS"}) is True
    assert names_the_agent_alone(UPDATE_TASK, {"task_id": 1, "status": "done", "agent_id": 267}) is False
