"""PRD-256 P256-FIX-RVW-14 (Decision D1 amended, FX-010): a playbook's timer set through
an update, and a plugin or skill taken from every agent, wait for the owner's click.

The fix-wave review: platform_schedule_playbook asked, but platform_update_playbook with a
schedule_config set the same timer live with no card ('set playbook 3 to run on a cron
schedule' is one of its own examples). And three tools changed every agent's settings
from the owner's chat with no card: platform_uninstall_plugin (unassigns the plugin from
every agent), platform_delete_workspace_skill (drops its agent_skills rows) and
platform_update_skill (forks a marketplace skill and moves its agents onto the fork).
Now each asks from a person's chat; its card names the timer from → to, or the plugin
or skill by id and the agents it is taken from or moved for; an agent's own run is
unchanged, and an update with no timer runs as before.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

from modules.tools.discovery import owner_only

KIT_TOOLS = ("platform_uninstall_plugin", "platform_delete_workspace_skill", "platform_update_skill")
TIMER = {"type": "cron", "cron_expression": "0 9 * * 1"}


def _owners_chat(user="7"):
    return {"driving_user_id": str(user), "conversation_id": str(uuid4())}


# ── The list, its verbs and its card text ─────────────────────────────────────────────

def test_each_tool_is_owner_only_with_a_verb_and_card_text():
    from modules.tools.discovery import get_action_registry
    from modules.tools.discovery.card_question import READERS

    for action in (*KIT_TOOLS, "platform_update_playbook"):
        assert get_action_registry().get(action) is not None, action
        assert action in owner_only.OWNER_ONLY_ACTIONS and owner_only.VERBS.get(action), action
        assert action in READERS, action


def test_an_update_is_owner_only_only_when_it_sets_the_timer():
    assert owner_only.is_owner_only("platform_update_playbook", {"playbook_id": 3, "schedule_config": TIMER})
    assert owner_only.is_owner_only("platform_update_playbook", '{"playbook_id": 3, "schedule_config": {"enabled": false}}')
    assert not owner_only.is_owner_only("platform_update_playbook", {"playbook_id": 3, "name": "Daily Digest"})
    assert not owner_only.is_owner_only("platform_update_playbook", {"playbook_id": 3, "schedule_config": None})
    for action in KIT_TOOLS:
        assert owner_only.is_owner_only(action, {})


# ── Rows: a café with a playbook on a Friday timer, a plugin and skills on its agents ───

def _user(db, name):
    return db.execute(text("INSERT INTO users (email, username) VALUES (:e, :u) RETURNING id"),
                      {"e": f"{name}-{uuid4().hex[:8]}@harbourline.test", "u": f"{name}-{uuid4().hex[:8]}"}).scalar()


def _agent(db, ws, name):
    from core.models import Agent

    agent = Agent(name=name, agent_type="worker", description="", status="active", configuration={},
                  workspace_id=ws, created_by="test", owner_type="workspace", owner_id=str(ws))
    db.add(agent)
    db.flush()
    return agent


def _skills(db, ws, other, buyer, roaster, theirs):
    """menu-writer (this workspace's own, on both agents and on another workspace's agent)
    and sourcing (a marketplace skill enabled here, on the buyer)."""
    from core.models.core import Skill, agent_skills
    from core.models.marketplace_plugins import WorkspaceEnabledSkill

    own = Skill(name=f"menu-writer-{uuid4().hex[:6]}", skill_type="technical", workspace_id=ws, is_active=True)
    market = Skill(name=f"sourcing-{uuid4().hex[:6]}", skill_type="technical", workspace_id=None, is_active=True)
    elsewhere = Skill(name=f"their-skill-{uuid4().hex[:6]}", skill_type="technical", workspace_id=other, is_active=True)
    db.add_all([own, market, elsewhere])
    db.flush()
    db.add(WorkspaceEnabledSkill(workspace_id=ws, skill_id=market.id))
    for agent, skill in ((buyer, own), (roaster, own), (theirs, own), (buyer, market)):
        db.execute(agent_skills.insert().values(agent_id=agent.id, skill_id=skill.id))
    db.flush()
    return own, market, elsewhere


def _plugin(db, ws, buyer, theirs):
    from core.models.marketplace_plugins import AgentAssignedPlugin, MarketplacePlugin, WorkspaceEnabledPlugin

    plugin = MarketplacePlugin(slug=f"rvw14-{uuid4().hex[:8]}", name="Shopify orders", version="1.0.0")
    idle = MarketplacePlugin(slug=f"rvw14-idle-{uuid4().hex[:8]}", name="Not enabled here", version="1.0.0")
    db.add_all([plugin, idle])
    db.flush()
    db.add(WorkspaceEnabledPlugin(workspace_id=ws, plugin_id=plugin.id))
    db.add_all([AgentAssignedPlugin(agent_id=agent.id, plugin_id=plugin.id, priority=0) for agent in (buyer, theirs)])
    db.flush()
    return plugin, idle


@pytest.fixture
def cafe(db_session, seed_workspace):
    from core.models.core import WorkflowTemplate

    db = db_session
    ws, other = UUID(seed_workspace()), UUID(seed_workspace())
    owner = _user(db, "gerard")
    db.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                    "VALUES (CAST(:ws AS uuid), :user, 'owner', TRUE)"), {"ws": str(ws), "user": owner})
    buyer, roaster, theirs = _agent(db, ws, "GREEN BUYER"), _agent(db, ws, "ROASTER"), _agent(db, other, "BRAVO BUYER")
    playbook = WorkflowTemplate(template_id=f"rvw14-{uuid4().hex[:8]}", name="Weekly cupping notes",
                                description="Write the cupping notes up.", workspace_id=ws, owner_type="workspace",
                                owner_id=str(ws), created_by="test", created_by_user_id=owner, steps=[],
                                schedule_config={"type": "manual", "cron_expression": "0 7 * * 5",
                                                 "timezone": "Europe/London", "enabled": False},
                                template_definition={"steps": [], "agents": [], "config": {}, "variables": []})
    db.add(playbook)
    db.flush()
    own, market, elsewhere = _skills(db, ws, other, buyer, roaster, theirs)
    plugin, idle = _plugin(db, ws, buyer, theirs)
    return NS(db=db, ws=ws, owner=owner, buyer=buyer, roaster=roaster, theirs=theirs, playbook=playbook,
              own=own, market=market, elsewhere=elsewhere, plugin=plugin, idle=idle)


# ── A timer through an update: the card, then the click sets it ──────────────────────

def _update(cafe, params, caller_context):
    """Through the executor, for the owner who made the playbook (F133: ``_agent_id`` is the
    agent making the call, the change made for the playbook's creator)."""
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    executor = PlatformActionExecutor(cafe.db, cafe.ws)
    executor._full_autonomy = lambda: False
    with patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)):
        return asyncio.run(executor.execute("platform_update_playbook", params, caller_context))


@pytest.fixture
def no_scheduler(monkeypatch):
    from modules.tools.discovery import handlers_playbooks

    synced = []
    monkeypatch.setattr(handlers_playbooks, "_sync_schedule", lambda playbook: synced.append(playbook.id) or (None, None))
    return synced


def test_a_timer_through_an_update_raises_the_card_and_the_click_sets_it(cafe, no_scheduler):
    from core.models.approval_grants import ApprovalGrant
    from core.services.approval_grants import grant_grant

    params = {"playbook_id": cafe.playbook.id, "schedule_config": dict(TIMER), "_agent_id": cafe.buyer.id}
    chat = _owners_chat(cafe.owner)
    ask = _update(cafe, params, chat)

    assert ask["requires_confirmation"] is True and ask["owner_only"] is True
    assert ask["question_md"].startswith(f"Set a playbook's timer 'Weekly cupping notes' (playbook #{cafe.playbook.id})")
    assert "- runs at (cron): 0 7 * * 5 → 0 9 * * 1" in ask["question_md"]
    assert "- runs: manual → cron" in ask["question_md"]
    cafe.db.refresh(cafe.playbook)
    assert cafe.playbook.schedule_config["cron_expression"] == "0 7 * * 5" and no_scheduler == []  # nothing ran

    grant_grant(cafe.db.get(ApprovalGrant, ask["grant_id"]), granted_by=f"user:{cafe.owner}")
    cafe.db.flush()
    done = _update(cafe, params, chat)

    assert done["success"] is True and done["approved_via_grant_id"] == ask["grant_id"]
    cafe.db.refresh(cafe.playbook)
    assert cafe.playbook.schedule_config["cron_expression"] == "0 9 * * 1" and no_scheduler == [cafe.playbook.id]


def test_an_update_with_no_timer_runs_with_no_card(cafe, no_scheduler):
    params = {"playbook_id": cafe.playbook.id, "name": "Friday cupping notes", "_agent_id": cafe.buyer.id}
    done = _update(cafe, params, _owners_chat(cafe.owner))
    assert done["success"] is True and "requires_confirmation" not in done
    cafe.db.refresh(cafe.playbook)
    assert cafe.playbook.name == "Friday cupping notes"


def test_a_timer_update_card_names_everything_else_the_click_changes(cafe):
    from modules.tools.discovery.card_question_playbooks import update_lines

    lines = update_lines(cafe.db, cafe.ws, "platform_update_playbook",
                         {"playbook_id": cafe.playbook.id, "name": "Friday cupping notes", "tags": ["cupping"],
                          "execution_config": {"max_retries": 0},
                          "schedule_config": {"enabled": True, "trigger_config": {"event": "order.created"}}})
    assert "- timer on: False → True" in lines
    assert "- name: Weekly cupping notes → Friday cupping notes" in lines
    assert '- trigger: (empty) → {"event": "order.created"}' in lines
    assert any(line.startswith("- tags: ") and line.endswith("→ cupping") for line in lines)
    assert any(line.startswith("- how its steps run: ") and '{"max_retries": 0}' in line for line in lines)


def test_a_schedule_config_never_names_the_playbook_the_card_reads(cafe):
    from modules.tools.discovery.card_question_playbooks import update_lines

    lines = update_lines(cafe.db, cafe.ws, "platform_update_playbook",
                         {"playbook_id": cafe.playbook.id, "schedule_config": {"playbook_id": 0, "enabled": True}})
    assert lines[0] == f"- playbook: 'Weekly cupping notes' (playbook #{cafe.playbook.id})"


# ── The three plugin and skill tools: from the owner's chat they ask; an agent's run runs ─

class _Executor:
    """PlatformActionExecutor._run_cleared's shape, on this workspace: the gates have cleared."""

    def __init__(self, db, ws):
        self.db, self.ws = db, ws

    @owner_only.asks_the_owner_first
    async def _run_cleared(self, action_name, params, caller_context, cleared, handler):
        return await handler(self.db, self.ws, params)


def _kit_params(cafe, action):
    if action == "platform_uninstall_plugin":
        return {"plugin_id": str(cafe.plugin.id)}
    if action == "platform_update_skill":
        return {"skill_id": cafe.market.id, "content": "---\nname: sourcing\n---\nBuy from Kerbside first."}
    return {"skill_id": cafe.own.id}


def _cleared(cafe, action, params, caller_context):
    handler = AsyncMock(return_value={"success": True})
    reply = asyncio.run(_Executor(cafe.db, cafe.ws)._run_cleared(action, params, caller_context, None, handler))
    return reply, handler


@pytest.mark.parametrize("action", KIT_TOOLS)
def test_each_from_the_owners_chat_asks_and_does_nothing(cafe, action):
    reply, handler = _cleared(cafe, action, _kit_params(cafe, action), _owners_chat(cafe.owner))
    assert reply["requires_confirmation"] is True and reply["owner_only"] is True and reply["grant_id"]
    handler.assert_not_called()


@pytest.mark.parametrize("action", KIT_TOOLS)
@pytest.mark.parametrize("lane", [None, {"board_task_id": 2318}, {"playbook_execution_id": "exec-1"}],
                         ids=["agent-run", "board-ticket", "playbook-step"])
def test_each_from_an_agents_run_runs(cafe, action, lane):
    reply, handler = _cleared(cafe, action, _kit_params(cafe, action), lane)
    assert reply == {"success": True}
    handler.assert_called_once()


def _asked(cafe, action, params):
    return owner_only.platform_ask(cafe.db, cafe.ws, action, params, _owners_chat(cafe.owner))


def test_the_uninstall_card_names_the_plugin_and_this_workspaces_agents(cafe):
    asked = _asked(cafe, "platform_uninstall_plugin", {"plugin_id": str(cafe.plugin.id)})["question_md"]
    assert asked.startswith("Turn a plugin off and take it from every agent:")
    assert f"- plugin: 'Shopify orders' ({cafe.plugin.slug}, plugin {cafe.plugin.id})" in asked
    assert f"- taken from: 'GREEN BUYER' (agent #{cafe.buyer.id})" in asked
    assert "BRAVO BUYER" not in asked                                   # another workspace's agent


def test_the_delete_card_names_the_skill_and_every_agent_it_is_taken_from(cafe):
    asked = _asked(cafe, "platform_delete_workspace_skill", {"skill_id": cafe.own.id})["question_md"]
    assert f"- skill: '{cafe.own.name}' (skill #{cafe.own.id}) is deleted for good" in asked
    assert (f"- taken from: 'GREEN BUYER' (agent #{cafe.buyer.id}), 'ROASTER' (agent #{cafe.roaster.id})"
            in asked) and "BRAVO BUYER" not in asked


def test_the_edit_card_says_the_fork_and_the_agents_moved_onto_it(cafe):
    asked = _asked(cafe, "platform_update_skill", _kit_params(cafe, "platform_update_skill"))["question_md"]
    assert f"(skill #{cafe.market.id}), a marketplace skill: your edit makes this workspace's own copy" in asked
    assert f"- moved onto the copy for: 'GREEN BUYER' (agent #{cafe.buyer.id})" in asked
    assert "- new content: --- name: sourcing --- Buy from Kerbside first." in asked
    own = _asked(cafe, "platform_update_skill", {"skill_id": cafe.own.id, "content": "x"})["question_md"]
    assert "this workspace's own: edited in place" in own and f"- used by: 'GREEN BUYER' (agent #{cafe.buyer.id})" in own
    assert "security scanner" not in own
    overridden = _asked(cafe, "platform_update_skill", {"skill_id": cafe.own.id, "content": "x",
                                                        "acknowledge_warnings": True})["question_md"]
    assert "- saved over the security scanner's high-severity findings" in overridden


# ── The click acts on what the card showed ───────────────────────────────────────────

def test_the_subject_is_bound_to_its_id_before_the_grant(cafe):
    from core.models.approval_grants import ApprovalGrant

    plugin = _asked(cafe, "platform_uninstall_plugin", {"plugin_id": str(cafe.plugin.id).upper()})
    assert plugin["params"]["plugin_id"] == str(cafe.plugin.id)
    assert cafe.db.get(ApprovalGrant, plugin["grant_id"]).details["params"]["plugin_id"] == str(cafe.plugin.id)
    skill = _asked(cafe, "platform_delete_workspace_skill", {"skill_id": str(cafe.own.id)})
    assert skill["params"]["skill_id"] == cafe.own.id
    assert cafe.db.get(ApprovalGrant, skill["grant_id"]).details["params"]["skill_id"] == cafe.own.id


@pytest.mark.parametrize("action, params", [
    ("platform_uninstall_plugin", lambda cafe: {"plugin_id": str(cafe.idle.id)}),          # not enabled here
    ("platform_uninstall_plugin", lambda cafe: {"plugin_id": "not-a-uuid"}),
    ("platform_delete_workspace_skill", lambda cafe: {"skill_id": cafe.market.id}),        # a marketplace skill
    ("platform_delete_workspace_skill", lambda cafe: {"skill_id": cafe.elsewhere.id}),     # another workspace's
    ("platform_update_skill", lambda cafe: {"skill_id": cafe.elsewhere.id, "content": "x"}),
], ids=["plugin-not-enabled", "plugin-not-an-id", "delete-marketplace", "delete-elsewhere", "edit-elsewhere"])
def test_what_the_tool_would_refuse_is_never_asked_about(cafe, action, params):
    from core.models.approval_grants import ApprovalGrant

    reply = _asked(cafe, action, params(cafe))
    assert reply["success"] is False and "requires_confirmation" not in reply and "nothing was asked" in reply["error"]
    assert cafe.db.query(ApprovalGrant).filter(ApprovalGrant.workspace_id == cafe.ws).count() == 0
