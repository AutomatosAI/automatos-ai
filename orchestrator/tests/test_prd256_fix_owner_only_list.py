"""PRD-256 FX-010 (Decisions D1 amended, D7): the owner-only list covers every
agent-setting change, and a brief that sends or orders is reviewed by a person.

Night 12 (M4, M6, F389): GREEN BUYER's heartbeat changed (A696), MARKET-MANAGER was
deleted on one word, eleven skills were assigned and playbooks made, none with a card.
Those calls now wait for the owner's click when a person's chat makes them, and their
card says what changes; an agent's own run is unchanged. Ticket #2318 'Confirm the
order with the supplier', filed by Auto from the owner's chat, closed itself after its
agent emailed the supplier: such a brief is now filed with ``review_mode: human`` unless
the call named one, and its receipt says so. A brief that only drafts keeps 'auto'.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from modules.tools.discovery import owner_only
from modules.tools.discovery.brief_sends import REVIEW_HELD, reviewed_by_a_person, sends_or_orders

ADDED = (
    "platform_configure_agent_heartbeat", "platform_delete_agent", "platform_assign_skill_to_agent",
    "platform_unassign_skill_from_agent", "platform_assign_plugin_to_agent", "platform_create_playbook",
    "platform_schedule_playbook", "platform_delete_playbook",
)
WS = uuid4()


def _owners_chat():
    return {"driving_user_id": "7", "user_id": "user_owner", "conversation_id": str(uuid4()), "turn_id": "t-1"}


# ── The list, its verbs and its card text ─────────────────────────────────────────────

def test_every_added_action_is_registered_owner_only_with_a_verb_and_card_text():
    from modules.tools.discovery import get_action_registry
    from modules.tools.discovery.card_question import READERS

    registry = get_action_registry()
    for action in ADDED:
        assert registry.get(action) is not None, action          # the registry holds it
        assert action in owner_only.OWNER_ONLY_ACTIONS and owner_only.is_owner_only(action, {})
        assert owner_only.VERBS.get(action), action               # the card's verb
        assert action in READERS, action                           # the card's change lines (FX-008)


def test_there_is_no_tool_that_takes_a_plugin_from_an_agent():
    """'whatever of these exist in the registry': nothing is listed that the registry lacks."""
    from modules.tools.discovery import get_action_registry

    assert get_action_registry().get("platform_unassign_plugin_from_agent") is None
    assert "platform_unassign_plugin_from_agent" not in owner_only.OWNER_ONLY_ACTIONS


def test_the_hierarchy_gate_still_passes():
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "check_hierarchy_gate.py"
    ran = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=False)
    assert ran.returncode == 0, ran.stdout + ran.stderr


# ── From the owner's chat it asks; from an agent's run it runs ───────────────────────

class _Executor:
    """PlatformActionExecutor._run_cleared's shape: the gates have cleared, the handler runs."""

    @owner_only.asks_the_owner_first
    async def _run_cleared(self, action_name, params, caller_context, cleared, handler):
        return await handler(None, WS, params)


@pytest.fixture
def asks(monkeypatch):
    asked = []
    monkeypatch.setattr(owner_only, "_the_click", lambda db, ws, action, params: None)
    monkeypatch.setattr(owner_only, "platform_ask", lambda db, ws, action, params, ctx: asked.append(action) or {
        "success": False, "requires_confirmation": True, "owner_only": True, "action": action})
    return asked


def _run(action, caller_context, cleared=None):
    handler = AsyncMock(return_value={"success": True})
    params = {"agent_id": 44, "playbook_id": 9, "name": "Monday reorder", "description": "Reorder what is low."}
    reply = asyncio.run(_Executor()._run_cleared(action, params, caller_context, cleared, handler))
    return reply, handler


@pytest.mark.parametrize("action", ADDED)
def test_each_added_action_from_the_owners_chat_asks_and_does_nothing(asks, action):
    reply, handler = _run(action, _owners_chat())
    assert reply["requires_confirmation"] is True and reply["owner_only"] is True
    assert asks == [action]
    handler.assert_not_called()


@pytest.mark.parametrize("action", ADDED)
@pytest.mark.parametrize("lane", [None, {"board_task_id": 2318}, {"playbook_execution_id": "exec-1"}],
                         ids=["agent-run", "board-ticket", "playbook-step"])
def test_the_same_call_from_an_agents_run_does_not_ask(asks, action, lane):
    reply, handler = _run(action, lane)
    assert reply == {"success": True} and asks == []
    handler.assert_called_once()


def test_a_destructive_call_runs_on_the_click_the_gate_already_claimed(asks, monkeypatch):
    """An editor's delete asks at the confirmation gate first, and the gate retires its
    single-use grant when it clears: that claim is the click, not a reason for a new card."""
    claimed = NS(id=51, granted_by="user:clicker-7")
    monkeypatch.setattr(owner_only, "_claimed_at_the_gate", lambda db, ws, action, grant_id: claimed if grant_id == 51 else None)
    destructive = NS(action_def=NS(permission_level="destructive"), approved_via_grant_id=51)

    reply, handler = _run("platform_delete_agent", _owners_chat(), destructive)

    assert reply["success"] is True and reply["approved_via_grant_id"] == 51 and asks == []
    handler.assert_called_once()


def test_a_writes_gate_grant_is_never_taken_as_the_click(asks, monkeypatch):
    """A write's grant stays granted at the gate: only ``_the_click`` claims it, once."""
    monkeypatch.setattr(owner_only, "_claimed_at_the_gate",
                        lambda db, ws, action, grant_id: NS(id=grant_id, granted_by="") if grant_id is not None else None)
    write = NS(action_def=NS(permission_level="write"), approved_via_grant_id=52)

    reply, handler = _run("platform_configure_agent_heartbeat", _owners_chat(), write)

    assert reply["requires_confirmation"] is True and asks == ["platform_configure_agent_heartbeat"]
    handler.assert_not_called()


# ── The card says what changes (FX-008's text, for each added action) ────────────────

@pytest.fixture
def buyer(db_session, seed_workspace):
    """GREEN BUYER with an hourly heartbeat and one skill; a playbook on a Monday timer;
    another workspace's agent of the same name."""
    from core.models.core import Agent, Skill, WorkflowTemplate, agent_skills

    ws, other = UUID(seed_workspace()), UUID(seed_workspace())
    agent = Agent(name="GREEN BUYER", agent_type="chatbot", description="Buys green coffee", status="active",
                  configuration={"heartbeat": {"enabled": True, "interval_minutes": 60}}, workspace_id=ws,
                  created_by="test", owner_type="workspace", owner_id=str(ws))
    theirs = Agent(name="GREEN BUYER", agent_type="chatbot", description="Bravo's buyer", status="active",
                   configuration={"heartbeat": {"interval_minutes": 5}}, workspace_id=other, created_by="test",
                   owner_type="workspace", owner_id=str(other))
    skill = Skill(name="menu-writer", skill_type="technical", workspace_id=ws)
    playbook = WorkflowTemplate(template_id=f"fx010-{uuid4().hex[:8]}", name="Monday green-coffee reorder",
                                description="Reorder what is low.", workspace_id=ws, owner_type="workspace",
                                owner_id=str(ws), created_by="test", steps=[],
                                schedule_config={"type": "cron", "cron_expression": "0 9 * * 1",
                                                 "timezone": "Europe/London", "enabled": True},
                                template_definition={"steps": [], "agents": [], "config": {}, "variables": []})
    db_session.add_all([agent, theirs, skill, playbook])
    db_session.flush()
    db_session.execute(agent_skills.insert().values(agent_id=agent.id, skill_id=skill.id))
    db_session.flush()
    return NS(db=db_session, ws=ws, agent=agent, theirs=theirs, playbook=playbook)


def _asked(buyer, action, params):
    return owner_only.platform_ask(buyer.db, buyer.ws, action, params, _owners_chat())["question_md"]


def test_a_heartbeat_card_says_from_and_to(buyer):
    asked = _asked(buyer, "platform_configure_agent_heartbeat", {"agent_id": buyer.agent.id, "interval_minutes": 15})
    assert asked.startswith("Change an agent's heartbeat")
    assert f"- agent: 'GREEN BUYER' (agent #{buyer.agent.id})" in asked
    assert "- every (minutes): 60 → 15" in asked


def test_another_workspaces_agent_is_never_read_onto_the_card(buyer):
    from modules.tools.discovery.card_question_agents import delete_agent_lines, heartbeat_lines

    params = {"agent_id": buyer.theirs.id, "interval_minutes": 15}
    assert heartbeat_lines(buyer.db, buyer.ws, "platform_configure_agent_heartbeat", params) == []
    assert delete_agent_lines(buyer.db, buyer.ws, "platform_delete_agent", params) == []
    refused = owner_only.platform_ask(buyer.db, buyer.ws, "platform_configure_agent_heartbeat", params, _owners_chat())
    assert refused["success"] is False and "question_md" not in refused      # no card for another's agent


def test_a_delete_agent_card_says_it_cannot_be_undone(buyer):
    asked = _asked(buyer, "platform_delete_agent", {"agent_name": "GREEN BUYER"})
    assert asked.startswith("Delete an agent")
    assert f"'GREEN BUYER' (agent #{buyer.agent.id}) is deleted for good" in asked


def test_a_skill_card_lists_the_agents_skills_before_and_after(buyer):
    asked = _asked(buyer, "platform_assign_skill_to_agent", {"agent_id": buyer.agent.id, "skill_name": "sourcing"})
    assert "- skills: menu-writer → menu-writer, sourcing" in asked
    taken = _asked(buyer, "platform_unassign_skill_from_agent", {"agent_id": buyer.agent.id, "skill_name": "menu-writer"})
    assert "- skills: menu-writer → (empty)" in taken


def test_a_playbook_card_says_its_timer_from_and_to_and_a_new_ones_name(buyer):
    timed = _asked(buyer, "platform_schedule_playbook", {"playbook_id": buyer.playbook.id, "cron_expression": "0 7 * * 1"})
    assert "- runs at (cron): 0 9 * * 1 → 0 7 * * 1" in timed
    made = _asked(buyer, "platform_create_playbook", {"name": "Friday cupping notes", "description": "Write them up."})
    assert made.startswith("Create a playbook") and "- name: Friday cupping notes" in made
    gone = _asked(buyer, "platform_delete_playbook", {"playbook_id": buyer.playbook.id})
    assert "'Monday green-coffee reorder'" in gone and "deleted for good" in gone


def test_only_this_workspaces_grant_for_this_action_claimed_at_the_gate_is_the_click(buyer, seed_workspace):
    from core.services.approval_grants import grant_grant, revoke_grant
    from modules.tools.execution.tool_grants import GRANT_CONSUMED_BY, issue_tool_grant

    params = {"agent_id": buyer.agent.id}
    grant = issue_tool_grant(buyer.db, buyer.ws, action="platform_delete_agent", params=params,
                             permission_level="destructive", description="Delete an agent",
                             caller_context=_owners_chat())
    grant_grant(grant, granted_by="user:clicker-7")
    claimed = owner_only._claimed_at_the_gate
    assert claimed(buyer.db, buyer.ws, "platform_delete_agent", grant.id) is None      # granted, not claimed
    revoke_grant(grant, revoked_by=GRANT_CONSUMED_BY)
    buyer.db.flush()
    assert claimed(buyer.db, buyer.ws, "platform_delete_agent", grant.id) is grant
    assert claimed(buyer.db, UUID(seed_workspace()), "platform_delete_agent", grant.id) is None
    assert claimed(buyer.db, buyer.ws, "platform_delete_playbook", grant.id) is None


# ── The click runs on the agent the card showed ───────────────────────────────────

def _market(buyer):
    from core.models.core import Agent

    made = [Agent(name=name, agent_type="chatbot", description="", status="active", configuration={},
                  workspace_id=buyer.ws, created_by="test", owner_type="workspace", owner_id=str(buyer.ws))
            for name in ("MARKET-MANAGER", "Market Research")]
    buyer.db.add_all(made)
    buyer.db.flush()
    return made


def test_an_agent_named_by_name_is_bound_to_its_id_before_the_card(buyer):
    from core.models.approval_grants import ApprovalGrant

    ask = owner_only.platform_ask(buyer.db, buyer.ws, "platform_delete_agent", {"agent_name": "green buyer"},
                                  _owners_chat())
    assert ask["requires_confirmation"] is True and ask["params"]["agent_id"] == buyer.agent.id
    assert buyer.db.get(ApprovalGrant, ask["grant_id"]).details["params"]["agent_id"] == buyer.agent.id


def test_a_whole_name_wins_over_a_name_that_contains_it(buyer):
    manager, _research = _market(buyer)
    ask = owner_only.platform_ask(buyer.db, buyer.ws, "platform_delete_agent", {"agent_name": "MARKET-MANAGER"},
                                  _owners_chat())
    assert ask["params"]["agent_id"] == manager.id


def test_a_name_two_agents_carry_is_refused_naming_them_and_nothing_is_asked(buyer):
    from core.models.approval_grants import ApprovalGrant

    manager, research = _market(buyer)
    reply = owner_only.platform_ask(buyer.db, buyer.ws, "platform_delete_agent", {"agent_name": "market"},
                                    _owners_chat())
    assert reply["success"] is False and "requires_confirmation" not in reply
    assert f"{manager.id}:MARKET-MANAGER" in reply["error"] and f"{research.id}:Market Research" in reply["error"]
    assert buyer.db.query(ApprovalGrant).filter(ApprovalGrant.workspace_id == buyer.ws).count() == 0


def test_a_name_no_agent_carries_is_refused_with_the_roster(buyer):
    reply = owner_only.platform_ask(buyer.db, buyer.ws, "platform_configure_agent_heartbeat",
                                    {"agent_name": "NOBODY", "interval_minutes": 15}, _owners_chat())
    assert reply["success"] is False and "GREEN BUYER" in reply["error"] and "requires_confirmation" not in reply


# ── A brief that sends or orders is reviewed by a person (Decision D7) ──────────────

def _brief(title, description="", **more):
    return {"title": title, "description": description, "_user_id": "user_owner", **more}


def test_a_brief_that_confirms_an_order_is_reviewed_by_a_person():
    params, held = reviewed_by_a_person(_brief("Confirm the order with the supplier", "Email Kerbside to confirm."))
    assert held is True and params["review_mode"] == "human"


def test_a_brief_that_drafts_a_reply_keeps_auto():
    params, held = reviewed_by_a_person(_brief("Draft a reply to the supplier", "Put it in Deliverables."))
    assert held is False and "review_mode" not in params


def test_a_review_mode_the_call_named_is_kept():
    params, held = reviewed_by_a_person(_brief("Send the invoice", review_mode="auto"))
    assert held is False and params["review_mode"] == "auto"


def test_a_ticket_an_agent_files_on_its_own_run_is_unchanged():
    params, held = reviewed_by_a_person({"title": "Send the invoice", "description": "Pay on receipt."})
    assert held is False and "review_mode" not in params


@pytest.mark.parametrize("brief, sends", [
    ("Send Kerbside the menu", True), ("Publish the spring post", True), ("Book the van for Friday", True),
    ("Pay the roaster", True), ("Reorder oat milk", True), ("Draft a reply", False),
    ("Summarise the cupping notes", False), ("Update the playbook", False),
])
def test_the_word_list(brief, sends):
    assert sends_or_orders(brief) is sends


def test_the_create_task_tool_files_it_for_review_and_its_receipt_says_so(monkeypatch):
    from consumers.chatbot.receipts import receipt
    from modules.tools.discovery import handlers_board_task_review, handlers_board_tasks

    filed = {}

    async def create(db, workspace_id, params):
        filed.update(params)
        return {"success": True, "task_id": 2318, "status": "assigned", "title": params["title"],
                "review_mode": params.get("review_mode", "auto")}

    monkeypatch.setattr(handlers_board_tasks, "create_board_task", create)
    params = _brief("Confirm the order with the supplier", "Email Kerbside.")
    result = asyncio.run(handlers_board_task_review.create_board_task(None, WS, params))

    assert filed["review_mode"] == "human" and result[REVIEW_HELD] is True
    assert receipt("platform_create_task", params, result)["effect"] == "card created, reviewed by you before it closes"


def test_a_drafting_ticket_closes_by_itself_and_its_receipt_says_nothing_of_review(monkeypatch):
    from consumers.chatbot.receipts import receipt
    from modules.tools.discovery import handlers_board_task_review, handlers_board_tasks

    async def create(db, workspace_id, params):
        return {"success": True, "task_id": 2319, "status": "assigned", "title": params["title"],
                "review_mode": params.get("review_mode", "auto")}

    monkeypatch.setattr(handlers_board_tasks, "create_board_task", create)
    params = _brief("Draft a reply to the supplier")
    result = asyncio.run(handlers_board_task_review.create_board_task(None, WS, params))

    assert result["review_mode"] == "auto" and REVIEW_HELD not in result
    assert "reviewed by you" not in receipt("platform_create_task", params, result)["effect"]
