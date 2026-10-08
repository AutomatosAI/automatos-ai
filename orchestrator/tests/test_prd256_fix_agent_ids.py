"""PRD-256 FX-012 (A1, F390, A430, A585): the task and playbook tools take the agent's id.

Night 12: platform_create_task took only ``assigned_agent_name``, platform_assign_task only
``agent_name``, platform_update_task no agent at all, and a name two active agents carry was
refused ("Rename or deactivate the duplicate"). c1 holds ten duplicated names over 26 agents:
39 writes refused, "Get OPS to…" fifteen times, and "267, the operations one", the id itself
sent as the name, too (A585). Asked "which agents did you create for me last night" (A430),
Auto's agent tools never said who made an agent.

These run Auto's tools as the platform executor binds them (``PLATFORM_HANDLERS``).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

from tests import test_1094_a_ticket_with_no_agent_is_never_in_progress as f1094

board = f1094.board        # a workspace, its Content Creator, Auto's real tools
TITLE = "Reorder the green stock"
BRIEF = "Reorder every green coffee under 50 kg from its usual supplier; hand back the order numbers."
STEP = "Report every coffee's green stock and flag anything under 50 kg."
_AGENT_SQL = text(
    "INSERT INTO agents (name, agent_type, workspace_id, status, configuration, job_title, created_by, tags, "
    "created_at) VALUES (:name, 'custom', CAST(:ws AS uuid), :status, CAST('{}' AS json), :job_title, 'platform', "
    "CAST('[\"ops\"]' AS json), now()) RETURNING id")


def _agent(db, ws, name, job_title=None, status="active"):
    return db.execute(_AGENT_SQL, {"name": name, "ws": str(ws), "status": status, "job_title": job_title}).scalar()


@pytest.fixture
def ops(board, seed_workspace):
    """c1's clash: two active agents called OPS; a switched-off one; and another workspace's agent."""
    elsewhere = UUID(seed_workspace())
    return NS(**vars(board), manager=_agent(board.db, board.ws, "OPS", "Operations Manager"),
              shop=_agent(board.db, board.ws, "OPS", "Shopify Operations"),
              retired=_agent(board.db, board.ws, "Old Buyer", "Buyer", status="inactive"),
              foreign=_agent(board.db, elsewhere, "Their Agent", "Buyer"))


def _tool(ops, action, params):
    from modules.tools.discovery.platform_executor import PLATFORM_HANDLERS

    return asyncio.run(PLATFORM_HANDLERS[action](ops.db, ops.ws, params))


def _cards(ops):
    from core.models.core import BoardTask

    return ops.db.query(BoardTask).filter(BoardTask.workspace_id == ops.ws, BoardTask.title == TITLE).all()


def _number(task):
    return f"#{task.workspace_seq:04d}"


# ── the clash lists its candidates; the id goes through ─────────────────────

def test_night_12s_get_ops_to_lists_both_ops_and_the_id_call_files_the_card(ops):
    refused = _tool(ops, "platform_create_task", {"title": TITLE, "description": BRIEF, "assigned_agent_name": "OPS"})

    assert refused["success"] is False and _cards(ops) == []
    assert f"{ops.manager} · OPS · Operations Manager" in refused["error"]
    assert f"{ops.shop} · OPS · Shopify Operations" in refused["error"]
    assert "call again with agent_id" in refused["error"]

    filed = _tool(ops, "platform_create_task", {"title": TITLE, "description": BRIEF, "assigned_agent_name": "OPS",
                                                "agent_id": f"#{ops.manager}"})       # an id wins over a name

    (card,) = _cards(ops)
    assert filed["success"] is True and (card.assigned_agent_id, card.status) == (ops.manager, "assigned")


def test_the_id_sent_as_the_name_assigns_the_card_a585(ops):
    card = f1094._ticket(ops, title=TITLE)

    refused = _tool(ops, "platform_assign_task", {"task_id": _number(card), "agent_name": "OPS"})
    given = _tool(ops, "platform_assign_task", {"task_id": _number(card), "agent_name": str(ops.shop)})

    ops.db.refresh(card)
    assert refused["success"] is False and "call again with agent_id" in refused["error"]
    assert given["success"] is True and given["assigned_agent"] == "OPS"
    assert (card.assigned_agent_id, card.status) == (ops.shop, "assigned")


def test_assign_takes_agent_id_alone(ops):
    card = f1094._ticket(ops, title=TITLE)

    given = _tool(ops, "platform_assign_task", {"task_id": _number(card), "agent_id": ops.manager})

    ops.db.refresh(card)
    assert given["success"] is True and card.assigned_agent_id == ops.manager


# ── an id the workspace can't use is refused, naming it ─────────────────────

@pytest.mark.parametrize("which, words", [("foreign", "is not in this workspace"), ("retired", "is switched off")])
def test_an_id_from_another_workspace_or_switched_off_is_refused_naming_it(ops, which, words):
    agent_id = getattr(ops, which)

    filed = _tool(ops, "platform_create_task", {"title": TITLE, "description": BRIEF, "agent_id": agent_id})
    card = f1094._ticket(ops, title="Another card")
    given = _tool(ops, "platform_assign_task", {"task_id": _number(card), "agent_id": str(agent_id)})

    ops.db.refresh(card)
    for out in (filed, given):
        assert out["success"] is False and f"Agent id {agent_id}" in out["error"] and words in out["error"]
    assert _cards(ops) == [] and card.assigned_agent_id is None


def test_an_agent_id_that_is_no_id_is_refused():
    from modules.tools.discovery.agent_refs import agent_id_said

    assert [agent_id_said(v) for v in (267, "267", "#267", " #0267 ")] == [267] * 4
    assert [agent_id_said(v) for v in ("OPS", "", None, True, 0, "-3", "26 7")] == [None] * 7


# ── platform_update_task gives the card to an agent ────────────────────────

def test_update_task_gives_the_card_by_id_and_edits_it(ops):
    card = f1094._ticket(ops, title=TITLE)

    out = _tool(ops, "platform_update_task", {"task_id": _number(card), "agent_id": ops.shop, "priority": "high"})

    ops.db.refresh(card)
    assert out["success"] is True and out["assigned_agent"] == "OPS"
    assert (card.assigned_agent_id, card.status, card.priority) == (ops.shop, "assigned", "high")


def test_update_task_with_an_id_it_cannot_use_edits_nothing(ops):
    card = f1094._ticket(ops, title=TITLE)

    out = _tool(ops, "platform_update_task", {"task_id": _number(card), "agent_id": ops.foreign, "priority": "high"})

    ops.db.refresh(card)
    assert out["success"] is False and f"Agent id {ops.foreign}" in out["error"]
    assert (card.assigned_agent_id, card.priority) == (None, "medium")


# ── the playbook step by id ────────────────────────────────────────────────

def _playbook(ops):
    from core.models.core import WorkflowTemplate

    made = WorkflowTemplate(template_id=f"fx012-{uuid4().hex[:8]}", name="Monday green stock", workspace_id=ops.ws,
                            description="Every coffee's green stock.", template_definition={"steps": []},
                            steps=[], created_by="platform")
    ops.db.add(made)
    ops.db.flush()
    return made


def test_a_playbook_step_takes_the_agent_by_id_and_a_clash_lists_the_ids(ops):
    playbook = _playbook(ops)

    clash = _tool(ops, "platform_add_playbook_step", {"playbook_id": playbook.id, "prompt_template": STEP,
                                                      "agent_name": "OPS"})
    by_id = _tool(ops, "platform_add_playbook_step", {"playbook_id": playbook.id, "prompt_template": STEP,
                                                      "agent_id": f"#{ops.shop}"})
    foreign = _tool(ops, "platform_add_playbook_step", {"playbook_id": playbook.id, "prompt_template": STEP,
                                                        "agent_id": ops.foreign})

    ops.db.refresh(playbook)
    assert clash["success"] is False and f"{ops.manager} · OPS · Operations Manager" in clash["error"]
    assert "agent_id" in clash["error"]
    assert by_id["success"] is True and [s["agent_id"] for s in playbook.steps] == [ops.shop]
    assert foreign["success"] is False and f"agent_id {ops.foreign} does not exist in this workspace" in foreign["error"]


# ── the agent tools say who made each agent, and when ──────────────────────

def test_list_and_get_agent_say_id_runtime_who_made_it_when_and_its_tags(ops):
    from modules.tools.discovery.agent_made_by import MADE_BY

    listed = {a["id"]: a for a in _tool(ops, "platform_list_agents", {})["agents"]}
    got = _tool(ops, "platform_get_agent", {"agent_id": ops.manager})["agent"]

    for agent in (listed[ops.manager], got):
        assert agent["id"] == ops.manager and agent["runtime"] == "api"
        assert agent["created_by"] == MADE_BY["platform"] and agent["created_at"] and agent["tags"] == ["ops"]
    assert ops.foreign not in listed


def test_a_shortened_listing_keeps_who_made_each_agent():
    from services.agent_roster_fit import every_agent_fits

    team = [{"id": 300 + i, "name": "OPS", "job_title": "Operations", "status": "active", "runtime": "api",
             "description": "d" * 600, "created_by": "platform: made with platform_create_agent, by Auto or an agent",
             "created_at": "2026-10-07T22:14:00", "tags": ["ops"]} for i in range(26)]

    fitted = every_agent_fits({"success": True, "agents": team, "count": len(team)})

    assert "note" in fitted and all("description" not in a for a in fitted["agents"])
    assert all(a["created_by"].startswith("platform") and a["created_at"] and a["tags"] for a in fitted["agents"])


# ── the schemas ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["platform_create_task", "platform_assign_task", "platform_update_task",
                                  "platform_add_playbook_step"])
def test_each_tool_takes_agent_id_as_a_number_or_its_hash_form(name):
    from modules.tools.discovery.action_registry import get_action_registry

    schema = get_action_registry().get(name).parameters
    assert set(schema["properties"]["agent_id"]["type"]) == {"integer", "string"}
    assert "agent_id" not in schema.get("required", [])
