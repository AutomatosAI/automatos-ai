"""P256-FIX-N14-7 (F421): a skill made from the owner's chat waits for their click.

Night 14: "give the WRITER a skill for writing product descriptions" made workspace skill
product-description-writer (#361) with no card: ``platform_create_workspace_skill`` was not
owner-only, and only the assignment after it asked. It is on the owner-only list now (Decision
D1 amended, FX-010's agent-setting rule): from the owner's chat it raises one grant card naming
the skill and what it does, and no skill row exists until the click, which makes exactly one.
A create the click could not make (no content, a name too long or one the workspace has, content
the security scanner refuses) is refused before any card; an agent's own run makes the skill as
before.

Each chat call runs through the executor's whole stack (the guards, the role gates, the rate
limit, the ask) onto the real create tool.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest

from modules.tools.discovery import owner_only

CREATE = "platform_create_workspace_skill"
NAME = "product-description-writer"
DOES = "Writes short product descriptions for the shop's coffees."
BODY = f"""---
name: {NAME}
description: {DOES}
version: 1.0.0
tags: [writing]
tools: []
---

# Product description writer

Write each description in 60 words, in the house voice.
"""


@pytest.fixture
def cafe(db_session, seed_workspace, monkeypatch):
    """A workspace whose owner drives the chat with Auto. The dial is on (an owner-only call still asks)."""
    from sqlalchemy import text

    import modules.tools.discovery.follows_the_owner as guard
    from core.models.core import Agent
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    ws = UUID(seed_workspace())
    auto = Agent(name="Auto", agent_type="system", description="", status="active", configuration={},
                 workspace_id=ws, created_by="test", owner_type="workspace", owner_id=str(ws), is_system_agent=True)
    db_session.add(auto)
    db_session.flush()
    tag = uuid4().hex[:8]
    owner = db_session.execute(text("INSERT INTO users (email, username) VALUES (:e, :u) RETURNING id"),
                               {"e": f"n147-{tag}@harbourline.test", "u": f"n147-{tag}"}).scalar()
    db_session.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                            "VALUES (CAST(:ws AS uuid), :user, 'owner', TRUE)"), {"ws": str(ws), "user": owner})
    monkeypatch.setattr(guard, "owner_turn", lambda db, ws, ctx: None)
    executor = PlatformActionExecutor(db_session, ws)
    executor._full_autonomy = lambda: True
    chat = {"driving_user_id": str(owner), "user_id": "user_owner", "conversation_id": str(uuid4()), "turn_id": "t-1"}
    return NS(db=db_session, ws=ws, owner=owner, auto=auto.id, executor=executor, chat=chat)


def _call(cafe, params):
    """The call as the chat makes it: Auto as ``_agent_id`` (the hierarchy check's actor)."""
    with patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)):
        return asyncio.run(cafe.executor.execute(CREATE, {**params, "_agent_id": cafe.auto}, cafe.chat))


def _grants(cafe):
    from core.models.approval_grants import ApprovalGrant

    return cafe.db.query(ApprovalGrant).filter(ApprovalGrant.workspace_id == cafe.ws).all()


def _skills(cafe, name=NAME):
    from core.models.core import Skill

    return cafe.db.query(Skill).filter(Skill.workspace_id == cafe.ws, Skill.name == name).all()


def _asked(reply):
    return (reply["success"] is False and reply.get("requires_confirmation") is True
            and reply.get("owner_only") is True and isinstance(reply.get("grant_id"), int))


def _click(cafe, ask):
    from core.models.approval_grants import ApprovalGrant
    from core.services.approval_grants import grant_grant

    grant_grant(cafe.db.get(ApprovalGrant, ask["grant_id"]), granted_by=f"user:{cafe.owner}")
    cafe.db.flush()


# ── The list, its verb and its card ─────────────────────────────────────────────────

def test_the_list_holds_a_skill_made_with_its_verb_and_its_card_lines():
    from modules.tools.discovery.card_question import READERS

    assert CREATE in owner_only.OWNER_ONLY_ACTIONS and owner_only.is_owner_only(CREATE, {"name": NAME})
    assert owner_only.VERBS[CREATE] == "create a skill for this workspace" and CREATE in READERS


# ── F421: Auto's create from the owner's chat ───────────────────────────────────────

def test_a_skill_made_from_the_owners_chat_raises_one_card_and_makes_no_skill(cafe):
    from consumers.chatbot.receipts import WAITING, receipt

    params = {"name": NAME, "content": BODY}
    reply = _call(cafe, params)

    assert _asked(reply), reply
    assert len(_grants(cafe)) == 1 and _grants(cafe)[0].tool_name == CREATE
    assert _skills(cafe) == []                                                # F421: #361 was made at once
    head, *lines = reply["question_md"].splitlines()
    assert head.startswith("Create a skill for this workspace"), head
    assert f"- skill: '{NAME}', new: this workspace's own, enabled for it and given to no agent yet" in lines
    assert f"- what it does: {DOES}" in lines                                 # the frontmatter's, as the tool saves it
    assert "- its instructions start: # Product description writer" in lines
    assert receipt(CREATE, params, reply)["status"] == WAITING                # FX-004: waiting, never a failure


def test_the_click_makes_exactly_one_skill(cafe):
    ask = _call(cafe, {"name": NAME, "content": BODY})
    _click(cafe, ask)

    made = _call(cafe, ask["params"])

    assert made["success"] is True and made["approved_via_grant_id"] == ask["grant_id"], made
    (skill,) = _skills(cafe)
    assert made["skill_id"] == skill.id and skill.description == DOES
    again = _call(cafe, ask["params"])                                         # one click, one skill
    assert again["success"] is False and len(_skills(cafe)) == 1, again


def test_the_calls_own_description_is_what_the_card_says_it_does(cafe):
    reply = _call(cafe, {"name": NAME, "content": BODY, "description": "Product copy for the web shop."})

    assert _asked(reply), reply
    assert "- what it does: Product copy for the web shop." in reply["question_md"].splitlines()


def test_a_scanner_override_is_on_the_card(cafe):
    reply = _call(cafe, {"name": NAME, "content": BODY, "acknowledge_warnings": True})

    assert _asked(reply), reply
    assert ("- saved over the security scanner's high-severity findings (acknowledge_warnings)"
            in reply["question_md"].splitlines())


# ── What the click could not make is refused before any card ─────────────────────────

def test_a_name_the_workspace_has_is_refused_before_any_card(cafe):
    from core.models.core import Skill

    kept = Skill(name=NAME, description="The one already here.", skill_type="cognitive", category="writing",
                 prompt_template=BODY, skill_source="workspace-user", skill_version="1.0.0",
                 workspace_id=cafe.ws, is_active=True, skill_metadata={})
    cafe.db.add(kept)
    cafe.db.flush()

    reply = _call(cafe, {"name": NAME, "content": BODY})

    assert reply["success"] is False and "requires_confirmation" not in reply, reply
    assert f"already has a skill named '{NAME}' (skill #{kept.id})" in reply["error"]
    assert _grants(cafe) == [] and _skills(cafe) == [kept]


def test_a_create_with_no_content_or_too_long_a_name_is_refused_before_any_card():
    from modules.tools.discovery.card_question_skill_create import NEEDS_BOTH, TOO_LONG, refused_before_the_create

    assert refused_before_the_create(None, uuid4(), {"name": NAME}) == {"success": False, "error": NEEDS_BOTH}
    assert refused_before_the_create(None, uuid4(), {"name": "  ", "content": BODY})["error"] == NEEDS_BOTH
    too_long = refused_before_the_create(None, uuid4(), {"name": "x" * 256, "content": BODY})
    assert too_long == {"success": False, "error": TOO_LONG.format(limit=255)}


@pytest.mark.parametrize("verdict", ["blocked", "warnings"])
def test_content_the_scanner_refuses_is_refused_before_any_card(cafe, monkeypatch, verdict):
    from modules.tools.discovery import handlers_skills
    from modules.tools.discovery.card_question_skill_create import SCANNED

    finding = {"severity": "high", "rule": "shell-exec", "line": 9}
    monkeypatch.setattr(handlers_skills, "_scan", lambda content, filename, acknowledge_warnings: {
        "status": verdict, "findings": [finding]})

    reply = _call(cafe, {"name": NAME, "content": BODY})

    assert reply["success"] is False and "requires_confirmation" not in reply, reply
    assert reply["error"] == SCANNED[verdict] and reply["findings"] == [finding]
    assert _grants(cafe) == [] and _skills(cafe) == []


def test_odd_frontmatter_still_gives_a_card_line():
    from modules.tools.discovery.card_question_skill_create import create_skill_lines

    lines = create_skill_lines(None, uuid4(), CREATE, {"name": NAME, "content": "---\n- a list\n---\nWrite well.\n"})

    assert "- its instructions start: Write well." in lines


# ── An agent's own run makes the skill as before ────────────────────────────────────

class _Executor:
    """PlatformActionExecutor._run_cleared's shape: the gates have cleared, the handler runs."""

    @owner_only.asks_the_owner_first
    async def _run_cleared(self, action_name, params, caller_context, cleared, handler):
        return await handler(None, uuid4(), params)


@pytest.mark.parametrize("lane", [None, {"board_task_id": 2318}, {"playbook_execution_id": "exec-1"}],
                         ids=["agent-run", "board-ticket", "playbook-step"])
def test_the_same_call_from_an_agents_run_does_not_ask(lane):
    handler = AsyncMock(return_value={"success": True, "skill_id": 361})

    reply = asyncio.run(_Executor()._run_cleared(CREATE, {"name": NAME, "content": BODY}, lane, None, handler))

    assert reply == {"success": True, "skill_id": 361}
    handler.assert_called_once()
