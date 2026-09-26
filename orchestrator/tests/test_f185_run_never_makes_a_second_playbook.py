"""F185 — "run it" never makes a second playbook, and a failed delete says why.

Night 6 (03:41:42Z): asked to run the "New Cafe Onboarding" playbook it had made
one turn before (#102), Auto made another of the same name (#103). The run tool
was not among the 17 actions offered that turn (the onboarding prior's list) and
every call it made was off that list: it repeated the create it had just used.
Its three deletes of the duplicate each asked for approval on a card that named
no playbook, and the turn's reply was "I encountered an issue … Please try again".

- F144's namesake rule now covers playbooks: a second one of the same name is
  refused, naming the one that exists and how to run it.
- The delete card names the playbook; one that is not in the workspace fails back
  without asking. A delete by name matches the whole name and refuses when two
  playbooks share it (it used to delete the first that contained it).
- When the model says nothing after an ask, the reply says what is waiting.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from sqlalchemy import text

NAME = "New Cafe Onboarding"


@pytest.fixture
def cafe(db_session, seed_workspace, monkeypatch):
    monkeypatch.setattr("modules.tools.execution.tool_grants._notify_approval_pending", lambda *a, **k: None)
    return NS(db=db_session, ws=UUID(seed_workspace()))


def _create(cafe, name=NAME):
    from modules.tools.discovery.handlers_playbooks import create_playbook

    return asyncio.run(create_playbook(cafe.db, cafe.ws, {"name": name, "description": "Welcome email and record card"}))


def _stored(cafe, name=NAME):
    """A playbook written straight to the table, as the night's duplicate was."""
    from core.models.core import WorkflowTemplate

    playbook = WorkflowTemplate(name=name, template_id=f"custom-{uuid.uuid4().hex[:8]}", description="d",
                                workspace_id=cafe.ws, owner_type="workspace", owner_id=str(cafe.ws),
                                created_by="platform", tags=[], template_definition={"steps": []})
    cafe.db.add(playbook)
    cafe.db.flush()
    return playbook.id


def _count(cafe):
    return cafe.db.execute(text("SELECT count(*) FROM workflow_recipes WHERE workspace_id = CAST(:w AS uuid)"),
                           {"w": str(cafe.ws)}).scalar()


def _delete(cafe, **params):
    from modules.tools.discovery.handlers_playbooks import delete_playbook

    return asyncio.run(delete_playbook(cafe.db, cafe.ws, params))


# ── one playbook per name ───────────────────────────────────────────────────

def test_a_second_playbook_of_the_same_name_is_refused_and_the_first_named(cafe):
    first = _create(cafe)["playbook"]["id"]
    reply = _create(cafe, "  new cafe onboarding ")
    assert reply["success"] is False and reply["existing_playbook_id"] == first
    assert f"(id {first})" in reply["error"] and "platform_execute_playbook" in reply["error"]
    assert _count(cafe) == 1


# ── a delete that fails says why ────────────────────────────────────────────

def _ask_to_delete(cafe, playbook_id):
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    executor = PlatformActionExecutor(cafe.db, cafe.ws)
    executor._full_autonomy = lambda: False
    return asyncio.run(executor.execute("platform_delete_playbook", {"playbook_id": playbook_id},
                                        {"mission_id": "m-1"}))


def test_the_delete_card_names_the_playbook(cafe):
    duplicate = _stored(cafe)
    reply = _ask_to_delete(cafe, duplicate)
    assert reply["requires_confirmation"] is True
    assert f"'{NAME}' (playbook #{duplicate})" in reply["message"]


def test_a_delete_of_a_playbook_that_is_not_here_asks_nothing(cafe):
    reply = _ask_to_delete(cafe, 987654)
    assert reply.get("requires_confirmation") is not True
    assert reply["error"].startswith("No playbook #987654 in this workspace — nothing was asked or done")


def test_a_delete_by_a_name_two_playbooks_share_deletes_neither(cafe):
    ids = [_stored(cafe), _stored(cafe)]
    reply = _delete(cafe, playbook_name=NAME)
    assert reply["success"] is False and f"(ids {ids[0]}, {ids[1]})" in reply["error"]
    assert _count(cafe) == 2


def test_a_delete_by_part_of_a_name_deletes_nothing(cafe):
    _stored(cafe)
    reply = _delete(cafe, playbook_name="Cafe")
    assert reply["success"] is False and "No playbook is called 'Cafe'" in reply["error"]
    assert _count(cafe) == 1


def test_an_empty_reply_after_an_ask_says_what_is_waiting():
    from consumers.chatbot.service import NOTHING_SAID, nothing_said_fallback

    ask = ("This action (destructive) requires confirmation. Action: platform_delete_playbook "
           f"on '{NAME}' (playbook #103) — Permanently delete a playbook")
    reply = nothing_said_fallback({"tool_approval": {"grant_id": 7, "message": ask}})
    assert reply.startswith("Nothing was done yet.") and ask in reply and "waits for your approval" in reply
    assert "try again" not in reply
    assert nothing_said_fallback({}) == NOTHING_SAID
