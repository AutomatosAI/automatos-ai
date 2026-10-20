"""PRD-256 F430 (D): Auto and the marketplace read what a CLI agent's sessions can call.

Socials day run (14 Oct, §46b): asked why the Social Media Director's post failed, Auto guessed:
platform_get_agent gave its runtime, never its session tools. And the marketplace listing that
sells the Director said nothing of what a copy switched to a CLI session could not do. Now:

* platform_get_agent on a CLI agent answers ``session_tools`` (what its sessions are offered) and
  ``session_tool_gaps`` (what its skills name that they cannot call); an API agent's answer is as
  it was;
* GET /api/marketplace/items/{id}?type=agent carries ``dependencies.session_tool_gaps``: a CLI
  copy's skill gaps on the default groups, so a skill naming workspace_html_to_png is said and
  the Socials post tools are not (a session has them now).
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from sqlalchemy import text

from services.session_capability_gaps import agent_session_view

SKILL_BODY = ("Draft with platform_create_social_post, then draw the card with workspace_html_to_png "
              "and check it with platform_get_social_post.")


def _skill(body=SKILL_BODY):
    return NS(id=11, name="quote-cards", description="Quote card posts.", category="social", skill_type="custom",
              is_active=True, prompt_template=body, content_hash=None, tools_schema=None)


def test_a_cli_agents_view_is_its_session_tools_and_its_skills_gaps():
    agent = NS(id=348, name="Social Media Director", skills=[_skill()],
               configuration={"runtime": "cli", "session_tool_groups": ["documents"]})
    view = agent_session_view(agent)
    assert "create_social_post" not in view["session_tools"] and "generate_document" in view["session_tools"]
    assert view["session_tool_gaps"] == [{"skill": "quote-cards", "tools": [
        "platform_create_social_post", "workspace_html_to_png", "platform_get_social_post"]}]


def test_on_the_default_groups_only_what_no_session_has_is_a_gap():
    agent = NS(id=348, name="Social Media Director", skills=[_skill()], configuration={"runtime": "cli"})
    (gap,) = agent_session_view(agent)["session_tool_gaps"]
    assert gap["tools"] == ["workspace_html_to_png"]
    assert gap["instead"] == {"platform_create_social_post": "create_social_post",
                              "platform_get_social_post": "get_social_post"}


def test_an_api_agent_has_no_session_view():
    assert agent_session_view(NS(id=1, name="Writer", skills=[_skill()], configuration={"runtime": "api"})) == {}
    assert agent_session_view(NS(id=2, name="Old", skills=[], configuration=None)) == {}


# ── platform_get_agent, on Postgres ─────────────────────────────────────────

@pytest.fixture
def workspace(db_session, seed_workspace):
    ws = UUID(seed_workspace())

    def agent(name, configuration):
        return db_session.execute(
            text("INSERT INTO agents (name, agent_type, workspace_id, status, configuration) "
                 "VALUES (:name, 'custom', CAST(:w AS uuid), 'active', CAST(:c AS json)) RETURNING id"),
            {"name": name, "w": str(ws), "c": json.dumps(configuration)}).scalar()

    return NS(db=db_session, ws=ws, director=agent("Social Media Director", {"runtime": "cli"}),
              writer=agent("Content Creator", {}))


def _get_agent(workspace, agent_id):
    from modules.tools.discovery.handlers_agents import get_agent

    answer = asyncio.run(get_agent(workspace.db, workspace.ws, {"agent_id": agent_id}))
    assert answer["success"] is True, answer
    return answer["agent"]


@pytest.mark.integration
def test_platform_get_agent_says_what_a_cli_agents_sessions_can_call(workspace):
    director = _get_agent(workspace, workspace.director)
    assert director["runtime"] == "cli"
    assert "create_social_post" in director["session_tools"] and director["session_tool_gaps"] == []


@pytest.mark.integration
def test_an_api_agents_answer_is_as_it_was(workspace):
    writer = _get_agent(workspace, workspace.writer)
    assert "session_tools" not in writer and "session_tool_gaps" not in writer


# ── the marketplace detail ──────────────────────────────────────────────────

class _NoToolsDB:
    """``_build_agent_detail``'s one read: the agent's tool assignments (none here)."""

    def execute(self, statement, params=None):
        return self

    def fetchall(self):
        return []


def _marketplace_agent():
    now = datetime(2026, 10, 14, tzinfo=timezone.utc)
    return NS(id=331, name="Social Media Director", description="Makes the posts.", created_by="platform",
              original_creator_id=None, skills=[_skill()], marketplace_icon=None, marketplace_category="social",
              tags=[], install_count=4, is_featured=False, version="1.0.0", agent_type="custom",
              configuration={}, model_config={}, created_at=now, updated_at=now)


def test_the_marketplace_detail_says_what_a_cli_copy_could_not_call(monkeypatch):
    import api.marketplace as marketplace

    item = _marketplace_agent()
    monkeypatch.setattr(marketplace, "_find_marketplace_item", lambda db, kind, item_id, approved_only: item)
    detail = asyncio.run(marketplace.get_item(331, type="agent", ctx=None, db=_NoToolsDB()))
    (gap,) = detail.dependencies["session_tool_gaps"]
    assert (gap["kind"], gap["skill"], gap["tools"]) == ("skill", "quote-cards", ["workspace_html_to_png"])
    assert not [name for name in gap["tools"] if name.endswith("_social_post")]
    assert detail.dependencies["skills"] == [{"id": 11, "name": "quote-cards"}]   # what it carried before


def test_gaps_that_cannot_be_read_leave_the_detail_as_it_was(monkeypatch):
    import api.agent_session_tools as agent_tools
    import api.marketplace as marketplace

    def broken(agent, groups=None):
        raise RuntimeError("the skills could not be read")

    item = _marketplace_agent()
    monkeypatch.setattr(marketplace, "_find_marketplace_item", lambda db, kind, item_id, approved_only: item)
    monkeypatch.setattr(agent_tools, "skill_gaps", broken)
    detail = asyncio.run(marketplace.get_item(331, type="agent", ctx=None, db=_NoToolsDB()))
    assert "session_tool_gaps" not in detail.dependencies and detail.name == "Social Media Director"
