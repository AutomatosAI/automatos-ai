"""F230 — the local edition has no plan agent limit; the hosted edition keeps it.

F200 put one agent-limit check on every create path, with no edition gate, so
the local edition enforced the hosted plans. c1 (49 agents, plan 'basic')
refused every new agent ("Your basic plan includes 5 agents and this workspace
has 49"), the build-4 PRD-253 probe agent was refused, and Auto went down
platform_create_agent in two chat checks. Plans are the hosted edition's: a
self-hoster has no plan limits.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from sqlalchemy import text

ON_C1 = 49


@pytest.fixture
def full_cafe(db_session, seed_workspace):
    ws = UUID(seed_workspace())
    db_session.execute(text("UPDATE workspaces SET plan = 'basic' WHERE id = :w"), {"w": ws})
    for n in range(ON_C1):
        db_session.execute(text(
            "INSERT INTO agents (name, agent_type, workspace_id, status, configuration, owner_type) "
            "VALUES (:n, 'custom', :w, 'active', CAST('{}' AS json), 'workspace')"), {"n": f"Barista {n}", "w": ws})
    db_session.flush()
    return NS(db=db_session, ws=ws, ctx=NS(workspace_id=ws, user=NS(id=None, email="owner@cafe.test")))


def _agents(cafe):
    return cafe.db.execute(text("SELECT count(*) FROM agents WHERE workspace_id = :w AND owner_type = 'workspace'"),
                           {"w": cafe.ws}).scalar()


def _edition(monkeypatch, edition):
    from config import config

    monkeypatch.setattr(config, "AUTH_EDITION", edition)


def test_the_local_edition_makes_agent_50_on_a_basic_plan(full_cafe, monkeypatch):
    import api.agents as agents_api
    from core.models.core import AgentCreate
    from modules.tools.discovery import handlers_agents
    from services.agent_quota import UNLIMITED, plan_agent_limit

    _edition(monkeypatch, "local")
    assert plan_agent_limit(NS(plan="basic")) == ("basic", UNLIMITED)

    made = asyncio.run(handlers_agents.create_agent(full_cafe.db, full_cafe.ws, {"name": "Probe"}))
    assert made["success"] is True and _agents(full_cafe) == ON_C1 + 1       # night: "includes 5 agents … has 49"
    asyncio.run(agents_api.create_agent(AgentCreate(name="Roaster", agent_type="custom"),
                                        ctx=full_cafe.ctx, db=full_cafe.db))
    assert _agents(full_cafe) == ON_C1 + 2


def test_the_hosted_edition_still_stops_at_the_plan_limit(full_cafe, monkeypatch):
    from modules.tools.discovery import handlers_agents

    _edition(monkeypatch, "saas")
    refused = asyncio.run(handlers_agents.create_agent(full_cafe.db, full_cafe.ws, {"name": "Probe"}))

    assert refused.get("over_quota") is True and "Your basic plan includes" in refused["message"]
    assert _agents(full_cafe) == ON_C1
