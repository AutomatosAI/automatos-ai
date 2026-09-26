"""F200 (b) — one agent-limit check on every create path, told before it is crossed.

Night 6: the Install button refused a package (over_quota: 8 agents on a basic
plan of 5) while Auto had just made three agents one by one through
platform_install_marketplace_agent, and the owner then made #330 by hand through
POST /api/agents. Neither was checked. Every create path now stops at the plan's
limit, creates nothing, and says the limit and the count.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from fastapi import HTTPException
from sqlalchemy import text


@pytest.fixture
def cafe(db_session, seed_workspace):
    from core.models.workspaces import Workspace
    from services.agent_quota import plan_agent_limit

    ws = UUID(seed_workspace())
    db_session.execute(text("UPDATE workspaces SET plan = 'basic' WHERE id = :w"), {"w": ws})
    _, limit = plan_agent_limit(db_session.get(Workspace, ws))
    assert limit > 0, "the basic plan has an agent limit"
    db_session.execute(text(
        "INSERT INTO agents (name, agent_type, status, configuration, owner_type) "
        "VALUES ('Shopify Support Agent', 'custom', 'active', CAST('{}' AS json), 'marketplace')"))
    return NS(db=db_session, ws=ws, limit=limit,
              ctx=NS(workspace_id=ws, user=NS(id=None, email="owner@cafe.test")))


def _hire(cafe, how_many):
    for n in range(how_many):
        cafe.db.execute(text(
            "INSERT INTO agents (name, agent_type, workspace_id, status, configuration, owner_type) "
            "VALUES (:n, 'custom', :w, 'active', CAST('{}' AS json), 'workspace')"), {"n": f"Barista {n}", "w": cafe.ws})
    cafe.db.flush()


def _agents(cafe):
    return cafe.db.execute(text("SELECT count(*) FROM agents WHERE workspace_id = :w AND owner_type = 'workspace'"),
                           {"w": cafe.ws}).scalar()


def _refused_over_the_limit(outcome, cafe):
    message = outcome["message"] if isinstance(outcome, dict) else outcome.detail
    assert f"includes {cafe.limit} agents and this workspace has {cafe.limit}" in message
    assert "Nothing was created" in message


def test_every_create_path_stops_at_the_plan_limit(cafe):
    import api.agents as agents_api
    from core.models.core import AgentCreate
    from modules.tools.discovery import handlers_agents, handlers_packages

    _hire(cafe, cafe.limit)

    made_by_auto = asyncio.run(handlers_agents.create_agent(cafe.db, cafe.ws, {"name": "Roaster"}))
    assert made_by_auto.get("over_quota") is True                       # night: created past the limit
    _refused_over_the_limit(made_by_auto, cafe)

    installed = asyncio.run(handlers_packages.install_marketplace_agent_tool(
        cafe.db, cafe.ws, {"agent_name": "Shopify Support Agent"}))
    assert installed.get("over_quota") is True                          # night: #327-329
    _refused_over_the_limit(installed, cafe)

    for call in (lambda: agents_api.create_agent(AgentCreate(name="Roaster", agent_type="custom"),
                                                 ctx=cafe.ctx, db=cafe.db),                # night: #330
                 lambda: agents_api.create_agents_bulk([AgentCreate(name="A", agent_type="custom"),
                                                        AgentCreate(name="B", agent_type="custom")],
                                                       ctx=cafe.ctx, db=cafe.db)):
        with pytest.raises(HTTPException) as refused:
            asyncio.run(call())
        assert refused.value.status_code == 402
        _refused_over_the_limit(refused.value, cafe)

    assert _agents(cafe) == cafe.limit


def test_the_last_agent_that_fits_is_made_and_a_batch_that_would_not_fit_is_not(cafe):
    import api.agents as agents_api
    from core.models.core import AgentCreate
    from modules.tools.discovery import handlers_agents

    _hire(cafe, cafe.limit - 2)
    with pytest.raises(HTTPException) as refused:                      # 2 slots left, 3 asked for
        asyncio.run(agents_api.create_agents_bulk(
            [AgentCreate(name=n, agent_type="custom") for n in ("A", "B", "C")], ctx=cafe.ctx, db=cafe.db))
    assert refused.value.status_code == 402 and _agents(cafe) == cafe.limit - 2

    made = asyncio.run(handlers_agents.create_agent(cafe.db, cafe.ws, {"name": "Roaster"}))
    assert made["success"] is True and _agents(cafe) == cafe.limit - 1


def test_a_package_that_hits_the_limit_mid_install_leaves_nothing_behind(monkeypatch):
    """Review HIGH: the plan can fill between the package's own check and a member's
    clone; the members cloned before it go with the refusal ('Nothing was created')."""
    import modules.tools.discovery.handlers_packages as hp
    from services.agent_quota import AgentLimitReached
    from tests.test_prd230_package_tools import FakeDB, FakeWS, _pkg

    ws = FakeWS(stage="completed")
    monkeypatch.setattr(hp, "_load_workspace", lambda db, wid: ws)
    monkeypatch.setattr(hp, "_workspace_agent_count", lambda db, wid: 0)
    monkeypatch.setattr("services.marketplace_packages.get_by_slug", lambda db, slug: _pkg(agents=3))

    async def _second_member_finds_the_plan_full(db, ws_id, slug, user_id=None):
        db.cloned = ["Operations"]                                   # the first member is in
        raise AgentLimitReached({"success": False, "over_quota": True, "message": "Your basic plan is full."})

    monkeypatch.setattr("services.package_installer.install_package", _second_member_finds_the_plan_full)
    db = FakeDB()

    out = asyncio.run(hp.install_package_tool(db, "ws-1", {"slug": "shopify-management"}))

    assert out == {"success": False, "over_quota": True, "message": "Your basic plan is full."}
    assert db.rolled_back_to_savepoint is True                      # before: the first member stayed


def test_a_package_install_that_breaks_leaves_no_savepoint_open(monkeypatch):
    """Review MEDIUM: an unexpected failure mid-install rolls its savepoint back
    rather than relying on every caller's own rollback."""
    import modules.tools.discovery.handlers_packages as hp
    from tests.test_prd230_package_tools import FakeDB, FakeWS, _pkg

    monkeypatch.setattr(hp, "_load_workspace", lambda db, wid: FakeWS(stage="completed"))
    monkeypatch.setattr(hp, "_workspace_agent_count", lambda db, wid: 0)
    monkeypatch.setattr("services.marketplace_packages.get_by_slug", lambda db, slug: _pkg(agents=2))

    async def _breaks(db, ws_id, slug, user_id=None):
        raise RuntimeError("the marketplace went away")

    monkeypatch.setattr("services.package_installer.install_package", _breaks)
    db = FakeDB()

    with pytest.raises(RuntimeError):
        asyncio.run(hp.install_package_tool(db, "ws-1", {"slug": "shopify-management"}))
    assert db.rolled_back_to_savepoint is True                      # before: left open
