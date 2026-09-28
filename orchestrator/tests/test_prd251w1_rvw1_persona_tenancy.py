"""PRD-251 P251W1-RVW-1: no workspace can change a marketplace agent that installs copy.

Every install copies a marketplace agent's persona, skills and plugin assignments
onto the installing workspace's clone, and the marketplace row is global (no
workspace). ``PUT /api/agents/{agent_id}/persona`` and ``PUT
/api/agents/{agent_id}/plugins`` skipped their workspace check for a row with no
workspace, so an editor of any workspace could plant a persona or plugins there, and
every later install, in any tenant, ran them.

PURE (no database):
  * a clone keeps a shared persona and drops a workspace's own (or a missing one);
  * the boot restore puts a changed persona back, leaves a matching one alone and
    logs what it replaced; every boot runs it for the Socials and Shopify rosters;
  * a request with no workspace finds no agent, before any query, and the plugins
    PUT checks the agent's workspace before its body runs.

@integration (the orchestrator-tests job's Postgres, one rolled-back transaction
each):
  * an owner, and an editor, PUT a custom prompt and then their workspace's own
    persona onto the seeded Social Media Director: 404 both, the row unchanged;
    another workspace's agent is a 404 too; their own agent takes both;
  * an editor's plugins PUT onto the Director is a 404, its plugins and skills
    unchanged; their own agent's is a 200;
  * a marketplace row carrying another workspace's persona clones without it, and
    one carrying a global persona clones with it;
  * a Director changed outside the seed is restored by the next boot and the
    package installs the seed's persona; a Shopify marketplace row the same.
"""
from __future__ import annotations

import logging
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import sqlalchemy as sa  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import api.agent_plugins as agent_plugins_api  # noqa: E402
import api.personas as personas_api  # noqa: E402
import core.auth.workspace_permission as permission_mod  # noqa: E402
import core.seeds.seed_shopify_agents as shopify_seed  # noqa: E402
import core.seeds.seed_socials_package as socials_seed  # noqa: E402
import services.agent_quota as agent_quota  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.auth.workspace_agent import workspace_agent_or_404  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.models.core import Agent  # noqa: E402
from core.models.marketplace_plugins import AgentAssignedPlugin, MarketplacePlugin  # noqa: E402
from core.models.personas import Persona  # noqa: E402
from core.seeds.marketplace_personas import restore_seeded_personas  # noqa: E402
from core.seeds.seed_shopify_agents import SHOPIFY_AGENTS  # noqa: E402
from core.seeds.seed_socials_package import DIRECTOR, SOCIALS_AGENTS  # noqa: E402
from modules.tools.discovery import cascade_installer as ci  # noqa: E402
from tests.test_prd251w1_socials_package import (  # noqa: E402
    INSTALL_ROUTE,
    SKILL_NAMES,
    _boot,
    _client,
    _marketplace_rows,
    _rolled_back,
    _skills_manifest,
    _workspace,
)

DIRECTOR_PERSONA = next(s["custom_persona_prompt"] for s in SOCIALS_AGENTS if s["slug"] == DIRECTOR)
PLANTED = "Planted instructions from another workspace."


# ---------------------------------------------------------------------------
# 1. The clone: a shared persona crosses, a workspace's own never does
# ---------------------------------------------------------------------------


def _marketplace_agent(**overrides):
    values = dict(
        id=11, name="Social Media Director", description="d", agent_type="specialized", configuration={},
        model_config=None, tags=["socials"], status="active", original_creator_id=None, version="1.0.0",
        persona_id=None, custom_persona_prompt=DIRECTOR_PERSONA, use_custom_persona=True, skills=[],
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _session_finding(row):
    """A session whose every ``query(...).filter(...).first()`` finds ``row``."""
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = row
    return db


def test_a_clone_drops_a_persona_that_is_not_a_shared_one(monkeypatch):
    # The shared-persona lookup finds nothing: a workspace's own persona, or none at all.
    monkeypatch.setattr(agent_quota, "require_agent_room", lambda db, workspace_id: None)
    marketplace = _marketplace_agent(persona_id=uuid.uuid4())
    cloned, name = ci.clone_agent_to_workspace(_session_finding(None), uuid.uuid4(), marketplace)
    assert cloned.persona_id is None
    assert cloned.custom_persona_prompt == DIRECTOR_PERSONA and cloned.use_custom_persona is True
    assert name == marketplace.name and cloned.cloned_from_id == marketplace.id
    assert cloned.owner_type == "workspace"


def test_a_clone_keeps_a_shared_persona(monkeypatch):
    monkeypatch.setattr(agent_quota, "require_agent_room", lambda db, workspace_id: None)
    persona_id = uuid.uuid4()
    db = _session_finding(SimpleNamespace(id=persona_id))
    cloned, _name = ci.clone_agent_to_workspace(db, uuid.uuid4(), _marketplace_agent(persona_id=persona_id))
    assert cloned.persona_id == persona_id


# ---------------------------------------------------------------------------
# 2. The boot restore
# ---------------------------------------------------------------------------


def _row(slug, **persona):
    values = dict(id=7, slug=slug, persona_id=None, custom_persona_prompt="the seed's", use_custom_persona=True)
    values.update(persona)
    return SimpleNamespace(**values)


def _session_listing(rows):
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = rows
    return db


@pytest.mark.parametrize("changed", [
    {"custom_persona_prompt": PLANTED},
    {"persona_id": uuid.uuid4()},
    {"use_custom_persona": False},
    {"custom_persona_prompt": None, "use_custom_persona": False, "persona_id": uuid.uuid4()},
])
def test_the_restore_puts_the_seeds_persona_back_and_leaves_a_matching_row_alone(changed):
    tampered, matching = _row("a", **changed), _row("b")
    restored = restore_seeded_personas(_session_listing([tampered, matching]), {"a": "the seed's", "b": "the seed's"})
    assert restored == ["a"]
    for row in (tampered, matching):
        assert (row.persona_id, row.custom_persona_prompt, row.use_custom_persona) == (None, "the seed's", True)


def test_the_restore_logs_what_it_replaced_on_one_line(caplog):
    planted = PLANTED + "\nINFO: a forged log line"
    with caplog.at_level(logging.WARNING, logger="core.seeds.marketplace_personas"):
        restore_seeded_personas(_session_listing([_row("a", custom_persona_prompt=planted)]), {"a": "the seed's"})
    [record] = [r for r in caplog.records if r.name == "core.seeds.marketplace_personas"]
    assert repr(planted) in record.getMessage() and "\n" not in record.getMessage()


def test_the_shopify_restore_covers_every_shopify_agent_with_its_seed(monkeypatch):
    seen = {}
    monkeypatch.setattr(shopify_seed, "restore_seeded_personas", lambda db, prompts: seen.update(prompts) or [])
    shopify_seed.restore_shopify_personas(MagicMock())
    assert seen and seen == {a["slug"]: a["custom_persona_prompt"] for a in SHOPIFY_AGENTS}


def test_every_boot_restores_the_socials_and_the_shopify_personas(monkeypatch):
    seen = []
    monkeypatch.setattr(socials_seed, "_ensure_agent", lambda db, spec: (SimpleNamespace(slug=spec["slug"]), False))
    monkeypatch.setattr(socials_seed, "_link_skills", lambda db, agent, wanted: {"attached": [], "missing": []})
    monkeypatch.setattr(socials_seed, "_ensure_playbook", lambda db, spec, agents: "present")
    monkeypatch.setattr(socials_seed, "restore_seeded_personas", lambda db, prompts: seen.append(dict(prompts)) or [DIRECTOR])
    monkeypatch.setattr(shopify_seed, "restore_shopify_personas", lambda db: ["shopify-ops"])
    outcome = socials_seed.seed_socials_marketplace(MagicMock())
    assert seen == [{s["slug"]: s["custom_persona_prompt"] for s in SOCIALS_AGENTS}]
    assert outcome["personas_restored"] == [DIRECTOR, "shopify-ops"]


# ---------------------------------------------------------------------------
# 3. The route
# ---------------------------------------------------------------------------


def test_a_request_without_a_workspace_finds_no_agent_before_any_query():
    # Agent.workspace_id == None would match every NULL-workspace (marketplace) row.
    db = MagicMock()
    with pytest.raises(HTTPException) as refused:
        workspace_agent_or_404(db, 5, None)
    assert refused.value.status_code == 404
    db.query.assert_not_called()


def test_the_plugins_put_checks_the_agents_workspace_before_its_body_runs():
    route = next(r for r in agent_plugins_api.router.routes
                 if r.path == "/api/agents/{agent_id}/plugins" and "PUT" in r.methods)
    assert agent_plugins_api.caller_owns_agent in [d.call for d in route.dependant.dependencies]


# ---------------------------------------------------------------------------
# 4. On Postgres
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pg_engine():
    # Its own copy, not test_prd251w1_socials_package's: it needs more tables, and an
    # imported fixture reads as an unused import to ruff (F401/F811).
    from core.database.database import get_database_url

    try:
        engine = sa.create_engine(get_database_url(), pool_pre_ping=True)
        with engine.connect() as conn:
            for table in ("agents", "personas", "skills", "agent_skills", "workflow_recipes", "marketplace_packages",
                          "workspace_enabled_skills", "agent_tool_assignments", "workspaces", "marketplace_plugins",
                          "agent_assigned_plugins"):
                conn.execute(sa.text(f"SELECT 1 FROM {table} LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the persona tenancy tests need the test database: {exc}")
    yield engine
    engine.dispose()


def _persona(session, workspace_id=None) -> Persona:
    """A workspace's own persona, or a global one when ``workspace_id`` is None."""
    tag = uuid.uuid4().hex[:10]
    persona = Persona(
        slug=f"rvw1-{tag}", name=f"Voice {tag}", system_prompt=f"Speak as voice {tag}.", is_active=True,
        scope="workspace" if workspace_id else "global", workspace_id=workspace_id,
    )
    session.add(persona)
    session.flush()
    return persona


def _own_agent(session, workspace_id) -> Agent:
    agent = Agent(name=f"Writer {uuid.uuid4().hex[:8]}", agent_type="custom", status="active",
                  owner_type="workspace", owner_id=str(workspace_id), workspace_id=workspace_id)
    session.add(agent)
    session.flush()
    return agent


def _persona_of(agent):
    return agent.persona_id, agent.custom_persona_prompt, bool(agent.use_custom_persona)


def _agents_client(session, ws) -> TestClient:
    """The persona and plugin routes, as a member of ``ws``."""
    app = FastAPI()
    app.include_router(personas_api.router)
    app.include_router(agent_plugins_api.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: RequestContext(
        workspace_id=ws,
        user=UserContext(id="member-1", clerk_user_id="clerk-member-1", system_role="user"),
        auth_type="clerk",
    )
    app.dependency_overrides[get_db] = lambda: session
    return TestClient(app)


def _reloaded(session, agent_id) -> Agent:
    session.expire_all()
    return session.get(Agent, agent_id)


@pytest.mark.integration
@pytest.mark.parametrize("role", ["owner", "editor"])
def test_no_workspace_can_set_a_marketplace_agents_persona_and_its_own_agent_still_can(
    pg_engine, monkeypatch, tmp_path, role,
):
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: role)
    with _rolled_back(pg_engine) as (conn, session):
        _boot(session, _skills_manifest(tmp_path, synced=SKILL_NAMES))
        ws, other = _workspace(conn), _workspace(conn)
        director = _marketplace_rows(session)[0][DIRECTOR]
        before = _persona_of(director)
        own_persona, own_agent, others_agent = _persona(session, ws), _own_agent(session, ws), _own_agent(session, other)
        client = _agents_client(session, ws)

        for agent_id in (director.id, others_agent.id):
            planted = client.put(f"/api/agents/{agent_id}/persona", json={"custom_prompt": PLANTED, "use_custom": True})
            borrowed = client.put(f"/api/agents/{agent_id}/persona", json={"persona_id": str(own_persona.id)})
            assert (planted.status_code, borrowed.status_code) == (404, 404), (planted.text, borrowed.text)
        assert _persona_of(_reloaded(session, director.id)) == before
        assert _persona_of(_reloaded(session, others_agent.id)) == (None, None, False)

        custom = client.put(f"/api/agents/{own_agent.id}/persona", json={"custom_prompt": "Plain and brief.", "use_custom": True})
        assert custom.status_code == 200, custom.text
        assert (custom.json()["custom_persona_prompt"], custom.json()["use_custom_persona"]) == ("Plain and brief.", True)
        chosen = client.put(f"/api/agents/{own_agent.id}/persona", json={"persona_id": str(own_persona.id)})
        assert chosen.status_code == 200, chosen.text
        assert (chosen.json()["persona_id"], chosen.json()["persona_name"]) == (str(own_persona.id), own_persona.name)
        assert _reloaded(session, own_agent.id).persona_id == own_persona.id


@pytest.mark.integration
def test_no_workspace_can_change_a_marketplace_agents_plugins_and_its_own_agent_still_can(
    pg_engine, monkeypatch, tmp_path,
):
    # Installs copy a marketplace agent's plugin assignments (and the skills they
    # bring) into every installing workspace: the same hole as the persona's.
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: "editor")
    with _rolled_back(pg_engine) as (conn, session):
        _boot(session, _skills_manifest(tmp_path, synced=SKILL_NAMES))
        ws = _workspace(conn)
        director = _marketplace_rows(session)[0][DIRECTOR]
        tag = uuid.uuid4().hex[:10]
        plugin = MarketplacePlugin(slug=f"rvw1-{tag}", name=f"Plugin {tag}", version="1.0.0")
        session.add(plugin)
        session.flush()
        session.add(AgentAssignedPlugin(agent_id=director.id, plugin_id=plugin.id, priority=0))
        session.flush()
        skills_before = sorted(s.name for s in director.skills)
        own_agent = _own_agent(session, ws)
        client = _agents_client(session, ws)

        refused = client.put(f"/api/agents/{director.id}/plugins", json={"plugin_ids": []})
        assert refused.status_code == 404, refused.text
        session.expire_all()
        kept = session.query(AgentAssignedPlugin).filter(AgentAssignedPlugin.agent_id == director.id).all()
        assert [a.plugin_id for a in kept] == [plugin.id]
        assert sorted(s.name for s in session.get(Agent, director.id).skills) == skills_before

        own = client.put(f"/api/agents/{own_agent.id}/plugins", json={"plugin_ids": []})
        assert own.status_code == 200, own.text


@pytest.mark.integration
def test_a_marketplace_row_carrying_a_workspaces_own_persona_clones_without_it(pg_engine, tmp_path):
    with _rolled_back(pg_engine) as (conn, session):
        _boot(session, _skills_manifest(tmp_path, synced=SKILL_NAMES))
        director = _marketplace_rows(session)[0][DIRECTOR]
        director.persona_id = _persona(session, _workspace(conn)).id  # however it got there
        session.flush()

        ws = _workspace(conn)
        clone, _name = ci.clone_agent_to_workspace(session, ws, director)
        assert clone.persona_id is None
        assert (clone.custom_persona_prompt, clone.use_custom_persona) == (DIRECTOR_PERSONA, True)
        assert (clone.workspace_id, clone.owner_type, clone.cloned_from_id) == (ws, "workspace", director.id)

        shared = _persona(session)
        director.persona_id = shared.id
        session.flush()
        again, _name = ci.clone_agent_to_workspace(session, _workspace(conn), director)
        assert again.persona_id == shared.id


@pytest.mark.integration
def test_a_director_changed_outside_the_seed_is_restored_at_boot_and_the_install_copies_the_seeds_persona(
    pg_engine, monkeypatch, tmp_path,
):
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: "owner")
    manifest = _skills_manifest(tmp_path, synced=SKILL_NAMES)
    with _rolled_back(pg_engine) as (conn, session):
        _boot(session, manifest)
        director = _marketplace_rows(session)[0][DIRECTOR]
        director.custom_persona_prompt, director.use_custom_persona = PLANTED, True
        director.persona_id = _persona(session, _workspace(conn)).id
        session.commit()

        outcome = _boot(session, manifest)
        assert DIRECTOR in outcome["personas_restored"]

        ws = _workspace(conn)
        installed = _client(session, ws).post(INSTALL_ROUTE)
        assert installed.status_code == 200 and installed.json()["success"] is True, installed.text
        director = _reloaded(session, director.id)
        clone = session.query(Agent).filter(Agent.workspace_id == ws, Agent.cloned_from_id == director.id).one()
        for row in (director, clone):
            assert _persona_of(row) == (None, DIRECTOR_PERSONA, True)


@pytest.mark.integration
def test_a_shopify_marketplace_agent_changed_outside_its_seed_is_restored_at_boot(pg_engine, tmp_path):
    spec = SHOPIFY_AGENTS[0]
    with _rolled_back(pg_engine) as (conn, session):
        row = session.query(Agent).filter(Agent.slug == spec["slug"], Agent.owner_type == "marketplace").first()
        if row is None:
            row = Agent(name=spec["name"], slug=spec["slug"], agent_type=spec["agent_type"], status="active",
                        owner_type="marketplace", owner_id="marketplace", workspace_id=None, is_approved=True)
            session.add(row)
        row.custom_persona_prompt, row.use_custom_persona = PLANTED, True
        session.flush()

        outcome = _boot(session, _skills_manifest(tmp_path, synced=SKILL_NAMES))

        assert spec["slug"] in outcome["personas_restored"]
        assert _persona_of(_reloaded(session, row.id)) == (None, spec["custom_persona_prompt"], True)
