"""PRD-270 W2 US-270-12a (K9, K10) — Automatos Cloud's export of approved community agents and Playbooks.

``services/marketplace_bundles/export_agents.exported_agents`` and ``export_playbooks.exported_playbooks``
on Postgres: the rows are flushed on one session and never committed (the teardown rolls them back), and
every assertion reads only this test's items, by their externalIds, so rows the suite seeded never
matter. Proven here:

* an approved community agent and Playbook export, each bundle one validate_bundle passes;
* an unapproved, inactive, slugged (seeded), system, role-restricted or workspace agent never exports; a
  seeded, unapproved, private or system Playbook never does;
* two approved rows of one lineage export once, the newer (the later row on a tie), and the item stands
  for both rows; a resubmission keeps the externalId and moves the revision;
* a Playbook step on the submitter's workspace agent names it {name}, on a seeded agent {slug}, on an
  exported agent {item}, and so does a step on the workspace agent that agent was submitted from; a
  document step names none;
* a step on another tenant's agent reads nothing of it: with no agent name of its own the Playbook is
  left out, and that agent's name never reaches any bundle;
* a shared persona travels by its slug, a workspace's own persona never does;
* a configuration holding a made-up key, the creator's email (as their name too), created_by, a
  schedule_config with a made-up webhook id, the workspace's id and the Playbook's extra keys never
  reach the bundle bytes;
* a bundle the validator refuses is logged as an error and left out;
* the round trip: rows built from the agent and Playbook fixtures export exactly the fixture bytes.

Without a database the pure checks still run: the lineage hash, the seeded template ids, the display
fields fitted to the format, and a step kept to the format's keys.
"""
from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from sqlalchemy import create_engine, text

from core.models.core import Agent, Skill, User, WorkflowTemplate as WorkflowRecipe, agent_skills
from core.models.marketplace_plugins import AgentAssignedPlugin, MarketplacePlugin
from core.models.personas import Persona
from services.marketplace_bundles import export_playbooks
from services.marketplace_bundles.canonical import canonical_bytes
from services.marketplace_bundles.export_agents import exported_agents
from services.marketplace_bundles.export_playbooks import exported_playbooks
from services.marketplace_bundles.exported import icon_of, publisher_name, tags_of
from services.marketplace_bundles.lineage import external_id, lineage_of
from services.marketplace_bundles.seeded import seeded_template_ids
from services.marketplace_bundles.validate import validate_bundle

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "marketplace_bundles"
# The natural keys the fixtures' externalIds were hashed from (row:<public_id>, row:<template_id>).
FIXTURE_AGENT_PUBLIC_ID = "00000000-0000-4000-8000-000000000001"
FIXTURE_TEMPLATE_ID = "test-weekly-digest"
MADE_UP = "sk-test-made-up"
MADE_UP_WEBHOOK = "whk-test-made-up"
WHEN = datetime(2026, 10, 1, 12, 0, 0)
TABLES = (
    "users", "workspaces", "agents", "workflow_recipes", "skills", "agent_skills", "agent_tool_assignments",
    "marketplace_plugins", "agent_assigned_plugins", "personas",
)


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            for table in TABLES:
                c.execute(text(f"SELECT 1 FROM {table} LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the export tests need Postgres with the marketplace tables: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def cloud(engine, new_session):
    """One uncommitted session: every row is flushed, and the teardown rolls them all back."""
    db = new_session()
    try:
        yield NS(db=db, tag=uuid.uuid4().hex[:10])
    finally:
        new_session.sweep().close()


# --- rows --------------------------------------------------------------------------------------


def _user(cloud, name=None, label="creator"):
    user = User(username=f"prd270w2-{label}-{cloud.tag}", email=f"{label}-{cloud.tag}@prd270w2.test", name=name)
    cloud.db.add(user)
    cloud.db.flush()
    return user


def _workspace(cloud):
    ws = uuid.uuid4()
    cloud.db.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'prd270w2 export')"),
                     {"id": str(ws)})
    return ws


def _agent(cloud, **columns):
    """A marketplace agent K9 exports unless ``columns`` say otherwise; its configuration holds MADE_UP."""
    values = {
        "name": f"Agent {cloud.tag}", "description": "An agent of the export test.", "agent_type": "custom",
        "status": "active", "owner_type": "marketplace", "owner_id": "marketplace", "is_approved": True,
        "version": "1.0.0", "created_at": WHEN, "configuration": {"runtime": "api", "api_key": MADE_UP},
        "created_by": f"created-by-{cloud.tag}@prd270w2.test", "marketplace_icon": "🧭", "tags": ["test"],
        **columns,
    }
    agent = Agent(**values)
    cloud.db.add(agent)
    cloud.db.flush()
    return agent


def _playbook(cloud, **columns):
    """A marketplace Playbook K9 exports unless ``columns`` say otherwise."""
    values = {
        "template_id": f"prd270w2-{uuid.uuid4().hex[:12]}", "name": f"Playbook {uuid.uuid4().hex[:8]}",
        "description": "A Playbook of the export test.", "owner_type": "marketplace", "owner_id": "marketplace",
        "is_approved": True, "is_public": True, "is_system": False, "version": "1.0", "created_at": WHEN,
        "created_by": f"created-by-{cloud.tag}@prd270w2.test", "template_definition": {"steps": []},
        "steps": [], "schedule_config": {"type": "trigger", "trigger_config": {"webhook_id": MADE_UP_WEBHOOK}},
        **columns,
    }
    recipe = WorkflowRecipe(**values)
    cloud.db.add(recipe)
    cloud.db.flush()
    return recipe


def _skill(cloud, agent, name, package=None, priority=0, workspace_id=None):
    skill = Skill(name=name, skill_type="technical", package_slug=package, workspace_id=workspace_id)
    cloud.db.add(skill)
    cloud.db.flush()
    cloud.db.execute(agent_skills.insert().values(agent_id=agent.id, skill_id=skill.id, priority=priority))


def _tool(cloud, agent, tool_id, enabled=True):
    cloud.db.execute(text("INSERT INTO agent_tool_assignments (agent_id, tool_id, enabled) VALUES (:a, :t, :e)"),
                     {"a": agent.id, "t": tool_id, "e": enabled})


def _plugin(cloud, agent, slug, priority=0):
    plugin = MarketplacePlugin(slug=slug, name=f"Plugin {slug}", version="1.0.0")
    cloud.db.add(plugin)
    cloud.db.flush()
    cloud.db.add(AgentAssignedPlugin(agent_id=agent.id, plugin_id=plugin.id, priority=priority))
    cloud.db.flush()


def _persona(cloud, scope="global", workspace_id=None):
    persona = Persona(slug=f"test-persona-{uuid.uuid4().hex[:12]}", name="Test persona", scope=scope,
                      workspace_id=workspace_id)
    cloud.db.add(persona)
    cloud.db.flush()
    return persona


# --- reads ---------------------------------------------------------------------------------------


def _export(cloud):
    agents = exported_agents(cloud.db)
    return NS(agents={item.entry.external_id: item for item in agents},
              playbooks={item.entry.external_id: item for item in exported_playbooks(cloud.db, agents)})


def _agent_id_of(agent):
    return external_id("agent", f"row:{agent.public_id}")


def _playbook_id_of(recipe):
    return external_id("playbook", f"row:{recipe.template_id}")


def _fixture(name):
    return (FIXTURES / f"{name}.json").read_bytes().rstrip(b"\n")


# --- without a database ------------------------------------------------------------------------


def test_the_external_id_is_the_hash_of_the_lineage():
    expected = hashlib.sha256(b"automatos-cloud:agent:from:42").hexdigest()[:32]
    assert external_id("agent", "from:42") == expected
    assert external_id("playbook", "from:42") != expected
    assert lineage_of("agent", NS(cloned_from_id=42, public_id="p")) == "from:42"
    assert lineage_of("agent", NS(cloned_from_id=None, public_id="p")) == "row:p"
    assert lineage_of("playbook", NS(cloned_from_id=None, template_id="t")) == "row:t"
    assert lineage_of("agent", NS(cloned_from_id=None, public_id=None)) is None
    assert external_id("agent", f"row:{FIXTURE_AGENT_PUBLIC_ID}") == json.loads(_fixture("agent"))["externalId"]
    assert external_id("playbook", f"row:{FIXTURE_TEMPLATE_ID}") == json.loads(_fixture("playbook"))["externalId"]


def test_the_seeded_playbooks_are_the_socials_playbooks():
    from core.seeds.seed_socials_package import SOCIALS_PLAYBOOKS

    assert seeded_template_ids() == {spec["template_id"] for spec in SOCIALS_PLAYBOOKS}


def test_display_fields_are_fitted_to_the_format():
    assert icon_of("🔎") == "🔎" and icon_of("robot") == "robot"
    for icon in ("https://cdn.test/icon.png", "//host/x", "a" * 17, None, 7):
        assert icon_of(icon) == ""
    assert tags_of(["a", "", "  ", 7, "a", "b" * 65, "c"]) == ["a", "c"]
    assert tags_of([f"t{n}" for n in range(30)]) == [f"t{n}" for n in range(20)]
    assert tags_of("not a list") == []
    assert publisher_name({1: "Test Publisher"}, 1) == "Test Publisher"
    assert publisher_name({1: "x" * 300}, 1) == "x" * 100
    assert publisher_name({}, 1) == "Automatos" and publisher_name({}, None) == "Automatos"


def test_a_step_keeps_only_the_formats_keys():
    stored = {
        "step_id": "s", "order": 1, "agent_id": 7, "agent_name": "Helper", "prompt_template": "Do it.",
        "until": None, "pass_to": None, "secret_extra": "no", "config": {"title": "T", "webhook": "no", "data": None},
    }
    step = export_playbooks._step(stored, "a1")
    assert step == {"step_id": "s", "order": 1, "prompt_template": "Do it.", "pass_to": None,
                    "config": {"title": "T"}, "agent": "a1"}
    assert stored["config"] == {"title": "T", "webhook": "no", "data": None}  # the row's step is untouched


# --- on Postgres ---------------------------------------------------------------------------------


def test_an_approved_community_agent_and_playbook_export(cloud):
    agent = _agent(cloud)
    recipe = _playbook(cloud)
    found = _export(cloud)
    item = found.agents[_agent_id_of(agent)]
    assert validate_bundle(item.bundle) is item.bundle and item.bundle["name"] == agent.name
    assert item.entry.row_ids == frozenset({agent.id}) and item.entry.version == "1.0.0"
    assert hashlib.sha256(canonical_bytes(item.bundle)).hexdigest() == item.revision
    playbook = found.playbooks[_playbook_id_of(recipe)]
    assert validate_bundle(playbook.bundle)["playbook"]["steps"] == []


def test_agents_k9_leaves_out_never_export(cloud):
    workspace_id = _workspace(cloud)
    left_out = [
        _agent(cloud, is_approved=False), _agent(cloud, status="inactive"),
        _agent(cloud, slug=f"seeded-{cloud.tag}"), _agent(cloud, is_system_agent=True),
        _agent(cloud, required_role="admin"),
        _agent(cloud, owner_type="workspace", owner_id=str(workspace_id), workspace_id=workspace_id),
    ]
    exported = _export(cloud).agents
    assert not {_agent_id_of(agent) for agent in left_out} & set(exported)


def test_playbooks_k9_leaves_out_never_export(cloud, monkeypatch):
    seeded = _playbook(cloud)
    monkeypatch.setattr(export_playbooks, "seeded_template_ids", lambda: frozenset({seeded.template_id}))
    left_out = [
        seeded, _playbook(cloud, is_approved=False), _playbook(cloud, is_public=False),
        _playbook(cloud, is_system=True), _playbook(cloud, owner_type="workspace"),
    ]
    exported = _export(cloud).playbooks
    assert not {_playbook_id_of(recipe) for recipe in left_out} & set(exported)


def test_one_lineage_exports_once_the_newer(cloud):
    workspace_id = _workspace(cloud)
    source = _agent(cloud, owner_type="workspace", workspace_id=workspace_id, is_approved=False)
    older = _agent(cloud, cloned_from_id=source.id, name=f"Older {cloud.tag}", created_at=WHEN)
    newer = _agent(cloud, cloned_from_id=source.id, name=f"Newer {cloud.tag}", created_at=WHEN + timedelta(days=1))
    source_recipe = _playbook(cloud, owner_type="workspace", is_approved=False)
    first = _playbook(cloud, cloned_from_id=source_recipe.id, name=f"First {cloud.tag}", created_at=WHEN)
    second = _playbook(cloud, cloned_from_id=source_recipe.id, name=f"Second {cloud.tag}",
                       created_at=WHEN + timedelta(days=1))
    found = _export(cloud)
    agent = found.agents[external_id("agent", f"from:{source.id}")]
    assert agent.bundle["name"] == newer.name and agent.entry.row_ids == frozenset({older.id, newer.id})
    assert _agent_id_of(older) not in found.agents and _agent_id_of(newer) not in found.agents
    playbook = found.playbooks[external_id("playbook", f"from:{source_recipe.id}")]
    assert playbook.bundle["name"] == second.name and playbook.entry.row_ids == frozenset({first.id, second.id})


def test_a_tie_on_created_at_goes_to_the_later_row(cloud):
    workspace_id = _workspace(cloud)
    source = _agent(cloud, owner_type="workspace", workspace_id=workspace_id, is_approved=False)
    _agent(cloud, cloned_from_id=source.id, name=f"Earlier {cloud.tag}", created_at=WHEN)
    later = _agent(cloud, cloned_from_id=source.id, name=f"Later {cloud.tag}", created_at=WHEN)
    assert _export(cloud).agents[external_id("agent", f"from:{source.id}")].bundle["name"] == later.name


def test_a_resubmission_keeps_the_external_id_and_moves_the_revision(cloud):
    workspace_id = _workspace(cloud)
    source = _agent(cloud, owner_type="workspace", workspace_id=workspace_id, is_approved=False)
    _agent(cloud, cloned_from_id=source.id, description="The first submission.", created_at=WHEN)
    key = external_id("agent", f"from:{source.id}")
    before = _export(cloud).agents[key]
    _agent(cloud, cloned_from_id=source.id, description="The resubmission.", created_at=WHEN + timedelta(days=2))
    after = _export(cloud).agents[key]
    assert after.entry.external_id == before.entry.external_id == after.bundle["externalId"]
    assert after.revision != before.revision and after.bundle["description"] == "The resubmission."


def test_a_step_names_its_agent_by_slug_item_or_name(cloud):
    workspace_id = _workspace(cloud)
    seeded = _agent(cloud, slug=f"seeded-{cloud.tag}")
    community = _agent(cloud)
    own = _agent(cloud, owner_type="workspace", workspace_id=workspace_id, name=f"Own helper {cloud.tag}")
    submitted = _agent(cloud, owner_type="workspace", workspace_id=workspace_id, name=f"Submitted {cloud.tag}")
    _agent(cloud, cloned_from_id=submitted.id)
    steps = [
        {"step_id": f"s{n}", "order": n, "agent_id": row.id, "agent_name": row.name, "prompt_template": "Go."}
        for n, row in enumerate((seeded, community, own, seeded, submitted), start=1)
    ] + [{"step_id": "doc", "order": 6, "type": "generate_document", "agent_id": community.id,
          "config": {"format": "report"}}]
    recipe = _playbook(cloud, steps=steps)
    playbook = _export(cloud).playbooks[_playbook_id_of(recipe)].bundle["playbook"]
    assert playbook["agents"] == {
        "a1": {"slug": seeded.slug}, "a2": {"item": _agent_id_of(community)}, "a3": {"name": own.name},
        "a4": {"item": external_id("agent", f"from:{submitted.id}")},
    }
    assert [step["agent"] for step in playbook["steps"]] == ["a1", "a2", "a3", "a1", "a4", None]


def test_a_step_on_another_tenants_agent_reads_nothing_of_it(cloud):
    theirs = _agent(cloud, owner_type="workspace", workspace_id=_workspace(cloud), is_approved=False,
                    name=f"Their private agent {cloud.tag}")
    unnamed = _playbook(cloud, steps=[{"step_id": "s1", "order": 1, "agent_id": theirs.id, "prompt_template": "Go."}])
    named = _playbook(cloud, steps=[
        {"step_id": "s1", "order": 1, "agent_id": theirs.id, "agent_name": "My helper", "prompt_template": "Go."},
    ])
    found = _export(cloud)
    assert _playbook_id_of(unnamed) not in found.playbooks
    assert found.playbooks[_playbook_id_of(named)].bundle["playbook"]["agents"] == {"a1": {"name": "My helper"}}
    for item in (*found.agents.values(), *found.playbooks.values()):
        assert theirs.name not in canonical_bytes(item.bundle).decode("utf-8")


def test_only_a_shared_persona_travels(cloud):
    workspace_id = _workspace(cloud)
    shared = _persona(cloud)
    own = _persona(cloud, scope="workspace", workspace_id=workspace_id)
    with_shared = _agent(cloud, persona_id=shared.id)
    with_own = _agent(cloud, persona_id=own.id)
    exported = _export(cloud).agents
    assert exported[_agent_id_of(with_shared)].bundle["agent"]["persona"] == shared.slug
    assert exported[_agent_id_of(with_own)].bundle["agent"]["persona"] is None


def test_nothing_private_reaches_the_bundle_bytes(cloud):
    workspace_id = _workspace(cloud)
    creator = _user(cloud, name=f"named-{cloud.tag}@prd270w2.test")
    agent = _agent(cloud, original_creator_id=creator.id)
    own = _agent(cloud, owner_type="workspace", workspace_id=workspace_id, name=f"Own {cloud.tag}")
    _skill(cloud, agent, f"own-skill-{cloud.tag}", workspace_id=workspace_id)
    recipe = _playbook(cloud, original_creator_id=creator.id, steps=[
        {"step_id": "s1", "order": 1, "agent_id": own.id, "agent_name": own.name, "prompt_template": "Go.",
         "workspace_id": str(workspace_id)},
    ], execution_config={"mode": "sequential", "quality_threshold": 0.8})
    found = _export(cloud)
    for item in (found.agents[_agent_id_of(agent)], found.playbooks[_playbook_id_of(recipe)]):
        data = canonical_bytes(item.bundle).decode("utf-8")
        assert item.bundle["publisher"] == {"name": "Automatos"}
        for private in (MADE_UP, MADE_UP_WEBHOOK, creator.email, creator.name, agent.created_by, str(workspace_id),
                        "schedule", "configuration", "quality_threshold", "created_by", "own-skill-"):
            assert private not in data


def test_a_refused_bundle_is_logged_and_left_out(cloud, caplog):
    agent = _agent(cloud, model_config={"model_id": "not a model id", "provider": "openrouter"})
    recipe = _playbook(cloud, steps=[{"step_id": "s1", "order": 1, "prompt_template": "No agent names this."}])
    with caplog.at_level(logging.ERROR, logger="services.marketplace_bundles.exported"):
        found = _export(cloud)
    assert _agent_id_of(agent) not in found.agents and _playbook_id_of(recipe) not in found.playbooks
    refused = [record.getMessage() for record in caplog.records if record.levelno == logging.ERROR]
    assert any(f"agent row {agent.id} is left out" in message for message in refused)
    assert any(f"playbook row {recipe.id} is left out" in message for message in refused)
    assert not any("not a model id" in message for message in refused)


# --- the round trip --------------------------------------------------------------------------------


def _fixture_agent(cloud):
    """Rows built from agent.json, with what never travels beside them (a disabled tool, a workspace's
    own skill, extra model settings, the made-up key in the configuration)."""
    bundle, workspace_id = json.loads(_fixture("agent")), _workspace(cloud)
    body = bundle["agent"]
    agent = _agent(
        cloud, public_id=uuid.UUID(FIXTURE_AGENT_PUBLIC_ID), name=bundle["name"],
        description=bundle["description"], agent_type=body["agentType"], marketplace_category=bundle["category"],
        marketplace_icon=bundle["icon"], tags=bundle["tags"], version=bundle["version"],
        original_creator_id=_user(cloud, name=bundle["publisher"]["name"]).id,
        custom_persona_prompt=body["prompt"]["text"], use_custom_persona=body["prompt"]["custom"],
        model_config={"model_id": body["model"]["modelId"], "provider": body["model"]["provider"], "temperature": 0.2},
    )
    for priority, skill in zip(range(len(body["skills"]), 0, -1), body["skills"]):
        _skill(cloud, agent, skill["name"], skill["package"], priority)
    _skill(cloud, agent, "workspace-own-skill", None, 99, workspace_id)
    for app in body["apps"]:
        _tool(cloud, agent, app.lower())
    _tool(cloud, agent, "notion", enabled=False)
    for priority, slug in enumerate(body["plugins"]):
        _plugin(cloud, agent, slug, priority)
    return agent


def _fixture_playbook(cloud, exported_agent):
    """Rows built from playbook.json: a1 the exported agent, a2 a seeded agent, a3 a workspace agent."""
    bundle, workspace_id = json.loads(_fixture("playbook")), _workspace(cloud)
    body = bundle["playbook"]
    rows = {
        "a1": exported_agent,
        "a2": _agent(cloud, slug=body["agents"]["a2"]["slug"], name=f"Seeded {cloud.tag}"),
        "a3": _agent(cloud, owner_type="workspace", workspace_id=workspace_id, name=body["agents"]["a3"]["name"]),
    }
    steps = [
        {**{key: value for key, value in step.items() if key != "agent"},
         **({"agent_id": rows[step["agent"]].id, "agent_name": rows[step["agent"]].name} if step["agent"] else {})}
        for step in body["steps"]
    ]
    return _playbook(
        cloud, template_id=FIXTURE_TEMPLATE_ID, name=bundle["name"], description=bundle["description"],
        marketplace_category=bundle["category"], marketplace_icon=bundle["icon"], tags=bundle["tags"],
        version=bundle["version"], original_creator_id=_user(cloud, name=bundle["publisher"]["name"], label="pb").id,
        steps=steps, inputs=body["inputs"], outputs=body["outputs"],
        execution_config={**body["execution"], "quality_threshold": 0.8},
        required_tools=[app.lower() for app in body["apps"]],
    )


def test_the_fixtures_round_trip(cloud):
    agent = _fixture_agent(cloud)
    _fixture_playbook(cloud, agent)
    found = _export(cloud)
    agent_item = found.agents[json.loads(_fixture("agent"))["externalId"]]
    playbook_item = found.playbooks[json.loads(_fixture("playbook"))["externalId"]]
    assert canonical_bytes(agent_item.bundle) == _fixture("agent")
    assert canonical_bytes(playbook_item.bundle) == _fixture("playbook")
    assert agent_item.revision == hashlib.sha256(_fixture("agent")).hexdigest()
    assert playbook_item.revision == hashlib.sha256(_fixture("playbook")).hexdigest()
