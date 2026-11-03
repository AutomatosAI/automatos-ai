"""PRD-270 W2 US-270-13b (§5.6, K14) — locally: an Automatos Cloud agent, written as a marketplace row.

``import_agents.upsert_agent`` on Postgres, each test in one transaction rolled back at its end (the
``test_prd251w1_socials_package.py`` pattern: the install commits, so the session joins the transaction
through a savepoint). The bundle is the corpus's agent (``tests/fixtures/marketplace_bundles/agent.json``)
with this test's own names, validated before use. Proven:

* a bundle writes an approved marketplace row (no workspace, no slug, ``local_version``, the publisher as
  its creator) that ``GET /api/marketplace/items`` answers a member, with its shared persona, its global
  skills in the bundle's order, its apps as ``agent_tool_assignments`` rows and its plugins, and its
  mapping;
* a persona, model, skill and plugin this install lacks are recorded in the mapping's ``unresolved`` and
  the agent is still written, without them; a workspace's persona or skill and an unapproved plugin never
  resolve; a skill, app or plugin named twice is linked once;
* the same revision again is unchanged and writes nothing; a new revision moves the same row and its
  mapping and leaves a workspace copy's columns, skills, tool and plugin rows byte-equal;
* a name a seeded or an operator-made marketplace agent holds (in another case) refuses the bundle and
  leaves that row as it was; so does an update renamed onto one, a second Cloud agent of one name (the
  first written keeps it), and a mapped row that is no longer a marketplace agent;
* an agent reference resolves ``{slug}`` and ``{item}`` to approved marketplace rows only, ``{name}`` never;
* ``install_item`` on the Cloud row clones it through the existing cascade, with its tools, skills, model
  and plugin, and the copy's version is the row's.
"""
from __future__ import annotations

import copy
import json
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest
import sqlalchemy as sa
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import api.marketplace as marketplace_api
import core.auth.workspace_permission as permission_mod
import core.models  # noqa: F401  (registers every table the foreign keys name)
import services.agent_quota as agent_quota
from core.auth.dependencies import RequestContext, UserContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from core.models.core import Agent, LLMModel, Skill, agent_skills
from core.models.marketplace_cloud import MarketplaceCloudItem
from core.models.marketplace_plugins import AgentAssignedPlugin, MarketplacePlugin
from core.models.personas import Persona
from services.marketplace_bundles import import_agents, resolve
from services.marketplace_bundles.canonical import local_version, revision_of
from services.marketplace_bundles.validate import validate_bundle
from services.marketplace_cloud.pull import NEW, REFUSED, UNCHANGED, UPDATED

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "marketplace_bundles" / "agent.json"
PUBLISHER = "Test Publisher"
APPS = ["GITHUB", "SLACK"]


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = sa.create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as conn:
            for probe in (
                "SELECT agent_id, unresolved FROM marketplace_cloud_items LIMIT 1",
                "SELECT tool_id, enabled FROM agent_tool_assignments LIMIT 1",
                "SELECT priority FROM agent_skills LIMIT 1",
                "SELECT plugin_id, priority FROM agent_assigned_plugins LIMIT 1",
                "SELECT slug, scope FROM personas LIMIT 1",
                "SELECT package_slug FROM skills LIMIT 1",
                "SELECT approval_status FROM marketplace_plugins LIMIT 1",
                "SELECT serving_provider FROM llm_models LIMIT 1",
                "SELECT id FROM composio_entities LIMIT 1",
            ):
                conn.execute(sa.text(probe))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the Cloud agent import tests need Postgres with the PRD-270 W2 schema: {exc}")
    yield eng
    eng.dispose()


@contextmanager
def _rolled_back(engine):
    with engine.connect() as conn:
        trans = conn.begin()
        session = Session(bind=conn, join_transaction_mode="create_savepoint")
        try:
            conn.execute(sa.text("SET LOCAL lock_timeout = '5s'"))
            yield session
        finally:
            session.close()
            trans.rollback()


@pytest.fixture
def db(engine):
    with _rolled_back(engine) as session:
        yield session


# --------------------------------------------------------------------------- bundles and rows


def _bundle(tag: str, external_id: str = "", **changes) -> dict:
    """The corpus's agent with this test's names (``changes`` replace keys of its agent object), validated."""
    bundle = json.loads(FIXTURE.read_text(encoding="utf-8"))
    body = {
        **bundle["agent"],
        "persona": f"t270ia-{tag}",
        "model": {"modelId": f"test-vendor/t270ia-{tag}", "provider": "openrouter"},
        "skills": [
            {"name": f"t270ia-summarise-{tag}", "package": f"t270ia-kit-{tag}"},
            {"name": f"t270ia-cite-{tag}", "package": None},
        ],
        "plugins": [f"t270ia-kit-{tag}"],
        "apps": list(APPS),
        **changes,
    }
    made = {**bundle, "externalId": external_id or uuid.uuid4().hex, "name": f"Test Cloud Helper {tag}", "agent": body}
    validate_bundle(made)
    return made


def _renamed(bundle: dict, name: str, **changes) -> dict:
    made = copy.deepcopy(bundle)
    made["name"] = name
    made["agent"].update(changes)
    validate_bundle(made)
    return made


def _hold_everything(db, bundle: dict) -> dict:
    """A shared persona, a model, the skills and the plugin the bundle names, as this install's rows."""
    body = bundle["agent"]
    persona = Persona(slug=body["persona"], name="Test persona", scope="global", workspace_id=None)
    model = LLMModel(
        provider="test-vendor", serving_provider="openrouter", model_id=body["model"]["modelId"],
        display_name="Test model", context_window=1000, max_output_tokens=100,
    )
    plugin = MarketplacePlugin(
        slug=body["plugins"][0], name="Test kit", version="1.0.0", approval_status="approved", is_active=True,
    )
    skills = [
        Skill(name=s["name"], description="Test skill", skill_type="technical", is_active=True, workspace_id=None,
              package_slug=s["package"])
        for s in body["skills"]
    ]
    db.add_all([persona, model, plugin, *skills])
    db.flush()
    return {"persona": persona.id, "model": model.id, "plugin": plugin.id, "skills": [s.id for s in skills]}


def _upsert(db, bundle: dict):
    return import_agents.upsert_agent(db, bundle, revision_of(bundle), PUBLISHER)


def _mapping(db, bundle: dict):
    return (
        db.query(MarketplaceCloudItem)
        .filter(MarketplaceCloudItem.item_type == "agent", MarketplaceCloudItem.external_id == bundle["externalId"])
        .one_or_none()
    )


def _workspace(db) -> uuid.UUID:
    ws = uuid.uuid4()
    db.execute(sa.text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'prd270w2 import agents')"),
               {"id": str(ws)})
    return ws


def _marketplace_agent(db, name: str, slug=None) -> int:
    row = Agent(name=name, agent_type="custom", owner_type="marketplace", owner_id="marketplace", workspace_id=None,
                is_approved=True, slug=slug, version="1.0.0", custom_persona_prompt="Its own prompt.")
    db.add(row)
    db.flush()
    db.execute(sa.text("INSERT INTO agent_tool_assignments (agent_id, tool_id, enabled) VALUES (:id, 'NOTION', true)"),
               {"id": row.id})
    return row.id


def _snapshot(db, agent_id: int):
    """The agent's row and every row of its skills, tools and plugins, as stored."""
    def rows(sql):
        return [dict(r) for r in db.execute(sa.text(sql), {"id": agent_id}).mappings().all()]
    return (
        rows("SELECT * FROM agents WHERE id = :id"),
        rows("SELECT * FROM agent_skills WHERE agent_id = :id ORDER BY skill_id"),
        rows("SELECT * FROM agent_tool_assignments WHERE agent_id = :id ORDER BY tool_id"),
        rows("SELECT * FROM agent_assigned_plugins WHERE agent_id = :id ORDER BY plugin_id"),
    )


def _skill_ids(db, agent_id: int):
    """The agent's skill ids, highest priority first."""
    return [r.skill_id for r in db.execute(
        sa.select(agent_skills.c.skill_id).where(agent_skills.c.agent_id == agent_id)
        .order_by(agent_skills.c.priority.desc()),
    )]


def _apps(db, agent_id: int):
    return sorted(r[0] for r in db.execute(
        sa.text("SELECT tool_id FROM agent_tool_assignments WHERE agent_id = :id AND enabled"), {"id": agent_id},
    ))


def _plugin_ids(db, agent_id: int):
    rows = db.query(AgentAssignedPlugin).filter(AgentAssignedPlugin.agent_id == agent_id)
    return [r.plugin_id for r in rows.order_by(AgentAssignedPlugin.priority)]


def _client(db, ws: uuid.UUID) -> TestClient:
    app = FastAPI()
    app.include_router(marketplace_api.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: RequestContext(
        workspace_id=ws,
        user=UserContext(id="t270ia-member", clerk_user_id="clerk-t270ia-member", system_role="user"),
        auth_type="clerk",
    )
    app.dependency_overrides[get_db] = lambda: db
    return TestClient(app)


# --------------------------------------------------------------------------- a new agent


def test_a_bundle_writes_an_approved_marketplace_agent_that_the_list_answers(db):
    tag = uuid.uuid4().hex[:10]
    bundle = _bundle(tag)
    held = _hold_everything(db, bundle)

    outcome, row = _upsert(db, bundle)
    db.commit()

    assert outcome == NEW
    revision = revision_of(bundle)
    assert (row.owner_type, row.owner_id, row.workspace_id, row.slug) == ("marketplace", "marketplace", None, None)
    assert (row.is_approved, row.is_featured, row.status) == (True, False, "active")
    assert row.version == local_version("1.0.0", revision) and row.created_by == PUBLISHER
    assert row.name == bundle["name"] and row.description == bundle["description"] and row.agent_type == "custom"
    assert (row.marketplace_category, row.marketplace_icon, row.tags) == ("research", "🔎", ["research", "test"])
    assert row.model_config == {"model_id": f"test-vendor/t270ia-{tag}", "provider": "openrouter"}
    assert row.persona_id == held["persona"] and row.configuration == {} and row.public_id is not None
    assert row.custom_persona_prompt == bundle["agent"]["prompt"]["text"] and row.use_custom_persona is True
    assert _skill_ids(db, row.id) == held["skills"]
    assert _apps(db, row.id) == APPS and _plugin_ids(db, row.id) == [held["plugin"]]

    mapping = _mapping(db, bundle)
    assert (mapping.agent_id, mapping.version, mapping.revision, mapping.unresolved) == (row.id, "1.0.0", revision, [])

    listed = _client(db, uuid.uuid4()).get("/api/marketplace/items", params={"type": "agent", "search": tag})
    assert listed.status_code == 200, listed.text
    [item] = [item for item in listed.json() if item["id"] == row.id]
    assert item["is_approved"] is True and item["version"] == row.version and item["creator_name"] == PUBLISHER
    assert set(item["metadata"]["tool_names"]) == set(APPS)


def test_what_this_install_lacks_is_recorded_unresolved_and_the_agent_is_still_written(db):
    tag = uuid.uuid4().hex[:10]
    bundle = _bundle(tag)

    outcome, row = _upsert(db, bundle)

    assert outcome == NEW and row.id is not None
    body = bundle["agent"]
    assert _mapping(db, bundle).unresolved == [
        {"kind": "persona", "name": body["persona"]},
        {"kind": "model", "name": body["model"]["modelId"]},
        {"kind": "skill", "name": body["skills"][0]["name"], "package": body["skills"][0]["package"]},
        {"kind": "skill", "name": body["skills"][1]["name"]},
        {"kind": "plugin", "name": body["plugins"][0]},
    ]
    assert row.persona_id is None and row.model_config["model_id"] == body["model"]["modelId"]
    assert _skill_ids(db, row.id) == [] and _plugin_ids(db, row.id) == [] and _apps(db, row.id) == APPS


def test_a_skill_app_or_plugin_named_twice_is_written_once(db):
    # The format caps each list but doesn't forbid a repeat; the link tables' unique keys would refuse one.
    tag = uuid.uuid4().hex[:10]
    plain = _bundle(tag)
    body = plain["agent"]
    bundle = _bundle(
        tag, external_id=plain["externalId"], apps=["SLACK", "GITHUB", "SLACK"],
        skills=[body["skills"][0], body["skills"][1], body["skills"][0]], plugins=body["plugins"] * 2,
    )
    held = _hold_everything(db, plain)  # the same names, each held once

    outcome, row = _upsert(db, bundle)
    db.commit()

    assert outcome == NEW
    assert _skill_ids(db, row.id) == held["skills"] and _plugin_ids(db, row.id) == [held["plugin"]]
    assert _apps(db, row.id) == APPS


def test_a_workspace_skill_persona_or_an_unapproved_plugin_never_resolves(db):
    tag = uuid.uuid4().hex[:10]
    bundle = _bundle(tag)
    body = bundle["agent"]
    ws = _workspace(db)
    db.add_all([
        Persona(slug=body["persona"], name="A workspace's own", scope="workspace", workspace_id=ws),
        Skill(name=body["skills"][1]["name"], skill_type="technical", is_active=True, workspace_id=ws),
        MarketplacePlugin(slug=body["plugins"][0], name="Pending kit", version="1.0.0", approval_status="pending"),
    ])
    db.flush()

    _outcome, row = _upsert(db, bundle)

    kinds = [entry["kind"] for entry in _mapping(db, bundle).unresolved]
    assert kinds == ["persona", "model", "skill", "skill", "plugin"]
    assert row.persona_id is None and _skill_ids(db, row.id) == [] and _plugin_ids(db, row.id) == []


# --------------------------------------------------------------------------- the same revision, a new one


def test_the_same_revision_again_is_unchanged_and_writes_nothing(db):
    bundle = _bundle(uuid.uuid4().hex[:10])
    _hold_everything(db, bundle)
    _outcome, row = _upsert(db, bundle)
    db.commit()
    before = (_snapshot(db, row.id), dict(db.execute(
        sa.text("SELECT * FROM marketplace_cloud_items WHERE agent_id = :id"), {"id": row.id}).mappings().one()))

    outcome, again = _upsert(db, bundle)
    db.commit()

    assert outcome == UNCHANGED and again.id == row.id
    after = (_snapshot(db, row.id), dict(db.execute(
        sa.text("SELECT * FROM marketplace_cloud_items WHERE agent_id = :id"), {"id": row.id}).mappings().one()))
    assert after == before
    assert db.query(Agent).filter(Agent.name == bundle["name"]).count() == 1


def _workspace_copy(db, row_id: int, ws: uuid.UUID) -> int:
    """A workspace's copy of the Cloud row, as an install leaves it: its own skills, tools and plugins."""
    cloud = db.get(Agent, row_id)
    made = Agent(
        name=cloud.name, agent_type=cloud.agent_type, owner_type="workspace", owner_id=str(ws), workspace_id=ws,
        cloned_from_id=cloud.id, version=cloud.version, model_config=cloud.model_config, persona_id=cloud.persona_id,
        custom_persona_prompt=cloud.custom_persona_prompt, use_custom_persona=True, is_approved=True,
    )
    db.add(made)
    db.flush()
    db.execute(sa.insert(agent_skills), [{"agent_id": made.id, "skill_id": s} for s in _skill_ids(db, cloud.id)])
    db.execute(sa.text("INSERT INTO agent_tool_assignments (agent_id, tool_id, enabled)"
                       " SELECT :copy, tool_id, enabled FROM agent_tool_assignments WHERE agent_id = :id"),
               {"copy": made.id, "id": cloud.id})
    db.add_all([AgentAssignedPlugin(agent_id=made.id, plugin_id=p) for p in _plugin_ids(db, cloud.id)])
    db.flush()
    return made.id


def test_a_new_revision_moves_the_same_row_and_leaves_a_workspace_copy_byte_equal(db):
    tag = uuid.uuid4().hex[:10]
    first = _bundle(tag)
    held = _hold_everything(db, first)
    _outcome, row = _upsert(db, first)
    db.commit()
    copy_id = _workspace_copy(db, row.id, _workspace(db))
    db.commit()
    copy_before = _snapshot(db, copy_id)

    second = _renamed(
        first, f"Test Cloud Helper renamed {tag}", prompt={"custom": False, "text": "A newer prompt."},
        model={"modelId": None, "provider": None}, apps=["NOTION"], skills=[first["agent"]["skills"][1]], plugins=[],
    )
    second["version"] = "1.1.0"
    validate_bundle(second)
    outcome, moved = _upsert(db, second)
    db.commit()

    assert outcome == UPDATED and moved.id == row.id
    db.expire_all()
    cloud = db.get(Agent, row.id)
    assert cloud.name == second["name"] and cloud.version == local_version("1.1.0", revision_of(second))
    assert (cloud.custom_persona_prompt, cloud.use_custom_persona, cloud.model_config) == ("A newer prompt.", False, None)
    assert _apps(db, row.id) == ["NOTION"] and _skill_ids(db, row.id) == [held["skills"][1]]
    assert _plugin_ids(db, row.id) == [] and cloud.is_approved is True and cloud.slug is None
    mapping = _mapping(db, second)
    assert (mapping.agent_id, mapping.version, mapping.revision) == (row.id, "1.1.0", revision_of(second))
    assert db.query(MarketplaceCloudItem).filter(MarketplaceCloudItem.agent_id == row.id).count() == 1
    assert _snapshot(db, copy_id) == copy_before


# --------------------------------------------------------------------------- refused


@pytest.mark.parametrize("seeded", [True, False], ids=["seeded", "operator-made"])
def test_a_name_held_by_a_marketplace_agent_without_a_mapping_refuses_it(db, seeded):
    tag = uuid.uuid4().hex[:10]
    held_id = _marketplace_agent(db, f"Held Helper {tag}", slug=f"t270ia-seeded-{tag}" if seeded else None)
    db.commit()
    held_before = _snapshot(db, held_id)
    bundle = _renamed(_bundle(tag), f"HELD helper {tag.upper()}")

    assert _upsert(db, bundle) == (REFUSED, None)
    db.commit()

    assert _mapping(db, bundle) is None
    assert db.query(Agent).filter(sa.func.lower(Agent.name) == f"held helper {tag}").count() == 1
    assert _snapshot(db, held_id) == held_before


def test_an_update_renamed_onto_a_held_name_is_refused_and_the_row_stays_at_its_revision(db):
    tag = uuid.uuid4().hex[:10]
    first = _bundle(tag)
    _outcome, row = _upsert(db, first)
    _marketplace_agent(db, f"Seeded Helper {tag}", slug=f"t270ia-seeded-{tag}")
    db.commit()
    before = _snapshot(db, row.id)

    assert _upsert(db, _renamed(first, f"Seeded Helper {tag}")) == (REFUSED, None)
    db.commit()

    assert _mapping(db, first).revision == revision_of(first) and _snapshot(db, row.id) == before


def test_a_second_cloud_agent_of_one_name_is_refused_and_so_is_a_rename_onto_one(db):
    # The Playbook install finds a helper by name, .first() of any: the first Cloud agent written keeps it.
    tag = uuid.uuid4().hex[:10]
    first = _bundle(tag)
    other = _renamed(_bundle(f"{tag}x"), first["name"].upper())
    _outcome, row = _upsert(db, first)

    assert _upsert(db, other) == (REFUSED, None)
    assert _mapping(db, other) is None

    third = _bundle(f"{tag}y")
    _outcome, third_row = _upsert(db, third)
    db.commit()
    before = _snapshot(db, third_row.id)
    assert _upsert(db, _renamed(third, first["name"])) == (REFUSED, None)
    db.commit()
    assert _snapshot(db, third_row.id) == before and _mapping(db, third).revision == revision_of(third)
    assert db.query(Agent).filter(sa.func.lower(Agent.name) == first["name"].lower()).one().id == row.id


def test_a_mapped_row_that_is_no_longer_a_marketplace_agent_is_never_written(db):
    tag = uuid.uuid4().hex[:10]
    first = _bundle(tag)
    _outcome, row = _upsert(db, first)
    ws = _workspace(db)
    db.execute(sa.text("UPDATE agents SET owner_type = 'workspace', owner_id = :ws, workspace_id = CAST(:ws AS uuid)"
                       " WHERE id = :id"), {"ws": str(ws), "id": row.id})
    db.commit()
    before = _snapshot(db, row.id)

    assert _upsert(db, _renamed(first, f"Test Cloud Helper newer {tag}")) == (REFUSED, None)
    db.commit()

    assert _snapshot(db, row.id) == before


# --------------------------------------------------------------------------- an agent reference


def test_an_agent_ref_resolves_only_to_an_approved_marketplace_row(db):
    tag = uuid.uuid4().hex[:10]
    bundle = _bundle(tag)
    _outcome, cloud = _upsert(db, bundle)
    seeded_id = _marketplace_agent(db, f"Seeded Helper {tag}", slug=f"t270ia-seeded-{tag}")
    pending_id = _marketplace_agent(db, f"Pending Helper {tag}", slug=f"t270ia-pending-{tag}")
    db.execute(sa.text("UPDATE agents SET is_approved = false WHERE id = :id"), {"id": pending_id})
    ws = _workspace(db)
    db.add(Agent(name=f"Workspace Helper {tag}", agent_type="custom", owner_type="workspace", owner_id=str(ws),
                 workspace_id=ws, slug=f"t270ia-own-{tag}"))
    db.flush()

    assert resolve.agent_for_ref(db, {"slug": f"t270ia-seeded-{tag}"}) == (seeded_id, f"Seeded Helper {tag}")
    assert resolve.agent_for_ref(db, {"item": bundle["externalId"]}) == (cloud.id, bundle["name"])
    assert resolve.agent_for_ref(db, {"name": f"Seeded Helper {tag}"}) is None
    assert resolve.agent_for_ref(db, {"slug": f"t270ia-pending-{tag}"}) is None
    assert resolve.agent_for_ref(db, {"slug": f"t270ia-own-{tag}"}) is None
    assert resolve.agent_for_ref(db, {"item": uuid.uuid4().hex}) is None


# --------------------------------------------------------------------------- the existing install


def test_install_item_clones_the_cloud_row_with_its_tools_skills_model_and_plugin(db, monkeypatch):
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: "owner")
    monkeypatch.setattr(agent_quota, "require_agent_room", lambda db, workspace_id: None)
    tag = uuid.uuid4().hex[:10]
    bundle = _bundle(tag)
    held = _hold_everything(db, bundle)
    _outcome, row = _upsert(db, bundle)
    db.commit()
    ws = _workspace(db)
    db.commit()
    _row, cloud_skills, cloud_tools, cloud_plugins = _snapshot(db, row.id)

    answer = _client(db, ws).post(f"/api/marketplace/items/{row.id}/install")

    assert answer.status_code == 200, answer.text
    assert answer.json()["success"] is True
    db.expire_all()
    [made] = db.query(Agent).filter(Agent.cloned_from_id == row.id, Agent.workspace_id == ws).all()
    cloud = db.get(Agent, row.id)
    assert made.owner_type == "workspace" and made.version == cloud.version == local_version("1.0.0", revision_of(bundle))
    assert made.persona_id == held["persona"] and made.custom_persona_prompt == cloud.custom_persona_prompt
    assert made.model_config["model_id"] == bundle["agent"]["model"]["modelId"]
    assert sorted(_skill_ids(db, made.id)) == sorted(held["skills"])
    assert _apps(db, made.id) == APPS and _plugin_ids(db, made.id) == [held["plugin"]]
    runtime = db.execute(sa.text("SELECT app_name FROM agent_app_assignments WHERE agent_id = :id"), {"id": made.id})
    assert sorted(r[0] for r in runtime) == APPS
    installed = db.execute(
        sa.text("SELECT count(*) FROM workspace_models WHERE workspace_id = CAST(:ws AS uuid) AND model_id = :m"),
        {"ws": str(ws), "m": held["model"]},
    ).scalar()
    assert installed == 1
    # The install reads the Cloud row's rows and changes only its install count.
    assert _snapshot(db, row.id)[1:] == (cloud_skills, cloud_tools, cloud_plugins) and cloud.install_count == 1
