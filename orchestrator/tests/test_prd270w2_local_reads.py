"""PRD-270 W2 US-270-14a (§5.6, K17, K18) — the local reads the web shows Cloud items with.

``api/marketplace_cloud_items.py`` on a mini FastAPI app over Postgres, each test in ONE transaction rolled back at
its end (the item pass's commit is a savepoint, ``test_prd270w2_import_items.py``'s pattern). The Cloud rows are
what a Sync writes: a plugin as plugin_import.py commits one (its row, its scan row and its mapping), then the
corpus's agents, Playbook and package through the item pass (``import_items``). A member calls; Automatos Cloud is
never asked (W1's guarded GET and its client fail the test). Proven:

* ``/items`` lists exactly the mapped rows, by the keys the install routes take, and never a seeded row;
* a seeded agent, a plugin without a mapping, an unknown kind and a key not of its kind's shape have no disclosure
  (404); a Cloud agent's shows its prompt (its own, or its persona's), its model and whether this install has it,
  its apps and which need a connection, its skills and plugins (here or not); a Playbook's each step's helper,
  prompt and needsHelper; a package's members (here or not), connects and its agents' prompts; a plugin's two
  reports, its contents and the skills it made;
* ``/updates`` lists a Cloud agent whose only copy here is older, not one with a copy at the current version, and
  never another workspace's copy; a Cloud plugin this workspace enabled before its version moved, not one enabled
  after, and not another workspace's enablement;
* every route is a plain ``def``, and SaaS answers 404 before the caller is read.
"""
from __future__ import annotations

import copy
import inspect
import json
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import sqlalchemy as sa
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import api.marketplace_cloud_items as items_api
import core.models  # noqa: F401  (registers every table the foreign keys name)
from config import config
from core.auth.dependencies import RequestContext, UserContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from core.models.composio_cache import ComposioAppCache
from core.models.core import Agent, LLMModel, Skill, WorkflowTemplate
from core.models.marketplace_cloud import MarketplaceCloudItem
from core.models.marketplace_plugins import MarketplacePlugin, PluginSecurityScan
from core.models.personas import Persona
from services.marketplace_bundles.canonical import revision_of
from services.marketplace_bundles.import_items import import_items
from services.marketplace_bundles.items_feed import CatalogEntry, FetchedBundle
from services.marketplace_bundles.validate import validate_bundle
from services.marketplace_cloud import fetch

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "marketplace_bundles"
BASE = "/api/marketplace/cloud-sync"
OLD_VERSION = "0.9.0+000000000000"
PUBLISHER = "Test Publisher"


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = sa.create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as conn:
            for probe in (
                "SELECT agent_id, recipe_id, package_id, plugin_id, unresolved, synced_at FROM marketplace_cloud_items LIMIT 1",
                "SELECT tool_id FROM agent_tool_assignments LIMIT 1",
                "SELECT auth_schemes FROM composio_apps_cache LIMIT 1",
                "SELECT enabled_at FROM workspace_enabled_plugins LIMIT 1",
                "SELECT static_findings, llm_findings FROM plugin_security_scans LIMIT 1",
                "SELECT materialized_skill_ids FROM marketplace_plugins LIMIT 1",
                "SELECT system_prompt, scope FROM personas LIMIT 1",
            ):
                conn.execute(sa.text(probe))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the Cloud local reads tests need Postgres with the PRD-270 W2 schema: {exc}")
    yield eng
    eng.dispose()


@contextmanager
def _rolled_back(engine):
    with engine.connect() as conn:
        trans = conn.begin()
        session = Session(bind=conn, join_transaction_mode="create_savepoint")
        try:
            conn.execute(sa.text("SET LOCAL lock_timeout = '5s'"))
            # An agent with no model hint has its model looked up in a legacy catalogue table (marketplace_items,
            # cascade_installer.cascade_agent_dependencies) that only alembic creates: an install's schema has
            # it, the test schema (create_all) has none. Rolled back below (test_prd251w1_socials_package's way).
            conn.execute(sa.text(
                "CREATE TABLE IF NOT EXISTS marketplace_items (id serial PRIMARY KEY, type text, name text, metadata jsonb)"
            ))
            yield session
        finally:
            session.close()
            trans.rollback()


def _never(*_args, **_kwargs):
    raise AssertionError("a local read never asks Automatos Cloud")


def _no_caller():
    raise AssertionError("SaaS answers before the caller is read")


def _workspace(db) -> uuid.UUID:
    ws = uuid.uuid4()
    db.execute(sa.text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'prd270w2 local reads')"),
               {"id": str(ws)})
    return ws


@pytest.fixture
def env(engine, monkeypatch):
    """The router over one rolled-back transaction, a member of ``state.workspace`` calling, no network."""
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(fetch, "fetch_bytes", _never)
    monkeypatch.setattr(fetch, "_client", _never)
    with _rolled_back(engine) as session:
        state = SimpleNamespace(db=session, workspace=_workspace(session))
        user = UserContext(id="clerk-14a", clerk_user_id="clerk-14a", system_role="user",
                           raw_claims={"sub": "clerk-14a", "iat": int(time.time())})
        app = FastAPI()
        app.include_router(items_api.router)
        app.dependency_overrides[get_db] = lambda: session
        app.dependency_overrides[get_request_context_hybrid] = (
            lambda: RequestContext(workspace_id=state.workspace, user=user, auth_type="clerk"))
        state.client = TestClient(app)
        state.cloud = _cloud_rows(session, uuid.uuid4().hex[:10])
        yield state


# --------------------------------------------------------------------------- what a Sync writes


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def _made(name: str, tag: str, title: str, body: dict) -> dict:
    base = _fixture(name)
    made = {**base, "externalId": uuid.uuid4().hex, "name": f"{title} {tag}", name: {**base[name], **body}}
    validate_bundle(made)
    return made


def _held(db, tag: str) -> SimpleNamespace:
    """What this install holds before the Sync: two Composio apps, a model, a shared persona, a seeded agent."""
    key_app, oauth_app, new_app = (f"T270LR_{kind}_{tag.upper()}" for kind in ("KEY", "OAUTH", "NEW"))
    for app, schemes in ((key_app, ["API_KEY"]), (oauth_app, ["OAUTH2"])):
        db.add(ComposioAppCache(app_name=app, app_slug=app.lower(), display_name=app, auth_schemes=schemes))
    model = f"test-vendor/t270lr-{tag}"
    db.add(LLMModel(provider="test-vendor", serving_provider="openrouter", model_id=model,
                    display_name="Test model", context_window=1000, max_output_tokens=100))
    db.add(Persona(slug=f"t270lr-persona-{tag}", name=f"Test Planner {tag}", scope="global", workspace_id=None,
                   system_prompt="You plan the week in plain sentences."))
    seeded = Agent(name=f"Seeded Writer {tag}", agent_type="custom", owner_type="marketplace",
                   owner_id="marketplace", workspace_id=None, is_approved=True, slug=f"t270lr-seeded-{tag}",
                   version="1.0.0", custom_persona_prompt="The seeded prompt.", use_custom_persona=True)
    db.add(seeded)
    db.flush()
    return SimpleNamespace(apps=(key_app, oauth_app, new_app), model=model, seeded=seeded)


def _cloud_plugin(db, tag: str) -> SimpleNamespace:
    """A Cloud plugin as plugin_import.py commits one: its skill, its scan row, its row and its mapping."""
    slug = f"t270lr-kit-{tag}"
    skill = Skill(name=f"t270lr-summarise-{tag}", description="Test skill", skill_type="technical", is_active=True,
                  workspace_id=None, package_slug=slug)
    scan = PluginSecurityScan(
        plugin_slug=slug, plugin_version="1.2.0", static_scan_status="flagged",
        static_findings=[{"type": "file_write", "severity": "medium", "file": "commands/save-notes.md", "line": 12,
                          "pattern": "open\\(", "matched_text": "open(notes_path, 'a')",
                          "description": "Opens file in append mode"}],
        llm_scan_status="passed", llm_risk_score=5,
        llm_findings=[{"severity": "low", "description": "The command writes a notes file."}],
        llm_summary="No security issues found.", overall_verdict="safe", scanned_by="automatos-cloud",
    )
    db.add_all([skill, scan])
    db.flush()
    plugin = MarketplacePlugin(
        slug=slug, name=f"Test Research Kit {tag}", version="1.2.0", approval_status="approved", is_active=True,
        original_author=PUBLISHER, skills_count=2, commands_count=1, agents_count=0, hooks_count=0,
        security_scan_id=scan.id, security_status="safe", source_type="automatos-cloud",
        materialized_skill_ids=[skill.id],
    )
    db.add(plugin)
    db.flush()
    db.add(MarketplaceCloudItem(item_type="plugin", external_id=uuid.uuid4().hex, version="1.2.0",
                                revision="c" * 64, plugin_id=plugin.id, unresolved=[]))
    db.flush()
    return SimpleNamespace(row=plugin, skill=skill.name)


def _bundles(tag: str, held: SimpleNamespace, plugin: SimpleNamespace) -> SimpleNamespace:
    """The corpus's agent (its own prompt) and a persona agent, a Playbook on the first and a package of them."""
    helper = _made("agent", tag, "Test Cloud Helper", {
        "persona": None, "model": {"modelId": held.model, "provider": "openrouter"}, "apps": list(held.apps),
        "skills": [{"name": plugin.skill, "package": plugin.row.slug}, {"name": f"t270lr-missing-{tag}", "package": None}],
        "plugins": [plugin.row.slug, f"t270lr-gone-{tag}"],
        "prompt": {"custom": True, "text": "You summarise the sources you are given."},
    })
    planner = _made("agent", tag, "Test Cloud Planner", {
        "persona": f"t270lr-persona-{tag}", "model": {"modelId": f"test-vendor/t270lr-absent-{tag}", "provider": None},
        "apps": [], "skills": [], "plugins": [], "prompt": {"custom": False, "text": ""},
    })
    steps = copy.deepcopy(_fixture("playbook")["playbook"]["steps"][:3])
    digest = _made("playbook", tag, "Test Digest", {
        "agents": {"a1": {"item": helper["externalId"]}, "a2": {"name": f"Test Editor {tag}"}}, "steps": steps,
    })
    members = [
        {"type": "agent", "ref": {"item": helper["externalId"]}, "name": "The helper", "description": ""},
        {"type": "agent", "ref": {"slug": held.seeded.slug}, "name": "The writer", "description": ""},
        {"type": "playbook", "ref": {"item": digest["externalId"]}, "name": "The digest", "description": ""},
        {"type": "plugin", "ref": {"slug": plugin.row.slug}, "name": "The kit", "description": ""},
        {"type": "skill", "ref": {"name": plugin.skill, "package": plugin.row.slug}, "name": "Summarise",
         "description": ""},
        {"type": "llm", "ref": {"modelId": held.model}, "name": "The model", "description": ""},
        {"type": "tool", "ref": {"app": held.apps[0]}, "name": "The app", "description": ""},
    ]
    pack = _made("package", tag, "Test Pack", {"slug": f"t270lr-pack-{tag}", "members": members})
    return SimpleNamespace(helper=helper, planner=planner, digest=digest, pack=pack)


def _fetched(bundle: dict) -> FetchedBundle:
    entry = CatalogEntry(bundle["type"], bundle["externalId"], bundle["version"], revision_of(bundle), bundle["name"])
    return FetchedBundle(entry, bundle)


def _local_id(db, bundle: dict):
    mapping = db.query(MarketplaceCloudItem).filter(
        MarketplaceCloudItem.item_type == bundle["type"], MarketplaceCloudItem.external_id == bundle["externalId"],
    ).one()
    return mapping.agent_id or mapping.recipe_id or mapping.package_id


def _cloud_rows(db, tag: str) -> SimpleNamespace:
    held = _held(db, tag)
    plugin = _cloud_plugin(db, tag)
    bundles = _bundles(tag, held, plugin)
    written = import_items(db, [_fetched(b) for b in (bundles.helper, bundles.planner, bundles.digest, bundles.pack)])
    assert written.ok, written.error
    assert {kind: counts["new"] for kind, counts in written.view().items()} == {"agent": 2, "playbook": 1, "package": 1}
    return SimpleNamespace(
        held=held, plugin=plugin.row, skill=plugin.skill, bundles=bundles, tag=tag,
        helper=db.get(Agent, _local_id(db, bundles.helper)), planner=db.get(Agent, _local_id(db, bundles.planner)),
        digest=db.get(WorkflowTemplate, _local_id(db, bundles.digest)),
    )


def _get(state, path: str, status: int = 200):
    answer = state.client.get(BASE + path)
    assert answer.status_code == status, answer.text
    return answer.json()


# --------------------------------------------------------------------------- /items


def test_items_lists_exactly_the_mapped_rows_by_the_keys_the_installs_take(env):
    cloud, items = env.cloud, _get(env, "/items")
    assert set(items) == {"agents", "playbooks", "packages", "plugins"}
    mapped = env.db.query(MarketplaceCloudItem).all()
    assert {row["id"] for row in items["agents"]} == {m.agent_id for m in mapped if m.item_type == "agent"}
    assert {row["id"] for row in items["playbooks"]} == {m.recipe_id for m in mapped if m.item_type == "playbook"}
    assert {row["id"] for row in items["plugins"]} == {str(m.plugin_id) for m in mapped if m.item_type == "plugin"}
    assert len(items["packages"]) == len([m for m in mapped if m.item_type == "package"])
    helper = next(row for row in items["agents"] if row["id"] == cloud.helper.id)
    assert helper == {"id": cloud.helper.id, "externalId": cloud.bundles.helper["externalId"], "version": "1.0.0",
                      "revision": revision_of(cloud.bundles.helper), "name": cloud.helper.name}
    assert cloud.held.seeded.id not in {row["id"] for row in items["agents"]}
    assert f"t270lr-pack-{cloud.tag}" in {row["slug"] for row in items["packages"]}
    assert cloud.digest.id in {row["id"] for row in items["playbooks"]}


# --------------------------------------------------------------------------- the disclosure


def test_only_a_cloud_item_of_a_known_kind_has_a_disclosure(env):
    cloud = env.cloud
    loose = MarketplacePlugin(slug=f"t270lr-local-{cloud.tag}", name="A local kit", version="1.0.0",
                              approval_status="approved", is_active=True)
    env.db.add(loose)
    env.db.flush()
    for path in (f"/agent/{cloud.held.seeded.id}", f"/plugin/{loose.id}", f"/theme/{cloud.helper.id}",
                 "/agent/not-an-id", "/agent/99999999999", "/agent/-1", "/playbook/0", "/package/no-such-pack",
                 "/package/a%00b", "/package/" + "a" * 121, "/package/Not-A-Slug", "/plugin/not-a-uuid"):
        assert _get(env, f"/items{path}/disclosure", 404)["detail"] == items_api.NOT_A_CLOUD_ITEM, path


def test_a_cloud_agents_disclosure_shows_its_prompt_model_apps_skills_and_plugins(env):
    cloud = env.cloud
    key_app, oauth_app, new_app = cloud.held.apps
    answer = _get(env, f"/items/agent/{cloud.helper.id}/disclosure")
    assert {k: answer[k] for k in ("kind", "name", "version", "trust", "publisher")} == {
        "kind": "agent", "name": cloud.helper.name, "version": "1.0.0", "trust": "reviewed", "publisher": PUBLISHER}
    assert answer["prompt"] == {"source": "custom", "text": "You summarise the sources you are given.",
                                "personaName": None}
    assert answer["model"] == {"modelId": cloud.held.model, "provider": "openrouter", "here": True}
    assert answer["apps"] == [{"name": key_app, "needsConnection": False}, {"name": new_app, "needsConnection": True},
                              {"name": oauth_app, "needsConnection": True}]
    assert answer["skills"] == [{"name": cloud.skill, "here": True},
                                {"name": f"t270lr-missing-{cloud.tag}", "here": False}]
    assert answer["plugins"] == [{"slug": cloud.plugin.slug, "here": True},
                                 {"slug": f"t270lr-gone-{cloud.tag}", "here": False}]
    assert {"kind": "skill", "name": f"t270lr-missing-{cloud.tag}"} in answer["unresolved"]
    assert answer["syncedAt"]

    planner = _get(env, f"/items/agent/{cloud.planner.id}/disclosure")
    assert planner["prompt"] == {"source": "persona", "text": "You plan the week in plain sentences.",
                                 "personaName": f"Test Planner {cloud.tag}"}
    assert planner["model"] == {"modelId": f"test-vendor/t270lr-absent-{cloud.tag}", "provider": None, "here": False}
    assert planner["apps"] == [] and planner["skills"] == [] and planner["plugins"] == []


def test_a_cloud_playbooks_disclosure_shows_each_steps_helper_prompt_and_need(env):
    cloud = env.cloud
    answer = _get(env, f"/items/playbook/{cloud.digest.id}/disclosure")
    assert answer["kind"] == "playbook" and answer["publisher"] == PUBLISHER and answer["trust"] == "reviewed"
    assert answer["steps"] == [
        {"order": 1, "helper": cloud.helper.name, "needsHelper": False,
         "prompt": "Gather this week's notes about {{ topic }}."},
        {"order": 2, "helper": None, "needsHelper": False, "prompt": None},
        {"order": 3, "helper": f"Test Editor {cloud.tag}", "needsHelper": True, "prompt": "Edit the digest: {{ digest }}"},
    ]
    assert answer["agents"] == [cloud.helper.name] and answer["apps"] == ["SLACK"]
    assert answer["unresolved"] == [{"kind": "helper", "name": f"Test Editor {cloud.tag}"}]


def test_a_cloud_packages_disclosure_shows_its_members_connects_and_agents_prompts(env):
    cloud, slug = env.cloud, f"t270lr-pack-{env.cloud.tag}"
    answer = _get(env, f"/items/package/{slug}/disclosure")
    assert answer["kind"] == "package" and answer["publisher"] == "Automatos" and answer["name"].startswith("Test Pack")
    assert [(m["type"], m["here"]) for m in answer["members"]] == [
        ("agent", True), ("agent", True), ("playbook", True), ("plugin", True), ("skill", True), ("llm", True),
        ("tool", True)]
    assert answer["connects"] == [{"app": "SLACK", "optional": True, "note": "Optional: post the digest to Slack."}]
    assert answer["agents"] == [
        {"name": cloud.helper.name, "prompt": {"source": "custom", "text": "You summarise the sources you are given.",
                                               "personaName": None}},
        {"name": cloud.held.seeded.name, "prompt": {"source": "custom", "text": "The seeded prompt.",
                                                    "personaName": None}},
    ]
    assert answer["playbooks"] == [{"name": cloud.digest.name, "steps": 3}]

    cloud.held.seeded.is_approved = False  # a member no longer installable here
    env.db.flush()
    again = _get(env, f"/items/package/{slug}/disclosure")
    assert [m["here"] for m in again["members"]][:2] == [True, False]
    assert [agent["name"] for agent in again["agents"]] == [cloud.helper.name]


def test_a_cloud_plugins_disclosure_shows_both_reports_its_contents_and_skills(env):
    plugin = env.cloud.plugin
    answer = _get(env, f"/items/plugin/{plugin.id}/disclosure")
    assert {k: answer[k] for k in ("kind", "name", "version", "trust", "publisher", "updatesInPlace")} == {
        "kind": "plugin", "name": plugin.name, "version": "1.2.0", "trust": "reviewed", "publisher": PUBLISHER,
        "updatesInPlace": True}
    assert answer["scan"] == {"verdict": "safe", "status": "passed", "riskScore": 5,
                              "summary": "No security issues found.",
                              "findings": [{"severity": "low", "description": "The command writes a notes file."}]}
    assert answer["localCheck"] == {"status": "flagged", "findings": [
        {"type": "file_write", "severity": "medium", "file": "commands/save-notes.md", "line": 12,
         "description": "Opens file in append mode"}]}
    assert answer["contents"] == {"skills": 2, "commands": 1, "agents": 0, "hooks": 0}
    assert answer["skills"] == [env.cloud.skill]


# --------------------------------------------------------------------------- /updates


def _agent_copy(db, source: Agent, workspace, version: str) -> Agent:
    row = Agent(name=f"{source.name}-copy", agent_type="custom", owner_type="workspace", workspace_id=workspace,
                cloned_from_id=source.id, version=version)
    db.add(row)
    db.flush()
    return row


def _playbook_copy(db, source: WorkflowTemplate, workspace, version: str) -> WorkflowTemplate:
    row = WorkflowTemplate(
        template_id=f"t270lr-copy-{uuid.uuid4().hex[:12]}", name=f"{source.name} (Copy)", description="A copy",
        workspace_id=workspace, owner_type="workspace", owner_id=str(workspace), cloned_from_id=source.id,
        template_definition={"steps": []}, created_by="tester", version=version,
    )
    db.add(row)
    db.flush()
    return row


def test_updates_list_a_cloud_copy_only_when_no_copy_here_is_at_the_current_version(env):
    db, cloud, other = env.db, env.cloud, _workspace(env.db)
    old = _agent_copy(db, cloud.helper, env.workspace, OLD_VERSION)
    _agent_copy(db, cloud.planner, env.workspace, OLD_VERSION)
    _agent_copy(db, cloud.planner, env.workspace, cloud.planner.version)  # one copy current: no notice
    _agent_copy(db, cloud.held.seeded, env.workspace, OLD_VERSION)  # not a Cloud row
    theirs = _playbook_copy(db, cloud.digest, other, OLD_VERSION)  # another workspace's copy
    listed = _get(env, "/updates")["updates"]
    assert listed == [{
        "kind": "agent", "cloudId": cloud.helper.id, "name": cloud.helper.name,
        "installed": [{"id": old.id, "name": old.name, "version": OLD_VERSION}],
        "available": {"version": "1.0.0", "revision": revision_of(cloud.bundles.helper),
                      "syncedAt": listed[0]["available"]["syncedAt"]},
    }]
    assert listed[0]["available"]["syncedAt"]
    env.workspace = other
    assert [(u["kind"], u["cloudId"], [c["id"] for c in u["installed"]]) for u in _get(env, "/updates")["updates"]] == [
        ("playbook", cloud.digest.id, [theirs.id])]


def _enable(db, workspace, plugin_id, days_ago: int) -> None:
    db.execute(sa.text(
        "INSERT INTO workspace_enabled_plugins (workspace_id, plugin_id, enabled_at)"
        " VALUES (CAST(:ws AS uuid), CAST(:plugin AS uuid), now() - make_interval(days => :days))"
    ), {"ws": str(workspace), "plugin": str(plugin_id), "days": days_ago})


def test_updates_list_a_cloud_plugin_this_workspace_enabled_before_it_moved(env):
    db, cloud, other = env.db, env.cloud, _workspace(env.db)
    later = _cloud_plugin(db, uuid.uuid4().hex[:10]).row
    db.execute(sa.text("UPDATE marketplace_cloud_items SET synced_at = now() - interval '1 day'"
                       " WHERE plugin_id IN (CAST(:a AS uuid), CAST(:b AS uuid))"),
               {"a": str(cloud.plugin.id), "b": str(later.id)})
    loose = MarketplacePlugin(slug=f"t270lr-own-{cloud.tag}", name="A local kit", version="1.0.0",
                              approval_status="approved", is_active=True)
    db.add(loose)
    db.flush()
    _enable(db, env.workspace, cloud.plugin.id, 2)  # enabled, then the plugin moved
    _enable(db, env.workspace, later.id, 0)  # enabled after it moved
    _enable(db, env.workspace, loose.id, 5)  # not a Cloud plugin
    _enable(db, other, later.id, 3)  # another workspace's enablement
    listed = _get(env, "/updates")["updates"]
    assert [(u["kind"], u["cloudId"], u["name"], u["installed"]) for u in listed] == [
        ("plugin", str(cloud.plugin.id), cloud.plugin.name, [])]
    assert {k: listed[0]["available"][k] for k in ("version", "revision")} == {"version": "1.2.0", "revision": "c" * 64}
    env.workspace = other
    assert [u["cloudId"] for u in _get(env, "/updates")["updates"]] == [str(later.id)]


# --------------------------------------------------------------------------- the editions and the handlers


def test_saas_answers_404_for_every_read_before_the_caller_is_read(env, monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    env.client.app.dependency_overrides[get_request_context_hybrid] = _no_caller
    for path in ("/items", f"/items/agent/{env.cloud.helper.id}/disclosure", "/updates"):
        assert _get(env, path, 404)["detail"] == items_api.HOSTED_NO_CLOUD_ITEMS, path


def test_every_handler_is_a_plain_def():
    routes = [route for route in items_api.router.routes if isinstance(route, APIRoute)]
    assert {(tuple(route.methods), route.path) for route in routes} == {
        (("GET",), BASE + "/items"), (("GET",), BASE + "/items/{kind}/{key}/disclosure"), (("GET",), BASE + "/updates")}
    for route in routes:
        assert not inspect.iscoroutinefunction(route.endpoint), route.path
