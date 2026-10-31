"""PRD-270 W2 US-270-13c (§5.6, K14) — locally: Cloud Playbooks and packages, and the item pass in one transaction.

``import_items.import_items`` (with ``import_playbooks.upsert_playbook`` and ``import_packages.upsert_package``)
on Postgres, each test in one transaction rolled back at its end (``test_prd270w2_import_agents.py``'s
pattern: the pass and the installs commit, so the session joins the transaction through a savepoint). The
bundles are the corpus's (``tests/fixtures/marketplace_bundles/``) with this test's own names, validated
before use. Proven:

* the pass writes agents, then Playbooks, then packages, whatever order it is handed them, and answers its
  counts by type; a Playbook step naming a Cloud agent of the same pass gets that agent's local id and name,
  one on a seeded agent the seeded agent's, one named only ``needs_helper`` (recorded unresolved); the row is
  a public, approved marketplace Playbook at ``cloud-<externalId>``, scheduled by hand;
* ``install_recipe_from_marketplace`` on it clones it with its agent (the existing cascade), the copy's step
  on the workspace's clone of the Cloud agent;
* a package whose members resolve is stored with this install's references and installs through
  ``install_package`` with no PackageInstallError; one with any member this install lacks, or a slug a row
  without a mapping holds, is refused and writes nothing; so is a Playbook whose template id another row holds;
* the same bundles again are unchanged; a new revision moves the same Playbook and package rows and their
  mappings, and keeps a local operator's featuring and showcase;
* a database error (or anything unforeseen) mid-pass leaves no row of the pass, records only its error in
  the ``marketplace_cloud_sync`` row, and the theme pass's rows stand.
"""
from __future__ import annotations

import asyncio
import copy
import json
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

import api.workflow_recipes as workflow_recipes
import core.models  # noqa: F401  (registers every table the foreign keys name)
import services.agent_quota as agent_quota
from core.auth.dependencies import RequestContext, UserContext
from core.models.core import Agent, LLMModel, Skill, WorkflowTemplate
from core.models.marketplace_cloud import MarketplaceCloudItem
from core.models.marketplace_packages import MarketplacePackage
from core.models.marketplace_plugins import MarketplacePlugin
from core.models.themes import ThemePackage
from services import package_installer
from services.marketplace_bundles import import_items as items_mod
from services.marketplace_bundles.canonical import local_version, revision_of
from services.marketplace_bundles.import_items import import_items
from services.marketplace_bundles.import_playbooks import helper_name, validation_problem
from services.marketplace_bundles.items_feed import CatalogEntry, FetchedBundle
from services.marketplace_bundles.validate import validate_bundle
from services.marketplace_cloud import pull
from services.marketplace_cloud.feed import CheckedItem

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "marketplace_bundles"
PUBLISHER = "Test Publisher"
ZERO = {"new": 0, "updated": 0, "unchanged": 0, "refused": 0}


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = sa.create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as conn:
            for probe in (
                "SELECT recipe_id, package_id, unresolved FROM marketplace_cloud_items LIMIT 1",
                "SELECT last_error FROM marketplace_cloud_sync LIMIT 1",
                "SELECT template_id, recommended_agents FROM workflow_recipes LIMIT 1",
                "SELECT slug, members, setup_manifest FROM marketplace_packages LIMIT 1",
                "SELECT source FROM theme_packages LIMIT 1",
                "SELECT plugin_id FROM workspace_enabled_plugins LIMIT 1",
            ):
                conn.execute(sa.text(probe))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the Cloud item pass tests need Postgres with the PRD-270 W2 schema: {exc}")
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


@pytest.fixture
def room(monkeypatch):
    """Every workspace has room for one more agent (the plan's limit isn't this test's)."""
    monkeypatch.setattr(agent_quota, "require_agent_room", lambda db, workspace_id: None)


# --------------------------------------------------------------------------- bundles and rows


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def _agent_bundle(tag: str) -> dict:
    """The corpus's agent, named for this test, with nothing to resolve (no persona, model, skill, app or plugin)."""
    base = _fixture("agent")
    body = {**base["agent"], "persona": None, "model": {"modelId": None, "provider": None},
            "skills": [], "apps": [], "plugins": []}
    made = {**base, "externalId": uuid.uuid4().hex, "name": f"Test Cloud Helper {tag}", "agent": body}
    validate_bundle(made)
    return made


def _playbook_bundle(tag: str, agents: dict, steps=None) -> dict:
    """The corpus's Playbook, named for this test, its agents ``agents`` (its steps the corpus's unless given)."""
    base = _fixture("playbook")
    body = {**base["playbook"], "agents": agents, "steps": base["playbook"]["steps"] if steps is None else steps}
    made = {**base, "externalId": uuid.uuid4().hex, "name": f"Test Digest {tag}", "playbook": body}
    validate_bundle(made)
    return made


def _one_step(tag: str, agents: dict) -> dict:
    """A Playbook of one agent step, on a1."""
    step = {key: value for key, value in _fixture("playbook")["playbook"]["steps"][0].items()}
    return _playbook_bundle(tag, agents, steps=[step])


def _package_bundle(tag: str, members: list) -> dict:
    base = _fixture("package")
    body = {**base["package"], "slug": f"t270ii-pack-{tag}", "members": members}
    made = {**base, "externalId": uuid.uuid4().hex, "name": f"Test Pack {tag}", "package": body}
    validate_bundle(made)
    return made


def _member(kind: str, ref: dict, name: str = "A member") -> dict:
    return {"type": kind, "ref": ref, "name": name, "description": ""}


def _fetched(bundle: dict) -> FetchedBundle:
    entry = CatalogEntry(bundle["type"], bundle["externalId"], bundle["version"], revision_of(bundle), bundle["name"])
    return FetchedBundle(entry, bundle)


def _pass(db, *bundles):
    return import_items(db, [_fetched(bundle) for bundle in bundles])


def _mapping(db, bundle: dict):
    return (
        db.query(MarketplaceCloudItem)
        .filter(MarketplaceCloudItem.item_type == bundle["type"],
                MarketplaceCloudItem.external_id == bundle["externalId"])
        .one_or_none()
    )


def _cloud_row(db, bundle: dict):
    mapping = _mapping(db, bundle)
    model, key = {"agent": (Agent, "agent_id"), "playbook": (WorkflowTemplate, "recipe_id"),
                  "package": (MarketplacePackage, "package_id")}[bundle["type"]]
    return db.get(model, getattr(mapping, key))


def _playbook_id(db, bundle: dict) -> int:
    return _mapping(db, bundle).recipe_id


def _seeded_agent(db, tag: str) -> Agent:
    row = Agent(name=f"Seeded Writer {tag}", agent_type="custom", owner_type="marketplace", owner_id="marketplace",
                workspace_id=None, is_approved=True, slug=f"t270ii-seeded-{tag}", version="1.0.0")
    db.add(row)
    db.flush()
    return row


def _workspace(db) -> uuid.UUID:
    ws = uuid.uuid4()
    db.execute(sa.text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'prd270w2 import items')"),
               {"id": str(ws)})
    return ws


def _view(agent=ZERO, playbook=ZERO, package=ZERO):
    return {"agent": dict(agent), "playbook": dict(playbook), "package": dict(package)}


def _new(count=1):
    return {**ZERO, "new": count}


# --------------------------------------------------------------------------- Playbooks


def test_a_playbook_step_takes_a_cloud_agent_of_the_pass_a_seeded_agent_or_needs_a_helper(db):
    tag = uuid.uuid4().hex[:10]
    seeded = _seeded_agent(db, tag)
    agent = _agent_bundle(tag)
    editor = f"Test Editor {tag}"
    playbook = _playbook_bundle(tag, {"a1": {"item": agent["externalId"]}, "a2": {"slug": seeded.slug},
                                      "a3": {"name": editor}})

    result = _pass(db, playbook, agent)  # handed the Playbook first: the pass writes the agent before it

    assert result.ok and result.view() == _view(agent=_new(), playbook=_new())
    cloud_agent, row = _cloud_row(db, agent), _cloud_row(db, playbook)
    gather, render, edit, review = row.steps
    assert (gather["agent_id"], gather["agent_name"]) == (cloud_agent.id, agent["name"])
    assert "agent_id" not in render and "agent" not in render and render["type"] == "generate_document"
    assert (edit["agent_id"], edit["agent_name"]) == (seeded.id, seeded.name)
    assert (review["agent_id"], review["agent_name"], review["needs_helper"]) == (None, editor, editor)
    assert all("agent" not in step for step in row.steps)
    assert gather["prompt_template"] == playbook["playbook"]["steps"][0]["prompt_template"]
    assert row.recommended_agents == [agent["name"], seeded.name]
    assert (row.template_id, row.owner_type, row.owner_id, row.workspace_id) == (
        f"cloud-{playbook['externalId']}", "marketplace", "marketplace", None)
    assert (row.is_public, row.is_approved, row.is_featured, row.is_system) == (True, True, False, False)
    assert row.schedule_config == {"type": "manual"} and row.required_tools == ["SLACK"]
    assert row.template_definition == {"steps": [], "agents": [], "config": {}, "variables": []}
    assert row.version == local_version("1.0.0", revision_of(playbook)) and row.created_by == PUBLISHER
    assert (row.inputs, row.outputs) == (playbook["playbook"]["inputs"], playbook["playbook"]["outputs"])
    assert row.execution_config == playbook["playbook"]["execution"]
    assert (row.marketplace_category, row.marketplace_icon, row.tags) == ("productivity", "📰", ["digest", "test"])
    mapping = _mapping(db, playbook)
    assert (mapping.recipe_id, mapping.version, mapping.revision) == (row.id, "1.0.0", revision_of(playbook))
    assert mapping.unresolved == [{"kind": "helper", "name": editor}]


def test_install_recipe_from_marketplace_clones_the_playbook_with_its_cloud_agent(db, room):
    tag = uuid.uuid4().hex[:10]
    agent = _agent_bundle(tag)
    playbook = _one_step(tag, {"a1": {"item": agent["externalId"]}})
    assert _pass(db, agent, playbook).ok
    row, cloud_agent = _cloud_row(db, playbook), _cloud_row(db, agent)
    ws = _workspace(db)
    db.commit()
    ctx = RequestContext(workspace_id=ws, auth_type="clerk",
                         user=UserContext(id="t270ii-member", clerk_user_id="clerk-t270ii-member"))

    answer = asyncio.run(workflow_recipes.install_recipe_from_marketplace(recipe_id=row.id, ctx=ctx, db=db))

    assert answer["success"] is True
    db.expire_all()
    [made] = db.query(WorkflowTemplate).filter(WorkflowTemplate.cloned_from_id == row.id,
                                               WorkflowTemplate.workspace_id == ws).all()
    [helper] = db.query(Agent).filter(Agent.cloned_from_id == cloud_agent.id, Agent.workspace_id == ws).all()
    assert made.owner_type == "workspace" and made.version == db.get(WorkflowTemplate, row.id).version
    assert made.steps[0]["agent_id"] == helper.id and "needs_helper" not in made.steps[0]
    # The Cloud row's own steps still name the marketplace agent.
    assert db.get(WorkflowTemplate, row.id).steps[0]["agent_id"] == cloud_agent.id


def test_a_template_id_another_row_holds_refuses_a_new_playbook(db):
    tag = uuid.uuid4().hex[:10]
    playbook = _one_step(tag, {"a1": {"name": "Anyone"}})
    db.add(WorkflowTemplate(template_id=f"cloud-{playbook['externalId']}", name="Someone's", description="Theirs.",
                            owner_type="marketplace", owner_id="marketplace", template_definition={},
                            created_by="someone"))
    db.flush()

    assert _pass(db, playbook).view() == _view(playbook={**ZERO, "refused": 1})
    assert _mapping(db, playbook) is None


def test_the_models_validators_say_what_is_wrong_and_a_helper_is_named_by_its_ref():
    fields = {"steps": [{"step_id": "s", "order": 1, "agent_id": None}], "execution_config": {},
              "schedule_config": {"type": "manual"}}
    assert validation_problem(fields) == "Step 0 missing required field: prompt_template"
    assert validation_problem({**fields, "steps": [], "execution_config": {"mode": "sideways"}}) is not None
    assert validation_problem({**fields, "steps": []}) is None
    names = {"0" * 32: "Test Helper"}
    assert helper_name({"name": "Ed"}, names) == "Ed" and helper_name({"slug": "writer"}, names) == "writer"
    assert helper_name({"item": "0" * 32}, names) == "Test Helper"
    assert helper_name({"item": "1" * 32}, names) == "a helper"


# --------------------------------------------------------------------------- packages


def _held(db, tag: str) -> dict:
    """A global skill, an approved plugin and a model, as this install holds them."""
    skill = Skill(name=f"t270ii-cite-{tag}", description="Test skill", skill_type="technical", is_active=True,
                  workspace_id=None, package_slug=None)
    plugin = MarketplacePlugin(slug=f"t270ii-kit-{tag}", name="Test kit", version="1.0.0",
                               approval_status="approved", is_active=True)
    model = LLMModel(provider="test-vendor", serving_provider="openrouter", model_id=f"test-vendor/t270ii-{tag}",
                     display_name="Test model", context_window=1000, max_output_tokens=100)
    db.add_all([skill, plugin, model])
    db.flush()
    return {"skill": skill, "plugin": plugin, "model": model}


def test_a_package_whose_members_resolve_installs_through_install_package(db, room):
    tag = uuid.uuid4().hex[:10]
    held = _held(db, tag)
    agent = _agent_bundle(tag)
    playbook = _one_step(tag, {"a1": {"item": agent["externalId"]}})
    package = _package_bundle(tag, [
        _member("agent", {"item": agent["externalId"]}), _member("playbook", {"item": playbook["externalId"]}),
        _member("skill", {"name": held["skill"].name}), _member("plugin", {"slug": held["plugin"].slug}),
        _member("llm", {"modelId": held["model"].model_id}), _member("tool", {"app": "SLACK"}, "Slack"),
    ])

    result = _pass(db, package, playbook, agent)

    assert result.view() == _view(agent=_new(), playbook=_new(), package=_new())
    row = _cloud_row(db, package)
    assert [member["ref"] for member in row.members] == [
        _cloud_row(db, agent).id, f"cloud-{playbook['externalId']}", held["skill"].id, str(held["plugin"].id),
        held["model"].model_id, "SLACK",
    ]
    assert [member["name"] for member in row.members][-1] == "Slack"
    assert (row.slug, row.name, row.showcase) == (package["package"]["slug"], package["name"], False)
    assert (row.vertical_tags, row.matching) == (package["package"]["verticalTags"], package["package"]["matching"])
    assert row.setup_manifest == package["package"]["setup"] and _mapping(db, package).unresolved == []
    ws = _workspace(db)
    db.commit()

    manifest = asyncio.run(package_installer.install_package(db, ws, row.slug))

    registered = {(registration.type, registration.ref, registration.status) for registration in manifest.registrations}
    # Each member's own registration (a cascade may add its dependencies' beside them).
    assert ("skill", str(held["skill"].id), "installed") in registered
    assert ("plugin", str(held["plugin"].id), "installed") in registered
    assert ("llm", held["model"].model_id, "installed") in registered and ("tool", "SLACK", "installed") in registered
    [helper] = db.query(Agent).filter(Agent.cloned_from_id == _cloud_row(db, agent).id, Agent.workspace_id == ws).all()
    [copied] = db.query(WorkflowTemplate).filter(WorkflowTemplate.cloned_from_id == _playbook_id(db, playbook),
                                                 WorkflowTemplate.workspace_id == ws).all()
    assert ("agent", str(helper.id), "cloned") in registered and ("playbook", str(copied.id), "cloned") in registered
    assert copied.steps[0]["agent_id"] == helper.id


@pytest.mark.parametrize("member", [
    _member("agent", {"slug": "t270ii-no-such-agent"}),
    _member("agent", {"item": "f" * 32}),
    _member("playbook", {"templateId": "t270ii-no-such-playbook"}),
    _member("skill", {"name": "t270ii-no-such-skill"}),
    _member("plugin", {"slug": "t270ii-no-such-plugin"}),
    _member("llm", {"modelId": "test-vendor/t270ii-no-such-model"}),
], ids=["agent-slug", "agent-item", "playbook", "skill", "plugin", "llm"])
def test_a_package_with_a_member_this_install_lacks_is_refused(db, member):
    tag = uuid.uuid4().hex[:10]
    package = _package_bundle(tag, [_member("tool", {"app": "SLACK"}), member])

    assert _pass(db, package).view() == _view(package={**ZERO, "refused": 1})
    assert _mapping(db, package) is None
    assert db.query(MarketplacePackage).filter(MarketplacePackage.slug == package["package"]["slug"]).count() == 0


def test_a_package_slug_held_without_a_mapping_refuses_it_and_leaves_that_row(db):
    tag = uuid.uuid4().hex[:10]
    package = _package_bundle(tag, [_member("tool", {"app": "SLACK"})])
    own = MarketplacePackage(slug=package["package"]["slug"], name="The operator's own", members=[], showcase=True)
    db.add(own)
    db.commit()

    assert _pass(db, package).view() == _view(package={**ZERO, "refused": 1})
    db.expire_all()
    assert (own.name, own.members, own.showcase) == ("The operator's own", [], True) and _mapping(db, package) is None


# --------------------------------------------------------------------------- the same pass again, and an update


def test_the_same_bundles_are_unchanged_and_a_new_revision_moves_the_same_rows(db):
    tag = uuid.uuid4().hex[:10]
    agent = _agent_bundle(tag)
    playbook = _one_step(tag, {"a1": {"item": agent["externalId"]}})
    package = _package_bundle(tag, [_member("playbook", {"item": playbook["externalId"]})])
    _pass(db, agent, playbook, package)
    row, pack = _cloud_row(db, playbook), _cloud_row(db, package)
    row.is_featured, pack.showcase = True, True  # a local operator's choices
    db.commit()

    unchanged = {**ZERO, "unchanged": 1}
    assert _pass(db, agent, playbook, package).view() == _view(unchanged, unchanged, unchanged)

    newer = copy.deepcopy(playbook)
    newer.update(version="1.1.0", description="A newer digest.")
    newer["playbook"]["steps"][0]["prompt_template"] = "Gather the notes again."
    newer_pack = copy.deepcopy(package)
    newer_pack.update(version="1.1.0", name=f"Test Pack newer {tag}")
    assert _pass(db, agent, newer, newer_pack).view() == _view(unchanged, {**ZERO, "updated": 1}, {**ZERO, "updated": 1})
    db.expire_all()
    assert _cloud_row(db, newer).id == row.id and _cloud_row(db, newer_pack).id == pack.id
    assert row.steps[0]["prompt_template"] == "Gather the notes again." and row.description == "A newer digest."
    assert row.version == local_version("1.1.0", revision_of(newer)) and row.is_featured is True
    assert (pack.name, pack.showcase) == (newer_pack["name"], True)
    assert _mapping(db, newer).revision == revision_of(newer) and _mapping(db, newer_pack).version == "1.1.0"


# --------------------------------------------------------------------------- one transaction


def _theme_pass(db, tag: str, now: datetime) -> CheckedItem:
    """What W1's theme pass committed before the item pass: a Cloud theme row and the Sync's success."""
    item = CheckedItem(f"test/t270ii-{tag}", "1.0.0", uuid.uuid4().hex * 2, {}, {}, "test-signature")
    assert pull.write_item(db, item, now)
    sync = pull.sync_row(db)
    sync.last_attempt_at = sync.last_success_at = now
    sync.last_error, sync.new_count = None, 1
    db.commit()
    return item


def _broken_package_writer(db, bundle, revision):
    db.execute(sa.text("SELECT 1 / 0"))  # the database refuses the statement mid-pass


def _crashing_package_writer(db, bundle, revision):
    raise RuntimeError("something unforeseen")


@pytest.mark.parametrize("writer, error", [
    (_broken_package_writer, items_mod.NOT_SAVED), (_crashing_package_writer, items_mod.UNEXPECTED),
], ids=["database-error", "unforeseen"])
def test_a_failure_mid_pass_leaves_no_row_records_the_error_and_the_theme_pass_stands(db, monkeypatch, writer, error):
    tag, now = uuid.uuid4().hex[:10], datetime.now(timezone.utc)
    theme = _theme_pass(db, tag, now)
    agent = _agent_bundle(tag)
    playbook = _one_step(tag, {"a1": {"item": agent["externalId"]}})
    package = _package_bundle(tag, [_member("tool", {"app": "SLACK"})])
    monkeypatch.setattr(items_mod, "upsert_package", writer)

    result = _pass(db, agent, playbook, package)

    refused = {**ZERO, "refused": 1}
    assert result.error == error and not result.ok and result.view() == _view(refused, refused, refused)
    db.expire_all()
    assert [_mapping(db, bundle) for bundle in (agent, playbook, package)] == [None, None, None]
    assert db.query(Agent).filter(Agent.name == agent["name"]).count() == 0
    assert db.query(WorkflowTemplate).filter(WorkflowTemplate.template_id == f"cloud-{playbook['externalId']}").count() == 0
    sync = pull.sync_row(db)
    assert (sync.last_error, sync.new_count) == (error, 1)
    assert sync.last_success_at is not None and sync.last_attempt_at == sync.last_success_at
    assert db.query(ThemePackage).filter(ThemePackage.theme_id == theme.theme_id, ThemePackage.source == "cloud").count() == 1
