"""P270W2-RVW-1 (PRD-270 §5.6, K13, K14, K15) — an item written while what it names hadn't arrived is resolved
again once it has, never frozen at its revision.

The Sync's passes (``sync_items.run_item_passes``: the plugin pass, then the item pass) run for real on
Postgres, a Sync after another, each test in ONE transaction rolled back at its end (the passes' commits are
savepoints, ``test_prd270w2_sync_routes.py``'s pattern). W1's theme pass is left out: the passes are handed
its outcome with the catalogue's item entries. Automatos Cloud is an ``httpx.MockTransport`` answering by URL
and recording each request, the object store a fake S3 client, and a model call fails the test
(``tests/helpers_prd270w2_plugins.py``). Proven:

* with AUTOMATOS_CLOUD_PLUGINS_PER_SYNC = 1 and two new plugins, an agent naming the one that waits gets that
  plugin's AgentAssignedPlugin row and the skill it makes on the next Sync, and its mapping no longer names
  them;
* a Playbook whose ``{item}`` agent's bundle couldn't be fetched finds that agent on the next Sync: its step's
  agent, its recommended agents, no helper needed;
* an agent left needing only a model this install lacks is unchanged on the next Sync, byte-equal, and no
  bundle of it is fetched; nor is one while the plugin it needs is still waiting;
* a Playbook is fetched again only once its helper's bundle is, is rewritten when its helper is named anew,
  and stands unchanged, byte-equal, when it resolves as it did or its own bundle can't be fetched again;
* an agent whose rewrite is refused (its name now held) stands as it was and counts unchanged.
"""
from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

import core.models  # noqa: F401  (registers every table the foreign keys name)
from config import config
from core.models.core import Agent, Skill, WorkflowTemplate, agent_skills
from core.models.marketplace_cloud import MarketplaceCloudItem
from core.models.marketplace_plugins import AgentAssignedPlugin, MarketplacePlugin
from services.marketplace_bundles import sync_items
from services.marketplace_bundles.canonical import canonical_bytes, local_version, revision_of
from services.marketplace_bundles.catalog_items import entry_of
from services.marketplace_bundles.exported import ExportedItem, ItemEntry
from services.marketplace_bundles.items_feed import Entries, parse_item_entry
from services.marketplace_bundles.validate import validate_bundle
from services.marketplace_cloud import feed, pull
from tests.helpers_prd270w2_plugins import (
    BUNDLES, Runner, bundle_of, cloud_settings, fake_bucket, files_of, package_url, refuse_models, zip_of,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "marketplace_bundles"
ZERO = {"new": 0, "updated": 0, "unchanged": 0, "refused": 0}


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = sa.create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as conn:
            for probe in (
                "SELECT agent_id, recipe_id, plugin_id, unresolved FROM marketplace_cloud_items LIMIT 1",
                "SELECT refused_count FROM marketplace_cloud_sync LIMIT 1",
                "SELECT security_scan_id, materialized_skill_ids FROM marketplace_plugins LIMIT 1",
                "SELECT scanned_by FROM plugin_security_scans LIMIT 1",
                "SELECT package_slug FROM skills LIMIT 1",
            ):
                conn.execute(sa.text(probe))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the re-resolve tests need Postgres with the PRD-270 W2 schema: {exc}")
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
def env(engine, monkeypatch):
    """One rolled-back transaction, the local edition syncing from a mocked Automatos Cloud, the store fake."""
    cloud = cloud_settings(monkeypatch)
    fake_bucket(monkeypatch)
    refuse_models(monkeypatch)
    with _rolled_back(engine) as session:
        yield SimpleNamespace(cloud=cloud, db=session, tag=uuid.uuid4().hex[:10])


# --------------------------------------------------------------------------- the catalogue and the Sync


def _fixture(name):
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def _listed(cloud, bundle):
    """``bundle`` served at its derived URL; its catalogue entry, as the passes read it."""
    validate_bundle(bundle)
    revision = revision_of(bundle)
    cloud.answer(BUNDLES + revision, canonical_bytes(bundle))
    identity = ItemEntry(bundle["type"], bundle["externalId"], bundle["version"], frozenset())
    return parse_item_entry(entry_of(ExportedItem(identity, bundle, revision)))


def _plugin(cloud, slug, name):
    """A plugin's bundle and package served; its entry."""
    bundle = bundle_of(slug, name=name)
    cloud.answer(package_url(slug, bundle["version"]), zip_of(files_of(slug, bundle["version"])))
    return _listed(cloud, bundle)


def _agent(tag, plugins=(), model=None):
    """The corpus's agent named for this test, using ``plugins`` and the skill each makes, and ``model``."""
    base = _fixture("agent")
    body = {**base["agent"], "persona": None, "model": {"modelId": model, "provider": None}, "apps": [],
            "skills": [{"name": f"{slug}-skill", "package": slug} for slug in plugins], "plugins": list(plugins)}
    return {**base, "externalId": uuid.uuid4().hex, "name": f"Test Re-resolve Helper {tag}", "agent": body}


def _playbook(tag, agent):
    """The corpus's Playbook, its one step on ``agent`` by ``{item}``."""
    base = _fixture("playbook")
    body = {**base["playbook"], "agents": {"a1": {"item": agent["externalId"]}}, "steps": base["playbook"]["steps"][:1]}
    return {**base, "externalId": uuid.uuid4().hex, "name": f"Test Re-resolve Digest {tag}", "playbook": body}


def _sync(env, listed):
    """One Sync's passes over the catalogue's item entries ``listed`` (W1's theme pass succeeded)."""
    catalog = feed.Catalog(None, (), (), 0, Entries(listed=tuple(listed)))
    outcome = pull.SyncOutcome(ok=True, counts=pull.SyncCounts(), catalog=catalog)
    answer = sync_items.run_item_passes(env.db, outcome, env.cloud.factory, Runner())
    env.db.expire_all()
    return answer


def _counts(**counted):
    return {**ZERO, **counted}


def _mapping(db, bundle):
    return (
        db.query(MarketplaceCloudItem)
        .filter(MarketplaceCloudItem.item_type == bundle["type"],
                MarketplaceCloudItem.external_id == bundle["externalId"])
        .one()
    )


def _plugin_ids(db, agent_id):
    rows = db.query(AgentAssignedPlugin).filter(AgentAssignedPlugin.agent_id == agent_id)
    return [row.plugin_id for row in rows.order_by(AgentAssignedPlugin.priority)]


def _skill_names(db, agent_id):
    return [row.name for row in db.query(Skill.name).join(agent_skills, agent_skills.c.skill_id == Skill.id)
            .filter(agent_skills.c.agent_id == agent_id).order_by(agent_skills.c.priority.desc())]


def _stored(db, agent_id):
    """The agent's row, its skill, tool and plugin rows and its mapping, as stored."""
    def rows(sql):
        return [dict(row) for row in db.execute(sa.text(sql), {"id": agent_id}).mappings().all()]
    return (
        rows("SELECT * FROM agents WHERE id = :id"),
        rows("SELECT * FROM agent_skills WHERE agent_id = :id ORDER BY skill_id"),
        rows("SELECT * FROM agent_tool_assignments WHERE agent_id = :id ORDER BY tool_id"),
        rows("SELECT * FROM agent_assigned_plugins WHERE agent_id = :id ORDER BY plugin_id"),
        rows("SELECT * FROM marketplace_cloud_items WHERE agent_id = :id"),
    )


def _stored_playbook(db, recipe_id):
    """The Playbook's row and its mapping, as stored."""
    def rows(sql):
        return [dict(row) for row in db.execute(sa.text(sql), {"id": recipe_id}).mappings().all()]
    return (
        rows("SELECT * FROM workflow_recipes WHERE id = :id"),
        rows("SELECT * FROM marketplace_cloud_items WHERE recipe_id = :id"),
    )


def _hold_name(db, name):
    """An operator's marketplace agent called ``name`` (in another case): a Cloud agent of that name is refused."""
    db.add(Agent(name=name.upper(), agent_type="custom", owner_type="marketplace", owner_id="marketplace",
                 workspace_id=None, is_approved=True, version="1.0.0"))
    db.commit()


# --------------------------------------------------------------------------- a plugin that waited


def test_an_agent_whose_plugin_waited_gets_it_and_its_skill_on_the_next_sync(env, monkeypatch):
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_PLUGINS_PER_SYNC", 1)
    first_slug, waiting_slug = f"t270rr-a-{env.tag}", f"t270rr-b-{env.tag}"
    agent = _agent(env.tag, plugins=(waiting_slug,))
    listed = [_plugin(env.cloud, first_slug, f"Test Kit A {env.tag}"),
              _plugin(env.cloud, waiting_slug, f"Test Kit B {env.tag}"), _listed(env.cloud, agent)]

    one = _sync(env, listed)

    assert one["plugins"] == {"waiting": 1} and one["items"]["agent"] == _counts(new=1)
    assert {"kind": "plugin", "name": waiting_slug} in _mapping(env.db, agent).unresolved

    two = _sync(env, listed)

    assert two["plugins"] == {"waiting": 0} and two["items"]["plugin"] == _counts(new=1, unchanged=1)
    assert two["items"]["agent"] == _counts(updated=1)
    mapping = _mapping(env.db, agent)
    plugin = env.db.query(MarketplacePlugin).filter(MarketplacePlugin.slug == waiting_slug).one()
    assert _plugin_ids(env.db, mapping.agent_id) == [plugin.id]
    assert _skill_names(env.db, mapping.agent_id) == [f"{waiting_slug}-skill"]
    assert mapping.unresolved == [] and mapping.revision == revision_of(agent)
    row = env.db.get(Agent, mapping.agent_id)
    assert row.version == local_version(agent["version"], revision_of(agent))  # still the bundle it was


# --------------------------------------------------------------------------- a helper whose bundle failed


def test_a_playbook_whose_helper_failed_to_fetch_finds_it_on_the_next_sync(env):
    agent = _agent(env.tag)
    playbook = _playbook(env.tag, agent)
    listed = [_listed(env.cloud, agent), _listed(env.cloud, playbook)]
    agent_url = BUNDLES + revision_of(agent)
    served = env.cloud.answers.pop(agent_url)  # Automatos Cloud fails the agent's bundle on the first Sync

    one = _sync(env, listed)

    assert one["items"]["agent"] == _counts(refused=1) and one["items"]["playbook"] == _counts(new=1)
    row = env.db.get(WorkflowTemplate, _mapping(env.db, playbook).recipe_id)
    assert row.steps[0]["needs_helper"] == "a helper" and row.recommended_agents == []

    env.cloud.answer(agent_url, served)
    two = _sync(env, listed)

    assert two["items"]["agent"] == _counts(new=1) and two["items"]["playbook"] == _counts(updated=1)
    cloud_agent = env.db.get(Agent, _mapping(env.db, agent).agent_id)
    row = env.db.get(WorkflowTemplate, _mapping(env.db, playbook).recipe_id)
    step = row.steps[0]
    assert (step["agent_id"], step["agent_name"]) == (cloud_agent.id, agent["name"]) and "needs_helper" not in step
    assert row.recommended_agents == [agent["name"]] and _mapping(env.db, playbook).unresolved == []
    assert row.version == local_version(playbook["version"], revision_of(playbook))


# --------------------------------------------------------------------------- what can't arrive by a Sync


def test_an_agent_left_needing_only_a_model_is_unchanged_and_never_fetched_again(env, monkeypatch):
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_PLUGINS_PER_SYNC", 0)
    slug, model = f"t270rr-m-{env.tag}", f"test-vendor/t270rr-absent-{env.tag}"
    agent = _agent(env.tag, plugins=(slug,), model=model)
    model_only = _agent(f"{env.tag} only", model=model)
    listed = [_plugin(env.cloud, slug, f"Test Kit M {env.tag}"), _listed(env.cloud, agent),
              _listed(env.cloud, model_only)]
    assert _sync(env, listed)["items"]["agent"] == _counts(new=2)
    env.cloud.requests = []

    waiting = _sync(env, listed)  # the plugin still waits: nothing it needs has arrived

    assert waiting["items"]["agent"] == _counts(unchanged=2) and env.cloud.urls() == []
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_PLUGINS_PER_SYNC", 1)
    arrived = _sync(env, listed)

    assert arrived["items"]["agent"] == _counts(updated=1, unchanged=1)
    assert BUNDLES + revision_of(model_only) not in env.cloud.urls()
    assert _mapping(env.db, agent).unresolved == [{"kind": "model", "name": model}]
    ids = [_mapping(env.db, bundle).agent_id for bundle in (agent, model_only)]
    before = [_stored(env.db, agent_id) for agent_id in ids]
    env.cloud.requests = []

    again = _sync(env, listed)

    assert again["items"]["agent"] == _counts(unchanged=2) and again["items"]["plugin"] == _counts(unchanged=1)
    assert env.cloud.urls() == [] and [_stored(env.db, agent_id) for agent_id in ids] == before


# --------------------------------------------------------------------------- what stands as it was


def test_a_playbook_is_fetched_again_only_once_its_helper_is_and_stands_when_nothing_moves(env):
    agent = _agent(env.tag)
    playbook = _playbook(env.tag, agent)
    listed = [_listed(env.cloud, agent), _listed(env.cloud, playbook)]
    agent_url, playbook_url = BUNDLES + revision_of(agent), BUNDLES + revision_of(playbook)
    served = env.cloud.answers.pop(agent_url)  # the agent's bundle fails: the Playbook needs 'a helper'
    _sync(env, listed)
    env.cloud.requests = []

    still = _sync(env, listed)  # it fails again: nothing the Playbook awaits has arrived

    assert still["items"]["playbook"] == _counts(unchanged=1) and playbook_url not in env.cloud.urls()
    _hold_name(env.db, agent["name"])  # the agent will be refused, its bundle fetched
    env.cloud.answer(agent_url, served)

    named = _sync(env, listed)

    assert named["items"]["agent"] == _counts(refused=1) and named["items"]["playbook"] == _counts(updated=1)
    mapping = _mapping(env.db, playbook)
    assert mapping.unresolved == [{"kind": "helper", "name": agent["name"], "item": agent["externalId"]}]
    before = _stored_playbook(env.db, mapping.recipe_id)
    env.cloud.requests = []

    again = _sync(env, listed)  # fetched again, it resolves as it did

    assert again["items"]["playbook"] == _counts(unchanged=1) and playbook_url in env.cloud.urls()
    assert _stored_playbook(env.db, mapping.recipe_id) == before
    env.cloud.answers.pop(playbook_url)

    kept = _sync(env, listed)  # its own bundle fails: it stands

    assert kept["items"]["playbook"] == _counts(unchanged=1) and _stored_playbook(env.db, mapping.recipe_id) == before


def test_an_agent_whose_rewrite_is_refused_stands_as_it_was_and_counts_unchanged(env, monkeypatch):
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_PLUGINS_PER_SYNC", 0)
    slug = f"t270rr-r-{env.tag}"
    agent = _agent(env.tag, plugins=(slug,))
    listed = [_plugin(env.cloud, slug, f"Test Kit R {env.tag}"), _listed(env.cloud, agent)]
    _sync(env, listed)  # the plugin waits
    agent_id = _mapping(env.db, agent).agent_id
    _hold_name(env.db, agent["name"])
    before = _stored(env.db, agent_id)
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_PLUGINS_PER_SYNC", 1)
    env.cloud.requests = []

    arrived = _sync(env, listed)

    assert arrived["items"]["plugin"] == _counts(new=1) and arrived["items"]["agent"] == _counts(unchanged=1)
    assert BUNDLES + revision_of(agent) in env.cloud.urls()  # resolved again; its rewrite refused
    assert _stored(env.db, agent_id) == before
