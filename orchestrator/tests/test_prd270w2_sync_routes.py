"""PRD-270 W2 US-270-13d (§5.1, §5.5, K16) — Sync with Automatos Cloud runs every pass.

W1's ``POST /api/marketplace/cloud-sync`` and ``GET .../status`` (``api/marketplace_cloud_sync.py``) on a mini
FastAPI app. The Sync runs for real on Postgres, each test in ONE transaction rolled back at its end (the passes'
commits are savepoints, ``test_prd270w2_import_items.py``'s pattern), so nothing it writes outlives the test.
Automatos Cloud is an ``httpx.MockTransport`` answering by URL and recording each request
(``tests/helpers_prd270w2_plugins.py``); its catalogue lists a signed theme (W1's test keys and package builder,
``tests/test_prd270w1_cloud_pull.py``), a plugin, an agent using the plugin and its skill, a Playbook on that agent
and a package of all three. The object stores are fake S3 clients; the LLM scan fails any test that reaches it.
Proven:

* an owner's Sync writes all five, runs the theme pass, the passes and the auto-update in that order (the plugin
  first: the agent finds its plugin and skill), holds W1's lock while the passes fetch, and answers each type's
  counts, the whole Sync's counts (W1's row the same) and the status's Cloud rows per type;
* a second Sync is all unchanged and fetches no bundle;
* a failing item pass keeps the theme and the plugin and records the error; a failing plugin pass records its
  error and runs no item pass; a plugin past the per-Sync limit waits, and the package needing it is refused;
  counts that can't be saved are recorded as the row's error;
* the passes run nothing with the URL empty, on SaaS, after a refused theme pass or while another Sync holds
  W1's lock; with the URL empty the Sync is W1's 409 and requests nothing;
* a member who isn't an owner or admin gets 403 and SaaS 404, before anything runs.
"""
from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import sqlalchemy as sa
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import api.marketplace_cloud_sync as sync_api
import core.auth.workspace_permission as permission_mod
import core.models  # noqa: F401  (registers every table the foreign keys name)
from config import config
from core.auth.dependencies import RequestContext, UserContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from core.models.core import Agent, WorkflowTemplate
from core.models.marketplace_cloud import MarketplaceCloudItem
from core.models.marketplace_packages import MarketplacePackage
from core.models.marketplace_plugins import MarketplacePlugin
from core.models.themes import ThemePackage
from services.marketplace_bundles import import_items as items_mod
from services.marketplace_bundles import plugin_import, sync_items
from services.marketplace_bundles.canonical import canonical_bytes, revision_of
from services.marketplace_bundles.catalog_items import entry_of
from services.marketplace_bundles.exported import ExportedItem, ItemEntry
from services.marketplace_bundles.held_items import items_held
from services.marketplace_bundles.items_feed import Entries, parse_item_entry
from services.marketplace_bundles.validate import validate_bundle
from services.marketplace_cloud import feed, fetch, pull, updates
from services.themes import serve, signature_verify, storage
from tests import helpers_theme_publish as hp
from tests.helpers_prd270w2_plugins import (
    BUNDLES, CLOUD, Runner, bundle_of, cloud_settings, external_id, fake_bucket, files_of, package_url,
    refuse_models, zip_of,
)
from tests.test_prd270w1_cloud_pull import _catalog, _item, _keys, _pkg

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "marketplace_bundles"
SYNC = "/api/marketplace/cloud-sync"
STATUS = SYNC + "/status"
CATALOG = CLOUD + "/api/marketplace/catalog"
TYPES = ("agent", "playbook", "package", "plugin")
ZERO = {"new": 0, "updated": 0, "unchanged": 0, "refused": 0}
WS = uuid.uuid4()


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = sa.create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as conn:
            for probe in (
                "SELECT agent_id, recipe_id, package_id, plugin_id, unresolved FROM marketplace_cloud_items LIMIT 1",
                "SELECT refused_count FROM marketplace_cloud_sync LIMIT 1",
                "SELECT source FROM theme_packages LIMIT 1",
                "SELECT security_scan_id, materialized_skill_ids FROM marketplace_plugins LIMIT 1",
                "SELECT scanned_by FROM plugin_security_scans LIMIT 1",
                "SELECT package_slug FROM skills LIMIT 1",
                "SELECT slug, members FROM marketplace_packages LIMIT 1",
            ):
                conn.execute(sa.text(probe))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the Sync route tests need Postgres with the PRD-270 W2 schema: {exc}")
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


def _caller():
    user = UserContext(id="clerk-9", clerk_user_id="clerk-9", system_role="user",
                       raw_claims={"sub": "clerk-9", "iat": int(time.time())})
    return RequestContext(workspace_id=WS, user=user, auth_type="clerk")


def _client(database, factory=None):
    app = FastAPI()
    app.include_router(sync_api.router)
    app.dependency_overrides[get_request_context_hybrid] = _caller
    app.dependency_overrides[get_db] = database
    app.dependency_overrides[sync_api.get_cloud_client_factory] = lambda: factory
    return TestClient(app)


def _never(*_args, **_kwargs):
    raise AssertionError("nothing may run here")


@pytest.fixture
def env(engine, monkeypatch, tmp_path):
    """The router over one rolled-back transaction, an owner calling, Automatos Cloud mocked, the stores fake."""
    cloud, theme_s3 = cloud_settings(monkeypatch), hp.FakeS3()
    fake_bucket(monkeypatch)
    refuse_models(monkeypatch)
    monkeypatch.setattr(storage, "get_s3_client", lambda: theme_s3)
    monkeypatch.setattr(storage, "ensure_bucket", lambda bucket: None)
    monkeypatch.setattr(serve, "get_s3_client", lambda: theme_s3)
    monkeypatch.setattr(signature_verify, "KEYS_PATH", _keys(tmp_path))
    monkeypatch.setattr(config, "THEMES_MARKETPLACE", "on", raising=False)
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_FEED_MAX_KB", 1024)
    state = SimpleNamespace(cloud=cloud, role="owner")
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: state.role)
    with _rolled_back(engine) as session:
        state.db = session
        state.client = _client(lambda: session, cloud.factory)
        yield state


# --------------------------------------------------------------------------- the catalogue


def _fixture(name):
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def _member(kind, ref):
    return {"type": kind, "ref": ref, "name": "A member", "description": ""}


def _agent_bundle(tag, slug):
    """The corpus's agent, named for this test, using the plugin ``slug`` and the skill it makes."""
    base = _fixture("agent")
    body = {**base["agent"], "persona": None, "model": {"modelId": None, "provider": None},
            "skills": [{"name": f"{slug}-skill", "package": slug}], "apps": [], "plugins": [slug]}
    return {**base, "externalId": uuid.uuid4().hex, "name": f"Test Sync Helper {tag}", "agent": body}


def _playbook_bundle(tag, agent):
    """The corpus's Playbook, its one step on ``agent``."""
    base = _fixture("playbook")
    body = {**base["playbook"], "agents": {"a1": {"item": agent["externalId"]}}, "steps": base["playbook"]["steps"][:1]}
    return {**base, "externalId": uuid.uuid4().hex, "name": f"Test Sync Digest {tag}", "playbook": body}


def _package_bundle(tag, agent, playbook, slug):
    base = _fixture("package")
    members = [_member("agent", {"item": agent["externalId"]}), _member("playbook", {"item": playbook["externalId"]}),
               _member("plugin", {"slug": slug})]
    body = {**base["package"], "slug": f"t270sr-pack-{tag}", "members": members}
    return {**base, "externalId": uuid.uuid4().hex, "name": f"Test Sync Pack {tag}", "package": body}


def _entry(cloud, bundle):
    """``bundle`` served at its derived URL; its catalogue entry."""
    validate_bundle(bundle)
    revision = revision_of(bundle)
    cloud.answer(BUNDLES + revision, canonical_bytes(bundle))
    identity = ItemEntry(bundle["type"], bundle["externalId"], bundle["version"], frozenset())
    return entry_of(ExportedItem(identity, bundle, revision))


def _offer(cloud):
    """A theme, a plugin, an agent, a Playbook and a package served; the catalogue lists them, its items last
    first (the passes' order is not the catalogue's)."""
    tag = uuid.uuid4().hex[:10]
    slug = f"t270sr-{tag}"
    plugin = bundle_of(slug, name=f"Test Sync Kit {tag}")
    agent = _agent_bundle(tag, slug)
    playbook = _playbook_bundle(tag, agent)
    package = _package_bundle(tag, agent, playbook, slug)
    cloud.answer(package_url(slug, plugin["version"]), zip_of(files_of(slug, plugin["version"])))
    theme = f"sync{tag}/arcade"
    entries = [_entry(cloud, bundle) for bundle in (package, playbook, agent, plugin)]
    cloud.answer(CATALOG, _catalog(_item(_pkg(theme)), *entries))
    bundles = {"agent": agent, "playbook": playbook, "package": package, "plugin": plugin}
    return SimpleNamespace(theme=theme, slug=slug, bundles=bundles, entries=entries)


def _mapping(db, bundle):
    return (
        db.query(MarketplaceCloudItem)
        .filter(MarketplaceCloudItem.item_type == bundle["type"],
                MarketplaceCloudItem.external_id == bundle["externalId"])
        .one_or_none()
    )


def _theme_rows(db, theme):
    return db.query(ThemePackage).filter(ThemePackage.theme_id == theme, ThemePackage.source == "cloud").count()


def _items(**by_type):
    return {item_type: {**ZERO, **by_type.get(item_type, {})} for item_type in TYPES}


# --------------------------------------------------------------------------- the Sync


def test_an_owners_sync_writes_all_five_and_answers_the_counts(env, monkeypatch):
    offer, before = _offer(env.cloud), items_held(env.db)
    order, held = [], []
    for module, name, label in ((pull, "sync_catalog", "themes"), (sync_api, "run_item_passes", "passes"),
                                (updates, "apply_cloud_updates", "updates")):
        real = getattr(module, name)
        monkeypatch.setattr(module, name, lambda *a, _real=real, _label=label: order.append(_label) or _real(*a))
    env.cloud.on_request = lambda: held.append(pull._SYNC_LOCK.locked())

    answer = env.client.post(SYNC)

    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert set(body) == {"counts", "items", "plugins", "updatesApplied", "updatesFailed", "status"}
    assert body["items"] == _items(**{item_type: {"new": 1} for item_type in TYPES})
    assert body["plugins"] == {"waiting": 0} and body["counts"] == {**ZERO, "new": 5}
    assert (body["updatesApplied"], body["updatesFailed"]) == (0, 0)
    status = body["status"]
    assert status["counts"] == body["counts"] and status["lastError"] is None
    assert status["items"] == {item_type: before[item_type] + 1 for item_type in TYPES}
    assert env.client.get(STATUS).json() == status
    assert order == ["themes", "passes", "updates"] and held and all(held)
    plugin_bundle, package_zip = env.cloud.urls()[1:3]
    assert env.cloud.urls()[0] == CATALOG and package_zip == package_url(offer.slug, "1.0.0")
    assert plugin_bundle == BUNDLES + revision_of(offer.bundles["plugin"])  # the plugin before any item
    _assert_all_five_written(env.db, offer)


def _assert_all_five_written(db, offer):
    db.expire_all()
    assert _theme_rows(db, offer.theme) == 1
    plugin = db.query(MarketplacePlugin).filter(MarketplacePlugin.slug == offer.slug).one()
    assert (plugin.approval_status, plugin.is_active) == ("approved", True)
    mappings = {item_type: _mapping(db, bundle) for item_type, bundle in offer.bundles.items()}
    assert mappings["plugin"].plugin_id == plugin.id and mappings["plugin"].external_id == external_id(offer.slug)
    agent = db.get(Agent, mappings["agent"].agent_id)
    assert agent.name == offer.bundles["agent"]["name"] and mappings["agent"].unresolved == []  # its plugin and skill
    playbook = db.get(WorkflowTemplate, mappings["playbook"].recipe_id)
    assert playbook.steps[0]["agent_id"] == agent.id
    package = db.get(MarketplacePackage, mappings["package"].package_id)
    assert package.slug == offer.bundles["package"]["package"]["slug"]


def test_a_second_sync_is_all_unchanged_and_fetches_no_bundle(env):
    _offer(env.cloud)
    assert env.client.post(SYNC).status_code == 200
    env.cloud.requests = []

    again = env.client.post(SYNC).json()

    assert again["items"] == _items(**{item_type: {"unchanged": 1} for item_type in TYPES})
    assert again["counts"] == {**ZERO, "unchanged": 5} and again["status"]["counts"] == again["counts"]
    assert env.cloud.urls() == [CATALOG]


def _broken_playbook_writer(db, *_args):
    db.execute(sa.text("SELECT 1 / 0"))  # the database refuses the statement mid-pass


def test_a_failing_item_pass_keeps_the_theme_and_the_plugin_and_records_the_error(env, monkeypatch):
    offer = _offer(env.cloud)
    monkeypatch.setattr(items_mod, "upsert_playbook", _broken_playbook_writer)

    body = env.client.post(SYNC).json()

    refused = {"refused": 1}
    assert body["items"] == _items(agent=refused, playbook=refused, package=refused, plugin={"new": 1})
    assert body["counts"] == {**ZERO, "new": 2, "refused": 3}
    status = body["status"]
    assert status["lastError"] == items_mod.NOT_SAVED and status["lastSuccessAt"] is not None
    assert status["counts"] == body["counts"]
    env.db.expire_all()
    assert _theme_rows(env.db, offer.theme) == 1 and _mapping(env.db, offer.bundles["plugin"]) is not None
    assert [_mapping(env.db, offer.bundles[name]) for name in ("agent", "playbook", "package")] == [None] * 3


def test_a_failing_plugin_pass_records_its_error_and_runs_no_item_pass(env, monkeypatch):
    offer = _offer(env.cloud)

    def unreadable(*_args):
        raise sa.exc.OperationalError("SELECT", {}, Exception("the database went away"))

    monkeypatch.setattr(plugin_import, "changed_entries", unreadable)

    body = env.client.post(SYNC).json()

    assert body["items"] == _items(**{item_type: {"refused": 1} for item_type in TYPES})
    assert body["status"]["lastError"] == sync_items.PLUGINS_FAILED
    assert body["counts"] == {**ZERO, "new": 1, "refused": 4}
    assert env.cloud.urls() == [CATALOG]  # no bundle fetched
    env.db.expire_all()
    assert _theme_rows(env.db, offer.theme) == 1
    assert [_mapping(env.db, bundle) for bundle in offer.bundles.values()] == [None] * 4


def test_a_plugin_past_the_limit_waits_and_the_package_needing_it_is_refused(env, monkeypatch):
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_PLUGINS_PER_SYNC", 0)
    offer = _offer(env.cloud)

    body = env.client.post(SYNC).json()

    new = {"new": 1}
    assert body["plugins"] == {"waiting": 1}
    assert body["items"] == _items(agent=new, playbook=new, package={"refused": 1})
    assert package_url(offer.slug, "1.0.0") not in env.cloud.urls()
    assert {"kind": "plugin", "name": offer.slug} in _mapping(env.db, offer.bundles["agent"]).unresolved


# --------------------------------------------------------------------------- when the passes run nothing


def _synced(entries=(), ok=True):
    """W1's result as the passes read it: a catalogue carrying ``entries`` (raw entries, parsed here)."""
    listed = tuple(parse_item_entry(entry) for entry in entries)
    catalog = feed.Catalog(None, (), (), 0, Entries(listed=listed))
    return pull.SyncOutcome(ok=ok, counts=pull.SyncCounts(new=1), catalog=catalog if ok else None)


@pytest.mark.parametrize("case", ["url-empty", "saas", "theme-pass-refused", "another-sync-running"])
def test_the_passes_run_nothing_unless_w1s_pass_succeeded_here_and_no_other_sync_runs(env, monkeypatch, case):
    entries = _offer(env.cloud).entries
    monkeypatch.setattr(sync_items, "import_plugins", _never)
    monkeypatch.setattr(sync_items, "import_items", _never)
    synced = _synced(entries, ok=case != "theme-pass-refused")
    if case == "url-empty":
        monkeypatch.setattr(config, "AUTOMATOS_CLOUD_URL", "")
    if case == "saas":
        monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    locked = case == "another-sync-running" and pull._SYNC_LOCK.acquire(blocking=False)
    try:
        answer = sync_items.run_item_passes(env.db, synced, env.cloud.factory, Runner())
    finally:
        if locked:
            pull._SYNC_LOCK.release()
    assert answer == {"counts": {**ZERO, "new": 1}, "items": _items(), "plugins": {"waiting": 0}}
    assert env.cloud.requests == [] and not pull._SYNC_LOCK.locked()


class _CountsRefused:
    """A session whose UPDATE of the status row fails; it records what else was done."""

    def __init__(self):
        self.row, self.ended = SimpleNamespace(last_error=None), []

    def query(self, *_args):
        return self

    def filter(self, *_args):
        return self

    def update(self, *_args, **_kwargs):
        raise sa.exc.OperationalError("UPDATE", {}, Exception("the database went away"))

    def get(self, *_args):
        return self.row

    def rollback(self):
        self.ended.append("rollback")

    def commit(self):
        self.ended.append("commit")


def test_counts_that_cant_be_saved_are_recorded_as_the_rows_error():
    session = _CountsRefused()
    sync_items._add_counts(session, pull.SyncCounts(new=2))
    assert session.row.last_error == sync_items.COUNTS_NOT_SAVED and session.ended == ["rollback", "commit"]


def test_with_the_url_empty_the_sync_is_w1s_409_and_requests_nothing(env, monkeypatch):
    _offer(env.cloud)
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_URL", "")
    monkeypatch.setattr(sync_api, "run_item_passes", _never)
    answer = env.client.post(SYNC)
    assert answer.status_code == 409 and answer.json()["detail"] == fetch.CLOUD_OFF
    assert env.cloud.requests == []


# --------------------------------------------------------------------------- who can Sync (no database)


@pytest.fixture
def gates(monkeypatch):
    """The router with no database, in the local edition with a Cloud URL; nothing behind a gate may run."""
    state = SimpleNamespace(role="owner")
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_URL", CLOUD)
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: state.role)
    for module, name in ((pull, "sync_catalog"), (pull, "sync_status"), (sync_api, "run_item_passes"),
                         (updates, "apply_cloud_updates")):
        monkeypatch.setattr(module, name, _never)
    state.client = _client(lambda: object())
    return state


@pytest.mark.parametrize("role", ["editor", "viewer"])
def test_a_member_who_is_not_an_owner_or_admin_gets_403_and_no_pass_runs(gates, role):
    gates.role = role
    answer = gates.client.post(SYNC)
    assert answer.status_code == 403 and answer.json()["detail"] == "Permission denied: workspace:manage"


@pytest.mark.parametrize("path, method", [(SYNC, "POST"), (STATUS, "GET")])
def test_on_saas_the_sync_and_its_status_are_404_and_no_pass_runs(gates, monkeypatch, path, method):
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    answer = gates.client.request(method, path)
    assert answer.status_code == 404 and answer.json()["detail"] == sync_api.HOSTED_NO_SYNC
