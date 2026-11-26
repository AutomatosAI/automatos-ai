"""PRD-277 US-704 (D-12): the operator switches on each call to Automatos's hosted services.

Three settings, one per hosted service the code calls (``core/editions.py`` ``HOSTED_SERVICE_SETTINGS``):
``ENTERPRISE_CALLS_THEME_REVOCATIONS`` (the revocation pull, its schedule, its sync status and Check now),
``ENTERPRISE_CALLS_CLOUD_SYNC`` (the Cloud Sync routes, the Sync and its package downloads, the Cloud file
addresses) and ``ENTERPRISE_CALLS_CLOUD_ITEMS`` (the Sync's item and plugin passes, and the bundle and package
requests they make). A setting stops requests; it never stops the install reading what it already holds.
Nothing leaves the process: Automatos's hosts are ``httpx.MockTransport``s, and while a service is off every
client seam and the real HTTP transports raise and record. Code under test catches errors in places, so every
"off" test asserts that record is empty, not only that no error surfaced. One test holds an in-memory SQLite
install; the rest use no database. Proven:

* in enterprise each capability is its own setting: off by default, on only for a plain true form, off for a
  malformed value, and one setting never turns on another; in ``local`` and ``saas`` the row decides and no
  setting moves it;
* start-up refuses, in enterprise only, a malformed value, Cloud sync without theme revocations and Cloud items
  without Cloud sync, naming the setting and never its value; ``config.py`` reads each setting, ``false`` unless
  set, and names it in the config surface;
* off, theme revocations: the shared scheduler registers no pull job, ``pull_feed`` sends nothing and records
  nothing, and both admin routes answer 404 with their plain sentence before anything runs; on: the job is
  registered, the pull reaches the feed, and the routes answer. A key a pull revoked while it was on still reads
  as revoked through ``signature_state`` once it is off; an install that never pulled holds no revoked key; the
  hosted edition reads no row;
* off, Cloud sync: ``sync_catalog`` and a package download send nothing, no Cloud file address is given, and
  both routes answer 404 before the caller is looked at; on: the Sync and the download reach Automatos Cloud;
* off, Cloud items: the Sync runs no item pass, a bundle fetch and a plugin package fetch each refuse before any
  request, and the three read routes still answer for what the install holds, before and after the setting is
  switched off; on: the passes run, and a plugin package is fetched;
* ``GET /api/hosted-services`` answers each capability, for any signed-in caller;
* ``local`` calls each service as before, and ``saas`` stays the host: 404 with its own sentences, no pull.
"""
from __future__ import annotations

import inspect
import itertools
import json
import os
import re
import uuid
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import api.admin_theme_revocation_sync as revocation_api  # noqa: E402
import api.hosted_services as hosted_api  # noqa: E402
import api.marketplace_cloud_items as items_api  # noqa: E402
import api.marketplace_cloud_sync as sync_api  # noqa: E402
import core.auth.workspace_permission as permission_mod  # noqa: E402
import services.scheduler as scheduler_mod  # noqa: E402
import services.theme_revocation_scheduler as pull_scheduler  # noqa: E402
from config import config  # noqa: E402
from core import edition_start_up, editions  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.edition_start_up import EditionNotReady, check_hosted_service_settings, check_start_up  # noqa: E402
from services import admin_audit  # noqa: E402
from services.marketplace_bundles import items_feed, items_gate, plugin_import, sync_items  # noqa: E402
from services.marketplace_cloud import fetch, package, pull, urls  # noqa: E402
from services.themes import revocation_pull, revocation_sync_state, signature_state  # noqa: E402
from tests.test_prd261_revocation_pull import KEY_A  # noqa: E402
from tests.test_prd261_revocation_pull import TABLES as THEME_TABLES  # noqa: E402
from tests.test_prd261_revocation_pull import _feed as _revocation_feed  # noqa: E402
from tests.test_prd277_us704_cloud_sync_admin import _person  # noqa: E402
from tests.test_prd277_enterprise_start_up import (  # noqa: E402,F401  (the reload fixture travels with _enterprise)
    _assert_no_value_in,
    _enterprise,
    _refusal,
    _restore_config_module,
)

REVOCATIONS, SYNC, ITEMS = (editions.HOSTED_SERVICE_SETTINGS[field] for field in (
    "calls_theme_revocations", "calls_cloud_sync", "calls_cloud_items"))
SETTINGS = (REVOCATIONS, SYNC, ITEMS)
FIELDS = ("calls_theme_revocations", "calls_cloud_sync", "calls_cloud_items")
FEED_URL = "https://feed.example.test/api/themes/revocations"
CLOUD = "https://cloud.example.test"
WS = uuid.uuid4()
DIGEST = "ab" * 32
MARKER = "prd277-us704-not-a-switch"  # a malformed value no message may echo
REVOCATION_ROUTES = (("GET", "/api/admin/themes/revocations/sync"), ("POST", "/api/admin/themes/revocations/sync/pull"))
SYNC_ROUTES = (("GET", "/api/marketplace/cloud-sync/status"), ("POST", "/api/marketplace/cloud-sync"))
ITEM_ROUTES = (
    ("GET", "/api/marketplace/cloud-sync/items"),
    ("GET", "/api/marketplace/cloud-sync/items/agent/a-key/disclosure"),
    ("GET", "/api/marketplace/cloud-sync/updates"),
)
ALL_ON = dict.fromkeys(SETTINGS, "true")
GARBAGE = dict.fromkeys(SETTINGS, MARKER)


def _switch(monkeypatch, edition, **values):
    """``edition`` with each named setting set (``revocations``, ``sync``, ``items``), the rest ``false``."""
    named = {"revocations": REVOCATIONS, "sync": SYNC, "items": ITEMS}
    monkeypatch.setattr(config, "AUTH_EDITION", edition)
    for key, setting in named.items():
        monkeypatch.setattr(config, setting, values.get(key, "false"))
    monkeypatch.setattr(config, revocation_pull.URL_SETTING, FEED_URL, raising=False)
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_URL", CLOUD)


@pytest.fixture
def no_requests(monkeypatch):
    """Every way out of the process raises and records: the module client seams and the real transports."""
    touched = []

    def refuse(*args, **kwargs):
        touched.append(args)
        raise AssertionError("nothing may be sent to Automatos while the service is off")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", refuse)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", refuse)
    monkeypatch.setattr(fetch, "_client", refuse)
    monkeypatch.setattr(revocation_pull, "_client", refuse)
    return SimpleNamespace(touched=touched, factory=refuse)


def _host(answer: bytes = b"{}"):
    """A mock Automatos host: ``(seen requests, client factory)``."""
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, content=answer)

    return seen, lambda: httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)


class _Db:
    """A session that only ends its transaction; any read is a test failure."""

    def commit(self):
        return None

    def rollback(self):
        return None

    def get(self, *args, **kwargs):
        raise AssertionError("no row may be read here")


# --------------------------------------------------------------------------- the capabilities


def test_each_hosted_service_has_its_own_setting():
    assert dict(editions.HOSTED_SERVICE_SETTINGS) == dict(zip(FIELDS, SETTINGS))
    assert edition_start_up.HOSTED_SERVICE_SETTING_NAMES == SETTINGS


@pytest.mark.parametrize("on", [combo for size in range(len(FIELDS) + 1) for combo in itertools.combinations(FIELDS, size)])
def test_in_enterprise_each_setting_decides_its_own_service_only(monkeypatch, on):
    _switch(monkeypatch, "enterprise")
    for field in on:
        monkeypatch.setattr(config, editions.HOSTED_SERVICE_SETTINGS[field], "true")
    caps = editions.current()
    assert {field: getattr(caps, field) for field in FIELDS} == {field: field in on for field in FIELDS}
    rest = {k: v for k, v in vars(caps).items() if k not in FIELDS}
    assert rest == {k: v for k, v in vars(editions.capabilities_for("enterprise")).items() if k not in FIELDS}


def test_in_enterprise_every_service_is_off_by_default(monkeypatch):
    _switch(monkeypatch, "enterprise")
    assert not any(getattr(editions.current(), field) for field in FIELDS)
    unset = editions.with_hosted_service_settings(editions.capabilities_for("enterprise"), {})
    assert not any(getattr(unset, field) for field in FIELDS)


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "on"])
def test_a_plain_true_form_switches_a_service_on(monkeypatch, value):
    _switch(monkeypatch, "enterprise", revocations=value)
    assert editions.current().calls_theme_revocations is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", MARKER, "tru", "2", "enabled", None])
def test_any_other_value_reads_as_off(monkeypatch, value):
    _switch(monkeypatch, "enterprise", revocations=value)
    assert editions.current().calls_theme_revocations is False


@pytest.mark.parametrize("edition", ["local", "saas"])
@pytest.mark.parametrize("settings", [ALL_ON, GARBAGE, dict.fromkeys(SETTINGS, "false")])
def test_local_and_saas_ignore_the_settings(monkeypatch, edition, settings):
    monkeypatch.setattr(config, "AUTH_EDITION", edition)
    for name, value in settings.items():
        monkeypatch.setattr(config, name, value)
    assert editions.current() == editions.capabilities_for(edition)
    assert editions.with_hosted_service_settings(editions.capabilities_for(edition), settings) is \
        editions.capabilities_for(edition)


def test_local_calls_each_service_and_saas_calls_none():
    local, saas = editions.capabilities_for("local"), editions.capabilities_for("saas")
    assert all(getattr(local, field) for field in FIELDS) and not any(getattr(saas, field) for field in FIELDS)
    assert saas.hosts_automatos_cloud and not local.hosts_automatos_cloud


# --------------------------------------------------------------------------- start-up


@pytest.mark.parametrize("name", SETTINGS)
def test_start_up_refuses_a_malformed_value_naming_the_setting_never_the_value(name):
    with pytest.raises(EditionNotReady) as refused:
        check_hosted_service_settings(editions.capabilities_for("enterprise"), {name: MARKER})
    assert name in str(refused.value) and edition_start_up.HOSTED_SWITCH_PROBLEM in str(refused.value)
    assert MARKER not in str(refused.value)


@pytest.mark.parametrize("settings, name, problem", [
    ({SYNC: "true"}, SYNC, edition_start_up.CLOUD_SYNC_NEEDS_REVOCATIONS),
    ({SYNC: "true", ITEMS: "true"}, SYNC, edition_start_up.CLOUD_SYNC_NEEDS_REVOCATIONS),
    ({ITEMS: "true"}, ITEMS, edition_start_up.CLOUD_ITEMS_NEED_SYNC),
    ({REVOCATIONS: "true", ITEMS: "true"}, ITEMS, edition_start_up.CLOUD_ITEMS_NEED_SYNC),
])
def test_start_up_refuses_a_service_switched_on_without_the_one_it_needs(settings, name, problem):
    with pytest.raises(EditionNotReady) as refused:
        check_hosted_service_settings(editions.capabilities_for("enterprise"), settings)
    assert str(refused.value) == edition_start_up.BAD_SESSION_SETTING.format(name=name, problem=problem)


@pytest.mark.parametrize("settings", [
    {}, {REVOCATIONS: "true"}, {REVOCATIONS: "1", SYNC: "yes"}, ALL_ON, dict.fromkeys(SETTINGS, "off"),
])
def test_start_up_accepts_every_combination_whose_needs_are_met(settings):
    check_hosted_service_settings(editions.capabilities_for("enterprise"), settings)  # must not raise


@pytest.mark.parametrize("edition", ["local", "saas"])
def test_local_and_saas_start_whatever_the_settings_hold(edition):
    check_start_up(editions.capabilities_for(edition), GARBAGE)  # must not raise
    check_start_up(editions.capabilities_for(edition), {SYNC: "true", ITEMS: "true"})


def test_config_reads_each_setting_false_unless_set():
    cfg = _enterprise()
    assert all(getattr(cfg, name) == "false" for name in SETTINGS)
    cfg.validate_auth_edition()  # an enterprise install with every service off starts


def test_config_folds_a_set_value_and_an_enterprise_start_checks_it(monkeypatch):
    monkeypatch.setenv(REVOCATIONS, " TRUE ")
    monkeypatch.setenv(SYNC, "true")
    cfg = _enterprise()
    assert (getattr(cfg, REVOCATIONS), getattr(cfg, SYNC)) == ("true", "true")
    cfg.validate_auth_edition()  # revocations and Cloud sync on, items off: starts


def test_an_enterprise_start_refuses_a_malformed_setting_without_echoing_it(monkeypatch):
    monkeypatch.setenv(ITEMS, MARKER)
    message = _refusal(_enterprise())
    assert ITEMS in message and MARKER.lower() not in message.lower()
    _assert_no_value_in(message)


def test_an_enterprise_start_refuses_cloud_sync_without_revocations(monkeypatch):
    monkeypatch.setenv(SYNC, "true")
    assert edition_start_up.CLOUD_SYNC_NEEDS_REVOCATIONS in _refusal(_enterprise())


def test_the_settings_are_in_the_config_surface():
    surface = json.loads((Path(__file__).resolve().parents[1] / "reports" / "config-surface.json").read_text())
    assert set(SETTINGS) <= set(surface["settings"])


# --------------------------------------------------------------------------- the route harness


class _Untouchable:
    """A session no code may use: any attribute is a test failure."""

    def __getattr__(self, name):
        raise AssertionError(f"the database was used ({name}) where nothing may run")


def _never(*args, **kwargs):
    raise AssertionError("nothing may run behind a service that is off")


def _admin():
    """A platform admin of the workspace (passes the admin and workspace checks behind the edition's)."""
    user = UserContext(id="admin-704", email="admin@example.test", system_role="super_admin")
    return RequestContext(workspace_id=WS, user=user, auth_type="session")


def _client(router, ran):
    """``router`` alone; ``ran`` records the caller and the database being looked at."""
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: ran.append("caller") or _admin()
    app.dependency_overrides[get_db] = lambda: ran.append("database") or _Untouchable()
    return TestClient(app)


def _answers(client, routes):
    return [(resp.status_code, resp.json()) for resp in (client.request(method, path) for method, path in routes)]


class _FakeAPScheduler:
    """Stands in for AsyncIOScheduler: records every job, starts nothing."""

    def __init__(self, **kwargs):
        self.jobs = {}
        self.running = False

    def start(self):
        self.running = True

    def add_job(self, func, trigger=None, **kwargs):
        self.jobs[kwargs["id"]] = SimpleNamespace(func=func, trigger=trigger, **kwargs)


def _scheduled_jobs(monkeypatch):
    """The jobs the shared scheduler registers at start."""
    monkeypatch.setattr(scheduler_mod, "AsyncIOScheduler", _FakeAPScheduler)
    unified = scheduler_mod.UnifiedScheduler()
    unified.start()
    return unified.apscheduler.jobs


# --------------------------------------------------------------------------- theme revocations


def test_off_no_revocation_pull_is_ever_scheduled(monkeypatch, no_requests):
    _switch(monkeypatch, "enterprise", sync="true", items="true")  # the other two never switch it on
    assert pull_scheduler.JOB_ID not in _scheduled_jobs(monkeypatch)
    recorder = _FakeAPScheduler()
    assert pull_scheduler.ThemeRevocationScheduler().start(recorder) is False and recorder.jobs == {}
    assert no_requests.touched == []


def test_on_the_revocation_pull_is_scheduled(monkeypatch):
    _switch(monkeypatch, "enterprise", revocations="true")
    job = _scheduled_jobs(monkeypatch)[pull_scheduler.JOB_ID]
    assert job.func is pull_scheduler.run_scheduled_pull and job.max_instances == 1


def test_off_a_pull_sends_nothing_and_records_nothing(monkeypatch, no_requests):
    _switch(monkeypatch, "enterprise", sync="true", items="true")
    outcome = revocation_pull.pull_feed(_Untouchable(), no_requests.factory)
    assert outcome == revocation_pull.PullOutcome(ok=False, error=revocation_pull.PULL_NOT_SWITCHED_ON)
    assert revocation_pull.pull_enabled() is False and no_requests.touched == []


def test_on_a_pull_reaches_the_feed(monkeypatch):
    _switch(monkeypatch, "enterprise", revocations="true")
    seen, factory = _host()
    monkeypatch.setattr(revocation_pull, "parse_feed", lambda data: data)
    monkeypatch.setattr(revocation_pull, "apply_feed", lambda db, feed, at: 0)
    assert revocation_pull.pull_feed(_Db(), factory) == revocation_pull.PullOutcome(ok=True, rows=0)
    assert [(request.method, str(request.url)) for request in seen] == [("GET", FEED_URL)]
    assert revocation_pull.pull_enabled() is True


@pytest.fixture
def held():
    """An in-memory install with the theme tables, and a mock feed that revokes ``KEY_A``."""
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    THEME_TABLES[0].metadata.create_all(engine, tables=THEME_TABLES)
    session = sessionmaker(bind=engine)()
    seen, factory = _host(_revocation_feed([], keys=[KEY_A]))
    try:
        yield SimpleNamespace(db=session, seen=seen, factory=factory)
    finally:
        session.close()
        engine.dispose()


def test_a_key_revoked_while_on_stays_revoked_once_switched_off(monkeypatch, held, no_requests):
    _switch(monkeypatch, "enterprise", revocations="true")
    assert revocation_pull.pull_feed(held.db, held.factory).ok
    assert signature_state.current_key_set(held.db).revoked == {KEY_A}
    _switch(monkeypatch, "enterprise")  # the operator switches theme revocations off
    assert signature_state.current_key_set(held.db).revoked == {KEY_A}
    assert revocation_pull.pull_feed(held.db, held.factory).error == revocation_pull.PULL_NOT_SWITCHED_ON
    assert len(held.seen) == 1 and no_requests.touched == []  # the one pull made while it was on


def test_an_install_that_never_pulled_holds_no_revoked_key(monkeypatch, held, no_requests):
    _switch(monkeypatch, "enterprise")
    assert signature_state.current_key_set(held.db).revoked == frozenset()
    assert held.seen == [] and no_requests.touched == []


def test_the_hosted_edition_reads_no_revoked_key_row(monkeypatch, no_requests):
    _switch(monkeypatch, "saas", revocations="true")
    assert revocation_sync_state.revoked_key_ids(_Untouchable()) == frozenset()
    assert no_requests.touched == []


def test_off_the_revocation_routes_answer_404_and_run_nothing(monkeypatch, no_requests):
    _switch(monkeypatch, "enterprise", sync="true", items="true")
    monkeypatch.setattr(revocation_pull, "sync_status", _never)
    monkeypatch.setattr(revocation_pull, "pull_feed", _never)
    off = {"detail": revocation_api.PULL_NOT_SWITCHED_ON}
    # The admin check runs first, as before; nothing past the 404 runs (sync_status and pull_feed refuse).
    assert _answers(_client(revocation_api.router, []), REVOCATION_ROUTES) == [(404, off), (404, off)]
    assert no_requests.touched == []


def test_on_the_revocation_routes_answer(monkeypatch):
    _switch(monkeypatch, "enterprise", revocations="true")
    monkeypatch.setattr(revocation_pull, "sync_status", lambda db: {"enabled": True})
    monkeypatch.setattr(revocation_pull, "pull_feed", lambda db, factory: revocation_pull.PullOutcome(ok=True))
    recorded = []  # the Check now's audit row (#1657) is test_prd261_revocation_pull's; here only that it's asked for
    monkeypatch.setattr(admin_audit, "record_after", lambda db, ctx, request, **row: recorded.append(row["action"]))
    answers = _answers(_client(revocation_api.router, []), REVOCATION_ROUTES)
    assert answers == [(200, {"enabled": True}), (200, {"enabled": True})]
    assert recorded == [admin_audit.REVOCATION_PULL]


# --------------------------------------------------------------------------- Cloud sync


def _version():
    return package.CloudVersion(id=uuid.uuid4(), theme_id="arcade", version="1.0.0", digest=DIGEST,
                                file_manifest=[], signature=None, storage_prefix="themes/arcade/")


def test_off_the_sync_sends_nothing_and_records_nothing(monkeypatch, no_requests):
    _switch(monkeypatch, "enterprise", revocations="true", items="true")
    outcome = pull.sync_catalog(_Untouchable(), no_requests.factory)
    assert (outcome.ok, outcome.off, outcome.error) == (False, True, pull.SYNC_OFF)
    assert pull.sync_enabled() is False and no_requests.touched == []


def test_on_the_sync_reaches_automatos_cloud(monkeypatch):
    _switch(monkeypatch, "enterprise", revocations="true", sync="true")
    seen, factory = _host()
    monkeypatch.setattr(pull, "parse_catalog", lambda data, db: SimpleNamespace(entries=()))
    monkeypatch.setattr(pull, "apply_catalog", lambda db, catalog, at: pull.SyncCounts())
    assert pull.sync_catalog(_Db(), factory).ok is True
    assert [(request.method, str(request.url)) for request in seen] == [("GET", urls.catalog_url(CLOUD))]


def test_off_no_package_is_downloaded(monkeypatch, no_requests):
    _switch(monkeypatch, "enterprise", revocations="true", items="true")
    with pytest.raises(package.CloudPackageError) as refused:
        package._download(_version(), no_requests.factory)
    assert refused.value.code == package.CLOUD_OFF and no_requests.touched == []


def test_on_a_package_is_downloaded(monkeypatch):
    _switch(monkeypatch, "enterprise", revocations="true", sync="true")
    seen, factory = _host(b"package-bytes")
    assert package._download(_version(), factory) == b"package-bytes"
    assert [str(request.url) for request in seen] == [urls.package_url(CLOUD, DIGEST)]


def test_off_no_cloud_file_address_is_given(monkeypatch, no_requests):
    _switch(monkeypatch, "enterprise", revocations="true", items="true")
    assert urls.cloud_file_url(DIGEST, "assets/previews/a.png") is None
    assert no_requests.touched == []
    _switch(monkeypatch, "enterprise", revocations="true", sync="true")
    assert urls.cloud_file_url(DIGEST, "assets/previews/a.png") == (
        f"{CLOUD}/api/marketplace/catalog/files/{DIGEST}/assets/previews/a.png")


def test_off_the_sync_routes_answer_404_before_the_caller_is_looked_at(monkeypatch, no_requests):
    _switch(monkeypatch, "enterprise", revocations="true", items="true")
    monkeypatch.setattr(pull, "sync_status", _never)
    monkeypatch.setattr(pull, "sync_catalog", _never)
    ran = []
    off = {"detail": sync_api.SYNC_NOT_SWITCHED_ON}
    assert _answers(_client(sync_api.router, ran), SYNC_ROUTES) == [(404, off), (404, off)]
    assert ran == [] and no_requests.touched == []


def test_on_the_sync_routes_answer(monkeypatch):
    _switch(monkeypatch, "enterprise", revocations="true", sync="true")
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: "owner")
    monkeypatch.setattr(pull, "sync_status", lambda db: {"enabled": True})
    tried = []
    monkeypatch.setattr(pull, "sync_catalog", lambda db, factory: tried.append(1) or pull.SyncOutcome(
        ok=False, error="busy", busy=True))
    client = _client(sync_api.router, [])
    # The Sync is an install admin's with a recent sign-in (test_prd277_us704_cloud_sync_admin.py holds that rule).
    overrides = client.app.dependency_overrides
    overrides[get_request_context_hybrid] = lambda: _person("super_admin", signed_in_minutes_ago=0)
    answers = _answers(client, SYNC_ROUTES)
    assert answers == [(200, {"enabled": True}), (409, {"detail": "busy"})] and tried == [1]


# --------------------------------------------------------------------------- Cloud items


def _synced():
    return pull.SyncOutcome(ok=True, catalog=SimpleNamespace(entries=()))


def test_off_the_sync_runs_no_item_pass(monkeypatch, no_requests):
    _switch(monkeypatch, "enterprise", revocations="true", sync="true")
    monkeypatch.setattr(sync_items, "_passes", _never)
    answer = sync_items.run_item_passes(_Untouchable(), _synced(), no_requests.factory, _never)
    assert answer == sync_items.Passes().view(pull.SyncCounts())
    assert sync_items.items_enabled() is False and no_requests.touched == []


def test_items_on_without_the_sync_still_run_no_pass(monkeypatch, no_requests):
    """One setting never turns on another: items on, Sync off, and no pass runs (start-up refuses it too)."""
    _switch(monkeypatch, "enterprise", revocations="true", items="true")
    monkeypatch.setattr(sync_items, "_passes", _never)
    sync_items.run_item_passes(_Untouchable(), _synced(), no_requests.factory, _never)
    assert sync_items.items_enabled() is False and no_requests.touched == []


def test_on_the_item_passes_run(monkeypatch):
    _switch(monkeypatch, "enterprise", revocations="true", sync="true", items="true")
    ran = []
    monkeypatch.setattr(sync_items, "_passes", lambda db, entries, factory, run: ran.append(1) or sync_items.Passes())
    monkeypatch.setattr(sync_items, "_add_counts", lambda db, counts: None)
    sync_items.run_item_passes(_Db(), _synced(), None, _never)
    assert ran == [1] and sync_items.items_enabled() is True


def _entry():
    return items_feed.CatalogEntry(type="agent", external_id="ab" * 16, version="1.0.0", revision="cd" * 32, name="A")


def _seams(factory):
    return plugin_import.Seams(origin=CLOUD, client_factory=factory, run_async=_never, s3=None)


@pytest.mark.parametrize("values", [{"revocations": "true", "sync": "true"}, {"revocations": "true", "items": "true"}])
def test_off_no_bundle_and_no_plugin_package_is_requested(monkeypatch, no_requests, values):
    _switch(monkeypatch, "enterprise", **values)
    fetched = items_feed.fetch_bundles([_entry()], no_requests.factory)
    assert fetched.bundles == () and [r.reason for r in fetched.refused] == [items_gate.ITEMS_NOT_SWITCHED_ON]
    with pytest.raises(items_feed.EntryRefused, match=re.escape(items_gate.ITEMS_NOT_SWITCHED_ON)):
        items_feed.fetch_bundle(_entry(), no_requests.factory)
    with pytest.raises(plugin_import.PluginRefused, match=re.escape(items_gate.ITEMS_NOT_SWITCHED_ON)):
        plugin_import._package(_seams(no_requests.factory), "a-plugin", "1.0.0")
    assert items_gate.brings_items() is False and no_requests.touched == []


def test_on_a_plugin_package_is_requested(monkeypatch):
    _switch(monkeypatch, "enterprise", revocations="true", sync="true", items="true")
    seen, factory = _host(b"zip-bytes")
    assert plugin_import._package(_seams(factory), "a-plugin", "1.0.0") == b"zip-bytes"
    assert [request.method for request in seen] == ["GET"] and items_gate.brings_items() is True


def _held_items(monkeypatch):
    """The three reads answer from what the install holds (they read rows only: local_reads, disclosure, updates)."""
    monkeypatch.setattr(items_api, "cloud_items", lambda db: {"agents": [{"id": 7}]})
    monkeypatch.setattr(items_api, "disclosure", lambda db, kind, key: {"kind": kind, "uses": ["an app"]})
    monkeypatch.setattr(items_api, "updates", lambda db, ws: {"updates": []})
    return [(200, {"agents": [{"id": 7}]}), (200, {"kind": "agent", "uses": ["an app"]}), (200, {"updates": []})]


def test_items_brought_while_on_are_still_listed_and_reviewed_once_switched_off(monkeypatch, no_requests):
    _switch(monkeypatch, "enterprise", revocations="true", sync="true", items="true")
    held_answers = _held_items(monkeypatch)
    assert _answers(_client(items_api.router, []), ITEM_ROUTES) == held_answers
    _switch(monkeypatch, "enterprise", revocations="true", sync="true")  # the operator switches Cloud items off
    assert _answers(_client(items_api.router, []), ITEM_ROUTES) == held_answers
    monkeypatch.setattr(sync_items, "_passes", _never)
    sync_items.run_item_passes(_Untouchable(), _synced(), no_requests.factory, _never)
    assert no_requests.touched == []


@pytest.mark.parametrize("values", [{}, {"revocations": "true"}, {"revocations": "true", "sync": "true"}])
def test_in_enterprise_the_item_reads_answer_whatever_the_settings_hold(monkeypatch, no_requests, values):
    _switch(monkeypatch, "enterprise", **values)
    held_answers = _held_items(monkeypatch)
    assert _answers(_client(items_api.router, []), ITEM_ROUTES) == held_answers
    assert no_requests.touched == []


# --------------------------------------------------------------------------- what the web app reads


@pytest.mark.parametrize("edition, values, expected", [
    ("enterprise", {}, (False, False, False)),
    ("enterprise", {"revocations": "true"}, (True, False, False)),
    ("enterprise", {"revocations": "true", "sync": "true"}, (True, True, False)),
    ("enterprise", {"revocations": "true", "sync": "true", "items": "true"}, (True, True, True)),
    ("local", {}, (True, True, True)),
    ("saas", {"revocations": "true", "sync": "true", "items": "true"}, (False, False, False)),
])
def test_the_web_app_reads_which_services_this_install_calls(monkeypatch, edition, values, expected):
    _switch(monkeypatch, edition, **values)
    ran = []
    resp = _client(hosted_api.router, ran).get("/api/hosted-services")
    assert resp.status_code == 200 and ran == ["caller"]
    assert resp.json() == dict(zip(("themeRevocations", "cloudSync", "cloudItems"), expected))


def test_the_hosted_services_route_needs_a_caller_and_is_a_plain_def():
    [route] = hosted_api.router.routes
    assert [dep.call for dep in route.dependant.dependencies] == [get_request_context_hybrid]
    assert not inspect.iscoroutinefunction(hosted_api.get_hosted_services)


def test_the_hosted_services_route_is_mounted_and_in_the_manifest():
    from router_manifest import MANIFEST_ROUTERS, RouterSpec

    manifest = json.loads((Path(__file__).resolve().parents[1] / "reports" / "route-manifest.json").read_text())
    assert RouterSpec("api.hosted_services") in MANIFEST_ROUTERS
    assert {"method": "GET", "path": "/api/hosted-services"} in manifest["routes"]


def test_auto_names_the_automatos_cloud_tab_in_an_organisation_only_while_it_syncs(monkeypatch):
    from modules.context.sections import product_facts as facts

    _switch(monkeypatch, "enterprise", revocations="true", items="false")
    assert facts.CLOUD_SYNC_TAB not in facts.edition_facts()
    _switch(monkeypatch, "enterprise", revocations="true", sync="true")
    tabs = facts.organisation_settings_tabs(cloud_sync_on=True)
    assert tabs[-1] == facts.CLOUD_SYNC_TAB and tabs[:-1] == facts.organisation_settings_tabs()
    assert f"Settings has these tabs: {', '.join(tabs)}." in facts.edition_facts()


# --------------------------------------------------------------------------- local and saas are unchanged


def test_local_calls_each_service_as_before(monkeypatch):
    _switch(monkeypatch, "local")  # every setting false: local ignores them
    assert pull_scheduler.JOB_ID in _scheduled_jobs(monkeypatch)
    assert revocation_pull.pull_enabled() and pull.sync_enabled() and sync_items.items_enabled()
    assert urls.cloud_file_url(DIGEST, "a.png") == f"{CLOUD}/api/marketplace/catalog/files/{DIGEST}/a.png"
    monkeypatch.setattr(pull, "sync_status", lambda db: {"enabled": True})
    monkeypatch.setattr(items_api, "cloud_items", lambda db: {"held": []})
    monkeypatch.setattr(revocation_pull, "sync_status", lambda db: {"enabled": True})
    assert _answers(_client(sync_api.router, []), SYNC_ROUTES[:1]) == [(200, {"enabled": True})]
    assert _answers(_client(items_api.router, []), ITEM_ROUTES[:1]) == [(200, {"held": []})]
    assert _answers(_client(revocation_api.router, []), REVOCATION_ROUTES[:1]) == [(200, {"enabled": True})]


def test_saas_stays_the_host_with_its_own_sentences(monkeypatch, no_requests):
    _switch(monkeypatch, "saas", revocations="true", sync="true", items="true")
    assert pull_scheduler.JOB_ID not in _scheduled_jobs(monkeypatch)
    assert not (revocation_pull.pull_enabled() or pull.sync_enabled() or sync_items.items_enabled())
    monkeypatch.setattr(revocation_pull, "sync_status", _never)
    monkeypatch.setattr(revocation_pull, "pull_feed", _never)
    ran = []
    assert _answers(_client(sync_api.router, ran), SYNC_ROUTES) == [(404, {"detail": sync_api.HOSTED_NO_SYNC})] * 2
    assert _answers(_client(items_api.router, ran), ITEM_ROUTES) == [
        (404, {"detail": items_api.HOSTED_NO_CLOUD_ITEMS})] * 3
    assert ran == []
    assert _answers(_client(revocation_api.router, []), REVOCATION_ROUTES) == [
        (404, {"detail": revocation_api.HOSTED_NO_PULL})] * 2
    assert no_requests.touched == []
