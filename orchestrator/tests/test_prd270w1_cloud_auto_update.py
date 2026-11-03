"""PRD-270 US-270-04b (§2, §5.3, K16) — after a Sync, unpinned installs move to the newest Cloud version.

``POST /api/marketplace/cloud-sync`` (``api/marketplace_cloud_sync.py``) runs on a mini FastAPI app as the
local operator (an owner), on Postgres with committing sessions through ``tests/test_prd270w1_cloud_pull.py``'s
``cloud`` fixture: the local edition, a test keys file with the RFC 8032 vector's keys active, and this test's
themes under its own publisher, swept on teardown with the workspaces it made (their installs go with the
packages). Automatos Cloud is an ``httpx.MockTransport`` that answers the catalogue and each package at its
derived URL and records every request, behind a client built with ``follow_redirects=True``; storage is a fake
S3 that answers NoSuchKey for a missing object. Installs are rows, each in a workspace of its own. Proven:

* an unpinned install on 1.0.0 moves to a synced 1.1.0 (the same install, 1.0.0 its rollback target, the offer
  cleared), the version's files stored before the move; the answer counts it; the next Sync moves and fetches
  nothing; no transaction is open on the route's session while Automatos Cloud sends the package;
* a pinned install stays on 1.0.0 with 1.1.0 offered; a theme whose only older install is pinned is offered it
  and nothing is fetched; an install unpinned after the plan was read is only offered it, and moves on the next
  Sync once the files are stored;
* a download that fails (Cloud down, a redirect, another version's bytes) leaves every install as it was, not
  even an offer, stores nothing, is logged with the theme and version and counted failed; the next Sync applies it;
* neither ``review.approve`` nor ``signing.request_signature`` runs: no ``signature_requested`` event is written;
* an install of a local-origin listing is never moved, even beside a Cloud row of its theme; an install of a
  workspace's own upload is never moved;
* a database error, or an unexpected one, in one theme's update rolls that theme back alone and counts it
  failed; the next theme moves;
* a pass that waits too long on another pass moves nothing, and the next Sync applies it.
"""
from __future__ import annotations

import io
import logging
import uuid
from types import SimpleNamespace

import httpx
import pytest
import sqlalchemy as sa
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

import api.marketplace_cloud_sync as sync_api
import core.auth.workspace_permission as permission_mod
from core.auth.dependencies import RequestContext, UserContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.principal import LOCAL_OPERATOR_CLAIM_SOURCE
from core.database.database import get_db
from core.models.themes import ThemeInstallation, ThemePackage, ThemeReviewEvent
from services.marketplace_cloud import updates as cloud_updates
from services.marketplace_cloud import urls
from services.themes import review, serve, signing, storage
from services.themes import updates as theme_updates
from services.themes.package import read_theme_package
from tests.test_prd270w1_cloud_pull import CATALOG_URL, CLOUD, _catalog, _item, _pkg, _row
from tests.test_prd270w1_cloud_pull import cloud, engine  # noqa: F401  (the Sync tests' fixtures)

SYNC = "/api/marketplace/cloud-sync"
OLD, NEW = "1.0.0", "1.1.0"
UNPINNED = None


class _Missing(Exception):
    """What boto3 raises for a key that isn't there (``serve.MISSING_OBJECT_CODES``)."""

    response = {"Error": {"Code": "NoSuchKey"}}


class FakeS3:
    """Objects by key; a missing one is NoSuchKey, as S3 answers it."""

    def __init__(self) -> None:
        self.objects = {}

    def put_object(self, Bucket, Key, Body, ContentType):  # noqa: N803 (boto3's names)
        self.objects[Key] = (Body, ContentType)

    def get_object(self, Bucket, Key):  # noqa: N803
        if Key not in self.objects:
            raise _Missing(Key)
        return {"Body": io.BytesIO(self.objects[Key][0])}

    def delete_object(self, Bucket, Key):  # noqa: N803
        self.objects.pop(Key, None)


class Cloud:
    """Automatos Cloud: the catalogue at its URL, each package served at its derived URL, every request recorded.

    ``package_status``, ``package_headers`` and ``wrong`` (bytes sent for every package) break the downloads.
    """

    def __init__(self) -> None:
        self.requests, self.packages = [], {}
        self.catalog = b""
        self.during_package = None  # runs while a package is being sent
        self.heal()

    def heal(self) -> None:
        self.package_status, self.package_headers, self.wrong = 200, {}, None

    def serve(self, *packages: bytes) -> None:
        for data in packages:
            self.packages[urls.package_url(CLOUD, read_theme_package(data).digest)] = data

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url)
        if url == CATALOG_URL:
            return httpx.Response(200, content=self.catalog)
        if self.during_package is not None:
            self.during_package()
        body = self.wrong if self.wrong is not None else self.packages.get(url, b"")
        return httpx.Response(self.package_status, content=body, headers=self.package_headers)

    def factory(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler), follow_redirects=True)

    def package_requests(self):
        return [str(request.url) for request in self.requests if str(request.url) != CATALOG_URL]


def _operator():
    user = UserContext(id="operator@local", email="operator@local", system_role="user",
                       raw_claims={"source": LOCAL_OPERATOR_CLAIM_SOURCE, "user_id": 1})
    return RequestContext(workspace_id=uuid.uuid4(), user=user, auth_type="anonymous")


def _never(*args, **kwargs):
    raise AssertionError("a Cloud version is never approved or sent for signing here")


@pytest.fixture
def env(cloud, monkeypatch):  # noqa: F811  (cloud: the Sync tests' fixture)
    """The Sync route over Postgres, the mock Automatos Cloud and a fake S3, the local operator calling."""
    s3, mock, sessions = FakeS3(), Cloud(), []
    monkeypatch.setattr(storage, "get_s3_client", lambda: s3)
    monkeypatch.setattr(serve, "get_s3_client", lambda: s3)
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: "owner")

    def database():
        session = cloud.new()
        sessions.append(session)
        try:
            yield session
        finally:
            session.close()

    app = FastAPI()
    app.include_router(sync_api.router)
    app.dependency_overrides[get_request_context_hybrid] = _operator
    app.dependency_overrides[get_db] = database
    app.dependency_overrides[sync_api.get_cloud_client_factory] = lambda: mock.factory
    return SimpleNamespace(cloud=cloud, s3=s3, mock=mock, sessions=sessions, client=TestClient(app),
                           theme=cloud.theme)


# --------------------------------------------------------------------------- helpers


def _add(env, *objects):
    session = env.cloud.new()
    session.add_all(objects)
    session.commit()
    session.close()


def _workspace(env):
    workspace_id = uuid.uuid4()
    session = env.cloud.new()
    session.execute(sa.text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), :n)"),
                    {"id": str(workspace_id), "n": env.cloud.publisher})
    session.commit()
    session.close()
    return workspace_id


def _install(env, package, pinned=UNPINNED, workspace_id=None):
    """An active install of ``package`` in a workspace of its own (or ``workspace_id``); its id."""
    row = ThemeInstallation(workspace_id=workspace_id or _workspace(env), theme_id=package.theme_id,
                            package_id=package.id, pinned_version=pinned)
    _add(env, row)
    return row.id


def _version(env, theme_id, version):
    session = env.cloud.new()
    row = session.query(ThemePackage).filter_by(theme_id=theme_id, version=version, owner_type="marketplace").one()
    session.close()
    return row


def _state(env, install_id):
    """``(package, previous, pending)`` of the install."""
    session = env.cloud.new()
    row = session.get(ThemeInstallation, install_id)
    session.close()
    return row.package_id, row.previous_package_id, row.pending_package_id


def _sync(env, *packages):
    env.mock.catalog = _catalog(*(_item(data) for data in packages))
    answer = env.client.post(SYNC)
    assert answer.status_code == 200, answer.text
    return answer.json()


def _keys(package):
    return {storage.theme_key(package.storage_prefix, path) for path in package.file_manifest}


def _installed(env, slug, *pins):
    """A Cloud theme synced at 1.0.0 and installed once per pin (``UNPINNED`` for none); 1.1.0 is served."""
    theme = env.theme(slug)
    old, new = _pkg(theme, OLD), _pkg(theme, NEW)
    _sync(env, old)
    first = _version(env, theme, OLD)
    env.mock.serve(new)
    return SimpleNamespace(theme=theme, old=old, new=new, first=first,
                           installs=[_install(env, first, pin) for pin in pins])


# --------------------------------------------------------------------------- the move


def test_an_unpinned_install_moves_to_the_synced_version_its_files_stored_first(env, monkeypatch):
    theme = _installed(env, "arcade", UNPINNED)
    stored_at_move, real_apply = [], theme_updates.apply_update

    def spy(db, workspace_id, install_id, package_id):
        stored_at_move.append(_keys(db.get(ThemePackage, package_id)) <= set(env.s3.objects))
        return real_apply(db, workspace_id, install_id, package_id)

    monkeypatch.setattr(theme_updates, "apply_update", spy)
    body = _sync(env, theme.old, theme.new)
    second = _version(env, theme.theme, NEW)
    assert body["counts"] == {"new": 0, "updated": 1, "unchanged": 1, "refused": 0}
    assert (body["updatesApplied"], body["updatesFailed"]) == (1, 0)
    assert _state(env, theme.installs[0]) == (second.id, theme.first.id, None)
    assert stored_at_move == [True]
    assert _keys(second) <= set(env.s3.objects) and not _keys(theme.first) & set(env.s3.objects)
    assert env.mock.package_requests() == [urls.package_url(CLOUD, second.digest)]
    again = _sync(env, theme.old, theme.new)
    assert (again["updatesApplied"], again["updatesFailed"]) == (0, 0)
    assert env.mock.package_requests() == [urls.package_url(CLOUD, second.digest)]


def test_no_transaction_is_open_while_automatos_cloud_sends_the_package(env):
    theme = _installed(env, "arcade", UNPINNED)
    open_during = []
    env.mock.during_package = lambda: open_during.append(env.sessions[-1].in_transaction())
    body = _sync(env, theme.old, theme.new)
    assert body["updatesApplied"] == 1
    assert open_during == [False]


def test_a_pinned_install_stays_on_its_version_with_the_new_one_offered(env):
    theme = _installed(env, "arcade", UNPINNED, OLD)
    body = _sync(env, theme.old, theme.new)
    second = _version(env, theme.theme, NEW)
    unpinned, pinned = theme.installs
    assert (body["updatesApplied"], body["updatesFailed"]) == (1, 0)
    assert _state(env, unpinned) == (second.id, theme.first.id, None)
    assert _state(env, pinned) == (theme.first.id, None, second.id)


def test_a_theme_whose_only_older_install_is_pinned_is_offered_the_version_and_nothing_is_fetched(env):
    theme = _installed(env, "ink", OLD)
    body = _sync(env, theme.old, theme.new)
    second = _version(env, theme.theme, NEW)
    assert (body["updatesApplied"], body["updatesFailed"]) == (0, 0)
    assert _state(env, theme.installs[0]) == (theme.first.id, None, second.id)
    assert env.mock.package_requests() == [] and env.s3.objects == {}


def test_an_install_unpinned_after_the_plan_was_read_is_only_offered_until_its_files_are_stored(env, monkeypatch):
    theme = _installed(env, "ink", OLD)
    real_offer, unpinned = cloud_updates.offer_listed_update, []

    def unpin_then_offer(db, package):
        if not unpinned:  # a person unpins the install between the plan and the offer
            session = env.cloud.new()
            session.get(ThemeInstallation, theme.installs[0]).pinned_version = None
            session.commit()
            session.close()
            unpinned.append(True)
        return real_offer(db, package)

    monkeypatch.setattr(cloud_updates, "offer_listed_update", unpin_then_offer)
    body = _sync(env, theme.old, theme.new)
    second = _version(env, theme.theme, NEW)
    assert (body["updatesApplied"], body["updatesFailed"]) == (0, 0)
    assert _state(env, theme.installs[0]) == (theme.first.id, None, second.id)
    assert env.mock.package_requests() == [] and env.s3.objects == {}
    again = _sync(env, theme.old, theme.new)
    assert (again["updatesApplied"], again["updatesFailed"]) == (1, 0)
    assert _state(env, theme.installs[0]) == (second.id, theme.first.id, None)
    assert _keys(second) <= set(env.s3.objects)


def _down(mock, theme):
    mock.package_status = 503


def _redirect(mock, theme):
    mock.package_status, mock.package_headers = 302, {"location": CLOUD + "/elsewhere/package.autotheme"}


def _another_version(mock, theme):
    mock.wrong = theme.old


@pytest.mark.parametrize("break_cloud", [_down, _redirect, _another_version], ids=["down", "redirect", "other"])
def test_a_failed_download_leaves_every_install_as_it_was_and_the_next_sync_applies_it(env, caplog, break_cloud):
    theme = _installed(env, "arcade", UNPINNED, OLD)
    break_cloud(env.mock, theme)
    with caplog.at_level(logging.WARNING, logger=cloud_updates.__name__):
        body = _sync(env, theme.old, theme.new)
    second = _version(env, theme.theme, NEW)
    assert body["counts"]["updated"] == 1
    assert (body["updatesApplied"], body["updatesFailed"]) == (0, 1)
    assert [_state(env, install) for install in theme.installs] == [(theme.first.id, None, None)] * 2
    assert env.s3.objects == {}
    assert env.mock.package_requests() == [urls.package_url(CLOUD, second.digest)]  # the redirect isn't followed
    logged = [record.getMessage() for record in caplog.records if record.name == cloud_updates.__name__]
    assert any(theme.theme in line and NEW in line for line in logged), logged

    env.mock.heal()
    again = _sync(env, theme.old, theme.new)
    assert again["counts"] == {"new": 0, "updated": 0, "unchanged": 2, "refused": 0}
    assert (again["updatesApplied"], again["updatesFailed"]) == (1, 0)
    unpinned, pinned = theme.installs
    assert _state(env, unpinned) == (second.id, theme.first.id, None)
    assert _state(env, pinned) == (theme.first.id, None, second.id)
    assert _keys(second) <= set(env.s3.objects)


def test_no_signature_request_is_written_and_review_approve_never_runs(env, monkeypatch):
    monkeypatch.setattr(review, "approve", _never)
    monkeypatch.setattr(signing, "request_signature", _never)
    theme = _installed(env, "arcade", UNPINNED)
    assert _sync(env, theme.old, theme.new)["updatesApplied"] == 1
    session = env.cloud.new()
    events = (
        session.query(ThemeReviewEvent.event)
        .join(ThemePackage, ThemePackage.id == ThemeReviewEvent.package_id)
        .filter(ThemePackage.theme_id == theme.theme)
        .all()
    )
    session.close()
    assert sorted(row.event for row in events) == ["approved", "approved"]  # the Sync's, one per Cloud row


# --------------------------------------------------------------------------- what never moves


def test_an_install_of_a_local_origin_listing_is_never_moved(env):
    theme = env.theme("own")
    old, new = _pkg(theme, OLD), _pkg(theme, NEW)
    local = _row(old)  # this install's own approved listing: source NULL
    _add(env, local)
    install = _install(env, local)
    env.mock.serve(new)
    body = _sync(env, new)
    assert body["counts"] == {"new": 0, "updated": 0, "unchanged": 0, "refused": 1}
    assert (body["updatesApplied"], body["updatesFailed"]) == (0, 0)
    assert _state(env, install) == (local.id, None, None)
    assert env.mock.package_requests() == []


def test_a_cloud_row_beside_a_local_origin_listing_never_moves_its_installs(env):
    """A Cloud version the Sync wrote before the operator listed their own: the theme is left whole."""
    theme = env.theme("both")
    old, new = _pkg(theme, OLD), _pkg(theme, NEW)
    local, synced = _row(old), _row(new, source="cloud")
    _add(env, local, synced)
    install = _install(env, local)
    env.mock.serve(new)
    session = env.cloud.new()
    assert theme not in cloud_updates.updatable_themes(session)
    cloud_updates.apply_cloud_updates(session, env.mock.factory)
    session.close()
    assert _state(env, install) == (local.id, None, None)
    assert env.mock.package_requests() == [] and env.s3.objects == {}


def test_an_install_of_a_workspaces_own_upload_is_never_moved(env):
    theme = env.theme("mine")
    old, new = _pkg(theme, OLD), _pkg(theme, NEW)
    workspace_id = _workspace(env)
    upload = _row(old, owner="workspace", workspace_id=workspace_id)
    _add(env, upload)
    install = _install(env, upload, workspace_id=workspace_id)
    env.mock.serve(new)
    body = _sync(env, new)
    assert body["counts"]["new"] == 1
    assert (body["updatesApplied"], body["updatesFailed"]) == (0, 0)
    assert _state(env, install) == (upload.id, None, None)
    assert env.mock.package_requests() == []


# --------------------------------------------------------------------------- one theme at a time


def _offer_fails_for(env, monkeypatch, slug):
    real_offer = cloud_updates.offer_listed_update

    def offer_then_fail(db, package):
        offered = real_offer(db, package)  # the offer is written, then the database fails
        if package.theme_id == env.theme(slug):
            raise OperationalError("UPDATE theme_installations", {}, Exception("the database went away"))
        return offered

    monkeypatch.setattr(cloud_updates, "offer_listed_update", offer_then_fail)


def _storing_breaks_for(env, monkeypatch, slug):
    real_ensure = cloud_updates.ensure_stored

    def ensure_or_break(db, package, client_factory=None):
        if package.theme_id == env.theme(slug):
            raise RuntimeError("an error nobody expected")
        return real_ensure(db, package, client_factory)

    monkeypatch.setattr(cloud_updates, "ensure_stored", ensure_or_break)


@pytest.mark.parametrize("break_alpha", [_offer_fails_for, _storing_breaks_for], ids=["database", "unexpected"])
def test_a_failure_in_one_themes_update_rolls_back_that_theme_alone(env, monkeypatch, break_alpha):
    slugs = ("alpha", "beta")  # updated in this order
    _sync(env, *(_pkg(env.theme(slug), OLD) for slug in slugs))
    firsts = [_version(env, env.theme(slug), OLD) for slug in slugs]
    installs = [_install(env, first) for first in firsts]
    newer = [_pkg(env.theme(slug), NEW) for slug in slugs]
    env.mock.serve(*newer)
    break_alpha(env, monkeypatch, "alpha")
    body = _sync(env, *(_pkg(env.theme(slug), OLD) for slug in slugs), *newer)
    seconds = [_version(env, env.theme(slug), NEW) for slug in slugs]
    assert (body["updatesApplied"], body["updatesFailed"]) == (1, 1)
    assert _state(env, installs[0]) == (firsts[0].id, None, None)  # the offer it wrote was rolled back
    assert _state(env, installs[1]) == (seconds[1].id, firsts[1].id, None)


def test_a_pass_that_waits_too_long_on_another_moves_nothing_and_the_next_sync_applies_it(env, monkeypatch):
    theme = _installed(env, "arcade", UNPINNED)
    monkeypatch.setattr(cloud_updates, "wait_seconds", lambda: 0.01)
    assert cloud_updates._UPDATE_LOCK.acquire(blocking=False)
    try:
        body = _sync(env, theme.old, theme.new)
    finally:
        cloud_updates._UPDATE_LOCK.release()
    assert body["counts"]["updated"] == 1
    assert (body["updatesApplied"], body["updatesFailed"]) == (0, 0)
    assert _state(env, theme.installs[0]) == (theme.first.id, None, None)
    assert env.mock.package_requests() == []
    again = _sync(env, theme.old, theme.new)
    assert again["updatesApplied"] == 1
    assert _state(env, theme.installs[0]) == (_version(env, theme.theme, NEW).id, theme.first.id, None)
