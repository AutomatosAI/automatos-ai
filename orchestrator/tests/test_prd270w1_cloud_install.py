"""PRD-270 US-270-04a (§5.4, K16) — installing a Cloud theme: its package fetched, checked and stored first.

The real routers (``api/marketplace_themes.py`` install, ``api/themes_install.py`` update and
``api/themes_files.py`` files) run on a mini FastAPI app over in-memory SQLite holding the theme tables,
as the local edition's operator, with the REAL gates. Cloud rows are what a Sync writes
(``services/marketplace_cloud/pull.py``): real ``.autotheme`` packages from ``tests/helpers_theme_publish.py``
signed with the RFC 8032 vector's seed, the keys file the test's own. Automatos Cloud is an
``httpx.MockTransport`` that records each request, behind a client built with ``follow_redirects=True``
(a redirect answered once proves the fetch turns them off itself); storage is a fake S3 that records
the order of its writes and answers NoSuchKey for a missing object. Proven:

* a stored Cloud package and a local-origin row install with no fetch;
* a Cloud row downloads once, from the configured origin's catalogue files path, stores every file
  (theme.json last), installs, and the files route serves them; a second install fetches nothing;
* oversize, a redirect, a non-200, a zip whose digest isn't the row's, a manifest of another id, bytes
  that aren't a package, and a signature no active key verifies are refused with their code: no file
  written, no install row;
* a failed storage write removes what it wrote; a storage read that fails, the Sync off, SaaS and a keys
  file that won't load are refused before anything is stored;
* an update to a Cloud version stores it first, and a failed download leaves the install where it was; an
  update naming another theme, a missing package, a local row or another workspace's install fetches nothing;
* a caller that waited on the same version's download finds it stored; one that waits too long is a 503;
  another version never waits; theme.json sorts after every path a package may hold (written last);
* the only URL ever requested is the one derived from ``AUTOMATOS_CLOUD_URL`` and the row's digest.
"""
from __future__ import annotations

import io
import json
import threading
import uuid
from types import SimpleNamespace

import httpx
import pytest
import sqlalchemy as sa
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from tests import helpers_theme_publish as hp  # first: it sets the database environment the imports below read

import api.marketplace_themes as listings_api
import api.themes_files as files_api
import api.themes_install as install_api
import core.auth.workspace_permission as permission_mod
from config import config
from core.auth import step_up
from core.auth.dependencies import RequestContext, UserContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.principal import LOCAL_OPERATOR_CLAIM_SOURCE
from core.database.database import get_db
from core.models.themes import (
    THEME_SOURCE_CLOUD, ThemeChoice, ThemeInstallation, ThemeLook, ThemeOptIn, ThemePackage, ThemePublisher,
    ThemeReviewEvent, ThemeRevocation, ThemeRevocationSync, ThemeWorkspaceSettings,
)
from services.marketplace_cloud import fetch, package as cloud_package, urls
from services.themes import listings, serve, signature_verify, storage
from services.themes.package import FILE_KINDS, read_theme_package

CLOUD = "https://cloud.example.test"
WS_A, WS_B = uuid.uuid4(), uuid.uuid4()
OPERATOR_PK = 1
THEME = hp.THEME_ID
LISTING = THEME.replace("/", ".")
TABLES = [model.__table__ for model in (
    ThemePackage, ThemeInstallation, ThemeChoice, ThemeLook, ThemeOptIn, ThemeWorkspaceSettings, ThemeRevocation,
    ThemePublisher, ThemeReviewEvent, ThemeRevocationSync,
)]


class _Missing(Exception):
    """What boto3 raises for a key that isn't there (``serve.MISSING_OBJECT_CODES``)."""

    response = {"Error": {"Code": "NoSuchKey"}}


class FakeS3:
    """Objects by key, the order they were written, what was deleted; ``fail_after`` writes, then every write raises."""

    def __init__(self) -> None:
        self.objects, self.order, self.deleted = {}, [], []
        self.fail_after = None
        self.read_error = None

    def put_object(self, Bucket, Key, Body, ContentType):  # noqa: N803 (boto3's names)
        if self.fail_after is not None and len(self.order) >= self.fail_after:
            raise ConnectionError("storage is down")
        self.objects[Key] = (Body, ContentType)
        self.order.append(Key)

    def get_object(self, Bucket, Key):  # noqa: N803
        if self.read_error is not None:
            raise self.read_error
        if Key not in self.objects:
            raise _Missing(Key)
        return {"Body": io.BytesIO(self.objects[Key][0])}

    def delete_object(self, Bucket, Key):  # noqa: N803
        self.deleted.append(Key)
        self.objects.pop(Key, None)


class Cloud:
    """Automatos Cloud as a mock transport: packages by URL, else ``status``; each request recorded."""

    def __init__(self) -> None:
        self.requests, self.served = [], {}
        self.status, self.headers, self.body = 200, {}, None

    def serve(self, data: bytes) -> None:
        self.served[urls.package_url(CLOUD, read_theme_package(data).digest)] = data

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = self.body if self.body is not None else self.served.get(str(request.url), b"")
        return httpx.Response(self.status, content=body, headers=self.headers)

    def factory(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler), follow_redirects=True)

    @property
    def urls(self):
        return [str(request.url) for request in self.requests]


def _operator(workspace_id=WS_A):
    user = UserContext(id="operator@local", email="operator@local", system_role="user",
                       raw_claims={"source": LOCAL_OPERATOR_CLAIM_SOURCE, "user_id": OPERATOR_PK})
    return RequestContext(workspace_id=workspace_id, user=user, auth_type="anonymous")


def _keys(tmp_path, retired=None):
    """The vector's SIGNER active; ``retired`` names another case kept as retired."""
    entries = [{"keyId": hp.SIGNER["keyId"], "spki": hp.SIGNER["spki"], "status": "active", "added": "2026-10-20"}]
    if retired is not None:
        entries.append({"keyId": retired["keyId"], "spki": retired["spki"], "status": "retired", "added": "2026-10-01"})
    path = tmp_path / "signing-keys.json"
    path.write_text(json.dumps({"schema": 1, "keys": entries}), encoding="utf-8")
    return path


@pytest.fixture
def env(monkeypatch, tmp_path):
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    ThemePackage.metadata.create_all(engine, tables=TABLES)
    session = sessionmaker(bind=engine)()
    s3, cloud = FakeS3(), Cloud()
    state = SimpleNamespace(db=session, ctx=_operator(), s3=s3, cloud=cloud, tmp=tmp_path)
    monkeypatch.setattr(storage, "get_s3_client", lambda: s3)
    monkeypatch.setattr(storage, "ensure_bucket", lambda bucket: None)
    monkeypatch.setattr(serve, "get_s3_client", lambda: s3)
    monkeypatch.setattr(fetch, "_client", cloud.factory)
    monkeypatch.setattr(signature_verify, "KEYS_PATH", _keys(tmp_path))
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: "owner")
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(config, "THEMES_MARKETPLACE", "on", raising=False)
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_URL", CLOUD)
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_TIMEOUT_SECONDS", 15)
    app = FastAPI()
    step_up.install_step_up_handler(app)
    for router in (listings_api.router, install_api.router, files_api.router):
        app.include_router(router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: state.ctx
    app.dependency_overrides[get_db] = lambda: session
    state.client = TestClient(app)
    try:
        yield state
    finally:
        session.close()
        engine.dispose()


def _pkg(version="1.0.0", theme_id=THEME, source=hp.SOURCE_BYTES):
    return hp.package(theme_id, version, source)


def _row(env, data, case=hp.SIGNER, source=THEME_SOURCE_CLOUD):
    """The Marketplace row a Sync writes for ``data`` (``source`` None: this install's own listing)."""
    checked = read_theme_package(data)
    theme_id, version = checked.manifest["id"], checked.manifest["version"]
    row = ThemePackage(
        id=uuid.uuid4(), theme_id=theme_id, version=version, owner_type="marketplace", workspace_id=None,
        digest=checked.digest, manifest=checked.manifest, file_manifest=checked.file_manifest,
        storage_prefix=storage.theme_prefix(theme_id, version), status="approved", reviewed=True,
        scan_verdict="not_run", signature=json.loads(hp.envelope(data, case))["signature"], source=source,
    )
    env.db.add(row)
    env.db.commit()
    return row


def _store(data):
    """Store ``data``'s files where its Marketplace row keeps them (what an earlier install did)."""
    checked = read_theme_package(data)
    storage.write_theme_files(storage.theme_prefix(checked.manifest["id"], checked.manifest["version"]), checked.files)


def _install(env, package_id=None):
    body = {"json": {"packageId": str(package_id)}} if package_id is not None else {}
    return env.client.post(f"/api/marketplace/themes/{LISTING}/install", **body)


def _installs(env):
    env.db.expire_all()
    return env.db.query(ThemeInstallation).all()


def _derived(row):
    return f"{CLOUD}/api/marketplace/catalog/files/{row.digest}/package.autotheme"


# --------------------------------------------------------------------------- installs


def test_a_stored_cloud_package_installs_with_no_fetch(env):
    data = _pkg()
    row = _row(env, data)
    _store(data)
    written = list(env.s3.order)
    resp = _install(env)
    assert resp.status_code == 201, resp.text
    assert env.cloud.requests == [] and env.s3.order == written
    assert [install.package_id for install in _installs(env)] == [row.id]


def test_a_local_origin_listing_installs_with_no_fetch_and_nothing_stored(env):
    row = _row(env, _pkg(), source=None)
    resp = _install(env)
    assert resp.status_code == 201, resp.text
    assert env.cloud.requests == [] and env.s3.objects == {}
    assert [install.package_id for install in _installs(env)] == [row.id]


def test_a_cloud_row_downloads_once_from_the_derived_url_stores_installs_and_is_served(env):
    data = _pkg()
    row = _row(env, data)
    env.cloud.serve(data)
    resp = _install(env)
    assert resp.status_code == 201, resp.text
    assert env.cloud.urls == [_derived(row)]
    sent = env.cloud.requests[0]
    assert sent.method == "GET" and sent.headers["accept"] == fetch.ACCEPT_PACKAGE
    assert not {"authorization", "cookie", "x-workspace-id"} & {name.lower() for name in sent.headers}
    checked = read_theme_package(data)
    assert sorted(env.s3.objects) == sorted(row.storage_prefix + path for path in checked.files)
    assert env.s3.order[-1] == row.storage_prefix + "theme.json"  # written last: stored means complete
    assert [install.package_id for install in _installs(env)] == [row.id]
    served = env.client.get(f"/api/themes/files/{row.digest}/theme.json")
    assert served.status_code == 200, served.text
    assert served.content == checked.files["theme.json"].data
    env.ctx = _operator(WS_B)
    again = _install(env)
    assert again.status_code == 201, again.text
    assert env.cloud.urls == [_derived(row)]  # the second install fetched nothing
    assert {install.workspace_id for install in _installs(env)} == {WS_A, WS_B}


def _oversize(env, row, data):
    env.cloud.body = b"\0" * (config.THEME_MAX_PACKAGE_MB * 1024 * 1024 + 1)


def _redirect(env, row, data):
    env.cloud.status, env.cloud.headers, env.cloud.body = 302, {"location": "https://other.example.test/x"}, b""


def _not_found(env, row, data):
    env.cloud.status, env.cloud.body = 404, b""


def _other_files(env, row, data):
    env.cloud.body = _pkg(source=b"export default { other: true }\n")


def _other_theme(env, row, data):
    env.cloud.body = _pkg(theme_id="acme-e2e/other")


def _not_a_zip(env, row, data):
    env.cloud.body = b"this is not a theme package"


@pytest.mark.parametrize("answer, status, code", [
    (_oversize, 413, cloud_package.PACKAGE_TOO_LARGE),
    (_redirect, 502, cloud_package.CLOUD_UNREACHABLE),
    (_not_found, 502, cloud_package.CLOUD_UNREACHABLE),
    (_other_files, 409, cloud_package.DIGEST_MISMATCH),
    (_other_theme, 409, cloud_package.OTHER_THEME),
    (_not_a_zip, 409, cloud_package.PACKAGE_REFUSED),
])
def test_a_refused_download_installs_nothing_and_stores_nothing(env, monkeypatch, answer, status, code):
    monkeypatch.setattr(config, "THEME_MAX_PACKAGE_MB", 1)
    data = _pkg()
    row = _row(env, data)
    answer(env, row, data)
    with pytest.raises(cloud_package.CloudPackageError) as refused:
        cloud_package.ensure_stored(env.db, row)
    assert (refused.value.status_code, refused.value.code) == (status, code)
    resp = _install(env)
    assert resp.status_code == status, resp.text
    assert resp.json()["detail"] == refused.value.message and THEME in refused.value.message
    assert env.cloud.urls == [_derived(row), _derived(row)]  # never the redirect's target, never twice per try
    assert env.s3.objects == {} and env.s3.order == []
    assert _installs(env) == []


# No key of the keys file made it: never listed (C-05c), so the route answers 404 and fetches nothing. A key
# retired since the Sync: still listed (a retired key's version keeps running), but no longer stored here.
@pytest.mark.parametrize("retired, route_status, fetches", [(None, 404, 1), (hp.OTHER_SIGNER, 409, 2)])
def test_a_signature_no_active_key_verifies_stores_nothing(env, monkeypatch, retired, route_status, fetches):
    monkeypatch.setattr(signature_verify, "KEYS_PATH", _keys(env.tmp, retired))
    data = _pkg()
    row = _row(env, data, case=hp.OTHER_SIGNER)
    env.cloud.serve(data)
    with pytest.raises(cloud_package.CloudPackageError) as refused:
        cloud_package.ensure_stored(env.db, row)
    assert (refused.value.status_code, refused.value.code) == (409, "SIGNATURE_BAD")
    assert env.cloud.urls == [_derived(row)] and env.s3.objects == {}
    resp = _install(env)
    assert resp.status_code == route_status, resp.text
    assert len(env.cloud.requests) == fetches and env.s3.objects == {} and _installs(env) == []


def test_a_failed_storage_write_removes_what_it_wrote_and_installs_nothing(env):
    data = _pkg()
    row = _row(env, data)
    env.cloud.serve(data)
    env.s3.fail_after = 2
    resp = _install(env)
    assert resp.status_code == 503, resp.text
    assert resp.json()["detail"] == cloud_package.STORAGE_DOWN
    assert env.s3.objects == {} and sorted(env.s3.deleted) == sorted(env.s3.order[:2])
    assert env.cloud.urls == [_derived(row)] and _installs(env) == []


def test_a_storage_read_that_fails_is_refused_before_any_fetch(env):
    row = _row(env, _pkg())
    env.s3.read_error = ConnectionError("storage is down")
    with pytest.raises(cloud_package.CloudPackageError) as refused:
        cloud_package.ensure_stored(env.db, row)
    assert (refused.value.status_code, refused.value.code) == (503, cloud_package.STORAGE_UNAVAILABLE)
    assert env.cloud.requests == [] and env.s3.order == []


@pytest.mark.parametrize("edition, url", [("local", ""), ("saas", CLOUD)])
def test_with_the_sync_off_or_on_saas_nothing_is_fetched(env, monkeypatch, edition, url):
    row = _row(env, _pkg())
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_URL", url)
    monkeypatch.setattr(config, "AUTH_EDITION", edition)
    with pytest.raises(cloud_package.CloudPackageError) as refused:
        cloud_package.ensure_stored(env.db, row)
    assert (refused.value.status_code, refused.value.code) == (409, cloud_package.CLOUD_OFF)
    assert env.cloud.requests == [] and env.s3.order == []


def test_a_keys_file_that_wont_load_stores_nothing(env, monkeypatch):
    data = _pkg()
    row = _row(env, data)
    env.cloud.serve(data)
    broken = env.tmp / "broken-keys.json"
    broken.write_text("not json", encoding="utf-8")
    monkeypatch.setattr(signature_verify, "KEYS_PATH", broken)
    with pytest.raises(cloud_package.CloudPackageError) as refused:
        cloud_package.ensure_stored(env.db, row)
    assert (refused.value.status_code, refused.value.code) == (503, cloud_package.SIGNING_KEYS_UNREADABLE)
    assert env.s3.order == []


def test_the_only_url_requested_is_derived_from_the_setting_and_the_rows_digest(env, monkeypatch):
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_URL", CLOUD + "/")  # a lone trailing slash is dropped
    data = _pkg()
    row = _row(env, data)
    env.cloud.serve(data)
    cloud_package.ensure_stored(env.db, row)
    cloud_package.ensure_stored(env.db, row)
    assert env.cloud.urls == [urls.package_url(CLOUD, row.digest)] == [_derived(row)]


# --------------------------------------------------------------------------- updates


def test_an_update_to_a_cloud_version_stores_it_first(env):
    first = _pkg("1.0.0")
    old = _row(env, first)
    _store(first)
    install_id = _install(env, old.id).json()["install"]["installId"]
    newer = _pkg("1.1.0")
    new = _row(env, newer)
    env.cloud.serve(newer)
    resp = env.client.post(f"/api/themes/installs/{install_id}/update", json={"packageId": str(new.id)})
    assert resp.status_code == 200, resp.text
    assert env.cloud.urls == [_derived(new)]
    assert new.storage_prefix + "theme.json" in env.s3.objects
    assert [(install.package_id, install.previous_package_id) for install in _installs(env)] == [(new.id, old.id)]


def test_a_failed_download_leaves_the_install_where_it_was(env):
    first = _pkg("1.0.0")
    old = _row(env, first)
    _store(first)
    install_id = _install(env, old.id).json()["install"]["installId"]
    new = _row(env, _pkg("1.1.0"))
    env.cloud.status = 404
    resp = env.client.post(f"/api/themes/installs/{install_id}/update", json={"packageId": str(new.id)})
    assert resp.status_code == 502, resp.text
    assert not any(key.startswith(new.storage_prefix) for key in env.s3.objects)
    assert [install.package_id for install in _installs(env)] == [old.id]


def _update(env, install_id, package_id):
    return env.client.post(f"/api/themes/installs/{install_id}/update", json={"packageId": str(package_id)})


def test_an_update_fetches_only_a_cloud_version_of_the_theme_this_workspaces_install_runs(env):
    old = _row(env, _pkg("1.0.0"), source=None)
    install_id = _install(env, old.id).json()["install"]["installId"]
    other_theme, newer = _row(env, _pkg(theme_id="acme-e2e/other")), _row(env, _pkg("1.1.0"))
    local = _row(env, _pkg("1.2.0"), source=None)
    assert _update(env, install_id, uuid.uuid4()).status_code == 404  # no such package
    assert _update(env, install_id, other_theme.id).status_code == 409  # another theme
    assert _update(env, uuid.uuid4(), newer.id).status_code == 404  # no such install
    env.ctx = _operator(WS_B)
    assert _update(env, install_id, newer.id).status_code == 404  # another workspace's install
    env.ctx = _operator(WS_A)
    assert _update(env, install_id, local.id).status_code == 200  # a local row: moved, nothing fetched
    assert env.cloud.requests == [] and env.s3.objects == {}
    assert [install.package_id for install in _installs(env)] == [local.id]


# --------------------------------------------------------------------------- one download of a version at a time


def test_a_caller_that_waited_on_the_same_versions_download_finds_it_stored(env, monkeypatch):
    data = _pkg()
    row = _row(env, data)
    checks, first_check, real = [], threading.Event(), cloud_package.is_stored

    def watched(package):
        stored = real(package)
        checks.append(stored)
        first_check.set()
        return stored

    monkeypatch.setattr(cloud_package, "is_stored", watched)
    with cloud_package._digest_lock(row.digest):  # another request's download of this version is running
        worker = threading.Thread(target=cloud_package.ensure_stored, args=(env.db, row))
        worker.start()
        assert first_check.wait(5)
        _store(data)  # what that download stored
    worker.join(5)
    assert not worker.is_alive()
    assert checks == [False, True] and env.cloud.requests == []


def test_a_caller_that_waits_past_the_other_download_is_answered_503(env, monkeypatch):
    row = _row(env, _pkg())
    monkeypatch.setattr(cloud_package, "wait_seconds", lambda: 0.01)
    with cloud_package._digest_lock(row.digest):
        with pytest.raises(cloud_package.CloudPackageError) as busy:
            cloud_package.ensure_stored(env.db, row)
    assert (busy.value.status_code, busy.value.code) == (503, cloud_package.DOWNLOAD_BUSY)
    assert env.cloud.requests == [] and env.s3.order == []


def test_another_versions_download_never_waits_for_this_one(env):
    first, second = _pkg("1.0.0"), _pkg("1.1.0")
    held, wanted = _row(env, first), _row(env, second)
    env.cloud.serve(second)
    with cloud_package._digest_lock(held.digest):
        cloud_package.ensure_stored(env.db, wanted)
    assert env.cloud.urls == [_derived(wanted)]


def _first_characters(pattern):
    """The first character of every path a ``FILE_KINDS`` pattern matches: its own, or each top-level branch's of
    a leading ``(?:…)`` group."""
    if not pattern.startswith("(?:"):
        return {pattern[0]}
    heads, depth, branch_starts = set(), 0, True
    for char in pattern[len("(?:"):]:
        if branch_starts:
            heads.add(char)
        branch_starts = char == "|" and depth == 0
        if char == "(":
            depth += 1
        elif char == ")" and depth == 0:
            break
        elif char == ")":
            depth -= 1
    return heads


def test_theme_json_sorts_after_every_other_path_a_package_may_hold():
    """``is_stored`` reads theme.json as the sign of a complete write: ``write_theme_files`` writes in path order."""
    for kind in FILE_KINDS:
        if kind.name == "manifest":
            continue
        heads = _first_characters(kind.pattern.pattern)
        assert heads and all(head.isalpha() and head < "theme.json"[0] for head in heads), kind.pattern.pattern
    assert _first_characters(r"(?:README\.md|CHANGELOG\.md|LICENSE(?:\.md|\.txt)?)") == {"R", "C", "L"}
    assert sorted(read_theme_package(_pkg()).files)[-1] == "theme.json"


# --------------------------------------------------------------------------- the version an install takes


def test_install_target_names_the_version_install_listing_installs(env):
    older, newest = _row(env, _pkg("1.0.0"), source=None), _row(env, _pkg("1.2.0"), source=None)
    assert listings.install_target(env.db, LISTING, None).id == newest.id
    assert listings.install_target(env.db, LISTING, older.id).id == older.id
    with pytest.raises(listings.ThemeInstallError) as missing:
        listings.install_target(env.db, "acme-e2e.nothing", None)
    assert missing.value.status_code == 404
    with pytest.raises(listings.ThemeInstallError) as unknown:
        listings.install_target(env.db, LISTING, uuid.uuid4())
    assert unknown.value.status_code == 404


def test_a_row_envelope_is_the_kits_form_with_the_key_that_verifies_it(env):
    data = _pkg()
    row = _row(env, data)
    assert json.loads(cloud_package.row_envelope(row)) == json.loads(hp.envelope(data))
