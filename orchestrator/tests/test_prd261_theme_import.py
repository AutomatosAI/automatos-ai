"""PRD-261 A-07a — ``POST /api/themes/import``: a workspace's private theme upload.

The router is mounted on a test app (B2-wire registers it in ``main.py``). The tables run on
SQLite (``create_all``; ``workspace_members`` by its own DDL, below) and a fake S3 client is
patched into ``services/themes/storage.py``. Two workspaces, A and B, prove tenant isolation.

Callers: the local edition's anonymous principal (the workspace gate treats it as the owner),
and, for the SaaS cases, a Clerk principal whose workspace role check is stubbed to grant
``workspace:manage`` (the gate itself is asserted on the route).
"""
from __future__ import annotations

import inspect
import io
import sys
import time
from pathlib import Path
from uuid import UUID, uuid4

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

import core.models  # noqa: E402,F401  (registers every table the FKs name)
from config import config  # noqa: E402
from core.auth import workspace_permission  # noqa: E402
from core.models.themes import (  # noqa: E402
    THEME_SCAN_VERDICT_DEFAULT, ThemeInstallation, ThemePackage, ThemeWorkspaceSettings,
)
from core.workspaces.models import WorkspaceMember  # noqa: E402
from services.themes import records, storage  # noqa: E402
from services.themes import settings as theme_settings  # noqa: E402
from services.themes.package import THEME_IMPORT_ROUTE  # noqa: E402
from services.themes.scan import CONTRAST_NOT_RUN, VERDICT_NOT_RUN  # noqa: E402
from tests.fixtures.themes import build  # noqa: E402

MB = 1024 * 1024
HELLO_ID = "automatos-e2e/hello"
SWITCH = records.PRIVATE_UPLOAD_SETTING
# The member table's own DDL: its model's postgresql UUID column has NUMERIC affinity on SQLite,
# which can turn a hex id into a number; CHAR(32) holds the portable Uuid's hex as text.
MEMBERS_DDL = (
    "CREATE TABLE workspace_members (id INTEGER PRIMARY KEY, workspace_id CHAR(32) NOT NULL, "
    "user_id INTEGER NOT NULL, role VARCHAR(50), invited_by INTEGER, invited_at DATETIME, "
    "joined_at DATETIME, is_active BOOLEAN)"
)
# ``workspaces.owner_id`` names an owner who may have no member row (records.people_count).
WORKSPACES_DDL = "CREATE TABLE workspaces (id CHAR(32) PRIMARY KEY, owner_id INTEGER)"


class _FakeS3:
    """Objects by key in one bucket; ``fail_after`` writes succeed, then every write raises."""

    def __init__(self):
        self.objects = {}
        self.fail_after = None

    def put_object(self, Bucket, Key, Body, ContentType):  # noqa: N803 (boto3's names)
        if self.fail_after is not None and len(self.objects) >= self.fail_after:
            raise ConnectionError("storage is down")
        self.objects[Key] = (Body, ContentType)

    def get_object(self, Bucket, Key):  # noqa: N803
        return {"Body": io.BytesIO(self.objects[Key][0])}

    def delete_object(self, Bucket, Key):  # noqa: N803
        self.objects.pop(Key, None)


@pytest.fixture
def env(monkeypatch):
    import api.themes_import as themes_import

    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    tables = [ThemePackage.__table__, ThemeInstallation.__table__, ThemeWorkspaceSettings.__table__]
    ThemePackage.metadata.create_all(engine, tables=tables)
    with engine.begin() as conn:
        conn.execute(sa.text(MEMBERS_DDL))
        conn.execute(sa.text(WORKSPACES_DDL))
    fake = _FakeS3()
    monkeypatch.setattr(storage, "get_s3_client", lambda: fake)
    monkeypatch.setattr(storage, "ensure_bucket", lambda bucket: None)
    monkeypatch.setattr(themes_import, "resolve_user_pk", lambda db, ctx: None)
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(config, SWITCH, "auto", raising=False)
    yield type("Env", (), {"sessions": sessionmaker(bind=engine), "s3": fake})
    engine.dispose()


def _local_ctx(workspace_id):
    from core.auth.dependencies import RequestContext, UserContext

    user = UserContext(id="local@example.com", email="local@example.com", system_role="user")
    return RequestContext(workspace_id=workspace_id, user=user, auth_type="anonymous")


def _saas_ctx(workspace_id, system_role="user"):
    """A Clerk caller who signed in a minute ago, so the route's step-up (A-07c) passes."""
    from core.auth.dependencies import RequestContext, UserContext

    claims = {"sub": "user_1", "fva": [1, -1], "iat": int(time.time())}
    user = UserContext(id="user_1", email="a@example.com", system_role=system_role, clerk_user_id="user_1",
                       raw_claims=claims)
    return RequestContext(workspace_id=workspace_id, user=user, auth_type="clerk")


def _client(env, ctx):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.themes_import import router
    from core.auth.hybrid import get_request_context_hybrid
    from core.database.database import get_db

    app = FastAPI()
    app.include_router(router)

    def _db():
        session = env.sessions()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[get_request_context_hybrid] = lambda: ctx
    return TestClient(app)


def _saas(env, monkeypatch, workspace_id, system_role="user"):
    """A SaaS client whose caller holds workspace:manage in ``workspace_id`` (and so, by default, may install)."""
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(workspace_permission, "workspace_permission_granted", lambda db, ctx, permission: True)
    monkeypatch.setattr(theme_settings, "workspace_permission_granted", lambda db, ctx, permission: True)
    return _client(env, _saas_ctx(workspace_id, system_role))


def _upload(client, data: bytes):
    return client.post(THEME_IMPORT_ROUTE, files={"file": ("hello.autotheme", data, "application/zip")})


def _hello(**fields) -> bytes:
    if not fields:
        return build.build_package()
    return build.build_package(replace={build.MANIFEST: build.changed_manifest(fields)})


def _rows(env, model):
    session = env.sessions()
    try:
        return session.query(model).all()
    finally:
        session.close()


def _add(env, *rows):
    session = env.sessions()
    try:
        session.add_all(rows)
        session.commit()
    finally:
        session.close()


def _members(env, workspace_id, count):
    _add(env, *[WorkspaceMember(workspace_id=workspace_id, user_id=n + 1, role="member", is_active=True) for n in range(count)])


def _prefix(workspace_id, version="1.0.0", theme_id=HELLO_ID):
    return f"themes/_workspace/{workspace_id}/{theme_id}/{version}/"


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------


def test_the_route_is_a_plain_def_gated_by_workspace_manage():
    from api.themes_import import import_theme, router

    (route,) = [r for r in router.routes if getattr(r, "path", None) == THEME_IMPORT_ROUTE]
    assert route.methods == {"POST"} and route.status_code == 201
    assert not inspect.iscoroutinefunction(import_theme)
    marks = [getattr(dep.call, workspace_permission.PERMISSION_MARKER_ATTR, None) for dep in route.dependant.dependencies]
    assert "workspace:manage" in marks
    from api.themes_import import STEP_UP, require_private_uploads

    calls = [dep.call for dep in route.dependant.dependencies]
    assert calls.index(require_private_uploads) < calls.index(STEP_UP)  # L4: refuse before any reverification


def test_a_caller_without_workspace_manage_is_refused(env, monkeypatch):
    monkeypatch.setattr(workspace_permission, "workspace_permission_granted", lambda db, ctx, permission: False)
    response = _upload(_client(env, _saas_ctx(uuid4())), _hello())
    assert response.status_code == 403
    assert _rows(env, ThemePackage) == [] and env.s3.objects == {}


# ---------------------------------------------------------------------------
# A good upload
# ---------------------------------------------------------------------------


def test_a_good_package_is_201_stored_privately_and_installed(env):
    workspace_a = uuid4()
    response = _upload(_client(env, _local_ctx(workspace_a)), _hello())

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == "installed" and body["duplicate"] is False
    assert body["package"]["themeId"] == HELLO_ID and body["package"]["ownerType"] == "workspace"
    assert body["package"]["reviewed"] is False and body["package"]["scanVerdict"] == VERDICT_NOT_RUN
    (row,) = _rows(env, ThemePackage)
    assert (row.owner_type, row.workspace_id, row.reviewed) == ("workspace", workspace_a, False)
    assert row.scan_verdict == THEME_SCAN_VERDICT_DEFAULT and row.storage_prefix == _prefix(workspace_a)
    assert set(env.s3.objects) == {_prefix(workspace_a) + path for path in row.file_manifest}
    (install,) = _rows(env, ThemeInstallation)
    assert (install.workspace_id, install.package_id, install.pending_package_id) == (workspace_a, row.id, None)
    assert body["install"]["id"] == str(install.id) and body["install"]["packageId"] == str(row.id)
    assert install.is_default is False and install.uninstalled_at is None


def test_the_scan_report_holds_the_server_checks_and_the_stages_not_run(env):
    body = _upload(_client(env, _local_ctx(uuid4())), _hello()).json()
    report = body["package"]["scanReport"]
    checks = report["checks"]

    assert report["verdict"] == VERDICT_NOT_RUN
    assert checks["sizes"]["status"] == "pass" and checks["sizes"]["entryBytes"] > 0
    assert checks["sizes"]["limits"] == {"package": 10 * MB, "code": 3072 * 1024, "entry": 300 * 1024, "clip": 1024 * 1024}
    assert checks["cssScope"] == {"status": "pass", "path": "dist/theme.css"}
    assert checks["looks"] == {"status": "pass", "presets": ["default", "night"]}
    assert checks["contrast"]["status"] == "not_run" and checks["contrast"]["reason"] == CONTRAST_NOT_RUN
    assert [preset["id"] for preset in checks["contrast"]["presets"]] == ["default", "night"]
    assert checks["automated"]["status"] == "not_run" and checks["automated"]["reason"]


# ---------------------------------------------------------------------------
# Refusals: 400, 413, 503
# ---------------------------------------------------------------------------


def test_a_refused_package_is_400_with_the_codes_and_stores_nothing(env):
    response = _upload(_client(env, _local_ctx(uuid4())), build.build_package(extra=[("../x", b"x")]))

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert [error["code"] for error in detail["errors"]] == ["UNSAFE_PATH"]
    assert detail["message"] and all(error["message"] for error in detail["errors"])
    assert env.s3.objects == {} and _rows(env, ThemePackage) == [] and _rows(env, ThemeInstallation) == []


def test_a_manifest_the_validator_refuses_is_400_with_its_code(env):
    response = _upload(_client(env, _local_ctx(uuid4())), _hello(htmlClass="studio"))
    assert response.status_code == 400
    assert "RESERVED_HTML_CLASS" in [error["code"] for error in response.json()["detail"]["errors"]]
    assert _rows(env, ThemePackage) == []


def test_a_package_over_the_cap_is_413(env, monkeypatch):
    monkeypatch.setattr(config, "THEME_MAX_PACKAGE_MB", 1, raising=False)
    response = _upload(_client(env, _local_ctx(uuid4())), b"PK\x03\x04" + b"\x00" * (MB + 1))
    assert response.status_code == 413
    assert [error["code"] for error in response.json()["detail"]["errors"]] == ["PACKAGE_TOO_LARGE"]
    assert _rows(env, ThemePackage) == []


def test_a_store_failure_is_503_and_leaves_no_row_and_no_file(env, caplog):
    env.s3.fail_after = 2
    response = _upload(_client(env, _local_ctx(uuid4())), _hello())

    assert response.status_code == 503
    assert "Traceback" not in response.text and response.json()["detail"] == "The theme's files couldn't be stored. Try again."
    assert _rows(env, ThemePackage) == [] and _rows(env, ThemeInstallation) == []
    assert env.s3.objects == {}  # the two files written before the failure were removed
    assert "rolling its rows back" in caplog.text  # the records layer logs before it rolls back


# ---------------------------------------------------------------------------
# Tenancy: identical bytes, another workspace
# ---------------------------------------------------------------------------


def test_two_workspaces_uploading_identical_bytes_both_get_201_and_their_own_rows(env):
    workspace_a, workspace_b = uuid4(), uuid4()
    first = _upload(_client(env, _local_ctx(workspace_a)), _hello())
    second = _upload(_client(env, _local_ctx(workspace_b)), _hello())

    assert (first.status_code, second.status_code) == (201, 201)
    assert first.json()["package"]["digest"] == second.json()["package"]["digest"]
    assert first.json()["package"]["id"] != second.json()["package"]["id"]
    rows = _rows(env, ThemePackage)
    assert {row.workspace_id for row in rows} == {workspace_a, workspace_b}
    assert {row.storage_prefix for row in rows} == {_prefix(workspace_a), _prefix(workspace_b)}
    assert {install.workspace_id for install in _rows(env, ThemeInstallation)} == {workspace_a, workspace_b}


def test_another_workspace_never_finds_the_package(env):
    workspace_a, workspace_b = uuid4(), uuid4()
    body = _upload(_client(env, _local_ctx(workspace_a)), _hello()).json()
    package_id, digest = UUID(body["package"]["id"]), body["package"]["digest"]

    session = env.sessions()
    try:
        assert records.workspace_package(session, workspace_a, package_id) is not None
        assert records.workspace_package(session, workspace_b, package_id) is None
        assert records.package_by_digest(session, workspace_b, digest) is None
        assert records.active_installation(session, workspace_b, HELLO_ID) is None
    finally:
        session.close()


def test_a_newer_version_in_another_workspace_never_touches_this_install(env):
    workspace_a, workspace_b = uuid4(), uuid4()
    _upload(_client(env, _local_ctx(workspace_a)), _hello())
    response = _upload(_client(env, _local_ctx(workspace_b)), _hello(version="1.1.0"))

    assert response.status_code == 201 and response.json()["status"] == "installed"
    install_a = next(install for install in _rows(env, ThemeInstallation) if install.workspace_id == workspace_a)
    assert install_a.pending_package_id is None


def test_the_same_bytes_again_in_one_workspace_return_the_existing_package(env):
    client = _client(env, _local_ctx(uuid4()))
    first = _upload(client, _hello()).json()
    again = _upload(client, _hello())

    assert again.status_code == 200
    assert again.json()["duplicate"] is True and again.json()["package"]["id"] == first["package"]["id"]
    assert len(_rows(env, ThemePackage)) == 1 and len(_rows(env, ThemeInstallation)) == 1


# ---------------------------------------------------------------------------
# Versions: an update waits, never switches
# ---------------------------------------------------------------------------


def test_a_higher_version_is_a_pending_update_never_a_silent_switch(env):
    workspace_a = uuid4()
    client = _client(env, _local_ctx(workspace_a))
    first = _upload(client, _hello()).json()
    second = _upload(client, _hello(version="1.1.0"))

    assert second.status_code == 201 and second.json()["status"] == "update_available"
    new_id = second.json()["package"]["id"]
    (install,) = _rows(env, ThemeInstallation)
    assert str(install.package_id) == first["package"]["id"] and str(install.pending_package_id) == new_id
    assert second.json()["install"]["pendingPackageId"] == new_id
    assert len(_rows(env, ThemePackage)) == 2
    assert any(key.startswith(_prefix(workspace_a, "1.1.0")) for key in env.s3.objects)


def test_the_same_version_with_other_files_is_409(env):
    client = _client(env, _local_ctx(uuid4()))
    _upload(client, _hello())
    response = _upload(client, _hello(name="Hello, changed"))

    assert response.status_code == 409 and "Raise the version" in response.json()["detail"]
    assert len(_rows(env, ThemePackage)) == 1


def test_a_version_not_newer_than_the_installed_one_is_409(env):
    client = _client(env, _local_ctx(uuid4()))
    _upload(client, _hello(version="1.2.0"))
    response = _upload(client, _hello(version="1.1.0"))

    assert response.status_code == 409 and "higher version" in response.json()["detail"]
    assert len(_rows(env, ThemePackage)) == 1


def _stale_first_check(monkeypatch):
    """The losing upload's pre-insert check ran before the winner committed: it saw nothing."""
    real, calls = records._check_new_version, []

    def check(*args):
        calls.append(args)
        return None if len(calls) == 1 else real(*args)

    monkeypatch.setattr(records, "_check_new_version", check)


def test_a_race_on_the_same_version_with_other_files_names_the_clash_not_busy(env, monkeypatch):
    client = _client(env, _local_ctx(uuid4()))
    assert _upload(client, _hello()).status_code == 201  # the winner, committed first
    _stale_first_check(monkeypatch)
    response = _upload(client, _hello(name="Hello, changed"))  # the loser hits the version index

    assert response.status_code == 409 and "Raise the version" in response.json()["detail"]
    assert len(_rows(env, ThemePackage)) == 1 and len(_rows(env, ThemeInstallation)) == 1


def test_a_race_that_leaves_no_clash_still_asks_to_try_again(env, monkeypatch):
    import api.themes_import as themes_import

    client = _client(env, _local_ctx(uuid4()))
    assert _upload(client, _hello()).status_code == 201
    monkeypatch.setattr(records, "active_installation", lambda *args, **kwargs: None)  # read before the winner's install
    response = _upload(client, _hello(version="1.1.0"))  # the loser hits the active-install index

    assert response.status_code == 409 and response.json()["detail"] == themes_import.BUSY
    assert len(_rows(env, ThemePackage)) == 1


# ---------------------------------------------------------------------------
# Who may upload: the platform switch and the workspace setting (D1-b)
# ---------------------------------------------------------------------------


def test_auto_in_a_two_member_saas_workspace_is_403_until_the_admin_turns_it_on(env, monkeypatch):
    workspace_a = uuid4()
    _members(env, workspace_a, 2)
    client = _saas(env, monkeypatch, workspace_a)

    refused = _upload(client, _hello())
    assert refused.status_code == 403 and refused.json()["detail"] == records.DEFAULT_OFF
    assert _rows(env, ThemePackage) == []

    _add(env, ThemeWorkspaceSettings(workspace_id=workspace_a, private_uploads="on"))
    assert _upload(client, _hello()).status_code == 201


def test_auto_in_a_single_member_saas_workspace_allows_uploads(env, monkeypatch):
    workspace_a = uuid4()
    _members(env, workspace_a, 1)
    assert _upload(_saas(env, monkeypatch, workspace_a), _hello()).status_code == 201


def test_auto_in_the_local_edition_allows_uploads_whatever_the_member_count(env):
    workspace_a = uuid4()
    _members(env, workspace_a, 3)
    assert _upload(_client(env, _local_ctx(workspace_a)), _hello()).status_code == 201


def test_a_workspace_that_turned_uploads_off_is_403(env):
    workspace_a = uuid4()
    _add(env, ThemeWorkspaceSettings(workspace_id=workspace_a, private_uploads="off"))
    response = _upload(_client(env, _local_ctx(workspace_a)), _hello())
    assert response.status_code == 403 and response.json()["detail"] == records.WORKSPACE_OFF


def test_off_is_a_ceiling_the_workspace_setting_cannot_lift(env, monkeypatch):
    workspace_a = uuid4()
    monkeypatch.setattr(config, SWITCH, "off", raising=False)
    _add(env, ThemeWorkspaceSettings(workspace_id=workspace_a, private_uploads="on"))
    response = _upload(_client(env, _local_ctx(workspace_a)), _hello())
    assert response.status_code == 403 and response.json()["detail"] == records.PLATFORM_OFF


def test_an_unknown_switch_value_fails_closed(env, monkeypatch):
    monkeypatch.setattr(config, SWITCH, "sometimes", raising=False)
    response = _upload(_client(env, _local_ctx(uuid4())), _hello())
    assert response.status_code == 403 and response.json()["detail"] == records.PLATFORM_OFF


def test_a_refused_upload_is_never_asked_to_step_up(env, monkeypatch):
    # L4: the private-upload gate runs before the step-up, so a caller who can't upload anyway
    # gets the plain reason, never a reverification prompt.
    from core.auth import step_up
    from core.auth.dependencies import RequestContext, UserContext

    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(workspace_permission, "workspace_permission_granted", lambda db, ctx, permission: True)
    monkeypatch.setattr(theme_settings, "workspace_permission_granted", lambda db, ctx, permission: True)
    monkeypatch.setattr(config, SWITCH, "off", raising=False)
    stale = UserContext(id="user_1", email="a@example.com", system_role="user", clerk_user_id="user_1",
                        raw_claims={"sub": "user_1", "iat": int(time.time())})
    client = _client(env, RequestContext(workspace_id=uuid4(), user=stale, auth_type="clerk"))
    step_up.install_step_up_handler(client.app)
    refused = _upload(client, _hello())
    assert refused.status_code == 403 and refused.json() == {"detail": records.PLATFORM_OFF}
    monkeypatch.setattr(config, SWITCH, "on", raising=False)
    asked = _upload(client, _hello())
    assert asked.status_code == 403 and "clerk_error" in asked.json()
    assert _rows(env, ThemePackage) == []


def test_admins_lets_only_a_platform_admin_upload(env, monkeypatch):
    monkeypatch.setattr(config, SWITCH, "admins", raising=False)
    refused = _upload(_saas(env, monkeypatch, uuid4()), _hello())
    assert refused.status_code == 403 and refused.json()["detail"] == records.ADMINS_ONLY
    assert _upload(_saas(env, monkeypatch, uuid4(), system_role="admin"), _hello()).status_code == 201


def test_on_defers_to_the_workspace_and_defaults_on(env, monkeypatch):
    workspace_a = uuid4()
    monkeypatch.setattr(config, SWITCH, "on", raising=False)
    _members(env, workspace_a, 2)
    assert _upload(_saas(env, monkeypatch, workspace_a), _hello()).status_code == 201


# ---------------------------------------------------------------------------
# Reserved publishers (carried from the W-B1 seam review)
# ---------------------------------------------------------------------------


def test_a_non_admin_uploading_under_the_automatos_publisher_is_refused(env, monkeypatch):
    workspace_a = uuid4()
    _members(env, workspace_a, 1)
    response = _upload(_saas(env, monkeypatch, workspace_a), _hello(id="automatos/play"))

    assert response.status_code == 403 and '"automatos" publisher' in response.json()["detail"]
    assert _rows(env, ThemePackage) == [] and env.s3.objects == {}


def test_a_platform_admin_may_upload_under_the_automatos_publisher(env, monkeypatch):
    workspace_a = uuid4()
    _members(env, workspace_a, 1)
    response = _upload(_saas(env, monkeypatch, workspace_a, system_role="admin"), _hello(id="automatos/play"))

    assert response.status_code == 201, response.text
    assert response.json()["package"]["themeId"] == "automatos/play"
    assert all(key.startswith(_prefix(workspace_a, theme_id="automatos/play")) for key in env.s3.objects)


def test_a_publisher_that_only_starts_like_a_reserved_one_is_not_reserved():
    assert records.reserved_publisher_refusal("automatos/play", None) is not None
    assert records.reserved_publisher_refusal("automatos-e2e/hello", None) is None


# ---------------------------------------------------------------------------
# Storage keys
# ---------------------------------------------------------------------------


def test_storage_keys_stay_under_the_workspace_prefix():
    workspace_a = uuid4()
    assert storage.theme_prefix(HELLO_ID, "1.0.0", workspace_a) == _prefix(workspace_a)
    assert storage.theme_prefix(HELLO_ID, "1.0.0") == f"themes/{HELLO_ID}/1.0.0/"
    for bad_id, bad_version in (("../x", "1.0.0"), ("Hello/x", "1.0.0"), (HELLO_ID, "1.0"), (HELLO_ID, "../1")):
        with pytest.raises(ValueError):
            storage.theme_prefix(bad_id, bad_version, workspace_a)
    for bad_path in ("../x", "a//b", "/abs", "a\\b", "./a", ""):
        with pytest.raises(ValueError):
            storage.theme_key(_prefix(workspace_a), bad_path)
