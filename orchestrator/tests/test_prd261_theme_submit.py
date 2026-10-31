"""PRD-261 C-01a — ``POST /api/marketplace/themes/submit``: a theme submitted to the Marketplace.

The router is mounted on a test app (C1-wire registers it in ``main.py``), next to A-07a's import
router for the private-upload-then-submit case. The tables run on SQLite (``create_all``;
``workspace_members`` and ``workspaces`` by their own DDL, as in ``test_prd261_theme_import.py``)
and a fake S3 client is patched into ``services/themes/storage.py``.

Callers: SaaS Clerk principals who signed in a minute ago (the step-up passes), with
``members:read`` stubbed granted and ``resolve_user_pk`` mapped from the Clerk id, so two authors
are two ``users.id`` values; and the local edition's anonymous operator.
"""
from __future__ import annotations

import inspect
import io
import sys
import time
from pathlib import Path
from uuid import uuid4

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
    ThemeInstallation, ThemePackage, ThemePublisher, ThemeReviewEvent, ThemeRevocation, ThemeWorkspaceSettings,
)
from services.themes import storage, submissions  # noqa: E402
from services.themes.package import THEME_IMPORT_ROUTE  # noqa: E402
from services.themes.llm_stage import DID_NOT_RUN  # noqa: E402
from services.themes.submissions import THEME_SUBMIT_ROUTE  # noqa: E402
from tests.fixtures.themes import build  # noqa: E402

HELLO_ID = "automatos-e2e/hello"
# A Marketplace submission must ship source/ (C-06p4); the hello fixture has none, so _hello adds one file.
SOURCE_FILE = "source/src/index.tsx"
SOURCE_BYTES = b"export default {}\n"
USER_PKS = {"user_1": 1, "user_2": 2, "admin_1": 9}
MEMBERS_DDL = (
    "CREATE TABLE workspace_members (id INTEGER PRIMARY KEY, workspace_id CHAR(32) NOT NULL, "
    "user_id INTEGER NOT NULL, role VARCHAR(50), invited_by INTEGER, invited_at DATETIME, "
    "joined_at DATETIME, is_active BOOLEAN)"
)
WORKSPACES_DDL = "CREATE TABLE workspaces (id CHAR(32) PRIMARY KEY, owner_id INTEGER)"
TABLES = (ThemePackage, ThemePublisher, ThemeReviewEvent, ThemeInstallation, ThemeWorkspaceSettings, ThemeRevocation)


class _FakeS3:
    """Objects by key; ``fail`` makes every write raise."""

    def __init__(self):
        self.objects = {}
        self.fail = False

    def put_object(self, Bucket, Key, Body, ContentType):  # noqa: N803 (boto3's names)
        if self.fail:
            raise ConnectionError("storage is down")
        self.objects[Key] = (Body, ContentType)

    def get_object(self, Bucket, Key):  # noqa: N803
        return {"Body": io.BytesIO(self.objects[Key][0])}

    def delete_object(self, Bucket, Key):  # noqa: N803
        self.objects.pop(Key, None)


def _user_pk(db, ctx):
    user = getattr(ctx, "user", None)
    return USER_PKS.get(getattr(user, "clerk_user_id", None))


@pytest.fixture
def env(monkeypatch):
    import api.admin_themes as admin_api
    import api.marketplace_themes_submit as submit_api
    import api.themes_import as themes_import

    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    ThemePackage.metadata.create_all(engine, tables=[model.__table__ for model in TABLES])
    with engine.begin() as conn:
        conn.execute(sa.text(MEMBERS_DDL))
        conn.execute(sa.text(WORKSPACES_DDL))
    fake = _FakeS3()
    monkeypatch.setattr(storage, "get_s3_client", lambda: fake)
    monkeypatch.setattr(storage, "ensure_bucket", lambda bucket: None)
    monkeypatch.setattr(submit_api, "resolve_user_pk", _user_pk)
    monkeypatch.setattr(themes_import, "resolve_user_pk", _user_pk)
    monkeypatch.setattr(admin_api, "resolve_user_pk", _user_pk)
    # The review stage (C-02a) runs inline by default: no key here, so no test calls the Anthropic API.
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(config, "THEMES_MARKETPLACE", "on", raising=False)
    monkeypatch.setattr(config, "THEMES_PRIVATE_UPLOAD", "on", raising=False)
    monkeypatch.setattr(config, submissions.ONE_OPEN_SUBMISSION_SETTING, False, raising=False)
    monkeypatch.setattr(workspace_permission, "workspace_permission_granted", lambda db, ctx, permission: True)
    from services.themes import settings as theme_settings

    monkeypatch.setattr(theme_settings, "workspace_permission_granted", lambda db, ctx, permission: True)
    yield type("Env", (), {"sessions": sessionmaker(bind=engine), "s3": fake})
    engine.dispose()


def _ctx(clerk_id="user_1", system_role="user", workspace_id=None, fresh=True):
    from core.auth.dependencies import RequestContext, UserContext

    claims = {"sub": clerk_id, "iat": int(time.time())}
    if fresh:
        claims["fva"] = [1, -1]
    user = UserContext(id=clerk_id, email=f"{clerk_id}@example.com", system_role=system_role,
                       clerk_user_id=clerk_id, raw_claims=claims)
    return RequestContext(workspace_id=workspace_id or uuid4(), user=user, auth_type="clerk")


def _local_ctx():
    from core.auth.dependencies import RequestContext, UserContext

    user = UserContext(id="local@example.com", email="local@example.com", system_role="user")
    return RequestContext(workspace_id=uuid4(), user=user, auth_type="anonymous")


def _client(env, ctx, stage=None):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.admin_themes import router as admin_router
    from api.marketplace_themes_submit import router, submission_review_stage
    from api.themes_import import router as import_router
    from core.auth.hybrid import get_request_context_hybrid
    from core.auth.step_up import install_step_up_handler
    from core.database.database import get_db

    app = FastAPI()
    app.include_router(router)
    app.include_router(import_router)
    app.include_router(admin_router)
    install_step_up_handler(app)

    def _db():
        session = env.sessions()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[get_request_context_hybrid] = lambda: ctx
    if stage is not None:
        app.dependency_overrides[submission_review_stage] = lambda: stage
    return TestClient(app)


def _submit(client, data: bytes):
    return client.post(THEME_SUBMIT_ROUTE, files={"file": ("hello.autotheme", data, "application/zip")})


def _hello(**fields) -> bytes:
    """The hello package with one source/ file; ``fields`` change its theme.json."""
    manifest = {build.MANIFEST: build.changed_manifest(fields)} if fields else {}
    return build.build_package(replace={SOURCE_FILE: SOURCE_BYTES, **manifest})


def _no_source(**fields) -> bytes:
    """The hello package as A-06a builds it: no source/."""
    if not fields:
        return build.build_package()
    return build.build_package(replace={build.MANIFEST: build.changed_manifest(fields)})


def _rows(env, model):
    session = env.sessions()
    try:
        return session.query(model).all()
    finally:
        session.close()


def _stage(verdict):
    calls = []
    budgets = []

    def stage(checked, package_bytes, token_budget=None):
        calls.append(sorted(checked.files))
        budgets.append(token_budget)
        return {"schema": "automatos-theme-scan/v1", "verdict": verdict, "checks": {}}

    stage.calls = calls
    stage.budgets = budgets
    return stage


def _marketplace_prefix(theme_id=HELLO_ID, version="1.0.0"):
    return f"themes/{theme_id}/{version}/"


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------


def test_the_route_is_a_plain_def_gated_by_members_read_then_the_switch_then_step_up():
    from api.marketplace_themes_submit import STEP_UP, require_marketplace, router, submit_theme

    (route,) = [r for r in router.routes if getattr(r, "path", None) == THEME_SUBMIT_ROUTE]
    assert route.methods == {"POST"} and route.status_code == 201
    assert not inspect.iscoroutinefunction(submit_theme)
    calls = [dep.call for dep in route.dependant.dependencies]
    marks = [getattr(call, workspace_permission.PERMISSION_MARKER_ATTR, None) for call in calls]
    assert marks[0] == "members:read"
    assert calls.index(require_marketplace) < calls.index(STEP_UP)


def test_a_caller_without_members_read_is_refused(env, monkeypatch):
    monkeypatch.setattr(workspace_permission, "workspace_permission_granted", lambda db, ctx, permission: False)
    assert _submit(_client(env, _ctx()), _hello()).status_code == 403
    assert _rows(env, ThemePackage) == []


# ---------------------------------------------------------------------------
# A good submission
# ---------------------------------------------------------------------------


def test_a_submission_is_a_pending_marketplace_row_with_its_stage_verdict(env):
    stage = _stage("review_required")
    response = _submit(_client(env, _ctx(), stage=stage), _hello())

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["duplicate"] is False and body["package"]["status"] == "pending"
    assert body["package"]["ownerType"] == "marketplace" and body["package"]["scanVerdict"] == "review_required"
    (row,) = _rows(env, ThemePackage)
    assert (row.owner_type, row.workspace_id, row.status, row.reviewed) == ("marketplace", None, "pending", False)
    assert row.scan_verdict == "review_required" and row.submitted_by == USER_PKS["user_1"]
    assert row.storage_prefix == _marketplace_prefix()
    assert set(env.s3.objects) == {_marketplace_prefix() + path for path in row.file_manifest}
    assert len(stage.calls) == 1
    events = sorted(_rows(env, ThemeReviewEvent), key=lambda event: event.event)
    assert [(e.event, e.to_status) for e in events] == [("scanned", "pending"), ("submitted", "pending")]
    assert events[0].detail == {"verdict": "review_required"}
    (publisher,) = _rows(env, ThemePublisher)
    assert (publisher.publisher, publisher.owner_user_id, publisher.reserved) == ("automatos-e2e", 1, False)


def test_by_default_the_c02a_stage_runs_on_dist_and_source_and_with_no_llm_key_a_person_may_approve(env):
    # C1-wire: submit's default stage is C-02a's, never A-07a's "not_run" report (approve refuses that).
    response = _submit(_client(env, _ctx()), _hello())
    assert response.status_code == 201, response.text
    (row,) = _rows(env, ThemePackage)
    llm = row.scan_report["checks"]["llm"]
    assert row.scan_verdict == "review_required" and row.scan_report["schema"] == "automatos-theme-scan/v2"
    assert llm["reason"].startswith(DID_NOT_RUN) and llm["read"] == "dist+source"

    admin = _client(env, _ctx("admin_1", system_role="admin"))
    note = {"summary": f"Outcome: pass\nPackage digest:  {row.digest}\nreproducible build", "outcome": "pass"}
    assert admin.post(f"/api/admin/themes/{row.id}/report-note", json=note).status_code == 200
    approved = admin.post(f"/api/admin/themes/{row.id}/approve")
    assert approved.status_code == 200, approved.text
    assert approved.json()["status"] == "approved" and approved.json()["signatureRequest"]["requestId"]


def test_the_same_bytes_again_return_the_existing_submission(env):
    client = _client(env, _ctx())
    first = _submit(client, _hello())
    again = _submit(client, _hello())

    assert first.status_code == 201 and again.status_code == 200
    assert again.json()["duplicate"] is True
    assert again.json()["package"]["id"] == first.json()["package"]["id"]
    assert len(_rows(env, ThemePackage)) == 1 and len(_rows(env, ThemeReviewEvent)) == 2


def test_the_same_bytes_again_never_run_the_review_stage_twice(env):
    stage = _stage("review_required")
    client = _client(env, _ctx(), stage=stage)
    assert _submit(client, _hello()).status_code == 201
    again = _submit(client, _hello())

    assert again.status_code == 200 and again.json()["package"]["scanVerdict"] == "review_required"
    assert len(stage.calls) == 1


def test_a_submission_overlapping_one_of_the_same_bytes_is_409_and_never_runs_the_stage(env):
    from api.marketplace_themes_submit import BUSY, CLAIM_SCOPE
    from services.themes.package import read_theme_package
    from services.themes.upload_claim import upload_claim

    stage = _stage("review_required")
    client = _client(env, _ctx(), stage=stage)
    session = env.sessions()
    try:
        with upload_claim(session, CLAIM_SCOPE, read_theme_package(_hello()).digest) as claimed:
            assert claimed
            overlapping = _submit(client, _hello())
    finally:
        session.close()

    assert overlapping.status_code == 409 and overlapping.json()["detail"] == BUSY
    assert stage.calls == [] and _rows(env, ThemePackage) == [] and env.s3.objects == {}
    assert _submit(client, _hello()).status_code == 201 and len(stage.calls) == 1


@pytest.mark.parametrize("edition", ["saas", "local"])
def test_the_llm_check_always_reads_dist_and_source_too_when_shipped(env, monkeypatch, edition):
    from api.marketplace_themes_submit import marketplace_review
    from services.themes.package import read_theme_package

    monkeypatch.setattr(config, "AUTH_EDITION", edition)
    data = _hello()
    llm = marketplace_review(read_theme_package(data), len(data))["checks"]["llm"]

    assert llm["read"] == "dist+source"
    assert "dist/entry.js" in llm["files"] and llm["files"][-1] == SOURCE_FILE


def test_the_submitter_sees_the_verdict_and_plain_checks_never_the_llm_check(env):
    response = _submit(_client(env, _ctx()), _hello())
    assert response.status_code == 201, response.text
    report = response.json()["package"]["scanReport"]
    (row,) = _rows(env, ThemePackage)
    assert "llm" in row.scan_report["checks"] and "summary" in row.scan_report  # the reviewer keeps it all
    assert report["verdict"] == row.scan_verdict and "llm" not in report["checks"] and "summary" not in report
    assert {"sizes", "cssScope", "looks", "uses", "clonedFrom"} <= set(report["checks"])
    again = _submit(_client(env, _ctx()), _hello())  # the duplicate answer hides it too
    assert again.status_code == 200 and "llm" not in again.json()["package"]["scanReport"]["checks"]


def test_a_private_upload_and_then_a_submit_of_the_same_bytes_both_succeed(env):
    data = _hello()
    client = _client(env, _ctx())
    upload = client.post(THEME_IMPORT_ROUTE, files={"file": ("hello.autotheme", data, "application/zip")})
    submit = _submit(client, data)

    assert upload.status_code == 201, upload.text
    assert submit.status_code == 201, submit.text
    rows = {row.owner_type: row for row in _rows(env, ThemePackage)}
    assert set(rows) == {"workspace", "marketplace"}
    assert rows["workspace"].digest == rows["marketplace"].digest
    assert rows["workspace"].storage_prefix != rows["marketplace"].storage_prefix


def test_the_local_operator_may_submit_and_owns_its_publisher(env, monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    client = _client(env, _local_ctx())
    assert _submit(client, _hello()).status_code == 201
    assert _submit(client, _hello(version="1.0.1")).status_code == 201
    (publisher,) = _rows(env, ThemePublisher)
    assert publisher.owner_user_id is None


# ---------------------------------------------------------------------------
# Publishers
# ---------------------------------------------------------------------------


def test_a_non_admin_submitting_under_the_automatos_publisher_is_403(env):
    stage = _stage("pass")
    response = _submit(_client(env, _ctx(), stage=stage), _hello(id="automatos/x"))

    assert response.status_code == 403 and '"automatos" publisher' in response.json()["detail"]
    assert _rows(env, ThemePackage) == [] and _rows(env, ThemePublisher) == [] and env.s3.objects == {}
    assert stage.calls == []  # refused before the review stage spends anything


def test_a_platform_admin_may_submit_under_the_automatos_publisher(env):
    response = _submit(_client(env, _ctx("admin_1", system_role="admin")), _hello(id="automatos/x"))

    assert response.status_code == 201, response.text
    (publisher,) = _rows(env, ThemePublisher)
    assert (publisher.publisher, publisher.reserved) == ("automatos", True)


def test_a_publisher_marked_reserved_takes_only_admins(env):
    session = env.sessions()
    session.add(ThemePublisher(publisher="automatos-e2e", owner_user_id=None, reserved=True))
    session.commit()
    session.close()
    assert _submit(_client(env, _ctx()), _hello()).status_code == 403
    assert _submit(_client(env, _ctx("admin_1", system_role="admin")), _hello()).status_code == 201


def test_an_id_under_another_authors_publisher_is_409(env):
    assert _submit(_client(env, _ctx("user_1")), _hello()).status_code == 201
    stage = _stage("pass")
    response = _submit(_client(env, _ctx("user_2"), stage=stage), _hello(version="1.0.1"))

    assert response.status_code == 409 and "belongs to another author" in response.json()["detail"]
    assert len(_rows(env, ThemePackage)) == 1 and stage.calls == []


def test_the_same_bytes_from_another_author_are_still_409(env):
    data = _hello()
    assert _submit(_client(env, _ctx("user_1")), data).status_code == 201
    response = _submit(_client(env, _ctx("user_2")), data)
    assert response.status_code == 409


def test_a_saas_caller_with_no_user_row_is_403(env):
    response = _submit(_client(env, _ctx("unknown")), _hello())
    assert response.status_code == 403 and response.json()["detail"] == submissions.NO_PERSON


# ---------------------------------------------------------------------------
# Versions and Q9
# ---------------------------------------------------------------------------


def test_the_same_version_with_other_files_is_409(env):
    client = _client(env, _ctx())
    assert _submit(client, _hello()).status_code == 201
    response = _submit(client, _hello(name="Hello again"))
    assert response.status_code == 409 and "Raise the version" in response.json()["detail"]


def test_a_version_lower_than_one_submitted_is_409(env):
    client = _client(env, _ctx())
    assert _submit(client, _hello(version="1.2.0")).status_code == 201
    response = _submit(client, _hello(version="1.1.0"))
    assert response.status_code == 409 and "1.2.0" in response.json()["detail"]


def test_one_open_submission_is_on_by_default_and_can_be_turned_off(env):
    assert submissions.ONE_OPEN_SUBMISSION_DEFAULT is True
    assert 'os.getenv("THEMES_ONE_OPEN_SUBMISSION", "true")' in (ORCH_ROOT / "config.py").read_text(encoding="utf-8")
    client = _client(env, _ctx())  # the env fixture turns it off
    assert _submit(client, _hello()).status_code == 201
    assert _submit(client, _hello(version="1.0.1")).status_code == 201


def test_one_open_submission_when_on_refuses_a_second_pending_one(env, monkeypatch):
    monkeypatch.setattr(config, submissions.ONE_OPEN_SUBMISSION_SETTING, "true", raising=False)
    client = _client(env, _ctx())
    assert _submit(client, _hello()).status_code == 201
    response = _submit(client, _hello(version="1.0.1"))
    assert response.status_code == 409 and "waiting for review" in response.json()["detail"]
    assert _submit(client, _hello()).status_code == 200  # the same bytes are not a second submission


# ---------------------------------------------------------------------------
# C-06p4: a submission ships source/ (QB-12)
# ---------------------------------------------------------------------------


def _spend_spies(monkeypatch):
    """Records each review slot, upload claim and token hold the route opens (all three stay real)."""
    import api.marketplace_themes_submit as submit_api
    from services.themes import review_limits

    opened = []
    real_slot, real_hold, real_claim = review_limits.review_slot, review_limits.token_hold, submit_api.upload_claim

    def slot():
        opened.append("slot")
        return real_slot()

    def hold(tokens_left):
        opened.append("hold")
        return real_hold(tokens_left)

    def claim(*args):
        opened.append("claim")
        return real_claim(*args)

    monkeypatch.setattr(review_limits, "review_slot", slot)
    monkeypatch.setattr(review_limits, "token_hold", hold)
    monkeypatch.setattr(submit_api, "upload_claim", claim)
    return opened


def _assert_source_required(response):
    from api.marketplace_themes_submit import SOURCE_REQUIRED, SOURCE_REQUIRED_CODE

    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert detail["message"] == SOURCE_REQUIRED
    assert detail["errors"] == [{"code": SOURCE_REQUIRED_CODE, "path": "source/", "message": SOURCE_REQUIRED}]


def _assert_nothing_spent_or_saved(env, opened):
    assert opened == []  # no review slot, no upload claim, no token hold
    assert _rows(env, ThemePackage) == [] and _rows(env, ThemePublisher) == [] and _rows(env, ThemeReviewEvent) == []
    assert env.s3.objects == {}


def test_a_package_with_no_source_is_422_source_required_before_the_stage_and_any_row(env, monkeypatch):
    opened = _spend_spies(monkeypatch)
    stage = _stage("pass")
    response = _submit(_client(env, _ctx(), stage=stage), _no_source())

    _assert_source_required(response)
    assert stage.calls == []  # the review stage (and its LLM check) never ran
    _assert_nothing_spent_or_saved(env, opened)


def test_with_no_source_the_default_stage_never_calls_the_llm(env, monkeypatch):
    from services.themes import llm_stage

    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "a-key-the-stage-would-use")
    monkeypatch.setattr(llm_stage, "_client_call", lambda: (_never_called, None))
    opened = _spend_spies(monkeypatch)
    response = _submit(_client(env, _ctx()), _no_source())

    _assert_source_required(response)
    _assert_nothing_spent_or_saved(env, opened)


def test_a_bare_source_folder_entry_is_no_source(env):
    stage = _stage("pass")
    response = _submit(_client(env, _ctx(), stage=stage), build.build_package(extra=[("source/", b"")]))

    _assert_source_required(response)
    assert stage.calls == [] and _rows(env, ThemePackage) == []


def test_the_package_checks_answer_before_the_source_rule(env):
    response = _submit(_client(env, _ctx()), _no_source(htmlClass="studio"))
    assert response.status_code == 400
    codes = [error["code"] for error in response.json()["detail"]["errors"]]
    assert "RESERVED_HTML_CLASS" in codes and "SOURCE_REQUIRED" not in codes


def test_with_source_the_same_package_is_submitted(env, monkeypatch):
    opened = _spend_spies(monkeypatch)
    stage = _stage("review_required")
    response = _submit(_client(env, _ctx(), stage=stage), _hello())

    assert response.status_code == 201, response.text
    assert SOURCE_FILE in stage.calls[0] and opened == ["slot", "claim", "hold"]


def test_a_private_import_with_no_source_is_unchanged(env):
    data = _no_source()
    client = _client(env, _ctx())
    upload = client.post(THEME_IMPORT_ROUTE, files={"file": ("hello.autotheme", data, "application/zip")})

    assert upload.status_code == 201, upload.text
    (row,) = _rows(env, ThemePackage)
    assert row.owner_type == "workspace" and not any(path.startswith("source/") for path in row.file_manifest)
    _assert_source_required(_submit(client, data))
    assert len(_rows(env, ThemePackage)) == 1


# ---------------------------------------------------------------------------
# Switch, step-up, storage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["off", "sometimes"])
def test_the_marketplace_switch_off_or_unknown_refuses_before_any_reverification(env, monkeypatch, value):
    monkeypatch.setattr(config, "THEMES_MARKETPLACE", value, raising=False)
    response = _submit(_client(env, _ctx(fresh=False)), _hello())
    assert response.status_code == 403 and response.json() == {"detail": submissions.MARKETPLACE_OFF}


def test_a_stale_sign_in_is_asked_to_step_up(env):
    response = _submit(_client(env, _ctx(fresh=False)), _hello())
    assert response.status_code == 403 and "clerk_error" in response.json()
    assert _rows(env, ThemePackage) == []


def test_a_package_the_checks_refuse_is_400_with_its_code_and_no_row(env):
    response = _submit(_client(env, _ctx()), _hello(htmlClass="studio"))
    assert response.status_code == 400
    assert "RESERVED_HTML_CLASS" in [error["code"] for error in response.json()["detail"]["errors"]]
    assert _rows(env, ThemePackage) == [] and env.s3.objects == {}


def test_a_store_failure_is_503_and_leaves_no_row(env):
    env.s3.fail = True
    response = _submit(_client(env, _ctx()), _hello())
    assert response.status_code == 503
    assert _rows(env, ThemePackage) == [] and _rows(env, ThemePublisher) == []
    assert _rows(env, ThemeReviewEvent) == []


# ---------------------------------------------------------------------------
# H1: what a submission may spend
# ---------------------------------------------------------------------------


def test_a_full_review_slot_is_503_with_retry_after_and_never_runs_the_stage(env, monkeypatch):
    from services.themes import review_limits

    monkeypatch.setattr(config, "THEME_REVIEW_MAX_CONCURRENT", 1, raising=False)
    stage = _stage("review_required")
    with review_limits.review_slot() as held:
        assert held
        response = _submit(_client(env, _ctx(), stage=stage), _hello())
    assert response.status_code == 503 and response.json()["detail"] == review_limits.REVIEWS_BUSY
    assert response.headers["Retry-After"] == str(review_limits.BUSY_RETRY_SECONDS)
    assert stage.calls == [] and _rows(env, ThemePackage) == []
    assert review_limits.stages_running() == 0
    assert _submit(_client(env, _ctx(), stage=stage), _hello()).status_code == 201
    assert review_limits.stages_running() == 0  # the slot is given back


def test_the_review_slot_is_taken_before_the_upload_claim(env, monkeypatch):
    import api.marketplace_themes_submit as submit_api
    from services.themes import review_limits

    order = []
    real_slot, real_claim = review_limits.review_slot, submit_api.upload_claim

    def slot():
        order.append("slot")
        return real_slot()

    def claim(*args):
        order.append("claim")
        return real_claim(*args)

    monkeypatch.setattr(review_limits, "review_slot", slot)
    monkeypatch.setattr(submit_api, "upload_claim", claim)
    assert _submit(_client(env, _ctx(), stage=_stage("pass")), _hello()).status_code == 201
    assert order == ["slot", "claim"]


def test_a_person_past_the_daily_submissions_is_429_before_the_stage(env, monkeypatch):
    from services.themes import review_limits

    monkeypatch.setattr(config, "THEME_SUBMISSIONS_PER_DAY", 2, raising=False)
    stage = _stage("review_required")
    client = _client(env, _ctx(), stage=stage)
    for version in ("1.0.0", "1.0.1"):
        assert _submit(client, _hello(version=version)).status_code == 201
    response = _submit(client, _hello(version="1.0.2"))
    assert response.status_code == 429
    assert response.json()["detail"] == review_limits.SUBMISSIONS_QUOTA.format(limit=2)
    assert len(stage.calls) == 2
    assert _submit(client, _hello(version="1.0.1")).status_code == 200  # the same bytes cost nothing
    other = _client(env, _ctx("user_2"), stage=stage)
    assert _submit(other, _hello(id="automatos-e2e-2/hello")).status_code == 201  # per person


def test_submissions_older_than_a_day_no_longer_count(env, monkeypatch):
    from datetime import datetime, timedelta, timezone

    monkeypatch.setattr(config, "THEME_SUBMISSIONS_PER_DAY", 1, raising=False)
    client = _client(env, _ctx(), stage=_stage("pass"))
    assert _submit(client, _hello()).status_code == 201
    assert _submit(client, _hello(version="1.0.1")).status_code == 429
    session = env.sessions()
    try:
        session.query(ThemePackage).update({"created_at": datetime.now(timezone.utc) - timedelta(hours=25)})
        session.commit()
    finally:
        session.close()
    assert _submit(client, _hello(version="1.0.1")).status_code == 201


def test_the_stage_gets_what_is_left_of_the_days_review_tokens(env, monkeypatch):
    monkeypatch.setattr(config, "THEME_REVIEW_DAILY_TOKENS", 5000, raising=False)
    session = env.sessions()
    try:
        session.add(ThemePackage(
            theme_id="acme/old", version="1.0.0", owner_type="workspace", workspace_id=uuid4(), digest="ab" * 32,
            manifest={}, file_manifest={}, storage_prefix="themes/x/", scan_verdict="pass",
            scan_report={"checks": {"llm": {"tokensUsed": 1200}}},
        ))
        session.commit()
    finally:
        session.close()
    stage = _stage("pass")
    assert _submit(_client(env, _ctx(), stage=stage), _hello()).status_code == 201
    assert [hold.budget for hold in stage.budgets] == [3800]


def test_with_the_days_tokens_spent_the_llm_check_never_runs_and_the_verdict_is_review_required(env, monkeypatch):
    from services.themes import llm_stage

    monkeypatch.setattr(config, "THEME_REVIEW_DAILY_TOKENS", 1, raising=False)
    monkeypatch.setattr(llm_stage, "_client_call", lambda: (_never_called, None))
    response = _submit(_client(env, _ctx()), _hello())
    assert response.status_code == 201, response.text
    (row,) = _rows(env, ThemePackage)
    llm = row.scan_report["checks"]["llm"]
    assert row.scan_verdict == "review_required" and llm["calls"] == 0
    assert llm["reason"] == "LLM stage did not run: daily budget reached."


def _never_called(model, text):
    raise AssertionError("the LLM was called although the day's budget was spent")
