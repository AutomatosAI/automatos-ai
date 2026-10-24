"""PRD-261 C-03c — ``GET /api/admin/themes/{packageId}/review-package``: the kit review job's link.

The real router (``api/admin_themes.py``) runs on a mini FastAPI app over SQLite with a fake S3
client patched into the three places the route reaches storage (the stored-file reader, the
writer and the presigner). The hello fixture package is stored as a Marketplace version. Pins
(plan §2.6 C-03c):

* an admin gets ``{url, digest, expiresAt}``: a presigned GET for 15 minutes of
  ``themes/_review/<digest>.autotheme``, a zip rebuilt from the stored files that
  ``read_theme_package`` reads with the row's digest; the same version gives the same bytes;
* the URL passes theme-review.yml's input check (``https://``, a 64-hex digest, the inputs under
  GitHub's dispatch limit); a link it would refuse is 503;
* a workspace's private upload and a missing row are 404; a non-admin is 403 before any storage
  call; a missing or changed stored file, or a file list that doesn't add up, is 409 with a plain
  message; storage failures are 503;
* the route is a plain ``def`` GET with no step-up, and checks ``caller_is_admin`` first.
"""
from __future__ import annotations

import inspect
import io
import os
import re
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import sqlalchemy as sa  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.routing import APIRoute  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import api.admin_themes as admin_api  # noqa: E402
import core.models  # noqa: E402,F401  (registers every table the FKs name)
from config import config  # noqa: E402
from core.auth import step_up  # noqa: E402
from core.auth.dependencies import RequestContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.models.themes import ThemePackage  # noqa: E402
from services.themes import review_package, serve, storage  # noqa: E402
from services.themes.contract import echo  # noqa: E402
from services.themes.package import read_theme_package  # noqa: E402
from tests.fixtures.themes import build  # noqa: E402
from tests.test_prd261_admin_themes import _ctx  # noqa: E402  (its caller helper; none of its tests)

ROUTE = "/api/admin/themes/{package_id}/review-package"
PUBLIC_ENDPOINT = "https://automatos-marketplace.s3.eu-west-1.amazonaws.com/"
# theme-review.yml's "Download the package" step: `case "$PACKAGE_URL" in https://*)` and this digest pattern.
JOB_URL_PREFIX = "https://"
JOB_DIGEST = re.compile(r"[0-9a-f]{64}")


class _Missing(Exception):
    """What boto3 raises for a key that isn't there (``serve.MISSING_OBJECT_CODES``)."""

    response = {"Error": {"Code": "NoSuchKey"}}


class _FakeS3:
    """Objects by key; the presigner records its calls and answers a URL under ``endpoint``."""

    def __init__(self):
        self.objects, self.presigned, self.endpoint = {}, [], PUBLIC_ENDPOINT
        self.read_error = self.write_error = None

    def put_object(self, Bucket, Key, Body, ContentType):  # noqa: N803 (boto3's names)
        if self.write_error is not None:
            raise self.write_error
        self.objects[Key] = (Body, ContentType)

    def get_object(self, Bucket, Key):  # noqa: N803
        if self.read_error is not None:
            raise self.read_error
        if Key not in self.objects:
            raise _Missing(Key)
        return {"Body": io.BytesIO(self.objects[Key][0])}

    def delete_object(self, Bucket, Key):  # noqa: N803
        self.objects.pop(Key, None)

    def generate_presigned_url(self, ClientMethod, Params, ExpiresIn):  # noqa: N803
        self.presigned.append((ClientMethod, dict(Params), ExpiresIn))
        return f"{self.endpoint}{Params['Key']}?X-Amz-Expires={ExpiresIn}&X-Amz-Signature={'ab' * 32}"


@pytest.fixture
def api(monkeypatch):
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    ThemePackage.metadata.create_all(engine, tables=[ThemePackage.__table__])
    fake = _FakeS3()
    for module in (serve, storage):
        monkeypatch.setattr(module, "get_s3_client", lambda: fake)
    monkeypatch.setattr(storage, "ensure_bucket", lambda bucket: None)
    monkeypatch.setattr(review_package, "get_public_s3_client", lambda: fake)
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    session = sessionmaker(bind=engine)()
    state = SimpleNamespace(db=session, ctx=_ctx(), s3=fake)
    app = FastAPI()
    app.include_router(admin_api.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: state.ctx
    app.dependency_overrides[get_db] = lambda: session
    state.client = TestClient(app)
    try:
        yield state
    finally:
        session.close()
        engine.dispose()


def _version(api, workspace_id=None):
    """The hello package stored as a Marketplace version (or a workspace's upload), files and row."""
    checked = read_theme_package(build.build_package())
    manifest = checked.manifest
    prefix = storage.theme_prefix(manifest["id"], manifest["version"], workspace_id)
    storage.write_theme_files(prefix, checked.files)
    row = ThemePackage(
        theme_id=manifest["id"], version=manifest["version"], digest=checked.digest, manifest=manifest,
        file_manifest=checked.file_manifest, storage_prefix=prefix, status="pending",
        owner_type="workspace" if workspace_id else "marketplace", workspace_id=workspace_id,
    )
    api.db.add(row)
    api.db.commit()
    return row


def _get(api, package_id):
    return api.client.get(ROUTE.format(package_id=package_id))


def _review_copy(api, row):
    return api.s3.objects.get(review_package.review_key(row.digest))


# --- the link -------------------------------------------------------------------------------


def test_an_admin_gets_a_15_minute_link_to_the_rebuilt_package(api):
    row = _version(api)
    before = datetime.now(timezone.utc)
    response = _get(api, row.id)
    after = datetime.now(timezone.utc)

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"url", "digest", "expiresAt"} and body["digest"] == row.digest
    key = f"themes/_review/{row.digest}.autotheme"
    assert review_package.review_key(row.digest) == key
    assert api.s3.presigned == [("get_object", {"Bucket": config.MARKETPLACE_S3_BUCKET, "Key": key}, 15 * 60)]
    assert body["url"].startswith(PUBLIC_ENDPOINT + key)
    expires = datetime.fromisoformat(body["expiresAt"])
    assert before + timedelta(minutes=15) <= expires <= after + timedelta(minutes=15)
    data, content_type = _review_copy(api, row)
    assert content_type == "application/zip"
    rebuilt = read_theme_package(data)
    assert rebuilt.digest == row.digest and rebuilt.file_manifest == row.file_manifest


def test_the_url_passes_the_review_jobs_input_check(api):
    row = _version(api)
    body = _get(api, row.id).json()
    assert body["url"].startswith(JOB_URL_PREFIX) and JOB_DIGEST.fullmatch(body["digest"])
    assert len(body["url"]) + len(body["digest"]) <= review_package.DISPATCH_INPUTS_MAX_CHARS == 65535
    assert review_package.job_accepts(body["url"], body["digest"])


def test_the_same_version_rebuilds_to_the_same_bytes(api):
    row = _version(api)
    assert _get(api, row.id).status_code == 200
    first = _review_copy(api, row)
    assert _get(api, row.id).status_code == 200
    assert _review_copy(api, row) == first
    assert review_package.rebuilt_package(row) == first[0]


@pytest.mark.parametrize("endpoint", ["http://localhost:9000/", "https://example.com/" + "x" * 65535])
def test_a_link_the_review_job_would_refuse_is_503(api, endpoint):
    row = _version(api)
    api.s3.endpoint = endpoint
    response = _get(api, row.id)
    assert response.status_code == 503 and response.json()["detail"] == review_package.LINK_UNUSABLE
    assert "localhost" not in response.text  # the link itself is never answered


# --- refusals -------------------------------------------------------------------------------


def test_a_workspace_upload_and_a_missing_row_are_404(api):
    private = _version(api, workspace_id=uuid.uuid4())
    for package_id in (private.id, uuid.uuid4()):
        response = _get(api, package_id)
        assert response.status_code == 404, response.text
    assert api.s3.presigned == [] and _review_copy(api, private) is None


@pytest.mark.parametrize("signed_in", [True, False])
def test_a_non_admin_is_403_before_any_storage_call(api, signed_in):
    row = _version(api)
    anonymous = RequestContext(workspace_id=uuid.uuid4(), user=None, auth_type="anonymous")
    api.ctx = _ctx(role="user", clerk="clerk-8") if signed_in else anonymous
    response = _get(api, row.id)
    assert response.status_code == 403 and response.json()["detail"] == admin_api.ADMIN_REQUIRED
    assert api.s3.presigned == [] and _review_copy(api, row) is None


def test_a_missing_stored_file_is_409_with_a_plain_message(api):
    row = _version(api)
    gone = sorted(row.file_manifest)[0]
    api.s3.delete_object(Bucket=config.MARKETPLACE_S3_BUCKET, Key=row.storage_prefix + gone)
    response = _get(api, row.id)
    assert response.status_code == 409
    assert response.json()["detail"] == review_package.FILE_MISSING.format(path=echo(gone))
    assert api.s3.presigned == [] and _review_copy(api, row) is None


def test_a_stored_file_that_changed_is_409(api):
    row = _version(api)
    changed = sorted(row.file_manifest)[0]
    api.s3.put_object(Bucket=config.MARKETPLACE_S3_BUCKET, Key=row.storage_prefix + changed, Body=b"x",
                      ContentType="text/plain")
    response = _get(api, row.id)
    assert response.status_code == 409
    assert response.json()["detail"] == review_package.FILE_CHANGED.format(path=echo(changed))


def test_a_file_list_that_does_not_add_up_to_the_digest_is_409(api):
    row = _version(api)
    row.digest = "cd" * 32
    api.db.commit()
    response = _get(api, row.id)
    assert response.status_code == 409 and response.json()["detail"] == review_package.FILE_LIST_BROKEN


def test_storage_that_fails_is_503(api):
    row = _version(api)
    api.s3.read_error = ConnectionError("storage is down")
    assert _get(api, row.id).json() == {"detail": review_package.STORAGE_DOWN}
    api.s3.read_error, api.s3.write_error = None, ConnectionError("storage is down")
    response = _get(api, row.id)
    assert response.status_code == 503 and response.json()["detail"] == review_package.STORAGE_DOWN
    assert api.s3.presigned == []


# --- the route ------------------------------------------------------------------------------


def test_the_route_is_a_plain_def_get_with_no_step_up_that_checks_admin_first():
    (route,) = [r for r in admin_api.router.routes if isinstance(r, APIRoute) and r.path == ROUTE]
    assert route.methods == {"GET"} and not inspect.iscoroutinefunction(route.endpoint)
    calls = [dep.call for dep in route.dependant.dependencies]
    assert not [call for call in calls if getattr(call, step_up.STEP_UP_MARKER_ATTR, None) is not None]
    body = inspect.getsource(route.endpoint).split('"""')[-1]
    first = next(line.strip() for line in body.splitlines() if line.strip() and not line.strip().startswith(")"))
    assert first == "_require_admin(ctx)"
