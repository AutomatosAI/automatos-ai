"""PRD-251 Wave 2, US-202 (S3.4, D9) — media hosting: presigned inline links.

* Links (the storage client mocked): ``media_urls`` mints presigned GETs served
  inline, with the file's own content type (video/mp4, image/jpeg, image/png)
  and the TTL from config. A post's media (a render's file records, or plain
  Deliverable ids) resolves through the ``deliverables`` rows of the post's own
  workspace: another workspace's, a deleted, a non-object-storage or a
  foreign-keyed Deliverable has no link and says why. The approval view's links
  are minted through the PUBLIC client, and a post with no stored file never
  touches storage.
* Public storage: without it ``media_public_url_available()`` is false and a
  platform link raises NeedsPublicStorage; AWS S3 links the documents bucket
  itself; with SOCIALS_PUBLIC_MEDIA_BUCKET set the object is copied into that
  bucket under social-media/{workspace}/{post}/{file} and linked from there.
* The route: GET /api/socials/posts/{post_id}/media answers the links for the
  caller's post, 404 for another workspace's post, [] for a post with no media,
  503 when there is no storage to link to. It is a plain ``def``, in the
  committed manifest, and apiClient calls it.
* MinIO, over real HTTP: the orchestrator-tests job starts a pinned MinIO
  (.github/workflows/test.yml). The committed MP4 goes up through the storage
  factory, media_urls mints its links, and a HEAD answers 200 with video/mp4 and
  inline; a Range GET answers 206 with exactly the first 100 bytes and a
  Content-Range. The public bucket's copy is served the same way. These FAIL,
  never skip, when CI=true and SOCIALS_TEST_S3_ENDPOINT is unset.
* Guards: modules/socials builds no storage client and never names the
  generated-images route; the fixture is the generator's output, under 100 KB;
  the self-hosting docs say what needs public storage.
"""
from __future__ import annotations

import inspect
import json
import os
import re
import struct
import sys
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
_ROOT = _ORCH.parent
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

# The real botocore and boto3, loaded before any sibling module can stub them.
import boto3  # noqa: E402,F401
import botocore.exceptions  # noqa: E402,F401
import httpx  # noqa: E402
import sqlalchemy as sa  # noqa: E402
import yaml  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.routing import APIRoute  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.dialects.postgresql import ARRAY, JSONB  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import api.socials as socials_api  # noqa: E402
import core.auth.workspace_permission as permission_mod  # noqa: E402
import modules.socials.media_urls as media_urls  # noqa: E402
import modules.socials.service as socials_service  # noqa: E402
import modules.socials.settings as socials_settings  # noqa: E402
from config import config  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.models.socials import SocialPost, SocialPostTarget  # noqa: E402
from core.models.workspaces import Workspace  # noqa: E402
from core.storage import get_public_s3_client, reset_s3_client  # noqa: E402
from modules.socials.media_store import MediaStore, media_key  # noqa: E402
from tests.fixtures.tiny_mp4 import tiny_mp4  # noqa: E402

WS = uuid.UUID("00000000-0000-0000-0000-00000000a202")
WS_OTHER = uuid.UUID("00000000-0000-0000-0000-00000000b202")
POST = uuid.UUID("00000000-0000-0000-0000-00000000c202")
NOW = datetime(2026, 9, 28, 9, 0)
TTL = 4321
DOCUMENTS = "automatos-docs"
PUBLIC_BUCKET = "automatos-public-media"
MINIO_ENDPOINT = "http://minio:9000"
MEDIA_ROUTE = "/api/socials/posts/{post_id}/media"

FIXTURE = _ORCH / "tests" / "fixtures" / "tiny.mp4"
MANIFEST = _ORCH / "reports" / "route-manifest.json"
WORKFLOW = _ROOT / ".github" / "workflows" / "test.yml"
DOCS = _ROOT / "docs" / "getting-started" / "self-hosting.md"
API_CLIENT = _ROOT / "frontend" / "lib" / "api-client.ts"
SOCIALS_SOURCES = [*sorted((_ORCH / "modules" / "socials").rglob("*.py")), _ORCH / "api" / "socials.py"]

# ``deliverables`` has no ORM model; a stand-in with the columns the link lookup reads.
_STANDIN = sa.MetaData()
DELIVERABLES = sa.Table(
    "deliverables",
    _STANDIN,
    sa.Column("id", sa.String(36), primary_key=True),
    sa.Column("workspace_id", sa.String(36), nullable=False),
    sa.Column("storage_type", sa.String(20), nullable=False),
    sa.Column("file_path", sa.String(1024), nullable=False),
    sa.Column("file_name", sa.String(255), nullable=True),
    sa.Column("file_size_bytes", sa.BigInteger, nullable=True),
    sa.Column("deleted_at", sa.DateTime, nullable=True),
)


# ---------------------------------------------------------------------------
# The harness
# ---------------------------------------------------------------------------


def _portable(col_type):
    if isinstance(col_type, (JSONB, ARRAY)):
        return sa.JSON()
    if isinstance(col_type, sa.Uuid) or type(col_type).__name__.upper() == "UUID":
        return sa.Uuid()
    return col_type


def _sqlite_copy(table: sa.Table, metadata: sa.MetaData) -> sa.Table:
    columns = [sa.Column(col.name, _portable(col.type), primary_key=col.primary_key) for col in table.columns]
    return sa.Table(table.name, metadata, *columns)


@pytest.fixture
def db():
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    copies = sa.MetaData()
    _sqlite_copy(Workspace.__table__, copies)
    copies.create_all(engine)
    _STANDIN.create_all(engine)
    SocialPost.metadata.create_all(engine, tables=[SocialPost.__table__, SocialPostTarget.__table__])
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _configure(monkeypatch, *, endpoint="", key="AKIATEST", secret="test-secret", public_bucket=""):
    """The storage knobs media_urls and the factory read (defaults: AWS S3, the hosted shape)."""
    for name, value in {
        "S3_ENDPOINT_URL": endpoint,
        "S3_PUBLIC_ENDPOINT_URL": "",
        "AWS_ACCESS_KEY_ID": key,
        "AWS_SECRET_ACCESS_KEY": secret,
        "S3_DOCUMENTS_BUCKET": DOCUMENTS,
        "SOCIALS_PUBLIC_MEDIA_BUCKET": public_bucket,
        "SOCIALS_MEDIA_URL_TTL_SECONDS": TTL,
    }.items():
        monkeypatch.setattr(config, name, value)


def _signed(operation, Params=None, ExpiresIn=None, HttpMethod=None):
    return f"https://signed.test/{Params['Bucket']}/{Params['Key']}?method={HttpMethod}&ttl={ExpiresIn}"


@pytest.fixture
def storage(monkeypatch):
    """The storage factory faked: the public client (links that leave the backend)
    apart from the backend client, so a test sees which one did what."""
    public, backend = MagicMock(name="public-client"), MagicMock(name="backend-client")
    public.generate_presigned_url.side_effect = _signed
    backend.generate_presigned_url.side_effect = AssertionError("a link that leaves the backend was minted on the backend client")
    ensured: List[str] = []
    monkeypatch.setattr(media_urls, "get_public_s3_client", lambda: public)
    monkeypatch.setattr(media_urls, "get_s3_client", lambda: backend)
    monkeypatch.setattr(media_urls, "ensure_bucket", ensured.append)
    _configure(monkeypatch, endpoint=MINIO_ENDPOINT)
    return SimpleNamespace(public=public, backend=backend, ensured=ensured)


def _deliverable(session, name, *, workspace=WS, storage="s3", key=None, size=2048, deleted=False) -> str:
    ident = str(uuid.uuid4())
    session.execute(
        DELIVERABLES.insert().values(
            id=ident,
            workspace_id=str(workspace),
            storage_type=storage,
            file_path=key or f"social-media/{workspace}/{POST}/{name}",
            file_name=name,
            file_size_bytes=size,
            deleted_at=NOW if deleted else None,
        )
    )
    session.commit()
    return ident


def _record(ident: str, name: str, size: int = 2048) -> dict:
    """A rendered file record, as a render writes it into the post's media."""
    return {"deliverable_id": ident, "name": name, "sha256": "a" * 64, "bytes": size, "content_type": "video/mp4"}


def _post(media, *, workspace=WS, post_id=POST):
    return SimpleNamespace(id=post_id, workspace_id=workspace, media=media)


def _presigned(client) -> List[dict]:
    return [
        {"operation": c.args[0], **c.kwargs}
        for c in client.generate_presigned_url.call_args_list
    ]


def _inline(bucket: str, key: str, content_type: str, method: str = "GET") -> dict:
    return {
        "operation": "get_object",
        "Params": {
            "Bucket": bucket,
            "Key": key,
            "ResponseContentDisposition": "inline",
            "ResponseContentType": content_type,
        },
        "ExpiresIn": TTL,
        "HttpMethod": method,
    }


# ---------------------------------------------------------------------------
# Links: inline, the file's content type, the TTL from config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "content_type"),
    [("render-9x16.mp4", "video/mp4"), ("slide-1.jpg", "image/jpeg"), ("card-1x1.png", "image/png")],
)
def test_a_link_is_presigned_inline_with_the_files_type_and_the_config_ttl(db, storage, name, content_type):
    ident = _deliverable(db, name, size=5120)
    links = media_urls.post_media_links(db, _post({"9:16": [_record(ident, name, 5120)]}))

    key = f"social-media/{WS}/{POST}/{name}"
    assert _presigned(storage.public) == [_inline(DOCUMENTS, key, content_type)]
    assert links == [
        {
            "aspect": "9:16",
            "deliverable_id": ident,
            "name": name,
            "url": f"https://signed.test/{DOCUMENTS}/{key}?method=GET&ttl={TTL}",
            "content_type": content_type,
            "bytes": 5120,
            "error": None,
        }
    ]
    storage.backend.generate_presigned_url.assert_not_called()


def test_a_link_is_signed_for_the_method_asked(storage):
    """A SigV4 link is bound to its method: a HEAD needs a link signed for HEAD."""
    url = media_urls.presigned_inline_url(storage.public, DOCUMENTS, "social-media/w/p/a.mp4", "video/mp4", method="HEAD")
    assert url.endswith(f"?method=HEAD&ttl={TTL}")
    assert _presigned(storage.public) == [_inline(DOCUMENTS, "social-media/w/p/a.mp4", "video/mp4", "HEAD")]


def test_a_posts_media_resolves_through_its_own_workspaces_deliverables(db, storage):
    video = _deliverable(db, "render-9x16.mp4")
    still = _deliverable(db, "4d7a9a0e-5d2c-4a4e-9a51-3c9f0f6d2b11.png", key=f"generated-images/{WS}/4d7a9a0e-5d2c-4a4e-9a51-3c9f0f6d2b11.png")
    theirs = _deliverable(db, "render-1x1.mp4", workspace=WS_OTHER)
    deleted = _deliverable(db, "old-9x16.mp4", deleted=True)
    workspace_file = _deliverable(db, "notes.png", storage="workspace", key="outputs/notes.png")
    foreign_key = _deliverable(db, "stolen.mp4", key=f"social-media/{WS_OTHER}/{POST}/stolen.mp4")
    media = {
        "9:16": [_record(video, "render-9x16.mp4"), theirs, deleted],
        "1:1": [still, workspace_file, foreign_key, "not-a-uuid"],
    }

    links = media_urls.post_media_links(db, _post(media))

    assert [(link["aspect"], link["deliverable_id"], link["error"]) for link in links] == [
        ("9:16", video, None),
        ("9:16", theirs, media_urls.NOT_IN_DELIVERABLES),
        ("9:16", deleted, media_urls.NOT_IN_DELIVERABLES),
        ("1:1", still, None),
        ("1:1", workspace_file, media_urls.NOT_IN_STORAGE),
        ("1:1", foreign_key, media_urls.NOT_IN_STORAGE),
    ]
    assert [link["url"] is None for link in links] == [False, True, True, False, True, True]
    assert links[3]["content_type"] == "image/png" and links[3]["bytes"] == 2048
    assert [call["Params"]["Key"] for call in _presigned(storage.public)] == [
        f"social-media/{WS}/{POST}/render-9x16.mp4",
        f"generated-images/{WS}/4d7a9a0e-5d2c-4a4e-9a51-3c9f0f6d2b11.png",
    ]


def test_a_post_with_nothing_stored_never_touches_storage(db, monkeypatch):
    def no_storage():
        raise AssertionError("storage was touched")

    monkeypatch.setattr(media_urls, "get_public_s3_client", no_storage)
    workspace_file = _deliverable(db, "notes.png", storage="workspace", key="outputs/notes.png")

    assert media_urls.post_media_links(db, _post({})) == []
    assert media_urls.post_media_links(db, _post(None)) == []
    (link,) = media_urls.post_media_links(db, _post({"1:1": [workspace_file]}))
    assert link["url"] is None and link["error"] == media_urls.NOT_IN_STORAGE


# ---------------------------------------------------------------------------
# Public storage: a link a platform fetches
# ---------------------------------------------------------------------------


def _stored_video(db) -> media_urls.MediaFile:
    ident = _deliverable(db, "render-9x16.mp4")
    (media,) = media_urls.resolve_post_media(db, _post({"9:16": [ident]}))
    return media


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param({"endpoint": MINIO_ENDPOINT}, id="minio-without-a-public-bucket"),
        pytest.param({"endpoint": "", "key": None, "secret": None}, id="no-storage-at-all"),
    ],
)
def test_without_public_storage_a_platform_gets_no_link(db, storage, monkeypatch, shape):
    media = _stored_video(db)
    _configure(monkeypatch, **shape)

    assert media_urls.media_public_url_available() is False
    with pytest.raises(media_urls.NeedsPublicStorage) as refused:
        media_urls.public_media_url(_post({}), media)
    assert str(refused.value).startswith(media_urls.NEEDS_PUBLIC_STORAGE)
    assert "SOCIALS_PUBLIC_MEDIA_BUCKET" in str(refused.value)
    storage.public.generate_presigned_url.assert_not_called()
    storage.backend.copy_object.assert_not_called()


def test_aws_s3_links_the_documents_bucket_itself(db, storage, monkeypatch):
    media = _stored_video(db)
    _configure(monkeypatch, endpoint="")

    assert media_urls.media_public_url_available() is True
    url = media_urls.public_media_url(_post({}), media)

    key = f"social-media/{WS}/{POST}/render-9x16.mp4"
    assert url.startswith(f"https://signed.test/{DOCUMENTS}/{key}")
    assert _presigned(storage.public) == [_inline(DOCUMENTS, key, "video/mp4")]
    storage.backend.copy_object.assert_not_called()
    assert storage.ensured == []


@pytest.mark.parametrize("endpoint", [MINIO_ENDPOINT, ""], ids=["minio", "aws"])
def test_a_public_bucket_gets_a_copy_under_social_media_workspace_post_file(db, storage, monkeypatch, endpoint):
    still_name = "4d7a9a0e-5d2c-4a4e-9a51-3c9f0f6d2b11.png"
    source = f"generated-images/{WS}/{still_name}"
    ident = _deliverable(db, still_name, key=source)
    (media,) = media_urls.resolve_post_media(db, _post({"1:1": [ident]}))
    _configure(monkeypatch, endpoint=endpoint, public_bucket=PUBLIC_BUCKET)

    assert media_urls.media_public_url_available() is True
    url = media_urls.public_media_url(_post({}), media)

    public_key = f"social-media/{WS}/{POST}/{still_name}"
    assert storage.ensured == [PUBLIC_BUCKET]
    storage.backend.copy_object.assert_called_once_with(
        Bucket=PUBLIC_BUCKET, Key=public_key, CopySource={"Bucket": DOCUMENTS, "Key": source}
    )
    assert _presigned(storage.public) == [_inline(PUBLIC_BUCKET, public_key, "image/png")]
    assert url.startswith(f"https://signed.test/{PUBLIC_BUCKET}/{public_key}")


def test_a_file_with_no_stored_object_has_no_platform_link(db, storage, monkeypatch):
    _configure(monkeypatch, endpoint="")
    (media,) = media_urls.resolve_post_media(db, _post({"9:16": [str(uuid.uuid4())]}))
    with pytest.raises(media_urls.MediaUnavailable, match="no longer in Deliverables"):
        media_urls.public_media_url(_post({}), media)
    storage.public.generate_presigned_url.assert_not_called()


# ---------------------------------------------------------------------------
# The route: GET /api/socials/posts/{post_id}/media
# ---------------------------------------------------------------------------


def _ctx(workspace_id):
    return RequestContext(
        workspace_id=workspace_id,
        user=UserContext(id="member-1", clerk_user_id="clerk-member-1", system_role="user"),
        auth_type="clerk",
    )


@pytest.fixture
def api(db, monkeypatch):
    for ws_id in (WS, WS_OTHER):
        db.add(
            Workspace(
                id=ws_id, name=f"ws-{ws_id.hex[-4:]}", plan="basic", plan_limits={},
                settings={"socials": {"enabled": True}}, onboarding={}, created_at=NOW, updated_at=NOW,
            )
        )
    db.commit()
    state = SimpleNamespace(session=db, ctx=_ctx(WS))
    monkeypatch.setattr(socials_settings, "read_system_setting", lambda category, key: "true")
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda session, ctx: "owner")
    app = FastAPI()
    app.include_router(socials_api.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: state.ctx
    app.dependency_overrides[get_db] = lambda: db
    state.client = TestClient(app)
    return state


def _saved_post(session, media, *, workspace=WS, post_id=POST) -> SocialPost:
    post = SocialPost(
        id=post_id, workspace_id=workspace, created_by="member-1", title="Launch", copy={"base": "Soon."},
        variables={}, sources={}, media=media, status="needs_approval", review_log=[], override_unsourced=False,
    )
    post.content_hash = socials_service.compute_content_hash(post)
    session.add(post)
    session.commit()
    return post


def test_the_route_answers_the_callers_posts_media_as_presigned_inline_links(api, storage):
    video = _deliverable(api.session, "render-9x16.mp4", size=4096)
    _saved_post(api.session, {"9:16": [_record(video, "render-9x16.mp4", 4096)]})

    resp = api.client.get(MEDIA_ROUTE.format(post_id=POST))

    assert resp.status_code == 200, resp.text
    key = f"social-media/{WS}/{POST}/render-9x16.mp4"
    assert resp.json() == [
        {
            "aspect": "9:16",
            "deliverable_id": video,
            "name": "render-9x16.mp4",
            "url": f"https://signed.test/{DOCUMENTS}/{key}?method=GET&ttl={TTL}",
            "content_type": "video/mp4",
            "bytes": 4096,
            "error": None,
        }
    ]
    assert _presigned(storage.public) == [_inline(DOCUMENTS, key, "video/mp4")]


def test_another_workspaces_post_is_404_and_nothing_is_signed(api, storage):
    theirs = _deliverable(api.session, "render-9x16.mp4", workspace=WS_OTHER)
    _saved_post(api.session, {"9:16": [theirs]}, workspace=WS_OTHER)

    resp = api.client.get(MEDIA_ROUTE.format(post_id=POST))

    assert resp.status_code == 404
    storage.public.generate_presigned_url.assert_not_called()


def test_a_post_with_no_media_answers_an_empty_list(api, storage):
    _saved_post(api.session, {})
    resp = api.client.get(MEDIA_ROUTE.format(post_id=POST))
    assert resp.status_code == 200 and resp.json() == []


def test_with_no_storage_configured_the_route_answers_503(api, monkeypatch):
    video = _deliverable(api.session, "render-9x16.mp4")
    _saved_post(api.session, {"9:16": [video]})
    _configure(monkeypatch, endpoint="", key=None, secret=None)
    reset_s3_client()
    try:
        resp = api.client.get(MEDIA_ROUTE.format(post_id=POST))
    finally:
        reset_s3_client()
    assert resp.status_code == 503
    assert resp.json()["detail"] == socials_api.MEDIA_STORAGE_UNAVAILABLE


def test_the_route_is_a_plain_def_in_the_committed_manifest_and_apiclient_calls_it():
    (route,) = [r for r in socials_api.router.routes if isinstance(r, APIRoute) and r.path == MEDIA_ROUTE]
    assert route.methods == {"GET"}
    assert not inspect.iscoroutinefunction(route.endpoint), "a route over a sync Session is a plain def (F105)"
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert {"method": "GET", "path": MEDIA_ROUTE} in manifest["routes"]
    assert manifest["route_count"] == len(manifest["routes"])
    client = API_CLIENT.read_text(encoding="utf-8")
    call = re.search(r"async getSocialPostMedia\(postId: string\)[^{]*\{\s*return this\.request<[^>]*>\(([^)]*)\)", client)
    assert call and call.group(1) == "`/api/socials/posts/${postId}/media`"


# ---------------------------------------------------------------------------
# MinIO, over real HTTP (CI starts it; see .github/workflows/test.yml)
# ---------------------------------------------------------------------------


def _minio_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    if os.environ.get("CI", "").strip().lower() == "true":
        pytest.fail(f"{name} is unset in CI: the orchestrator-tests job must start MinIO and pass it (US-202)")
    pytest.skip(f"no MinIO here: set {name} (CI starts one)")


@pytest.fixture
def minio(monkeypatch):
    """The storage factory pointed at the real MinIO, a fresh documents bucket."""
    endpoint = _minio_env("SOCIALS_TEST_S3_ENDPOINT")
    for name, value in {
        "S3_ENDPOINT_URL": endpoint,
        "S3_PUBLIC_ENDPOINT_URL": endpoint,
        "S3_USE_PATH_STYLE": True,
        "AWS_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": _minio_env("SOCIALS_TEST_S3_ACCESS_KEY"),
        "AWS_SECRET_ACCESS_KEY": _minio_env("SOCIALS_TEST_S3_SECRET_KEY"),
        "S3_DOCUMENTS_BUCKET": f"socials-docs-{uuid.uuid4().hex[:12]}",
        "SOCIALS_PUBLIC_MEDIA_BUCKET": "",
        "SOCIALS_MEDIA_URL_TTL_SECONDS": 600,
    }.items():
        monkeypatch.setattr(config, name, value)
    reset_s3_client()
    try:
        yield endpoint
    finally:
        reset_s3_client()


def _upload_fixture(db) -> Tuple[str, str]:
    """The fixture stored as a render stores its files, registered as a Deliverable.
    Stored as bytes of no particular type: the link sets the type D9 asks for."""
    key = media_key(WS, POST, "render-9x16.mp4")
    MediaStore().put_file(key, FIXTURE, "application/octet-stream")
    return key, _deliverable(db, "render-9x16.mp4", size=FIXTURE.stat().st_size)


def _assert_served_inline_with_ranges(url: str, data: bytes) -> None:
    with httpx.Client(timeout=15) as http:
        ranged = http.get(url, headers={"Range": "bytes=0-99"})
        assert ranged.status_code == 206, ranged.text
        assert len(ranged.content) == 100 and ranged.content == data[:100]
        assert ranged.headers["content-range"] == f"bytes 0-99/{len(data)}"
        assert ranged.headers["content-type"] == "video/mp4"
        assert ranged.headers["content-disposition"] == "inline"
        whole = http.get(url)
        assert whole.status_code == 200 and whole.content == data


def test_minio_serves_the_link_inline_and_answers_a_range_request(minio, db):
    data = FIXTURE.read_bytes()
    key, ident = _upload_fixture(db)
    post = _post({"9:16": [_record(ident, "render-9x16.mp4", len(data))]})

    (link,) = media_urls.post_media_links(db, post)
    assert link["url"].startswith(f"{minio}/{config.S3_DOCUMENTS_BUCKET}/{key}?")
    (media,) = media_urls.resolve_post_media(db, post)
    head_url = media_urls.presigned_inline_url(
        get_public_s3_client(), config.S3_DOCUMENTS_BUCKET, media.key, media.content_type, method="HEAD"
    )

    with httpx.Client(timeout=15) as http:
        head = http.head(head_url)
    assert head.status_code == 200, head.text
    assert head.headers["content-type"] == "video/mp4"
    assert head.headers["content-disposition"] == "inline"
    assert int(head.headers["content-length"]) == len(data)
    _assert_served_inline_with_ranges(link["url"], data)


def test_minio_serves_the_public_buckets_copy(minio, db, monkeypatch):
    data = FIXTURE.read_bytes()
    _, ident = _upload_fixture(db)
    public_bucket = f"socials-public-{uuid.uuid4().hex[:12]}"
    monkeypatch.setattr(config, "SOCIALS_PUBLIC_MEDIA_BUCKET", public_bucket)
    post = _post({"9:16": [ident]})
    (media,) = media_urls.resolve_post_media(db, post)

    assert media_urls.media_public_url_available() is True
    url = media_urls.public_media_url(post, media)

    assert url.startswith(f"{minio}/{public_bucket}/social-media/{WS}/{POST}/render-9x16.mp4?")
    _assert_served_inline_with_ranges(url, data)


# ---------------------------------------------------------------------------
# Guards: CI, the fixture, the docs, the Socials sources
# ---------------------------------------------------------------------------


def _orchestrator_steps() -> list:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]["orchestrator-tests"]["steps"]


def test_ci_starts_a_pinned_minio_before_the_test_net_and_hands_the_tests_its_endpoint():
    steps = _orchestrator_steps()
    (minio,) = [i for i, step in enumerate(steps) if "minio" in (step.get("run") or "").lower()]
    (net,) = [i for i, step in enumerate(steps) if step.get("name") == "Run test net (with coverage)"]
    assert minio < net
    run, step_env = steps[minio]["run"], steps[minio].get("env") or {}
    assert "docker run -d" in run and "server /data" in run and "/minio/health/live" in run
    assert re.search(r"minio/minio:RELEASE\.\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z", run), "the MinIO image is pinned"
    env = steps[net]["env"]
    assert env["SOCIALS_TEST_S3_ENDPOINT"] == "http://127.0.0.1:9000"
    assert env["SOCIALS_TEST_S3_ACCESS_KEY"] == step_env["MINIO_ROOT_USER"]
    assert env["SOCIALS_TEST_S3_SECRET_KEY"] == step_env["MINIO_ROOT_PASSWORD"]


def _boxes(data: bytes, start: int = 0, end: Optional[int] = None) -> List[Tuple[str, int, int]]:
    """(type, payload start, end) of each ISO-BMFF box in data[start:end]."""
    end = len(data) if end is None else end
    found, at = [], start
    while at < end:
        size, kind = struct.unpack(">I4s", data[at:at + 8])
        assert size >= 8 and at + size <= end, (kind, size)
        found.append((kind.decode("latin-1"), at + 8, at + size))
        at += size
    assert at == end
    return found


def _child(data: bytes, parent: Tuple[str, int, int], kind: str, skip: int = 0) -> Tuple[str, int, int]:
    (box,) = [b for b in _boxes(data, parent[1] + skip, parent[2]) if b[0] == kind]
    return box


def test_the_fixture_is_a_tiny_mp4_this_repo_generates():
    data = FIXTURE.read_bytes()
    assert data == tiny_mp4(), "tiny.mp4 is not the generator's output: run python -m tests.fixtures.tiny_mp4"
    assert 100 < len(data) < 100 * 1024
    ftyp, moov, mdat = _boxes(data)
    assert (ftyp[0], moov[0], mdat[0]) == ("ftyp", "moov", "mdat")
    assert data[ftyp[1]:ftyp[1] + 4] == b"isom"
    stbl = moov
    for kind in ("trak", "mdia", "minf", "stbl"):
        stbl = _child(data, stbl, kind)
    stsd = _child(data, stbl, "stsd")
    assert _boxes(data, stsd[1] + 8, stsd[2])[0][0] == "avc1"
    stsz, stco = _child(data, stbl, "stsz"), _child(data, stbl, "stco")
    count = struct.unpack(">I", data[stsz[1] + 8:stsz[1] + 12])[0]
    sizes = struct.unpack(f">{count}I", data[stsz[1] + 12:stsz[2]])
    assert sum(sizes) == mdat[2] - mdat[1]
    assert struct.unpack(">I", data[stco[1] + 8:stco[2]])[0] == mdat[1]


def test_socials_builds_no_storage_client_and_never_uses_the_generated_images_route():
    for path in SOCIALS_SOURCES:
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"\bboto3\b", text), f"{path}: storage clients come from core/storage/s3.py"
        assert "/api/generated-images" not in text, f"{path}: D9 never serves Socials media there"


def test_the_self_hosting_docs_say_what_needs_public_storage():
    docs = DOCS.read_text(encoding="utf-8")
    for needle in (
        "SOCIALS_PUBLIC_MEDIA_BUCKET",
        "S3_PUBLIC_ENDPOINT_URL",
        "SOCIALS_MEDIA_URL_TTL_SECONDS",
        "YouTube",
        "thumbnail",
        media_urls.NEEDS_PUBLIC_STORAGE,
    ):
        assert needle in docs, needle
