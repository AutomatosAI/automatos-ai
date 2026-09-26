"""PRD-251 Wave 1, US-104 (S1.1c) — rendering a post: the lifecycle, the quota, the client.

The real Socials router runs on a mini FastAPI app over in-memory SQLite (the
social tables, SQLite copies of ``workspaces`` / ``document_templates`` and an
untyped ``llm_usage``), with the real gate and permission check; the master
switch and the caller's role are stubbed. media-render is mocked at the HTTP
layer (``httpx.MockTransport``), so the real client runs; storage, the
Deliverable registry and the usage tracker are recording fakes. Pins:

* POST /render moves a draft to ``rendering`` (202) and hands the render to the
  background. Run, the render copies the MP4 into storage under
  ``social-media/{workspace}/{post}/``, registers it as a video Deliverable, ends
  the post in ``needs_approval`` with the file's sha256 in ``media`` (the
  content hash covers it) and books the rendered seconds on the media lane;
* a composition the renderer's check refuses ends the post in ``failed`` with
  the findings in ``review_log``; a failed post is edited and rendered again,
  and nothing else; an approved post never renders;
* the quota: basic / pro / business read 10 / 60 / 240 minutes from config,
  enterprise and the local edition have none; the month's media-lane render
  units are the minutes used; a render past the quota is refused (429) BEFORE
  any call to media-render;
* no template → 422; no renderer → 503, "Rendering needs the media profile" in
  the local edition; GET /usage; GET /media streams a rendered file;
* the client sends X-Internal-Token with config's timeouts, and maps 202 / 422 /
  503 / connection and read failures; a download streams to disk with its sha256;
* the boot reaper fails a render stranded by a restart;
* compose runs media-render under the ``media`` profile only, and the Railway
  manifest carries it at 4 vCPU / 8 GB with one replica;
* US-107: a carousel (a social_image template) asks for a still per slide it
  shows, and its PNGs are stored, registered and recorded one per slide, in
  order, at no render minutes.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
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

import httpx  # noqa: E402
import sqlalchemy as sa  # noqa: E402
import yaml  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.dialects.postgresql import ARRAY, JSONB  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import core.models  # noqa: E402,F401  (registers every mapper)
import api.socials as socials_api  # noqa: E402
import core.auth.workspace_permission as permission_mod  # noqa: E402
import core.boot.reaper as reaper  # noqa: E402
import core.media_render_client as media_render_client  # noqa: E402
import core.media_render_quota as render_quota  # noqa: E402  (US-105 moved it into core)
import modules.socials.media_store as media_store  # noqa: E402
import modules.socials.render as render  # noqa: E402
import modules.socials.service as service  # noqa: E402
import modules.socials.settings as socials_settings  # noqa: E402
import services.deliverable_service as deliverable_service  # noqa: E402
import services.plan_tiers as plan_tiers  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.llm import usage_context as uc  # noqa: E402
from core.llm.providers import MEDIA_RENDER_PROVIDER  # noqa: E402
from core.llm.usage_tracker import UsageTracker  # noqa: E402
from core.media_render_bundle import NO_LOGO  # noqa: E402
from core.media_render_client import MediaRenderClient, MediaRenderError, MediaRenderUnavailable  # noqa: E402
from core.models.core import DocumentTemplate, LLMUsage  # noqa: E402
from core.models.socials import SocialPost, SocialPostTarget  # noqa: E402
from core.models.workspaces import Workspace  # noqa: E402
from modules.documents.brand_kit import DEFAULT_ACCENT, DEFAULT_FONT, DEFAULT_PRIMARY  # noqa: E402

# Hex with letters, so SQLite keeps every UUID column as text.
WS = uuid.UUID("00000000-0000-0000-0000-0000000000a1")
WS_PRO = uuid.UUID("00000000-0000-0000-0000-0000000000a2")
WS_OTHER = uuid.UUID("00000000-0000-0000-0000-0000000000a3")
CREATED = datetime(2026, 9, 1, 9, 0)
RENDER_URL = "http://media-render:8090"
TOKEN = "render-secret"
JOB_ID = "b" * 32
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"frame-bytes " * 400
MP4_SHA = hashlib.sha256(MP4).hexdigest()
OUTPUT = {"name": "render.mp4", "aspect": "9:16", "width": 1080, "height": 1920, "bytes": len(MP4), "duration": 39.5}
COMPOSITION = {
    # A social template as S1.2 defines it (core/social_templates.py): a full document.
    "html": '<!doctype html><html><head></head><body><div id="root" data-composition-id="main" data-width="1080" '
            'data-height="1920" data-duration="39.5"><h1>{{ headline }}</h1></div></body></html>',
    "css": "h1 { color: var(--brand-text); }",
    "variables_schema": {"headline": {"type": "text"}},
    "sizes": ["1080x1920"],
    "audio_plan": {"voice": {"voice": "af_heart", "speed": 0.95, "lines": [{"id": "l01", "at": 0.3, "text": "Three weeks to go."}]}},
}
CHECK_FINDING = {
    "section": "layout", "severity": "error", "code": "text_overflow",
    "message": "the headline overflows its box at 2.4 s", "selector": "h1", "fixHint": "shorten it",
}


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
    """A column-for-column copy SQLite can build (JSONB / ARRAY → JSON, UUID → CHAR(32))."""
    columns = [sa.Column(c.name, _portable(c.type), primary_key=c.primary_key) for c in table.columns]
    return sa.Table(table.name, metadata, *columns)


def _ctx(workspace_id, user_id="member-1"):
    return RequestContext(
        workspace_id=workspace_id,
        user=UserContext(id=user_id, clerk_user_id=f"clerk-{user_id}", system_role="user"),
        auth_type="clerk",
    )


def _set_config(monkeypatch, **values):
    """Patch the config object every module under test reads (one object, unless a reload split it)."""
    targets = {id(m.config): m.config for m in (render, render_quota, media_render_client, media_store)}
    for cfg in targets.values():
        for name, value in values.items():
            monkeypatch.setattr(cfg, name, value, raising=False)


class FakeStore:
    """The documents bucket, in memory."""

    def __init__(self, configured=True, fail_put=False):
        self.objects = {}
        self._configured = configured
        self.fail_put = fail_put

    def configured(self):
        return self._configured

    def put_file(self, key, path, content_type):
        if self.fail_put:
            raise RuntimeError("the bucket refused the write")
        self.objects[key] = (Path(path).read_bytes(), content_type)

    def open(self, key):
        if key not in self.objects:
            return None
        data, content_type = self.objects[key]
        return media_store.MediaObject(body=iter([data]), content_type=content_type, content_length=len(data))


class Deliverables:
    """Stands in for DeliverableService (its SQL is Postgres-only): records register()."""

    calls: list = []

    def __init__(self, db, workspace_id):
        self.workspace_id = workspace_id

    def register(self, **kwargs):
        type(self).calls.append({"workspace_id": self.workspace_id, **kwargs})
        return {"success": True, "deliverable_id": f"d-video-{len(type(self).calls)}", "created": True}


@pytest.fixture
def env(monkeypatch):
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    copies = sa.MetaData()
    _sqlite_copy(Workspace.__table__, copies)
    _sqlite_copy(DocumentTemplate.__table__, copies)
    copies.create_all(engine)
    SocialPost.metadata.create_all(engine, tables=[SocialPost.__table__, SocialPostTarget.__table__])
    # llm_usage is raw DDL over the model's own names, untyped: SQLAlchemy 2.0.23
    # cannot compile the Postgres UUID type for SQLite (test_prd251w1_media_lane.py).
    columns = ", ".join(
        f"{c.name} INTEGER PRIMARY KEY" if c.primary_key else c.name for c in LLMUsage.__table__.columns
    )
    with engine.begin() as conn:
        conn.exec_driver_sql(f"CREATE TABLE {LLMUsage.__tablename__} ({columns})")

    factory = sessionmaker(bind=engine)
    session = factory()
    for ws_id, plan in ((WS, "basic"), (WS_PRO, "pro"), (WS_OTHER, "basic")):
        session.add(
            Workspace(
                id=ws_id, name=f"ws-{ws_id.hex[-2:]}", plan=plan, plan_limits={},
                settings={"socials": {"enabled": True}}, onboarding={}, created_at=CREATED, updated_at=CREATED,
            )
        )
    session.commit()

    state = SimpleNamespace(session=session, factory=factory, ctx=_ctx(WS), role="owner", launched=[], health=[])

    monkeypatch.setattr(socials_settings, "read_system_setting", lambda category, key: "true")
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: state.role)
    monkeypatch.setattr(socials_api, "_launch_render", lambda job: state.launched.append(job))

    async def renderer_up(client=None, store=None):
        state.health.append(True)

    monkeypatch.setattr(render, "ensure_renderer", renderer_up)
    _set_config(
        monkeypatch,
        AUTH_EDITION="saas",
        SOCIALS_RENDER_URL=RENDER_URL,
        SOCIALS_RENDER_TOKEN=TOKEN,
        SOCIALS_RENDER_POLL_SECONDS=0,
        SOCIALS_RENDER_MAX_WAIT_SECONDS=60,
    )
    Deliverables.calls = []
    monkeypatch.setattr(deliverable_service, "DeliverableService", Deliverables)
    state.booked = []

    def track_media(**kwargs):
        state.booked.append({**kwargs, "scope": dict(uc.current_usage_scope())})

    monkeypatch.setattr(UsageTracker, "track_media", staticmethod(track_media))

    app = FastAPI()
    app.include_router(socials_api.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: state.ctx
    app.dependency_overrides[get_db] = lambda: session
    state.client = TestClient(app)
    try:
        yield state
    finally:
        session.close()
        engine.dispose()


def _template(env, *, workspace_id=WS, blocks=COMPOSITION, fmt="social_video"):
    template_id = uuid.uuid4()
    env.session.execute(
        sa.text(
            "INSERT INTO document_templates (id, workspace_id, name, format, data_schema, blocks) "
            "VALUES (:id, :ws, :name, :fmt, '{}', :blocks)"
        ),
        {
            "id": template_id.hex, "ws": workspace_id.hex, "name": f"tpl-{template_id.hex[:6]}", "fmt": fmt,
            "blocks": json.dumps(blocks) if blocks is not None else None,
        },
    )
    env.session.commit()
    return template_id


def _create(env, **body):
    payload = {
        "title": "Web Summit countdown",
        "copy": {"base": "Three weeks to go."},
        "format": "video",
        "variables": {"headline": {"value": "Three weeks to go", "claim": False}},
    }
    payload.update(body)
    resp = env.client.post("/api/socials/posts", json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()


def _renderable(env, **body):
    return _create(env, template_id=str(_template(env)), **body)


def _post(env, post_id):
    env.session.expire_all()
    return env.session.get(SocialPost, uuid.UUID(post_id))


def _usage_row(workspace_id, seconds, *, provider=MEDIA_RENDER_PROVIDER, lane="media", at=None):
    return LLMUsage(
        workspace_id=workspace_id, model_id="hyperframes", provider=provider, tier="direct",
        request_type=lane, input_tokens=seconds, output_tokens=0, total_tokens=seconds,
        cache_read_tokens=0, cache_write_tokens=0, input_cost=0.0, output_cost=0.0, total_cost=0.0,
        is_byok=False, status="success",
        created_at=at or datetime.now(timezone.utc).replace(tzinfo=None),
    )


class Renderer:
    """media-render over httpx.MockTransport: the real client talks to it."""

    def __init__(self, *, reject=None, busy_first=False, status="done", error=None, outputs=(OUTPUT,), files=None):
        self.reject = reject
        self.busy_first = busy_first
        self.status = status
        self.error = error
        self.outputs = list(outputs)
        # The files GET /render/{id}/output/{name} serves (US-107: a carousel's PNGs).
        self.files = dict(files) if files is not None else {"render.mp4": MP4}
        self.bundles = []
        self.seen = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.seen.append((request.method, path, request.headers.get("X-Internal-Token")))
        if request.method == "POST" and path == "/render":
            self.bundles.append(json.loads(request.content))
            if self.busy_first:
                self.busy_first = False
                return httpx.Response(
                    503, json={"error": "busy", "message": "20 renders are already in progress"},
                    headers={"Retry-After": "0"},
                )
            if self.reject is not None:
                return self.reject
            return httpx.Response(202, json={"id": JOB_ID, "status": "rendering", "outputs": [], "report": {}})
        if request.method == "GET" and path == f"/render/{JOB_ID}":
            body = {
                "id": JOB_ID, "status": self.status, "outputs": self.outputs if self.status == "done" else [],
                "report": {"check": {"ok": True, "errors": 0}, "timings": {"render_seconds": 12.5}},
                "error": self.error,
            }
            return httpx.Response(200, json=body)
        name = path.rsplit("/", 1)[-1]
        if request.method == "GET" and path == f"/render/{JOB_ID}/output/{name}" and name in self.files:
            return httpx.Response(200, content=self.files[name])
        return httpx.Response(404, json={"error": "not_found", "message": f"no route {path}"})


def _run(job, renderer, store, factory):
    async def go():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(renderer.handler), headers={"X-Internal-Token": TOKEN}
        ) as http:
            return await render.run_render(job, client=MediaRenderClient(http), store=store, session_factory=factory)

    return asyncio.run(go())


def _start(env, post):
    resp = env.client.post(f"/api/socials/posts/{post['id']}/render")
    assert resp.status_code == 202, resp.text
    (job,) = env.launched[-1:]
    return resp.json(), job


# ---------------------------------------------------------------------------
# The lifecycle: draft → rendering → needs_approval, stored and registered
# ---------------------------------------------------------------------------


def test_rendering_a_draft_stores_the_mp4_registers_a_deliverable_and_awaits_approval(env):
    post = _renderable(env)
    started, job = _start(env, post)

    # 202: the post is rendering, and the render was handed to the background.
    assert started["status"] == "rendering"
    assert started["review_log"][-1]["action"] == "render" and started["review_log"][-1]["by"] == "member-1"
    assert _post(env, post["id"]).status == "rendering"
    assert job.post_id == uuid.UUID(post["id"]) and job.workspace_id == WS and job.actor == "member-1"
    assert job.content_hash == post["content_hash"]
    assert env.health == [True]

    renderer, store = Renderer(), FakeStore()
    assert _run(job, renderer, store, env.factory) is True

    # The bundle: the template's composition, the post's variable values, the audio plan,
    # and (US-105, S1.2) the workspace brand kit as --brand-* tokens with the brand and
    # size variables. This workspace has no kit: the neutral defaults, its own name, no logo.
    (bundle,) = renderer.bundles
    assert bundle["workspace_id"] == str(WS) and bundle["reference"] == f"social_post:{post['id']}"
    assert bundle["composition"] == {"html": COMPOSITION["html"], "css": COMPOSITION["css"]}
    assert bundle["variables"] == {
        "headline": "Three weeks to go",
        "brand.name": "ws-a1",
        "brand.tagline": "",
        "brand.logo": NO_LOGO,
        "brand.logo_mark": NO_LOGO,
        "size.width": 1080,
        "size.height": 1920,
    }
    tokens = bundle["brand"]["tokens"]
    assert tokens["primary"] == DEFAULT_PRIMARY and tokens["accent"] == DEFAULT_ACCENT
    assert tokens["body-font"] == tokens["heading-font"] == DEFAULT_FONT
    assert "files" not in bundle
    assert bundle["audio"] == COMPOSITION["audio_plan"]
    assert all(token == TOKEN for _, _, token in renderer.seen)

    # Stored under social-media/{workspace}/{post}/ (D9).
    key = f"social-media/{WS}/{post['id']}/video-9x16.mp4"
    assert store.objects == {key: (MP4, "video/mp4")}

    # Registered as a video Deliverable whose preview is the post's media route.
    (registered,) = Deliverables.calls
    assert registered["workspace_id"] == WS and registered["file_path"] == key
    assert registered["artifact_type"] == "video" and registered["storage_type"] == "s3"
    assert registered["source_type"] == "social_post" and registered["source_id"] == post["id"]
    assert registered["file_size_bytes"] == len(MP4)
    assert registered["preview_url"] == f"/api/socials/posts/{post['id']}/media/video-9x16.mp4"
    assert registered["extra"]["sha256"] == MP4_SHA

    # needs_approval, with the file and its digest in media; the hash covers it.
    row = _post(env, post["id"])
    assert row.status == "needs_approval"
    (record,) = row.media["9:16"]
    assert record == {
        "deliverable_id": "d-video-1", "name": "video-9x16.mp4", "sha256": MP4_SHA, "bytes": len(MP4),
        "content_type": "video/mp4", "duration": 39.5, "width": 1080, "height": 1920,
    }
    assert row.content_hash == service.compute_content_hash(row) != post["content_hash"]
    done = row.review_log[-1]
    assert done["action"] == "render_done" and done["comment"] == "Rendered 39.5 s 1080×1920."
    assert done["report"]["check"] == {"ok": True, "errors": 0}

    # The rendered seconds are booked on the media lane at $0: the quota counts them.
    (booking,) = env.booked
    assert booking["provider"] == MEDIA_RENDER_PROVIDER and booking["model_id"] == "hyperframes"
    assert booking["units"] == 39.5 and booking["usd"] == 0.0
    assert booking["scope"]["workspace_id"] == WS and booking["scope"]["request_type"] == "media"
    assert booking["scope"]["execution_id"] == f"social_post:{post['id']}"


def test_the_approval_binds_to_the_rendered_bytes(env):
    post = _renderable(env)
    _, job = _start(env, post)
    _run(job, Renderer(), FakeStore(), env.factory)
    rendered = env.client.get(f"/api/socials/posts/{post['id']}").json()

    swapped = SimpleNamespace(**{f: rendered[f] for f in service.CONTENT_FIELDS})
    swapped.media = {"9:16": [{**rendered["media"]["9:16"][0], "sha256": "c" * 64}]}
    assert service.compute_content_hash(swapped) != rendered["content_hash"]

    approved = env.client.post(f"/api/socials/posts/{post['id']}/approve", json={"content_hash": rendered["content_hash"]})
    assert approved.status_code == 200 and approved.json()["approved_hash"] == rendered["content_hash"]
    # An approved post never renders: edit it first, which voids the approval.
    assert env.client.post(f"/api/socials/posts/{post['id']}/render").status_code == 409


def test_a_busy_renderer_is_asked_again(env):
    post = _renderable(env)
    _, job = _start(env, post)
    renderer = Renderer(busy_first=True)
    assert _run(job, renderer, FakeStore(), env.factory) is True
    assert len(renderer.bundles) == 2
    assert _post(env, post["id"]).status == "needs_approval"


def test_a_refused_check_fails_the_post_with_the_findings(env):
    post = _renderable(env)
    _, job = _start(env, post)
    reject = httpx.Response(
        422,
        json={
            "error": "check_failed",
            "message": "the composition failed its check with 1 error(s); nothing was rendered",
            "findings": [CHECK_FINDING],
            "report": {"check": {"ok": False, "errors": 1}},
        },
    )
    store = FakeStore()
    assert _run(job, Renderer(reject=reject), store, env.factory) is False

    row = _post(env, post["id"])
    assert row.status == "failed"
    failed = row.review_log[-1]
    assert failed["action"] == "render_failed"
    assert failed["comment"] == (
        "The composition failed its check with 1 error(s): the headline overflows its box at 2.4 s. "
        "Nothing was rendered."
    )
    assert failed["report"]["code"] == "check_failed"
    assert failed["report"]["findings"] == [CHECK_FINDING]
    assert failed["report"]["check"] == {"ok": False, "errors": 1}
    # Nothing stored, registered or booked; the content is untouched.
    assert store.objects == {} and Deliverables.calls == [] and env.booked == []
    assert row.content_hash == post["content_hash"] and row.media == {}


def test_a_render_the_renderer_fails_ends_the_post_in_failed(env):
    post = _renderable(env)
    _, job = _start(env, post)
    renderer = Renderer(status="failed", error={"code": "render_timed_out", "message": "the render ran past 900 s"})
    assert _run(job, renderer, FakeStore(), env.factory) is False
    row = _post(env, post["id"])
    assert row.status == "failed"
    assert row.review_log[-1]["comment"] == "The render failed: the render ran past 900 s."
    assert row.review_log[-1]["report"]["code"] == "render_timed_out"


def test_an_output_that_never_arrives_leaves_nothing_behind(env):
    """Every output is fetched before any is stored: a render whose second file
    cannot be fetched stores and registers neither."""
    post = _renderable(env)
    _, job = _start(env, post)
    square = {**OUTPUT, "name": "render-square.mp4", "aspect": "1:1", "width": 1080, "height": 1080}
    store = FakeStore()
    assert _run(job, Renderer(outputs=(OUTPUT, square)), store, env.factory) is False
    row = _post(env, post["id"])
    assert row.status == "failed" and row.review_log[-1]["report"]["code"] == "not_found"
    assert store.objects == {} and Deliverables.calls == [] and env.booked == []


def test_a_render_that_outlasts_its_budget_fails_the_post(env, monkeypatch):
    """The whole render (queue, render, storing) gets SOCIALS_RENDER_MAX_WAIT_SECONDS."""
    _set_config(monkeypatch, SOCIALS_RENDER_MAX_WAIT_SECONDS=1, SOCIALS_RENDER_POLL_SECONDS=0.01)
    post = _renderable(env)
    _, job = _start(env, post)
    assert _run(job, Renderer(status="rendering"), FakeStore(), env.factory) is False
    row = _post(env, post["id"])
    assert row.status == "failed" and row.review_log[-1]["report"]["code"] == "timed_out"
    assert env.booked == []


def test_a_storage_failure_fails_the_post_and_books_nothing(env):
    post = _renderable(env)
    _, job = _start(env, post)
    assert _run(job, Renderer(), FakeStore(fail_put=True), env.factory) is False
    row = _post(env, post["id"])
    assert row.status == "failed" and row.review_log[-1]["report"]["code"] == "storage_failed"
    assert Deliverables.calls == [] and env.booked == []


def test_an_unexpected_error_fails_the_post_and_is_raised(env, monkeypatch):
    post = _renderable(env)
    _, job = _start(env, post)

    def broken(*args, **kwargs):
        raise RuntimeError("the registry fell over")

    monkeypatch.setattr(render, "_register", broken)
    with pytest.raises(RuntimeError):
        _run(job, Renderer(), FakeStore(), env.factory)
    row = _post(env, post["id"])
    assert row.status == "failed" and row.review_log[-1]["report"]["code"] == "internal_error"


def test_a_post_that_moved_on_keeps_its_state_and_nothing_is_booked(env):
    post = _renderable(env)
    _, job = _start(env, post)
    row = _post(env, post["id"])
    service.fail_render(row, "orphaned_on_restart", "The render was lost when the server restarted.")
    env.session.commit()

    assert _run(job, Renderer(), FakeStore(), env.factory) is False
    row = _post(env, post["id"])
    assert row.status == "failed" and row.review_log[-1]["by"] == "orphaned_on_restart"
    assert env.booked == []


def test_the_render_is_launched_as_a_guarded_background_task(monkeypatch):
    calls = []

    def fake_launch(coro, **kwargs):
        calls.append((coro.cr_code.co_name, kwargs))
        coro.close()

    monkeypatch.setattr(socials_api, "launch_guarded", fake_launch)
    job = render.RenderJob(
        post_id=uuid.uuid4(), workspace_id=WS, actor="member-1", content_hash="a" * 64,
        title="T", format="video", bundle={},
    )
    socials_api._launch_render(job)
    assert calls == [("run_render", {"subsystem": "socials", "operation": "render", "workspace_id": WS})]


# ---------------------------------------------------------------------------
# The status machine's render moves (service, pure)
# ---------------------------------------------------------------------------


def _draft_post():
    class _Added:
        def add(self, obj):
            pass

    return service.create_draft(_Added(), workspace_id=WS, created_by="author", title="T", format="video")


def _rendered():
    return {"9:16": [{"deliverable_id": "d-1", "name": "video-9x16.mp4", "sha256": MP4_SHA, "bytes": 10}]}


def test_a_failed_post_is_edited_and_rendered_again_and_nothing_else():
    post = _draft_post()
    service.start_render(post, "author")
    service.fail_render(post, "author", "The check failed.")
    assert post.status == service.FAILED

    for action in (
        lambda p: service.submit(p, "author"),
        lambda p: service.approve(p, "reviewer", content_hash=p.content_hash),
        lambda p: service.schedule(p, "reviewer", datetime.now(timezone.utc) + timedelta(days=1), "UTC"),
        lambda p: service.request_changes(p, "reviewer", "No."),
        lambda p: service.reject(p, "reviewer"),
    ):
        with pytest.raises(service.IllegalTransition):
            action(post)
        assert post.status == service.FAILED

    before = post.content_hash
    service.update_post(post, "author", {"variables": {"headline": {"value": "Shorter", "claim": False}}})
    assert post.status == service.FAILED and post.content_hash != before
    service.start_render(post, "author")
    assert post.status == service.RENDERING


@pytest.mark.parametrize("status", ["approved", "scheduled", "archived", "rendering", "publishing", "published"])
def test_a_post_holding_an_approval_or_in_flight_never_renders(status):
    post = _draft_post()
    post.status = status
    with pytest.raises(service.IllegalTransition):
        service.assert_can_render(post)
    with pytest.raises(service.IllegalTransition):
        service.start_render(post, "author")
    assert post.status == status


def test_a_rendering_post_cannot_be_edited():
    post = _draft_post()
    service.start_render(post, "author")
    with pytest.raises(service.IllegalTransition):
        service.update_post(post, "author", {"copy": {"base": "changed mid-render"}})


@pytest.mark.parametrize(
    "media",
    [
        {},
        {"9:16": []},
        {"9:16": [{"deliverable_id": "d-1", "name": "v.mp4", "sha256": "not-a-digest", "bytes": 10}]},
        {"9:16": [{"deliverable_id": "d-1", "name": "v.mp4", "sha256": MP4_SHA, "bytes": 0}]},
        {"9:16": [{"deliverable_id": "", "name": "v.mp4", "sha256": MP4_SHA, "bytes": 10}]},
        {"9:16": [{"deliverable_id": "d-1", "name": "v.mp4", "sha256": MP4_SHA, "bytes": 10, "url": "x"}]},
        {"9:16": [{"deliverable_id": "d-1", "name": "v.mp4", "sha256": MP4_SHA, "bytes": 10, "duration": -1}]},
    ],
)
def test_a_render_records_only_well_formed_files(media):
    post = _draft_post()
    service.start_render(post, "author")
    with pytest.raises(service.InvalidPost):
        service.finish_render(post, "author", media)
    assert post.status == service.RENDERING and post.media == {}


def test_an_edit_can_never_write_a_rendered_file_record():
    """Records come only from a render, so no client can forge a digest (D6)."""
    post = _draft_post()
    with pytest.raises(service.InvalidPost):
        service.update_post(post, "author", {"media": _rendered()})
    service.update_post(post, "author", {"media": {"9:16": ["d-attached"]}})
    assert post.media == {"9:16": ["d-attached"]}


# ---------------------------------------------------------------------------
# The route's refusals
# ---------------------------------------------------------------------------


def test_a_post_without_a_template_or_a_composition_is_422(env):
    bare = _create(env)
    resp = env.client.post(f"/api/socials/posts/{bare['id']}/render")
    assert resp.status_code == 422 and "template" in resp.json()["detail"]

    empty = _create(env, template_id=str(_template(env, blocks={"version": 1, "blocks": []})))
    resp = env.client.post(f"/api/socials/posts/{empty['id']}/render")
    assert resp.status_code == 422 and "composition" in resp.json()["detail"]
    assert env.launched == [] and _post(env, empty["id"]).status == "draft"


def test_a_viewer_cannot_render(env):
    post = _renderable(env)
    env.role = "viewer"
    assert env.client.post(f"/api/socials/posts/{post['id']}/render").status_code == 403
    assert env.launched == [] and _post(env, post["id"]).status == "draft"


def test_another_workspaces_post_is_404(env):
    env.ctx = _ctx(WS_OTHER, "member-other")
    theirs = _create(env, template_id=str(_template(env, workspace_id=WS_OTHER)))
    env.ctx = _ctx(WS)
    assert env.client.post(f"/api/socials/posts/{theirs['id']}/render").status_code == 404
    assert env.client.get(f"/api/socials/posts/{theirs['id']}/media/video-9x16.mp4").status_code == 404
    assert env.launched == []


def _real_renderer_check(monkeypatch, handler=None, *, storage=True):
    """The real ensure_renderer, against a mocked transport (None = never called)."""
    monkeypatch.setattr(render, "ensure_renderer", _REAL_ENSURE_RENDERER)
    monkeypatch.setattr(media_store, "is_storage_configured", lambda: storage)
    calls = []

    def recording(request):
        calls.append(request.url.path)
        if handler is None:
            raise AssertionError(f"media-render was called: {request.url.path}")
        return handler(request)

    def client():
        return httpx.AsyncClient(transport=httpx.MockTransport(recording))

    monkeypatch.setattr(media_render_client, "_get_client", client)
    return calls


_REAL_ENSURE_RENDERER = render.ensure_renderer


@pytest.mark.parametrize(
    "edition, url, failure, message",
    [
        ("local", "", None, render.MEDIA_PROFILE_MESSAGE),
        ("local", RENDER_URL, httpx.ConnectError("Name or service not known"), render.MEDIA_PROFILE_MESSAGE),
        ("saas", "", None, render.NOT_CONFIGURED_MESSAGE),
        ("saas", RENDER_URL, httpx.ConnectError("connection refused"), render.UNREACHABLE_MESSAGE),
    ],
)
def test_no_renderer_is_503_and_changes_nothing(env, monkeypatch, edition, url, failure, message):
    post = _renderable(env)

    def handler(request):
        raise failure

    _real_renderer_check(monkeypatch, handler if failure else None)
    _set_config(monkeypatch, AUTH_EDITION=edition, SOCIALS_RENDER_URL=url)
    resp = env.client.post(f"/api/socials/posts/{post['id']}/render")
    assert resp.status_code == 503 and resp.json()["detail"] == message
    assert env.launched == [] and _post(env, post["id"]).status == "draft"


def test_no_storage_is_503_before_the_renderer_is_asked(env, monkeypatch):
    post = _renderable(env)
    calls = _real_renderer_check(monkeypatch, storage=False)
    resp = env.client.post(f"/api/socials/posts/{post['id']}/render")
    assert resp.status_code == 503 and resp.json()["detail"] == render.STORAGE_MESSAGE
    assert calls == [] and env.launched == []


def test_a_healthy_renderer_is_checked_before_the_render_starts(env, monkeypatch):
    post = _renderable(env)
    calls = _real_renderer_check(monkeypatch, lambda request: httpx.Response(200, json={"status": "healthy"}))
    resp = env.client.post(f"/api/socials/posts/{post['id']}/render")
    assert resp.status_code == 202 and calls == ["/health"]


# ---------------------------------------------------------------------------
# The quota (owner, 2026-09-23): Basic 10, Pro 60, Business 240; none locally
# ---------------------------------------------------------------------------


def test_the_plan_tiers_carry_the_owners_render_quotas():
    from config import PLAN_TIERS

    assert {plan: PLAN_TIERS[plan]["render_minutes_month"] for plan in ("basic", "pro", "business")} == {
        "basic": 10, "pro": 60, "business": 240,
    }
    assert "render_minutes_month" not in PLAN_TIERS["enterprise"]
    for plan, minutes in (("basic", 10), ("pro", 60), ("business", 240)):
        assert plan_tiers.plan_limits_for_tier(plan)["render_minutes_month"] == minutes


def test_a_plan_change_rewrites_the_quota():
    workspace = SimpleNamespace(plan="basic", plan_limits={})
    plan_tiers.assign_plan(None, workspace, "pro")
    assert workspace.plan_limits["render_minutes_month"] == 60
    plan_tiers.assign_plan(None, workspace, "business")
    assert workspace.plan_limits["render_minutes_month"] == 240
    # A tier without the key writes None, so no stale quota survives the move.
    tiers = {"custom": {"assignable": True, "seats": 1, "max_agents": 1, "mission_concurrency": 1,
                        "watcher_limit": 1, "marketplace_depth": 1, "budget_usd": 0}}
    plan_tiers.assign_plan(None, workspace, "custom", tiers=tiers)
    assert workspace.plan_limits["render_minutes_month"] is None
    assert render_quota.quota_minutes_for(workspace) is None


@pytest.mark.parametrize(
    "plan, limits, expected",
    [
        ("basic", {}, 10.0),
        ("pro", {}, 60.0),
        ("business", {}, 240.0),
        ("enterprise", {}, None),
        (None, {}, 10.0),                          # no plan: the entry tier
        ("starter", {}, 10.0),                     # a stray plan string: the entry tier
        ("basic", {"render_minutes_month": 30}, 30.0),  # the assigned value wins
        ("basic", {"render_minutes_month": 0}, None),   # 0 = no quota, as for max_agents
        ("basic", {"render_minutes_month": -1}, None),
        ("basic", {"render_minutes_month": None}, None),
        ("basic", {"render_minutes_month": "ten"}, None),
    ],
)
def test_the_quota_a_workspace_gets(monkeypatch, plan, limits, expected):
    _set_config(monkeypatch, AUTH_EDITION="saas")
    workspace = SimpleNamespace(id=WS, plan=plan, plan_limits=limits)
    assert render_quota.quota_minutes_for(workspace) == expected


def test_the_local_edition_has_no_quota(monkeypatch):
    _set_config(monkeypatch, AUTH_EDITION="local")
    assert render_quota.quota_minutes_for(SimpleNamespace(plan="basic", plan_limits={"render_minutes_month": 10})) is None


def test_minutes_used_are_this_months_render_units_on_the_media_lane(env):
    now = datetime.now(timezone.utc)
    start, _ = render_quota.month_window_utc(now)
    last_month = (start - timedelta(days=2)).replace(tzinfo=None)
    env.session.add_all([
        _usage_row(WS, 300),
        _usage_row(WS, 90),
        _usage_row(WS, 900, at=last_month),                   # last month's renders are not this month's
        _usage_row(WS, 5, provider="fal_ai"),                 # footage spend is not render minutes
        _usage_row(WS, 100, provider="openrouter", lane="chat"),
        _usage_row(WS_OTHER, 600),                            # another workspace's renders never count
    ])
    env.session.commit()

    workspace = env.session.get(Workspace, WS)
    reading = render_quota.render_quota(env.session, workspace, now)
    assert reading.used_seconds == 390 and reading.quota_minutes == 10.0 and not reading.exhausted
    assert reading.to_dict()["used_minutes"] == 6.5 and reading.to_dict()["remaining_minutes"] == 3.5

    body = env.client.get("/api/socials/usage").json()["render_minutes"]
    assert body["used_minutes"] == 6.5 and body["used_seconds"] == 390 and body["quota_minutes"] == 10.0
    assert body["exhausted"] is False and body["period_start"] == start.isoformat()


def test_a_render_past_the_quota_is_refused_before_any_call_to_media_render(env, monkeypatch):
    post = _renderable(env)
    calls = _real_renderer_check(monkeypatch)                # any call to media-render fails the test
    env.session.add(_usage_row(WS, 600))                     # Basic: 10 minutes, all used
    env.session.commit()

    resp = env.client.post(f"/api/socials/posts/{post['id']}/render")

    assert resp.status_code == 429
    detail = resp.json()["detail"]
    assert "used 10.0 of its 10 render minutes this month on the Basic plan" in detail
    assert "Rendering resumes on 1 " in detail
    assert calls == [] and env.launched == []
    assert _post(env, post["id"]).status == "draft"


def test_a_render_with_minutes_left_starts(env):
    post = _renderable(env)
    env.session.add(_usage_row(WS, 599))
    env.session.commit()
    assert env.client.post(f"/api/socials/posts/{post['id']}/render").status_code == 202


def test_a_bigger_plan_has_room_where_basic_has_none(env):
    env.session.add_all([_usage_row(WS, 600), _usage_row(WS_PRO, 600)])
    env.session.commit()
    env.ctx = _ctx(WS_PRO, "member-pro")
    post = _create(env, template_id=str(_template(env, workspace_id=WS_PRO)))
    assert env.client.post(f"/api/socials/posts/{post['id']}/render").status_code == 202
    assert env.client.get("/api/socials/usage").json()["render_minutes"]["quota_minutes"] == 60.0


def test_the_local_edition_renders_past_any_count(env, monkeypatch):
    _set_config(monkeypatch, AUTH_EDITION="local")
    post = _renderable(env)
    env.session.add(_usage_row(WS, 100_000))
    env.session.commit()
    assert env.client.post(f"/api/socials/posts/{post['id']}/render").status_code == 202
    assert env.client.get("/api/socials/usage").json()["render_minutes"]["quota_minutes"] is None


# ---------------------------------------------------------------------------
# GET /media — a rendered file, streamed from storage
# ---------------------------------------------------------------------------


def test_the_media_route_streams_a_rendered_file(env, monkeypatch):
    post = _renderable(env)
    _, job = _start(env, post)
    store = FakeStore()
    _run(job, Renderer(), store, env.factory)
    monkeypatch.setattr(media_store, "MediaStore", lambda *a, **k: store)

    resp = env.client.get(f"/api/socials/posts/{post['id']}/media/video-9x16.mp4")
    assert resp.status_code == 200 and resp.content == MP4
    assert resp.headers["content-type"] == "video/mp4"
    assert resp.headers["content-disposition"] == 'inline; filename="video-9x16.mp4"'
    assert resp.headers["cache-control"] == "private, no-cache"

    assert env.client.get(f"/api/socials/posts/{post['id']}/media/image-1x1.png").status_code == 404
    assert env.client.get(f"/api/socials/posts/{post['id']}/media/VIDEO.mp4").status_code == 404
    assert env.client.get(f"/api/socials/posts/{post['id']}/media/..mp4").status_code == 404


def test_media_keys_never_leave_their_posts_prefix():
    assert media_store.media_key(WS, "p1", "video-9x16.mp4") == f"social-media/{WS}/p1/video-9x16.mp4"
    for bad in ("../x.mp4", "a/b.mp4", "Video.mp4", "..", ".hidden", "", "x" * 121):
        with pytest.raises(media_store.MediaNameError):
            media_store.media_key(WS, "p1", bad)


# ---------------------------------------------------------------------------
# The client (core/media_render_client.py)
# ---------------------------------------------------------------------------


def test_the_shared_client_sends_the_token_with_configs_timeouts(monkeypatch):
    _set_config(
        monkeypatch, SOCIALS_RENDER_TOKEN=TOKEN, SOCIALS_RENDER_TIMEOUT_SECONDS=900,
        SOCIALS_RENDER_CONNECT_TIMEOUT_SECONDS=10,
    )
    monkeypatch.setattr(media_render_client, "_client", None)

    async def build():
        client = media_render_client._get_client()
        try:
            return dict(client.headers), client.timeout
        finally:
            await client.aclose()

    headers, timeout = asyncio.run(build())
    assert headers["x-internal-token"] == TOKEN
    assert timeout.read == 900.0 and timeout.connect == 10.0
    monkeypatch.setattr(media_render_client, "_client", None)


def _call(handler, method, *args):
    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await getattr(MediaRenderClient(http), method)(*args)

    return asyncio.run(go())


def test_the_client_reads_the_renderers_answers(monkeypatch):
    _set_config(monkeypatch, SOCIALS_RENDER_URL=RENDER_URL + "/")

    accepted = _call(lambda r: httpx.Response(202, json={"id": JOB_ID, "status": "queued"}), "submit", {"workspace_id": "w"})
    assert accepted == {"id": JOB_ID, "status": "queued"}

    with pytest.raises(MediaRenderError) as refused:
        _call(
            lambda r: httpx.Response(422, json={"error": "check_failed", "message": "1 error", "findings": [CHECK_FINDING], "report": {"check": {}}}),
            "submit", {},
        )
    assert refused.value.code == "check_failed" and refused.value.findings == (CHECK_FINDING,)
    assert refused.value.status == 422 and not isinstance(refused.value, MediaRenderUnavailable)

    with pytest.raises(MediaRenderError) as busy:
        _call(lambda r: httpx.Response(503, json={"error": "busy", "message": "full"}, headers={"Retry-After": "30"}), "submit", {})
    assert busy.value.code == "busy" and busy.value.retry_after == 30.0

    with pytest.raises(MediaRenderError) as gone:
        _call(lambda r: httpx.Response(404, json={"error": "not_found", "message": "no such render"}), "job", JOB_ID)
    assert gone.value.code == "not_found" and gone.value.status == 404


def test_the_client_maps_transport_failures(monkeypatch):
    _set_config(monkeypatch, SOCIALS_RENDER_URL=RENDER_URL)

    def refused(request):
        raise httpx.ConnectError("connection refused", request=request)

    def slow(request):
        raise httpx.ReadTimeout("read timed out", request=request)

    with pytest.raises(MediaRenderUnavailable) as unreachable:
        _call(refused, "health")
    assert unreachable.value.code == "unreachable"
    with pytest.raises(MediaRenderError) as timeout:
        _call(slow, "submit", {})
    assert timeout.value.code == "timeout" and not isinstance(timeout.value, MediaRenderUnavailable)

    _set_config(monkeypatch, SOCIALS_RENDER_URL="")
    with pytest.raises(MediaRenderUnavailable) as unset:
        _call(lambda r: httpx.Response(200), "health")
    assert unset.value.code == "not_configured"


def test_a_download_streams_to_disk_with_its_sha256(monkeypatch, tmp_path):
    _set_config(monkeypatch, SOCIALS_RENDER_URL=RENDER_URL)
    paths = []

    def handler(request):
        paths.append(request.url.path)
        return httpx.Response(200, content=MP4)

    target = tmp_path / "render.mp4"
    size, digest = _call(handler, "download", JOB_ID, "render.mp4", target)
    assert (size, digest) == (len(MP4), MP4_SHA) and target.read_bytes() == MP4
    assert paths == [f"/render/{JOB_ID}/output/render.mp4"]


# ---------------------------------------------------------------------------
# The boot reaper: a render stranded by a restart
# ---------------------------------------------------------------------------


def test_the_boot_reaper_fails_a_render_stranded_by_a_restart(env, monkeypatch):
    monkeypatch.setattr(reaper, "record_error", MagicMock())
    stranded, fresh = _renderable(env), _renderable(env)
    for post in (stranded, fresh):
        _start(env, post)
    now = datetime.now(timezone.utc)
    env.session.execute(
        sa.update(SocialPost.__table__)
        .where(SocialPost.__table__.c.id == uuid.UUID(stranded["id"]))
        .values(updated_at=now - timedelta(hours=2))
    )
    env.session.commit()

    cutoff = now - timedelta(minutes=30)
    assert reaper._reap_social_renders(env.session, cutoff, now) == 1
    env.session.commit()

    row = _post(env, stranded["id"])
    assert row.status == "failed"
    assert row.review_log[-1]["action"] == "render_failed" and row.review_log[-1]["by"] == "orphaned_on_restart"
    assert _post(env, fresh["id"]).status == "rendering"
    kwargs = reaper.record_error.call_args.kwargs
    assert kwargs["subsystem"] == "socials" and kwargs["operation"] == "boot_reap"


def test_the_reaper_sweeps_social_renders_with_the_other_surfaces():
    swept = []

    class _Session:
        def query(self, model):
            swept.append(model)
            return SimpleNamespace(filter=lambda *a: SimpleNamespace(all=lambda: []))

        def commit(self):
            pass

    reaper.reap_orphaned_runs(_Session(), now=datetime(2026, 9, 25, 12, tzinfo=timezone.utc))
    assert SocialPost in swept


# ---------------------------------------------------------------------------
# Compose, Railway, the config surface
# ---------------------------------------------------------------------------


def test_compose_runs_media_render_only_under_the_media_profile():
    compose = yaml.safe_load((_ROOT / "docker-compose.yml").read_text())
    service_def = compose["services"]["media-render"]
    assert service_def["profiles"] == ["media"]
    assert service_def["build"]["context"] == "./services/media-render"
    assert service_def["deploy"]["resources"]["limits"] == {"cpus": "4.0", "memory": "8G"}
    assert "SOCIALS_RENDER_TOKEN" in service_def["environment"]
    # The default stack never waits on the optional profile.
    assert "media-render" not in (compose["services"]["backend"].get("depends_on") or {})
    assert "SOCIALS_RENDER_TOKEN" in compose["services"]["backend"]["environment"]
    defaults = (_ROOT / "envs" / "api.defaults").read_text().splitlines()
    assert "SOCIALS_RENDER_URL=http://media-render:8090" in defaults


def test_the_railway_manifest_carries_media_render_at_the_owners_size():
    manifest = json.loads((_ROOT / "infrastructure" / "railway-manifest.json").read_text())
    entry = manifest["services"]["media-render"]
    assert entry["source"] == {"repo": "AutomatosAI/automatos-ai", "root_dir": "/services/media-render"}
    assert entry["builder"] == "DOCKERFILE" and entry["port"] == 8090 and entry["healthcheck"] == "/health"
    assert entry["resources"] == {"vcpu": 4, "memory_gb": 8} and entry["replicas"] == 1
    assert "SOCIALS_RENDER_TOKEN" in entry["env_keys"] and "MEDIA_RENDER_MEDIA_URL_PREFIXES" in entry["env_keys"]
    assert "media-render" in manifest["service_groups"]["core"]["services"]
    assert "media-render" in manifest["network_topology"]["internal_only"]
    assert {"SOCIALS_RENDER_URL", "SOCIALS_RENDER_TOKEN"} <= set(manifest["services"]["automatos-ai-api"]["env_keys"])


def test_the_config_surface_guards_the_render_settings_and_wave_0s():
    names = json.loads((_ORCH / "reports" / "config-surface.json").read_text())["settings"]
    assert names == sorted(names)
    wanted = {
        "SOCIALS_RENDER_TOKEN", "SOCIALS_RENDER_TIMEOUT_SECONDS", "SOCIALS_RENDER_CONNECT_TIMEOUT_SECONDS",
        "SOCIALS_RENDER_POLL_SECONDS", "SOCIALS_RENDER_MAX_WAIT_SECONDS",
        # Wave 0's six, which it never added.
        "SOCIALS_ENABLED_DEFAULT", "SOCIALS_MEDIA_URL_TTL_SECONDS", "SOCIALS_MISFIRE_GRACE_SECONDS",
        "SOCIALS_MAX_TARGET_ATTEMPTS", "SOCIALS_RENDER_URL", "SOCIALS_PUBLIC_MEDIA_BUCKET",
    }
    assert wanted <= set(names)


def test_the_render_wait_stays_under_the_reaper_cutoff():
    """A live render is never reaped: the longest wait is under the stale cutoff."""
    from config import config

    assert config.SOCIALS_RENDER_MAX_WAIT_SECONDS < config.BOOT_REAPER_STALE_MINUTES * 60


# ---------------------------------------------------------------------------
# US-107: an image renders as stills; a carousel keeps every slide it shows
# ---------------------------------------------------------------------------


def _carousel_blocks():
    from modules.documents.social_starters import social_starters

    (carousel,) = [s for s in social_starters("social_image") if s["slug"] == "carousel"]
    return carousel["blocks"]


def test_a_carousel_renders_a_png_per_slide_it_shows_stored_and_recorded_in_order(env):
    post = _create(
        env,
        template_id=str(_template(env, blocks=_carousel_blocks(), fmt="social_image")),
        format="carousel",
        variables={
            "headline": {"value": "4 SIGNS|YOUR AGENT|NEEDS A|HARNESS", "claim": False},
            "point_1_title": {"value": "It forgets what it just did", "claim": False},
            "point_2_title": {"value": "It picks the wrong tool", "claim": False},
            "closing_title": {"value": "Build the harness first.", "claim": False},
        },
    )
    _, job = _start(env, post)
    pngs = {f"render-{i:02d}.png": b"\x89PNG\r\n\x1a\n" + bytes([i]) * 64 for i in range(1, 5)}
    outputs = [
        {"name": name, "kind": "still", "index": i, "at": at, "aspect": "4:5", "width": 1080, "height": 1350, "bytes": 72}
        for i, (name, at) in enumerate(zip(sorted(pngs), (0.5, 1.5, 2.5, 7.5)), start=1)
    ]
    renderer, store = Renderer(outputs=outputs, files=pngs), FakeStore()
    assert _run(job, renderer, store, env.factory) is True

    # Points 3-6 are empty, so their slides are not taken: a still for the cover, the
    # two points and the close. An image has no sound and is no preview.
    (bundle,) = renderer.bundles
    assert bundle["still"] == {"at": [0.5, 1.5, 2.5, 7.5]}
    assert "audio" not in bundle and "preview" not in bundle
    assert bundle["variables"]["size.width"] == 1080 and bundle["variables"]["size.height"] == 1350

    # Each slide is its own file, numbered in order, stored and registered as an image.
    names = [f"carousel-4x5-{i:02d}.png" for i in range(1, 5)]
    keys = [f"social-media/{WS}/{post['id']}/{name}" for name in names]
    assert sorted(store.objects) == keys
    assert [store.objects[key] for key in keys] == [(pngs[f"render-{i:02d}.png"], "image/png") for i in range(1, 5)]
    assert [call["artifact_type"] for call in Deliverables.calls] == ["image"] * 4
    assert [call["file_path"] for call in Deliverables.calls] == keys

    row = _post(env, post["id"])
    assert row.status == "needs_approval"
    records = row.media["4:5"]
    assert [record["name"] for record in records] == names
    assert all("duration" not in r and (r["width"], r["height"]) == (1080, 1350) for r in records)
    assert row.content_hash == service.compute_content_hash(row)
    assert row.review_log[-1]["comment"] == "Rendered 4 images at 1080×1350."
    # A still has no duration: the render spends no minutes.
    (booking,) = env.booked
    assert booking["units"] == 0.0


def test_one_output_keeps_its_plain_name_and_several_are_numbered():
    job = SimpleNamespace(format="image")
    still = {"name": "render.png", "index": 1, "aspect": "4:5"}
    assert render.stored_file_name(job, still) == "image-4x5.png"
    assert render.stored_file_name(job, still, several=True) == "image-4x5-01.png"
    assert render.stored_file_name(job, {**OUTPUT, "index": 3}) == "image-9x16.mp4"

