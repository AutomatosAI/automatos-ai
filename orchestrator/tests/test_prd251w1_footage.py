"""PRD-251 Wave 1, US-114 (S1.8, D12, D13) — footage and stills from the workspace's Composio toolkits.

The real Socials router runs on a mini FastAPI app over in-memory SQLite (the
social tables, SQLite copies of ``workspaces``, ``document_templates`` and the
Composio tables, an untyped ``llm_usage`` the real usage tracker writes, and
``system_settings`` seeded by the two migrations: the Wave 0 deny list and the
US-110 media allowlist). The media capability registry reads them for real.
Composio is mocked where the footage recipe calls it (``ComposioToolExecutor``),
a toolkit's file link where it is fetched, media-render at the HTTP layer
(``httpx.MockTransport``, the real client), and storage and the Deliverable
registry are recording fakes. Pins:

* AC1 — with ``fal_ai`` connected, the template's hook slot is filled by a
  generated 5 s 9:16 clip: priced by fal's estimate action, submitted, polled,
  its file copied into our storage and handed to media-render at the slot's
  path, registered as a Deliverable, and the post's footage and the media lane
  both carry the estimate;
* AC2 — an estimate over the workspace's monthly media cap (or the post's cap)
  submits nothing and the render says why; a cap that cannot be read spends
  nothing; a booking is seen by the next render's cap check;
* AC3 — the file is in our storage before its slot is marked done; a file that
  cannot be stored is never marked done, and what the toolkit spent is booked;
* AC4 — Higgsfield MCP (credit-billed): one ``params`` string, JOBS_WAIT polled,
  the balance difference booked at the credit's price; an unreadable balance
  submits nothing. Kie.ai makes a still and books its credits the same way;
* AC5 — with no generation toolkit connected, the footage slots play the
  template's own motion graphics, and the report says why;
* a slot made for its prompt is reused, a new prompt asks again, and footage is
  a render setting outside the content hash; a save names only slots the
  template has and lets a toolkit fill; a denied action is never used;
* the column: ``social_posts.footage`` comes from the wave's one migration,
  create_all-first safe (``@integration`` on Postgres), and the footage window
  holds across worker processes (``@integration``).
"""
from __future__ import annotations

import ast
import asyncio
import importlib.util
import json
import os
import sys
import uuid
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
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

import httpx  # noqa: E402
import sqlalchemy as sa  # noqa: E402
from alembic.operations import Operations  # noqa: E402
from alembic.runtime.migration import MigrationContext  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.dialects.postgresql import ARRAY, JSONB  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import core.models  # noqa: E402,F401  (registers every mapper)
import api.socials as socials_api  # noqa: E402
import core.auth.workspace_permission as permission_mod  # noqa: E402
import core.best_effort as best_effort  # noqa: E402
import core.composio.deny_list as deny_list  # noqa: E402
import core.database.database as database_mod  # noqa: E402
import core.media_render_client as media_render_client  # noqa: E402
import core.media_render_quota as render_quota  # noqa: E402
import modules.socials.media_caps as media_caps  # noqa: E402
import modules.socials.media_store as media_store  # noqa: E402
import modules.socials.recipes.footage as footage  # noqa: E402
import modules.socials.recipes.footage_toolkits as toolkits  # noqa: E402
import modules.socials.recipes.toolkit as toolkit  # noqa: E402
import modules.socials.render as render  # noqa: E402
import modules.socials.service as service  # noqa: E402
import modules.socials.settings as socials_settings  # noqa: E402
import services.deliverable_service as deliverable_service  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.database.database import Base, get_db  # noqa: E402
from core.media_render_client import MediaRenderClient  # noqa: E402
from core.models.composio import ComposioConnection, ComposioEntity  # noqa: E402
from core.models.composio_cache import ComposioActionCache  # noqa: E402
from core.models.core import DocumentTemplate, LLMUsage  # noqa: E402
from core.models.socials import SocialPost, SocialPostTarget  # noqa: E402
from core.models.system_settings import SystemSetting  # noqa: E402
from core.models.workspaces import Workspace  # noqa: E402
from modules.socials.recipes.files import ReturnedFile  # noqa: E402

VERSIONS = _ORCH / "alembic" / "versions"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WAVE0 = _load(VERSIONS / "prd251_socials.py", "prd251_socials_migration_footage")
WAVE1 = _load(VERSIONS / "prd251_wave1.py", "prd251_wave1_migration_footage")

WS = uuid.UUID("00000000-0000-0000-0000-0000000001b4")
CREATED = datetime(2026, 9, 1, 9, 0)
RENDER_URL = "http://media-render:8090"
TOKEN = "render-secret"
JOB_ID = "f" * 32
RENDERED = b"\x00\x00\x00\x18ftypmp42" + b"rendered-frames " * 300
OUTPUT = {"name": "render.mp4", "aspect": "9:16", "width": 1080, "height": 1920, "bytes": len(RENDERED), "duration": 6.0}

# What the toolkits return: a clip, a still, and audio (not footage).
CLIP = b"\x00\x00\x00\x18ftypisom" + b"generated-clip " * 500
STILL = b"\x89PNG\r\n\x1a\n" + b"generated-still " * 200
AUDIO = b"\x00\x00\x00\x20ftypM4A " + b"not-footage " * 100

FAL_MODEL = "fal-ai/kling-video/v2.1/standard/text-to-video"
FAL_IMAGE_MODEL = "fal-ai/flux-pro/v1.1-ultra"
FAL_SUBMIT, FAL_STATUS = "FAL_AI_SUBMIT_ASYNC_JOB", "FAL_AI_QUEUE_GET_STATUS"
FAL_RESULT, FAL_ESTIMATE = "FAL_AI_GET_QUEUE_REQUEST_RESULT", "FAL_AI_ESTIMATE_PRICING"
KIE_FLUX, KIE_FLUX_DETAILS, KIE_CREDITS = (
    "KIEAI_GENERATE_FLUX_KONTEXT_IMAGE", "KIEAI_GET_FLUX_KONTEXT_IMAGE_DETAILS", "KIEAI_GET_ACCOUNT_CREDITS",
)
HF_VIDEO, HF_WAIT, HF_BALANCE = "HIGGSFIELD_MCP_GENERATE_VIDEO", "HIGGSFIELD_MCP_JOBS_WAIT", "HIGGSFIELD_MCP_BALANCE"
PROMPT = "a phone lighting up with orders on a café counter at night, warm practicals, slow dolly-in"
GUARD = "no readable text, no logos"

# The cached input schemas, as docs.composio.dev lists the actions (2026-09-26).
SCHEMAS = {
    FAL_SUBMIT: {
        "type": "object",
        "properties": {"model_id": {"type": "string"}, "input": {"type": "object"}, "priority": {"type": "string"},
                       "webhook_url": {"type": "string"}, "timeout_seconds": {"type": "integer"}},
        "required": ["model_id", "input"],
    },
    FAL_STATUS: {
        "type": "object",
        "properties": {"model_id": {"type": "string"}, "request_id": {"type": "string"}, "logs": {"type": "integer"}},
        "required": ["model_id", "request_id"],
    },
    FAL_RESULT: {
        "type": "object",
        "properties": {"model_id": {"type": "string"}, "request_id": {"type": "string"}},
        "required": ["model_id", "request_id"],
    },
    FAL_ESTIMATE: {
        "type": "object",
        "properties": {"estimate_type": {"type": "string", "enum": ["historical_api_price", "unit_price"]},
                       "endpoints": {"type": "object"}},
        "required": ["estimate_type", "endpoints"],
    },
    KIE_FLUX: {
        "type": "object",
        "properties": {"prompt": {"type": "string"}, "model": {"type": "string"}, "aspect_ratio": {"type": "string"},
                       "output_format": {"type": "string"}, "input_image": {"type": "string"}},
        "required": ["prompt"],
    },
    KIE_FLUX_DETAILS: {"type": "object", "properties": {"task_id": {"type": "string"}}, "required": ["task_id"]},
    HF_VIDEO: {"type": "object", "properties": {"params": {"type": "string"}}, "required": ["params"]},
    HF_WAIT: {
        "type": "object",
        "properties": {"jobs": {"type": "array", "items": {"type": "string"}}, "timeout_seconds": {"type": "integer"}},
        "required": ["jobs"],
    },
}

SLOTS = {
    "hook": {"kind": "video", "path": "assets/slots/hook.mp4", "label": "Hook footage",
             "description": "9:16, 4-5 s, no readable text or logos."},
    "still": {"kind": "image", "path": "assets/slots/still.png", "label": "Still behind the chart"},
    "app_loop": {"kind": "video", "path": "assets/slots/app_loop.mp4", "label": "The app's own loop", "generate": False},
}
HTML = (
    '<!doctype html><html><head></head><body><div id="root" data-composition-id="main" data-width="1080" '
    'data-height="1920" data-duration="6">'
    '<div class="bg"><video id="hook" class="clip" data-slot="hook" src="assets/slots/hook.mp4" muted playsinline></video></div>'
    '<div class="plate"><img id="still" data-slot="still" src="assets/slots/still.png"></div>'
    '<div class="phone"><video id="loop" data-slot="app_loop" src="assets/slots/app_loop.mp4" muted></video></div>'
    "<h1>{{ headline }}</h1></div></body></html>"
)
COMPOSITION = {
    "html": HTML,
    "css": "h1 { color: var(--brand-text); }",
    "variables_schema": {"headline": {"type": "text"}},
    "sizes": ["1080x1920"],
    "slots": SLOTS,
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
    return sa.Table(table.name, metadata, *[sa.Column(c.name, _portable(c.type), primary_key=c.primary_key) for c in table.columns])


def _ctx(workspace_id, user_id="member-1"):
    return RequestContext(
        workspace_id=workspace_id,
        user=UserContext(id=user_id, clerk_user_id=f"clerk-{user_id}", system_role="user"),
        auth_type="clerk",
    )


class FakeStore:
    """The documents bucket, in memory; a presigned link names the key it signs."""

    def __init__(self, events=None, fail=False):
        self.objects = {}
        self.events = events if events is not None else []
        self.fail = fail

    def configured(self):
        return True

    def put_file(self, key, path, content_type):
        if self.fail:
            raise OSError("the bucket is down")
        self.objects[key] = (Path(path).read_bytes(), content_type)
        self.events.append(("stored", key))

    def presigned_get(self, key, ttl_seconds):
        return f"http://minio:9000/automatos-ai/{key}?X-Amz-Expires={ttl_seconds}&X-Amz-Signature=sig"

    def open(self, key):
        return None


class Deliverables:
    """Stands in for DeliverableService (its SQL is Postgres-only): records register()."""

    calls: list = []

    def __init__(self, db, workspace_id):
        self.workspace_id = workspace_id

    def register(self, **kwargs):
        type(self).calls.append(kwargs)
        return {"success": True, "deliverable_id": f"d-{len(type(self).calls)}", "created": True}


class Renderer:
    """media-render over httpx.MockTransport; it notes what storage held when each bundle arrived."""

    def __init__(self, store):
        self.store = store
        self.bundles = []
        self.stored_at_submit = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path == "/render":
            self.bundles.append(json.loads(request.content))
            self.stored_at_submit.append(sorted(self.store.objects))
            return httpx.Response(202, json={"id": JOB_ID, "status": "checking", "outputs": [], "report": {}})
        if request.method == "GET" and path == f"/render/{JOB_ID}":
            return httpx.Response(200, json={"id": JOB_ID, "status": "done", "outputs": [OUTPUT], "report": {"check": {"ok": True}}})
        if request.method == "GET" and path == f"/render/{JOB_ID}/output/render.mp4":
            return httpx.Response(200, content=RENDERED)
        return httpx.Response(404, json={"error": "not_found"})


class Composio:
    """ComposioToolExecutor as the footage recipe calls it: every call recorded,
    answered from the test's script (slug → answer, or a list of answers)."""

    def __init__(self):
        self.calls = []
        self.answers = {}

    def executor(self):
        state = self

        class Executor:
            def __init__(self, db):
                self.db = db

            async def execute(self, *, action, params, agent_id, workspace_id, app_name=None, skip_validation=False):
                state.calls.append(
                    {"action": action, "params": params, "agent_id": agent_id, "workspace_id": workspace_id,
                     "app_name": app_name, "skip_validation": skip_validation}
                )
                answer = state.answers[action]
                if isinstance(answer, list):
                    answer = answer.pop(0)
                return answer(params) if callable(answer) else answer

        return Executor

    def slugs(self):
        return [call["action"] for call in self.calls]

    def params(self, slug):
        return [call["params"] for call in self.calls if call["action"] == slug]


def _ok(data):
    return {"success": True, "data": {"data": data, "successful": True, "error": None}, "error": None}


def _failed(error):
    return {"success": False, "data": None, "error": error}


_COPIES = sa.MetaData()
for _table in (
    Workspace.__table__,
    DocumentTemplate.__table__,
    ComposioEntity.__table__,
    ComposioConnection.__table__,
    ComposioActionCache.__table__,
):
    _sqlite_copy(_table, _COPIES)

CONFIG = dict(
    AUTH_EDITION="saas",
    SOCIALS_RENDER_URL=RENDER_URL,
    SOCIALS_RENDER_TOKEN=TOKEN,
    SOCIALS_RENDER_POLL_SECONDS=0,
    SOCIALS_RENDER_MAX_WAIT_SECONDS=60,
    SOCIALS_RENDER_MEDIA_URL_TTL_SECONDS=3600,
    SOCIALS_MEDIA_FETCH_TIMEOUT_SECONDS=60,
    SOCIALS_FOOTAGE_TOOLKITS="fal_ai,kieai,higgsfield_mcp",
    SOCIALS_FOOTAGE_FAL_VIDEO_MODEL=FAL_MODEL,
    SOCIALS_FOOTAGE_FAL_IMAGE_MODEL=FAL_IMAGE_MODEL,
    SOCIALS_FOOTAGE_KIEAI_VIDEO_MODEL="veo3_fast",
    SOCIALS_FOOTAGE_KIEAI_IMAGE_MODEL="flux-kontext-pro",
    SOCIALS_FOOTAGE_HIGGSFIELD_VIDEO_MODEL="kling3_0",
    SOCIALS_FOOTAGE_HIGGSFIELD_IMAGE_MODEL="gpt_image_2",
    SOCIALS_FOOTAGE_CLIP_SECONDS=5,
    SOCIALS_FOOTAGE_PROMPT_GUARD=GUARD,
    SOCIALS_FOOTAGE_POLL_SECONDS=0,
    SOCIALS_FOOTAGE_MAX_WAIT_SECONDS=60,
    SOCIALS_FOOTAGE_MAX_BYTES=64 * 1024 * 1024,
    SOCIALS_FOOTAGE_CEILING_VIDEO_USD=2.5,
    SOCIALS_FOOTAGE_CEILING_IMAGE_USD=0.3,
    SOCIALS_MEDIA_POST_CAP_USD=10.0,
    SOCIALS_MEDIA_MONTHLY_CAP_USD=30.0,
    SOCIALS_KIEAI_USD_PER_CREDIT=0.005,
    SOCIALS_HIGGSFIELD_USD_PER_CREDIT=0.0625,
)


def _set_config(monkeypatch, **values):
    """Patch the config object every module under test reads (one object, unless a reload split it)."""
    modules = (render, render_quota, media_render_client, media_store, footage, toolkits, toolkit, media_caps,
               socials_api, socials_settings)
    for cfg in {id(m.config): m.config for m in modules}.values():
        for name, value in values.items():
            monkeypatch.setattr(cfg, name, value, raising=False)


@pytest.fixture
def env(monkeypatch):
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    _COPIES.create_all(engine)
    SocialPost.metadata.create_all(engine, tables=[SocialPost.__table__, SocialPostTarget.__table__])
    SystemSetting.__table__.create(bind=engine)
    # llm_usage untyped: SQLAlchemy 2.0.23 cannot compile the Postgres UUID type for SQLite.
    columns = ", ".join(f"{c.name} INTEGER PRIMARY KEY" if c.primary_key else c.name for c in LLMUsage.__table__.columns)
    with engine.begin() as conn:
        conn.exec_driver_sql(f"CREATE TABLE {LLMUsage.__tablename__} ({columns})")
        WAVE0._seed_settings(conn, WAVE0._composio_settings_seed())
        WAVE1.seed_settings(conn, WAVE1.settings_seed())

    factory = sessionmaker(bind=engine)
    monkeypatch.setattr(database_mod, "SessionLocal", factory)
    deny_list.reset_cache()
    session = factory()
    session.add(
        Workspace(
            id=WS, name="Harbourline", plan="pro", plan_limits={},
            settings={"socials": {"enabled": True}}, onboarding={}, created_at=CREATED, updated_at=CREATED,
        )
    )
    session.commit()

    state = SimpleNamespace(session=session, engine=engine, factory=factory, ctx=_ctx(WS), role="owner", launched=[])
    monkeypatch.setattr(socials_settings, "read_system_setting", lambda category, key: "true")
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: state.role)
    monkeypatch.setattr(socials_api, "_launch_render", lambda job: state.launched.append(job))

    async def renderer_up(client=None, store=None):
        return None

    monkeypatch.setattr(render, "ensure_renderer", renderer_up)
    _set_config(monkeypatch, **CONFIG)
    Deliverables.calls = []
    monkeypatch.setattr(deliverable_service, "DeliverableService", Deliverables)

    state.composio = Composio()
    monkeypatch.setattr(footage, "ComposioToolExecutor", state.composio.executor())
    state.fetched = []
    state.files = {}

    async def fake_fetch(url, *, max_bytes, timeout_seconds):
        state.fetched.append((url, max_bytes, timeout_seconds))
        return state.files[url]

    monkeypatch.setattr(footage, "fetch", fake_fetch)

    app = FastAPI()
    app.include_router(socials_api.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: state.ctx
    app.dependency_overrides[get_db] = lambda: session
    state.client = TestClient(app)
    try:
        yield state
    finally:
        best_effort.drain(timeout=5)
        session.close()
        deny_list.reset_cache()
        engine.dispose()


def _cache(env, app, *slugs):
    for slug in slugs:
        env.session.add(
            ComposioActionCache(
                app_name=app, action_name=slug, action_slug=slug.lower().replace("_", "-"),
                display_name=slug.replace("_", " ").title(), parameters=SCHEMAS.get(slug, {"type": "object"}),
            )
        )
    env.session.commit()


def _cache_fal(env):
    _cache(env, "FAL_AI", FAL_SUBMIT, FAL_STATUS, FAL_RESULT, FAL_ESTIMATE, "FAL_AI_UPLOAD_FILE", "FAL_AI_CANCEL_QUEUE_REQUEST")


def _cache_kie(env):
    _cache(env, "KIEAI", "KIEAI_GENERATE_VEO_VIDEO", "KIEAI_GET_VEO_VIDEO_DETAILS", KIE_FLUX, KIE_FLUX_DETAILS, KIE_CREDITS)


def _cache_higgsfield(env):
    _cache(env, "HIGGSFIELD_MCP", HF_VIDEO, "HIGGSFIELD_MCP_GENERATE_IMAGE", HF_WAIT, "HIGGSFIELD_MCP_JOB_STATUS",
           HF_BALANCE, "HIGGSFIELD_MCP_CONFIRM_BILLING_PURCHASE")


def _connect(env, *apps):
    entity = env.session.query(ComposioEntity).filter(ComposioEntity.workspace_id == WS).first()
    if entity is None:
        entity = ComposioEntity(workspace_id=WS, composio_entity_id=str(WS))
        env.session.add(entity)
        env.session.flush()
    for app in apps:
        env.session.add(ComposioConnection(entity_id=entity.id, app_name=app, status="active", connection_id=f"ca_{app.lower()}"))
    env.session.commit()


def _deny(env, *slugs):
    table = SystemSetting.__table__
    denied = json.dumps(list(WAVE0.COMPOSIO_DENIED_ACTIONS_SEED) + list(slugs))
    with env.engine.begin() as conn:
        conn.execute(table.update().where(table.c.category == "composio", table.c.key == "denied_actions").values(value=denied))
    deny_list.reset_cache()


def _settings(env, **socials):
    workspace = env.session.get(Workspace, WS)
    workspace.settings = {"socials": {"enabled": True, **socials}}
    env.session.commit()


def _template(env, composition=COMPOSITION):
    template_id = uuid.uuid4()
    env.session.execute(
        sa.text(
            "INSERT INTO document_templates (id, workspace_id, name, format, data_schema, blocks) "
            "VALUES (:id, :ws, :name, 'social_video', '{}', :blocks)"
        ),
        {"id": template_id.hex, "ws": WS.hex, "name": "Night orders", "blocks": json.dumps(composition)},
    )
    env.session.commit()
    return template_id


def _create(env, **body):
    payload = {
        "title": "Night orders",
        "copy": {"base": "Orders at midnight, handled."},
        "format": "video",
        "template_id": str(_template(env)),
        "variables": {"headline": {"value": "Orders at midnight.", "claim": False}},
        "footage": {"hook": {"prompt": PROMPT}},
    }
    payload.update(body)
    resp = env.client.post("/api/socials/posts", json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()


def _render(env, post_id, store):
    resp = env.client.post(f"/api/socials/posts/{post_id}/render")
    assert resp.status_code == 202, resp.text
    job = env.launched[-1]
    renderer = Renderer(store)

    async def go():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(renderer.handler), headers={"X-Internal-Token": TOKEN}
        ) as http:
            return await render.run_render(job, client=MediaRenderClient(http), store=store, session_factory=env.factory)

    result = asyncio.run(go())
    # run_render books the rendered seconds on the best-effort threads, and their
    # session shares this fixture's one SQLite connection (StaticPool): its close
    # can roll back whatever the test writes next (CI run 36220301412 lost the next
    # post's template that way). Let that write land first.
    best_effort.drain(timeout=5)
    return result, renderer, job


def _post(env, post_id):
    env.session.expire_all()
    return env.session.get(SocialPost, uuid.UUID(post_id))


def _media_rows(env, post_id=None):
    """What the media lane booked (footage and stills: not the renderer's own seconds)."""
    best_effort.drain(timeout=5)
    env.session.expire_all()
    query = env.session.query(LLMUsage).filter(LLMUsage.request_type == "media", LLMUsage.provider != "media_render")
    if post_id is not None:
        query = query.filter(LLMUsage.execution_id == f"social_post:{post_id}")
    return query.order_by(LLMUsage.id).all()


def _spent(env, usd, *, execution_id="social_post:other", created_at=None):
    """A booking already on the media lane this month."""
    env.session.add(
        LLMUsage(
            workspace_id=WS, model_id=FAL_MODEL, provider="fal_ai", tier="direct", execution_id=execution_id,
            request_type="media", input_tokens=5, output_tokens=0, total_tokens=5, cache_read_tokens=0,
            cache_write_tokens=0, input_cost=usd, output_cost=0.0, total_cost=usd, is_byok=True, status="success",
            created_at=created_at or datetime.now(timezone.utc).replace(tzinfo=None),
        )
    )
    env.session.commit()


def _script_fal(env, *, estimate=0.46, statuses=("IN_QUEUE", "IN_PROGRESS", "COMPLETED"), clip=CLIP):
    """fal.ai's answers: a price, a request id (as JSON text, as its docs list data), the statuses, then the clip."""
    link = "https://v3.fal.media/files/koala/hook-clip.mp4?expires=soon"
    env.files[link] = clip
    env.composio.answers[FAL_ESTIMATE] = _ok({"estimate_type": "historical_api_price", "total_cost": estimate, "currency": "USD"})
    env.composio.answers[FAL_SUBMIT] = _ok(json.dumps({
        "request_id": "req-hook-1", "status_url": "https://queue.fal.run/x/status", "response_url": "https://queue.fal.run/x",
    }))
    env.composio.answers[FAL_STATUS] = [_ok({"status": status, "queue_position": 0}) for status in statuses]
    env.composio.answers[FAL_RESULT] = _ok({"video": {"url": link, "content_type": "video/mp4", "file_size": len(clip)}})
    return link


def _last_log(env, post_id):
    return _post(env, post_id).review_log[-1]


# ---------------------------------------------------------------------------
# AC1 — fal.ai fills the hook with a generated 5 s 9:16 clip; the cost carries the estimate
# ---------------------------------------------------------------------------


def test_with_fal_connected_the_hook_is_a_generated_5s_9x16_clip_and_the_cost_carries_the_estimate(env):
    _cache_fal(env)
    _connect(env, "FAL_AI")
    link = _script_fal(env)
    post = _create(env)

    store = FakeStore()
    ok, renderer, job = _render(env, post["id"], store)

    assert ok is True
    # Priced first, then submitted and polled: never one long call.
    assert env.composio.slugs() == [FAL_ESTIMATE, FAL_SUBMIT, FAL_STATUS, FAL_STATUS, FAL_STATUS, FAL_RESULT]
    assert env.composio.params(FAL_ESTIMATE) == [
        {"estimate_type": "historical_api_price", "endpoints": {FAL_MODEL: {"call_quantity": 1}}}
    ]
    (submitted,) = env.composio.params(FAL_SUBMIT)
    assert submitted == {
        "model_id": FAL_MODEL,
        "input": {"prompt": f"{PROMPT}, {GUARD}", "aspect_ratio": "9:16", "duration": "5"},
    }
    assert env.composio.params(FAL_STATUS)[0] == {"model_id": FAL_MODEL, "request_id": "req-hook-1"}
    for call in env.composio.calls:
        assert (call["agent_id"], call["workspace_id"], call["app_name"], call["skip_validation"]) == (0, WS, "FAL_AI", True)
    assert env.fetched == [(link, 64 * 1024 * 1024, 60)]

    # The clip is copied into our storage, and media-render gets it at the hook's path.
    (key,) = [k for k in store.objects if "/footage-hook-" in k]
    assert key.startswith(f"social-media/{WS}/{post['id']}/footage-hook-") and key.endswith(".mp4")
    assert store.objects[key] == (CLIP, "video/mp4")
    (bundle,) = renderer.bundles
    assert key in renderer.stored_at_submit[0]
    assert {"path": "assets/slots/hook.mp4", "url": store.presigned_get(key, 3600)} in bundle["media"]
    html = bundle["composition"]["html"]
    assert 'data-slot="hook"' in html and 'src="assets/slots/hook.mp4"' in html
    # Only the slot the post asked for is shown: the others play the motion graphics.
    assert 'data-slot="still"' not in html and 'data-slot="app_loop"' not in html

    # Registered as a video Deliverable of the post.
    footage_deliverable = next(c for c in Deliverables.calls if c["file_path"] == key)
    assert footage_deliverable["artifact_type"] == "video" and footage_deliverable["source_type"] == "social_post"
    assert footage_deliverable["source_id"] == post["id"]
    assert footage_deliverable["extra"]["footage"] == {
        "slot": "hook", "toolkit": "fal_ai", "model": FAL_MODEL, "prompt": PROMPT, "estimate_usd": 0.46,
    }

    # The post's footage carries the file and the estimate, and the render went through.
    saved = _post(env, post["id"])
    assert saved.status == "needs_approval"
    record = saved.footage["hook"]
    assert record["status"] == "done" and record["prompt"] == PROMPT
    assert record["deliverable_id"] == "d-1" and record["name"] == key.rsplit("/", 1)[1]
    assert record["estimate_usd"] == pytest.approx(0.46) and record["cost_usd"] == pytest.approx(0.46)
    assert (record["toolkit"], record["model"], record["content_type"], record["bytes"]) == ("fal_ai", FAL_MODEL, "video/mp4", len(CLIP))
    assert saved.review_log[-1]["report"]["footage"] == {"generated": {"hook": "fal_ai"}}

    # The media lane books the estimate against the post: 5 s of footage at $0.46.
    (row,) = _media_rows(env, post["id"])
    assert (row.provider, row.model_id, row.input_tokens, row.tier) == ("fal_ai", FAL_MODEL, 5, "direct")
    assert row.total_cost == pytest.approx(0.46) and row.request_type == "media" and bool(row.is_byok) is True


def test_a_slot_made_for_its_prompt_is_reused_and_a_new_prompt_asks_again(env):
    _cache_fal(env)
    _connect(env, "FAL_AI")
    _script_fal(env)
    post = _create(env)
    store = FakeStore()
    assert _render(env, post["id"], store)[0] is True
    first = dict(_post(env, post["id"]).footage["hook"])

    # A second render reuses the stored clip: no call, no spend.
    env.composio.calls.clear()
    ok, renderer, _ = _render(env, post["id"], store)
    assert ok is True and env.composio.calls == []
    key = media_store.media_key(WS, post["id"], first["name"])
    assert {"path": "assets/slots/hook.mp4", "url": store.presigned_get(key, 3600)} in renderer.bundles[0]["media"]
    assert len(_media_rows(env, post["id"])) == 1
    assert _last_log(env, post["id"])["report"]["footage"] == {"reused": ["hook"]}

    # Sending the post's footage back as it is keeps what was made (a client's round trip).
    before = _post(env, post["id"])
    kept = env.client.patch(f"/api/socials/posts/{post['id']}", json={"footage": {"hook": first}})
    assert kept.status_code == 200, kept.text
    assert kept.json()["footage"]["hook"] == first

    # A new prompt asks again: a render setting, so the hash and the status stand.
    changed = env.client.patch(f"/api/socials/posts/{post['id']}", json={"footage": {"hook": {"prompt": "a harbour at dawn"}}})
    assert changed.status_code == 200, changed.text
    assert changed.json()["footage"] == {"hook": {"prompt": "a harbour at dawn"}}
    assert (changed.json()["status"], changed.json()["content_hash"]) == (before.status, before.content_hash)
    _script_fal(env)
    env.composio.calls.clear()
    assert _render(env, post["id"], store)[0] is True
    assert FAL_SUBMIT in env.composio.slugs()
    assert env.composio.params(FAL_SUBMIT)[0]["input"]["prompt"] == f"a harbour at dawn, {GUARD}"


def test_footage_is_a_render_setting_outside_the_content_hash(env):
    post = _create(env, footage=None)
    hashed = post["content_hash"]
    asked = env.client.patch(f"/api/socials/posts/{post['id']}", json={"footage": {"hook": {"prompt": PROMPT}}})
    assert asked.status_code == 200, asked.text
    assert asked.json()["content_hash"] == hashed and asked.json()["footage"] == {"hook": {"prompt": PROMPT}}
    assert service.compute_content_hash(_post(env, post["id"])) == hashed
    cleared = env.client.patch(f"/api/socials/posts/{post['id']}", json={"footage": None})
    assert cleared.status_code == 200 and cleared.json()["footage"] is None


# ---------------------------------------------------------------------------
# AC2 — over a cap nothing is submitted, and the render says why
# ---------------------------------------------------------------------------


def test_an_estimate_over_the_monthly_cap_submits_nothing_and_says_why(env):
    _cache_fal(env)
    _connect(env, "FAL_AI")
    _script_fal(env, estimate=0.46)
    _settings(env, media_monthly_cap_usd=1.0)
    _spent(env, 0.80)  # another post's footage this month
    post = _create(env)

    ok, renderer, _ = _render(env, post["id"], FakeStore())

    assert ok is False and renderer.bundles == []
    assert env.composio.slugs() == [FAL_ESTIMATE], "priced, then refused: nothing was submitted"
    saved = _post(env, post["id"])
    assert saved.status == "failed" and saved.footage == {"hook": {"prompt": PROMPT}}
    entry = saved.review_log[-1]
    assert entry["action"] == "render_failed" and entry["report"]["code"] == "footage_refused"
    resets = datetime.now(timezone.utc).replace(day=1) + timedelta(days=32)
    assert entry["comment"] == (
        "The hook footage would cost about $0.46, and this workspace has spent $0.80 of its $1.00 monthly media "
        f"cap, which resets on 1 {resets:%B}: nothing was submitted. Nothing was rendered."
    )
    assert [row.total_cost for row in _media_rows(env)] == [pytest.approx(0.80)], "nothing new was booked"


def test_an_estimate_over_the_posts_cap_submits_nothing_and_says_why(env):
    _cache_fal(env)
    _connect(env, "FAL_AI")
    _script_fal(env, estimate=0.46)
    post = _create(env)
    _spent(env, 9.80, execution_id=f"social_post:{post['id']}")

    ok, _, _ = _render(env, post["id"], FakeStore())

    assert ok is False and FAL_SUBMIT not in env.composio.slugs()
    assert _last_log(env, post["id"])["comment"].startswith(
        "The hook footage would cost about $0.46, and this post has spent $9.80 of its $10.00 media cap: nothing was submitted."
    )


def test_a_booking_is_seen_by_the_next_renders_cap_check(env):
    _cache_fal(env)
    _connect(env, "FAL_AI")
    _settings(env, media_monthly_cap_usd=0.80)
    _script_fal(env, estimate=0.46)
    first = _create(env)
    assert _render(env, first["id"], FakeStore())[0] is True

    _script_fal(env, estimate=0.46)
    second = _create(env)
    env.composio.calls.clear()
    ok, _, _ = _render(env, second["id"], FakeStore())

    assert ok is False and env.composio.slugs() == [FAL_ESTIMATE]
    assert "spent $0.46 of its $0.80 monthly media cap" in _last_log(env, second["id"])["comment"]


@pytest.mark.parametrize("stored", ["lots", -5, True])
def test_a_monthly_cap_that_is_not_a_number_of_dollars_spends_nothing(env, stored):
    _cache_fal(env)
    _connect(env, "FAL_AI")
    _script_fal(env)
    _settings(env, media_monthly_cap_usd=stored)
    post = _create(env)

    ok, _, _ = _render(env, post["id"], FakeStore())

    assert ok is False and FAL_SUBMIT not in env.composio.slugs()
    comment = _last_log(env, post["id"])["comment"]
    assert comment.startswith(f"The workspace's monthly media cap ({stored!r}) is not a number of dollars")
    assert "nothing was submitted" in comment


def test_the_month_spend_counts_this_months_media_lane_only(env):
    now = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)
    _spent(env, 1.25, created_at=datetime(2026, 9, 2, 8, 0))
    _spent(env, 0.75, created_at=datetime(2026, 9, 25, 23, 0), execution_id="social_post:p1")
    _spent(env, 9.00, created_at=datetime(2026, 8, 31, 23, 59))  # last month
    workspace = env.session.get(Workspace, WS)
    spend = media_caps.media_spend(env.session, workspace, "p1", now=now)
    assert spend.month_usd == pytest.approx(2.0) and spend.post_usd == pytest.approx(0.75)
    assert (spend.monthly_cap_usd, spend.post_cap_usd, spend.problem) == (30.0, 10.0, None)
    assert spend.period_end == datetime(2026, 10, 1, tzinfo=timezone.utc)
    # The post's cap is checked first, then the month's.
    assert spend.refusal(9.25, "It") is None
    assert "this post has spent $0.75 of its $10.00 media cap" in spend.refusal(9.26, "It")
    roomy = replace(spend, post_cap_usd=100.0)
    assert roomy.refusal(28.0, "It") is None
    assert roomy.refusal(28.01, "It") == (
        "It would cost about $28.01, and this workspace has spent $2.00 of its $30.00 monthly media cap, "
        "which resets on 1 October: nothing was submitted."
    )


@pytest.mark.parametrize(
    "value, normalized",
    [
        ({"media_monthly_cap_usd": 25}, {"media_monthly_cap_usd": 25.0}),
        ({"media_monthly_cap_usd": 0}, {"media_monthly_cap_usd": 0.0}),
        ({"enabled": True, "media_monthly_cap_usd": 12.5}, {"enabled": True, "media_monthly_cap_usd": 12.5}),
        ({"enabled": False}, {"enabled": False}),
    ],
)
def test_the_workspace_monthly_cap_is_a_socials_setting(value, normalized):
    assert socials_settings.validate_socials_update(value) == normalized


@pytest.mark.parametrize("bad", [-1, "20", True, float("inf"), float("nan"), None])
def test_a_cap_that_is_not_dollars_is_refused_on_save(bad):
    with pytest.raises(ValueError) as refused:
        socials_settings.validate_socials_update({"media_monthly_cap_usd": bad})
    assert "must be a number of dollars" in str(refused.value)


def test_without_a_stored_cap_the_config_default_applies(monkeypatch):
    monkeypatch.setattr(socials_settings.config, "SOCIALS_MEDIA_MONTHLY_CAP_USD", 42.0, raising=False)
    assert socials_settings.media_monthly_cap_usd({"socials": {"enabled": True}}) == (42.0, None)
    assert socials_settings.media_monthly_cap_usd({"socials": {"media_monthly_cap_usd": 7}}) == (7.0, None)
    cap, why = socials_settings.media_monthly_cap_usd({"socials": {"media_monthly_cap_usd": "7"}})
    assert cap == 0.0 and "not a number of dollars" in why


# ---------------------------------------------------------------------------
# AC3 — copied into our storage before the slot is marked done
# ---------------------------------------------------------------------------


def test_the_file_is_in_our_storage_before_the_slot_is_marked_done(env, monkeypatch):
    _cache_fal(env)
    _connect(env, "FAL_AI")
    _script_fal(env)
    post = _create(env)
    events = []
    store = FakeStore(events)
    record = footage._record

    def recording(session_factory, workspace_id, post_id, slot, done):
        events.append(("marked done", slot, done["name"], sorted(store.objects)))
        return record(session_factory, workspace_id, post_id, slot, done)

    monkeypatch.setattr(footage, "_record", recording)
    assert _render(env, post["id"], store)[0] is True

    stored = [event for event in events if event[0] == "stored" and "/footage-hook-" in event[1]]
    marked = [event for event in events if event[0] == "marked done"]
    assert len(stored) == 1 and len(marked) == 1
    assert events.index(stored[0]) < events.index(marked[0])
    assert stored[0][1] in marked[0][3] and stored[0][1].endswith(marked[0][2])


def test_a_file_that_cannot_be_stored_is_never_marked_done_and_the_spend_is_booked(env):
    _cache_fal(env)
    _connect(env, "FAL_AI")
    _script_fal(env)
    post = _create(env)

    ok, renderer, _ = _render(env, post["id"], FakeStore(fail=True))

    assert ok is False and renderer.bundles == []
    saved = _post(env, post["id"])
    assert saved.status == "failed" and saved.footage == {"hook": {"prompt": PROMPT}}
    assert saved.review_log[-1]["comment"] == (
        "The footage could not be made: Hook footage: fal.ai's file could not be stored. Nothing was rendered."
    )
    assert Deliverables.calls == []
    # fal completed the job, so it billed it: the estimate is booked all the same.
    assert [row.total_cost for row in _media_rows(env, post["id"])] == [pytest.approx(0.46)]


def test_what_a_toolkit_returns_that_is_not_the_slots_kind_is_never_kept(env):
    _cache_fal(env)
    _connect(env, "FAL_AI")
    _script_fal(env, clip=AUDIO)
    post = _create(env)
    store = FakeStore()

    assert _render(env, post["id"], store)[0] is False
    assert store.objects == {} and "fal.ai returned something that is not footage" in _last_log(env, post["id"])["comment"]


def test_a_job_the_toolkit_failed_books_nothing(env):
    _cache_fal(env)
    _connect(env, "FAL_AI")
    _script_fal(env, statuses=("IN_QUEUE", "ERROR"))
    post = _create(env)

    assert _render(env, post["id"], FakeStore())[0] is False
    assert FAL_RESULT not in env.composio.slugs()
    assert "fal.ai's job ended ERROR" in _last_log(env, post["id"])["comment"]
    assert _media_rows(env, post["id"]) == []


def test_a_status_that_cannot_be_read_is_asked_again_then_given_up_and_booked(env):
    _cache_fal(env)
    _connect(env, "FAL_AI")
    _script_fal(env)
    env.composio.answers[FAL_STATUS] = [_failed("gateway timeout")] * footage.MAX_POLL_FAILURES
    post = _create(env)

    assert _render(env, post["id"], FakeStore())[0] is False
    assert env.composio.slugs().count(FAL_STATUS) == footage.MAX_POLL_FAILURES
    assert "its status could not be read" in _last_log(env, post["id"])["comment"]
    # The toolkit may still finish it: the estimate is booked.
    assert [row.total_cost for row in _media_rows(env, post["id"])] == [pytest.approx(0.46)]


# ---------------------------------------------------------------------------
# AC4 — a credit-billed toolkit books its balance difference
# ---------------------------------------------------------------------------


def _script_higgsfield(env, *, before=100, after=92):
    link = "https://cdn.higgsfield.ai/jobs/job-7/result.mp4"
    env.files[link] = CLIP
    env.composio.answers[HF_BALANCE] = [_ok({"credits": before, "subscription_plan_type": "pro"}),
                                        _ok({"credits": after, "subscription_plan_type": "pro"})]
    env.composio.answers[HF_VIDEO] = _ok({"results": [{"id": "job-7", "status": "queued"}], "request_id": "r-7",
                                          "cost": None, "error": None})
    env.composio.answers[HF_WAIT] = [
        _ok({"jobs": [{"id": "job-7", "status": "in_progress"}], "all_terminal": False, "timed_out": True}),
        _ok({"jobs": [{"id": "job-7", "status": "completed", "results": [{"url": link, "type": "video"}]}],
             "all_terminal": True}),
    ]
    return link


def test_a_credit_billed_toolkit_higgsfield_books_its_balance_difference(env):
    _cache_higgsfield(env)
    _connect(env, "HIGGSFIELD_MCP")
    _script_higgsfield(env, before=100, after=92)
    post = _create(env)

    ok, renderer, _ = _render(env, post["id"], FakeStore())

    assert ok is True
    assert env.composio.slugs() == [HF_BALANCE, HF_VIDEO, HF_WAIT, HF_WAIT, HF_BALANCE]
    (generated,) = env.composio.params(HF_VIDEO)
    assert set(generated) == {"params"}, "Higgsfield takes ONE params string"
    assert json.loads(generated["params"]) == {
        "model": "kling3_0", "prompt": f"{PROMPT}, {GUARD}", "aspect_ratio": "9:16", "duration": 5,
    }
    assert env.composio.params(HF_WAIT)[0] == {"jobs": ["job-7"], "timeout_seconds": 15}
    assert any(entry["path"] == "assets/slots/hook.mp4" for entry in renderer.bundles[0]["media"])

    # 8 credits at $0.0625 = $0.50, against the post; the hook's cost says so.
    (row,) = _media_rows(env, post["id"])
    assert (row.provider, row.model_id, row.input_tokens) == ("higgsfield_mcp", "kling3_0", 8)
    assert row.total_cost == pytest.approx(0.50) and row.error_message is None
    record = _post(env, post["id"]).footage["hook"]
    assert record["estimate_usd"] == pytest.approx(2.5), "Higgsfield prices nothing: the ceiling was checked"
    assert record["cost_usd"] == pytest.approx(0.50)


def test_an_unreadable_balance_submits_nothing_to_a_credit_billed_toolkit(env):
    _cache_higgsfield(env)
    _connect(env, "HIGGSFIELD_MCP")
    _script_higgsfield(env)
    env.composio.answers[HF_BALANCE] = _failed("unauthorised")
    post = _create(env)

    assert _render(env, post["id"], FakeStore())[0] is False
    assert HF_VIDEO not in env.composio.slugs()
    comment = _last_log(env, post["id"])["comment"]
    assert "Higgsfield's balance could not be read" in comment and "nothing was submitted to it" in comment
    assert _media_rows(env, post["id"]) == []


def test_kie_makes_a_still_and_books_its_credits(env):
    _cache_kie(env)
    _connect(env, "KIEAI")
    link = "https://tempfile.redpandaai.co/kieai/still.png"
    env.files[link] = STILL
    env.composio.answers[KIE_CREDITS] = [_ok({"code": 200, "msg": "success", "data": 1000}),
                                         _ok({"code": 200, "msg": "success", "data": 996})]
    env.composio.answers[KIE_FLUX] = _ok({"code": 200, "msg": "success", "data": {"taskId": "task-9"}})
    env.composio.answers[KIE_FLUX_DETAILS] = [
        _ok({"code": 200, "data": {"taskId": "task-9", "successFlag": 0}}),
        _ok({"code": 200, "data": {"taskId": "task-9", "successFlag": 1,
                                   "response": {"originImageUrl": None, "resultImageUrl": link}}}),
    ]
    post = _create(env, footage={"still": {"prompt": "an open logbook on a chart table"}})
    store = FakeStore()

    ok, renderer, _ = _render(env, post["id"], store)

    assert ok is True
    assert env.composio.params(KIE_FLUX) == [{
        "prompt": f"an open logbook on a chart table, {GUARD}", "model": "flux-kontext-pro",
        "aspect_ratio": "9:16", "output_format": "png",
    }]
    assert env.composio.params(KIE_FLUX_DETAILS)[0] == {"task_id": "task-9"}
    (key,) = [k for k in store.objects if "/footage-still-" in k]
    assert store.objects[key] == (STILL, "image/png") and key.endswith(".png")
    assert {"path": "assets/slots/still.png", "url": store.presigned_get(key, 3600)} in renderer.bundles[0]["media"]
    (row,) = _media_rows(env, post["id"])
    assert (row.provider, row.input_tokens) == ("kieai", 4) and row.total_cost == pytest.approx(0.02)
    assert next(c for c in Deliverables.calls if c["file_path"] == key)["artifact_type"] == "image"


# ---------------------------------------------------------------------------
# One render, several shots: one toolkit making two kinds, or two toolkits
# ---------------------------------------------------------------------------

VIDEO_LINK = "https://v3.fal.media/files/koala/hook.mp4"
STILL_LINK = "https://v3.fal.media/files/koala/still.png"


def _script_fal_both(env):
    """fal.ai making the hook (its video model) and the still (its image model) in one render."""
    env.files.update({VIDEO_LINK: CLIP, STILL_LINK: STILL})
    prices = {FAL_MODEL: 0.46, FAL_IMAGE_MODEL: 0.05}

    def estimate(params):
        (model,) = params["endpoints"]
        return _ok({"estimate_type": "historical_api_price", "total_cost": prices[model], "currency": "USD"})

    env.composio.answers[FAL_ESTIMATE] = estimate
    env.composio.answers[FAL_SUBMIT] = lambda params: _ok({"request_id": f"req-{params['model_id'].rsplit('/', 1)[1]}"})
    env.composio.answers[FAL_STATUS] = lambda params: _ok({"status": "COMPLETED"})
    env.composio.answers[FAL_RESULT] = lambda params: _ok(
        {"video": {"url": VIDEO_LINK}} if params["model_id"] == FAL_MODEL else {"images": [{"url": STILL_LINK}]}
    )


def test_one_toolkit_makes_footage_and_a_still_in_one_render(env):
    _cache_fal(env)
    _connect(env, "FAL_AI")
    _script_fal_both(env)
    post = _create(env, footage={"hook": {"prompt": PROMPT}, "still": {"prompt": "an open logbook"}})
    store = FakeStore()

    ok, renderer, _ = _render(env, post["id"], store)

    assert ok is True
    # Priced once per model, then both submitted before either is polled.
    assert [next(iter(p["endpoints"])) for p in env.composio.params(FAL_ESTIMATE)] == [FAL_MODEL, FAL_IMAGE_MODEL]
    assert env.composio.params(FAL_SUBMIT) == [
        {"model_id": FAL_MODEL, "input": {"prompt": f"{PROMPT}, {GUARD}", "aspect_ratio": "9:16", "duration": "5"}},
        {"model_id": FAL_IMAGE_MODEL, "input": {"prompt": f"an open logbook, {GUARD}", "aspect_ratio": "9:16"}},
    ]
    assert env.composio.slugs().index(FAL_STATUS) > env.composio.slugs().index(FAL_SUBMIT) + 1
    paths = {entry["path"] for entry in renderer.bundles[0]["media"]}
    assert paths == {"assets/slots/hook.mp4", "assets/slots/still.png"}
    rows = [(row.model_id, row.input_tokens, round(row.total_cost, 2)) for row in _media_rows(env, post["id"])]
    assert rows == [(FAL_MODEL, 5, 0.46), (FAL_IMAGE_MODEL, 1, 0.05)]
    footage_of = _post(env, post["id"]).footage
    assert (footage_of["hook"]["cost_usd"], footage_of["still"]["cost_usd"]) == (pytest.approx(0.46), pytest.approx(0.05))
    assert footage_of["still"]["content_type"] == "image/png"


def test_two_toolkits_make_one_renders_footage_each_booked_its_own_way(env):
    _cache_fal(env)
    _cache_kie(env)
    _connect(env, "FAL_AI", "KIEAI")
    # The allowlist is data: without fal's image capability, the still goes to Kie.ai.
    allowlist = json.loads(json.dumps(WAVE1.SOCIALS_MEDIA_ACTIONS_SEED))
    del allowlist["fal_ai"]["generate_image"]
    table = SystemSetting.__table__
    with env.engine.begin() as conn:
        conn.execute(table.update().where(table.c.category == "socials", table.c.key == "media_actions")
                     .values(value=json.dumps(allowlist)))
    _script_fal(env, statuses=("COMPLETED",))
    link = "https://tempfile.redpandaai.co/kieai/still.png"
    env.files[link] = STILL
    env.composio.answers[KIE_CREDITS] = [_ok({"code": 200, "data": 1000}), _ok({"code": 200, "data": 996})]
    env.composio.answers[KIE_FLUX] = _ok({"code": 200, "data": {"taskId": "task-9"}})
    env.composio.answers[KIE_FLUX_DETAILS] = _ok({"code": 200, "data": {"successFlag": 1, "response": {"resultImageUrl": link}}})
    post = _create(env, footage={"hook": {"prompt": PROMPT}, "still": {"prompt": "an open logbook"}})

    ok, renderer, _ = _render(env, post["id"], FakeStore())

    assert ok is True
    assert env.composio.slugs() == [
        FAL_ESTIMATE, FAL_SUBMIT, FAL_STATUS, FAL_RESULT, KIE_CREDITS, KIE_FLUX, KIE_FLUX_DETAILS, KIE_CREDITS,
    ]
    assert {entry["path"] for entry in renderer.bundles[0]["media"]} == {"assets/slots/hook.mp4", "assets/slots/still.png"}
    rows = [(row.provider, row.input_tokens, round(row.total_cost, 2)) for row in _media_rows(env, post["id"])]
    assert rows == [("fal_ai", 5, 0.46), ("kieai", 4, 0.02)]
    assert _last_log(env, post["id"])["report"]["footage"] == {"generated": {"hook": "fal_ai", "still": "kieai"}}


@pytest.mark.parametrize("priced", [{"total_cost": 0.46}, {"total_cost": 0.46, "currency": "EUR"}, {"currency": "USD"}])
def test_a_fal_price_that_is_not_in_dollars_submits_nothing(env, priced):
    _cache_fal(env)
    _connect(env, "FAL_AI")
    _script_fal(env)
    env.composio.answers[FAL_ESTIMATE] = _ok(priced)
    post = _create(env)

    assert _render(env, post["id"], FakeStore())[0] is False
    assert env.composio.slugs() == [FAL_ESTIMATE]
    entry = _last_log(env, post["id"])
    assert entry["report"]["code"] == "footage_refused"
    assert entry["comment"] == (
        f"fal.ai did not price {FAL_MODEL} in dollars, so its spend could not be checked against the caps: "
        "nothing was submitted. Nothing was rendered."
    )


# ---------------------------------------------------------------------------
# AC5 — no generation toolkit: the template's own motion graphics
# ---------------------------------------------------------------------------


def test_with_no_generation_toolkit_connected_the_footage_slots_fall_back_to_motion_graphics(env):
    post = _create(env, footage={"hook": {"prompt": PROMPT}, "still": {"prompt": "a quiet desk"}})

    ok, renderer, job = _render(env, post["id"], FakeStore())

    assert ok is True and env.composio.calls == []
    (bundle,) = renderer.bundles
    assert "media" not in bundle
    html = bundle["composition"]["html"]
    assert "data-slot" not in html and "assets/slots/" not in html
    assert '<div class="bg"></div>' in html and '<div class="plate"></div>' in html
    assert bundle["variables"]["headline"] == "Orders at midnight."
    saved = _post(env, post["id"])
    assert saved.status == "needs_approval"
    fallback = saved.review_log[-1]["report"]["footage"]["motion_graphics"]
    assert fallback == {
        "hook": "no generation toolkit that makes footage is connected: connect fal.ai or Kie.ai or Higgsfield in Composio",
        "still": "no generation toolkit that makes a still is connected: connect fal.ai or Kie.ai or Higgsfield in Composio",
    }
    assert _media_rows(env) == []


def test_a_denied_generate_action_is_never_used(env):
    _cache_fal(env)
    _connect(env, "FAL_AI")
    _deny(env, FAL_SUBMIT)
    post = _create(env)

    ok, renderer, _ = _render(env, post["id"], FakeStore())

    assert ok is True and env.composio.calls == []
    assert "data-slot" not in renderer.bundles[0]["composition"]["html"]
    why = _last_log(env, post["id"])["report"]["footage"]["motion_graphics"]["hook"]
    assert why.startswith("fal.ai does not offer FAL_AI_SUBMIT_ASYNC_JOB here")


def test_the_footage_sources_say_what_the_slots_can_be_filled_with(env):
    _cache_fal(env)
    _connect(env, "FAL_AI")
    resp = env.client.get("/api/socials/footage")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["kinds"]["video"] == {"available": True, "toolkit": "fal_ai", "label": "fal.ai", "model": FAL_MODEL}
    assert body["kinds"]["image"]["model"] == FAL_IMAGE_MODEL
    assert body["toolkits"] == [
        {"toolkit": "fal_ai", "label": "fal.ai", "status": "available", "makes": ["video", "image"]},
        {"toolkit": "kieai", "label": "Kie.ai", "status": "connect"},
        {"toolkit": "higgsfield_mcp", "label": "Higgsfield", "status": "connect"},
    ]
    assert body["spend"]["monthly_cap_usd"] == 30.0 and body["spend"]["month_usd"] == 0.0
    assert body["problem"] is None


# ---------------------------------------------------------------------------
# A save names only slots the template has and lets a toolkit fill
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "asked, message",
    [
        ({"nope": {"prompt": "x"}}, "footage.nope: the post's template has no slot nope (its slots are app_loop, hook, still)"),
        ({"app_loop": {"prompt": "an app demo"}}, "footage.app_loop: The app's own loop takes the workspace's own file, never generated footage"),
        ({"hook": {"prompt": "  "}}, "footage.hook.prompt is required"),
        ({"hook": {"prompt": "x", "colour": "red"}}, "footage.hook takes a prompt, got ['colour']"),
        ({"hook": "a sea"}, "footage.hook must be an object"),
        ({"9lives": {"prompt": "x"}}, "footage.9lives is not a slot name"),
    ],
)
def test_a_save_names_only_slots_the_template_has_and_lets_a_toolkit_fill(env, asked, message):
    resp = env.client.post(
        "/api/socials/posts",
        json={"title": "T", "format": "video", "template_id": str(_template(env)), "footage": asked},
    )
    assert resp.status_code == 422, resp.text
    assert message in resp.json()["detail"]


def test_a_client_cannot_forge_what_a_render_recorded(env):
    post = _create(env, footage={"hook": {"prompt": PROMPT, "status": "done", "deliverable_id": "forged", "name": "x.mp4"}})
    assert post["footage"] == {"hook": {"prompt": PROMPT}}


# ---------------------------------------------------------------------------
# The recipes' pieces
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "width, height, ratio",
    [(1080, 1920, "9:16"), (1080, 1350, "4:5"), (1200, 628, "16:9"), (1600, 900, "16:9"), (1080, 1080, "1:1")],
)
def test_the_aspect_ratio_is_the_compositions(width, height, ratio):
    assert toolkits.aspect_ratio(width, height) == ratio


def test_every_prompt_ends_with_the_guard_once(monkeypatch):
    monkeypatch.setattr(toolkits.config, "SOCIALS_FOOTAGE_PROMPT_GUARD", GUARD, raising=False)
    assert toolkits.prompt_for("A calm sea.") == f"A calm sea, {GUARD}"
    assert toolkits.prompt_for(f"A calm sea, {GUARD.upper()}") == f"A calm sea, {GUARD.upper()}"


@pytest.mark.parametrize(
    "data, found",
    [
        (CLIP, ("video", "mp4", "video/mp4")),
        (b"\x00\x00\x00\x14ftypqt  " + b"x" * 40, ("video", "mov", "video/quicktime")),
        (b"\x1a\x45\xdf\xa3" + b"x" * 40, ("video", "webm", "video/webm")),
        (STILL, ("image", "png", "image/png")),
        (b"\xff\xd8\xff\xe0" + b"x" * 40, ("image", "jpg", "image/jpeg")),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"x" * 40, ("image", "webp", "image/webp")),
        (AUDIO, None),
        (b"<html>not media</html>", None),
    ],
)
def test_generated_media_is_known_by_its_first_bytes(data, found):
    assert footage.media_type(data) == found


@pytest.mark.parametrize(
    "answer, kind, url",
    [
        ({"video": {"url": "https://v3.fal.media/a.mp4"}}, "video", "https://v3.fal.media/a.mp4"),
        ({"images": [{"url": "https://v3.fal.media/b.png", "width": 1}]}, "image", "https://v3.fal.media/b.png"),
        ({"code": 200, "data": {"response": {"resultUrls": ["https://kie.ai/c.mp4"]}, "successFlag": 1}}, "video", "https://kie.ai/c.mp4"),
        ({"id": "j", "status": "completed", "results": [{"url": "https://cdn.higgsfield.ai/d.mp4"}]}, "video", "https://cdn.higgsfield.ai/d.mp4"),
        ({"file": {"name": "e.mp4", "mimetype": "video/mp4", "s3url": "https://r2.composio.dev/e.mp4"}}, "video", "https://r2.composio.dev/e.mp4"),
    ],
)
def test_a_finished_jobs_file_is_found_in_each_toolkits_answer(answer, kind, url):
    found = toolkits.media_file(answer, kind)
    assert isinstance(found, ReturnedFile) and found.url == url


def test_a_balance_is_read_bare_or_under_its_key():
    assert toolkit.balance_of(12, ("credits",)) == Decimal(12)
    assert toolkit.balance_of({"code": 200, "msg": "success", "data": 996}, ("credits", "data", "balance")) == Decimal(996)
    assert toolkit.balance_of({"credits": "92.5"}, ("credits",)) == Decimal("92.5")
    assert toolkit.balance_of({"error": "nope"}, ("credits",)) is None


def test_the_tools_output_is_read_out_of_composios_envelope():
    assert toolkit.output_of(_ok({"a": 1})) == {"a": 1}
    assert toolkit.output_of(_ok(json.dumps({"a": 1}))) == {"a": 1}
    assert toolkit.output_of({"success": True, "data": {"a": 1}}) == {"a": 1}


def test_a_footage_file_name_never_replaces_an_earlier_one():
    first = footage.footage_file_name("Hook", "ab" * 32, "mp4")
    assert first == "footage-hook-abababab.mp4" and media_store.valid_file_name(first)
    assert footage.footage_file_name("hook", "cd" * 32, "mp4") != first


def test_the_recipes_write_no_provider_client():
    """D15: every call goes through the workspace's Composio connection; a file link
    is read only through files.fetch (public, pinned, capped)."""
    for name in ("footage.py", "footage_toolkits.py", "toolkit.py"):
        tree = ast.parse((_ORCH / "modules" / "socials" / "recipes" / name).read_text(encoding="utf-8"))
        imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        imported |= {(node.module or "").split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        assert not imported & {"httpx", "requests", "aiohttp", "urllib3", "fal_client", "elevenlabs"}, name


def test_the_seeded_app_promo_never_generates_the_apps_own_loop():
    starter = json.loads((_ORCH / "modules" / "documents" / "templates" / "social" / "app-promo.json").read_text())
    assert starter["slots"]["app_loop"]["generate"] is False
    assert all(spec.get("generate", True) is True for name, spec in starter["slots"].items() if name != "app_loop")


# ---------------------------------------------------------------------------
# The column (the wave's one migration), and the window across processes
# ---------------------------------------------------------------------------


def _columns(conn):
    return {c["name"] for c in sa.inspect(conn).get_columns("social_posts")}


def _step(conn, name):
    with Operations.context(MigrationContext.configure(conn)):
        getattr(WAVE1, name)()


def test_the_wave_migration_adds_the_footage_column_once():
    """On SQLite the add; the drop is Postgres DDL, proved by the @integration test below."""
    engine = sa.create_engine("sqlite://", poolclass=StaticPool)
    try:
        with engine.begin() as conn:
            with Operations.context(MigrationContext.configure(conn)):
                WAVE0._create_social_posts()
            assert "footage" not in _columns(conn)
            _step(conn, "add_post_footage_column")
            _step(conn, "add_post_footage_column")
            (column,) = [c for c in sa.inspect(conn).get_columns("social_posts") if c["name"] == "footage"]
            assert column["nullable"] is True
    finally:
        engine.dispose()


def test_the_upgrade_runs_the_footage_column_step():
    source = (VERSIONS / "prd251_wave1.py").read_text(encoding="utf-8")
    upgrade = source[source.index("def upgrade()"): source.index("def downgrade()")]
    downgrade = source[source.index("def downgrade()"):]
    assert "add_post_footage_column()" in upgrade and "drop_post_footage_column()" in downgrade


@pytest.fixture(scope="module")
def pg_engine():
    from core.database.database import get_database_url

    try:
        engine = sa.create_engine(get_database_url(), pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(sa.text("SELECT 1 FROM system_settings LIMIT 1"))
            conn.execute(sa.text("SELECT 1 FROM document_templates LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the Postgres checks need the test database: {exc}")
    yield engine
    engine.dispose()


@pytest.mark.integration
def test_a_footage_window_held_by_another_worker_is_waited_for(pg_engine, monkeypatch):
    """Two renders of one workspace never price, cap and submit against the same
    headroom: a window another process holds keeps this one out until it ends."""
    monkeypatch.setattr(toolkit.config, "SOCIALS_RENDER_POLL_SECONDS", 0.05, raising=False)
    factory = sessionmaker(bind=pg_engine)
    params = {"namespace": footage.FOOTAGE_LOCK_NAMESPACE, "key": str(WS)}

    async def scenario():
        entered = asyncio.Event()

        async def second():
            async with toolkit.window(factory, footage.FOOTAGE_LOCK_NAMESPACE, str(WS)):
                entered.set()

        with pg_engine.connect() as holder:
            trans = holder.begin()
            held = holder.execute(sa.text("SELECT pg_try_advisory_xact_lock(:namespace, hashtext(:key))"), params)
            assert held.scalar() is True
            task = asyncio.create_task(second())
            await asyncio.sleep(0.5)
            assert not entered.is_set(), "entered while another connection held the window"
            trans.rollback()
        await asyncio.wait_for(task, timeout=10)
        assert entered.is_set()

    asyncio.run(scenario())


@pytest.mark.integration
def test_create_all_first_then_the_upgrade_twice_leaves_one_jsonb_footage_column(pg_engine):
    """A backend that already loaded the new models runs create_all before the migration."""
    with pg_engine.connect() as conn:
        trans = conn.begin()
        try:
            conn.execute(sa.text("SET LOCAL lock_timeout = '5s'"))
            Base.metadata.create_all(bind=conn)
            with Operations.context(MigrationContext.configure(conn)):
                WAVE1.upgrade()
                WAVE1.upgrade()
            kind = conn.execute(
                sa.text(
                    "SELECT udt_name FROM information_schema.columns "
                    "WHERE table_name = 'social_posts' AND column_name = 'footage'"
                )
            ).scalars().all()
            assert kind == ["jsonb"]
            with Operations.context(MigrationContext.configure(conn)):
                WAVE1.downgrade()
            assert "footage" not in _columns(conn)
            with Operations.context(MigrationContext.configure(conn)):
                WAVE1.upgrade()
            assert "footage" in _columns(conn)
        finally:
            trans.rollback()
