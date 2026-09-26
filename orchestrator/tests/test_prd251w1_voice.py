"""PRD-251 Wave 1, US-111 (S1.5, D11) — voice: Kokoro, or a connected Composio voice toolkit.

The real Socials router runs on a mini FastAPI app over in-memory SQLite (the
social tables, SQLite copies of ``workspaces``, ``document_templates`` and the
Composio tables, an untyped ``llm_usage``, and ``system_settings`` seeded by the
two migrations: the Wave 0 deny list and the US-110 media allowlist). The media
capability registry reads them for real. Composio is mocked where it is called
(``ComposioToolExecutor`` in the voice recipe), media-render at the HTTP layer
(``httpx.MockTransport``, the real client), and storage, the Deliverable
registry and the usage tracker are recording fakes. Pins:

* AC2 — with ``fish_audio`` connected and chosen, the same script renders with
  no other change: one Composio call per line through the allowlisted speech
  action, each line's audio copied into our storage before the bundle is sent,
  and the bundle media-render gets is the Kokoro bundle with each spoken line
  naming its stored file instead of its text; Fish Audio's balance difference
  is booked on the media lane against the post;
* AC3 — without a voice toolkit, the voice picker offers Kokoro only, and Fish
  Audio and ElevenLabs as links to the Composio connect flow; a connected
  toolkit the workspace cannot use is unavailable and says why; a denied speech
  action is never offered;
* a toolkit's voices are listed through its allowlisted ``voices`` action; the
  voice is saved only when the workspace can speak with it, and it is a render
  setting: outside the content hash, so an approval stands;
* the recipe: a Composio file output, a link or inline bytes; audio by its
  first bytes; a link fetched from public addresses only, pinned, size-capped;
  a failed line still books what was spent; a balance that cannot be read
  first means nothing is spoken; ElevenLabs books characters at $0;
* the column: ``social_posts.voice`` comes from the wave's one migration,
  create_all-first safe (``@integration`` on Postgres: create_all, then the
  upgrade twice).
"""
from __future__ import annotations

import asyncio
import base64
import copy
import importlib.util
from contextlib import asynccontextmanager
import json
import os
import sys
import uuid
from datetime import datetime
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
import core.composio.deny_list as deny_list  # noqa: E402
import core.database.database as database_mod  # noqa: E402
import core.media_render_client as media_render_client  # noqa: E402
import core.media_render_quota as render_quota  # noqa: E402
import modules.socials.media_store as media_store  # noqa: E402
import modules.socials.recipes.files as files  # noqa: E402
import modules.socials.recipes.toolkit as toolkit  # noqa: E402
import modules.socials.recipes.voice as voice  # noqa: E402
import modules.socials.render as render  # noqa: E402
import modules.socials.service as service  # noqa: E402
import modules.socials.settings as socials_settings  # noqa: E402
import services.deliverable_service as deliverable_service  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.database.database import Base, get_db  # noqa: E402
from core.llm import usage_context as uc  # noqa: E402
from core.llm.providers import MEDIA_RENDER_PROVIDER  # noqa: E402
from core.llm.usage_tracker import UsageTracker  # noqa: E402
from core.media_render_bundle import VOICE_DIR, voice_script, with_voice_files  # noqa: E402
from core.media_render_client import MediaRenderClient  # noqa: E402
from core.models.composio import ComposioConnection, ComposioEntity  # noqa: E402
from core.models.composio_cache import ComposioActionCache  # noqa: E402
from core.models.core import DocumentTemplate, LLMUsage  # noqa: E402
from core.models.socials import SocialPost, SocialPostTarget  # noqa: E402
from core.models.system_settings import SystemSetting  # noqa: E402
from core.models.workspaces import Workspace  # noqa: E402
from core.security.web_access import OutboundTarget  # noqa: E402
from modules.socials.capabilities import media_capabilities  # noqa: E402

VERSIONS = _ORCH / "alembic" / "versions"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WAVE0 = _load(VERSIONS / "prd251_socials.py", "prd251_socials_migration_voice")
WAVE1 = _load(VERSIONS / "prd251_wave1.py", "prd251_wave1_migration_voice")

WS = uuid.UUID("00000000-0000-0000-0000-0000000001a1")
CREATED = datetime(2026, 9, 1, 9, 0)
RENDER_URL = "http://media-render:8090"
TOKEN = "render-secret"
JOB_ID = "e" * 32
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"frame-bytes " * 400
OUTPUT = {"name": "render.mp4", "aspect": "9:16", "width": 1080, "height": 1920, "bytes": len(MP4), "duration": 6.0}
FISH_VOICE = "802e3bc2b27e49c2995d23ef70e6ac89"
SPEAK = "FISH_AUDIO_SYNTHESIZE_SPEECH"
BALANCE = "FISH_AUDIO_GET_ACCOUNT_BALANCE"
LIST_VOICES = "FISH_AUDIO_LIST_VOICE_MODELS"
ELEVEN_SPEAK = "ELEVENLABS_TEXT_TO_SPEECH"
# The cached input schemas, as the metadata sync stores them (docs.composio.dev, 2026-09-26).
FISH_SPEAK_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "voice_model_ids": {"type": "array", "items": {"type": "string"}},
        "speed": {"type": "number"},
        "format": {"type": "string", "enum": ["mp3", "wav", "pcm", "opus"]},
        "latency": {"type": "string"},
    },
    "required": ["text", "voice_model_ids"],
}
FISH_LIST_SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string"}, "page_size": {"type": "integer"}, "sort_by": {"type": "string"}},
}
ELEVEN_SPEAK_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}, "voice_id": {"type": "string"}, "model_id": {"type": "string"}},
    "required": ["text", "voice_id"],
}
SCHEMAS = {SPEAK: FISH_SPEAK_SCHEMA, LIST_VOICES: FISH_LIST_SCHEMA, ELEVEN_SPEAK: ELEVEN_SPEAK_SCHEMA}
# A two-line script: the first line reads a variable, as the seeded templates' lines do.
SCRIPT = [{"id": "l01", "at": 0.3, "text": "{{ headline }}"}, {"id": "l02", "at": 2.4, "text": "Three weeks to go."}]
COMPOSITION = {
    "html": '<!doctype html><html><head></head><body><div id="root" data-composition-id="main" data-width="1080" '
            'data-height="1920" data-duration="6"><h1>{{ headline }}</h1></div></body></html>',
    "css": "h1 { color: var(--brand-text); }",
    "variables_schema": {"headline": {"type": "text"}},
    "sizes": ["1080x1920"],
    "audio_plan": {"voice": {"voice": "af_heart", "speed": 0.95, "lines": SCRIPT}, "music": {"track": "deep-house-003", "start": 32.0}},
}


# US-112: media-render reports the library track a render mixes (COMPOSITION names Deep House 003),
# and a render whose report does not name it fails closed (core/music_credit.credit_for_render).
MUSIC_REPORT = {
    "track": "deep-house-003", "title": "Deep House 003", "artist": "Sascha Ende", "licence": "CC BY 4.0",
    "licence_url": "https://creativecommons.org/licenses/by/4.0/",
    "attribution": 'Music: "Deep House 003" by Sascha Ende (ende.app), licensed CC BY 4.0.',
    "credit_required": True, "start": 32.0, "end": 38.0,
}


def _mp3(tag: str) -> bytes:
    """An MP3 as a voice toolkit returns one: an ID3 header, then frames."""
    return b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\xff\xfb\x90\x64" + tag.encode() * 64


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

    def __init__(self):
        self.objects = {}
        self.linked = []

    def configured(self):
        return True

    def put_file(self, key, path, content_type):
        self.objects[key] = (Path(path).read_bytes(), content_type)

    def presigned_get(self, key, ttl_seconds):
        self.linked.append((key, ttl_seconds))
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
            report = {"check": {"ok": True}, "music": MUSIC_REPORT}
            return httpx.Response(200, json={"id": JOB_ID, "status": "done", "outputs": [OUTPUT], "report": report})
        if request.method == "GET" and path == f"/render/{JOB_ID}/output/render.mp4":
            return httpx.Response(200, content=MP4)
        return httpx.Response(404, json={"error": "not_found"})


class Composio:
    """ComposioToolExecutor as the voice recipe calls it: every call recorded,
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


def _ok(data):
    return {"success": True, "data": {"data": data, "successful": True, "error": None}, "error": None}


def _failed(error):
    return {"success": False, "data": None, "error": error}


def _file_output(url, mimetype="audio/mpeg"):
    return _ok({"file": {"name": "speech.mp3", "mimetype": mimetype, "s3url": url}})


def _balance(credit):
    # Fish Audio's reading: the package allowance (free quota, not money) and the API credit.
    return _ok({"package": {"type": "free", "total": 100000, "balance": 99000}, "api_credit": {"credit": credit}})


_COPIES = sa.MetaData()
for _table in (
    Workspace.__table__,
    DocumentTemplate.__table__,
    ComposioEntity.__table__,
    ComposioConnection.__table__,
    ComposioActionCache.__table__,
):
    _sqlite_copy(_table, _COPIES)


def _set_config(monkeypatch, **values):
    """Patch the config object every module under test reads (one object, unless a reload split it)."""
    modules = (render, render_quota, media_render_client, media_store, voice, socials_api)
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
    _set_config(
        monkeypatch,
        AUTH_EDITION="saas",
        SOCIALS_RENDER_URL=RENDER_URL,
        SOCIALS_RENDER_TOKEN=TOKEN,
        SOCIALS_RENDER_POLL_SECONDS=0,
        SOCIALS_RENDER_MAX_WAIT_SECONDS=60,
        SOCIALS_RENDER_MEDIA_URL_TTL_SECONDS=3600,
        SOCIALS_VOICE_LINE_MAX_BYTES=16 * 1024 * 1024,
        SOCIALS_MEDIA_FETCH_TIMEOUT_SECONDS=60,
        SOCIALS_VOICE_LIST_LIMIT=30,
    )
    Deliverables.calls = []
    monkeypatch.setattr(deliverable_service, "DeliverableService", Deliverables)
    state.booked = []

    def track_media(**kwargs):
        state.booked.append({**kwargs, "scope": dict(uc.current_usage_scope())})

    monkeypatch.setattr(UsageTracker, "track_media", staticmethod(track_media))

    state.composio = Composio()
    monkeypatch.setattr(voice, "ComposioToolExecutor", state.composio.executor())
    state.fetched = []
    state.audio = {}

    async def fake_fetch(url, *, max_bytes, timeout_seconds):
        state.fetched.append((url, max_bytes, timeout_seconds))
        return state.audio[url]

    monkeypatch.setattr(voice, "fetch", fake_fetch)

    app = FastAPI()
    app.include_router(socials_api.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: state.ctx
    app.dependency_overrides[get_db] = lambda: session
    state.client = TestClient(app)
    try:
        yield state
    finally:
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


def _cache_voice_toolkits(env):
    _cache(env, "FISH_AUDIO", SPEAK, LIST_VOICES, BALANCE, "FISH_AUDIO_CREATE_VOICE_MODEL")
    _cache(env, "ELEVENLABS", ELEVEN_SPEAK, "ELEVENLABS_GET_VOICES_LIST", "ELEVENLABS_DELETE_VOICE_BY_ID")


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


def _template(env):
    template_id = uuid.uuid4()
    env.session.execute(
        sa.text(
            "INSERT INTO document_templates (id, workspace_id, name, format, data_schema, blocks) "
            "VALUES (:id, :ws, :name, 'social_video', '{}', :blocks)"
        ),
        {"id": template_id.hex, "ws": WS.hex, "name": "Countdown", "blocks": json.dumps(COMPOSITION)},
    )
    env.session.commit()
    return template_id


def _create(env, **body):
    payload = {
        "title": "Web Summit countdown",
        "copy": {"base": "Three weeks to go."},
        "format": "video",
        "template_id": str(_template(env)),
        "variables": {"headline": {"value": "Lisbon, here we come.", "claim": False}},
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

    return asyncio.run(go()), renderer, job


def _post(env, post_id):
    env.session.expire_all()
    return env.session.get(SocialPost, uuid.UUID(post_id))


def _script_fish(env):
    """Fish Audio's answers: a balance before and after, and one MP3 per line on a Composio file link."""
    env.composio.answers[BALANCE] = [_balance("10.00"), _balance("9.99")]
    links = []

    def speak(params):
        link = f"https://r2.composio.example/fish/{len(links) + 1}.mp3?X-Amz-Signature=short-lived"
        links.append(link)
        env.audio[link] = _mp3(params["text"])
        return _file_output(link)

    env.composio.answers[SPEAK] = speak
    return links


# ---------------------------------------------------------------------------
# AC2 — fish_audio connected and chosen: the same script renders with no other change
# ---------------------------------------------------------------------------


def test_with_fish_audio_chosen_the_same_script_renders_with_no_other_change(env):
    _cache_voice_toolkits(env)
    _connect(env, "FISH_AUDIO")
    post = _create(env)

    # Kokoro first: the bundle carries the script's text, and media-render speaks it.
    store = FakeStore()
    ok, kokoro, _ = _render(env, post["id"], store)
    assert ok is True and env.composio.calls == []
    (kokoro_bundle,) = kokoro.bundles
    assert voice_script(kokoro_bundle) == [("l01", "Lisbon, here we come."), ("l02", "Three weeks to go.")]

    # Fish Audio chosen: a render setting, so the post keeps its hash and status.
    before = _post(env, post["id"])
    chosen = env.client.patch(
        f"/api/socials/posts/{post['id']}",
        json={"voice": {"toolkit": "fish_audio", "voice_id": FISH_VOICE, "name": "Energetic narrator"}},
    )
    assert chosen.status_code == 200, chosen.text
    assert chosen.json()["voice"] == {"toolkit": "fish_audio", "voice_id": FISH_VOICE, "name": "Energetic narrator"}
    assert (chosen.json()["status"], chosen.json()["content_hash"]) == (before.status, before.content_hash)

    links = _script_fish(env)
    env.booked.clear()
    ok, fish, job = _render(env, post["id"], store)
    assert ok is True and _post(env, post["id"]).status == "needs_approval"
    assert job.voice.toolkit == "fish_audio" and job.voice.speak_action.slug == SPEAK

    # One Composio call per line through the allowlisted speech action, the balance read around them.
    assert env.composio.slugs() == [BALANCE, SPEAK, SPEAK, BALANCE]
    speaks = [call for call in env.composio.calls if call["action"] == SPEAK]
    assert [call["params"] for call in speaks] == [
        {"text": "Lisbon, here we come.", "voice_model_ids": [FISH_VOICE], "format": "mp3"},
        {"text": "Three weeks to go.", "voice_model_ids": [FISH_VOICE], "format": "mp3"},
    ]
    assert all(
        (call["workspace_id"], call["app_name"], call["agent_id"], call["skip_validation"]) == (WS, "FISH_AUDIO", 0, True)
        for call in env.composio.calls
    )

    # Each line's audio was fetched from its short-lived link and copied into our storage
    # before the bundle went to media-render.
    assert [url for url, _, _ in env.fetched] == links
    keys = [f"social-media/{WS}/{post['id']}/voice-l01.mp3", f"social-media/{WS}/{post['id']}/voice-l02.mp3"]
    assert store.objects[keys[0]] == (_mp3("Lisbon, here we come."), "audio/mpeg")
    assert store.objects[keys[1]] == (_mp3("Three weeks to go."), "audio/mpeg")
    (stored_at_submit,) = fish.stored_at_submit
    assert set(keys) <= set(stored_at_submit)

    # The same bundle, but each spoken line names its stored file instead of its text.
    (fish_bundle,) = fish.bundles
    lines = fish_bundle["audio"]["voice"]["lines"]
    assert lines == [
        {"id": "l01", "at": 0.3, "path": f"{VOICE_DIR}l01.mp3"},
        {"id": "l02", "at": 2.4, "path": f"{VOICE_DIR}l02.mp3"},
    ]
    assert fish_bundle["media"] == [
        {"path": f"{VOICE_DIR}l01.mp3", "url": f"http://minio:9000/automatos-ai/{keys[0]}?X-Amz-Expires=3600&X-Amz-Signature=sig"},
        {"path": f"{VOICE_DIR}l02.mp3", "url": f"http://minio:9000/automatos-ai/{keys[1]}?X-Amz-Expires=3600&X-Amz-Signature=sig"},
    ]
    stripped = copy.deepcopy(fish_bundle)
    stripped.pop("media")
    stripped["audio"]["voice"]["lines"] = kokoro_bundle["audio"]["voice"]["lines"]
    assert stripped == kokoro_bundle, "anything but the spoken lines changed"

    # D13: Fish Audio's balance difference is booked on the media lane against the post.
    voice_booking, render_booking = env.booked
    assert voice_booking["provider"] == "fish_audio" and voice_booking["model_id"] == SPEAK.lower()
    assert voice_booking["usd"] == pytest.approx(0.01)
    assert voice_booking["units"] == len("Lisbon, here we come.".encode()) + len("Three weeks to go.".encode())
    assert voice_booking["error_message"] is None
    assert voice_booking["scope"]["request_type"] == "media" and voice_booking["scope"]["workspace_id"] == WS
    assert voice_booking["scope"]["execution_id"] == f"social_post:{post['id']}"
    assert render_booking["provider"] == MEDIA_RENDER_PROVIDER


def test_a_line_the_toolkit_cannot_speak_fails_the_render_and_still_books_the_spend(env):
    _cache_voice_toolkits(env)
    _connect(env, "FISH_AUDIO")
    post = _create(env, voice={"toolkit": "fish_audio", "voice_id": FISH_VOICE})
    links = _script_fish(env)
    first = env.composio.answers[SPEAK]
    env.composio.answers[SPEAK] = [first, lambda params: _failed("voice model not found")]

    store = FakeStore()
    ok, renderer, _ = _render(env, post["id"], store)

    assert ok is False and renderer.bundles == [], "nothing reached media-render"
    row = _post(env, post["id"])
    assert row.status == "failed"
    assert "Fish Audio could not speak line l02: voice model not found" in row.review_log[-1]["comment"]
    assert list(store.objects) == [f"social-media/{WS}/{post['id']}/voice-l01.mp3"] and len(links) == 1
    (booking,) = env.booked
    assert booking["provider"] == "fish_audio" and booking["usd"] == pytest.approx(0.01)


def test_a_balance_that_cannot_be_read_first_speaks_nothing(env):
    _cache_voice_toolkits(env)
    _connect(env, "FISH_AUDIO")
    post = _create(env, voice={"toolkit": "fish_audio", "voice_id": FISH_VOICE})
    env.composio.answers[BALANCE] = _failed("rate limited")

    ok, renderer, _ = _render(env, post["id"], FakeStore())

    assert ok is False and renderer.bundles == []
    assert env.composio.slugs() == [BALANCE]
    assert env.booked == []
    assert "balance could not be read" in _post(env, post["id"]).review_log[-1]["comment"]


def test_a_toolkit_that_returns_no_audio_fails_the_render_saying_so(env):
    _cache_voice_toolkits(env)
    _connect(env, "FISH_AUDIO")
    post = _create(env, voice={"toolkit": "fish_audio", "voice_id": FISH_VOICE})
    env.composio.answers[BALANCE] = [_balance("5"), _balance("5")]
    env.composio.answers[SPEAK] = lambda params: _ok({"file": {"name": "x.mp3", "mimetype": "audio/mpeg", "s3url": "https://r2.composio.example/x"}})
    env.audio["https://r2.composio.example/x"] = b"<html>expired</html>"

    ok, _, _ = _render(env, post["id"], FakeStore())

    assert ok is False
    assert "returned something that is not audio for line l01" in _post(env, post["id"]).review_log[-1]["comment"]


def test_elevenlabs_books_characters_at_no_price(env):
    _cache_voice_toolkits(env)
    _connect(env, "ELEVENLABS")
    post = _create(env, voice={"toolkit": "elevenlabs", "voice_id": "21m00Tcm4TlvDq8ikWAM"})
    inline = "data:audio/mpeg;base64," + base64.b64encode(_mp3("eleven")).decode()
    env.composio.answers[ELEVEN_SPEAK] = lambda params: _ok({"audio": inline})

    store = FakeStore()
    ok, renderer, _ = _render(env, post["id"], store)

    assert ok is True and env.fetched == [], "inline audio needs no fetch"
    assert [call["params"] for call in env.composio.calls] == [
        {"text": "Lisbon, here we come.", "voice_id": "21m00Tcm4TlvDq8ikWAM"},
        {"text": "Three weeks to go.", "voice_id": "21m00Tcm4TlvDq8ikWAM"},
    ]
    voice_booking = env.booked[0]
    assert voice_booking["provider"] == "elevenlabs" and voice_booking["usd"] == 0.0
    assert voice_booking["units"] == len("Lisbon, here we come.") + len("Three weeks to go.")
    assert [line["path"] for line in renderer.bundles[0]["audio"]["voice"]["lines"]] == [f"{VOICE_DIR}l01.mp3", f"{VOICE_DIR}l02.mp3"]


def test_a_render_with_a_voice_the_workspace_lost_is_refused_with_nothing_changed(env):
    _cache_voice_toolkits(env)
    _connect(env, "FISH_AUDIO")
    post = _create(env, voice={"toolkit": "fish_audio", "voice_id": FISH_VOICE})
    env.session.query(ComposioConnection).delete()
    env.session.commit()

    resp = env.client.post(f"/api/socials/posts/{post['id']}/render")

    assert resp.status_code == 422
    assert "Fish Audio is not connected in this workspace" in resp.json()["detail"]
    assert env.launched == [] and _post(env, post["id"]).status == "draft"


def test_only_a_credit_billed_toolkit_takes_the_credit_window(env, monkeypatch):
    """Fish Audio's balance difference is read inside the one credit window per
    workspace and toolkit; ElevenLabs reads no balance and takes none."""
    _cache_voice_toolkits(env)
    _connect(env, "FISH_AUDIO", "ELEVENLABS")
    windows = []

    @asynccontextmanager
    async def window(factory, workspace_id, toolkit):
        windows.append((workspace_id, toolkit))
        yield

    monkeypatch.setattr(voice, "credit_window", window)
    _script_fish(env)
    inline = "data:audio/mpeg;base64," + base64.b64encode(_mp3("eleven")).decode()
    env.composio.answers[ELEVEN_SPEAK] = lambda params: _ok({"audio": inline})
    caps = media_capabilities(env.session, WS)
    post_id = uuid.uuid4()
    for chosen in ({"toolkit": "fish_audio", "voice_id": FISH_VOICE}, {"toolkit": "elevenlabs", "voice_id": "e1"}):
        plan = voice.plan_for(chosen, caps)
        spoken = asyncio.run(
            voice.speak(plan, workspace_id=WS, post_id=post_id, lines=[("l01", "One line.")],
                        session_factory=env.factory, store=FakeStore())
        )
        assert list(spoken) == ["l01"]
    assert windows == [(WS, "fish_audio")]


def test_on_a_database_without_advisory_locks_the_window_is_the_process_lock(env):
    async def go():
        async with toolkit.credit_window(env.factory, WS, "fish_audio"):
            return toolkit.account_lock(WS, "fish_audio").locked()

    assert asyncio.run(go()) is True


# ---------------------------------------------------------------------------
# AC3 — without a voice toolkit, Kokoro only; the others link to the connect flow
# ---------------------------------------------------------------------------


def _sources(env):
    resp = env.client.get("/api/socials/voices")
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_without_a_voice_toolkit_only_kokoro_is_offered_and_the_others_link_to_connect(env):
    _cache_voice_toolkits(env)
    body = _sources(env)
    assert body["problem"] is None
    assert body["sources"] == [
        {"toolkit": "kokoro", "label": "Kokoro (built in)", "status": "available", "builtin": True, "lists_voices": False},
        {"toolkit": "elevenlabs", "label": "ElevenLabs", "status": "connect", "builtin": False, "lists_voices": False},
        {"toolkit": "fish_audio", "label": "Fish Audio", "status": "connect", "builtin": False, "lists_voices": False},
    ]
    offered = [s["toolkit"] for s in body["sources"] if s["status"] == "available"]
    assert offered == ["kokoro"]


def test_a_connected_voice_toolkit_is_offered_beside_kokoro(env):
    _cache_voice_toolkits(env)
    _connect(env, "FISH_AUDIO")
    sources = {s["toolkit"]: s for s in _sources(env)["sources"]}
    assert (sources["fish_audio"]["status"], sources["fish_audio"]["lists_voices"]) == ("available", True)
    assert sources["elevenlabs"]["status"] == "connect"
    assert sources["kokoro"]["status"] == "available"


def test_a_connected_toolkit_the_workspace_cannot_use_says_why(env):
    # Fish Audio connected, but its balance action is not in the synced schemas:
    # its spend could not be booked, so it is not offered.
    _cache(env, "FISH_AUDIO", SPEAK, LIST_VOICES)
    _connect(env, "FISH_AUDIO")
    fish = {s["toolkit"]: s for s in _sources(env)["sources"]}["fish_audio"]
    assert fish["status"] == "unavailable" and "balance action is not available" in fish["reason"]


def test_a_denied_speech_action_is_never_offered(env):
    _cache_voice_toolkits(env)
    _connect(env, "FISH_AUDIO")
    _deny(env, SPEAK)
    fish = {s["toolkit"]: s for s in _sources(env)["sources"]}["fish_audio"]
    assert fish["status"] == "unavailable"
    with pytest.raises(voice.VoiceUnavailable, match="speech action is not available"):
        voice.plan_for({"toolkit": "fish_audio", "voice_id": FISH_VOICE}, media_capabilities(env.session, WS))


def test_an_allowlist_that_cannot_be_used_offers_kokoro_alone_and_says_why(env):
    _cache_voice_toolkits(env)
    _connect(env, "FISH_AUDIO")
    table = SystemSetting.__table__
    with env.engine.begin() as conn:
        conn.execute(table.update().where(table.c.category == "socials", table.c.key == "media_actions").values(value="not json"))
    body = _sources(env)
    assert body["problem"] and "not usable" in body["problem"]
    assert [s["toolkit"] for s in body["sources"] if s["status"] == "available"] == ["kokoro"]


# ---------------------------------------------------------------------------
# A toolkit's voices; saving a voice
# ---------------------------------------------------------------------------


def test_a_toolkits_voices_are_listed_through_its_allowlisted_action(env):
    _cache_voice_toolkits(env)
    _connect(env, "FISH_AUDIO")
    env.composio.answers[LIST_VOICES] = _ok(
        {"total": 2, "items": [
            {"_id": "v1", "title": "Energetic narrator", "description": "Upbeat, clear", "author": {"_id": "a1"}},
            {"_id": "v2", "title": "Calm narrator"},
        ]}
    )
    resp = env.client.get("/api/socials/voices/fish_audio", params={"q": "narr"})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "toolkit": "fish_audio",
        "voices": [
            {"id": "v1", "name": "Energetic narrator", "description": "Upbeat, clear"},
            {"id": "v2", "name": "Calm narrator"},
        ],
    }
    (call,) = env.composio.calls
    assert call["action"] == LIST_VOICES
    assert call["params"] == {"sort_by": "score", "page_size": 30, "title": "narr"}


def test_listing_voices_needs_the_toolkit_and_says_when_it_does_not_answer(env):
    _cache_voice_toolkits(env)
    resp = env.client.get("/api/socials/voices/fish_audio")
    assert resp.status_code == 422 and "not connected" in resp.json()["detail"]
    assert env.client.get("/api/socials/voices/playht").status_code == 422
    _connect(env, "FISH_AUDIO")
    env.composio.answers[LIST_VOICES] = _failed("upstream 500")
    resp = env.client.get("/api/socials/voices/fish_audio")
    assert resp.status_code == 502 and "did not list its voices: upstream 500" in resp.json()["detail"]
    env.role = "viewer"
    assert env.client.get("/api/socials/voices/fish_audio").status_code == 403


def test_a_voice_is_saved_only_when_the_workspace_can_speak_with_it(env):
    _cache_voice_toolkits(env)
    post = _create(env)
    url = f"/api/socials/posts/{post['id']}"
    fish = {"toolkit": "fish_audio", "voice_id": FISH_VOICE}

    refused = env.client.patch(url, json={"voice": fish})
    assert refused.status_code == 422 and "Fish Audio is not connected in this workspace" in refused.json()["detail"]
    assert _post(env, post["id"]).voice is None

    _connect(env, "FISH_AUDIO")
    assert env.client.patch(url, json={"voice": fish}).json()["voice"] == fish
    # Kokoro again, however it is said.
    for kokoro in (None, {}, {"toolkit": "kokoro"}):
        assert env.client.patch(url, json={"voice": kokoro}).json()["voice"] is None
        env.client.patch(url, json={"voice": fish})
    assert env.client.patch(url, json={"voice": {"toolkit": "fish_audio"}}).status_code == 422
    assert env.client.patch(url, json={"voice": {"toolkit": "kokoro", "voice_id": "x"}}).status_code == 422
    create_refused = env.client.post("/api/socials/posts", json={"title": "T", "voice": {"toolkit": "elevenlabs", "voice_id": "x"}})
    assert create_refused.status_code == 422


def test_changing_the_voice_leaves_an_approval_standing(env):
    _cache_voice_toolkits(env)
    _connect(env, "FISH_AUDIO")
    post = _create(env)
    submitted = env.client.post(f"/api/socials/posts/{post['id']}/submit").json()
    approved = env.client.post(f"/api/socials/posts/{post['id']}/approve", json={"content_hash": submitted["content_hash"]}).json()
    assert approved["status"] == "approved"

    changed = env.client.patch(f"/api/socials/posts/{post['id']}", json={"voice": {"toolkit": "fish_audio", "voice_id": FISH_VOICE}})

    assert changed.status_code == 200
    body = changed.json()
    assert (body["status"], body["content_hash"], body["approved_hash"]) == ("approved", approved["content_hash"], approved["approved_hash"])


# ---------------------------------------------------------------------------
# The shape, the bundle, the recipe's parts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        ({}, None),
        ({"toolkit": "kokoro"}, None),
        ({"toolkit": " Fish_Audio ", "voice_id": " v1 ", "name": " Narrator "}, {"toolkit": "fish_audio", "voice_id": "v1", "name": "Narrator"}),
        ({"toolkit": "elevenlabs", "voice_id": "v2", "name": "  "}, {"toolkit": "elevenlabs", "voice_id": "v2"}),
    ],
)
def test_the_voice_shape(value, expected):
    assert service.validate_voice(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "fish_audio",
        {"toolkit": "fish_audio"},
        {"toolkit": "fish_audio", "voice_id": 7},
        {"toolkit": "fish audio", "voice_id": "v1"},
        {"toolkit": "kokoro", "name": "Heart"},
        {"toolkit": "fish_audio", "voice_id": "v1", "speed": 1.1},
        {"toolkit": "fish_audio", "voice_id": "x" * 201},
    ],
)
def test_a_voice_that_is_not_the_shape_is_refused(value):
    with pytest.raises(service.InvalidPost):
        service.validate_voice(value)


def test_the_voice_is_outside_the_content_hash():
    post = SimpleNamespace(copy={"base": "x"}, variables={}, sources={}, format="video", template_id=None, media={}, voice=None)
    spoken = SimpleNamespace(**{**vars(post), "voice": {"toolkit": "fish_audio", "voice_id": "v1"}})
    assert service.compute_content_hash(post) == service.compute_content_hash(spoken)
    assert "voice" in service.EDITABLE_FIELDS and "voice" not in service.CONTENT_FIELDS


def test_with_voice_files_changes_only_the_spoken_lines():
    bundle = {
        "workspace_id": "w", "composition": {"html": "<html></html>"}, "media": [{"path": "assets/slots/hook.mp4", "url": "u0"}],
        "audio": {"voice": {"voice": "af_heart", "lines": [
            {"id": "l01", "at": 0.3, "text": "One"}, {"id": "l02", "at": 2.0, "path": "assets/voice/kept.wav"},
        ]}, "music": {"track": "t"}},
    }
    out = with_voice_files(bundle, {"l01": ("mp3", "u1")})
    assert out["audio"]["voice"]["lines"] == [
        {"id": "l01", "at": 0.3, "path": "assets/voice/l01.mp3"}, {"id": "l02", "at": 2.0, "path": "assets/voice/kept.wav"},
    ]
    assert out["media"] == [{"path": "assets/slots/hook.mp4", "url": "u0"}, {"path": "assets/voice/l01.mp3", "url": "u1"}]
    assert out["audio"]["music"] == {"track": "t"} and bundle["audio"]["voice"]["lines"][0]["text"] == "One"
    assert with_voice_files(bundle, {}) == bundle
    with pytest.raises(ValueError, match="no line l09"):
        with_voice_files(bundle, {"l09": ("mp3", "u")})


@pytest.mark.parametrize(
    "response, url, data",
    [
        (_file_output("https://r2.example/a.mp3")["data"], "https://r2.example/a.mp3", None),
        ({"data": {"result": {"audio_url": "https://cdn.example/b.wav"}}}, "https://cdn.example/b.wav", None),
        ({"data": {"audio": "data:audio/mpeg;base64," + base64.b64encode(b"ID3" + b"x" * 80).decode()}}, None, b"ID3" + b"x" * 80),
        ({"data": {"audio_base64": base64.b64encode(b"RIFF" + b"y" * 90).decode()}}, None, b"RIFF" + b"y" * 90),
    ],
)
def test_the_file_a_toolkit_returned_is_found(response, url, data):
    found = files.returned_file(response)
    assert (found.url, found.data) == (url, data)


def test_no_file_in_an_answer_is_none():
    assert files.returned_file({"data": {"message": "queued", "id": "job-1"}, "successful": True}) is None
    assert files.returned_file(None) is None


@pytest.mark.parametrize(
    "head, extension",
    [
        (b"ID3\x04\x00", "mp3"),
        (b"\xff\xfb\x90\x64", "mp3"),
        (b"RIFF\x24\x08\x00\x00WAVEfmt ", "wav"),
        (b"OggS\x00\x02", "ogg"),
        (b"fLaC\x00\x00", "flac"),
        (b"\x00\x00\x00\x20ftypM4A ", "m4a"),
        (b"\xff\xf1\x50\x80", "aac"),
        (b"<html><body>", None),
        (b"{\"error\": 1}", None),
    ],
)
def test_audio_is_known_by_its_first_bytes(head, extension):
    assert voice.audio_extension(head + b"\x00" * 16) == extension


def test_the_balance_read_is_the_money_not_the_free_quota():
    reading = _balance("12.3456")["data"]
    assert voice.balance_of(reading, voice.RECIPES["fish_audio"].balance_keys) == Decimal("12.3456")
    assert voice.balance_of({"data": {"package": {"balance": 99000}}}, voice.RECIPES["fish_audio"].balance_keys) is None
    assert voice.balance_of({"data": {"credit": "not a number"}}, ("credit",)) is None


def test_a_parameter_the_cached_schema_does_not_take_refuses_the_call():
    action = SimpleNamespace(slug=SPEAK, parameters={"properties": {"text": {}}})
    with pytest.raises(voice.VoiceToolError, match="takes no voice_model_ids"):
        voice._params(action, {"text": "x", "voice_model_ids": ["v"]}, ("text", "voice_model_ids"))
    assert voice._params(action, {"text": "x", "format": "mp3"}, ("text",)) == {"text": "x"}
    unsynced = SimpleNamespace(slug=SPEAK, parameters={})
    assert voice._params(unsynced, {"text": "x", "format": "mp3"}, ("text",)) == {"text": "x", "format": "mp3"}


def test_the_voices_are_read_from_either_toolkits_listing():
    eleven = {"data": {"voices": [{"voice_id": "e1", "name": "Rachel", "labels": {"accent": "american"}}, {"voice_id": "e2", "name": "Adam"}]}}
    assert voice.parse_voices(eleven, limit=5, query="ada") == [{"id": "e2", "name": "Adam"}]
    assert voice.parse_voices(eleven, limit=1) == [{"id": "e1", "name": "Rachel"}]
    assert voice.parse_voices({"data": {"items": []}}, limit=5) == []


# ── the fetch: public addresses only, pinned, capped ────────────────────────
def _fetch_with(monkeypatch, handler, ok=True):
    """The outbound check and the network, stood in: every URL checked is recorded."""
    target = OutboundTarget(True, "OK", "r2.example", "93.184.216.34", 443, "https") if ok else OutboundTarget(False, "resolves into a blocked range (private)", "minio")
    checked = []

    async def resolve(url, *, enforce_switch=True):
        assert enforce_switch is False
        checked.append(url)
        return target

    monkeypatch.setattr(files, "resolve_outbound_async", resolve)
    real = httpx.AsyncClient

    def client(**kwargs):
        return real(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(files.httpx, "AsyncClient", client)
    return checked


def test_a_link_into_the_private_network_is_never_fetched(monkeypatch):
    seen = []
    _fetch_with(monkeypatch, lambda request: seen.append(request) or httpx.Response(200, content=b"x"), ok=False)
    with pytest.raises(files.FileOutputError, match="not an address this server may fetch"):
        asyncio.run(files.fetch("http://minio:9000/bucket/x", max_bytes=100, timeout_seconds=5))
    assert seen == []


def test_the_fetch_is_pinned_and_capped(monkeypatch):
    seen = []

    def handler(request):
        seen.append((request.url.host, request.headers["host"]))
        return httpx.Response(200, content=b"ID3" + b"a" * 97)

    _fetch_with(monkeypatch, handler)
    assert asyncio.run(files.fetch("https://r2.example/a.mp3", max_bytes=100, timeout_seconds=5)) == b"ID3" + b"a" * 97
    assert seen == [("93.184.216.34", "r2.example")]
    with pytest.raises(files.FileOutputError, match="larger than the 50-byte limit"):
        asyncio.run(files.fetch("https://r2.example/a.mp3", max_bytes=50, timeout_seconds=5))


def test_a_redirect_is_checked_again_and_an_error_status_refused(monkeypatch):
    def handler(request):
        if request.url.path == "/moved":
            return httpx.Response(302, headers={"location": "https://r2.example/gone"})
        return httpx.Response(403, content=b"expired")

    checked = _fetch_with(monkeypatch, handler)
    with pytest.raises(files.FileOutputError, match="answered 403"):
        asyncio.run(files.fetch("https://r2.example/moved", max_bytes=100, timeout_seconds=5))
    assert checked == ["https://r2.example/moved", "https://r2.example/gone"]


# ---------------------------------------------------------------------------
# The column: the wave's one migration, create_all-first safe
# ---------------------------------------------------------------------------


def _columns(conn):
    return {c["name"] for c in sa.inspect(conn).get_columns("social_posts")}


def _step(conn, name):
    with Operations.context(MigrationContext.configure(conn)):
        getattr(WAVE1, name)()


def test_the_wave_migration_adds_the_voice_column_once():
    """On SQLite the add; the drop is Postgres DDL, proved by the @integration test below."""
    engine = sa.create_engine("sqlite://", poolclass=StaticPool)
    try:
        with engine.begin() as conn:
            with Operations.context(MigrationContext.configure(conn)):
                WAVE0._create_social_posts()
            assert "voice" not in _columns(conn)
            _step(conn, "add_post_voice_column")
            _step(conn, "add_post_voice_column")
            (column,) = [c for c in sa.inspect(conn).get_columns("social_posts") if c["name"] == "voice"]
            assert column["nullable"] is True
    finally:
        engine.dispose()


def test_a_schema_without_the_posts_table_is_left_alone():
    engine = sa.create_engine("sqlite://", poolclass=StaticPool)
    try:
        with engine.begin() as conn:
            _step(conn, "add_post_voice_column")
            _step(conn, "drop_post_voice_column")
            assert not sa.inspect(conn).has_table("social_posts")
    finally:
        engine.dispose()


def test_the_upgrade_runs_the_column_step():
    source = (VERSIONS / "prd251_wave1.py").read_text(encoding="utf-8")
    upgrade = source[source.index("def upgrade()"): source.index("def downgrade()")]
    downgrade = source[source.index("def downgrade()"):]
    assert "add_post_voice_column()" in upgrade and "drop_post_voice_column()" in downgrade


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
def test_a_credit_window_held_by_another_worker_is_waited_for(pg_engine, monkeypatch):
    """Production runs several uvicorn workers: a window another process holds (here,
    another connection) keeps this one out until it ends, so two renders' balance
    readings never overlap."""
    monkeypatch.setattr(toolkit.config, "SOCIALS_RENDER_POLL_SECONDS", 0.05, raising=False)
    factory = sessionmaker(bind=pg_engine)
    params = {"namespace": toolkit.CREDIT_LOCK_NAMESPACE, "key": toolkit.credit_lock_key(WS, "fish_audio")}

    async def scenario():
        entered = asyncio.Event()

        async def second():
            async with toolkit.advisory_lock(factory, params["namespace"], params["key"]):
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
def test_create_all_first_then_the_upgrade_twice_leaves_one_jsonb_voice_column(pg_engine):
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
                    "WHERE table_name = 'social_posts' AND column_name = 'voice'"
                )
            ).scalars().all()
            assert kind == ["jsonb"]
            with Operations.context(MigrationContext.configure(conn)):
                WAVE1.downgrade()
            assert "voice" not in _columns(conn)
            with Operations.context(MigrationContext.configure(conn)):
                WAVE1.upgrade()
            assert "voice" in _columns(conn)
        finally:
            trans.rollback()


# ---------------------------------------------------------------------------
# AC1 — the CI render of the fixture script, pinned
# ---------------------------------------------------------------------------


def test_the_media_render_job_renders_the_fixture_script_with_speech_in_every_window():
    """US-111: the media-render job renders the fixture script through the API and
    asserts speech in every script window, with the over-long first line fitted."""
    import yaml

    workflow = yaml.safe_load((_ORCH.parent / ".github" / "workflows" / "test.yml").read_text())
    commands = "\n".join(step.get("run", "") for step in workflow["jobs"]["media-render"]["steps"])
    assert '"$IMAGE" fixture-bundle script' in commands
    assert "ci/render_bundle.py" in commands and "--save-as script.mp4" in commands
    assert "ci/assert_script_windows.py" in commands and "--expect-fitted l01" in commands
    assert "--duration 9.0" in commands
    script = json.loads((_ORCH.parent / "services" / "media-render" / "fixtures" / "script.bundle.json").read_text())
    assert [line["id"] for line in script["audio"]["voice"]["lines"]] == ["l01", "l02", "l03", "l04"]
