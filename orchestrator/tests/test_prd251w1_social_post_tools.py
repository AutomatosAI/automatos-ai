"""PRD-251 Wave 1, US-116 (S4.1 pulled forward) — agents draft Socials posts; people approve them.

The five tools (``platform_create_social_post``, ``platform_update_social_post``,
``platform_submit_social_post``, ``platform_get_social_post`` and
``platform_list_social_posts``) run through the real ``PlatformActionExecutor``
over in-memory SQLite, beside the real Socials router in the same workspace:
the real switches (only the master switch's system setting is stubbed), the
real render lifecycle and quota, media-render mocked at the HTTP layer
(``httpx.MockTransport``). An agent's call carries what ``exec_platform``
mints for it (``_agent_id``, ``_agent_name``). Pins:

* **AC1**: a post the tool drafts is the post REST drafts: the Socials API
  lists it with the same status, content_hash and sources, and the tool
  records the acting agent (``created_by`` ``agent:<id>``, a ``draft`` entry
  naming it). The template is taken by id or by name, a social one of the
  workspace only. Sources must resolve in the workspace (D7).
* **AC2**: editing an approved post through the tool voids the approval,
  through the Wave 0 service's ``update_post``; the edit names the agent.
* **AC3**: list filters by status and returns only the caller's workspace;
  submit moves a draft to needs_approval; no registered tool approves,
  schedules or publishes, no socials schema has such a parameter, and nothing
  the tools run reaches such a step. A field the post does not take is refused.
* **AC4**: with the platform or the workspace switch off, every tool refuses
  saying which, and nothing is read or written.
* **AC5**: ``render`` (create's default) is the US-104 render: the post
  renders through the mocked media-render and waits for approval with its
  files; past the plan's render minutes the draft is saved and the render is
  refused before media-render. Update renders a failed post again when asked.
* **S1.7**: a chart template's rows come from a report (``chart_report``),
  exactly as the composer's binding route gives them, and are never typed.
"""
from __future__ import annotations

import ast
import asyncio
import inspect
import json
import os
import re
import sys
import textwrap
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

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
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.dialects.postgresql import ARRAY, JSONB  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import core.models  # noqa: E402,F401  (registers every mapper)
import api.socials as socials_api  # noqa: E402
import core.auth.workspace_permission as permission_mod  # noqa: E402
import core.media_render_client as media_render_client  # noqa: E402
import core.media_render_quota as render_quota  # noqa: E402
import modules.socials.media_store as media_store  # noqa: E402
import modules.socials.render as render  # noqa: E402
import modules.socials.report_charts as report_charts  # noqa: E402
import modules.socials.service as service  # noqa: E402
import modules.socials.settings as socials_settings  # noqa: E402
import modules.tools.discovery.handlers_socials as handlers_socials  # noqa: E402
import services.deliverable_service as deliverable_service  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.chart_binding import CHART_KINDS  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.llm.providers import MEDIA_RENDER_PROVIDER  # noqa: E402
from core.llm.usage_context import LANE_MEDIA  # noqa: E402
from core.llm.usage_tracker import UsageTracker  # noqa: E402
from core.media_render_client import MediaRenderClient  # noqa: E402
from core.models.core import DocumentTemplate, LLMUsage  # noqa: E402
from core.models.socials import SOCIAL_POST_FORMATS, SOCIAL_POST_STATUSES, SocialPost, SocialPostTarget  # noqa: E402
from core.models.workspaces import Workspace  # noqa: E402
from core.report_tables import PARTS  # noqa: E402
from modules.documents.social_starters import social_starters  # noqa: E402
from modules.tools.discovery import actions_socials  # noqa: E402
from modules.tools.discovery.action_registry import get_action_registry  # noqa: E402
from modules.tools.discovery.platform_executor import PlatformActionExecutor  # noqa: E402

# Hex with letters, so SQLite keeps every UUID column as text.
WS = uuid.UUID("00000000-0000-0000-0000-0000000001c6")
WS_OTHER = uuid.UUID("00000000-0000-0000-0000-0000000001c7")
WS_OFF = uuid.UUID("00000000-0000-0000-0000-0000000001c8")
CREATED = datetime(2026, 9, 1, 9, 0)
MADE = datetime(2026, 9, 21, 9, 30, tzinfo=timezone.utc)
SOCIALS_ON = {"socials": {"enabled": True}}

AGENT = {"_agent_id": 7, "_agent_name": "Social Media Director"}
ACTOR = "agent:7"
TOOLS = (
    "platform_create_social_post",
    "platform_update_social_post",
    "platform_submit_social_post",
    "platform_get_social_post",
    "platform_list_social_posts",
)
# The steps no tool may reach: the review, the schedule and the way out (D6, D14).
NOT_FOR_AGENTS = {
    "approve", "approve_social_post", "request_changes", "request_changes_social_post",
    "reject", "reject_social_post", "schedule", "schedule_social_post", "unschedule",
    "unschedule_social_post", "publish_post", "publish_social_post_now",
}
FORBIDDEN_WORD = re.compile(r"approv|schedul|publish", re.IGNORECASE)

RENDER_URL = "http://media-render:8090"
TOKEN = "render-secret"
JOB_ID = "c" * 32
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"frame-bytes " * 300
OUTPUT = {"name": "render.mp4", "aspect": "9:16", "width": 1080, "height": 1920, "bytes": len(MP4), "duration": 30.0}
COMPOSITION = {
    # A social template as S1.2 defines it (core/social_templates.py): a full document.
    "html": '<!doctype html><html><head></head><body><div id="root" data-composition-id="main" data-width="1080" '
            'data-height="1920" data-duration="30"><h1>{{ headline }}</h1></div></body></html>',
    "css": "h1 { color: var(--brand-text); }",
    "variables_schema": {"headline": {"type": "text", "claim": True}},
    "sizes": ["1080x1920"],
    "audio_plan": {"voice": {"voice": "af_heart", "speed": 0.95, "lines": [{"id": "l01", "at": 0.3, "text": "Three weeks to go."}]}},
}
LAUNCH_SOURCE = {"kind": "url", "ref": "https://example.com/launch", "as_of": "2026-09-01T00:00:00+00:00"}
REPORT_TITLE = "September social report"
CHANNELS = (
    "# September social report\n\n"
    "| Channel | Posts | Reach |\n"
    "| --- | ---: | ---: |\n"
    "| Instagram | 14 | 48,210 |\n"
    "| LinkedIn | 9 | 21,480 |\n"
    "| TikTok | 6 | 19,305 |\n"
    "| Threads | 8 | 7,940 |\n"
    "| X | 11 | 6,115 |\n"
    "| YouTube Shorts | 2 | 3,020 |\n"
)
REACH = [("Instagram", "48,210"), ("LinkedIn", "21,480"), ("TikTok", "19,305"), ("Threads", "7,940"), ("X", "6,115")]


# ---------------------------------------------------------------------------
# The harness
# ---------------------------------------------------------------------------


def _portable(col_type):
    if isinstance(col_type, (JSONB, ARRAY)):
        return sa.JSON()
    if isinstance(col_type, sa.Uuid) or type(col_type).__name__.upper() == "UUID":
        return sa.Uuid()
    return col_type


def _sqlite_copy(table, metadata):
    """A column-for-column copy SQLite can build (JSONB / ARRAY → JSON, UUID → CHAR(32))."""
    columns = [sa.Column(c.name, _portable(c.type), primary_key=c.primary_key) for c in table.columns]
    return sa.Table(table.name, metadata, *columns)


_TABLES = sa.MetaData()
_sqlite_copy(Workspace.__table__, _TABLES)
_sqlite_copy(DocumentTemplate.__table__, _TABLES)
# agent_reports has no ORM model (alembic prd76_agent_reports + prd133b's deleted_at): the columns read here.
AGENT_REPORTS = sa.Table(
    "agent_reports",
    _TABLES,
    sa.Column("id", sa.String(36), primary_key=True),
    sa.Column("workspace_id", sa.String(36), nullable=False),
    sa.Column("report_type", sa.String(30)),
    sa.Column("title", sa.String(255), nullable=False),
    sa.Column("summary", sa.String(500)),
    sa.Column("file_path", sa.String(1024), nullable=False),
    sa.Column("metrics", sa.JSON),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("deleted_at", sa.DateTime(timezone=True)),
)


class FakeWorkspaceClient:
    """The workspace worker's file read (core.workspace_client.WorkspaceClient), from the test's files."""

    files: dict = {}

    def __init__(self, workspace_id):
        self.workspace_id = workspace_id

    async def read_file(self, path):
        content = type(self).files.get((self.workspace_id, path))
        if content is None:
            return {"success": False, "error": "File not found", "status_code": 404}
        return {"success": True, "content": content}


class FakeStore:
    """The documents bucket, in memory."""

    def __init__(self):
        self.objects = {}

    def configured(self):
        return True

    def put_file(self, key, path, content_type):
        self.objects[key] = (Path(path).read_bytes(), content_type)


class Deliverables:
    """Stands in for DeliverableService (its SQL is Postgres-only): records register()."""

    calls: list = []

    def __init__(self, db, workspace_id):
        self.workspace_id = workspace_id

    def register(self, **kwargs):
        type(self).calls.append({"workspace_id": self.workspace_id, **kwargs})
        return {"success": True, "deliverable_id": f"d-video-{len(type(self).calls)}", "created": True}


class Renderer:
    """media-render over httpx.MockTransport: the real client talks to it."""

    def __init__(self):
        self.bundles = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path == "/render":
            self.bundles.append(json.loads(request.content))
            return httpx.Response(202, json={"id": JOB_ID, "status": "rendering", "outputs": [], "report": {}})
        if request.method == "GET" and path == f"/render/{JOB_ID}":
            return httpx.Response(200, json={
                "id": JOB_ID, "status": "done", "outputs": [OUTPUT],
                "report": {"check": {"ok": True, "errors": 0}}, "error": None,
            })
        if request.method == "GET" and path == f"/render/{JOB_ID}/output/render.mp4":
            return httpx.Response(200, content=MP4)
        return httpx.Response(404, json={"error": "not_found", "message": f"no route {path}"})


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


@pytest.fixture
def env(monkeypatch):
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    _TABLES.create_all(engine)
    SocialPost.metadata.create_all(engine, tables=[SocialPost.__table__, SocialPostTarget.__table__])
    # llm_usage (the render quota reads it) as raw DDL: CI's SQLAlchemy cannot compile its UUID type for SQLite.
    columns = ", ".join(f"{c.name} INTEGER PRIMARY KEY" if c.primary_key else c.name for c in LLMUsage.__table__.columns)
    with engine.begin() as conn:
        conn.exec_driver_sql(f"CREATE TABLE {LLMUsage.__tablename__} ({columns})")
    factory = sessionmaker(bind=engine)
    session = factory()
    for ws_id, settings in ((WS, SOCIALS_ON), (WS_OTHER, SOCIALS_ON), (WS_OFF, {})):
        session.add(
            Workspace(
                id=ws_id, name=f"ws-{ws_id.hex[-2:]}", plan="basic", plan_limits={},
                settings=settings, onboarding={}, created_at=CREATED, updated_at=CREATED,
            )
        )
    session.commit()

    state = SimpleNamespace(
        session=session, factory=factory, ctx=_ctx(WS), role="owner", master="true",
        launched=[], health=[], booked=[],
    )
    monkeypatch.setattr(socials_settings, "read_system_setting", lambda category, key: state.master)
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
    monkeypatch.setattr(UsageTracker, "track_media", staticmethod(lambda **kwargs: state.booked.append(kwargs)))
    FakeWorkspaceClient.files = {}
    monkeypatch.setattr(report_charts, "WorkspaceClient", FakeWorkspaceClient)

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


def _tool(env, action, workspace_id=WS, **params):
    """An agent's call as exec_platform makes it (the agent keys minted), through the executor."""
    executor = PlatformActionExecutor(env.session, workspace_id)
    with patch.object(PlatformActionExecutor, "_full_autonomy", return_value=False), patch(
        "core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)
    ):
        return asyncio.run(executor.execute(action, {**AGENT, **params}))


def _template(env, *, name="Countdown", blocks=COMPOSITION, fmt="social_video", ws=WS):
    template_id = uuid.uuid4()
    env.session.execute(
        sa.text(
            "INSERT INTO document_templates (id, workspace_id, name, format, data_schema, blocks, is_active, version) "
            "VALUES (:id, :ws, :name, :fmt, '{}', :blocks, :active, 1)"
        ),
        {
            "id": template_id.hex, "ws": ws.hex, "name": name, "fmt": fmt,
            "blocks": json.dumps(blocks) if blocks is not None else None, "active": True,
        },
    )
    env.session.commit()
    return template_id


def _draft_fields(**overrides):
    fields = {
        "title": "Web Summit countdown",
        "brief": "Three weeks to Lisbon",
        "copy": {"base": "Three weeks to go.", "channels": {"linkedin": "Three weeks to Web Summit."}},
        "format": "video",
        "variables": {"headline": {"value": "Three weeks to go", "claim": True}},
        "sources": {"headline": dict(LAUNCH_SOURCE)},
    }
    fields.update(overrides)
    return fields


def _created(result):
    assert result["success"] is True, result
    return result["post"]


def _row(env, post_id):
    env.session.expire_all()
    return env.session.get(SocialPost, uuid.UUID(str(post_id)))


def _posts(env):
    env.session.expire_all()
    return env.session.query(SocialPost).count()


def _usage_row(workspace_id, seconds):
    return LLMUsage(
        workspace_id=workspace_id, model_id="hyperframes", provider=MEDIA_RENDER_PROVIDER, tier="direct",
        request_type=LANE_MEDIA, input_tokens=seconds, output_tokens=0, total_tokens=seconds,
        cache_read_tokens=0, cache_write_tokens=0, input_cost=0.0, output_cost=0.0, total_cost=0.0,
        is_byok=False, status="success", created_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )


def _run(job, env):
    """The render the background would run, against the mocked media-render."""
    renderer, store = Renderer(), FakeStore()

    async def go():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(renderer.handler), headers={"X-Internal-Token": TOKEN}
        ) as http:
            return await render.run_render(job, client=MediaRenderClient(http), store=store, session_factory=env.factory)

    return asyncio.run(go()), renderer, store


# ---------------------------------------------------------------------------
# AC1: a post the tool drafts is the post REST drafts
# ---------------------------------------------------------------------------


def test_a_post_the_tool_drafts_is_listed_as_the_post_rest_drafts(env):
    template_id = _template(env)
    rest = env.client.post("/api/socials/posts", json={**_draft_fields(), "template_id": str(template_id)})
    assert rest.status_code == 201, rest.text

    drafted = _created(_tool(env, "platform_create_social_post", **_draft_fields(), template="Countdown", render=False))

    listed = {p["id"]: p for p in env.client.get("/api/socials/posts").json()["posts"]}
    by_rest, by_tool = listed[rest.json()["id"]], listed[drafted["id"]]
    for field in ("status", "content_hash", "sources", "copy", "variables", "template_id", "format"):
        assert by_tool[field] == by_rest[field], field
    assert by_tool["status"] == "draft" and by_tool["sources"] == {"headline": LAUNCH_SOURCE}
    assert by_tool["content_hash"] == service.compute_content_hash(_row(env, drafted["id"]))

    # The acting agent is recorded; a person's own draft logs nothing (Wave 0 as it was).
    assert by_tool["created_by"] == ACTOR and by_rest["created_by"] == "member-1"
    (entry,) = by_tool["review_log"]
    assert (entry["action"], entry["by"], entry["comment"]) == ("draft", ACTOR, "Drafted by Social Media Director.")
    assert entry["agent"] == "Social Media Director"
    assert by_rest["review_log"] == []


def test_the_tool_takes_a_social_template_of_the_workspace_by_id_or_by_name(env):
    template_id = _template(env, name="Countdown")
    by_id = _created(_tool(env, "platform_create_social_post", title="By id", template=str(template_id), render=False))
    by_name = _created(_tool(env, "platform_create_social_post", title="By name", template="Countdown", render=False))
    assert by_id["template_id"] == by_name["template_id"] == str(template_id)
    # template_id, the name the other document tools give it, is taken the same way.
    alias = _created(_tool(env, "platform_create_social_post", title="Alias", template_id=str(template_id), render=False))
    assert alias["template_id"] == str(template_id)

    _template(env, name="Board pack", blocks=None, fmt="pdf")
    other = _template(env, name="Theirs", ws=WS_OTHER)
    before = _posts(env)
    for ref, reason in (
        ("Board pack", "'Board pack' is a pdf template: a post takes a social_video or social_image template."),
        (str(other), f"No template {str(other)!r} in this workspace"),
        ("Theirs", "No template 'Theirs' in this workspace"),
        ("No such template", "No template 'No such template' in this workspace"),
    ):
        result = _tool(env, "platform_create_social_post", title="Refused", template=ref, render=False)
        assert result["success"] is False and reason in result["error"], result
    assert _posts(env) == before


def test_the_tool_takes_the_shapes_the_socials_skills_write_and_stores_the_posts_own(env):
    """social-ops and social-video-director draft with copy per channel, bare variable
    values and a list of sources: the post keeps its own shapes, as REST would."""
    template_id = _template(env)
    rest = env.client.post(
        "/api/socials/posts",
        json={
            "title": "Countdown", "template_id": str(template_id),
            "copy": {"channels": {"linkedin": "Three weeks to Web Summit.", "x": "3 weeks."}},
            "variables": {"headline": {"value": "Three weeks to go", "claim": True}},
            "sources": {"headline": dict(LAUNCH_SOURCE)},
        },
    ).json()

    drafted = _created(_tool(
        env, "platform_create_social_post", title="Countdown", template_id=str(template_id), render=False,
        copy={"linkedin": "Three weeks to Web Summit.", "x": "3 weeks."},
        variables={"headline": "Three weeks to go"},
        sources=[{"claim": "headline", **LAUNCH_SOURCE}],
    ))

    for field in ("copy", "variables", "sources", "content_hash"):
        assert drafted[field] == rest[field], field
    # headline is a claim because the template says so, though the agent sent a bare value.
    assert drafted["variables"] == {"headline": {"value": "Three weeks to go", "claim": True}}

    unnamed = _tool(env, "platform_create_social_post", title="Unnamed", render=False, sources=[{"kind": "url", "ref": "https://a.b"}])
    assert unnamed == {"success": False, "error": "sources[0] needs claim: the name of the variable it backs."}


def test_a_figure_the_template_marks_a_claim_stays_a_claim_whatever_the_agent_says(env):
    template_id = _template(env)
    unclaimed = _tool(
        env, "platform_create_social_post", title="Sneaky", template=str(template_id), render=False,
        variables={"headline": {"value": "Fastest in Europe", "claim": False}},
    )
    post = _created(unclaimed)
    assert post["variables"]["headline"] == {"value": "Fastest in Europe", "claim": True}
    # So its approval needs a source (D7), and the reviewer is told which claim lacks one.
    assert _tool(env, "platform_submit_social_post", post_id=post["id"])["success"] is True
    approve = env.client.post(f"/api/socials/posts/{post['id']}/approve", json={"content_hash": _row(env, post["id"]).content_hash})
    assert approve.status_code == 422 and approve.json()["detail"]["claims"] == ["headline"]
    # An empty value claims nothing.
    empty = _created(_tool(
        env, "platform_create_social_post", title="Empty", template=str(template_id), render=False,
        variables={"headline": ""},
    ))
    assert empty["variables"]["headline"] == {"value": "", "claim": False}


def test_an_update_changes_only_what_it_sends(env):
    post = _created(_tool(
        env, "platform_create_social_post", render=False, title="Countdown",
        copy={"base": "Three weeks to go.", "channels": {"linkedin": "Three weeks to Web Summit.", "x": "3 weeks."}},
        variables={"headline": {"value": "Three weeks to go", "claim": True}, "eyebrow": {"value": "Lisbon", "claim": False}},
        sources={"headline": dict(LAUNCH_SOURCE)},
    ))

    fixed = _tool(
        env, "platform_update_social_post", post_id=post["id"],
        copy={"linkedin": "Two weeks to Web Summit.", "x": ""},
        variables={"eyebrow": None, "cta": "Book a demo"},
        sources=[{"claim": "cta", "kind": "url", "ref": "https://example.com/demo"}],
    )

    assert fixed["success"] is True, fixed
    row = _row(env, post["id"])
    assert row.copy == {"base": "Three weeks to go.", "channels": {"linkedin": "Two weeks to Web Summit."}}
    assert row.variables == {
        "headline": {"value": "Three weeks to go", "claim": True},
        "cta": {"value": "Book a demo", "claim": False},
    }
    assert row.sources == {"headline": LAUNCH_SOURCE, "cta": {"kind": "url", "ref": "https://example.com/demo"}}
    edit = row.review_log[-1]
    assert edit["action"] == "edit" and edit["fields"] == ["copy", "variables", "sources"]


def test_a_source_that_does_not_resolve_in_the_workspace_saves_nothing(env):
    missing = str(uuid.uuid4())
    result = _tool(
        env, "platform_create_social_post", render=False,
        **_draft_fields(sources={"headline": {"kind": "report", "ref": missing, "as_of": None}}),
    )
    assert result["success"] is False
    assert set(result["unresolved"]) == {"headline"}
    assert _posts(env) == 0


# ---------------------------------------------------------------------------
# AC2: an edit through the tool voids the approval, through the Wave 0 service
# ---------------------------------------------------------------------------


def test_editing_an_approved_post_through_the_tool_voids_its_approval_through_the_service(env, monkeypatch):
    post = _created(_tool(env, "platform_create_social_post", **_draft_fields(), render=False))
    submitted = _tool(env, "platform_submit_social_post", post_id=post["id"])
    assert submitted["success"] is True, submitted
    # A person approves in the Socials tab (D6).
    approved = env.client.post(
        f"/api/socials/posts/{post['id']}/approve", json={"content_hash": submitted["post"]["content_hash"]}
    )
    assert approved.status_code == 200 and approved.json()["status"] == "approved", approved.text

    calls = []
    real_update = service.update_post

    def spy(post_row, actor, changes, **kwargs):
        calls.append((actor, dict(changes), kwargs))
        return real_update(post_row, actor, changes, **kwargs)

    monkeypatch.setattr(service, "update_post", spy)
    edited = _tool(env, "platform_update_social_post", post_id=post["id"], copy={"base": "Two weeks to go."})

    assert edited["success"] is True, edited
    # The tool merged the change into the post's copy (its LinkedIn text kept) and handed it to the service.
    merged = {"base": "Two weeks to go.", "channels": {"linkedin": "Three weeks to Web Summit."}}
    assert calls == [(ACTOR, {"copy": merged}, {"agent": "Social Media Director"})]
    row = _row(env, post["id"])
    assert row.status == "needs_approval"
    assert row.approved_hash == approved.json()["approved_hash"] != row.content_hash
    assert [e["action"] for e in row.review_log] == ["draft", "submit", "approve", "edit", "approval_voided"]
    edit, voided = row.review_log[-2:]
    assert (edit["by"], edit["fields"], edit["comment"]) == (ACTOR, ["copy"], "Edited by Social Media Director: copy.")
    assert voided["by"] == ACTOR
    # The platform would refuse to publish it now (the approval no longer covers it).
    with pytest.raises(service.NotPublishable):
        service.assert_publishable(row)


def test_an_edit_that_changes_nothing_logs_nothing_and_an_empty_one_is_refused(env):
    post = _created(_tool(env, "platform_create_social_post", **_draft_fields(), render=False))
    same = _tool(env, "platform_update_social_post", post_id=post["id"], title=post["title"])
    assert same["success"] is True
    assert [e["action"] for e in _row(env, post["id"]).review_log] == ["draft"]

    empty = _tool(env, "platform_update_social_post", post_id=post["id"])
    assert empty == {"success": False, "error": "Nothing to change: send the fields to change, or render true."}
    elsewhere = _tool(env, "platform_update_social_post", workspace_id=WS_OTHER, post_id=post["id"], title="Theirs now")
    assert elsewhere["success"] is False and "Post not found in this workspace" in elsewhere["error"]
    assert _row(env, post["id"]).title == post["title"]


# ---------------------------------------------------------------------------
# AC3: list, submit, and nothing that approves, schedules or publishes
# ---------------------------------------------------------------------------


def test_list_filters_by_status_and_returns_only_the_callers_workspace(env):
    drafts = [_created(_tool(env, "platform_create_social_post", title=f"Draft {n}", render=False)) for n in (1, 2)]
    waiting = _created(_tool(env, "platform_create_social_post", title="Waiting", render=False))
    assert _tool(env, "platform_submit_social_post", post_id=waiting["id"])["success"] is True
    theirs = _created(_tool(env, "platform_create_social_post", workspace_id=WS_OTHER, title="Theirs", render=False))
    # SQLite's CURRENT_TIMESTAMP has whole seconds: give each post its own moment, oldest first.
    posts = SocialPost.__table__
    for minute, post in enumerate((drafts[0], drafts[1], waiting, theirs)):
        env.session.execute(
            sa.update(posts).where(posts.c.id == uuid.UUID(post["id"])).values(created_at=datetime(2026, 9, 2, 9, minute))
        )
    env.session.commit()

    everything = _tool(env, "platform_list_social_posts")
    assert everything["success"] is True
    assert {p["id"] for p in everything["posts"]} == {drafts[0]["id"], drafts[1]["id"], waiting["id"]}
    assert theirs["id"] not in {p["id"] for p in everything["posts"]}

    queued = _tool(env, "platform_list_social_posts", status=["needs_approval"])
    assert [p["id"] for p in queued["posts"]] == [waiting["id"]] and queued["count"] == 1
    both = _tool(env, "platform_list_social_posts", status="draft,needs_approval", limit=2)
    assert both["count"] == 2 and both["limit"] == 2
    assert [p["id"] for p in both["posts"]] == [waiting["id"], drafts[1]["id"]]  # newest first

    unknown = _tool(env, "platform_list_social_posts", status=["pending"])
    assert unknown["success"] is False and "Unknown status pending" in unknown["error"]
    too_many = _tool(env, "platform_list_social_posts", limit=actions_socials.LIST_MAX_LIMIT + 1)
    assert too_many["success"] is False and "limit must be" in too_many["error"]
    assert _tool(env, "platform_list_social_posts", workspace_id=WS_OTHER)["posts"][0]["id"] == theirs["id"]


def test_submit_moves_a_draft_to_needs_approval_and_no_further(env):
    post = _created(_tool(env, "platform_create_social_post", **_draft_fields(), render=False))
    submitted = _tool(env, "platform_submit_social_post", post_id=post["id"])
    assert submitted["success"] is True and submitted["post"]["status"] == "needs_approval"
    assert "a person approves it" in submitted["message"]
    last = submitted["post"]["review_log"][-1]
    assert (last["action"], last["by"]) == ("submit", ACTOR)

    again = _tool(env, "platform_submit_social_post", post_id=post["id"])
    assert again == {"success": False, "error": "cannot submit a post that is needs approval"}
    assert _tool(env, "platform_submit_social_post", post_id=post["id"], approve=True) == {
        "success": False,
        "error": "platform_submit_social_post takes post_id and note; not approve. Nothing was sent.",
    }
    assert _tool(env, "platform_get_social_post", post_id=post["id"])["post"]["status"] == "needs_approval"
    assert _tool(env, "platform_get_social_post", post_id="not-a-post")["success"] is False


def test_a_submit_note_tells_the_reviewer_what_to_look_at(env):
    post = _created(_tool(env, "platform_create_social_post", **_draft_fields(), render=False))
    note = "The caption on shot 3 sits over the bright sky; the second round of fixes did not move it."
    submitted = _tool(env, "platform_submit_social_post", post_id=post["id"], note=note)
    assert submitted["success"] is True
    last = _row(env, post["id"]).review_log[-1]
    assert (last["action"], last["by"], last["comment"]) == ("submit", ACTOR, note)


def _parameter_names(schema):
    """Every parameter a schema declares: nested objects and array items included."""
    names = []
    if isinstance(schema, dict):
        for name, sub in (schema.get("properties") or {}).items():
            names.append(name)
            names.extend(_parameter_names(sub))
        names.extend(_parameter_names(schema.get("items")))
    return names


def _names_in(source):
    tree = ast.parse(textwrap.dedent(source))
    return {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {
        n.id for n in ast.walk(tree) if isinstance(n, ast.Name)
    }


def test_no_registered_tool_approves_schedules_or_publishes_a_post():
    registry = get_action_registry()
    socials = {a.name: a for a in registry.get_all() if a.category == "socials"}
    assert set(socials) == set(TOOLS)
    assert [a.name for a in registry.get_all() if "social" in a.name and FORBIDDEN_WORD.search(a.name)] == []
    for action in socials.values():
        assert not FORBIDDEN_WORD.search(action.name)
        assert [n for n in _parameter_names(action.parameters) if FORBIDDEN_WORD.search(n)] == [], action.name
        assert action.requires_confirmation is False and action.admin_only is False
    assert {n: socials[n].permission_level for n in TOOLS} == {
        "platform_create_social_post": "write",
        "platform_update_social_post": "write",
        "platform_submit_social_post": "write",
        "platform_get_social_post": "read",
        "platform_list_social_posts": "read",
    }
    # Every tool is routed to its handler here, and nothing a tool runs reaches a review, a
    # schedule or the way out: neither the handlers nor the flows they share with the routes.
    handlers = PlatformActionExecutor(None, WS)._handlers
    assert {handlers[n].__module__ for n in TOOLS} == {handlers_socials.__name__}
    reached = _names_in(inspect.getsource(handlers_socials))
    for flow in (socials_api.create_post, socials_api.edit_post, socials_api.submit_post, socials_api.render_post):
        reached |= _names_in(inspect.getsource(flow))
    assert reached & NOT_FOR_AGENTS == set()


def test_a_field_the_post_does_not_take_is_refused_and_nothing_is_saved(env):
    for field in ("approve", "scheduled_for", "publish_now", "status"):
        result = _tool(env, "platform_create_social_post", title="Sneaky", render=False, **{field: True})
        assert result["success"] is False, result
        assert f"{field}: Extra inputs are not permitted" in result["error"]
    assert _posts(env) == 0


def test_the_schemas_speak_the_lifecycles_own_vocabulary():
    assert actions_socials.POST_FORMATS == list(SOCIAL_POST_FORMATS)
    assert actions_socials.POST_STATUSES == list(SOCIAL_POST_STATUSES)
    assert actions_socials.CHART_KINDS == list(CHART_KINDS)
    assert actions_socials.CHART_PARTS == list(PARTS)
    create = get_action_registry().get("platform_create_social_post")
    fields = set(create.parameters["properties"]) - {"template", "chart_report", "render"}
    # The REST body's own fields, template_id taken as ``template`` (an id or a name).
    rest = {f.alias or name for name, f in socials_api.CreateSocialPostRequest.model_fields.items()} - {"template_id"}
    assert fields == rest


# ---------------------------------------------------------------------------
# AC4: Socials off → every tool refuses, and nothing is read or written
# ---------------------------------------------------------------------------


def _calls(post_id):
    return {
        "platform_create_social_post": {"title": "Off", "render": False},
        "platform_update_social_post": {"post_id": post_id, "title": "Off"},
        "platform_submit_social_post": {"post_id": post_id},
        "platform_get_social_post": {"post_id": post_id},
        "platform_list_social_posts": {},
    }


@pytest.mark.parametrize(
    ("switch", "message"),
    [("platform", socials_settings.SOCIALS_OFF_FOR_PLATFORM), ("workspace", socials_settings.SOCIALS_OFF_FOR_WORKSPACE)],
)
def test_every_socials_tool_refuses_while_socials_is_off(env, switch, message):
    post = _created(_tool(env, "platform_create_social_post", title="Before", render=False))
    before = _row(env, post["id"]).to_dict()
    workspace = WS
    if switch == "platform":
        env.master = "false"
    else:
        workspace = WS_OFF
    for action, params in _calls(post["id"]).items():
        result = _tool(env, action, workspace_id=workspace, **params)
        assert result == {"success": False, "error": message, "socials_off": True}, action
    assert _posts(env) == 1 and _row(env, post["id"]).to_dict() == before


def test_a_master_switch_that_cannot_be_read_is_off_for_the_tools(env, monkeypatch):
    def unreadable(category, key):
        raise RuntimeError("the pool is exhausted")

    monkeypatch.setattr(socials_settings, "read_system_setting", unreadable)
    result = _tool(env, "platform_create_social_post", title="Off", render=False)
    assert result["error"] == socials_settings.SOCIALS_OFF_FOR_PLATFORM
    assert _posts(env) == 0


# ---------------------------------------------------------------------------
# AC5: render is the US-104 render, within the plan's quota
# ---------------------------------------------------------------------------


def test_a_draft_renders_by_default_through_media_render_and_waits_for_approval(env):
    template_id = _template(env)
    result = _tool(env, "platform_create_social_post", **_draft_fields(), template=str(template_id))

    assert result["success"] is True and result["render"] == {"started": True}, result
    assert result["post"]["status"] == "rendering" and "waits for approval" in result["message"]
    assert env.health == [True]
    (job,) = env.launched
    assert job.actor == ACTOR and str(job.post_id) == result["post"]["id"] and job.workspace_id == WS
    assert [e["action"] for e in result["post"]["review_log"]] == ["draft", "render"]

    done, renderer, store = _run(job, env)
    assert done is True
    (bundle,) = renderer.bundles
    assert bundle["variables"]["headline"] == "Three weeks to go"
    row = _row(env, result["post"]["id"])
    assert row.status == "needs_approval"
    (record,) = row.media["9:16"]
    assert record["deliverable_id"] == "d-video-1" and record["duration"] == 30.0
    assert list(store.objects) == [f"social-media/{WS}/{row.id}/video-9x16.mp4"]
    (booking,) = env.booked
    assert booking["units"] == 30.0 and booking["usd"] == 0.0


def test_past_the_render_quota_the_draft_is_saved_and_the_render_refused_before_media_render(env):
    template_id = _template(env)
    env.session.add(_usage_row(WS, 600))  # Basic: 10 minutes, all used
    env.session.commit()

    result = _tool(env, "platform_create_social_post", **_draft_fields(), template=str(template_id), render=True)

    assert result["success"] is True and result["render"]["started"] is False, result
    assert "used 10.0 of its 10 render minutes this month on the Basic plan" in result["render"]["error"]
    assert result["message"].startswith("The post was saved but not rendered: ")
    assert env.health == [] and env.launched == []
    row = _row(env, result["post"]["id"])
    assert row.status == "draft" and [e["action"] for e in row.review_log] == ["draft"]


def test_a_post_without_a_template_is_saved_and_says_why_it_was_not_rendered(env):
    result = _tool(env, "platform_create_social_post", title="Text only", copy={"base": "Hello."})
    assert result["success"] is True and result["post"]["status"] == "draft"
    assert result["render"] == {
        "started": False, "error": "this post has no template to render: choose a social template first",
    }
    assert env.launched == []


def test_update_with_render_renders_a_failed_post_again(env):
    template_id = _template(env)
    post = _created(_tool(env, "platform_create_social_post", **_draft_fields(), template=str(template_id), render=False))
    row = _row(env, post["id"])
    service.start_render(row, ACTOR)
    service.fail_render(row, ACTOR, "The composition failed its check.")
    env.session.commit()

    fixed = _tool(
        env, "platform_update_social_post", post_id=post["id"],
        variables={"headline": {"value": "Two weeks to go", "claim": True}}, render=True,
    )

    assert fixed["success"] is True and fixed["render"] == {"started": True}, fixed
    assert fixed["post"]["status"] == "rendering"
    (job,) = env.launched
    assert job.bundle["variables"]["headline"] == "Two weeks to go"
    actions = [e["action"] for e in _row(env, post["id"]).review_log]
    assert actions == ["draft", "render", "render_failed", "edit", "render"]
    assert _tool(env, "platform_update_social_post", post_id=post["id"], render="yes") == {
        "success": False, "error": "render must be true or false.",
    }


# ---------------------------------------------------------------------------
# S1.7: a chart template's rows come from a report, never typed
# ---------------------------------------------------------------------------


def _infographic():
    (starter,) = [s for s in social_starters("social_image") if s["slug"] == "infographic"]
    return starter


def _report(env, ws=WS) -> str:
    ident = str(uuid.uuid4())
    path = f"reports/scout/2026-09-21_{ident[:6]}_september.md"
    env.session.execute(
        AGENT_REPORTS.insert().values(
            id=ident, workspace_id=str(ws), report_type="summary", title=REPORT_TITLE, summary=None,
            file_path=path, metrics={}, created_at=MADE, deleted_at=None,
        )
    )
    env.session.commit()
    FakeWorkspaceClient.files[(str(ws), path)] = CHANNELS
    return ident


def test_a_chart_template_is_filled_from_a_report_as_the_composer_fills_it_never_typed(env):
    starter = _infographic()
    template_id = _template(env, name=starter["name"], blocks=starter["blocks"], fmt="social_image")
    report_id = _report(env)
    sample = starter["sample_data"]
    words = {name: {"value": sample[name], "claim": False} for name in ("headline", "eyebrow", "cta")}

    typed = _tool(
        env, "platform_create_social_post", title="Typed", template=starter["name"], render=False,
        variables={**words, "row_1_value": {"value": "99,999", "claim": True}},
    )
    assert typed["success"] is False and typed["error"].startswith("row_1_value: a chart's rows")
    assert _posts(env) == 0

    drafted = _created(_tool(
        env, "platform_create_social_post", title="September reach", format="infographic",
        template=starter["name"], variables=words, render=False,
        chart_report={"report_id": report_id, "column": "Reach"},
    ))

    binding = env.client.get(
        f"/api/socials/sources/reports/{report_id}/chart", params={"template_id": str(template_id), "column": "Reach"}
    ).json()
    assert drafted["variables"] == {**words, **binding["variables"]}
    assert drafted["sources"] == binding["sources"]
    assert [(drafted["variables"][f"row_{n}_label"]["value"], drafted["variables"][f"row_{n}_value"]["value"])
            for n in range(1, 6)] == REACH

    # Sending the rows back as they are is no typing; changing one is.
    echoed = _tool(env, "platform_update_social_post", post_id=drafted["id"], variables=drafted["variables"], title="Reach")
    assert echoed["success"] is True, echoed
    edited = dict(drafted["variables"], row_2_value={"value": "30,000", "claim": True})
    refused = _tool(env, "platform_update_social_post", post_id=drafted["id"], variables=edited)
    assert refused["success"] is False and "row_2_value" in refused["error"]
    assert _row(env, drafted["id"]).variables["row_2_value"]["value"] == "21,480"

    unknown = _tool(env, "platform_update_social_post", post_id=drafted["id"], chart_report={"report_id": str(uuid.uuid4())})
    assert unknown["success"] is False and "report" in unknown["error"].lower()
