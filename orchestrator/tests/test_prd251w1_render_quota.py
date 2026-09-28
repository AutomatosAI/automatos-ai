"""P251W1-RVW-3 — the render quota holds when renders run at once.

Wave 1's quota read only the seconds finished renders had booked, so renders
asked for together all read the same total and all started. A render now holds
its composition's declared duration (a ``reserved`` row in ``llm_usage``) from
before anything reaches media-render until it ends, and the check and the hold
are one step under one lock per workspace (``core/media_render_quota.py``). Pins:

* 20 renders of a 60-second template asked for at once on Basic (10 minutes,
  nothing used), through the real route: exactly 10 start (202) and 10 are
  refused (429) before any request reaches media-render; the ten submit, book
  their seconds and give their holds back, and the month is then used up. The
  same with 20 concurrent ``generate_document`` renders of a social template,
  each held in flight until all twenty are decided;
* the refusal still names the plan and the minutes used, and adds the minutes
  renders in progress hold; GET /usage shows them;
* a hold is given back however the render ends: done (after its booking),
  failed, timed out, an unexpected error, a post that moved on, or a refusal
  after the hold; one nothing gave back stops counting at its render's
  deadline, and the boot reaper deletes it;
* the rendered seconds are in llm_usage when run_render / generate return, with
  no room in the connection pool and no ``best_effort.drain()``;
* a render holds its declared duration, rounded up and at most the longest
  media-render renders; a still holds nothing; no quota, no hold;
* @integration, on the CI Postgres: the quota lock holds across two sessions,
  each in its own thread with its own SessionLocal.

The harness is a file-backed SQLite with a connection per session, so renders
running at once never share one; media-render is mocked at the HTTP layer and
the real client runs; the usage tracker writes into this database.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import sys
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import httpx  # noqa: E402
import sqlalchemy as sa  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import NullPool  # noqa: E402

from tests.test_prd251w1_render_lifecycle import (  # noqa: E402  (the lifecycle's harness pieces)
    MP4, RENDER_URL, TOKEN, Deliverables, FakeStore, _ctx, _set_config, _sqlite_copy,
)
import api.socials as socials_api  # noqa: E402
import core.auth.workspace_permission as permission_mod  # noqa: E402
import core.best_effort as best_effort  # noqa: E402
import core.boot.reaper as reaper  # noqa: E402
import core.database.database as database_mod  # noqa: E402
import core.media_render_client as media_render_client  # noqa: E402
import core.media_render_quota as render_quota  # noqa: E402
import modules.documents.generation_service as generation_service  # noqa: E402
import modules.socials.media_store as media_store  # noqa: E402
import modules.socials.render as render  # noqa: E402
import modules.socials.service as service  # noqa: E402
import modules.socials.settings as socials_settings  # noqa: E402
import services.deliverable_service as deliverable_service  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.llm.providers import MEDIA_RENDER_PROVIDER  # noqa: E402
from core.media_render_client import MediaRenderClient  # noqa: E402
from core.models.core import DocumentTemplate, LLMUsage  # noqa: E402
from core.models.socials import SocialPost, SocialPostTarget  # noqa: E402
from core.models.workspaces import Workspace  # noqa: E402
from modules.documents.generation_service import DocumentGenerationService  # noqa: E402

# Hex with letters, so SQLite keeps every UUID column as text.
WS = uuid.UUID("00000000-0000-0000-0000-0000000000b1")
TEMPLATE_ID = uuid.UUID("00000000-0000-0000-0000-0000000000bc")
CREATED = datetime(2026, 9, 1, 9, 0)
RENDERS = 20
# Basic: 10 minutes; a 60-second composition: ten fit.
STARTED = 10
HTML_60 = (
    '<!doctype html><html><head></head><body><div id="root" data-composition-id="main" data-width="1080" '
    'data-height="1920" data-duration="60"><h1>{{ headline }}</h1></div></body></html>'
)
BLOCKS_60 = {
    "html": HTML_60,
    "css": "h1 { color: var(--brand-text); }",
    "variables_schema": {"headline": {"type": "text"}},
    "sizes": ["1080x1920"],
}
OUTPUT_60 = {"name": "render.mp4", "aspect": "9:16", "width": 1080, "height": 1920, "bytes": len(MP4), "duration": 60.0}


class MediaRender:
    """media-render over httpx.MockTransport, recording every request that reaches
    it. ``hold`` keeps each job rendering until ``released``: renders in flight."""

    def __init__(self):
        self.seen = []
        self.hold = False
        self.released = False
        self.health_status = 200
        self.reject = None

    @property
    def submits(self) -> int:
        return self.seen.count(("POST", "/render"))

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.seen.append((request.method, path))
        if path == "/health":
            return httpx.Response(self.health_status, json={"status": "healthy", "error": "down", "message": "down"})
        if request.method == "POST" and path == "/render":
            if self.reject is not None:
                return self.reject
            return httpx.Response(202, json={"id": f"{self.submits:032x}", "status": "rendering", "outputs": [], "report": {}})
        if path.endswith("/output/render.mp4"):
            return httpx.Response(200, content=MP4)
        if request.method == "GET" and path.startswith("/render/"):
            done = not self.hold or self.released
            body = {
                "id": path.rsplit("/", 1)[-1], "status": "done" if done else "rendering",
                "outputs": [OUTPUT_60] if done else [], "report": {"check": {"ok": True, "errors": 0}},
            }
            return httpx.Response(200, json=body)
        return httpx.Response(404, json={"error": "not_found", "message": f"no route {path}"})


def _database(tmp_path):
    """A file-backed SQLite, a connection per session: the social tables, SQLite
    copies of workspaces and document_templates, and an untyped llm_usage."""
    engine = sa.create_engine(
        f"sqlite:///{tmp_path / 'quota.db'}", connect_args={"check_same_thread": False, "timeout": 30}, poolclass=NullPool
    )
    with engine.connect() as conn:
        conn.exec_driver_sql("PRAGMA journal_mode=WAL")
    copies = sa.MetaData()
    _sqlite_copy(Workspace.__table__, copies)
    _sqlite_copy(DocumentTemplate.__table__, copies)
    copies.create_all(engine)
    SocialPost.metadata.create_all(engine, tables=[SocialPost.__table__, SocialPostTarget.__table__])
    # SQLAlchemy 2.0.23 cannot compile the Postgres UUID type for SQLite: llm_usage as raw DDL.
    columns = ", ".join(f"{c.name} INTEGER PRIMARY KEY" if c.primary_key else c.name for c in LLMUsage.__table__.columns)
    with engine.begin() as conn:
        conn.exec_driver_sql(f"CREATE TABLE {LLMUsage.__tablename__} ({columns})")
    return engine


@pytest.fixture
def env(tmp_path, monkeypatch):
    engine = _database(tmp_path)
    factory = sessionmaker(bind=engine)
    with factory() as db:
        db.add(
            Workspace(
                id=WS, name="Harbourline", plan="basic", plan_limits={},
                settings={"socials": {"enabled": True}}, onboarding={}, created_at=CREATED, updated_at=CREATED,
            )
        )
        db.commit()
    renderer = MediaRender()
    state = SimpleNamespace(engine=engine, factory=factory, renderer=renderer, launched=[])
    monkeypatch.setattr(socials_settings, "read_system_setting", lambda category, key: "true")
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: "owner")
    monkeypatch.setattr(socials_api, "_launch_render", lambda job: state.launched.append(job))
    monkeypatch.setattr(render_quota, "_PROCESS_LOCKS", {})
    # The real renderer check, and the real client, against the mocked media-render.
    monkeypatch.setattr(media_store, "is_storage_configured", lambda: True)
    monkeypatch.setattr(
        media_render_client, "_get_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(renderer.handler))
    )
    monkeypatch.setattr(database_mod, "SessionLocal", factory)  # the usage tracker books into this database
    Deliverables.calls = []
    monkeypatch.setattr(deliverable_service, "DeliverableService", Deliverables)
    _set_config(
        monkeypatch, AUTH_EDITION="saas", SOCIALS_RENDER_URL=RENDER_URL, SOCIALS_RENDER_TOKEN=TOKEN,
        SOCIALS_RENDER_POLL_SECONDS=0, SOCIALS_RENDER_MAX_WAIT_SECONDS=60, SOCIALS_RENDER_MAX_DURATION_SECONDS=180,
    )

    def session():
        db = factory()
        try:
            yield db
        finally:
            db.close()

    app = FastAPI()
    app.include_router(socials_api.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: _ctx(WS)
    app.dependency_overrides[get_db] = session
    state.app = app
    try:
        yield state
    finally:
        engine.dispose()


def _template(env, blocks=BLOCKS_60, fmt="social_video"):
    template_id = uuid.uuid4()
    with env.factory() as db:
        db.execute(
            sa.text(
                "INSERT INTO document_templates (id, workspace_id, name, format, data_schema, blocks) "
                "VALUES (:id, :ws, :name, :fmt, '{}', :blocks)"
            ),
            {"id": template_id.hex, "ws": WS.hex, "name": f"tpl-{template_id.hex[:6]}", "fmt": fmt, "blocks": json.dumps(blocks)},
        )
        db.commit()
    return template_id


def _drafts(env, count):
    """``count`` drafts of the 60-second template, ready to render."""
    template_id = _template(env)
    with env.factory() as db:
        posts = [
            service.create_draft(
                db, workspace_id=WS, created_by="member-1", title=f"Countdown {i:02d}", format="video",
                template_id=template_id, variables={"headline": {"value": "Three weeks to go", "claim": False}},
            )
            for i in range(count)
        ]
        db.commit()
        return [str(post.id) for post in posts]


async def _ask(env, post_ids):
    """POST /render for every post at once, through the real route."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=env.app), base_url="http://socials.test") as client:
        return await asyncio.gather(*(client.post(f"/api/socials/posts/{post_id}/render") for post_id in post_ids))


async def _usage(env):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=env.app), base_url="http://socials.test") as client:
        return (await client.get("/api/socials/usage")).json()["render_minutes"]


async def _render_jobs(env, jobs):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(env.renderer.handler), headers={"X-Internal-Token": TOKEN}
    ) as http:
        client, store = MediaRenderClient(http), FakeStore()
        return await asyncio.gather(
            *(render.run_render(job, client=client, store=store, session_factory=env.factory) for job in jobs)
        )


def _render_rows(env):
    """The renderer's rows in llm_usage as they are now: read at once, never after a best_effort.drain()."""
    with env.factory() as db:
        rows = db.query(LLMUsage).filter(LLMUsage.provider == MEDIA_RENDER_PROVIDER).order_by(LLMUsage.id).all()
        return [(row.status, int(row.input_tokens), row.execution_id) for row in rows]


def _held(env):
    return [seconds for status, seconds, _ in _render_rows(env) if status == render_quota.RESERVED_STATUS]


def _booked(env):
    return [seconds for status, seconds, _ in _render_rows(env) if status != render_quota.RESERVED_STATUS]


def _resumes():
    _, end = render_quota.month_window_utc()
    return f"{end.day} {end:%B}"


# ---------------------------------------------------------------------------
# Renders asked for at once: exactly the quota's worth start
# ---------------------------------------------------------------------------


def test_twenty_renders_asked_at_once_on_basic_start_ten_and_refuse_ten_before_media_render(env):
    post_ids = _drafts(env, RENDERS)

    responses = asyncio.run(_ask(env, post_ids))

    started = [r for r in responses if r.status_code == 202]
    refused = [r for r in responses if r.status_code == 429]
    assert (len(started), len(refused)) == (STARTED, RENDERS - STARTED), [r.status_code for r in responses]
    for resp in refused:
        assert resp.json()["detail"] == (
            "This workspace has used 0.0 of its 10 render minutes this month on the Basic plan, and renders in "
            f"progress hold 10.0 more until they finish. Rendering resumes on {_resumes()}, or sooner if they use less."
        )
    # Only the renders that started reached media-render (its health check); none submitted yet.
    assert env.renderer.seen == [("GET", "/health")] * STARTED
    assert sorted(str(job.post_id) for job in env.launched) == sorted(r.json()["id"] for r in started)
    assert _held(env) == [60] * STARTED and _booked(env) == []

    # The ten render: ten submits; each books its 60 seconds, then gives its hold back.
    assert asyncio.run(_render_jobs(env, env.launched)) == [True] * STARTED
    assert env.renderer.submits == STARTED
    assert _held(env) == [] and _booked(env) == [60] * STARTED
    # The month's ten minutes are all booked now: the next render is refused, as before.
    (resp,) = asyncio.run(_ask(env, _drafts(env, 1)))
    assert resp.status_code == 429
    assert resp.json()["detail"] == (
        f"This workspace has used 10.0 of its 10 render minutes this month on the Basic plan. Rendering resumes on {_resumes()}."
    )


def _documents(monkeypatch, tmp_path):
    monkeypatch.setattr(generation_service, "GENERATED_DIR", str(tmp_path / "generated"))
    monkeypatch.setattr(generation_service, "is_storage_configured", lambda: False)
    monkeypatch.setattr(generation_service, "brand_kit_for_media_render", lambda kit: dict(kit))


async def _generate(env, http, index):
    """generate_document on the 60-second social template, as the tool and the route call it."""
    template = SimpleNamespace(id=TEMPLATE_ID, name="Countdown", format="social_video", blocks=BLOCKS_60)
    documents = DocumentGenerationService(env.factory(), WS)
    documents.template_service = SimpleNamespace(get_template=lambda template_id, workspace_id: template)
    documents._render_client = lambda: MediaRenderClient(http)
    try:
        return await documents.generate(
            title=f"Countdown {index:02d}", format="social_video", data={"headline": "Three weeks to go"},
            workspace_id=WS, template_id=TEMPLATE_ID,
        )
    finally:
        documents.db.close()


def test_twenty_documents_rendered_at_once_on_basic_render_ten_and_refuse_ten(env, monkeypatch, tmp_path):
    _documents(monkeypatch, tmp_path)
    _set_config(monkeypatch, SOCIALS_RENDER_POLL_SECONDS=0.01)
    env.renderer.hold = True  # every render stays in flight until all twenty are decided
    refused = []

    async def one(http, index):
        try:
            await _generate(env, http, index)
            return "rendered"
        except render_quota.RenderQuotaExceeded as exc:
            refused.append(str(exc))
            return "refused"

    async def release_once_all_are_decided():
        while env.renderer.submits + len(refused) < RENDERS:
            await asyncio.sleep(0.01)
        env.renderer.released = True

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(env.renderer.handler)) as http:
            outcomes = await asyncio.gather(*(one(http, i) for i in range(RENDERS)), release_once_all_are_decided())
        return outcomes[:-1]

    outcomes = asyncio.run(scenario())

    assert sorted(outcomes) == ["refused"] * (RENDERS - STARTED) + ["rendered"] * STARTED
    assert env.renderer.submits == STARTED, "a refused render never reaches media-render"
    assert all("and renders in progress hold 10.0 more until they finish" in message for message in refused)
    assert _held(env) == [] and _booked(env) == [60] * STARTED


# ---------------------------------------------------------------------------
# The booking is written before the render returns (no best_effort.drain())
# ---------------------------------------------------------------------------


def test_the_rendered_seconds_are_in_llm_usage_when_run_render_returns_with_no_pool_room(env, monkeypatch):
    """A booking handed to the best-effort threads waits BEST_EFFORT_POOL_WAIT_S for
    pool room, then is dropped: the render's own is written inline, off the loop."""
    monkeypatch.setattr(best_effort, "_pool_has_room", lambda: False)
    (resp,) = asyncio.run(_ask(env, _drafts(env, 1)))
    assert resp.status_code == 202

    assert asyncio.run(_render_jobs(env, env.launched)) == [True]

    assert _render_rows(env) == [("success", 60, f"social_post:{resp.json()['id']}")]


def test_a_documents_rendered_seconds_are_in_llm_usage_when_generate_returns_with_no_pool_room(env, monkeypatch, tmp_path):
    _documents(monkeypatch, tmp_path)
    monkeypatch.setattr(best_effort, "_pool_has_room", lambda: False)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(env.renderer.handler)) as http:
            return await _generate(env, http, 1)

    result = asyncio.run(scenario())

    assert result.format == "mp4" and Path(result.path).read_bytes() == MP4
    assert _render_rows(env) == [("success", 60, f"document_template:{TEMPLATE_ID}")]


# ---------------------------------------------------------------------------
# However a render ends, its hold is given back
# ---------------------------------------------------------------------------


def _check_refused(env, monkeypatch):
    env.renderer.reject = httpx.Response(
        422, json={"error": "check_failed", "message": "1 error", "findings": [], "report": {"check": {"ok": False}}}
    )


def _times_out(env, monkeypatch):
    env.renderer.hold = True
    _set_config(monkeypatch, SOCIALS_RENDER_MAX_WAIT_SECONDS=1, SOCIALS_RENDER_POLL_SECONDS=0.01)


def _moved_on(env, monkeypatch):
    (job,) = env.launched
    with env.factory() as db:
        post = service.get_post(db, WS, job.post_id)
        service.fail_render(post, "orphaned_on_restart", "The render was lost when the server restarted.")
        db.commit()


@pytest.mark.parametrize("arrange", [_check_refused, _times_out, _moved_on], ids=["failed", "timed_out", "moved_on"])
def test_a_render_that_does_not_finish_gives_its_hold_back_and_books_nothing(env, monkeypatch, arrange):
    (resp,) = asyncio.run(_ask(env, _drafts(env, 1)))
    assert resp.status_code == 202 and _held(env) == [60]
    arrange(env, monkeypatch)

    assert asyncio.run(_render_jobs(env, env.launched)) == [False]

    assert _render_rows(env) == []


def test_a_render_that_breaks_unexpectedly_gives_its_hold_back(env, monkeypatch):
    (resp,) = asyncio.run(_ask(env, _drafts(env, 1)))
    assert resp.status_code == 202 and _held(env) == [60]

    def broken(*args, **kwargs):
        raise RuntimeError("the registry fell over")

    monkeypatch.setattr(render, "_register", broken)
    with pytest.raises(RuntimeError):
        asyncio.run(_render_jobs(env, env.launched))

    assert _render_rows(env) == []


def test_a_render_refused_after_its_hold_gives_it_back(env):
    """The quota held the seconds, then the renderer was not there (503): nothing is left held."""
    env.renderer.health_status = 503

    (resp,) = asyncio.run(_ask(env, _drafts(env, 1)))

    assert resp.status_code == 503
    assert env.launched == [] and _render_rows(env) == []


def test_a_document_render_that_fails_gives_its_hold_back_and_books_nothing(env, monkeypatch, tmp_path):
    _documents(monkeypatch, tmp_path)
    _check_refused(env, monkeypatch)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(env.renderer.handler)) as http:
            return await _generate(env, http, 1)

    with pytest.raises(media_render_client.MediaRenderError):
        asyncio.run(scenario())

    assert env.renderer.submits == 1 and _render_rows(env) == []


def test_a_hold_nothing_gave_back_stops_counting_at_its_deadline_and_the_boot_reaper_deletes_it(env, monkeypatch):
    now = datetime.now(timezone.utc)
    with env.factory() as db:
        db.add_all([
            # A render that died with its process: past its deadline (60 s here) and the grace.
            render_quota._reservation_row(WS, 600, "social_post:gone", now - render_quota.reservation_lifetime() - timedelta(seconds=1)),
            # A render still in flight.
            render_quota._reservation_row(WS, 120, "social_post:live", now - timedelta(seconds=5)),
        ])
        db.commit()
        reading = render_quota.render_quota(db, db.get(Workspace, WS), now)
    assert (reading.used_seconds, reading.reserved_seconds, reading.exhausted) == (0, 120, False)

    recorded = MagicMock()
    monkeypatch.setattr(reaper, "record_error", recorded)
    with env.factory() as db:
        assert reaper._reap_render_reservations(db, now - timedelta(minutes=30), now) == 1
        db.commit()

    assert _render_rows(env) == [("reserved", 120, "social_post:live")]
    assert recorded.call_args.kwargs["subsystem"] == "media_render"


def test_the_boot_reaper_sweeps_render_holds_with_the_other_surfaces():
    swept = []

    class _Session:
        def query(self, model):
            swept.append(model)
            return SimpleNamespace(filter=lambda *a: SimpleNamespace(all=lambda: []))

        def commit(self):
            pass

        def rollback(self):
            pass

    asyncio.run(reaper.reap_orphaned_runs(_Session(), now=datetime(2026, 9, 28, 12, tzinfo=timezone.utc)))
    assert LLMUsage in swept


def test_a_hold_outlives_its_renders_deadline_and_expires_before_the_reaper_cutoff():
    """Its render's deadline starts a moment after the hold; a post the reaper fails
    (stale past BOOT_REAPER_STALE_MINUTES) holds nothing by then."""
    from config import config

    lifetime = render_quota.reservation_lifetime().total_seconds()
    assert config.SOCIALS_RENDER_MAX_WAIT_SECONDS < lifetime < config.BOOT_REAPER_STALE_MINUTES * 60


# ---------------------------------------------------------------------------
# What a render holds, and what the tab is told
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "blocks, fmt, expected",
    [
        (BLOCKS_60, "social_video", 60),
        ({"html": HTML_60.replace('data-duration="60"', 'data-duration="39.5"')}, "social_video", 40),  # rounded up
        ({"html": HTML_60.replace('data-duration="60"', 'data-duration="500"')}, "social_video", 180),  # the longest
        ({"html": HTML_60.replace('data-duration="60"', "data-duration=60")}, "social_video", 180),  # unread: the longest
        (None, "social_video", 180),
        (BLOCKS_60, "social_image", 0),  # a still books no seconds (open owner call): it holds none
    ],
)
def test_a_render_holds_its_declared_duration(monkeypatch, blocks, fmt, expected):
    _set_config(monkeypatch, SOCIALS_RENDER_MAX_DURATION_SECONDS=180)
    assert render_quota.declared_seconds(blocks, fmt) == expected


def test_the_refusal_names_the_plan_the_minutes_used_and_the_minutes_held():
    start, end = render_quota.month_window_utc(datetime(2026, 9, 28, tzinfo=timezone.utc))
    reading = render_quota.RenderQuota(
        used_seconds=240, quota_minutes=10.0, period_start=start, period_end=end, plan_label="Basic", reserved_seconds=360
    )
    assert reading.exhausted
    assert reading.refusal() == (
        "This workspace has used 4.0 of its 10 render minutes this month on the Basic plan, and renders in progress "
        "hold 6.0 more until they finish. Rendering resumes on 1 October, or sooner if they use less."
    )
    nothing_held = dataclasses.replace(reading, used_seconds=600, reserved_seconds=0)
    assert nothing_held.refusal() == (
        "This workspace has used 10.0 of its 10 render minutes this month on the Basic plan. Rendering resumes on 1 October."
    )
    # A render starts only while the minutes used plus the minutes held are under the quota.
    assert not dataclasses.replace(reading, reserved_seconds=359).exhausted


def test_get_usage_shows_the_minutes_renders_in_progress_hold(env):
    responses = asyncio.run(_ask(env, _drafts(env, 2)))
    assert [r.status_code for r in responses] == [202, 202]

    body = asyncio.run(_usage(env))

    assert (body["used_minutes"], body["reserved_minutes"], body["reserved_seconds"]) == (0.0, 2.0, 120)
    assert body["remaining_minutes"] == 8.0 and body["quota_minutes"] == 10.0 and body["exhausted"] is False


def test_a_render_with_nothing_to_hold_never_touches_the_callers_session(monkeypatch):
    """Sessions of the hold's own are opened only when needed: a caller's stand-in
    session (a Playbook step's, a tool's) is never asked for its database."""
    _set_config(monkeypatch, AUTH_EDITION="local")
    sessions = render_quota.sessions_for(object())  # no get_bind(): any use would raise
    workspace = SimpleNamespace(id=WS, plan="basic", plan_limits={})

    held = asyncio.run(render_quota.reserve_render(sessions, workspace, 60, execution_id="document_template:t"))

    assert held == render_quota.RenderReservation(WS)
    assert asyncio.run(render_quota.release_render(sessions, held)) is True
    assert asyncio.run(render_quota.release_render(sessions, None)) is True


@pytest.mark.parametrize("edition, plan", [("local", "basic"), ("saas", "enterprise")])
def test_with_no_quota_a_render_holds_nothing(env, monkeypatch, edition, plan):
    _set_config(monkeypatch, AUTH_EDITION=edition)
    with env.factory() as db:
        db.get(Workspace, WS).plan = plan
        db.commit()

    responses = asyncio.run(_ask(env, _drafts(env, 3)))

    assert [r.status_code for r in responses] == [202] * 3
    assert _render_rows(env) == [] and all(job.reservation.row_id is None for job in env.launched)


# ---------------------------------------------------------------------------
# @integration: the lock holds across sessions on the CI Postgres
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pg_engine():
    from core.database.database import get_database_url

    try:
        engine = sa.create_engine(get_database_url(), pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(sa.text("SELECT 1 FROM llm_usage LIMIT 1"))
            conn.execute(sa.text("SELECT 1 FROM workspaces LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the Postgres check needs the test database: {exc}")
    yield engine
    engine.dispose()


@pytest.mark.integration
def test_the_quota_lock_holds_across_two_sessions_on_postgres(pg_engine, monkeypatch):
    """One thread takes the workspace's quota lock and holds the whole quota; a
    second thread, with its own SessionLocal, reserves meanwhile: it waits for the
    lock, then sees the first one's hold and is refused."""
    monkeypatch.setattr(render_quota, "_PROCESS_LOCKS", {})
    first_sessions, second_sessions = sessionmaker(bind=pg_engine), sessionmaker(bind=pg_engine)
    workspace_id = uuid.uuid4()
    with first_sessions() as db:
        db.add(Workspace(id=workspace_id, name="rvw3-quota-lock", plan="basic", plan_limits={}, settings={}))
        db.commit()
    holder = render_quota._Holder(workspace_id, 1.0, "Basic")  # one minute
    locked, go_on, second_done = threading.Event(), threading.Event(), threading.Event()
    outcome = {}

    def first():
        db = first_sessions()
        try:
            render_quota._lock_quota(db, workspace_id)  # the database lock alone, not this process's
            db.add(render_quota._reservation_row(workspace_id, 60, "rvw3:first", None))
            db.flush()
            locked.set()
            go_on.wait(10)
            db.commit()
        finally:
            db.close()

    def second():
        try:
            outcome["reservation"] = render_quota._reserve(second_sessions, holder, 60, "rvw3:second", None)
        except render_quota.RenderQuotaExceeded as exc:
            outcome["refused"] = str(exc)
        finally:
            second_done.set()

    threads = [threading.Thread(target=first), threading.Thread(target=second)]
    try:
        threads[0].start()
        assert locked.wait(10)
        threads[1].start()
        assert not second_done.wait(0.5), "the second session reserved while the first held the lock"
        go_on.set()
        assert second_done.wait(10)
        assert "reservation" not in outcome
        assert "and renders in progress hold 1.0 more until they finish" in outcome["refused"]
    finally:
        go_on.set()
        for thread in threads:
            thread.join(10)
        with first_sessions() as db:
            db.execute(sa.text("DELETE FROM llm_usage WHERE workspace_id = CAST(:ws AS uuid)"), {"ws": str(workspace_id)})
            db.execute(sa.text("DELETE FROM workspaces WHERE id = CAST(:ws AS uuid)"), {"ws": str(workspace_id)})
            db.commit()
