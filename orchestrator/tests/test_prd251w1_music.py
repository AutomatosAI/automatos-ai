"""PRD-251 Wave 1, US-112 (S1.6) — the music library: a CC BY track's credit reaches the post's copy.

The library itself (the manifest, the build-time fetch and hash check, the cue
analysis, the render's report) is media-render's, proved inside its image by the
media-render CI job (services/media-render/tests/test_music.py, ci/assert_music.py).
This suite pins what the repo commits and what the orchestrator does with it:

* AC1 — the committed manifest lists every track with its licence, attribution,
  source URL and sha256; no audio file is committed; the image build fetches and
  checks every track, and the CI job builds from a bad hash and requires the build
  to fail;
* AC2/AC3 — the CI job asserts Deep House 003's break at 34.0-35.8 s, and renders
  a video with library music and measures -14 ± 1 LUFS on it (the steps are pinned
  here); every seeded video names its reference track; the template contract
  checks a music cue's shape;
* AC4 — a post using a CC BY track carries its credit line in its default copy:
  - its render appends the line of the music the render mixed, to the base text
    and to every channel's own text, once; the content hash covers it and the
    history says so; CC0 music adds none; a report that asks for credit and gives
    no line, or does not name the track the bundle asked for, fails the render and
    stores nothing;
  - every rendered file records its music on its Deliverable, and a post whose
    media names a Deliverable with a credited track carries the line: at create,
    and again after an edit that drops it; another workspace's Deliverable, or a
    deleted one, credits nothing;
  - ``generate_document`` with a social format records the music on its
    Deliverable and puts the credit in its content.

The render runs over in-memory SQLite (the social tables, a SQLite copy of
``workspaces`` and a stand-in ``deliverables`` table), media-render mocked at the
HTTP layer (``httpx.MockTransport``, the real client), storage a recording fake.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import uuid
from datetime import datetime
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
import core.media_render_client as media_render_client  # noqa: E402
import core.media_render_quota as render_quota  # noqa: E402
import modules.documents.generation_service as generation_service  # noqa: E402
import modules.socials.media_store as media_store  # noqa: E402
import modules.socials.render as render  # noqa: E402
import modules.socials.service as service  # noqa: E402
import modules.socials.settings as socials_settings  # noqa: E402
import services.deliverable_service as deliverable_service  # noqa: E402
from core import music_credit  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.media_render_bundle import build_bundle  # noqa: E402
from core.media_render_client import MediaRenderClient  # noqa: E402
from core.models.socials import SocialPost, SocialPostTarget  # noqa: E402
from core.models.workspaces import Workspace  # noqa: E402
from core.social_templates import SocialTemplateError, resolve_variables, validate_social_blocks  # noqa: E402
from modules.documents.generation_service import DocumentGenerationService  # noqa: E402
from modules.documents.social_starters import social_starters  # noqa: E402

MEDIA_RENDER = _ROOT / "services" / "media-render"
MANIFEST = MEDIA_RENDER / "music" / "manifest.json"
DOCKERFILE = MEDIA_RENDER / "Dockerfile"
WORKFLOW = _ROOT / ".github" / "workflows" / "test.yml"

# Hex with letters, so SQLite keeps every UUID column as text.
WS = uuid.UUID("00000000-0000-0000-0000-0000000001b2")
WS_OTHER = uuid.UUID("00000000-0000-0000-0000-0000000001b3")
CREATED = datetime(2026, 9, 1, 9, 0)
RENDER_URL = "http://media-render:8090"
TOKEN = "render-secret"
JOB_ID = "c" * 32
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"frames " * 300
OUTPUT = {"name": "render.mp4", "aspect": "9:16", "width": 1080, "height": 1920, "bytes": len(MP4), "duration": 40.0}

CREDIT = 'Music: "Deep House 003" by Sascha Ende (ende.app), licensed CC BY 4.0.'
# The music block of a render's report, as media-render writes it (music.Track.report + the window).
DEEP_HOUSE = {
    "track": "deep-house-003", "title": "Deep House 003", "artist": "Sascha Ende", "licence": "CC BY 4.0",
    "licence_url": "https://creativecommons.org/licenses/by/4.0/", "attribution": CREDIT, "credit_required": True,
    "start": 32.0, "end": 72.0,
}
CC0_BED = {
    "track": "quiet-bed", "title": "Quiet Bed", "artist": "Someone", "licence": "CC0 1.0",
    "licence_url": "https://creativecommons.org/publicdomain/zero/1.0/", "attribution": 'Music: "Quiet Bed" by Someone, CC0 1.0.',
    "credit_required": False, "start": 0.0, "end": 40.0,
}
DEEP_HOUSE_EXTRA = {"track": "deep-house-003", "title": "Deep House 003", "licence": "CC BY 4.0", "credit": CREDIT}
COPY = {"base": "Three weeks to go.", "channels": {"linkedin": "Three weeks to go, on LinkedIn."}}

# The four reference videos' music windows (PRD-251A stage 3 and the reference mixes).
REFERENCE_MUSIC = {
    "ui-story-promo": ("where-the-night-begins", 190.1),
    "cinematic-product-promo": ("da-da-da-da-da-de-de-de-de", 50.05),
    "app-promo": ("spring-of-2026", 47.9),
    "data-story": ("deep-house-003", 32.0),
}
PINNED = {
    "where-the-night-begins": "2feeff3d7bfffc24712bf0b9df1efc76a1777973fde3143b061579c4541d6c83",
    "da-da-da-da-da-de-de-de-de": "5e71ada9dc33cd4fc21e69a38008b4b263d45de8aca2b2b5ed466816f66e9792",
    "spring-of-2026": "ab1805743e5c3a728013ad45e06a80fe6cd22384ea9f2602e4d9209cf585016d",
    "deep-house-003": "ddb3174cb4f2337a04c918902e4a766a427e947c6b04ae8d406317db07b32109",
}

HTML = (
    '<!doctype html><html><head></head><body><div id="root" data-composition-id="main" data-width="1080" '
    'data-height="1920" data-duration="40"><h1>{{ headline }}</h1></div></body></html>'
)
BLOCKS = {
    "html": HTML,
    "css": "h1 { color: var(--brand-text); }",
    "variables_schema": {"headline": {"type": "text"}},
    "sizes": ["1080x1920"],
    "audio_plan": {
        "voice": {"voice": "af_heart", "speed": 0.95, "lines": [{"id": "l01", "at": 0.3, "text": "{{ headline }}"}]},
        "music": {"track": "deep-house-003", "start": 32.0, "fade_out": 1.2},
    },
}

# ``deliverables`` has no ORM model; a stand-in with the columns the credit lookup reads.
_STANDIN = sa.MetaData()
DELIVERABLES = sa.Table(
    "deliverables",
    _STANDIN,
    sa.Column("id", sa.String(36), primary_key=True),
    sa.Column("workspace_id", sa.String(36), nullable=False),
    sa.Column("extra", sa.Text, nullable=True),
    sa.Column("deleted_at", sa.DateTime, nullable=True),
)


# ---------------------------------------------------------------------------
# AC1: the committed manifest, the build, the CI job
# ---------------------------------------------------------------------------


def _manifest():
    return json.loads(MANIFEST.read_text())


def test_every_committed_track_has_licence_attribution_url_and_sha256():
    tracks = _manifest()["tracks"]
    assert [t["id"] for t in tracks] == list(PINNED)
    for track in tracks:
        for field in ("title", "artist", "licence", "attribution", "url", "sha256"):
            assert isinstance(track.get(field), str) and track[field].strip(), (track["id"], field)
        assert track["sha256"] == PINNED[track["id"]]
        assert re.fullmatch(r"https://ende\.app/storage/mp3low/[0-9a-f-]{36}\.mp3", track["url"]), track["url"]
        assert track["licence"] == "CC-BY-4.0"
        assert track["attribution"] == f'Music: "{track["title"]}" by Sascha Ende (ende.app), licensed CC BY 4.0.'


def test_no_audio_is_committed_and_the_image_build_fetches_and_checks_every_track():
    assert sorted(p.name for p in MANIFEST.parent.iterdir()) == ["manifest.json"]
    dockerfile = DOCKERFILE.read_text()
    step = dockerfile.index("COPY music/manifest.json media_render/music_build.py /opt/media-render/build/")
    run = dockerfile.index("RUN python /opt/media-render/build/music_build.py /opt/media-render/build/manifest.json /opt/media-render/music")
    # After numpy is installed (the analysis) and before the package is copied (so a code change re-uses the layer).
    assert dockerfile.index("pip install --no-cache-dir -r requirements.txt") < step < run < dockerfile.index("\nCOPY . .")
    build = (MEDIA_RENDER / "media_render" / "music_build.py").read_text()
    assert "sha256 mismatch" in build and "def verify(" in build
    assert not re.search(r"^\s*(?:from|import)\s+(?:\.|media_render)", build, re.M), "the build step runs before the package is copied"


def _media_render_runs():
    workflow = yaml.safe_load(WORKFLOW.read_text())
    return [step.get("run") or "" for step in workflow["jobs"]["media-render"]["steps"]]


def test_the_ci_job_fails_the_image_build_on_a_bad_hash_and_checks_the_markets_break():
    (library,) = [run for run in _media_render_runs() if "media-render-bad-hash" in run]
    assert "assert_music.py --source services/media-render/music/manifest.json" in library
    assert "--break deep-house-003:34.0-35.8" in library
    assert '"sha256"' in library and "docker build" in library
    assert 'if [ "$code" -eq 0 ]' in library and 'grep -q "sha256 mismatch"' in library


def test_the_ci_job_measures_a_video_with_library_music_at_minus_14_lufs():
    (music,) = [run for run in _media_render_runs() if "fixture-bundle music" in run]
    assert "ebur128" in music and "--lufs -14 --lufs-tolerance 1" in music
    assert "assert_music.py --job" in music and f"--credit '{CREDIT}'" in music


# ---------------------------------------------------------------------------
# The seeded videos name their reference tracks; the contract checks a music cue
# ---------------------------------------------------------------------------


def test_every_seeded_video_names_its_reference_track_from_the_library():
    library = {t["id"] for t in _manifest()["tracks"]}
    videos = {s["slug"]: s for s in social_starters("social_video")}
    assert set(videos) == set(REFERENCE_MUSIC)
    for slug, (track, start) in REFERENCE_MUSIC.items():
        cue = videos[slug]["blocks"]["audio_plan"]["music"]
        assert (cue["track"], cue["start"]) == (track, start) and track in library, slug
        assert "CC BY 4.0" in videos[slug]["description"], "the template says whose music it plays"


@pytest.mark.parametrize(
    ("music", "field"),
    [
        ("deep-house-003", "audio_plan.music"),
        ({"track": "Deep House 003"}, "audio_plan.music.track"),
        ({"start": 3}, "audio_plan.music.track"),
        ({"track": "deep-house-003", "start": -1}, "audio_plan.music.start"),
        ({"track": "deep-house-003", "fade_out": True}, "audio_plan.music.fade_out"),
        ({"track": "deep-house-003", "volume": 0.5}, "audio_plan.music.volume"),
    ],
)
def test_the_contract_refuses_a_music_cue_media_render_could_not_play(music, field):
    blocks = {**BLOCKS, "audio_plan": {**BLOCKS["audio_plan"], "music": music}}
    with pytest.raises(SocialTemplateError) as refused:
        validate_social_blocks(blocks, "social_video")
    assert field in [error["field"] for error in refused.value.errors]


def test_a_music_cue_rides_the_bundle_as_the_template_names_it():
    blocks = validate_social_blocks(BLOCKS, "social_video")
    values = resolve_variables(blocks["variables_schema"], {"headline": "Launch day"}).values
    bundle = build_bundle(workspace_id=WS, reference="r", blocks=blocks, values=values, brand_kit={}, fmt="social_video")
    assert bundle["audio"]["music"] == {"track": "deep-house-003", "start": 32.0, "fade_out": 1.2}


# ---------------------------------------------------------------------------
# The credit: from the render's report, onto the copy
# ---------------------------------------------------------------------------


def test_the_report_gives_the_music_and_the_line_a_cc_by_track_asks_for():
    credit = music_credit.credit_from_report({"music": DEEP_HOUSE})
    assert credit == music_credit.MusicCredit(track="deep-house-003", title="Deep House 003", licence="CC BY 4.0", line=CREDIT)
    assert credit.extra() == DEEP_HOUSE_EXTRA
    assert music_credit.credit_from_report({"music": CC0_BED}).line is None, "CC0 asks for no credit"
    assert music_credit.credit_from_report({"check": {"ok": True}}) is None
    assert music_credit.credit_from_report(None) is None
    folded = music_credit.credit_from_report({"music": {**DEEP_HOUSE, "attribution": '  Music:\n "Deep House 003"  by Sascha Ende '}})
    assert folded.line == 'Music: "Deep House 003" by Sascha Ende'
    for broken in ("", None, "x" * (music_credit.CREDIT_MAX_CHARS + 1)):
        with pytest.raises(music_credit.MusicCreditMissing):
            music_credit.credit_from_report({"music": {**DEEP_HOUSE, "attribution": broken}})
    assert music_credit.deliverable_credit({"music": DEEP_HOUSE_EXTRA}) == CREDIT
    assert music_credit.deliverable_credit({"music": {**DEEP_HOUSE_EXTRA, "credit": None}}) is None
    assert music_credit.deliverable_credit(None) is None and music_credit.deliverable_credit({"sha256": "x"}) is None


def test_a_render_must_report_the_track_its_bundle_asked_for():
    asked = {"audio": {"music": {"track": "deep-house-003", "start": 32.0}}}
    assert music_credit.bundle_track(asked) == "deep-house-003"
    assert music_credit.bundle_track({"audio": {"voice": {}}}) is None and music_credit.bundle_track(None) is None
    assert music_credit.credit_for_render(asked, {"music": DEEP_HOUSE}).line == CREDIT
    assert music_credit.credit_for_render({"audio": {}}, {"music": DEEP_HOUSE}).line == CREDIT
    assert music_credit.credit_for_render({}, {"check": {"ok": True}}) is None
    with pytest.raises(music_credit.MusicCreditMissing, match="asked to mix deep-house-003, and its report names no music"):
        music_credit.credit_for_render(asked, {"check": {"ok": True}})
    with pytest.raises(music_credit.MusicCreditMissing, match="its report names quiet-bed"):
        music_credit.credit_for_render(asked, {"music": CC0_BED})


def test_with_credits_appends_each_line_once_to_the_base_and_every_channel():
    copy = {"base": "Three weeks to go.", "channels": {"linkedin": "On LinkedIn.", "x": "On X.\n" + CREDIT}}
    kept = json.loads(json.dumps(copy))
    credited = service.with_credits(copy, [CREDIT, CREDIT])
    assert credited == {
        "base": "Three weeks to go.\n\n" + CREDIT,
        "channels": {"linkedin": "On LinkedIn.\n\n" + CREDIT, "x": "On X.\n" + CREDIT},
    }
    assert copy == kept, "the copy it was given is left as it was"
    assert service.with_credits(credited, [CREDIT]) is credited
    assert service.with_credits(None, [CREDIT]) == service.with_credits({}, [CREDIT]) == {"base": CREDIT}
    assert service.with_credits({"channels": {"x": "On X."}}, [CREDIT]) == {"base": CREDIT, "channels": {"x": "On X.\n\n" + CREDIT}}
    assert service.with_credits(copy, []) is copy
    # Copy the validators refuse (a null text, say) is left for them to refuse, never filled in.
    odd = {"base": None, "channels": {"x": None}}
    assert service.with_credits(odd, [CREDIT]) is odd


def test_finish_render_puts_the_credit_in_the_copy_the_hash_covers():
    post = SocialPost(
        workspace_id=WS, created_by="member-1", status="rendering", title="Launch", copy={"base": "Go."},
        variables={}, sources={}, media={}, review_log=[], override_unsourced=False,
    )
    post.content_hash = before = service.compute_content_hash(post)
    record = {"deliverable_id": "d-1", "name": "video-9x16.mp4", "sha256": "a" * 64, "bytes": 10}
    service.finish_render(post, "member-1", {"9:16": [record]}, credits=[CREDIT])
    assert post.copy == {"base": "Go.\n\n" + CREDIT}
    assert post.content_hash == service.compute_content_hash(post) != before
    assert post.status == "needs_approval" and post.review_log[-1]["credits_added"] == [CREDIT]


# ---------------------------------------------------------------------------
# The harness: a render over SQLite, and the Socials API
# ---------------------------------------------------------------------------


def _portable(col_type):
    if isinstance(col_type, (JSONB, ARRAY)):
        return sa.JSON()
    if isinstance(col_type, sa.Uuid) or type(col_type).__name__.upper() == "UUID":
        return sa.Uuid()
    return col_type


def _sqlite_copy(table: sa.Table, metadata: sa.MetaData) -> sa.Table:
    columns = [sa.Column(c.name, _portable(c.type), primary_key=c.primary_key) for c in table.columns]
    return sa.Table(table.name, metadata, *columns)


def _set_config(monkeypatch, **values):
    """Patch the config object every module under test reads."""
    targets = {id(m.config): m.config for m in (render, render_quota, media_render_client, media_store)}
    for cfg in targets.values():
        for name, value in values.items():
            monkeypatch.setattr(cfg, name, value, raising=False)


class FakeStore:
    def __init__(self):
        self.objects = {}

    def configured(self):
        return True

    def put_file(self, key, path, content_type):
        self.objects[key] = (Path(path).read_bytes(), content_type)


class Deliverables:
    """DeliverableService (its SQL is Postgres-only), writing the stand-in table the credit lookup reads."""

    registered: list = []

    def __init__(self, db, workspace_id):
        self.db, self.workspace_id = db, workspace_id

    def register(self, **kwargs):
        ident = str(uuid.uuid4())
        self.db.execute(
            DELIVERABLES.insert().values(
                id=ident, workspace_id=str(self.workspace_id), extra=json.dumps(kwargs.get("extra") or {}), deleted_at=None
            )
        )
        self.db.commit()
        type(self).registered.append({"deliverable_id": ident, **kwargs})
        return {"success": True, "deliverable_id": ident, "created": True}


class Renderer:
    """media-render over httpx.MockTransport; its finished job reports ``music``."""

    def __init__(self, music):
        self.music = music

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path == "/render":
            return httpx.Response(202, json={"id": JOB_ID, "status": "rendering", "outputs": [], "report": {}})
        if request.method == "GET" and path == f"/render/{JOB_ID}":
            report = {"check": {"ok": True, "errors": 0}, "timings": {"render_seconds": 30.0}, "audio": {"integrated_lufs": -14.0}}
            if self.music is not None:
                report["music"] = self.music
            return httpx.Response(200, json={"id": JOB_ID, "status": "done", "outputs": [OUTPUT], "report": report})
        if request.method == "GET" and path == f"/render/{JOB_ID}/output/render.mp4":
            return httpx.Response(200, content=MP4)
        return httpx.Response(404, json={"error": "not_found", "message": f"no route {path}"})


@pytest.fixture
def env(monkeypatch):
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    copies = sa.MetaData()
    _sqlite_copy(Workspace.__table__, copies)
    copies.create_all(engine)
    SocialPost.metadata.create_all(engine, tables=[SocialPost.__table__, SocialPostTarget.__table__])
    _STANDIN.create_all(engine)
    factory = sessionmaker(bind=engine)
    with factory() as db:
        for ws_id in (WS, WS_OTHER):
            db.add(
                Workspace(
                    id=ws_id, name=f"ws-{ws_id.hex[-2:]}", plan="basic", plan_limits={},
                    settings={"socials": {"enabled": True}}, onboarding={}, created_at=CREATED, updated_at=CREATED,
                )
            )
        db.commit()

    state = SimpleNamespace(factory=factory, booked=[])
    monkeypatch.setattr(socials_settings, "read_system_setting", lambda category, key: "true")
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: "owner")
    monkeypatch.setattr(render, "book_render_seconds", lambda **kwargs: state.booked.append(kwargs))
    _set_config(
        monkeypatch,
        AUTH_EDITION="saas",
        SOCIALS_RENDER_URL=RENDER_URL,
        SOCIALS_RENDER_TOKEN=TOKEN,
        SOCIALS_RENDER_POLL_SECONDS=0,
        SOCIALS_RENDER_MAX_WAIT_SECONDS=60,
    )
    Deliverables.registered = []
    monkeypatch.setattr(deliverable_service, "DeliverableService", Deliverables)

    def session():
        db = factory()
        try:
            yield db
        finally:
            db.close()

    ctx = RequestContext(
        workspace_id=WS,
        user=UserContext(id="member-1", clerk_user_id="clerk-member-1", system_role="user"),
        auth_type="clerk",
    )
    app = FastAPI()
    app.include_router(socials_api.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: ctx
    app.dependency_overrides[get_db] = session
    state.client = TestClient(app)
    try:
        yield state
    finally:
        engine.dispose()


def _rendering_post(env, *, copy=COPY, music=None):
    """A draft with ``copy``, moved to rendering: the job a render starts with; its bundle names ``music``."""
    bundle = {"workspace_id": str(WS), "composition": {"html": HTML}}
    if music is not None:
        bundle["audio"] = {"music": music}
    with env.factory() as db:
        post = service.create_draft(db, workspace_id=WS, created_by="member-1", title="Launch video", copy=copy, format="video")
        db.commit()
        service.start_render(post, "member-1")
        db.commit()
        return render.RenderJob(
            post_id=post.id, workspace_id=WS, actor="member-1", content_hash=post.content_hash,
            title=post.title, format=post.format, bundle=bundle,
        )


def _run(job, renderer, store, factory):
    async def go():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(renderer.handler), headers={"X-Internal-Token": TOKEN}
        ) as http:
            return await render.run_render(job, client=MediaRenderClient(http), store=store, session_factory=factory)

    return asyncio.run(go())


def _post(env, post_id):
    with env.factory() as db:
        post = db.get(SocialPost, post_id)
        db.expunge(post)
        return post


def _deliverable(env, workspace_id, music=None, *, deleted=False):
    ident = str(uuid.uuid4())
    extra = {"sha256": "a" * 64, **({"music": music} if music is not None else {})}
    with env.factory() as db:
        db.execute(
            DELIVERABLES.insert().values(
                id=ident, workspace_id=str(workspace_id), extra=json.dumps(extra), deleted_at=CREATED if deleted else None
            )
        )
        db.commit()
    return ident


# ---------------------------------------------------------------------------
# AC4: a post that renders with a CC BY track carries its credit line
# ---------------------------------------------------------------------------


def test_a_render_with_a_cc_by_track_puts_its_credit_line_in_the_default_copy(env):
    job = _rendering_post(env, music={"track": "deep-house-003", "start": 32.0})
    store = FakeStore()
    assert _run(job, Renderer(DEEP_HOUSE), store, env.factory) is True

    post = _post(env, job.post_id)
    assert post.status == "needs_approval"
    assert post.copy == {
        "base": "Three weeks to go.\n\n" + CREDIT,
        "channels": {"linkedin": "Three weeks to go, on LinkedIn.\n\n" + CREDIT},
    }
    # The approver reviews the credited copy: the hash covers it with the rendered file.
    assert post.content_hash == service.compute_content_hash(post)
    done = post.review_log[-1]
    assert done["action"] == "render_done" and done["credits_added"] == [CREDIT]
    assert done["report"]["music"]["track"] == "deep-house-003" and done["report"]["music"]["attribution"] == CREDIT
    # The file records its music, so a post that attaches it later carries the credit too.
    (registered,) = Deliverables.registered
    assert registered["extra"]["music"] == DEEP_HOUSE_EXTRA
    assert [b["seconds"] for b in env.booked] == [40.0]

    # An edit that drops the line gets it back: the post's media still plays the track.
    resp = env.client.patch(f"/api/socials/posts/{job.post_id}", json={"copy": {"base": "New words."}})
    assert resp.status_code == 200, resp.text
    assert resp.json()["copy"] == {"base": "New words.\n\n" + CREDIT}
    # A label edit changes no content: the hash (and so an approval) stands.
    hashed = resp.json()["content_hash"]
    renamed = env.client.patch(f"/api/socials/posts/{job.post_id}", json={"title": "Launch video, final"})
    assert renamed.status_code == 200, renamed.text
    assert renamed.json()["content_hash"] == hashed and renamed.json()["copy"] == {"base": "New words.\n\n" + CREDIT}


def test_cc0_music_asks_for_no_credit(env):
    job = _rendering_post(env)
    assert _run(job, Renderer(CC0_BED), FakeStore(), env.factory) is True
    post = _post(env, job.post_id)
    assert post.status == "needs_approval" and post.copy == COPY
    assert "credits_added" not in post.review_log[-1]
    (registered,) = Deliverables.registered
    assert registered["extra"]["music"] == {"track": "quiet-bed", "title": "Quiet Bed", "licence": "CC0 1.0", "credit": None}


def test_a_render_without_music_leaves_the_copy_and_records_none(env):
    job = _rendering_post(env)
    assert _run(job, Renderer(None), FakeStore(), env.factory) is True
    post = _post(env, job.post_id)
    assert post.copy == COPY and "music" not in Deliverables.registered[0]["extra"]


@pytest.mark.parametrize(
    "reported",
    [pytest.param({**DEEP_HOUSE, "attribution": ""}, id="no-credit-line"), pytest.param(None, id="no-music-reported")],
)
def test_a_render_that_cannot_credit_the_track_it_was_asked_to_mix_fails_and_stores_nothing(env, reported):
    job = _rendering_post(env, music={"track": "deep-house-003", "start": 32.0})
    store = FakeStore()
    assert _run(job, Renderer(reported), store, env.factory) is False
    post = _post(env, job.post_id)
    assert post.status == "failed" and post.copy == COPY
    assert post.review_log[-1]["report"]["code"] == "music_credit_missing"
    assert "credit line" in post.review_log[-1]["comment"]
    assert store.objects == {} and Deliverables.registered == [] and env.booked == []


# ---------------------------------------------------------------------------
# AC4: a post whose media is a file with a CC BY track carries its credit line
# ---------------------------------------------------------------------------


def test_a_post_whose_media_plays_a_cc_by_track_carries_its_credit_from_the_start(env):
    credited = _deliverable(env, WS, DEEP_HOUSE_EXTRA)
    elsewhere = _deliverable(env, WS_OTHER, {**DEEP_HOUSE_EXTRA, "credit": "Music: another workspace's track."})
    deleted = _deliverable(env, WS, {**DEEP_HOUSE_EXTRA, "credit": "Music: a deleted file's track."}, deleted=True)
    silent = _deliverable(env, WS)
    media = {"9:16": [credited, elsewhere, deleted, silent, "not-a-uuid"]}
    resp = env.client.post(
        "/api/socials/posts", json={"title": "Launch", "copy": {"base": "Launch day."}, "format": "video", "media": media}
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["copy"] == {"base": "Launch day.\n\n" + CREDIT}
    assert body["media"] == media and body["content_hash"] == service.compute_content_hash(_post(env, uuid.UUID(body["id"])))


def test_an_edit_that_attaches_a_credited_file_adds_its_credit_and_one_without_adds_none(env):
    created = env.client.post("/api/socials/posts", json={"title": "Launch", "copy": {"base": "Launch day."}, "format": "video"})
    assert created.status_code == 201 and created.json()["copy"] == {"base": "Launch day."}
    post_id = created.json()["id"]

    plain = env.client.patch(f"/api/socials/posts/{post_id}", json={"media": {"9:16": [_deliverable(env, WS)]}})
    assert plain.status_code == 200 and plain.json()["copy"] == {"base": "Launch day."}

    attached = env.client.patch(f"/api/socials/posts/{post_id}", json={"media": {"9:16": [_deliverable(env, WS, DEEP_HOUSE_EXTRA)]}})
    assert attached.status_code == 200, attached.text
    assert attached.json()["copy"] == {"base": "Launch day.\n\n" + CREDIT}


def test_the_credit_lookup_reads_only_the_callers_live_deliverables(env):
    from modules.socials import credits

    mine = _deliverable(env, WS, DEEP_HOUSE_EXTRA)
    record = {"deliverable_id": mine, "name": "video-9x16.mp4", "sha256": "a" * 64, "bytes": 10}
    with env.factory() as db:
        assert credits.media_credits(db, WS, {"9:16": [record], "1:1": [mine]}) == [CREDIT]
        assert credits.media_credits(db, WS_OTHER, {"9:16": [mine]}) == []
        assert credits.media_credits(db, WS, {"9:16": ["not-a-uuid", 7, None]}) == []
        assert credits.media_credits(db, WS, None) == [] and credits.media_credits(db, WS, {}) == []
    assert credits.media_deliverable_ids({"a": [mine, record, mine.upper()], "b": "no"}) == [mine]


# ---------------------------------------------------------------------------
# generate_document with a social format: the file's Deliverable records its music
# ---------------------------------------------------------------------------

TEMPLATE_ID = uuid.UUID("00000000-0000-0000-0000-0000000001c1")


class DocRenderer:
    def __init__(self, music):
        self.music = music

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path == "/render":
            return httpx.Response(202, json={"id": JOB_ID, "status": "rendering", "outputs": [], "report": {}})
        if request.method == "GET" and path == f"/render/{JOB_ID}":
            report = {"check": {"ok": True}, "music": self.music} if self.music else {"check": {"ok": True}}
            return httpx.Response(200, json={"id": JOB_ID, "status": "done", "outputs": [OUTPUT], "report": report})
        if request.method == "GET" and path == f"/render/{JOB_ID}/output/render.mp4":
            return httpx.Response(200, content=MP4)
        return httpx.Response(404, json={"error": "not_found"})


@pytest.fixture
def documents(monkeypatch, tmp_path):
    for cfg in {id(m.config): m.config for m in (generation_service, media_render_client)}.values():
        monkeypatch.setattr(cfg, "SOCIALS_RENDER_URL", RENDER_URL, raising=False)
        monkeypatch.setattr(cfg, "SOCIALS_RENDER_TOKEN", "", raising=False)
        monkeypatch.setattr(cfg, "SOCIALS_RENDER_POLL_SECONDS", 0, raising=False)
        monkeypatch.setattr(cfg, "SOCIALS_RENDER_MAX_WAIT_SECONDS", 60, raising=False)
    monkeypatch.setattr(generation_service, "GENERATED_DIR", str(tmp_path))
    monkeypatch.setattr(generation_service, "is_storage_configured", lambda: False)
    monkeypatch.setattr(generation_service, "brand_kit_for_media_render", lambda kit: dict(kit))
    monkeypatch.setattr(generation_service, "enforce_render_quota", lambda db, workspace: None)
    state = SimpleNamespace(booked=[], registered=[], dir=tmp_path)
    monkeypatch.setattr(generation_service, "book_render_seconds", lambda **kwargs: state.booked.append(kwargs))

    class Recorded:
        def __init__(self, db, workspace_id):
            pass

        def register(self, **kwargs):
            state.registered.append(kwargs)
            return {"success": True, "deliverable_id": "d-1"}

    monkeypatch.setattr(deliverable_service, "DeliverableService", Recorded)
    template = SimpleNamespace(id=TEMPLATE_ID, name="Launch video", format="social_video", blocks=BLOCKS)
    state.template = template
    service_ = DocumentGenerationService(MagicMock(), WS)
    service_.template_service = SimpleNamespace(
        get_template=lambda template_id, workspace_id: template, get_template_by_name=lambda workspace_id, name: None
    )
    service_.db.query.return_value.filter.return_value.first.return_value = SimpleNamespace(
        id=WS, name="Workspace One", plan="basic", plan_limits={}, settings={}
    )
    state.service = service_
    return state


def _generate(documents, renderer):
    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(renderer.handler)) as http:
            documents.service._render_client = lambda: MediaRenderClient(http)
            return await documents.service.generate(
                title="Launch video", format="social_video", data={"headline": "Launch day"},
                workspace_id=WS, template_id=TEMPLATE_ID,
            )

    return asyncio.run(go())


def test_generate_document_records_the_music_on_the_video_deliverable_and_in_its_content(documents):
    result = _generate(documents, DocRenderer(DEEP_HOUSE))
    assert result.music == DEEP_HOUSE_EXTRA
    assert result.content.rstrip().endswith(CREDIT)
    documents.service.register_as_deliverable(result, title="Launch video")
    (registered,) = documents.registered
    assert registered["artifact_type"] == "video" and registered["extra"]["music"] == DEEP_HOUSE_EXTRA


def test_generate_document_without_music_records_none(documents):
    documents.template.blocks = {**BLOCKS, "audio_plan": {"voice": BLOCKS["audio_plan"]["voice"]}}
    result = _generate(documents, DocRenderer(None))
    assert result.music is None and CREDIT not in result.content
    documents.service.register_as_deliverable(result, title="Launch video")
    assert "music" not in documents.registered[0]["extra"]


@pytest.mark.parametrize(
    "reported",
    [pytest.param({**DEEP_HOUSE, "attribution": None}, id="no-credit-line"), pytest.param(None, id="no-music-reported")],
)
def test_generate_document_refuses_a_render_it_cannot_credit_and_keeps_no_file(documents, reported):
    with pytest.raises(music_credit.MusicCreditMissing):
        _generate(documents, DocRenderer(reported))
    assert documents.booked == [] and not any(documents.dir.rglob("*.mp4"))
