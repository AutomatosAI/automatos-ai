"""PRD-251 Wave 1, US-106 (S1.2b) — the four reference videos, seeded as social video templates.

Pins:

* **The four starters.** "UI story promo", "Cinematic product promo", "App promo"
  and "Data story" (``modules/documents/social_starters.py``): social_video at
  1080x1920, each passing the template contract and the brand rule (D4), each
  with a voice line per beat.
* **The port keeps the reference's timing.** Every scene and caption clip of
  each reference composition (docs/PRDS/prd251-reference/) starts and lasts
  exactly as it did, and the whole runs as long.
* **Every word is a variable.** No line of a reference's copy is baked into its
  template; the starter's sample data carries the reference's own copy, and it
  fills every variable a render needs.
* **Seeding (AC 1).** Turning Socials on seeds the four into the workspace in the
  switch's own commit, with the eight image templates US-107 and US-113 add
  beside them (test_prd251w1_image_templates.py); turning it on twice leaves the same rows;
  a person's own template of the same name is never touched. On real Postgres
  (``@integration``): the rows land once, and ``GET /api/documents/templates``
  lists them.
* **Footage slots (AC 4).** A filled slot reaches media-render as a media file at
  the slot's path, and its element stays; an empty slot's element is taken out,
  the template's own motion graphics stay, and nothing points at a missing file.
  media-render's own bundle parser takes every starter both ways. The contract
  refuses a slot the removal could not take whole.
* **Voice lines are template text.** Each line's ``{{ name }}`` is filled from the
  variables (the brand's name too); a line that fills in empty is dropped; an
  undeclared name is refused on save.
* **The stage palette.** Any brand kit gives a dark ink and readable text tones;
  the Automatos kit reproduces the reference palette exactly.
* **The CI driver** (scripts/ci/social_template_previews.py) builds every seeded
  video template's preview bundle, and its PNG coder round-trips.
"""
from __future__ import annotations

import html
import importlib.util
import json
import os
import re
import sys
import uuid
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
_ROOT = _ORCH.parent
_MEDIA_RENDER = _ROOT / "services" / "media-render"
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))
# media-render's bundle parser (standard library only), found LAST so no orchestrator module is shadowed.
if str(_MEDIA_RENDER) not in sys.path:
    sys.path.append(str(_MEDIA_RENDER))

import sqlalchemy as sa  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

import core.models  # noqa: E402,F401  (registers every mapper)
import api.document_generation as documents_api  # noqa: E402
import api.workspaces as workspaces_api  # noqa: E402
import core.auth.workspace_permission as permission_mod  # noqa: E402
import modules.socials.settings as socials_settings  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.brand_palette import STAGE_TOKENS, contrast, luminance, parse_hex, stage_palette  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.media_render_bundle import build_bundle  # noqa: E402
from core.models.core import DocumentTemplate  # noqa: E402
from core.social_templates import (  # noqa: E402
    SocialTemplateError,
    brand_literals,
    resolve_variables,
    validate_social_blocks,
    without_slots,
)
from media_render.bundle import parse_bundle  # noqa: E402  (media-render's own parser: stdlib only)
from media_render.config import load_settings  # noqa: E402
from media_render.media_urls import parse_prefixes  # noqa: E402
from media_render.music import load_library  # noqa: E402
from modules.documents import seed_templates  # noqa: E402
from modules.documents.social_starters import (  # noqa: E402
    SOCIAL_IMAGE_STARTER_SLUGS,
    SOCIAL_STARTER_SLUGS,
    SOCIAL_VIDEO_STARTER_SLUGS,
    social_starters,
)
from modules.documents.template_summary import STARTER_CREATOR, summarize_template  # noqa: E402

WS = uuid.UUID("00000000-0000-0000-0000-0000000001a6")
STARTER_NAMES = ["UI story promo", "Cinematic product promo", "App promo", "Data story"]
# US-107 (and US-113's infographic): the image templates seed through the same path, after the videos.
IMAGE_STARTER_NAMES = [
    "Title card", "Definition card", "Stats card", "Quote card", "Announcement card", "Fact card", "Carousel", "Infographic",
]
ALL_STARTER_NAMES = STARTER_NAMES + IMAGE_STARTER_NAMES
REFERENCES = _ROOT / "docs" / "PRDS" / "prd251-reference"
REFERENCE_OF = {
    "ui-story-promo": "v1-ui-story.html",
    "cinematic-product-promo": "v2-cinematic-product.html",
    "app-promo": "academy-app-promo.html",
    "data-story": "markets-posh.html",
}
STORAGE = "https://storage.example-ci.test/automatos/"
SWITCH_ROUTE = "/api/workspaces/current/socials"
KIT = {
    "name": "Automatos",
    "tagline": "An operating system for autonomous agent teams",
    "primary_color": "#e96235",
    "secondary_color": "#1a1714",
    "accent_color": "#90af5a",
    "text_color": "#f0e8db",
    "font_family": "Geist, sans-serif",
}


def _starter(slug):
    return next(s for s in social_starters() if s["slug"] == slug)


def _bundle(starter, kit=KIT, slot_media=None):
    resolved = resolve_variables(starter["blocks"]["variables_schema"], starter["sample_data"])
    assert resolved.missing == [] and resolved.invalid == [], (starter["name"], resolved)
    return build_bundle(
        workspace_id=WS,
        reference=f"seeded:{starter['slug']}",
        blocks=starter["blocks"],
        values=resolved.values,
        brand_kit=kit,
        slot_media=slot_media,
    )


def _renderer_settings():
    from dataclasses import replace

    return replace(load_settings({}), media_url_prefixes=parse_prefixes(STORAGE))


@pytest.fixture(scope="module")
def music_library(tmp_path_factory):
    """media-render's music library as the image carries it (US-112): the committed
    tracks, each a stub file (the build fetches the real ones), with no analysis."""
    root = tmp_path_factory.mktemp("music")
    tracks = []
    for entry in json.loads((_MEDIA_RENDER / "music" / "manifest.json").read_text())["tracks"]:
        (root / f"{entry['id']}.mp3").write_bytes(b"stub")
        tracks.append({**entry, "file": f"{entry['id']}.mp3"})
    (root / "manifest.json").write_text(json.dumps({"tracks": tracks}))
    return load_library(str(root))


# ---------------------------------------------------------------------------
# The four starters
# ---------------------------------------------------------------------------


def test_the_four_reference_videos_are_seeded_as_social_video_starters():
    starters = social_starters("social_video")
    assert [s["name"] for s in starters] == STARTER_NAMES
    assert [s["slug"] for s in starters] == list(SOCIAL_VIDEO_STARTER_SLUGS) == list(REFERENCE_OF)
    assert list(SOCIAL_STARTER_SLUGS) == list(SOCIAL_VIDEO_STARTER_SLUGS) + list(SOCIAL_IMAGE_STARTER_SLUGS)
    for starter in starters:
        blocks = starter["blocks"]
        assert (starter["format"], starter["category"], blocks["sizes"]) == ("social_video", "social", ["1080x1920"])
        assert validate_social_blocks(blocks, "social_video") == blocks
        assert brand_literals(blocks["html"], blocks.get("css") or "") == []
        assert len(blocks["audio_plan"]["voice"]["lines"]) >= 9, starter["name"]
        assert starter["description"] and starter["preview"]["at"]


def test_the_seeds_are_data_the_seeder_writes_and_rendering_never_reads_a_file():
    """Rendering reads the row (CLAUDE.md §4): nothing outside the seed loader names the seed files."""
    offenders = [
        str(path.relative_to(_ORCH))
        for path in (_ORCH / "modules").rglob("*.py")
        if "templates/social" in path.read_text(encoding="utf-8") or "templates\" / \"social" in path.read_text(encoding="utf-8")
    ]
    assert offenders == ["modules/documents/social_starters.py"]


class _Clips(HTMLParser):
    """The scene and caption clips of a composition, ``{id: (start, duration)}``, and the root's duration."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.clips, self.duration = {}, None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("data-composition-id") == "main":
            self.duration = float(attrs["data-duration"])
        classes = (attrs.get("class") or "").split()
        scene = tag == "section" or "cap" in classes or attrs.get("id") == "bg"
        if scene and "clip" in classes and "data-start" in attrs:
            self.clips[attrs["id"]] = (float(attrs["data-start"]), float(attrs["data-duration"]))


def _clips(text):
    parser = _Clips()
    parser.feed(text)
    return parser


@pytest.mark.parametrize("slug", list(REFERENCE_OF))
def test_each_template_keeps_its_references_timing_structure(slug):
    reference = _clips((REFERENCES / REFERENCE_OF[slug]).read_text(encoding="utf-8"))
    ported = _clips(_starter(slug)["blocks"]["html"])
    assert ported.duration == reference.duration
    assert len(reference.clips) >= 8
    assert ported.clips == reference.clips


def test_no_line_of_the_references_copy_is_baked_into_a_template():
    for starter in social_starters():
        page = starter["blocks"]["html"]
        baked = [
            value for value in starter["sample_data"].values()
            if isinstance(value, str) and len(value) >= 6 and value in page
        ]
        assert baked == [], (starter["name"], baked)
        for word in ("Harbourline", "Harvest Club", "Shopify", "Gerard", "automatos.app", "86,370"):
            assert word not in page, (starter["name"], word)


def test_the_sample_data_is_the_references_own_copy_and_fills_every_variable():
    reference = html.unescape((REFERENCES / "v1-ui-story.html").read_text(encoding="utf-8"))
    sample = _starter("ui-story-promo")["sample_data"]
    for key in ("hook_eyebrow", "chat_request", "task_3_title", "ask_question", "done_6", "cap_outputs", "end_accent"):
        assert sample[key] in reference, key
    for starter in social_starters("social_video"):
        resolved = resolve_variables(starter["blocks"]["variables_schema"], starter["sample_data"])
        assert resolved.missing == [] and resolved.invalid == [], starter["name"]
        # Every story line has no default: a post must write it, never inherit a reference's copy.
        required = [n for n, spec in starter["blocks"]["variables_schema"].items() if "default" not in spec]
        assert len(required) >= 50, starter["name"]


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


class _Session:
    """Keeps the template rows the seeder adds; answers its lookup by workspace and name."""

    def __init__(self, rows=()):
        self.rows = list(rows)
        self.commits = 0
        self._criteria = {}

    def query(self, _model):
        self._criteria = {}
        return self

    def filter(self, *criteria):
        for criterion in criteria:
            self._criteria[criterion.left.key] = criterion.right.value
        return self

    def first(self):
        return next((r for r in self.rows if all(getattr(r, k) == v for k, v in self._criteria.items())), None)

    def add(self, row):
        self.rows.append(row)

    def commit(self):
        self.commits += 1


def test_the_social_starters_seed_through_the_starter_path_once():
    db = _Session()
    assert seed_templates.seed_social_starters(db, WS) == {"created": len(ALL_STARTER_NAMES), "refreshed": 0}
    assert [row.name for row in db.rows] == ALL_STARTER_NAMES and db.commits == 1
    starters = social_starters()
    assert len(starters) == len(db.rows) == 12
    for row, starter in zip(db.rows, starters):
        assert (row.workspace_id, row.format, row.category, row.created_by) == (WS, starter["format"], "social", STARTER_CREATOR)
        assert row.blocks == starter["blocks"] and row.sample_data == starter["sample_data"]
    assert [row.format for row in db.rows] == ["social_video"] * 4 + ["social_image"] * 8
    # Twice is the same rows: nothing added, nothing refreshed, nothing committed.
    assert seed_templates.seed_social_starters(db, WS) == {"created": 0, "refreshed": 0}
    assert len(db.rows) == 12 and db.commits == 1


def test_a_drifted_starter_is_refreshed_and_a_persons_own_is_never_touched():
    stale = DocumentTemplate(workspace_id=WS, name="Data story", format="social_video", category="social",
                             blocks={"old": True}, sample_data={}, description="", created_by=STARTER_CREATOR, is_active=True)
    mine = DocumentTemplate(workspace_id=WS, name="App promo", format="social_video", category="social",
                            blocks={"mine": True}, sample_data={}, description="", created_by="user-7", is_active=True)
    db = _Session([stale, mine])
    assert seed_templates.seed_social_starters(db, WS, commit=False) == {"created": len(ALL_STARTER_NAMES) - 2, "refreshed": 1}
    assert stale.blocks == _starter("data-story")["blocks"]
    assert mine.blocks == {"mine": True} and mine.created_by == "user-7"
    assert db.commits == 0, "commit=False leaves the commit to the caller"


def test_the_gallery_lists_a_seeded_starter_by_its_variables():
    db = _Session()
    seed_templates.seed_social_starters(db, WS)
    listed = [summarize_template(row) for row in db.rows]
    assert [entry["name"] for entry in listed] == ALL_STARTER_NAMES
    for entry, starter in zip(listed, social_starters()):
        assert entry["format"] == starter["format"] and entry["is_starter"] is True and entry["has_blocks"] is False
        assert entry["data_fields"] and entry["sample_data"]


class _SwitchDB(_Session):
    """The workspace for the switch route (``query(Workspace).get``), and the template rows."""

    def __init__(self, workspace):
        super().__init__()
        self.workspace = workspace

    def query(self, model):
        super().query(model)
        return self

    def get(self, ident):
        return self.workspace if ident == self.workspace.id else None


def _ctx(workspace_id=WS):
    return RequestContext(
        workspace_id=workspace_id,
        user=UserContext(id="owner-1", clerk_user_id="clerk-owner-1", system_role="user"),
        auth_type="clerk",
    )


def test_turning_socials_on_seeds_the_starters_in_the_switchs_own_commit(monkeypatch):
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: "owner")
    monkeypatch.setattr(socials_settings, "read_system_setting", lambda category, key: "true")
    monkeypatch.setattr("sqlalchemy.orm.attributes.flag_modified", lambda instance, key: None)
    workspace = SimpleNamespace(id=WS, settings={})
    db = _SwitchDB(workspace)
    app = FastAPI()
    app.include_router(workspaces_api.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: _ctx()
    app.dependency_overrides[get_db] = lambda: db
    client = TestClient(app)

    resp = client.put(SWITCH_ROUTE, json={"socials": {"enabled": True}})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"status": "saved", "socials": {"available": True, "enabled": True}}
    assert [row.name for row in db.rows] == ALL_STARTER_NAMES and db.commits == 1

    # On again: the same rows. Off: nothing seeded, nothing removed.
    assert client.put(SWITCH_ROUTE, json={"socials": {"enabled": True}}).status_code == 200
    assert client.put(SWITCH_ROUTE, json={"socials": {"enabled": False}}).status_code == 200
    assert [row.name for row in db.rows] == ALL_STARTER_NAMES and db.commits == 3


def test_a_seed_that_breaks_the_contract_never_reaches_a_workspace(monkeypatch, tmp_path):
    import modules.documents.social_starters as starters_module

    for slug in SOCIAL_STARTER_SLUGS:
        for suffix in (".html", ".json"):
            (tmp_path / f"{slug}{suffix}").write_text((starters_module.STARTERS_DIR / f"{slug}{suffix}").read_text())
    broken = tmp_path / "data-story.html"
    broken.write_text(broken.read_text().replace("background: var(--ink);", "background: #1a1714;", 1))
    monkeypatch.setattr(starters_module, "STARTERS_DIR", tmp_path)
    starters_module._starters.cache_clear()
    try:
        with pytest.raises(SocialTemplateError) as refused:
            starters_module.social_starters()
        assert any(e["field"].startswith("data-story.") and "#1a1714" in e["message"] for e in refused.value.errors)
    finally:
        starters_module._starters.cache_clear()


# ---------------------------------------------------------------------------
# Seeding on real Postgres, through the routes (AC 1)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pg_engine():
    from core.database.database import get_database_url

    try:
        engine = sa.create_engine(get_database_url(), pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(sa.text("SELECT 1 FROM document_templates LIMIT 1"))
            conn.execute(sa.text("SELECT 1 FROM workspaces LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the Postgres checks need the test database: {exc}")
    yield engine
    engine.dispose()


@pytest.mark.integration
def test_turning_socials_on_seeds_the_four_templates_once_and_get_templates_lists_them(pg_engine, monkeypatch):
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: "owner")
    monkeypatch.setattr(socials_settings, "read_system_setting", lambda category, key: "true")
    with pg_engine.connect() as conn:
        trans = conn.begin()
        session = Session(bind=conn, join_transaction_mode="create_savepoint")
        try:
            conn.execute(sa.text("SET LOCAL lock_timeout = '5s'"))
            ws_id = uuid.uuid4()
            conn.execute(
                sa.text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), :name)"),
                {"id": str(ws_id), "name": f"prd251w1-seed-{str(ws_id)[:8]}"},
            )
            app = FastAPI()
            app.include_router(workspaces_api.router)
            app.include_router(documents_api.router)
            app.dependency_overrides[get_request_context_hybrid] = lambda: _ctx(ws_id)
            app.dependency_overrides[get_db] = lambda: session
            client = TestClient(app)

            def rows():
                return conn.execute(
                    sa.text(
                        "SELECT id, name, format, created_by, blocks FROM document_templates "
                        "WHERE workspace_id = CAST(:ws AS uuid) ORDER BY name"
                    ),
                    {"ws": str(ws_id)},
                ).all()

            assert rows() == []
            resp = client.put(SWITCH_ROUTE, json={"socials": {"enabled": True}})
            assert resp.status_code == 200, resp.text
            first = rows()
            assert sorted(r.name for r in first) == sorted(ALL_STARTER_NAMES)
            assert {(r.format, r.created_by) for r in first} == {("social_video", STARTER_CREATOR), ("social_image", STARTER_CREATOR)}
            by_name = {s["name"]: s for s in social_starters()}
            assert all(r.blocks == by_name[r.name]["blocks"] and r.format == by_name[r.name]["format"] for r in first)

            # Turning it on twice leaves the same rows.
            assert client.put(SWITCH_ROUTE, json={"socials": {"enabled": True}}).status_code == 200
            assert [r.id for r in rows()] == [r.id for r in first]

            listed = client.get("/api/documents/templates")
            assert listed.status_code == 200, listed.text
            entries = {entry["name"]: entry for entry in listed.json()}
            assert sorted(entries) == sorted(ALL_STARTER_NAMES)
            for name, entry in entries.items():
                assert entry["format"] == by_name[name]["format"] and entry["is_starter"] is True and entry["data_fields"]
            videos = client.get("/api/documents/templates", params={"format": "social_video"}).json()
            assert sorted(e["name"] for e in videos) == sorted(STARTER_NAMES)
            images = client.get("/api/documents/templates", params={"format": "social_image"}).json()
            assert sorted(e["name"] for e in images) == sorted(IMAGE_STARTER_NAMES)
        finally:
            session.close()
            trans.rollback()


# ---------------------------------------------------------------------------
# Footage slots (AC 4)
# ---------------------------------------------------------------------------


def test_an_empty_slot_falls_back_to_the_templates_own_motion_graphics(music_library):
    starter = _starter("cinematic-product-promo")
    bundle = _bundle(starter)
    page = bundle["composition"]["html"]
    assert "media" not in bundle
    assert "data-slot" not in page and "assets/slots/" not in page
    # The motion graphics the footage played over are all still there.
    for kept in ('id="bg"', 'class="bg-plane"', 'id="glowA"', 'id="s1"', 'id="s1head"', 'id="s9shade"', 'id="endhalo"'):
        assert kept in page, kept
    # The camera moves act on the (now empty) wrappers, so nothing targets a missing element.
    assert 'id="w2"' in page and 'id="w4"' in page
    parsed = parse_bundle(bundle, _renderer_settings(), music_library)
    assert parsed.media == () and parsed.composition.duration == 40.0


def test_a_filled_slot_takes_the_clip_and_keeps_its_element(music_library):
    starter = _starter("cinematic-product-promo")
    url = STORAGE + "social-media/ws/post/hook.mp4?X-Amz-Signature=abc"
    bundle = _bundle(starter, slot_media={"hook": url})
    page = bundle["composition"]["html"]
    assert bundle["media"] == [{"path": "assets/slots/hook.mp4", "url": url}]
    assert re.search(r'<video id="cv1"[^>]*data-slot="hook"[^>]*src="assets/slots/hook\.mp4"', page)
    assert page.count("data-slot=") == 1, "only the filled slot's element stays"
    parsed = parse_bundle(bundle, _renderer_settings(), music_library)
    assert [(m.path, m.url) for m in parsed.media] == [("assets/slots/hook.mp4", url)]
    with pytest.raises(ValueError, match="no slot"):
        _bundle(starter, slot_media={"b_roll": url})


@pytest.mark.parametrize("slug", list(REFERENCE_OF))
def test_media_render_takes_every_starter_with_its_slots_empty_or_filled(slug, music_library):
    starter = _starter(slug)
    slots = starter["blocks"].get("slots") or {}
    settings = _renderer_settings()
    for filled in ({}, {name: f"{STORAGE}{name}.bin?sig=1" for name in slots}):
        bundle = _bundle(starter, slot_media=filled)
        parsed = parse_bundle(bundle, settings, music_library)
        assert len(parsed.media) == len(filled)
        assert parsed.composition.width == 1080 and parsed.composition.height == 1920
        assert len(parsed.audio.voice.lines) == len(starter["blocks"]["audio_plan"]["voice"]["lines"])
        # US-112: the reference's own track, from the library, with its credit.
        cue = starter["blocks"]["audio_plan"]["music"]
        assert (parsed.audio.music.track, parsed.audio.music.start) == (cue["track"], cue["start"])
        assert parsed.audio.music.about["credit_required"] is True


def test_the_image_slots_of_the_data_story_are_stills():
    slots = _starter("data-story")["blocks"]["slots"]
    assert {name: spec["kind"] for name, spec in slots.items()} == {
        "hook": "video", "tide": "video", "still_1": "image", "still_2": "image", "still_3": "image", "end": "video",
    }
    page = _bundle(_starter("data-story"), slot_media={"still_2": STORAGE + "still.png"})["composition"]["html"]
    assert page.count("<img data-slot=") == 1 and 'src="assets/slots/still_2.png"' in page


PAGE = (
    '<!doctype html><html><head></head><body><div id="root" data-composition-id="main" data-duration="3">'
    "{body}"
    '<audio src="assets/audio/mix.wav"></audio></div></body></html>'
)
VIDEO_SLOT = {"hook": {"kind": "video", "label": "Hook", "path": "assets/slots/hook.mp4"}}


def _slot_blocks(body, slots=VIDEO_SLOT):
    return {"html": PAGE.format(body=body), "variables_schema": {}, "sizes": ["1080x1920"], "slots": slots}


@pytest.mark.parametrize(
    "body, slots, message",
    [
        pytest.param("", VIDEO_SLOT, "no element shows this slot", id="unused-slot"),
        pytest.param('<video data-slot="hook" src="assets/slots/other.mp4"></video>', VIDEO_SLOT, 'must show src="assets/slots/hook.mp4"', id="wrong-src"),
        pytest.param('<video data-slot="hook" src="assets/slots/hook.mp4"><track></video>', VIDEO_SLOT, "whole", id="children"),
        pytest.param('<div data-slot="hook"><video src="assets/slots/hook.mp4"></video></div>', VIDEO_SLOT, "whole", id="wrapper-marked"),
        pytest.param('<video data-slot="hook" src="assets/slots/hook.mp4"></video><img src="assets/slots/hook.mp4">', VIDEO_SLOT, "outside its data-slot", id="stray-reference"),
        pytest.param('<video data-slot="nope" src="x.mp4"></video>', {}, "names no slot", id="undeclared"),
        pytest.param('<video data-slot="hook" src="assets/slots/hook.gif"></video>',
                     {"hook": {"kind": "video", "path": "assets/slots/hook.gif"}}, "must be one of", id="bad-path"),
        pytest.param('<img data-slot="hook" src="assets/slots/hook.png">',
                     {"hook": {"kind": "gif", "path": "assets/slots/hook.png"}}, "kind", id="bad-kind"),
    ],
)
def test_the_contract_refuses_a_slot_the_removal_could_not_take_whole(body, slots, message):
    with pytest.raises(SocialTemplateError) as refused:
        validate_social_blocks(_slot_blocks(body, slots), "social_video")
    assert message in str(refused.value), str(refused.value)


def test_a_well_formed_slot_passes_and_is_removed_exactly():
    body = '<div class="vwrap"><video id="v" class="clip" data-slot="hook" src="assets/slots/hook.mp4" muted></video></div><p>kept</p>'
    blocks = validate_social_blocks(_slot_blocks(body), "social_video")
    stripped = without_slots(blocks["html"], blocks["slots"])
    assert '<div class="vwrap"></div><p>kept</p>' in stripped and "<video" not in stripped
    assert without_slots(blocks["html"], blocks["slots"], keep=["hook"]) == blocks["html"]


# ---------------------------------------------------------------------------
# Voice lines are template text
# ---------------------------------------------------------------------------


def test_the_voice_lines_are_filled_from_the_variables_and_the_brand():
    lines = _bundle(_starter("ui-story-promo"))["audio"]["voice"]["lines"]
    spoken = {line["id"]: line["text"] for line in lines}
    assert spoken["l01"] == "You clock off at six."
    assert spoken["l05"] == "The right agent takes it. Research. Writing. The website. The books."
    assert spoken["l10"] == "Automatos. Your business, running — even when you're not."
    assert all("{{" not in text for text in spoken.values())


def test_a_voice_line_that_fills_in_empty_is_dropped_and_an_undeclared_name_is_refused():
    blocks = {
        "html": PAGE.format(body="<p>{{ headline }}</p>"),
        "variables_schema": {"headline": {"type": "text"}, "aside": {"type": "text", "default": ""}},
        "sizes": ["1080x1920"],
        "audio_plan": {"voice": {"lines": [{"id": "l01", "at": 0.2, "text": "{{ headline }}"}, {"id": "l02", "at": 1.5, "text": " {{ aside }} "}]}},
    }
    checked = validate_social_blocks(blocks, "social_video")
    values = resolve_variables(checked["variables_schema"], {"headline": "Three weeks to go."}).values
    bundle = build_bundle(workspace_id=WS, reference="r", blocks=checked, values=values, brand_kit=KIT)
    assert bundle["audio"]["voice"]["lines"] == [{"id": "l01", "at": 0.2, "text": "Three weeks to go."}]
    blocks["audio_plan"]["voice"]["lines"][0]["text"] = "{{ tagline }}"
    with pytest.raises(SocialTemplateError, match="tagline"):
        validate_social_blocks(blocks, "social_video")


# ---------------------------------------------------------------------------
# The stage palette
# ---------------------------------------------------------------------------


def test_the_automatos_kit_reproduces_the_reference_palette():
    stage = stage_palette(KIT)
    assert set(stage) == set(STAGE_TOKENS)
    assert (stage["ink"], stage["on-ink"], stage["primary-on-ink"], stage["accent-on-ink"]) == ("#1a1714", "#f0e8db", "#e96235", "#90af5a")
    tokens = _bundle(_starter("ui-story-promo"))["brand"]["tokens"]
    assert {name: tokens[name] for name in STAGE_TOKENS} == stage


@pytest.mark.parametrize(
    "kit",
    [
        pytest.param({"primary_color": "#1a1a2e", "secondary_color": "#16213e", "accent_color": "#0f3460", "text_color": "#1a1a2e"}, id="the-document-defaults"),
        pytest.param({"primary_color": "#ffcc00", "secondary_color": "#f3ede2", "accent_color": "#00aa88", "text_color": "#222222"}, id="a-light-kit"),
        pytest.param({"primary_color": "#c8742c", "secondary_color": "#1f3b2d", "accent_color": "#f3ede2", "text_color": "#f3ede2"}, id="harbourline"),
    ],
)
def test_any_kit_gives_a_dark_stage_with_readable_text(kit):
    stage = {name: parse_hex(value) for name, value in stage_palette(kit).items()}
    ink = stage["ink"]
    assert luminance(ink) <= 0.025
    assert contrast(stage["on-ink"], ink) >= 10
    assert contrast(stage["on-ink-muted"], ink) >= 6.9 and contrast(stage["on-ink-dim"], ink) >= 5.3
    assert contrast(stage["primary-on-ink"], ink) >= 4.5 and contrast(stage["accent-on-ink"], ink) >= 4.5
    assert contrast(stage["primary-light"], ink) >= 6.0


def test_a_kit_without_a_usable_secondary_leaves_the_template_fallbacks():
    assert stage_palette({"secondary_color": "navy", "text_color": "#ffffff"}) == {}


# ---------------------------------------------------------------------------
# The CI driver
# ---------------------------------------------------------------------------


def _driver():
    spec = importlib.util.spec_from_file_location("social_template_previews", _ROOT / "scripts" / "ci" / "social_template_previews.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_ci_driver_builds_a_checked_preview_bundle_for_every_seeded_video(music_library):
    driver = _driver()
    kit = {**driver.KIT, "logo_url": driver.logo_png(driver.KIT["primary_color"])}
    settings = _renderer_settings()
    probed = []
    for starter in social_starters("social_video"):
        bundle = driver.bundle_for(starter, kit, starter["preview"]["at"])
        parsed = parse_bundle(bundle, settings, music_library)
        assert parsed.preview.at == tuple(sorted(starter["preview"]["at"]))
        assert [f.path for f in parsed.files] == ["assets/brand/logo.png"]
        probe = starter["preview"].get("probe")
        if probe:
            assert probe["at"] in starter["preview"]["at"] and probe["token"] in bundle["brand"]["tokens"]
            probed.append(starter["name"])
    assert probed == ["UI story promo", "Cinematic product promo"]


def test_the_ci_drivers_png_coder_round_trips():
    driver = _driver()
    data = driver.encode_png(6, 4, lambda x, y: (x * 40, y * 60, 200, 255))
    width, height, channels, rows = driver.decode_png(data)
    assert (width, height, channels) == (6, 4, 4)
    assert tuple(rows[3][5 * 4 : 5 * 4 + 3]) == (200, 180, 200)
    assert driver.sample(data, 1.0, 1.0, radius=0) == (200, 180, 200)
    assert driver.hex_rgb("#2f7bf6") == (47, 123, 246)


def test_the_media_render_job_checks_and_previews_every_seeded_template():
    import yaml

    workflow = yaml.safe_load((_ROOT / ".github" / "workflows" / "test.yml").read_text())
    job = workflow["jobs"]["media-render"]
    commands = "\n".join(step.get("run", "") for step in job["steps"])
    assert "PYTHONPATH=orchestrator python3 scripts/ci/social_template_previews.py" in commands
    assert "SOCIALS_RENDER_TOKEN=\"$TOKEN\"" in commands and "exit $code" in commands
    uploads = [step.get("with", {}).get("name") for step in job["steps"] if "upload-artifact" in str(step.get("uses"))]
    assert "media-render-template-previews" in uploads
