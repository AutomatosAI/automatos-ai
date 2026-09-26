"""PRD-251 Wave 1, US-105 (S1.2a) — social templates as data: the format, the blocks, the dispatch.

Pins:

* **The wave's one migration.** ``prd251_wave1`` is the only revision chained
  onto the base head ``f049_prd251_merge_heads``, and it and the model declare
  the same format list (the head pins in test_prd209 / test_prd236_w1_routes
  move with it). On real Postgres (``@integration``, rolled back): every model
  built by ``create_all`` first, then the upgrade run twice, and the CHECK takes
  ``social_image`` and ``social_video`` (the 89d89c250 lesson); a database still
  on the narrow CHECK gains them, and the downgrade keeps the social rows.
* **The contract, on save.** A social template's ``blocks`` are
  ``{html, css, variables_schema, sizes, audio_plan}``; one with a missing or
  malformed ``variables_schema`` is refused by the service and by the API (422
  naming each problem), and nothing is written.
* **The brand rule (D4).** No seeded social template hardcodes a hex colour, a
  font family or a logo outside a ``var()`` fallback, and the checker finds each
  kind (so that net is not vacuous).
* **The dispatch.** ``generate()`` sends social formats to ``generate_social``
  and pdf / docx / xlsx exactly where they went before. ``generate_social``
  builds the bundle (brand tokens as CSS variables, the logo and fonts inlined
  as data: URIs), checks the month's render minutes BEFORE media-render is
  called, renders through the real client over ``httpx.MockTransport``, books
  the seconds on the media lane and registers a video Deliverable. US-107: an
  image asks for one still and gets its PNG; a template that renders several
  (a carousel) is refused as a document before the quota is touched.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import sys
import uuid
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
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import httpx  # noqa: E402
import sqlalchemy as sa  # noqa: E402
from alembic.operations import Operations  # noqa: E402
from alembic.runtime.migration import MigrationContext  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.exc import IntegrityError  # noqa: E402

import core.models  # noqa: E402,F401  (registers every mapper)
import core.auth.workspace_permission as permission_mod  # noqa: E402
import core.media_render_client as media_render_client  # noqa: E402
import modules.documents.generation_service as generation_service  # noqa: E402
import services.deliverable_service as deliverable_service  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.database.database import Base, get_db  # noqa: E402
from core.brand_palette import PAPER_TOKENS, STAGE_TOKENS  # noqa: E402
from core.media_render_bundle import NO_LOGO, build_bundle  # noqa: E402
from core.media_render_client import MediaRenderClient, MediaRenderError  # noqa: E402
from core.media_render_quota import RenderQuotaExceeded  # noqa: E402
from core.models.core import DOCUMENT_TEMPLATE_FORMATS, SOCIAL_TEMPLATE_FORMATS, DocumentTemplate  # noqa: E402
from core.social_templates import (  # noqa: E402
    InvalidVariableValues,
    SocialTemplateError,
    brand_literals,
    is_social_format,
    resolve_variables,
    validate_social_blocks,
)
from modules.documents.generation_service import DocumentGenerationService  # noqa: E402
from modules.documents.models import GeneratedDocument, UnresolvedDeliverableError  # noqa: E402
from modules.documents.template_service import DocumentTemplateService, UnknownTemplateFormat  # noqa: E402
from modules.documents.template_summary import summarize_template  # noqa: E402

VERSIONS = _ORCH / "alembic" / "versions"
MIGRATION = VERSIONS / "prd251_wave1.py"
BASE_HEAD = "f049_prd251_merge_heads"
FORMAT_CHECK = "check_document_template_format"

WS = uuid.UUID("00000000-0000-0000-0000-0000000000b1")
TEMPLATE_ID = uuid.UUID("00000000-0000-0000-0000-0000000000c1")
RENDER_URL = "http://media-render:8090"
JOB_ID = "d" * 32
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"frame " * 300
PNG_URI = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg=="
WOFF2_URI = "data:font/woff2;base64,d09GMgABAAAAAA=="

HTML = (
    "<!doctype html><html><head><style>"
    "#root { background: var(--brand-secondary, #16213e); color: var(--brand-text, #f3efe6);"
    " font-family: var(--brand-body-font, sans-serif); }"
    ' h1 { font: 700 96px/1.1 var(--brand-heading-font, "Geist", sans-serif); }'
    " #bead { opacity: 0; }"
    "</style></head><body>"
    '<div id="root" data-composition-id="main" data-start="0" data-duration="3"'
    ' data-width="{{ size.width }}" data-height="{{ size.height }}">'
    '<img id="logo" src="{{ brand.logo }}" alt="{{ brand.name }}" />'
    '<h1 id="bead">{{ headline }}</h1><p>{{ stat }}</p><p>{{ subtitle }}</p>'
    '<audio id="mix" src="assets/audio/mix.wav" data-start="0" data-duration="3"></audio></div>'
    '<script>const tl = gsap.timeline({ paused: true }); tl.to("#bead", { opacity: 1 }, 0.2);'
    ' window.__timelines = { main: tl };</script>'
    "</body></html>"
)
BLOCKS = {
    "html": HTML,
    "css": "h1 { color: var(--brand-primary, #1a1a2e); }",
    "variables_schema": {
        "headline": {"type": "text", "label": "Headline", "max_chars": 60},
        "stat": {"type": "number", "claim": True, "min": 0},
        "subtitle": {"type": "text", "default": ""},
    },
    "sizes": ["1080x1920", "1080x1350"],
    "audio_plan": {"voice": {"voice": "af_heart", "lines": [{"id": "l01", "at": 0.3, "text": "Three weeks to go."}]}},
}


def _without(key):
    return {k: v for k, v in BLOCKS.items() if k != key}


def _ctx(workspace_id=WS):
    return RequestContext(
        workspace_id=workspace_id,
        user=UserContext(id="member-1", clerk_user_id="clerk-member-1", system_role="user"),
        auth_type="clerk",
    )


def _load_migration():
    spec = importlib.util.spec_from_file_location("prd251_wave1_migration", MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# The wave's one migration
# ---------------------------------------------------------------------------


def test_the_one_wave_revision_chains_onto_the_base_head():
    mod = _load_migration()
    assert (mod.revision, mod.down_revision) == ("prd251_wave1", BASE_HEAD)
    chained = [
        p.name
        for p in VERSIONS.glob("*.py")
        if re.search(rf"^down_revision\s*=\s*['\"]{BASE_HEAD}['\"]", p.read_text(encoding="utf-8"), re.M)
    ]
    # The night fixes' first revision (F155) chains onto the same base.
    assert sorted(chained) == ["f155_chats_widget_key_id.py", "prd251_wave1.py"]


def test_the_migration_and_the_model_declare_the_same_formats():
    mod = _load_migration()
    assert mod.FORMATS_AFTER == DOCUMENT_TEMPLATE_FORMATS
    assert mod.FORMATS_BEFORE + SOCIAL_TEMPLATE_FORMATS == DOCUMENT_TEMPLATE_FORMATS
    assert SOCIAL_TEMPLATE_FORMATS == ("social_image", "social_video")
    (check,) = [
        c for c in DocumentTemplate.__table__.constraints if getattr(c, "name", None) == FORMAT_CHECK
    ]
    assert str(check.sqltext) == mod.format_check(mod.FORMATS_AFTER)
    assert str(check.sqltext) == "format IN ('pdf', 'docx', 'xlsx', 'social_image', 'social_video')"
    # The step tolerates what create_all already built (89d89c250): drop IF EXISTS, then add.
    source = MIGRATION.read_text(encoding="utf-8")
    assert "DROP CONSTRAINT IF EXISTS" in source and "ADD CONSTRAINT" in source


def _run_migration(conn, *steps):
    """Run the revision's upgrade/downgrade on ``conn`` through a real alembic context."""
    mod = _load_migration()
    ctx = MigrationContext.configure(conn)
    with Operations.context(ctx):
        for step in steps:
            getattr(mod, step)()


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


def _format_check_definition(conn) -> str:
    return conn.execute(
        sa.text(
            "SELECT pg_get_constraintdef(c.oid) FROM pg_constraint c "
            "WHERE c.conname = :name AND c.conrelid = 'document_templates'::regclass"
        ),
        {"name": FORMAT_CHECK},
    ).scalar_one()


def _insert_workspace(conn) -> str:
    ws_id = str(uuid.uuid4())
    conn.execute(
        sa.text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), :name)"),
        {"id": ws_id, "name": f"prd251w1-{ws_id[:8]}"},
    )
    return ws_id


def _insert_template(conn, ws_id: str, fmt: str) -> str:
    template_id = str(uuid.uuid4())
    conn.execute(
        sa.text(
            "INSERT INTO document_templates (id, workspace_id, name, format) "
            "VALUES (CAST(:id AS uuid), CAST(:ws AS uuid), :name, :fmt)"
        ),
        {"id": template_id, "ws": ws_id, "name": f"tpl-{template_id[:8]}", "fmt": fmt},
    )
    return template_id


def _refused(conn, ws_id: str, fmt: str) -> bool:
    try:
        with conn.begin_nested():
            _insert_template(conn, ws_id, fmt)
    except IntegrityError:
        return True
    return False


@pytest.mark.integration
def test_create_all_first_then_the_upgrade_twice_and_the_check_takes_the_social_formats(pg_engine):
    """A backend that already loaded the new models runs create_all before the migration."""
    with pg_engine.connect() as conn:
        trans = conn.begin()
        try:
            conn.execute(sa.text("SET LOCAL lock_timeout = '5s'"))
            Base.metadata.create_all(bind=conn)
            _run_migration(conn, "upgrade")
            _run_migration(conn, "upgrade")

            definition = _format_check_definition(conn)
            assert "social_image" in definition and "social_video" in definition
            ws_id = _insert_workspace(conn)
            for fmt in ("social_image", "social_video", "pdf", "docx", "xlsx"):
                _insert_template(conn, ws_id, fmt)
            assert _refused(conn, ws_id, "gif")
        finally:
            trans.rollback()


@pytest.mark.integration
def test_a_database_on_the_narrow_check_gains_the_formats_and_the_downgrade_keeps_the_rows(pg_engine):
    mod = _load_migration()
    with pg_engine.connect() as conn:
        trans = conn.begin()
        try:
            conn.execute(sa.text("SET LOCAL lock_timeout = '5s'"))
            # Where a database migrated up to the base head stands: the narrow CHECK.
            conn.execute(sa.text(f"ALTER TABLE document_templates DROP CONSTRAINT IF EXISTS {FORMAT_CHECK}"))
            conn.execute(
                sa.text(
                    f"ALTER TABLE document_templates ADD CONSTRAINT {FORMAT_CHECK} "
                    f"CHECK ({mod.format_check(mod.FORMATS_BEFORE)})"
                )
            )
            ws_id = _insert_workspace(conn)
            assert _refused(conn, ws_id, "social_video")

            _run_migration(conn, "upgrade")
            kept = _insert_template(conn, ws_id, "social_video")

            # The downgrade deletes nothing: the social row stays, new rows follow the old rule.
            _run_migration(conn, "downgrade")
            left = conn.execute(
                sa.text("SELECT format FROM document_templates WHERE id = CAST(:id AS uuid)"), {"id": kept}
            ).scalar_one()
            assert left == "social_video"
            assert _refused(conn, ws_id, "social_image")
            _insert_template(conn, ws_id, "pdf")

            _run_migration(conn, "upgrade")
            assert "social_image" in _format_check_definition(conn)
        finally:
            trans.rollback()


# ---------------------------------------------------------------------------
# The contract, checked on save
# ---------------------------------------------------------------------------


def test_a_well_formed_social_template_passes_and_comes_back_as_a_copy():
    checked = validate_social_blocks(BLOCKS, "social_video")
    assert checked == BLOCKS and checked is not BLOCKS
    assert checked["variables_schema"] is not BLOCKS["variables_schema"]


@pytest.mark.parametrize(
    "schema, field",
    [
        pytest.param(None, "variables_schema", id="missing"),
        pytest.param(["headline", "stat"], "variables_schema", id="a-list"),
        pytest.param({"headline": "text"}, "variables_schema.headline", id="a-bare-type"),
        pytest.param({"headline": {"type": "paragraph"}}, "variables_schema.headline.type", id="unknown-type"),
        pytest.param({"headline": {"type": "text", "maxlength": 60}}, "variables_schema.headline.maxlength", id="unknown-setting"),
        pytest.param({"1headline": {"type": "text"}}, "variables_schema.1headline", id="bad-name"),
        pytest.param({"stat": {"type": "number", "default": "ten"}}, "variables_schema.stat.default", id="bad-default"),
        pytest.param({"stat": {"type": "number", "min": 5, "max": 1}}, "variables_schema.stat", id="min-over-max"),
        pytest.param({"headline": {"type": "text", "claim": "yes"}}, "variables_schema.headline.claim", id="bad-claim"),
    ],
)
def test_a_missing_or_malformed_variables_schema_is_refused(schema, field):
    blocks = _without("variables_schema") if schema is None else {**BLOCKS, "variables_schema": schema}
    with pytest.raises(SocialTemplateError) as refused:
        validate_social_blocks(blocks, "social_video")
    assert field in [e["field"] for e in refused.value.errors], refused.value.errors


@pytest.mark.parametrize(
    "blocks, field",
    [
        pytest.param({**BLOCKS, "html": HTML.replace("{{ subtitle }}", "{{ tagline }}")}, "html", id="undeclared-variable"),
        pytest.param({**BLOCKS, "html": HTML.replace("</head>", "")}, "html", id="no-head"),
        pytest.param({**BLOCKS, "sizes": []}, "sizes", id="no-sizes"),
        pytest.param({**BLOCKS, "sizes": ["1080 by 1920"]}, "sizes[0]", id="bad-size"),
        pytest.param({**BLOCKS, "sizes": ["1080x9999"]}, "sizes[0]", id="size-too-big"),
        pytest.param({**BLOCKS, "audio_plan": {"narration": {}}}, "audio_plan.narration", id="bad-audio-key"),
        pytest.param({**BLOCKS, "script": "…"}, "script", id="unknown-block"),
        pytest.param({**BLOCKS, "css": "h1 { color: #e96235; }"}, "css", id="hardcoded-colour"),
    ],
)
def test_the_rest_of_the_contract_is_checked_too(blocks, field):
    with pytest.raises(SocialTemplateError) as refused:
        validate_social_blocks(blocks, "social_video")
    assert field in [e["field"] for e in refused.value.errors], refused.value.errors


def test_an_image_template_carries_no_audio():
    with pytest.raises(SocialTemplateError) as refused:
        validate_social_blocks(BLOCKS, "social_image")
    assert [e["field"] for e in refused.value.errors] == ["audio_plan"]
    assert validate_social_blocks(_without("audio_plan"), "social_image")["sizes"] == BLOCKS["sizes"]


def test_the_service_refuses_a_social_template_without_a_good_variables_schema_and_writes_nothing():
    db = MagicMock()
    service = DocumentTemplateService(db)
    for blocks in (_without("variables_schema"), {**BLOCKS, "variables_schema": "headline, stat"}):
        with pytest.raises(SocialTemplateError) as refused:
            service.create_template(workspace_id=WS, name="Countdown", format="social_video", blocks=blocks)
        assert any(e["field"].startswith("variables_schema") for e in refused.value.errors)
    with pytest.raises(SocialTemplateError):
        service.create_template(workspace_id=WS, name="Countdown", format="social_image", blocks=None)
    with pytest.raises(UnknownTemplateFormat):
        service.create_template(workspace_id=WS, name="Countdown", format="gif")
    db.add.assert_not_called()
    db.commit.assert_not_called()

    created = service.create_template(workspace_id=WS, name="Countdown", format="social_video", blocks=BLOCKS)
    assert created.blocks == BLOCKS and created.blocks is not BLOCKS
    db.add.assert_called_once_with(created)


def test_an_update_to_a_social_template_is_checked_and_a_bad_one_changes_nothing():
    db = MagicMock()
    existing = DocumentTemplate(id=TEMPLATE_ID, workspace_id=WS, name="Countdown", format="social_video", blocks=BLOCKS)
    db.query.return_value.filter.return_value.first.return_value = existing
    service = DocumentTemplateService(db)

    with pytest.raises(SocialTemplateError):
        service.update_template(TEMPLATE_ID, WS, blocks={**BLOCKS, "variables_schema": {"headline": {"type": "rich"}}})
    assert existing.blocks == BLOCKS
    db.commit.assert_not_called()

    renamed = service.update_template(TEMPLATE_ID, WS, name="Countdown 2")
    assert renamed.name == "Countdown 2" and renamed.blocks == BLOCKS


@pytest.fixture
def documents_api(monkeypatch):
    import api.document_generation as documents_module

    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: "owner")
    db = MagicMock()
    app = FastAPI()
    app.include_router(documents_module.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: _ctx()
    app.dependency_overrides[get_db] = lambda: db
    return SimpleNamespace(client=TestClient(app), db=db)


def test_the_api_answers_422_naming_each_problem_and_saves_nothing(documents_api):
    cases = [
        (_without("variables_schema"), "variables_schema"),
        ({**BLOCKS, "variables_schema": {"headline": {"type": "paragraph"}}}, "variables_schema.headline.type"),
    ]
    for blocks, field in cases:
        resp = documents_api.client.post(
            "/api/documents/templates", json={"name": "Countdown", "format": "social_video", "blocks": blocks}
        )
        assert resp.status_code == 422, resp.text
        detail = resp.json()["detail"]
        assert detail["message"] == "Invalid template"
        assert field in [e["field"] for e in detail["errors"]]
    resp = documents_api.client.post("/api/documents/templates", json={"name": "Clip", "format": "gif"})
    assert resp.status_code == 422 and resp.json()["detail"]["errors"][0]["field"] == "format"
    documents_api.db.add.assert_not_called()


# ---------------------------------------------------------------------------
# The brand rule (D4)
# ---------------------------------------------------------------------------


class _SeedRecorder:
    """A session for the starter seeder: nothing exists yet, every row it adds is kept."""

    def __init__(self):
        self.rows = []

    def query(self, *args):
        return self

    def filter(self, *args):
        return self

    def first(self):
        return None

    def add(self, row):
        self.rows.append(row)

    def commit(self):
        pass


def _seeded_templates():
    """Every template a workspace is seeded with, ``(name, format, blocks)``, from each seed source."""
    from modules.documents import presets, seed_templates

    recorder = _SeedRecorder()
    seed_templates.seed_starter_templates(recorder, WS)
    # US-106: the social video starters, which turning Socials on seeds through the same path.
    seed_templates.seed_social_starters(recorder, WS)
    seeded = [(row.name, row.format, row.blocks) for row in recorder.rows]
    seeded += [(p["name"], p["format"], p.get("blocks")) for p in presets.PRESETS]
    seeded += [(t["name"], t["format"], t.get("blocks")) for t in seed_templates.STARTER_TEMPLATES]
    return seeded


def test_no_seeded_social_template_hardcodes_a_colour_a_font_or_a_logo():
    seeded = _seeded_templates()
    # The net runs the real starter seeders, so the social starters US-106 and US-107
    # seed through them are checked the moment they exist; the tests below prove it bites.
    assert seeded, "the starter seeder wrote nothing: this net is not attached to it"
    assert sum(1 for _, fmt, _ in seeded if is_social_format(fmt)) >= 4, "the social starters are not under this net"
    for name, fmt, blocks in seeded:
        if not is_social_format(fmt):
            continue
        blocks = blocks if isinstance(blocks, dict) else {}
        found = brand_literals(blocks.get("html") or "", blocks.get("css") or "")
        assert found == [], f"seeded template {name!r} hardcodes its brand: {found}"
        validate_social_blocks(blocks, fmt)


@pytest.mark.parametrize(
    "html, css, expected",
    [
        pytest.param("", "h1 { color: #e96235; }", "hex colour #e96235", id="css-hex"),
        pytest.param('<p style="background:#fff">x</p>', "", "hex colour #fff", id="inline-style-hex"),
        pytest.param('<svg><rect fill="#E96235"/></svg>', "", "hex colour #E96235", id="svg-fill-hex"),
        pytest.param('<script>tl.to("#bead", { color: "#e96235" });</script>', "", "hex colour #e96235", id="script-hex"),
        pytest.param(
            "", '@font-face { font-family: "Geist"; src: url("assets/fonts/Geist.woff2"); }',
            "font family 'Geist'", id="font-face",
        ),
        pytest.param("", '.m { font: 500 42px/1.2 "Geist Mono", monospace; }', "font family 'Geist Mono'", id="font-shorthand"),
        pytest.param('<svg><text font-family="Newsreader">x</text></svg>', "", "font family 'Newsreader'", id="svg-font"),
        pytest.param('<script>tl.set("#t", { fontFamily: "Geist" });</script>', "", "font family 'Geist'", id="script-font"),
        pytest.param('<img src="assets/brand/logo_wordmark_white.svg" alt="">', "", "bakes a logo", id="logo-file"),
        pytest.param('<img src="https://cdn.example.com/logo.png" alt="">', "", "is a URL", id="logo-url"),
        pytest.param("", ".x { left: {{ offset }}px; color: #000; }", "hex colour #000", id="rule-with-a-placeholder"),
        pytest.param("", ".g { background: linear-gradient(var(--brand-primary, #111), #222); }", "hex colour #222", id="outside-the-var"),
    ],
)
def test_the_brand_rule_finds_what_a_template_hardcodes(html, css, expected):
    found = " | ".join(message for _field, message in brand_literals(html, css))
    assert expected in found, found


def test_the_brand_rule_passes_what_reads_the_brand_kit():
    css = (
        'h1 { color: var(--brand-primary, #1a1a2e); font-family: var(--brand-heading-font, "Geist", sans-serif); }'
        " #bead { opacity: 0; } p { font-family: sans-serif; } em { font: inherit; }"
        " .g { background: linear-gradient(var(--brand-primary, #111), var(--brand-accent, #222)); }"
        ' .n { background-image: url("data:image/svg+xml;utf8,<svg xmlns=\'http://www.w3.org/2000/svg\'>'
        "<rect fill='%23000'/></svg>\"); }"
    )
    html = (
        '<img src="{{ brand.logo }}" alt=""><a href="#main">x</a><audio src="assets/audio/mix.wav"></audio>'
        '<script>tl.to("#bead", { opacity: 1 });</script>'
    )
    assert brand_literals(html, css) == []
    assert brand_literals(HTML, BLOCKS["css"]) == []


# ---------------------------------------------------------------------------
# The bundle
# ---------------------------------------------------------------------------

KIT = {
    "name": "Acme",
    "tagline": "Build better",
    "primary_color": "#ff0000",
    "secondary_color": "#00ff00",
    "accent_color": "#0000ff",
    "text_color": "#111111",
    "font_family": "Inter, sans-serif",
    "logo_url": PNG_URI,
    "font_files": [{"family": "Geist", "weight": 700, "style": "normal", "data_uri": WOFF2_URI}],
}


def test_the_bundle_carries_the_brand_kit_as_tokens_inlined_files_and_variables():
    values = resolve_variables(BLOCKS["variables_schema"], {"headline": "Hi", "stat": 3}).values
    bundle = build_bundle(workspace_id=WS, reference="document_template:x", blocks=BLOCKS, values=values, brand_kit=KIT)
    assert bundle["composition"] == {"html": HTML, "css": BLOCKS["css"]}
    tokens = bundle["brand"]["tokens"]
    raw = ("primary", "secondary", "accent", "text", "body-font", "heading-font")
    assert {name: tokens[name] for name in raw} == {
        "primary": "#ff0000", "secondary": "#00ff00", "accent": "#0000ff", "text": "#111111",
        "body-font": "Inter, sans-serif", "heading-font": "Inter, sans-serif",
    }
    # US-106: and the dark stage a social video reads, derived from those colours (core/brand_palette.py);
    # US-107: and the paper a social image reads.
    assert set(tokens) - set(raw) == set(STAGE_TOKENS) | set(PAPER_TOKENS)
    assert bundle["brand"]["fonts"] == [
        {"family": "Geist", "weight": "700", "style": "normal", "path": "assets/brand/fonts/font-0.woff2"}
    ]
    assert bundle["files"] == [
        {"path": "assets/brand/logo.png", "data_uri": PNG_URI},
        {"path": "assets/brand/fonts/font-0.woff2", "data_uri": WOFF2_URI},
    ]
    assert bundle["variables"] == {
        "headline": "Hi", "stat": 3, "subtitle": "",
        "brand.name": "Acme", "brand.tagline": "Build better", "brand.logo": "assets/brand/logo.png",
        # US-108: the square mark, which without an uploaded mark is the logo.
        "brand.logo_mark": "assets/brand/logo.png",
        "size.width": 1080, "size.height": 1920,
    }
    assert bundle["audio"] == BLOCKS["audio_plan"]

    square = build_bundle(workspace_id=WS, reference="r", blocks=BLOCKS, values=values, brand_kit=KIT, size="1080x1350")
    assert (square["variables"]["size.width"], square["variables"]["size.height"]) == (1080, 1350)
    with pytest.raises(ValueError):
        build_bundle(workspace_id=WS, reference="r", blocks=BLOCKS, values=values, brand_kit=KIT, size="640x640")


def test_what_the_bundle_leaves_out_of_the_brand():
    kit = {
        **KIT,
        "name": "",
        "company": {"name": "Acme Ltd"},
        "logo_url": "https://example.com/logo.png",  # never fetched: a logo is uploaded to reach a render
        "font_family": "Inter; } body { color: red",  # not one CSS value: the template's fallback applies
        "heading_font": "Geist, sans-serif",
        "font_files": [{"family": "Geist", "data_uri": "data:application/zip;base64,UEsDBA=="}],
    }
    bundle = build_bundle(workspace_id=WS, reference="r", blocks=BLOCKS, values={}, brand_kit=kit)
    assert "files" not in bundle and "fonts" not in bundle["brand"]
    assert bundle["variables"]["brand.logo"] == NO_LOGO
    assert bundle["variables"]["brand.name"] == "Acme Ltd"
    assert "body-font" not in bundle["brand"]["tokens"]
    assert bundle["brand"]["tokens"]["heading-font"] == "Geist, sans-serif"
    unnamed = build_bundle(
        workspace_id=WS, reference="r", blocks=BLOCKS, values={}, brand_kit={}, fallback_name="Workspace One"
    )
    assert unnamed["variables"]["brand.name"] == "Workspace One"


def test_variables_take_their_defaults_and_a_number_counts_as_text():
    resolved = resolve_variables(BLOCKS["variables_schema"], {"headline": 42, "stat": -1, "extra": "ignored"})
    assert resolved.values == {"headline": "42", "subtitle": ""}
    assert resolved.missing == [] and resolved.invalid == ["stat must be at least 0"]
    assert resolve_variables(BLOCKS["variables_schema"], {}).missing == ["headline", "stat"]


# ---------------------------------------------------------------------------
# The dispatch: generate() → generate_social → media-render
# ---------------------------------------------------------------------------


def _social_template(fmt="social_video", blocks=BLOCKS):
    return SimpleNamespace(id=TEMPLATE_ID, name="Countdown", format=fmt, blocks=blocks)


def _service_with(template):
    service = DocumentGenerationService(MagicMock(), WS)
    service.template_service = SimpleNamespace(
        get_template=lambda template_id, workspace_id: template,
        get_template_by_name=lambda workspace_id, name: None,
    )
    return service


def test_generate_sends_the_social_formats_to_media_render_and_the_documents_where_they_went(monkeypatch):
    template = _social_template()
    service = _service_with(template)
    calls = []

    def lane(name):
        async def run(*args, **kwargs):
            calls.append((name, args, kwargs))
            return GeneratedDocument(path="p", format=name, filename=f"f.{name}", size=1)

        return run

    for name in ("social", "pdf", "docx", "xlsx"):
        monkeypatch.setattr(service, f"generate_{name}", lane(name))

    for fmt, expected in (
        ("social_video", "social"), ("social_image", "social"), ("pdf", "pdf"), ("docx", "docx"), ("xlsx", "xlsx"),
    ):
        calls.clear()
        template_id = TEMPLATE_ID if is_social_format(fmt) else None
        asyncio.run(
            service.generate(title="T", format=fmt, data={"columns": ["a"]}, workspace_id=WS, template_id=template_id, user_id=7)
        )
        assert [c[0] for c in calls] == [expected], fmt
        name, args, kwargs = calls[0]
        if name == "social":
            assert args[0] is template and args[2] == WS and kwargs == {"format": fmt}
        elif name in ("pdf", "docx"):
            assert args[0] is None and args[2:] == (WS, "T") and kwargs == {"user_id": 7}
        else:
            assert kwargs == {"title": "T", "template": None}

    with pytest.raises(ValueError, match="Unsupported format"):
        asyncio.run(service.generate(title="T", format="gif", data={}, workspace_id=WS))


class _Renderer:
    """media-render over httpx.MockTransport: the real client talks to it."""

    def __init__(self, events, *, status="done", error=None, output=None, content=MP4):
        self.events = events
        self.status = status
        self.error = error
        self.output = output or {"name": "render.mp4", "aspect": "9:16", "width": 1080, "height": 1920, "duration": 12.5}
        self.content = content
        self.bundles = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.events.append(f"{request.method} {path}")
        if request.method == "POST" and path == "/render":
            self.bundles.append(json.loads(request.content))
            return httpx.Response(202, json={"id": JOB_ID, "status": "rendering", "outputs": [], "report": {}})
        if request.method == "GET" and path == f"/render/{JOB_ID}":
            done = self.status == "done"
            return httpx.Response(
                200,
                json={"id": JOB_ID, "status": self.status, "outputs": [self.output] if done else [], "report": {}, "error": self.error},
            )
        if request.method == "GET" and path == f"/render/{JOB_ID}/output/{self.output['name']}":
            return httpx.Response(200, content=self.content)
        return httpx.Response(404, json={"error": "not_found", "message": f"no route {path}"})


@pytest.fixture
def social_env(monkeypatch, tmp_path):
    configs = {id(m.config): m.config for m in (generation_service, media_render_client)}
    for cfg in configs.values():
        monkeypatch.setattr(cfg, "SOCIALS_RENDER_URL", RENDER_URL, raising=False)
        monkeypatch.setattr(cfg, "SOCIALS_RENDER_TOKEN", "", raising=False)
        monkeypatch.setattr(cfg, "SOCIALS_RENDER_POLL_SECONDS", 0, raising=False)
        monkeypatch.setattr(cfg, "SOCIALS_RENDER_MAX_WAIT_SECONDS", 60, raising=False)
    monkeypatch.setattr(generation_service, "GENERATED_DIR", str(tmp_path))
    monkeypatch.setattr(generation_service, "is_storage_configured", lambda: False)
    # An uploaded logo, inlined as the render-ready kit carries it (US-108: brand_fonts.brand_kit_for_media_render).
    monkeypatch.setattr(generation_service, "brand_kit_for_media_render", lambda kit: {**kit, "logo_url": PNG_URI})

    state = SimpleNamespace(events=[], booked=[])

    def enforce(db, workspace):
        state.events.append(f"quota {workspace.plan}")

    def book(**kwargs):
        state.booked.append(kwargs)

    monkeypatch.setattr(generation_service, "enforce_render_quota", enforce)
    monkeypatch.setattr(generation_service, "book_render_seconds", book)
    workspace = SimpleNamespace(
        id=WS, name="Workspace One", plan="basic", plan_limits={},
        settings={"brand_kit": {"name": "Acme", "primary_color": "#ff0000", "font_family": "Inter, sans-serif"}},
    )
    state.workspace = workspace
    return state


def _generate(service, renderer, **kwargs):
    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(renderer.handler)) as http:
            service._render_client = lambda: MediaRenderClient(http)
            return await service.generate(**kwargs)

    return asyncio.run(go())


def test_generate_social_renders_the_template_with_the_brand_kit_through_media_render(social_env):
    service = _service_with(_social_template())
    service.db.query.return_value.filter.return_value.first.return_value = social_env.workspace
    renderer = _Renderer(social_env.events)

    result = _generate(
        service, renderer, title="Countdown", format="social_video",
        data={"headline": "Three weeks to go", "stat": 21}, workspace_id=WS, template_id=TEMPLATE_ID,
    )

    # The month's minutes are checked before anything reaches media-render.
    assert social_env.events[:2] == ["quota basic", "POST /render"]
    (bundle,) = renderer.bundles
    assert bundle["workspace_id"] == str(WS) and bundle["reference"] == f"document_template:{TEMPLATE_ID}"
    assert bundle["composition"] == {"html": HTML, "css": BLOCKS["css"]}
    assert bundle["brand"]["tokens"]["primary"] == "#ff0000"
    assert bundle["brand"]["tokens"]["body-font"] == "Inter, sans-serif"
    assert bundle["files"] == [{"path": "assets/brand/logo.png", "data_uri": PNG_URI}]
    assert bundle["variables"]["brand.logo"] == "assets/brand/logo.png"
    assert bundle["variables"]["brand.name"] == "Acme"
    assert (bundle["variables"]["headline"], bundle["variables"]["stat"]) == ("Three weeks to go", 21)
    assert bundle["variables"]["subtitle"] == ""
    assert bundle["audio"] == BLOCKS["audio_plan"]

    # The MP4 is the file generated; the seconds are booked on the media lane.
    assert result.format == "mp4" and result.template_lane == "social"
    assert Path(result.path).read_bytes() == MP4 and result.size == len(MP4)
    assert result.download_url == f"/api/documents/generated/{result.filename}"
    assert result.template_id == str(TEMPLATE_ID) and result.template_name == "Countdown"
    (booking,) = social_env.booked
    assert booking["workspace_id"] == WS and booking["seconds"] == 12.5
    assert booking["execution_id"] == f"document_template:{TEMPLATE_ID}"


def test_a_rendered_social_file_is_a_video_deliverable(monkeypatch):
    registered = []

    class Deliverables:
        def __init__(self, db, workspace_id):
            pass

        def register(self, **kwargs):
            registered.append(kwargs)
            return {"success": True, "deliverable_id": "d-1"}

    monkeypatch.setattr(deliverable_service, "DeliverableService", Deliverables)
    service = DocumentGenerationService(MagicMock(), WS)
    video = GeneratedDocument(path="p", format="mp4", filename="countdown.mp4", size=9, template_lane="social")
    pdf = GeneratedDocument(path="p", format="pdf", filename="report.pdf", size=9, template_lane="block")
    service.register_as_deliverable(video, title="Countdown")
    service.register_as_deliverable(pdf, title="Report")
    assert [r["artifact_type"] for r in registered] == ["video", "document"]
    assert [r["file_type"] for r in registered] == ["mp4", "pdf"]


def test_a_variable_without_a_value_or_a_default_blocks_before_anything_renders(social_env):
    service = _service_with(_social_template())
    service.db.query.return_value.filter.return_value.first.return_value = social_env.workspace
    renderer = _Renderer(social_env.events)
    with pytest.raises(UnresolvedDeliverableError) as blocked:
        _generate(service, renderer, title="T", format="social_video", data={}, workspace_id=WS, template_id=TEMPLATE_ID)
    assert blocked.value.unresolved == ["data.headline", "data.stat"]
    assert social_env.events == [] and social_env.booked == []


def test_a_value_that_does_not_fit_its_variable_is_refused_naming_it(social_env):
    service = _service_with(_social_template())
    service.db.query.return_value.filter.return_value.first.return_value = social_env.workspace
    renderer = _Renderer(social_env.events)
    with pytest.raises(InvalidVariableValues) as refused:
        _generate(
            service, renderer, title="T", format="social_video",
            data={"headline": "H", "stat": -1}, workspace_id=WS, template_id=TEMPLATE_ID,
        )
    assert refused.value.errors == [{"field": "data", "message": "stat must be at least 0"}]
    assert social_env.events == [] and social_env.booked == []


def test_a_social_format_needs_a_template_of_that_format(social_env):
    renderer = _Renderer(social_env.events)
    for template in (None, _social_template(fmt="social_image")):
        service = _service_with(template)
        with pytest.raises(ValueError, match="social_video renders a social_video template"):
            _generate(service, renderer, title="T", format="social_video", data={}, workspace_id=WS, template_id=TEMPLATE_ID)
    assert social_env.events == []


def test_a_render_past_the_quota_never_reaches_media_render(social_env, monkeypatch):
    def exhausted(db, workspace):
        raise RenderQuotaExceeded("This workspace has used 10.0 of its 10 render minutes this month.")

    monkeypatch.setattr(generation_service, "enforce_render_quota", exhausted)
    service = _service_with(_social_template())
    service.db.query.return_value.filter.return_value.first.return_value = social_env.workspace
    renderer = _Renderer(social_env.events)
    with pytest.raises(RenderQuotaExceeded):
        _generate(
            service, renderer, title="T", format="social_video",
            data={"headline": "H", "stat": 1}, workspace_id=WS, template_id=TEMPLATE_ID,
        )
    assert social_env.events == [] and social_env.booked == []


def test_a_render_the_renderer_fails_raises_its_code_and_books_nothing(social_env):
    service = _service_with(_social_template())
    service.db.query.return_value.filter.return_value.first.return_value = social_env.workspace
    renderer = _Renderer(social_env.events, status="failed", error={"code": "render_timed_out", "message": "past 900 s"})
    with pytest.raises(MediaRenderError) as failed:
        _generate(
            service, renderer, title="T", format="social_video",
            data={"headline": "H", "stat": 1}, workspace_id=WS, template_id=TEMPLATE_ID,
        )
    assert failed.value.code == "render_timed_out" and str(failed.value) == "past 900 s"
    assert social_env.booked == []


# ---------------------------------------------------------------------------
# What callers see of a social template
# ---------------------------------------------------------------------------


def test_the_gallery_lists_a_social_template_by_its_variables_never_as_block_editable():
    row = SimpleNamespace(id=TEMPLATE_ID, name="Countdown", format="social_video", blocks=BLOCKS, created_by="system")
    summary = summarize_template(row)
    assert summary["has_blocks"] is False and summary["is_starter"] is True
    assert summary["data_fields"] == ["headline", "stat", "subtitle"]
    assert summary["variable_paths"] == ["data.headline", "data.stat", "data.subtitle"]
    assert summary["list_fields"] == []


def test_an_agent_reads_a_social_templates_variables():
    from modules.tools.discovery.handlers_documents import get_template_schema

    db = MagicMock()
    row = DocumentTemplate(id=TEMPLATE_ID, workspace_id=WS, name="Countdown", format="social_video", blocks=BLOCKS)
    db.query.return_value.filter.return_value.first.return_value = row
    schema = asyncio.run(get_template_schema(db, WS, {"template_id": str(TEMPLATE_ID)}))
    assert schema["success"] is True and schema["uses_blocks"] is False
    assert schema["data_fields"] == ["data.headline", "data.stat", "data.subtitle"]
    assert schema["variables_schema"] == BLOCKS["variables_schema"]
    assert schema["sizes"] == BLOCKS["sizes"]


# ---------------------------------------------------------------------------
# US-107: an image is one still; a carousel is not a document
# ---------------------------------------------------------------------------

PNG = b"\x89PNG\r\n\x1a\n" + b"pixels " * 80
IMAGE_HTML = HTML.replace('data-duration="3"', 'data-duration="1"')
IMAGE_BLOCKS = {
    "html": IMAGE_HTML,
    "css": BLOCKS["css"],
    "variables_schema": BLOCKS["variables_schema"],
    "sizes": ["1080x1350", "1200x628"],
}


def _starter_template(slug):
    from modules.documents.social_starters import social_starters

    (starter,) = [s for s in social_starters() if s["slug"] == slug]
    return SimpleNamespace(id=TEMPLATE_ID, name=starter["name"], format=starter["format"], blocks=starter["blocks"])


def test_generate_social_renders_an_image_as_one_still_and_keeps_its_png(social_env):
    service = _service_with(_social_template(fmt="social_image", blocks=IMAGE_BLOCKS))
    service.db.query.return_value.filter.return_value.first.return_value = social_env.workspace
    still = {"name": "render.png", "kind": "still", "index": 1, "at": 0.0, "aspect": "4:5", "width": 1080, "height": 1350}
    renderer = _Renderer(social_env.events, output=still, content=PNG)

    result = _generate(
        service, renderer, title="Countdown card", format="social_image",
        data={"headline": "Three weeks to go", "stat": 21}, workspace_id=WS, template_id=TEMPLATE_ID,
    )

    assert social_env.events[:2] == ["quota basic", "POST /render"]
    (bundle,) = renderer.bundles
    # One still at the template's first moment (it declares none: 0 s), no sound, at the first size.
    assert bundle["still"] == {"at": [0.0]}
    assert "audio" not in bundle and "preview" not in bundle
    assert (bundle["variables"]["size.width"], bundle["variables"]["size.height"]) == (1080, 1350)
    assert result.format == "png" and Path(result.path).read_bytes() == PNG
    # A still spends no render minutes.
    (booking,) = social_env.booked
    assert booking["seconds"] == 0.0


def test_a_carousel_is_refused_as_a_document_before_the_quota_is_touched(social_env):
    template = _starter_template("carousel")
    service = _service_with(template)
    service.db.query.return_value.filter.return_value.first.return_value = social_env.workspace
    renderer = _Renderer(social_env.events)
    data = {"headline": "4 SIGNS", "point_1_title": "One", "point_2_title": "Two", "closing_title": "Done."}
    with pytest.raises(ValueError, match="renders 4 images, one per slide.*Socials post"):
        _generate(service, renderer, title="T", format="social_image", data=data, workspace_id=WS, template_id=TEMPLATE_ID)
    assert social_env.events == [] and social_env.booked == [] and renderer.bundles == []

