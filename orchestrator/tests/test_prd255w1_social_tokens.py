"""PRD-255 Wave 1, US-006 — socials read the same tokens as the kit's documents.

Pins (``core/media_render_bundle.brand_tokens``, ``core/brand_palette.paper_palette``,
the 18 seeded social templates, and the media-render CI driver):

* **The v2 roles reach the paper.** A role the kit stores maps onto the paper token
  it plays (paper → ``paper``, ink → ``on-paper``, muted → ``on-paper-muted``,
  accent → ``primary-on-paper`` / ``-large`` / ``accent-on-paper``, accent_2 →
  ``accent-on-paper``) when it meets that token's own target; else, and for every
  v1 kit, today's derivation. Every token name the base emitted is still emitted,
  and media-render's own parser accepts the bundle.
* **The type scale reaches the templates.** ``display-scale`` / ``body-scale`` are
  the kit's display and body sizes over the default (a ratio, bounded), and every
  font size in every seeded template is multiplied by one of them, with a
  fallback of 1: a template without the token renders as it always did.
* **The dark logo (FR-9).** An uploaded logo for dark backgrounds is staged and is
  what ``{{ brand.logo_on_dark }}`` fills, with nothing behind it; without one,
  the mark on a light chip in the paper's colour. It is never generated: an
  external URL is never fetched. Every dark stage (the four videos, the six
  photo cards) shows it; the paper cards show the mark.
* **Sparing.** The accent is never the background of a text-heavy card's surfaces
  (quote, definition, fact, stats, data story, carousel) under ``accent_use:
  sparing``, the default for every kit.
* **The CI driver renders with both night kits** (Automatos #c44a1a as its v1 kit
  has it, Harbourline #1e3a5f as a v2 kit): every seeded template, each bundle
  checked against the kit before it is sent, Harbourline's Title card page read
  back from its PNG as the stored paper.
"""
from __future__ import annotations

import base64
import importlib.util
import re
import sys
from pathlib import Path
from uuid import UUID

import pytest

_ORCH = Path(__file__).resolve().parents[1]
_ROOT = _ORCH.parent
_MEDIA_RENDER = _ROOT / "services" / "media-render"
if str(_MEDIA_RENDER) not in sys.path:
    sys.path.append(str(_MEDIA_RENDER))

import modules.documents.brand_logo as bl  # noqa: E402
from core.brand_palette import (  # noqa: E402
    ACCENT_ON_PAPER,
    CARD_CONTRAST,
    LARGE_TEXT_MIN_CONTRAST,
    MUTED_CONTRAST,
    ON_PAPER,
    ON_PAPER_MIN_CONTRAST,
    ON_PAPER_MUTED,
    PAPER,
    PAPER_CARD,
    PAPER_TEXT_MIN_CONTRAST,
    PAPER_TOKENS,
    PRIMARY_ON_PAPER,
    PRIMARY_ON_PAPER_LARGE,
    STAGE_TOKENS,
    contrast,
    paper_palette,
    parse_hex,
)
from core.media_render_bundle import (  # noqa: E402
    LOGO_CHIP_CLEAR,
    MAX_SOCIAL_TYPE_SCALE,
    MIN_SOCIAL_TYPE_SCALE,
    NO_LOGO,
    brand_tokens,
    build_bundle,
    type_scale_tokens,
)
from core.social_templates import SOCIAL_IMAGE, SOCIAL_VIDEO, resolve_variables  # noqa: E402
from media_render.bundle import parse_bundle  # noqa: E402  (media-render's own parser: stdlib only)
from media_render.config import load_settings  # noqa: E402
from modules.documents.brand_fonts import brand_kit_for_media_render  # noqa: E402
from modules.documents.brand_kit import get_brand_kit  # noqa: E402
from modules.documents.social_starters import social_starters  # noqa: E402

WS = UUID("6d0b5c1e-8f1a-4c2b-9d3e-0a1b2c3d4e66")
# The US-003 fixtures: Automatos as its v1 kit has it; Harbourline with its v2 rules stored.
AUTOMATOS = {"name": "Automatos", "primary_color": "#c44a1a", "secondary_color": "#1d3658",
             "accent_color": "#0f3460", "text_color": "#1a1a2e"}
HARBOURLINE_PAPER, HARBOURLINE_ACCENT = "#faf7f2", "#1e3a5f"
HARBOURLINE = {"name": "Harbourline Coffee Roasters", "primary_color": "#1E3A5F", "secondary_color": "#C26A2E",
               "accent_color": "#0f3460", "text_color": "#1a1a2e", "accent_use": "sparing",
               "palette": {"paper": HARBOURLINE_PAPER, "accent": HARBOURLINE_ACCENT},
               "type_scale": {"display": {"size_pt": 36}, "body": {"size_pt": 11}}}
# Every token name brand_tokens emitted on the base (PRD-251): all 18 templates read them.
BASE_TOKENS = {"primary", "secondary", "accent", "text", "body-font", "heading-font"} | set(STAGE_TOKENS) | set(PAPER_TOKENS)
DARK_STAGES = {"app-promo", "cinematic-product-promo", "data-story", "ui-story-promo",
               "photo-headline", "photo-highlights", "photo-offer", "photo-only", "photo-review", "before-after"}
TEXT_HEAVY = ("quote-card", "definition-card", "fact-card", "stats-card", "data-story", "carousel")
# A text-heavy card's surfaces: its page, its cards and panels, its pills.
SURFACES = {"html", "body", "#root", ".clip", ".page", ".main", ".card", ".note", ".bg-base", ".glass",
            ".cap-in", ".pill", ".cta-pill", ".eyebrow", ".panel"}
ACCENT_TOKENS = {"brand-primary", "brand-primary-on-paper", "brand-primary-on-paper-large", "brand-accent-on-paper",
                 "brand-primary-on-ink", "brand-primary-light", "brand-accent-on-ink", "brand-accent"}


def _png(width: int = 400, height: int = 100) -> bytes:
    """A PNG signature and IHDR: enough to sniff and size."""
    ihdr = width.to_bytes(4, "big") + height.to_bytes(4, "big") + b"\x08\x06\x00\x00\x00"
    return b"\x89PNG\r\n\x1a\n" + (13).to_bytes(4, "big") + b"IHDR" + ihdr + b"\x00" * 16


def _uri(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _bundle(kit, slug="title-card"):
    starter = next(s for s in social_starters() if s["slug"] == slug)
    values = resolve_variables(starter["blocks"]["variables_schema"], starter["sample_data"]).values
    fmt = SOCIAL_IMAGE if starter["format"] == SOCIAL_IMAGE else SOCIAL_VIDEO
    return build_bundle(workspace_id=WS, reference="r", blocks=starter["blocks"], values=values, brand_kit=kit, fmt=fmt)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── the v2 roles on the paper ────────────────────────────────────────────────

def test_every_base_token_is_still_emitted_and_media_render_accepts_the_bundle():
    for kit in (AUTOMATOS, HARBOURLINE):
        tokens = brand_tokens({**kit, "font_family": "Inter, sans-serif"})
        assert BASE_TOKENS <= set(tokens)
        assert {"display-scale", "body-scale", "logo-chip"} <= set(tokens)
        parse_bundle(_bundle(kit), load_settings({}), {})  # the tokens are single CSS values, within the cap


def test_a_kit_without_stored_roles_keeps_todays_paper():
    # Automatos is a v1 kit: its paper is derived exactly as before, and an empty palette changes nothing.
    assert paper_palette(AUTOMATOS) == paper_palette({**AUTOMATOS, "palette": {}})
    assert paper_palette(AUTOMATOS) == paper_palette({**AUTOMATOS, "palette": {"paper": "", "accent": None}})
    assert parse_hex(paper_palette(AUTOMATOS)[PAPER]) != parse_hex(HARBOURLINE_PAPER)


def test_harbourlines_stored_paper_and_accent_are_its_paper_tokens():
    tokens = brand_tokens(HARBOURLINE)
    assert tokens[PAPER] == HARBOURLINE_PAPER
    paper, card = parse_hex(tokens[PAPER]), parse_hex(tokens[PAPER_CARD])
    assert card != paper and contrast(card, paper) <= CARD_CONTRAST  # the card is derived from the stored paper
    assert tokens[PRIMARY_ON_PAPER] == tokens[PRIMARY_ON_PAPER_LARGE] == tokens[ACCENT_ON_PAPER] == HARBOURLINE_ACCENT
    # No ink stored: on-paper is derived, against the stored paper's card, and reads at 10:1.
    assert contrast(parse_hex(tokens[ON_PAPER]), card) >= ON_PAPER_MIN_CONTRAST


def test_stored_ink_muted_and_accent_2_take_their_tokens():
    kit = {**HARBOURLINE, "palette": {**HARBOURLINE["palette"], "ink": "#141414", "muted": "#3d3d3d", "accent_2": "#7a3b12"}}
    tokens = paper_palette(kit)
    assert (tokens[ON_PAPER], tokens[ON_PAPER_MUTED], tokens[ACCENT_ON_PAPER]) == ("#141414", "#3d3d3d", "#7a3b12")
    assert tokens[PRIMARY_ON_PAPER] == HARBOURLINE_ACCENT


def test_a_stored_role_that_would_not_read_there_is_left_to_todays_derivation():
    v1 = paper_palette(AUTOMATOS)
    # A paper too dark to be a page, an accent too pale for text on it, a muted tone too faint.
    weak = paper_palette({**AUTOMATOS, "palette": {"paper": "#808080", "accent": "#f3d9cf", "muted": "#d0d0d0"}})
    assert weak == v1
    # Large text needs less: a mid accent is the large token and not the small one.
    mid = "#b8491f"
    card = parse_hex(v1[PAPER_CARD])
    assert LARGE_TEXT_MIN_CONTRAST <= contrast(parse_hex(mid), card) < PAPER_TEXT_MIN_CONTRAST
    tokens = paper_palette({**AUTOMATOS, "palette": {"accent": mid}})
    assert tokens[PRIMARY_ON_PAPER_LARGE] == mid and tokens[PRIMARY_ON_PAPER] == v1[PRIMARY_ON_PAPER]
    assert contrast(parse_hex(tokens[ON_PAPER_MUTED]), card) >= MUTED_CONTRAST


# ── the type scale ────────────────────────────────────────────────────────────

def test_the_type_scale_is_a_ratio_to_the_default_and_bounded():
    assert type_scale_tokens(AUTOMATOS) == {"display-scale": "1", "body-scale": "1"}
    assert type_scale_tokens(HARBOURLINE) == {"display-scale": "1.125", "body-scale": "1.1"}
    huge = {"type_scale": {"display": {"size_pt": 96}, "body": {"size_pt": 5}}}
    assert type_scale_tokens(huge) == {"display-scale": f"{MAX_SOCIAL_TYPE_SCALE:g}", "body-scale": f"{MIN_SOCIAL_TYPE_SCALE:g}"}
    junk = {"type_scale": {"display": {"size_pt": "big"}, "body": {"size_pt": True}}}
    assert type_scale_tokens(junk) == {"display-scale": "1", "body-scale": "1"}


def _style(html: str) -> str:
    """The template's css, its ``{{ size.width }}``-style placeholders masked so its rules parse."""
    css = "".join(re.findall(r"<style>(.*?)</style>", html, re.S))
    return re.sub(r"\{\{[^{}]*\}\}", "0", css)


def test_every_seeded_template_scales_every_font_size_by_the_kits_type_scale():
    starters = social_starters()
    assert len(starters) == 18
    for starter in starters:
        css = _style(starter["blocks"]["html"])
        assert "--display-scale: var(--brand-display-scale, 1);" in css, starter["slug"]
        assert "--body-scale: var(--brand-body-scale, 1);" in css, starter["slug"]
        sizes = re.findall(r"font-size:\s*([^;}]+)", css)
        assert sizes, starter["slug"]
        unscaled = [size for size in sizes if not re.search(r"var\(--(display|body)-scale\)", size)]
        assert unscaled == [], (starter["slug"], unscaled)
        assert "var(--display-scale)" in css, starter["slug"]  # every template has display type


# ── the logo on a dark stage (FR-9) ──────────────────────────────────────────

def test_the_dark_logo_is_staged_and_shown_with_nothing_behind_it():
    logo, mark, dark = _uri(_png()), _uri(_png(128, 128)), _uri(_png(401, 100))
    bundle = _bundle({**HARBOURLINE, "logo_url": logo, "logo_mark_url": mark, "logo_dark_url": dark}, "data-story")
    assert {"path": "assets/brand/logo-dark.png", "data_uri": dark} in bundle["files"]
    assert bundle["variables"]["brand.logo_on_dark"] == "assets/brand/logo-dark.png"
    assert bundle["variables"]["brand.logo_mark"] == "assets/brand/logo-mark.png"
    assert bundle["brand"]["tokens"]["logo-chip"] == LOGO_CHIP_CLEAR


def test_without_a_dark_logo_the_mark_sits_on_a_light_chip_and_nothing_is_generated():
    mark = _uri(_png(128, 128))
    bundle = _bundle({**AUTOMATOS, "logo_mark_url": mark, "logo_dark_url": "https://cdn.example/dark.png"}, "data-story")
    assert bundle["variables"]["brand.logo_on_dark"] == "assets/brand/logo-mark.png"  # the URL is never fetched
    assert not any("logo-dark" in f["path"] for f in bundle.get("files", []))
    tokens = bundle["brand"]["tokens"]
    assert tokens["logo-chip"] == tokens[PAPER]
    bare = _bundle({}, "data-story")
    assert bare["variables"]["brand.logo_on_dark"] == NO_LOGO and bare["brand"]["tokens"]["logo-chip"] == "#ffffff"


def test_every_dark_stage_shows_the_logo_for_dark_backgrounds_on_its_chip():
    for starter in social_starters():
        html, slug = starter["blocks"]["html"], starter["slug"]
        if slug in DARK_STAGES:
            assert "{{ brand.logo_on_dark }}" in html and "{{ brand.logo_mark }}" not in html, slug
            assert "--logo-chip: var(--brand-logo-chip, transparent);" in html, slug
            assert re.search(r'<img class="dark-logo[^"]*"[^>]*src="\{\{ brand\.logo_on_dark \}\}"', html), slug
            assert ".dark-logo { background: var(--logo-chip);" in html, slug
        else:
            assert "{{ brand.logo_mark }}" in html and "{{ brand.logo_on_dark }}" not in html, slug


@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.setattr(bl.config, "DOCUMENT_STORAGE_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(bl, "is_storage_configured", lambda: False)
    return tmp_path


def test_the_social_render_inlines_the_uploaded_dark_logo(storage):
    dark = _png(300, 120)
    kit = get_brand_kit({"brand_kit": {"logo_dark_path": bl.save_brand_logo(WS, dark, stem=bl.LOGO_DARK_STEM)}})
    rendered = brand_kit_for_media_render(kit)
    assert rendered["logo_dark_url"] == _uri(dark)
    assert "logo_dark_url" not in kit  # the input is untouched
    assert "logo_dark_url" not in brand_kit_for_media_render(get_brand_kit({"brand_kit": {}}))


# ── sparing: the accent stays off a text-heavy card's surfaces ───────────────

def _aliases(css: str) -> dict:
    """The template's own ``--name: var(--brand-x, …)`` aliases."""
    return dict(re.findall(r"--([a-z0-9-]+):\s*var\(--(brand-[a-z0-9-]+)", css))


def test_under_sparing_the_accent_is_never_a_text_heavy_cards_background():
    assert HARBOURLINE["accent_use"] == "sparing" and "accent_use" not in AUTOMATOS  # sparing is every kit's default
    for slug in TEXT_HEAVY:
        starter = next(s for s in social_starters() if s["slug"] == slug)
        css = _style(starter["blocks"]["html"])
        aliases = _aliases(css)
        surfaces = 0
        for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
            if not {sel.strip() for sel in selectors.split(",")} & SURFACES:
                continue
            for value in re.findall(r"background(?:-color)?:\s*([^;]+)", body):
                surfaces += 1
                used = {aliases.get(name, name) for name in re.findall(r"var\(--([a-z0-9-]+)", value)}
                assert not used & ACCENT_TOKENS, (slug, selectors.strip(), value)
        assert surfaces, slug


# ── the CI driver: both night kits, every template ────────────────────────────

class _FakeRenderer:
    """media-render as the driver sees it: every bundle checks clean and returns one still of ``colour``."""

    def __init__(self, driver, colour: str) -> None:
        self.driver, self.colour, self.bundles = driver, driver.hex_rgb(colour), []

    def render(self, bundle):
        self.bundles.append(bundle)
        png = self.driver.encode_png(8, 8, lambda x, y: (*self.colour, 255))
        return {"report": {"check": {"errors": 0}}, "outputs": [{"name": "render.png", "data": png}], "seconds": 0.1}


@pytest.fixture(scope="module")
def driver():
    return _load("social_template_previews", _ROOT / "scripts" / "ci" / "social_template_previews.py")


def test_the_driver_carries_both_night_kits(driver):
    assert driver.NIGHT_KITS["automatos"]["primary_color"] == AUTOMATOS["primary_color"]
    harbourline = driver.NIGHT_KITS["harbourline"]
    assert harbourline["primary_color"].lower() == HARBOURLINE_ACCENT
    assert harbourline["palette"] == HARBOURLINE["palette"] and harbourline["type_scale"] == HARBOURLINE["type_scale"]
    assert harbourline["accent_use"] == "sparing"


def test_every_template_renders_with_both_night_kits_each_bundle_true_to_the_kit(driver, tmp_path):
    kits = driver._sibling("social_template_kits")
    renderer = _FakeRenderer(driver, HARBOURLINE_PAPER)
    report: dict = {}
    assert kits.run_night_kits(driver, renderer, tmp_path, report) == []
    slugs = {s["slug"] for s in social_starters()}
    for name in ("automatos", "harbourline"):
        rendered = {key.split(" ")[0] for key in report["night_kits"][name]}
        assert rendered == slugs, name
    images = social_starters(SOCIAL_IMAGE)
    assert len(report["night_kits"]["harbourline"]) == 4 + sum(len(s["blocks"]["sizes"]) for s in images)
    assert len(report["night_kits"]["automatos"]) == 4 + len(images)
    dark = [b for b in renderer.bundles if b["variables"].get("brand.logo_on_dark") == "assets/brand/logo-dark.png"]
    assert dark  # Harbourline's dark logo reached its dark stages
    assert (tmp_path / "kit-harbourline" / "title-card" / "1080x1350" / "render.png").exists()


def test_the_night_kit_pass_fails_a_page_that_is_not_the_stored_paper(driver, tmp_path):
    kits = driver._sibling("social_template_kits")
    failures = kits.run_night_kits(driver, _FakeRenderer(driver, "#202020"), tmp_path, {})
    assert failures and all("harbourline kit, Title card" in f and "stored paper" in f for f in failures)


def test_the_night_kit_pass_flags_a_bundle_untrue_to_the_kit(driver):
    kits = driver._sibling("social_template_kits")
    kit = kits.night_kit(driver, "harbourline")
    starter = next(s for s in social_starters() if s["slug"] == "data-story")
    bundle = driver.bundle_for(starter, kit, starter["preview"]["at"])
    assert kits.bundle_findings(kit, starter, bundle) == []
    bundle["brand"]["tokens"]["paper"] = "#ffffff"
    bundle["variables"]["brand.logo_on_dark"] = NO_LOGO
    findings = kits.bundle_findings(kit, starter, bundle)
    assert any("token paper" in f for f in findings) and any("dark logo" in f for f in findings)
