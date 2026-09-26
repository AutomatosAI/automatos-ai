"""PRD-251 Wave 1, US-107 (S1.2c) — the image families, seeded as social image templates.

Pins:

* **The eight starters.** The five automatos-social families (title, definition,
  stats, quote, announcement), two variants (a fact card, a carousel) and the
  infographic (US-113, test_prd251w1_infographic.py), each a ``social_image`` at
  the four sizes of automatos-social's schema.json (1080x1350, 1080x1920,
  1200x628, 1600x900), each passing the template contract and the brand rule
  (D4). They seed through the same starter path as the videos
  (test_prd251w1_seeded_templates.py pins the twelve rows).
* **Every word is a variable.** No text node of a card's markup is anything
  but a ``{{ placeholder }}``; the sample data fills every variable; the
  figures are claims (D7).
* **Stills.** An image renders as PNG stills: one for a card, one per slide
  for a carousel, where a point left empty drops its slide. The contract
  checks the ``stills`` block; the bundle asks media-render for the moments; and
  media-render's own parser takes every image at every size, and refuses a
  still with sound, a still that is also a preview, and moments out of order or
  past the end.
* **The paper.** The brand kit gives a light page with readable text on it
  and on its cards (WCAG, with a margin), and the Automatos kit reproduces the
  automatos-social cream, card and ink.
* **The CI driver** renders every image template at every size it declares
  and probes the brand stripe on two of them.
* **The clone is gone.** No code under orchestrator/, services/ or frontend/
  names ``repos/automatos-social``; the generated skill seed is the owner's to
  change in automatos-skills.
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys
from dataclasses import replace
from html.parser import HTMLParser
from pathlib import Path

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

from core.builtin_skills import builtin_skill_paths  # noqa: E402
from core.brand_palette import (  # noqa: E402
    PAPER_TOKENS,
    contrast,
    luminance,
    paper_palette,
    parse_hex,
)
from core.media_render_bundle import build_bundle  # noqa: E402
from core.social_templates import (  # noqa: E402
    MAX_STILLS,
    PLACEHOLDER,
    SocialTemplateError,
    brand_literals,
    claim_names,
    parse_size,
    resolve_variables,
    root_duration,
    still_moments,
    validate_social_blocks,
)
from media_render.bundle import parse_bundle  # noqa: E402  (media-render's own parser: stdlib only)
from media_render.config import load_settings  # noqa: E402
from media_render.validate import BundleError  # noqa: E402
from modules.documents.social_starters import SOCIAL_IMAGE_STARTER_SLUGS, social_starters  # noqa: E402

WS = "00000000-0000-0000-0000-0000000001a7"
# automatos-social/schema.json "sizes": ig_post, ig_story, linkedin, twitter.
SCHEMA_SIZES = ["1080x1350", "1080x1920", "1200x628", "1600x900"]
IMAGE_NAMES = {
    "title-card": "Title card",
    "definition-card": "Definition card",
    "stats-card": "Stats card",
    "quote-card": "Quote card",
    "announcement-card": "Announcement card",
    "fact-card": "Fact card",
    "carousel": "Carousel",
    "infographic": "Infographic",
}
KIT = {
    "name": "Automatos",
    "primary_color": "#e96235",
    "secondary_color": "#1a1714",
    "accent_color": "#90af5a",
    "text_color": "#f0e8db",
    "font_family": "Geist, sans-serif",
}
KITS = [
    pytest.param(KIT, id="automatos-studio-dark"),
    pytest.param({"primary_color": "#1a1a2e", "secondary_color": "#16213e", "accent_color": "#0f3460", "text_color": "#1a1a2e"}, id="the-document-defaults"),
    pytest.param({"primary_color": "#ffcc00", "secondary_color": "#f3ede2", "accent_color": "#00aa88", "text_color": "#222222"}, id="a-light-kit"),
    pytest.param({"primary_color": "#c8742c", "secondary_color": "#1f3b2d", "accent_color": "#f3ede2", "text_color": "#f3ede2"}, id="harbourline"),
    pytest.param({"primary_color": "#2f7bf6", "secondary_color": "#ffffff", "accent_color": "#ff00aa", "text_color": "#ffffff"}, id="all-light"),
]


def _images():
    return social_starters("social_image")


def _starter(slug):
    return next(s for s in social_starters() if s["slug"] == slug)


def _values(starter, **overrides):
    resolved = resolve_variables(starter["blocks"]["variables_schema"], {**starter["sample_data"], **overrides})
    assert resolved.missing == [] and resolved.invalid == [], (starter["name"], resolved)
    return resolved.values


def _bundle(starter, size=None, kit=KIT, **overrides):
    return build_bundle(
        workspace_id=WS,
        reference=f"seeded:{starter['slug']}",
        blocks=starter["blocks"],
        values=_values(starter, **overrides),
        brand_kit=kit,
        size=size,
        fmt="social_image",
    )


def _renderer_settings():
    return load_settings({})


# ---------------------------------------------------------------------------
# The eight starters
# ---------------------------------------------------------------------------


def test_the_five_families_two_variants_and_the_infographic_are_seeded_as_social_image_starters():
    images = _images()
    assert [s["slug"] for s in images] == list(SOCIAL_IMAGE_STARTER_SLUGS) == list(IMAGE_NAMES)
    assert [s["name"] for s in images] == list(IMAGE_NAMES.values())
    for starter in images:
        blocks = starter["blocks"]
        assert (starter["format"], starter["category"], blocks["sizes"]) == ("social_image", "social", SCHEMA_SIZES)
        assert validate_social_blocks(blocks, "social_image") == blocks
        assert brand_literals(blocks["html"], blocks.get("css") or "") == []
        assert "audio_plan" not in blocks and "slots" not in blocks
        assert starter["description"]


class _Text(HTMLParser):
    """The text nodes of a document's body, outside <script> and <style>."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.texts, self._skip, self._body = [], 0, False

    def handle_starttag(self, tag, attrs):
        self._body = self._body or tag == "body"
        if tag in ("script", "style"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip -= 1

    def handle_data(self, data):
        if self._body and not self._skip and data.strip():
            self.texts.append(data.strip())


@pytest.mark.parametrize("slug", list(IMAGE_NAMES))
def test_every_word_on_a_card_is_a_variable(slug):
    starter = _starter(slug)
    parser = _Text()
    parser.feed(starter["blocks"]["html"])
    # The page's only text is placeholders; the numbers a card counts are written by its script.
    assert parser.texts and all(PLACEHOLDER.fullmatch(text) for text in parser.texts), parser.texts
    values = _values(starter)
    assert set(values) == set(starter["blocks"]["variables_schema"])
    for value in starter["sample_data"].values():
        if isinstance(value, str) and len(value) >= 6:
            assert value not in starter["blocks"]["html"], value


def test_the_figures_on_the_cards_are_claims():
    assert claim_names(_starter("stats-card")["blocks"]["variables_schema"]) == ["stat_1_value", "stat_2_value", "stat_3_value"]
    assert claim_names(_starter("fact-card")["blocks"]["variables_schema"]) == ["fact_value"]
    assert claim_names(_starter("infographic")["blocks"]["variables_schema"]) == [f"row_{n}_value" for n in range(1, 6)]


@pytest.mark.parametrize("slug", list(IMAGE_NAMES))
def test_each_card_is_one_scene_on_a_paused_timeline_that_spans_it(slug):
    html = _starter(slug)["blocks"]["html"]
    duration = root_duration(html)
    assert duration == (8.0 if slug == "carousel" else 1.0)
    assert html.count('<audio id="mix" src="assets/audio/mix.wav"') == 1
    assert "gsap.timeline({ paused: true })" in html and f"tl.set({{}}, {{}}, {duration:g});" in html
    assert 'window.__timelines["main"] = tl;' in html
    # The brand stripe the CI probe reads, in the kit's own primary.
    assert '<div id="stripe"></div>' in html and "--brand: var(--brand-primary," in html
    for forbidden in ("Math.random", "Date.now", "performance.now", "requestAnimationFrame"):
        assert forbidden not in html, forbidden


def test_the_carousel_has_eight_one_second_slides():
    html = _starter("carousel")["blocks"]["html"]
    slides = re.findall(r'<section id="s(\d)" class="clip slide" data-start="(\d)" data-duration="1"', html)
    assert slides == [(str(i), str(i)) for i in range(8)]
    assert re.findall(r'data-point="(\d)"', html) == [str(i) for i in range(1, 7)]


# ---------------------------------------------------------------------------
# Stills: the contract, the bundle, media-render's parser
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("slug", list(IMAGE_NAMES))
def test_every_image_builds_a_still_bundle_media_render_takes_at_every_size(slug):
    starter = _starter(slug)
    settings = _renderer_settings()
    for size in SCHEMA_SIZES:
        bundle = _bundle(starter, size)
        assert bundle["still"] == {"at": still_moments(starter["blocks"], _values(starter))}
        assert "audio" not in bundle and "preview" not in bundle
        assert set(PAPER_TOKENS) <= set(bundle["brand"]["tokens"])
        parsed = parse_bundle(bundle, settings, {})
        assert (parsed.composition.width, parsed.composition.height) == parse_size(size)
        assert parsed.still.at == tuple(bundle["still"]["at"])
        assert parsed.composition.duration == root_duration(starter["blocks"]["html"])


def test_a_card_is_one_still_and_a_carousel_one_per_slide_it_shows():
    for slug in IMAGE_NAMES:
        if slug != "carousel":
            assert _bundle(_starter(slug))["still"] == {"at": [0.0]}, slug
    carousel = _starter("carousel")
    empty_points = {f"point_{n}_title": "" for n in range(3, 7)}
    assert _bundle(carousel, **empty_points)["still"] == {"at": [0.5, 1.5, 2.5, 7.5]}
    assert _bundle(carousel)["still"] == {"at": [0.5, 1.5, 2.5, 3.5, 4.5, 7.5]}  # the sample: four points
    every_point = {f"point_{n}_title": f"Point {n}" for n in range(1, 7)}
    assert _bundle(carousel, **every_point)["still"] == {"at": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]}
    # A point left out in the middle drops only its own slide.
    assert _bundle(carousel, **{**every_point, "point_4_title": "  "})["still"]["at"] == [0.5, 1.5, 2.5, 3.5, 5.5, 6.5, 7.5]


PAGE = (
    '<!doctype html><html><head></head><body><div id="root" data-composition-id="main" data-start="0" '
    'data-duration="3" data-width="{{ size.width }}" data-height="{{ size.height }}"><p>{{ a }}</p></div></body></html>'
)
SCHEMA = {"a": {"type": "text"}, "b": {"type": "text", "default": ""}}


@pytest.mark.parametrize(
    "stills, message",
    [
        pytest.param([], "non-empty list", id="empty"),
        pytest.param({"at": 0}, "non-empty list", id="not-a-list"),
        pytest.param([{"at": 0}] * (MAX_STILLS + 1), f"at most {MAX_STILLS}", id="too-many"),
        pytest.param([{"at": -1}], "0 or more", id="negative"),
        pytest.param([{"at": "0.5"}], "number of seconds", id="text"),
        pytest.param([{"at": 1}, {"at": 1}], "time order", id="twice"),
        pytest.param([{"at": 2}, {"at": 1}], "time order", id="backwards"),
        pytest.param([{"at": 3}], "past the composition's end", id="at-the-end"),
        pytest.param([{"at": 0, "when": "b"}], "first still is always taken", id="first-with-when"),
        pytest.param([{"at": 0}, {"at": 1, "when": "nope"}], "must name a variable", id="unknown-when"),
        pytest.param([{"at": 0, "size": "1080x1350"}], "not a still setting", id="unknown-key"),
    ],
)
def test_the_contract_refuses_stills_a_render_could_not_take(stills, message):
    with pytest.raises(SocialTemplateError, match=message):
        validate_social_blocks({"html": PAGE, "variables_schema": SCHEMA, "sizes": ["1080x1350"], "stills": stills}, "social_image")


def test_stills_are_for_an_image_and_a_well_formed_list_passes():
    blocks = {"html": PAGE, "variables_schema": SCHEMA, "sizes": ["1080x1350"], "stills": [{"at": 0.5}, {"at": 1.5, "when": "b"}]}
    with pytest.raises(SocialTemplateError, match="stills are for an image"):
        validate_social_blocks(blocks, "social_video")
    checked = validate_social_blocks(blocks, "social_image")
    assert checked["stills"] == blocks["stills"]
    assert still_moments(checked, {"a": "x", "b": ""}) == [0.5]
    assert still_moments(checked, {"a": "x", "b": "shown"}) == [0.5, 1.5]
    assert still_moments({"html": PAGE}, {}) == [0.0]
    assert root_duration(PAGE) == 3.0 and root_duration(PAGE.replace('"3"', '"{{ d }}"')) is None


@pytest.mark.parametrize(
    "extra, message",
    [
        pytest.param({"still": {"at": [0.5], "extra": 1}}, "still", id="unknown-key"),
        pytest.param({"still": {"at": []}}, "at least one moment", id="no-moment"),
        pytest.param({"still": {"at": [1.0]}}, "still.at", id="at-the-end"),
        pytest.param({"still": {"at": [0.6, 0.2]}}, "time order", id="backwards"),
        pytest.param({"still": {"at": [0.1] * 2}}, "time order", id="twice"),
        pytest.param({"still": {"at": [i / 20 for i in range(11)]}}, "still.at", id="more-than-the-limit"),
        pytest.param({"still": {"at": [0.0]}, "preview": {"at": [0.0]}}, "its own preview", id="and-a-preview"),
        pytest.param(
            {"still": {"at": [0.0]}, "audio": {"voice": {"lines": [{"id": "l01", "at": 0.1, "text": "Hi"}]}}},
            "no sound",
            id="and-a-voice",
        ),
    ],
)
def test_media_render_refuses_a_still_it_could_not_take(extra, message):
    bundle = {**_bundle(_starter("title-card")), **extra}
    with pytest.raises(BundleError, match=message):
        parse_bundle(bundle, _renderer_settings(), {})


def test_media_render_takes_as_many_stills_as_its_setting_allows():
    settings = replace(_renderer_settings(), still_max_frames=2)
    bundle = _bundle(_starter("carousel"))
    with pytest.raises(BundleError, match="still.at"):
        parse_bundle(bundle, settings, {})
    assert parse_bundle({**bundle, "still": {"at": [0.5, 1.5]}}, settings, {}).still.at == (0.5, 1.5)
    assert _renderer_settings().still_max_frames == MAX_STILLS == 10


# ---------------------------------------------------------------------------
# The paper palette
# ---------------------------------------------------------------------------


def test_the_automatos_kit_reproduces_the_automatos_social_paper():
    paper = paper_palette(KIT)
    assert set(paper) == set(PAPER_TOKENS)
    # The cream is the kit's own text colour, the ink its own secondary: exactly.
    assert (paper["paper"], paper["on-paper"]) == ("#f0e8db", "#1a1714")
    # The card sits a shade below the paper, as automatos-social's #e3d9c8 sits below #f1e9dd.
    assert contrast(parse_hex(paper["paper-card"]), parse_hex(paper["paper"])) == pytest.approx(1.16, abs=0.02)
    # The display colour is the brand orange, deepened only as far as large text needs.
    r, g, b = parse_hex(paper["primary-on-paper-large"])
    assert r > g > b and contrast(parse_hex(paper["primary-on-paper-large"]), parse_hex(paper["paper-card"])) < 3.5


@pytest.mark.parametrize("kit", KITS)
def test_any_kit_gives_a_light_page_with_readable_text(kit):
    paper = {name: parse_hex(value) for name, value in paper_palette(kit).items()}
    page, card = paper["paper"], paper["paper-card"]
    assert luminance(page) >= 0.80
    assert contrast(card, page) <= 1.17
    # Every tone is measured on the card, the darker surface, so it reads on both.
    for surface in (card, page):
        assert contrast(paper["on-paper"], surface) >= 10
        assert contrast(paper["on-paper-muted"], surface) >= 6.9 and contrast(paper["on-paper-dim"], surface) >= 5.3
        assert contrast(paper["primary-on-paper"], surface) >= 4.7 and contrast(paper["accent-on-paper"], surface) >= 4.7
        assert contrast(paper["primary-on-paper-large"], surface) >= 3.2
    # The numbered circles set the paper on the primary.
    assert contrast(page, paper["primary-on-paper"]) >= 4.7


def test_a_kit_without_a_usable_text_colour_leaves_the_template_fallbacks():
    assert paper_palette({"text_color": "cream", "secondary_color": "#1a1714"}) == {}
    assert set(paper_palette({"text_color": "#f0e8db"})) == {"paper", "paper-card"}


# ---------------------------------------------------------------------------
# The CI driver, and the clone
# ---------------------------------------------------------------------------


def _driver():
    spec = importlib.util.spec_from_file_location("social_template_previews", _ROOT / "scripts" / "ci" / "social_template_previews.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_ci_driver_renders_every_image_at_every_size_and_probes_the_stripe():
    driver = _driver()
    kit = {**driver.KIT, "logo_url": driver.logo_png(driver.KIT["primary_color"])}
    settings = _renderer_settings()
    probed = []
    for starter in driver.social_starters("social_image"):
        for size in starter["blocks"]["sizes"]:
            bundle = driver.image_bundle_for(starter, kit, size)
            parsed = parse_bundle(bundle, settings, {})
            assert (parsed.composition.width, parsed.composition.height) == parse_size(size)
            assert [f.path for f in parsed.files] == ["assets/brand/logo.png"]
        probe = starter["preview"].get("probe")
        if probe:
            assert 0 < probe["x"] < 1 and 0 < probe["y"] < 1 and probe["token"] in bundle["brand"]["tokens"]
            # The probe reads the stripe: 14 design pixels tall at every size.
            for size in starter["blocks"]["sizes"]:
                width, height = parse_size(size)
                assert probe["y"] * height + 1 < 14 * min(width, height) / 1080
            probed.append(starter["name"])
    assert probed == ["Title card", "Carousel"]
    assert driver._first_still(driver.image_bundle_for(_starter("carousel"), kit, "1080x1350")) == "render-01.png"
    assert driver._first_still(driver.image_bundle_for(_starter("quote-card"), kit, "1080x1350")) == "render.png"


TEXT_SUFFIXES = {".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".json", ".md", ".yaml", ".yml", ".html", ".txt", ".sh", ".toml"}
SKIPPED_DIRS = {"node_modules", ".next", "__pycache__", ".git", "dist", "build", "coverage", ".venv"}
GENERATED_SEED = _ORCH / "core" / "seeds" / "platform-management-skill.md"
# Every built-in skill's seed is generated from automatos-skills (US-119): its
# wording is the owner's to change there, never in this repo.
GENERATED_SEEDS = frozenset(builtin_skill_paths().values())


def test_no_code_names_the_automatos_social_clone():
    offenders = []
    for top in (_ORCH, _ROOT / "services", _ROOT / "frontend"):
        for dirpath, dirnames, filenames in os.walk(top):
            dirnames[:] = [d for d in dirnames if d not in SKIPPED_DIRS]
            for filename in filenames:
                path = Path(dirpath) / filename
                if path.suffix not in TEXT_SUFFIXES or path in GENERATED_SEEDS or path == Path(__file__).resolve():
                    continue
                try:
                    text = path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue
                if "repos/automatos-social" in text:
                    offenders.append(str(path.relative_to(_ROOT)))
    assert offenders == []
    # The generated Auto skill seed still carries its lines: the owner changes them in automatos-skills.
    assert "repos/automatos-social" in GENERATED_SEED.read_text(encoding="utf-8")
