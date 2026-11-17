"""PRD-251D Wave 3, US-D301 (E10) — look options on a post.

Pinned (``modules/socials/post_look.py``, ``core/brand_palette.stage_palette``'s ``ink-card``,
``modules/socials/render.bundle_for``, the gallery entry, the service and the post routes):

* **The look** is ``{"mode", "accent", "logo"}``, each key optional: ``light``/``dark``,
  ``bold``/``sparing``, ``true``/``false``. A choice that is the template as authored (light, the
  logo shown) is left out, and a look with nothing left is ``None``. Anything else is refused.
* **Built on paper.** A template whose composition reads ``var(--brand-paper)`` is built on paper:
  exactly the nine paper starters. Only they take light or dark; the gallery entry's ``looks``
  says so, and every template takes the accent and the logo.
* **Dark** maps each paper token to the stage token that plays its part (and back, for the dark
  pieces a paper card lays on its page), the accent's fill read again from the swapped tokens,
  under both accents. Every word a paper card sets still reads on its ground in the dark look, for
  five kits, and the ink's card is a lift every stage tone reads on. A kit without the whole stage
  keeps its tokens; nothing is changed in place.
* **The accent** is the kit's ``accent_use`` on a copy of the kit; **the logo off** empties every
  logo variable (each starter hides the empty logo) and keeps the brand name.
* **The render** (``bundle_for``) applies it; a post with no look, or a thumbnail's stub with no
  ``look`` at all, renders as before, and dark on a template not built on paper changes nothing.
* **A render setting.** The look is never hashed: a post with one hashes as one without, and a look
  changed on an approved post keeps its approval, its hash and its status, and its render is still
  refused (409). The create and update routes take it, answer with it, and refuse a bad one (422).
* **The CI pass** (``scripts/ci/social_template_looks.py``) renders every starter in every look it
  is offered in, each bundle true to its look.
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys
import uuid
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
if str(_MEDIA_RENDER) not in sys.path:
    sys.path.append(str(_MEDIA_RENDER))

import tests.test_prd251_api as api_harness  # noqa: E402
from core.brand_palette import (  # noqa: E402
    INK_CARD_CONTRAST,
    LARGE_TEXT_MIN_CONTRAST,
    PAPER_TEXT_MIN_CONTRAST,
    contrast,
    luminance,
    parse_hex,
    stage_palette,
)
from core.media_render_bundle import NO_LOGO, brand_tokens  # noqa: E402
from media_render.bundle import parse_bundle  # noqa: E402  (media-render's own parser: stdlib only)
from media_render.config import load_settings  # noqa: E402
from media_render.voices_build import load_catalogue  # noqa: E402
from modules.documents.social_starters import social_starters  # noqa: E402
from modules.socials import post_look, render, service, template_gallery  # noqa: E402
from tests.test_prd251_api import _approved, _create  # noqa: E402

api = api_harness.api

WS, POST_ID = uuid.UUID("6d0b5c1e-8f1a-4c2b-9d3e-0a1b2c3d4e77"), uuid.UUID("6d0b5c1e-8f1a-4c2b-9d3e-0a1b2c3d4e78")
AUTHOR, REVIEWER = "user-1", "reviewer-1"
PNG = "data:image/png;base64,iVBORw0KGgo="
LOGO_VARIABLES = ("brand.logo", "brand.logo_mark", "brand.logo_on_dark")
# The reference's Studio Dark (the media-render CI kit), the two night kits, the documents' defaults, a light kit.
STUDIO_DARK = {"name": "Automatos", "primary_color": "#e96235", "secondary_color": "#1a1714", "text_color": "#f0e8db",
               "palette": {"accent_2": "#90af5a"}, "font_family": "Inter, sans-serif"}
KITS = [
    pytest.param(STUDIO_DARK, id="studio-dark"),
    pytest.param({"primary_color": "#c44a1a", "secondary_color": "#1d3658", "text_color": "#1a1a2e"}, id="automatos-v1"),
    pytest.param({"primary_color": "#1E3A5F", "secondary_color": "#C26A2E", "text_color": "#1a1a2e",
                  "palette": {"paper": "#faf7f2", "accent": "#1e3a5f"}}, id="harbourline-v2"),
    pytest.param({"primary_color": "#1a1a2e", "secondary_color": "#16213e", "text_color": "#1a1a2e"}, id="document-defaults"),
    pytest.param({"primary_color": "#ffcc00", "secondary_color": "#f3ede2", "text_color": "#222222"}, id="a-light-kit"),
]
PAPER_STARTERS = {
    "announcement-card", "brand-board", "carousel", "definition-card", "fact-card", "infographic", "quote-card",
    "stats-card", "title-card",
}
# The dark look on a paper template: each token and the token whose value it takes.
DARK_TAKES = {
    "paper": "ink", "paper-card": "ink-card", "on-paper": "on-ink", "on-paper-muted": "on-ink-muted",
    "on-paper-dim": "on-ink-dim", "primary-on-paper": "primary-light", "primary-on-paper-large": "primary-on-ink",
    "accent-on-paper": "accent-on-ink",
    "ink": "paper", "ink-card": "paper-card", "on-ink": "on-paper", "on-ink-muted": "on-paper-muted",
    "on-ink-dim": "on-paper-dim", "primary-on-ink": "primary-on-paper", "primary-light": "primary-on-paper",
    "accent-on-ink": "accent-on-paper",
}
FILL = ("accent-fill", "on-accent-fill")
# WCAG AA, what the renderer's contrast pass holds words to: small text, and large (display) text.
AA_TEXT, AA_LARGE = 4.5, 3.0
# A paper card's words: small text on the page and on its cards, display words, and the dark pill's.
SMALL_TEXT = ("on-paper", "on-paper-muted", "on-paper-dim", "primary-on-paper")
PILL_TEXT = ("on-ink", "on-ink-muted")
LOGO_IMG = re.compile(r'<img class="([^"]+)"[^>]*src="\{\{ brand\.logo')
FLOAT_SLACK = 1e-9


def _starter(slug):
    return next(s for s in social_starters() if s["slug"] == slug)


# ── the look ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value, clean", [
    (None, None),
    ({}, None),
    ({"mode": "dark"}, {"mode": "dark"}),
    ({"accent": "bold"}, {"accent": "bold"}),
    ({"accent": "sparing"}, {"accent": "sparing"}),
    ({"logo": False}, {"logo": False}),
    ({"mode": "light", "logo": True}, None),  # the template as authored: the kit's own, left out
    ({"logo": False, "accent": "bold", "mode": "dark"}, {"mode": "dark", "accent": "bold", "logo": False}),
])
def test_a_look_is_three_optional_choices_and_the_kits_own_is_left_out(value, clean):
    checked = post_look.validate_look(value)
    assert checked == clean
    assert list(checked or {}) == [key for key in ("mode", "accent", "logo") if key in (clean or {})]
    assert service._VALIDATORS["look"] is post_look.validate_look


@pytest.mark.parametrize("value", [
    {"mode": "dim"}, {"mode": True}, {"accent": "loud"}, {"accent": None}, {"logo": 1}, {"logo": "off"},
    {"colour": "red"}, {"mode": "dark", "font": "serif"}, "dark", ["dark"],
])
def test_anything_else_is_refused(value):
    with pytest.raises(service.InvalidPost):
        post_look.validate_look(value)


# ── which templates take which choices ──────────────────────────────────────

def test_the_nine_paper_starters_take_light_or_dark_and_every_starter_the_accent_and_the_logo():
    starters = social_starters()
    assert {s["slug"] for s in starters if post_look.built_on_paper(s["blocks"])} == PAPER_STARTERS
    for starter in starters:
        expected = ["mode", "accent", "logo"] if starter["slug"] in PAPER_STARTERS else ["accent", "logo"]
        assert post_look.looks_of(starter["blocks"]) == expected, starter["slug"]


def test_built_on_paper_is_the_paper_ground_token_not_a_card_on_it():
    assert post_look.built_on_paper({"html": "<style>:root { --page: var(--brand-paper, #fff); }</style>"})
    assert post_look.built_on_paper({"html": "<p/>", "css": "body { background: var( --brand-paper ) }"})
    assert not post_look.built_on_paper({"html": "<style>.c { background: var(--brand-paper-card, #eee); }</style>"})
    assert not post_look.built_on_paper({"html": "<style>body { background: var(--brand-ink); }</style>"})
    assert not post_look.built_on_paper(None)


def _row(starter):
    return SimpleNamespace(
        id=uuid.uuid4(), name=starter["name"], description=starter["description"], format=starter["format"],
        blocks=starter["blocks"], thumbnail_url=None, created_by="system", updated_at=None, sample_data=starter["sample_data"],
    )


def test_the_gallery_entry_says_which_looks_a_template_takes():
    assert template_gallery.entry(_row(_starter("quote-card")), store=None)["looks"] == ["mode", "accent", "logo"]
    assert template_gallery.entry(_row(_starter("photo-headline")), store=None)["looks"] == ["accent", "logo"]
    assert template_gallery.entry(_row(_starter("data-story")), store=None)["looks"] == ["accent", "logo"]


# ── the dark look's tokens ──────────────────────────────────────────────────

@pytest.mark.parametrize("accent", ["bold", "sparing"])
def test_dark_maps_each_paper_token_to_the_stage_token_that_plays_its_part(accent):
    kit = {**STUDIO_DARK, "accent_use": accent}
    tokens = brand_tokens(kit)
    dark = post_look.dark_tokens(tokens, kit)
    assert {name: dark[name] for name in DARK_TAKES} == {name: tokens[source] for name, source in DARK_TAKES.items()}
    assert dark["paper"] != tokens["paper"] and dark["ink"] != tokens["ink"]  # non-vacuous: the grounds swapped
    if accent == "bold":  # the fill is the small accent on the dark page, its words the page
        assert (dark["accent-fill"], dark["on-accent-fill"]) == (tokens["primary-light"], tokens["ink"])
    else:  # clear, its words the small accent
        assert (dark["accent-fill"], dark["on-accent-fill"]) == ("transparent", tokens["primary-light"])
    untouched = set(tokens) - set(DARK_TAKES) - set(FILL)
    assert {"primary", "secondary", "accent", "text", "logo-chip", "accent-bar-span", "mono-font", "display-scale"} <= untouched
    assert {name: dark[name] for name in untouched} == {name: tokens[name] for name in untouched}
    assert tokens == brand_tokens(kit)  # nothing changed in place


def test_a_kit_without_the_whole_stage_keeps_its_tokens():
    kit = {"primary_color": "#c44a1a", "secondary_color": "navy", "text_color": "#1a1a2e"}  # no hex secondary: no stage
    tokens = brand_tokens(kit)
    assert "ink" not in tokens and "paper" in tokens
    assert not post_look.takes_dark(tokens) and post_look.dark_tokens(tokens, kit) == tokens
    bundle = {"brand": {"tokens": tokens}, "variables": {"brand.logo_mark": "assets/brand/logo.png",
                                                          "brand.logo_on_dark": "assets/brand/logo-dark.png"}}
    paper_card = {"html": "<style>body { background: var(--brand-paper, #fff); }</style>"}
    kept = post_look.with_look(bundle, paper_card, {"mode": "dark"}, kit)
    assert kept["brand"]["tokens"] == tokens and kept["variables"] == bundle["variables"]  # the light page, its own logos


@pytest.mark.parametrize("kit", KITS)
def test_the_ink_card_is_a_lift_every_stage_tone_still_reads_on(kit):
    stage = {name: parse_hex(value) for name, value in stage_palette(kit).items()}
    ink, card = stage["ink"], stage["ink-card"]
    assert luminance(card) > luminance(ink)
    assert contrast(card, ink) <= INK_CARD_CONTRAST + FLOAT_SLACK
    for tone in ("on-ink", "on-ink-muted", "on-ink-dim", "primary-light"):
        assert contrast(stage[tone], card) >= PAPER_TEXT_MIN_CONTRAST - FLOAT_SLACK, tone
    assert contrast(stage["primary-on-ink"], card) >= LARGE_TEXT_MIN_CONTRAST


@pytest.mark.parametrize("accent", ["bold", "sparing"])
@pytest.mark.parametrize("kit", KITS)
def test_in_the_dark_look_every_word_a_paper_card_sets_reads_on_its_ground(kit, accent):
    looked = {**kit, "accent_use": accent}
    raw = post_look.dark_tokens(brand_tokens(looked), looked)
    dark = {name: parse_hex(value) for name, value in raw.items() if parse_hex(value) is not None}
    for ground in ("paper", "paper-card"):  # the page and its cards, now the ink and its card
        for text in SMALL_TEXT:
            assert contrast(dark[text], dark[ground]) >= AA_TEXT, (text, ground)
        assert contrast(dark["primary-on-paper-large"], dark[ground]) >= AA_LARGE, ground
        if accent == "sparing":
            assert raw["accent-fill"] == "transparent" and contrast(dark["on-accent-fill"], dark[ground]) >= AA_TEXT
    for text in PILL_TEXT:  # the dark pill and the dark note, now the paper
        assert contrast(dark[text], dark["ink"]) >= AA_TEXT, text
    if accent == "bold":
        assert contrast(dark["on-accent-fill"], dark["accent-fill"]) >= AA_TEXT


# ── the accent and the logo ─────────────────────────────────────────────────

def test_the_accent_is_the_kits_accent_use_on_a_copy_of_the_kit():
    kit = dict(STUDIO_DARK)
    bold = post_look.kit_with_look(kit, {"accent": "bold"})
    assert bold == {**kit, "accent_use": "bold"} and "accent_use" not in kit
    kept = post_look.kit_with_look(kit, None)
    assert kept == kit and kept is not kit
    assert post_look.kit_with_look(None, {"accent": "sparing"}) == {"accent_use": "sparing"}
    assert brand_tokens(bold)["accent-bar-span"] == "1" and brand_tokens(kit)["accent-bar-span"] == "0"


def test_the_logo_off_empties_every_logo_variable_and_keeps_the_name():
    variables = {"brand.name": "Automatos", "brand.logo": "assets/brand/logo.png", "brand.logo_mark": "assets/brand/logo-mark.png",
                 "brand.logo_on_dark": "assets/brand/logo-dark.png", "headline": "Hi"}
    bundle = {"brand": {"tokens": {"paper": "#ffffff"}}, "variables": variables}
    off = post_look.with_look(bundle, {"html": "<p/>"}, {"logo": False}, {})
    assert {name: off["variables"][name] for name in LOGO_VARIABLES} == {name: NO_LOGO for name in LOGO_VARIABLES}
    assert (off["variables"]["brand.name"], off["variables"]["headline"]) == ("Automatos", "Hi")
    assert bundle["variables"]["brand.logo"] == "assets/brand/logo.png"  # the bundle itself is untouched


def test_every_starter_hides_the_empty_logo_wherever_it_shows_one():
    for starter in social_starters():
        html = starter["blocks"]["html"]
        images = LOGO_IMG.findall(html)
        assert images, starter["slug"]
        for classes in images:
            assert any(f'.{name}[src^="data:image/gif"]' in html for name in classes.split()), (starter["slug"], classes)


def test_on_a_dark_page_a_kit_with_a_dark_logo_shows_it_where_a_paper_card_shows_its_logo():
    variables = {"brand.logo": "assets/brand/logo.png", "brand.logo_mark": "assets/brand/logo-mark.png",
                 "brand.logo_on_dark": "assets/brand/logo-dark.png"}
    assert post_look.dark_logos(variables) == {
        "brand.logo": "assets/brand/logo-dark.png", "brand.logo_mark": "assets/brand/logo-dark.png",
        "brand.logo_on_dark": "assets/brand/logo-mark.png",
    }
    without = {"brand.logo": "assets/brand/logo.png", "brand.logo_mark": "assets/brand/logo.png", "brand.logo_on_dark": "assets/brand/logo.png"}
    assert post_look.dark_logos(without) == without


# ── the render ──────────────────────────────────────────────────────────────

_NO_LOOK = object()


def _bundle(slug, look=_NO_LOOK, kit=None):
    """``bundle_for`` of a post of the starter's sample data; with no ``look`` attribute at all, a thumbnail's stub."""
    starter = _starter(slug)
    template = SimpleNamespace(id=uuid.uuid4(), format=starter["format"], blocks=starter["blocks"])
    fields = {"id": POST_ID, "workspace_id": WS, "length_seconds": None, "music": None,
              "variables": {name: {"value": value} for name, value in starter["sample_data"].items()}}
    if look is not _NO_LOOK:
        fields["look"] = look
    return render.bundle_for(SimpleNamespace(**fields), template, kit or {**STUDIO_DARK, "logo_url": PNG, "logo_dark_url": PNG})


def test_bundle_for_sets_a_paper_card_dark_with_its_logo_off_and_no_template_change():
    plain, dark = _bundle("quote-card"), _bundle("quote-card", {"mode": "dark", "logo": False})
    before, after = plain["brand"]["tokens"], dark["brand"]["tokens"]
    assert (after["paper"], after["ink"], after["paper-card"]) == (before["ink"], before["paper"], before["ink-card"])
    assert {name: dark["variables"][name] for name in LOGO_VARIABLES} == {name: NO_LOGO for name in LOGO_VARIABLES}
    assert dark["variables"]["brand.name"] == plain["variables"]["brand.name"] == "Automatos"
    assert dark["composition"] == plain["composition"]  # the template's own file, unchanged
    assert _bundle("quote-card", None) == plain == _bundle("quote-card", {"mode": "light", "logo": True})
    parse_bundle(dark, load_settings({}), {}, frozenset(load_catalogue()))  # media-render takes it


def test_bundle_for_shows_the_dark_logo_on_a_dark_paper_card():
    dark = _bundle("quote-card", {"mode": "dark"})
    assert dark["variables"]["brand.logo_mark"] == "assets/brand/logo-dark.png"


def test_dark_on_a_template_not_built_on_paper_is_left_out_of_its_render():
    plain, dark = _bundle("photo-headline"), _bundle("photo-headline", {"mode": "dark"})
    assert dark["brand"]["tokens"] == plain["brand"]["tokens"] and dark["variables"] == plain["variables"]


def test_the_accent_choice_reaches_the_render_through_accent_use():
    bold, sparing = _bundle("announcement-card", {"accent": "bold"}), _bundle("announcement-card", {"accent": "sparing"})
    assert bold["brand"]["tokens"]["accent-bar-span"] == "1" and sparing["brand"]["tokens"]["accent-bar-span"] == "0"
    assert bold["brand"]["tokens"]["accent-fill"] == bold["brand"]["tokens"]["primary-on-paper"]
    assert sparing["brand"]["tokens"]["accent-fill"] == "transparent"
    assert "accent_use" not in STUDIO_DARK


# ── a render setting: never hashed, never an approval ───────────────────────

class _Db:
    def add(self, obj) -> None:
        pass


def _draft(**fields):
    return service.create_draft(_Db(), workspace_id=WS, created_by=AUTHOR, title="A post", format="image",
                                copy={"base": "Hello"}, **fields)


def test_a_look_changed_after_a_render_leaves_the_file_older_and_a_post_without_one_reads_as_before():
    # PRD-251D US-D108 + US-D301: the look is a render setting outside the approved hash, so the record
    # of what a file was made from is what tells a look changed since. A post with no look must hash
    # exactly as it did before posts carried one, or every file recorded then would read as older.
    from core import social_file_current as file_state
    from core.social_render_state import RENDER_DONE

    post = SimpleNamespace(format="image", template_id="tpl", length_seconds=None, variables={}, footage=None,
                           look=None, review_log=[])
    assert "look" not in file_state.render_inputs(post)
    made_plain = file_state.render_inputs_hash(post)

    post.look = {"mode": "dark"}
    assert file_state.render_inputs(post)["look"] == {"mode": "dark"}
    assert file_state.render_inputs_hash(post) != made_plain

    post.review_log = [{"action": RENDER_DONE, file_state.MADE_FROM: made_plain}]  # rendered before the look was chosen
    assert file_state.file_current(post) is False
    post.look = None
    assert file_state.file_current(post) is True


def test_the_look_is_kept_on_the_post_and_never_hashed():
    plain, looked = _draft(), _draft(look={"logo": False, "mode": "dark", "accent": "bold"})
    assert looked.look == {"mode": "dark", "accent": "bold", "logo": False}
    assert looked.to_dict()["look"] == looked.look and plain.to_dict()["look"] is None
    assert looked.content_hash == plain.content_hash == service.compute_content_hash(plain)
    assert "look" in service.RENDER_FIELDS and "look" not in service.CONTENT_FIELDS
    with pytest.raises(service.InvalidPost):
        _draft(look={"mode": "night"})


def test_a_look_changed_on_an_approved_post_keeps_its_approval_and_its_render_is_still_refused():
    post = _draft()
    service.submit(post, AUTHOR)
    service.approve(post, REVIEWER, content_hash=post.content_hash)
    approved = post.content_hash
    service.update_post(post, AUTHOR, {"look": {"mode": "dark", "accent": "sparing"}})
    assert (post.status, post.content_hash, post.approved_hash) == (service.APPROVED, approved, approved)
    assert post.look == {"mode": "dark", "accent": "sparing"}
    assert not any(entry["action"] == service.ACTION_APPROVAL_VOIDED for entry in post.review_log)
    with pytest.raises(service.IllegalTransition):
        service.assert_can_render(post)


def test_the_post_routes_take_a_look_and_answer_with_it(api):
    looked = _create(api, look={"logo": False, "mode": "dark"})
    plain = _create(api)
    assert looked["look"] == {"mode": "dark", "logo": False} and plain["look"] is None
    assert looked["content_hash"] == plain["content_hash"]  # outside the hash
    changed = api.client.patch(f"/api/socials/posts/{plain['id']}", json={"look": {"accent": "bold"}})
    assert changed.status_code == 200, changed.text
    assert changed.json()["look"] == {"accent": "bold"} and changed.json()["content_hash"] == plain["content_hash"]
    cleared = api.client.patch(f"/api/socials/posts/{plain['id']}", json={"look": {"mode": "light", "logo": True}})
    assert cleared.status_code == 200 and cleared.json()["look"] is None  # the kit's own: left out
    assert api.client.patch(f"/api/socials/posts/{plain['id']}", json={"look": {"mode": "dim"}}).status_code == 422
    assert api.client.post("/api/socials/posts", json={"title": "Bad", "look": {"logo": "off"}}).status_code == 422


def test_an_approved_posts_look_is_its_approved_files_and_a_render_is_still_refused(api):
    post = _approved(api)
    changed = api.client.patch(f"/api/socials/posts/{post['id']}", json={"look": {"mode": "dark"}})
    assert changed.status_code == 200, changed.text
    body = changed.json()
    assert body["look"] == {"mode": "dark"}
    assert (body["status"], body["content_hash"], body["approved_hash"]) == ("approved", post["content_hash"], post["content_hash"])
    assert api.client.post(f"/api/socials/posts/{post['id']}/render").status_code == 409


# ── the CI pass: every starter in every look it is offered in ───────────────

class _FakeRenderer:
    """media-render as the driver sees it: every bundle checks clean and returns one small still."""

    def __init__(self, driver) -> None:
        self.driver, self.bundles = driver, []

    def render(self, bundle):
        self.bundles.append(bundle)
        png = self.driver.encode_png(4, 4, lambda x, y: (0, 0, 0, 255))
        return {"report": {"check": {"errors": 0}}, "outputs": [{"name": "render.png", "data": png}], "seconds": 0.1}


@pytest.fixture(scope="module")
def driver():
    spec = importlib.util.spec_from_file_location("social_template_previews", _ROOT / "scripts" / "ci" / "social_template_previews.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PAPER_LOOKS = {"light-bold-logo", "light-bold-no-logo", "light-sparing-no-logo",
               "dark-bold-logo", "dark-bold-no-logo", "dark-sparing-logo", "dark-sparing-no-logo"}
OTHER_LOOKS = {"bold-logo", "bold-no-logo", "sparing-no-logo"}


def test_the_ci_pass_renders_every_starter_in_every_look_it_is_offered_in(driver, tmp_path):
    looks = driver._sibling("social_template_looks")
    kit = {**driver.KIT, "logo_url": driver.logo_png(driver.KIT["primary_color"])}
    renderer, report = _FakeRenderer(driver), {}
    assert looks.run_looks(driver, renderer, tmp_path, kit, report) == []
    assert set(report["looks"]) == {s["slug"] for s in social_starters()}
    for slug, rendered in report["looks"].items():
        assert set(rendered) == (PAPER_LOOKS if slug in PAPER_STARTERS else OTHER_LOOKS), slug
    assert len(renderer.bundles) == 7 * len(PAPER_STARTERS) + 3 * (len(report["looks"]) - len(PAPER_STARTERS))
    assert (tmp_path / "look-quote-card" / "dark-sparing-logo" / "render.png").exists()


def test_the_ci_pass_flags_a_bundle_untrue_to_its_look(driver):
    looks = driver._sibling("social_template_looks")
    kit = {**driver.KIT, "logo_url": driver.logo_png(driver.KIT["primary_color"])}
    starter, look = _starter("quote-card"), {"mode": "dark", "accent": "bold", "logo": False}
    plain, looked = looks.look_bundle(driver, starter, kit, {}), looks.look_bundle(driver, starter, kit, look)
    assert looks.look_findings(plain, looked, look) == []
    untrue = {**looked, "brand": {"tokens": plain["brand"]["tokens"]}, "variables": plain["variables"]}
    findings = looks.look_findings(plain, untrue, look)
    assert any("dark" in f for f in findings) and any("logo is off" in f for f in findings)
    assert any("brand bar" in f for f in findings)
