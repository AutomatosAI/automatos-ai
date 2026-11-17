"""PRD-251D US-D201 (E8) — the instant preview of an image post: the two routes.

On the S1.1c render harness (SQLite, the real Socials router with its gate and the real
permission check; the caller's role stubbed). ``POST /api/socials/templates/{id}/preview-page``
answers the template as one self-contained page for the words on screen; ``GET
/api/socials/brand-kit/preview-fonts`` answers the kit's font faces as data: URIs. Pinned:

* the words are filled on the server and HTML-escaped: a value holding ``<script>``, quotes or
  ``&`` stays text; a number goes into a script; a text value inside an event attribute is
  refused (422); the brand tokens, the size variables and the font marker are in the page;
* the page opens with its own content security policy and the timeline stand-in, its first
  script, which takes the peer-connection API away (the policy does not cover WebRTC); no staged
  path is left, every src is a data: URI (the logo too), the timeline library's script and the
  audio are gone;
* an empty photo spot's element is removed; a spot the post holds a picture for shows its
  marker; a story's 9:16 keeps the safe zone;
* a field left empty is empty text (an empty number is 0 and an empty true/false is false, so one
  in a script still previews); a value over its field's limit is refused, saying why;
* only a starter the platform ships, unchanged, previews: a workspace's own template and an edited
  starter are refused, so no script but the platform's runs in the editor;
* only an image template of one still previews; a composition a render refuses is refused;
* the request is bounded (values, their length, photos, the size, unknown keys);
* another workspace's template is 404, a viewer is 403, a workspace with Socials off is 404;
* the page reads the kit's logos and none of its fonts, uploaded or bundled; the fonts route gives the
  kit's faces as the renderer writes them, with each file inline; ``fonts_version`` follows them;
* the editor looks for the page's own policy and markers, as the server writes them;
* both routes are plain ``def`` and in the committed route manifest.
"""
from __future__ import annotations

import inspect
import json
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
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import api.socials_instant_preview as instant_api  # noqa: E402
import tests.test_prd251w1_render_lifecycle as render_harness  # noqa: E402
from core.media_render_bundle import BUNDLED_FONT_FILES, NO_LOGO  # noqa: E402
from core.models.workspaces import Workspace  # noqa: E402
from modules.documents.brand_kit import get_brand_kit  # noqa: E402
from modules.documents.social_starters import social_starters  # noqa: E402
from modules.socials import instant_preview  # noqa: E402
from tests.test_prd251w1_render_lifecycle import COMPOSITION, CREATED, WS, WS_OTHER, _ctx, _template  # noqa: E402

env = render_harness.env
# The real rule, kept before the fixture below stands in for it.
SHIPS_WITH_THE_PLATFORM = instant_api.ships_with_the_platform
MANIFEST = _ORCH / "reports" / "route-manifest.json"
STUDIO = _ORCH.parent / "frontend" / "components" / "deliverables" / "socials" / "studio"
WS_OFF = uuid.UUID("00000000-0000-0000-0000-0000000d201f")
PNG_URI = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
FONT = {"family": "Brand Sans", "weight": 400, "style": "normal", "data_uri": "data:font/woff2;base64,d09GMgABAAA="}
HOSTILE = "<script>alert(\"x\")</script> & \"quoted\" 'single'"
ESCAPED = "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; &amp; &quot;quoted&quot; &#x27;single&#x27;"
SAFE_ZONE = "--story-top: 250px; --story-bottom: 340px;"
REFERENCE = re.compile(r"""(?<![\w-])(?:src|href)\s*=\s*["']([^"']*)["']""", re.IGNORECASE)

# An image template as the seeded ones are made: the timeline library staged, a photo spot, a
# logo, the mix, and its own script that runs the timeline once its layout is done.
PAGE_HTML = (
    '<!doctype html><html lang="en"><head><meta charset="UTF-8" />'
    '<script src="assets/vendor/gsap.min.js"></script></head><body>'
    '<div id="root" data-composition-id="main" data-start="0" data-duration="1" '
    'data-width="{{ size.width }}" data-height="{{ size.height }}">'
    '<img class="photo" data-slot="photo" src="assets/slots/photo.png" alt="" />'
    '<h1 class="v">{{ headline }}</h1><p class="v" data-accent="{{ accent }}">{{ quote }}</p>'
    '<img class="logo" src="{{ brand.logo_mark }}" alt="" />'
    '<audio id="mix" src="assets/audio/mix.wav" data-start="0" data-duration="1"></audio></div>'
    "<script>window.__timelines = {}; const tl = gsap.timeline({ paused: true }); tl.set({}, {}, {{ seconds }});</script>"
    "</body></html>"
)
PAGE_TEMPLATE = {
    "html": PAGE_HTML,
    "css": "h1 { color: var(--brand-text); width: {{ size.width }}px; }",
    "variables_schema": {
        "headline": {"type": "text", "label": "Headline", "max_chars": 60},
        "quote": {"type": "text", "default": ""},
        "accent": {"type": "text", "default": ""},
        "seconds": {"type": "number", "default": 1},
    },
    "sizes": ["1080x1350", "1080x1920"],
    "slots": {"photo": {"kind": "image", "path": "assets/slots/photo.png", "label": "Photo"}},
}


@pytest.fixture(autouse=True)
def _made_here_stands_for_a_starter(monkeypatch):
    """The templates these tests make stand for starters the platform ships. The rule that only a
    starter previews has its own test, which puts the real rule back."""
    monkeypatch.setattr(instant_api, "ships_with_the_platform", lambda template: True)


def _image(env, *, blocks=PAGE_TEMPLATE, workspace_id=WS):
    return _template(env, workspace_id=workspace_id, blocks=blocks, fmt="social_image")


def _page(env, template_id, **body):
    return env.client.post(f"/api/socials/templates/{template_id}/preview-page", json=body)


def _html(env, template_id, **body):
    resp = _page(env, template_id, **body)
    assert resp.status_code == 200, resp.text
    return resp.json()["html"]


def _with_html(markup):
    return {**PAGE_TEMPLATE, "html": PAGE_HTML.replace('<h1 class="v">{{ headline }}</h1>', markup)}


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------


def test_the_words_are_filled_escaped_with_the_brand_tokens_the_size_and_the_font_marker(env):
    template_id = _image(env)
    resp = _page(env, template_id, values={"headline": HOSTILE, "seconds": 2}, size="1080x1350")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"html", "fonts_version"}
    page = body["html"]
    assert f'<h1 class="v">{ESCAPED}</h1>' in page
    assert "<script>alert" not in page and HOSTILE not in page
    assert 'data-width="1080" data-height="1350"' in page and "width: 1080px;" in page
    assert "tl.set({}, {}, 2);" in page  # a number goes into the template's own script
    brand = re.search(r"<style>(:root\{--brand-[^<]*)</style><style>h1 \{", page)
    assert brand is not None, "the brand styles come before the template css, in their own <style>"
    assert brand.group(1).endswith("}" + instant_preview.FONTS_MARKER)
    assert "{{" not in page


def test_the_page_opens_with_its_own_policy_and_fetches_nothing(env):
    page = _html(env, _image(env), values={"headline": "Open late"})
    assert page.startswith("<!doctype html>" + instant_preview.POLICY_META + instant_preview.TIMELINE_STUB)
    assert page.index(instant_preview.POLICY_META) < page.lower().index("<script")
    # The stand-in is the page's first script, and before it stands in for the timeline it takes the
    # peer-connection API away: the policy does not cover WebRTC.
    assert page.lower().index("<script") == page.index(instant_preview.TIMELINE_STUB)
    for name in ("RTCPeerConnection", "webkitRTCPeerConnection", "RTCDataChannel"):
        assert f'"{name}"' in instant_preview.TIMELINE_STUB
    assert instant_preview.TIMELINE_STUB.index("Object.defineProperty(window, name") < instant_preview.TIMELINE_STUB.index("window.gsap")
    assert "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; font-src data:" in page
    assert "window.gsap = stub;" in page
    assert "gsap.min.js" not in page and "<audio" not in page and "mix.wav" not in page and "assets/" not in page
    references = REFERENCE.findall(page)
    assert references and all(ref.startswith("data:") for ref in references), references
    assert f'<img class="logo" src="{NO_LOGO}" alt="" />' in page  # no uploaded logo: the renderer's empty one


def test_only_a_starter_the_platform_ships_unchanged_previews_as_you_type(env, monkeypatch):
    monkeypatch.setattr(instant_api, "ships_with_the_platform", SHIPS_WITH_THE_PLATFORM)
    # A workspace's own template: its script is not the platform's, so it never runs in the editor.
    own = _page(env, _image(env), values={"headline": "Open late"})
    assert own.status_code == 422, own.text
    assert own.json()["detail"] == instant_api.STARTERS_ONLY

    starters = social_starters("social_image")
    assert starters and all(SHIPS_WITH_THE_PLATFORM(SimpleNamespace(blocks=starter["blocks"])) for starter in starters)
    blocks = starters[0]["blocks"]
    # A starter someone edited is no longer the platform's: one more script is enough.
    edited = {**blocks, "html": blocks["html"].replace("</body>", "<script>window.added = 1;</script></body>")}
    assert edited["html"] != blocks["html"]
    assert not SHIPS_WITH_THE_PLATFORM(SimpleNamespace(blocks=edited))
    assert not SHIPS_WITH_THE_PLATFORM(SimpleNamespace(blocks=None))
    assert not SHIPS_WITH_THE_PLATFORM(SimpleNamespace(blocks={"html": PAGE_HTML}))


def test_a_value_in_an_event_attribute_is_refused_and_a_forbidden_element_is_too(env):
    handler = _image(env, blocks=_with_html("<h1 onclick=\"go('{{ headline }}')\">x</h1>"))
    resp = _page(env, handler, values={"headline": HOSTILE})
    assert resp.status_code == 422
    assert "inside a <script> or an event handler" in resp.json()["detail"]
    framed = _image(env, blocks=_with_html('<h1 class="v">{{ headline }}</h1><iframe title="elsewhere"></iframe>'))
    resp = _page(env, framed, values={"headline": "Open late"})
    assert resp.status_code == 422 and "the composition contains <iframe>" in resp.json()["detail"]


def test_an_empty_photo_spot_is_removed_and_one_the_post_holds_shows_its_marker(env):
    template_id = _image(env)
    assert 'data-slot="photo"' not in _html(env, template_id, values={"headline": "Open late"})
    held = _html(env, template_id, values={"headline": "Open late"}, photos=["photo"])
    marker = instant_preview.photo_marker("photo")
    assert f'<img class="photo" data-slot="photo" src="{marker}" alt="" />' in held
    resp = _page(env, template_id, values={}, photos=["hook"])
    assert resp.status_code == 422 and "no photo spot hook" in resp.json()["detail"]


def test_a_story_keeps_its_words_in_the_safe_zone(env):
    template_id = _image(env)
    assert SAFE_ZONE in _html(env, template_id, values={}, size="1080x1920", story=True)
    assert SAFE_ZONE not in _html(env, template_id, values={}, size="1080x1920", story=False)
    assert "--story-top" not in _html(env, template_id, values={}, size="1080x1350", story=True)  # not 9:16


def test_a_field_left_empty_is_empty_text_and_a_value_over_its_limit_is_refused(env):
    template_id = _image(env)
    assert '<h1 class="v"></h1>' in _html(env, template_id, values={})
    # An empty number or true/false with no default stays that kind (0, false): only those may sit in
    # a script, so a template that puts one there still previews with the field left empty.
    in_script = {
        **PAGE_TEMPLATE,
        "html": PAGE_HTML.replace("</body>", "<script>window.shown = {{ shown }};</script></body>"),
        "variables_schema": {**PAGE_TEMPLATE["variables_schema"], "seconds": {"type": "number"}, "shown": {"type": "boolean"}},
    }
    page = _html(env, _image(env, blocks=in_script), values={})
    assert "tl.set({}, {}, 0);" in page and "window.shown = false;" in page
    resp = _page(env, template_id, values={"headline": "x" * 61})
    assert resp.status_code == 422
    assert resp.json()["detail"] == "headline is longer than 60 characters (61 given)"


def test_only_an_image_template_of_one_still_has_an_instant_preview(env):
    video = _template(env, blocks=COMPOSITION, fmt="social_video")
    resp = _page(env, video, values={})
    assert resp.status_code == 422 and resp.json()["detail"] == instant_preview.NOT_AN_IMAGE
    slides = _image(env, blocks={**PAGE_TEMPLATE, "stills": [{"at": 0}, {"at": 0.5}]})
    resp = _page(env, slides, values={})
    assert resp.status_code == 422 and resp.json()["detail"] == instant_preview.ONE_STILL_ONLY
    empty = _template(env, blocks=None, fmt="social_image")
    assert _page(env, empty, values={}).status_code == 422


@pytest.mark.parametrize(
    "body",
    [
        {"values": {f"field_{i}": "x" for i in range(161)}},
        {"values": {"headline": "x" * 2001}},
        {"values": {"x" * 65: "a"}},
        {"values": {"headline": ["a list"]}},
        {"values": {}, "photos": [f"spot_{i}" for i in range(13)]},
        {"values": {}, "size": "10800x13500"},
        {"values": {}, "colour": "red"},
    ],
    ids=["too-many-values", "value-too-long", "name-too-long", "not-a-value", "too-many-photos", "size-too-long", "unknown-key"],
)
def test_the_request_is_bounded(env, body):
    assert _page(env, _image(env), **body).status_code == 422


def test_a_size_the_template_does_not_declare_is_refused(env):
    resp = _page(env, _image(env), values={}, size="1080x1080")
    assert resp.status_code == 422 and "is not one this template declares" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Who may ask
# ---------------------------------------------------------------------------


def test_another_workspaces_template_is_404_a_viewer_is_403_and_socials_off_is_404(env):
    theirs = _image(env, workspace_id=WS_OTHER)
    assert _page(env, theirs, values={}).status_code == 404
    assert _page(env, uuid.uuid4(), values={}).status_code == 404
    ours = _image(env)
    env.role = "viewer"
    assert _page(env, ours, values={}).status_code == 403
    env.role = "owner"
    env.session.add(Workspace(
        id=WS_OFF, name="ws-off", plan="basic", plan_limits={}, settings={"socials": {"enabled": False}}, onboarding={},
        created_at=CREATED, updated_at=CREATED,
    ))
    env.session.commit()
    env.ctx = _ctx(WS_OFF)
    assert _page(env, _image(env, workspace_id=WS_OFF), values={}).status_code == 404
    assert env.client.get("/api/socials/brand-kit/preview-fonts").status_code == 404


# ---------------------------------------------------------------------------
# The brand kit: logos in the page, fonts once per kit
# ---------------------------------------------------------------------------


def test_the_page_shows_the_uploaded_logo_inline_and_reads_no_uploaded_font(env, monkeypatch):
    real = instant_api.brand_kit_for_media_render
    real_bundle = instant_preview.build_bundle
    asked, bundled = [], []

    def render_ready(kit):
        asked.append(kit)
        return {**real(kit), "logo_url": PNG_URI, BUNDLED_FONT_FILES: [FONT]}

    def bundle(**parts):
        bundled.append(parts["brand_kit"])
        return real_bundle(**parts)

    monkeypatch.setattr(instant_api, "brand_kit_for_media_render", render_ready)
    monkeypatch.setattr(instant_preview, "build_bundle", bundle)
    resp = _page(env, _image(env), values={"headline": "Open late"})
    assert resp.status_code == 200, resp.text
    page = resp.json()["html"]
    assert f'<img class="logo" src="{PNG_URI}" alt="" />' in page and "assets/brand/" not in page
    assert [kit["font_files"] for kit in asked] == [[]]
    # No font goes into the page's bundle, a bundled face included: the editor has them from the fonts route.
    assert [(kit["font_files"], kit[BUNDLED_FONT_FILES]) for kit in bundled] == [([], [])]
    assert resp.json()["fonts_version"] == instant_preview.fonts_version(get_brand_kit({"socials": {"enabled": True}}))


def test_the_fonts_are_the_kits_faces_as_the_renderer_writes_them_with_each_file_inline(env, monkeypatch):
    real = instant_api.brand_kit_for_media_render
    monkeypatch.setattr(
        instant_api, "brand_kit_for_media_render", lambda kit: {**real(kit), "font_files": [FONT], BUNDLED_FONT_FILES: []},
    )
    resp = env.client.get("/api/socials/brand-kit/preview-fonts")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "css": '@font-face{font-family:"Brand Sans";src:url("data:font/woff2;base64,d09GMgABAAA=") format("woff2");'
        "font-weight:400;font-style:normal;font-display:block}"
    }


def test_the_fonts_version_follows_the_kits_faces_and_nothing_else():
    kit = get_brand_kit({})
    same = instant_preview.fonts_version(kit)
    assert same == instant_preview.fonts_version({**kit, "primary_color": "#123456"})
    stored_font = {"id": "f1", "family": "Brand Sans", "weight": 400, "style": "normal", "path": "w/brand/fonts/f1.woff2"}
    assert instant_preview.fonts_version({**kit, "font_files": [stored_font]}) != same
    assert instant_preview.fonts_version({**kit, "font_family": "Newsreader, serif"}) != same
    assert instant_preview.fonts_version({**kit, "heading_font": "Geist"}) != same


# ---------------------------------------------------------------------------
# The routes
# ---------------------------------------------------------------------------


def test_the_editor_looks_for_the_pages_own_policy_and_markers():
    """The editor shows a page only when it opens with this policy and puts the kit's faces and the
    photos at these markers (instant-preview-model.ts): a drift would hide every preview."""
    model = (STUDIO / "instant-preview-model.ts").read_text(encoding="utf-8")
    assert instant_preview.POLICY_META in model
    assert f"'{instant_preview.FONTS_MARKER}'" in model
    assert instant_preview.photo_marker("${slot}") in model


def test_both_routes_are_plain_defs_in_the_committed_manifest():
    assert not inspect.iscoroutinefunction(instant_api.social_template_preview_page)
    assert not inspect.iscoroutinefunction(instant_api.social_preview_fonts)
    routes = json.loads(MANIFEST.read_text())["routes"]
    assert {"method": "POST", "path": "/api/socials/templates/{template_id}/preview-page"} in routes
    assert {"method": "GET", "path": "/api/socials/brand-kit/preview-fonts"} in routes
