"""PRD-251D US-D201 — the instant preview's page is the renderer's own fill, so the two cannot drift.

The renderer's page assembly (``services/media-render/media_render/composition.py``) lives in
another container and is not importable by the orchestrator at run time, so the orchestrator
restates its rules (``core/social_composition.py``) and assembles the preview page with them
(``modules/socials/instant_preview.py``). This test imports the renderer's own module, its
package found LAST on ``sys.path`` as ``test_prd251w1_seeded_templates.py`` finds it (the
package reaches only ``validate.py`` and ``espeak_build.py``, both the standard library only),
and holds the two equal:

* for every seeded image template of one still, at every size it declares, a story or not, its
  photo spots held or empty: the page is byte for byte the renderer's ``build_document`` of the
  same markup, css, variables and brand tokens (``brand_stylesheet``), with the font marker
  after the tokens and the page's own head (its policy and the timeline stand-in) first;
* the fill, value by value: text, numbers, true/false and every character an escape changes;
* every refusal, word for word: an unset variable, a value that is not text, a number or
  true/false, text in a script or an event handler (the renderer's closing-tag cases), a
  placeholder that is no name, a style closed early, a document without exactly one ``</head>``,
  each forbidden element and a refresh;
* the brand tokens and a font face as the renderer writes them, the kit's faces with each file
  inline, and the patterns and constants the rules are made of.
"""
from __future__ import annotations

import os
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
_MEDIA_RENDER = _ORCH.parent / "services" / "media-render"
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))
# The renderer's package (standard library only), found LAST so no orchestrator module is shadowed.
if str(_MEDIA_RENDER) not in sys.path:
    sys.path.append(str(_MEDIA_RENDER))

from media_render import composition as renderer  # noqa: E402
from media_render.validate import BundleError  # noqa: E402

from core import social_composition as ours  # noqa: E402
from core.media_render_bundle import build_bundle  # noqa: E402
from core.social_templates import IMAGE_SLOT, SOCIAL_IMAGE, resolve_variables  # noqa: E402
from modules.documents.social_starters import social_starters  # noqa: E402
from modules.socials import instant_preview  # noqa: E402

WS = uuid.UUID("00000000-0000-0000-0000-0000000d2010")
KIT = {
    "name": "Automatos",
    "tagline": "An operating system for autonomous agent teams",
    "primary_color": "#e96235",
    "secondary_color": "#1a1714",
    "palette": {"accent_2": "#90af5a"},
    "text_color": "#f0e8db",
    "font_family": "Geist, sans-serif",
}
FONT_A = {"family": "Brand Sans", "weight": 400, "style": "normal", "data_uri": "data:font/woff2;base64,d09GMkFBQUE="}
FONT_B = {"family": "Brand Sans", "weight": 700, "style": "italic", "data_uri": "data:font/woff2;base64,d09GMkJCQkI="}
PATTERNS = ("_PLACEHOLDER", "_LEFTOVER", "_HEAD_CLOSE", "_STYLE_CLOSE", "_SCRIPT_BODY", "_TAG", "_EVENT_HANDLER")
HEAD = "composition.html"
STORY_SIZE = "1080x1920"  # the seeded templates' one 9:16 size, at the safe zone's own scale
SAFE_ZONE = ":root { --story-top: 250px; --story-bottom: 340px; }"


def _one_still_images():
    return [s for s in social_starters(SOCIAL_IMAGE) if len(s["blocks"].get("stills") or []) <= 1]


def _photo_spots(blocks):
    return tuple(name for name, spec in (blocks.get("slots") or {}).items() if spec.get("kind") == IMAGE_SLOT)


PAGE_CASES = [
    pytest.param(starter["slug"], size, story, photos, id=f"{starter['slug']}-{size}-{'story' if story else 'feed'}-{'photos' if photos else 'empty'}")
    for starter in _one_still_images()
    for size in starter["blocks"]["sizes"]
    for story in (False, True)
    for photos in (False, True)
]


def test_the_cases_cover_the_seeded_image_templates():
    slugs = {starter["slug"] for starter in _one_still_images()}
    assert len(slugs) >= 13 and "quote-card" in slugs and "photo-headline" in slugs and "before-after" in slugs
    assert "carousel" not in slugs  # several slides: it keeps its rendered preview (instant_preview.ONE_STILL_ONLY)


@pytest.mark.parametrize("slug, size, story, photos", PAGE_CASES)
def test_the_page_is_the_renderers_fill_of_the_same_template_and_values(slug, size, story, photos):
    starter = next(s for s in social_starters(SOCIAL_IMAGE) if s["slug"] == slug)
    template = SimpleNamespace(id=uuid.uuid4(), format=SOCIAL_IMAGE, blocks=starter["blocks"])
    request = instant_preview.PreviewRequest(
        values=starter["sample_data"], size=size, story=story, photos=_photo_spots(starter["blocks"]) if photos else (),
    )
    parts = instant_preview.page_parts(template, KIT, request, workspace_id=WS, fallback_name="Acme")
    brand_css = renderer.brand_stylesheet(parts.tokens, []) + instant_preview.FONTS_MARKER
    theirs = renderer.build_document(parts.html, parts.css, parts.variables, brand_css)
    page = instant_preview.preview_page(template, KIT, request, workspace_id=WS, fallback_name="Acme")
    assert page == instant_preview.with_page_head(theirs)
    # Not vacuous: the brand styles are there, nothing is left unfilled, and a story's 9:16 has its
    # safe zone (every still template reads --story-top in its own css; only a story's sets it).
    assert brand_css in page and "{{" not in page
    assert (SAFE_ZONE in page) == (story and size == STORY_SIZE)


VALUES = [
    "plain words",
    '<script>alert("x")</script>',
    "Tom & Jerry's \"best\" <b>",
    "a | b",
    "",
    "ünïcödé — “quotes” ‘and’ ©",
    "&amp; already escaped",
    0,
    42,
    -3.5,
    1e21,
    True,
    False,
]
FILL_TEXT = (
    '<p title="{{ v }}">{{ v }}</p><script>var n = {{ n }};</script>'
    '<b onclick="f({{ n }})" data-x=\'{{ v }}\'>{{v}}</b><style>p::after { content: "{{ v }}"; }</style>'
)


@pytest.mark.parametrize("value", VALUES, ids=[repr(value) for value in VALUES])
def test_the_fill_is_the_renderers_value_by_value(value):
    for variables in ({"v": value, "n": 7}, {"v": "x", "n": value} if not isinstance(value, str) else {"v": value, "n": 1.5}):
        assert ours.fill_placeholders(FILL_TEXT, variables, HEAD) == renderer.fill_placeholders(FILL_TEXT, variables, HEAD)


REFUSALS = [
    ("<p>{{ missing }}</p>", {}),
    ("<p>{{ a }}{{ b }}</p>", {"a": ["not", "a", "value"]}),
    ("<script>var t = '{{ word }}';</script>", {"word": "hi"}),
    ("<b onclick=\"go('{{ word }}')\">x</b>", {"word": "hi"}),
    ("<b onmouseover={{ word }}>x</b>", {"word": "hi"}),
    ("<script>{{ word }}</script\t\n bar>", {"word": "hi"}),
    ('<script>const s="</script-x>"; const x=`{{word}}`;</script>', {"word": "${alert(1)}"}),
    ("<p>{{ not-a-name }}</p>", {}),
]


@pytest.mark.parametrize("text, variables", REFUSALS)
def test_every_refusal_of_the_fill_is_the_renderers_word_for_word(text, variables):
    with pytest.raises(BundleError) as theirs:
        renderer.fill_placeholders(text, variables, HEAD)
    with pytest.raises(ours.CompositionError) as mine:
        ours.fill_placeholders(text, variables, HEAD)
    assert str(mine.value) == str(theirs.value)


DOCUMENTS = [
    ("<html><head></head><body>{{ v }}</body></html>", "p { color: red; } </style>", ""),
    ("<html><head></head><body></body></html>", "", ":root{--brand-x:1;}</STYLE >"),
    ("<html><head></head><head></head><body></body></html>", "", ""),
    ("<html><body>{{ v }}</body></html>", "", ""),
    ("<html><head><title>{{ v }}</title></head><body>{{ v }}</body></html>", "p::before { content: '{{ v }}'; }", ":root{--brand-x:1;}"),
]


@pytest.mark.parametrize("document, css, brand_css", DOCUMENTS)
def test_build_document_is_the_renderers_in_what_it_makes_and_what_it_refuses(document, css, brand_css):
    variables = {"v": "<i>it's</i>"}
    try:
        theirs = renderer.build_document(document, css, variables, brand_css)
    except BundleError as refused:
        with pytest.raises(ours.CompositionError) as mine:
            ours.build_document(document, css, variables, brand_css)
        assert str(mine.value) == str(refused)
        return
    assert ours.build_document(document, css, variables, brand_css) == theirs


@pytest.mark.parametrize("element", sorted(renderer.FORBIDDEN_ELEMENTS) + ["refresh"])
def test_a_forbidden_element_is_named_as_the_renderer_names_it(element):
    held = '<meta http-equiv="Refresh" content="0">' if element == "refresh" else f"<{element}></{element}>"
    document = f'<!doctype html><html><head></head><body><div data-composition-id="main">{held}<embed></div></body></html>'
    with pytest.raises(BundleError) as theirs:
        renderer.inspect_document(document, known_paths=frozenset(), max_duration=60)
    assert f"the composition contains {', '.join(ours.forbidden_elements(document))}" == str(theirs.value)


def test_a_seeded_page_holds_nothing_forbidden():
    starter = next(s for s in social_starters(SOCIAL_IMAGE) if s["slug"] == "photo-headline")
    template = SimpleNamespace(id=uuid.uuid4(), format=SOCIAL_IMAGE, blocks=starter["blocks"])
    request = instant_preview.PreviewRequest(values=starter["sample_data"], photos=("photo",))
    assert ours.forbidden_elements(instant_preview.preview_page(template, KIT, request, workspace_id=WS)) == []


def test_the_brand_tokens_and_a_font_face_are_written_as_the_renderer_writes_them():
    tokens = {"primary": "#e96235", "paper": "#f1e9dd", "body-font": "Geist, sans-serif", "display-scale": "1.05"}
    assert ours.token_stylesheet(tokens) == renderer.brand_stylesheet(tokens, [])
    assert ours.token_stylesheet({}) == renderer.brand_stylesheet({}, []) == ""
    for extension in ("woff2", "woff", "ttf", "otf"):
        path = f"assets/brand/fonts/font-3.{extension}"
        face = renderer.FontFace(family="Brand Sans", weight="700", style="italic", path=path)
        assert ours.font_face_rule("Brand Sans", "700", "italic", path, extension) == renderer.brand_stylesheet({}, [face])


def test_the_kits_faces_are_the_ones_a_render_stages_with_each_file_inline():
    kit = {**KIT, "font_files": [FONT_A, FONT_B]}
    starter = next(s for s in social_starters(SOCIAL_IMAGE) if s["slug"] == "quote-card")
    values = resolve_variables(starter["blocks"]["variables_schema"], starter["sample_data"]).values
    bundle = build_bundle(workspace_id=WS, reference="t", blocks=starter["blocks"], values=values, brand_kit=kit, fmt=SOCIAL_IMAGE)
    staged = renderer.brand_stylesheet({}, [
        renderer.FontFace(family=face["family"], weight=face["weight"], style=face["style"], path=face["path"])
        for face in bundle["brand"]["fonts"]
    ])
    inline = instant_preview.font_faces(kit)
    for entry in bundle["files"]:
        inline = inline.replace(f'url("{entry["data_uri"]}")', f'url("{entry["path"]}")')
    assert inline == staged and staged.count("@font-face") == 2


def test_the_patterns_and_constants_are_the_renderers():
    for name in PATTERNS:
        mine, theirs = getattr(ours, name), getattr(renderer, name)
        assert (mine.pattern, mine.flags) == (theirs.pattern, theirs.flags), name
    assert ours.GSAP_PATH == renderer.GSAP_PATH and ours.MIX_PATH == renderer.MIX_PATH
    assert ours.FORBIDDEN_ELEMENTS == renderer.FORBIDDEN_ELEMENTS
