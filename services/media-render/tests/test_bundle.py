"""The render bundle's contract (US-102): what POST /render accepts, and what it
refuses with 400 before anything is fetched, spoken or rendered.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from helpers import STORAGE, bundle, data_uri, page
from media_render.bundle import parse_bundle
from media_render.fixture import fixture_bundle
from media_render.music import load_library
from media_render.validate import BundleError

SIGNED = "?X-Amz-Signature=deadbeef&X-Amz-Expires=600"


def refused(settings, body, library=None) -> str:
    with pytest.raises(BundleError) as caught:
        parse_bundle(body, settings, library or {})
    return str(caught.value)


def test_the_fixture_bundle_parses(settings):
    parsed = parse_bundle(fixture_bundle(), settings, {})
    assert parsed.workspace_id == "media-render-fixture"
    composition = parsed.composition
    assert (composition.duration, composition.width, composition.height, composition.aspect) == (3.0, 1080, 1920, "9:16")
    assert '<h1 id="title">Rendered on brand</h1>' in composition.html
    voice = parsed.audio.voice
    assert (voice.voice, voice.speed, voice.lang) == ("af_heart", 0.95, "en-us")
    assert [(line.id, line.at, line.text) for line in voice.lines] == [("l01", 0.3, "Rendered on brand, in three seconds.")]


def test_variables_are_escaped_text(settings):
    parsed = parse_bundle(bundle(variables={"headline": '<img src=x onerror="alert(1)"> & more'}), settings, {})
    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt; &amp; more" in parsed.composition.html
    assert "<img src=x" not in parsed.composition.html


def test_a_missing_variable_is_named(settings):
    message = refused(settings, bundle(variables={}))
    assert "headline" in message


@pytest.mark.parametrize(
    "markup",
    [
        "<script>const title = `{{ label }}`;</script>",
        '<div onclick="go({{ label }})"></div>',
        "<div onclick=go({{label}})></div>",
    ],
)
def test_text_never_lands_in_a_script_or_an_event_handler(settings, markup):
    body = bundle(composition={"html": page(markup)}, variables={"headline": "On brand", "label": "`);fetch(`//x"})
    assert "inside a <script> or an event handler" in refused(settings, body)


def test_numbers_and_flags_may_time_a_script(settings):
    markup = "<script>window.__cue = {{ scene_at }}; window.__loop = {{ loop }};</script>"
    body = bundle(composition={"html": page(markup)}, variables={"headline": "On brand", "scene_at": 2.4, "loop": False})
    assert "window.__cue = 2.4; window.__loop = false;" in parse_bundle(body, settings, {}).composition.html


def test_a_malformed_placeholder_is_refused(settings):
    body = bundle(composition={"html": page("<p>{{ 1bad }}</p>")})
    assert "placeholder" in refused(settings, body)


def test_brand_tokens_become_css_variables_before_head_closes(settings):
    parsed = parse_bundle(bundle(brand={"tokens": {"accent": "#3a7bd5", "heading-font": '"Geist", sans-serif'}}), settings, {})
    html = parsed.composition.html
    style = html.index(":root{--brand-accent:#3a7bd5;--brand-heading-font:\"Geist\", sans-serif;}")
    assert style < html.index("</head>")


@pytest.mark.parametrize("value", ["#fff; } body { display:none", "url(https://evil.example/x.png)", "</style><script>"])
def test_a_token_is_one_css_value(settings, value):
    assert "single CSS value" in refused(settings, bundle(brand={"tokens": {"accent": value}}))


def test_fonts_are_inlined_files_with_font_face_rules(settings):
    body = bundle(
        files=[{"path": "assets/brand/geist-700.woff2", "data_uri": data_uri("font/woff2", b"wOF2-font-bytes")}],
        brand={"fonts": [{"family": "Geist", "weight": 700, "path": "assets/brand/geist-700.woff2"}]},
    )
    parsed = parse_bundle(body, settings, {})
    assert parsed.files[0].data == b"wOF2-font-bytes"
    assert '@font-face{font-family:"Geist";src:url("assets/brand/geist-700.woff2") format("woff2");' in parsed.composition.html


def test_a_font_must_be_an_inlined_file(settings):
    body = bundle(brand={"fonts": [{"family": "Geist", "path": "assets/brand/missing.woff2"}]})
    assert "inlined in files" in refused(settings, body)


@pytest.mark.parametrize(
    "path, uri, expected",
    [
        ("assets/brand/logo.svg", "data:image/png;base64,AAAA", "media type image/png"),
        ("assets/brand/logo.svg", "data:image/svg+xml,<svg/>", "base64"),
        ("assets/audio/logo.svg", "data:image/svg+xml;base64,PHN2Zy8+", "stages itself"),
        ("assets/brand/.env.svg", "data:image/svg+xml;base64,PHN2Zy8+", "relative path"),
        ("assets/../index.svg", "data:image/svg+xml;base64,PHN2Zy8+", "relative path"),
        ("brand/logo.svg", "data:image/svg+xml;base64,PHN2Zy8+", "under assets/"),
        ("assets/brand/logo.exe", "data:image/svg+xml;base64,PHN2Zy8+", "must end in one of"),
    ],
)
def test_inline_files_are_checked(settings, path, uri, expected):
    assert expected in refused(settings, bundle(files=[{"path": path, "data_uri": uri}]))


def test_an_inline_file_over_the_size_limit_is_refused(settings):
    small = replace(settings, max_asset_bytes=16)
    body = bundle(files=[{"path": "assets/brand/logo.png", "data_uri": data_uri("image/png", b"x" * 64)}])
    assert "byte limit" in refused(small, body)


def test_a_media_url_on_our_storage_is_accepted(settings):
    body = bundle(media=[{"path": "assets/cine/hook.mp4", "url": STORAGE + "ws-a/post-1/hook.mp4" + SIGNED}])
    parsed = parse_bundle(body, settings, {})
    assert parsed.media[0].path == "assets/cine/hook.mp4"


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.example/automatos/hook.mp4",
        "https://media.example-storage.test/other-bucket/hook.mp4",
        "http://media.example-storage.test/automatos/hook.mp4",
        "https://media.example-storage.test:8443/automatos/hook.mp4",
        "https://user@media.example-storage.test/automatos/hook.mp4",
        "https://media.example-storage.test/automatos/../other/hook.mp4",
        "https://media.example-storage.test/automatos/%2e%2e/other/hook.mp4",
        "https://media.example-storage.test.evil.example/automatos/hook.mp4",
        "file:///etc/passwd",
    ],
)
def test_a_media_url_off_the_allowlist_is_refused(settings, url):
    message = refused(settings, bundle(media=[{"path": "assets/cine/hook.mp4", "url": url + SIGNED}]))
    assert "storage allowlist" in message
    assert "deadbeef" not in message, "a refusal must never echo a presigned signature"


def test_no_media_url_is_accepted_without_an_allowlist(settings):
    open_door = replace(settings, media_url_prefixes=())
    body = bundle(media=[{"path": "assets/cine/hook.mp4", "url": STORAGE + "hook.mp4"}])
    assert "storage allowlist" in refused(open_door, body)


@pytest.mark.parametrize(
    "markup, expected",
    [
        ('<img src="https://evil.example/x.png">', "points outside"),
        ('<img src="//evil.example/x.png">', "points outside"),
        ('<video src="assets/cine/missing.mp4"></video>', "does not provide"),
        ('<img src="/etc/passwd">', "relative to the composition"),
        ('<img src="assets\\\\x.png">', "backslash"),
        ('<iframe src="about:blank"></iframe>', "<iframe>"),
        ('<meta http-equiv="refresh" content="0;url=https://evil.example">', "refresh"),
        ('<div style="background: url(https://evil.example/x.png)"></div>', "points outside"),
        ("<style>@import url(theme.css);</style>", "@import"),
        ('<style>.a { background: image-set("https://evil.example/x.png" 1x); }</style>', "points outside"),
        ('<div data-composition-src="scenes/intro.html"></div>', "does not provide"),
    ],
)
def test_the_composition_never_reaches_outside_itself(settings, markup, expected):
    assert expected in refused(settings, bundle(composition={"html": page(markup)}))


def test_inline_svg_data_and_fragment_references_are_fine(settings):
    # The references' grain texture: an SVG data: URI whose own url(%23n) is not a reference.
    grain = "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg'><filter id='n'/><rect filter='url(%23n)'/></svg>\")"
    markup = (
        f"<style>.grain {{ background-image: {grain}; }} .glow {{ filter: url(#g); }}</style>"
        '<svg xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="g"/></defs>'
        '<rect fill="url(#g)"/><use href="#g"/></svg>'
    )
    parse_bundle(bundle(composition={"html": page(markup)}), settings, {})


def test_media_and_inline_files_can_be_referenced(settings):
    markup = '<img src="assets/brand/logo.svg"><video src="assets/cine/hook.mp4" muted></video>'
    body = bundle(
        composition={"html": page(markup)},
        files=[{"path": "assets/brand/logo.svg", "data_uri": data_uri("image/svg+xml", b"<svg/>")}],
        media=[{"path": "assets/cine/hook.mp4", "url": STORAGE + "hook.mp4"}],
    )
    parse_bundle(body, settings, {})


@pytest.mark.parametrize(
    "html, expected",
    [
        (page().replace('data-composition-id="main" ', ""), "exactly one root"),
        (page().replace('data-duration="3" data-width', 'data-duration="soon" data-width'), "numeric data-duration"),
        (page(duration=500), "at most"),
        (page().replace('data-width="1080"', 'data-width="1080.5"'), "whole numbers"),
        (page('<audio src="assets/audio/mix.wav"></audio>'), "exactly one <audio>"),
        (page().replace('src="assets/audio/mix.wav"', 'src="assets/cine/other.wav"'), "exactly one <audio>"),
        (page().replace("</head>", "</head></head>"), "exactly one </head>"),
    ],
)
def test_the_composition_root_and_mix_are_required(settings, html, expected):
    assert expected in refused(settings, bundle(composition={"html": html}))


def test_template_css_cannot_close_its_style(settings):
    body = bundle(composition={"html": page(), "css": "</style><script>alert(1)</script>"})
    assert "must not close" in refused(settings, body)


def test_unknown_fields_are_refused(settings):
    assert "unknown field(s) callback_url" in refused(settings, bundle(callback_url="https://x.example"))


def test_voice_lines_are_checked(settings):
    lines = [{"id": "l01", "at": 3.0, "text": "Too late."}]
    assert "must be under 3" in refused(settings, bundle(audio={"voice": {"lines": lines}}))
    both = [{"id": "l01", "at": 0.2, "text": "Hi.", "path": "assets/vo/l01.wav"}]
    assert "either text" in refused(settings, bundle(audio={"voice": {"lines": both}}))
    missing = [{"id": "l01", "at": 0.2, "path": "assets/vo/l01.wav"}]
    assert "listed in media" in refused(settings, bundle(audio={"voice": {"lines": missing}}))
    twice = [{"id": "l01", "at": 0.2, "text": "Hi."}, {"id": "l01", "at": 1.2, "text": "Again."}]
    assert "used twice" in refused(settings, bundle(audio={"voice": {"lines": twice}}))
    fast = {"lines": [{"id": "l01", "at": 0.2, "text": "Hi."}], "speed": 3}
    assert "at most 2" in refused(settings, bundle(audio={"voice": fast}))


def test_a_voice_file_from_storage_can_be_a_line(settings):
    body = bundle(
        media=[{"path": "assets/vo/l01.wav", "url": STORAGE + "vo/l01.wav"}],
        audio={"voice": {"lines": [{"id": "l01", "at": 0.2, "path": "assets/vo/l01.wav"}]}},
    )
    line = parse_bundle(body, settings, {}).audio.voice.lines[0]
    assert (line.path, line.text) == ("assets/vo/l01.wav", None)


def _library(tmp_path, duration):
    music = tmp_path / "music"
    music.mkdir()
    (music / "bed.mp3").write_bytes(b"not really audio")
    track = {
        "id": "deep-house-003", "file": "bed.mp3", "duration": duration, "title": "Deep House 003",
        "artist": "Sascha Ende", "licence": "CC-BY-4.0",
        "attribution": 'Music: "Deep House 003" by Sascha Ende (ende.app), licensed CC BY 4.0.',
    }
    manifest = {"tracks": [track]}
    (music / "manifest.json").write_text(json.dumps(manifest))
    return load_library(str(music))


def test_music_comes_from_the_library(settings, tmp_path):
    library = _library(tmp_path, 120.0)
    cue = parse_bundle(bundle(audio={"music": {"track": "deep-house-003", "start": 32.0}}), settings, library).audio.music
    assert (cue.track, cue.start, cue.fade_in, cue.fade_out) == ("deep-house-003", 32.0, 0.02, 1.7)
    # The cue carries the library's word on the track: the report's licence and credit line (S1.6).
    assert cue.about["credit_required"] is True and cue.about["licence"] == "CC BY 4.0"
    assert cue.about["attribution"] == 'Music: "Deep House 003" by Sascha Ende (ende.app), licensed CC BY 4.0.'
    assert "not in the music library" in refused(settings, bundle(audio={"music": {"track": "nope"}}), library)
    late = bundle(audio={"music": {"track": "deep-house-003", "start": 118.5}})
    assert "runs past its end" in refused(settings, late, library)


def test_sfx_come_from_storage_at_a_volume_up_to_one(settings):
    media = [{"path": "assets/sfx/click.ogg", "url": STORAGE + "sfx/click.ogg"}]
    cue = parse_bundle(bundle(media=media, audio={"sfx": [{"path": "assets/sfx/click.ogg", "at": 1.0}]}), settings, {})
    assert cue.audio.sfx[0].volume == 0.5
    loud = bundle(media=media, audio={"sfx": [{"path": "assets/sfx/click.ogg", "at": 1.0, "volume": 1.5}]})
    assert "at most 1" in refused(settings, loud)


def test_a_workspace_id_is_required(settings):
    assert "missing workspace_id" in refused(settings, {"composition": {"html": page()}})
    assert "workspace id" in refused(settings, bundle(workspace="../../etc"))


def test_a_preview_names_moments_inside_the_composition(settings):
    parsed = parse_bundle(bundle(preview={"at": [2.5, 0.5, 2.5]}), settings, {})
    assert parsed.preview.at == (0.5, 2.5)
    assert parse_bundle(bundle(), settings, {}).preview is None


@pytest.mark.parametrize(
    "preview, message",
    [
        pytest.param({"at": []}, "at least one moment", id="no-moments"),
        pytest.param({"at": [3.0]}, "under 3", id="at-the-end"),
        pytest.param({"at": [-1]}, "at least 0", id="before-the-start"),
        pytest.param({"at": ["1"]}, "must be a number", id="not-a-number"),
        pytest.param({"at": [0.5], "scale": 2}, "unknown field", id="unknown-key"),
        pytest.param({"frames": 5}, "missing at", id="no-at"),
        pytest.param("soon", "must be an object", id="not-an-object"),
        pytest.param({"at": [0.1] * 13}, "the limit is 12", id="too-many"),
    ],
)
def test_a_preview_outside_the_composition_is_refused(settings, preview, message):
    assert message in refused(settings, bundle(preview=preview))


def test_a_still_names_its_moments_in_order(settings):
    """US-107: an image, one full-size PNG per moment (a carousel's slides)."""
    parsed = parse_bundle(bundle(still={"at": [0.0, 1.5, 2.5]}), settings, {})
    assert parsed.still.at == (0.0, 1.5, 2.5) and parsed.preview is None
    assert parse_bundle(bundle(), settings, {}).still is None


@pytest.mark.parametrize(
    "extra, message",
    [
        pytest.param({"still": {"at": []}}, "at least one moment", id="no-moments"),
        pytest.param({"still": {"at": [3.0]}}, "under 3", id="at-the-end"),
        pytest.param({"still": {"at": [-1]}}, "at least 0", id="before-the-start"),
        pytest.param({"still": {"at": [2.0, 1.0]}}, "time order", id="out-of-order"),
        pytest.param({"still": {"at": [1.0, 1.0]}}, "time order", id="twice"),
        pytest.param({"still": {"at": [0.5], "size": "1080x1350"}}, "unknown field", id="unknown-key"),
        pytest.param({"still": {"at": [0.1 * i for i in range(11)]}}, "the limit is 10", id="too-many"),
        pytest.param({"still": {"at": [0.5]}, "preview": {"at": [0.5]}}, "its own preview", id="and-a-preview"),
        pytest.param(
            {"still": {"at": [0.5]}, "audio": {"voice": {"lines": [{"id": "l01", "at": 0.3, "text": "Hi"}]}}},
            "no sound",
            id="and-a-voice",
        ),
    ],
)
def test_a_still_it_could_not_take_is_refused(settings, extra, message):
    assert message in refused(settings, bundle(**extra))
