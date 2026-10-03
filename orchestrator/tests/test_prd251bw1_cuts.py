"""PRD-251B Wave 1, US-B104 (second half) — the 15 s and 30 s cuts.

Pinned:

* the contract: ``cuts`` maps a declared length to the stretches of the authored timeline
  it keeps, in order and inside the timeline, adding up to the length; a declared length
  more than a second shorter than the timeline needs one; a cut names only declared
  lengths; an image has none;
* ``cut_to_length``: the root carries the length and the stretches; every clip moves
  through the cut, and a clip outside it is hidden but stays in the page; scripts are left
  alone; the runtime that moves the GSAP tweens sits after the GSAP the page loads and
  before its own script; voice lines, SFX and snapshot moments move or drop; a length
  without a cut only sets the root's duration;
* a post that chose a length renders that cut, reserves that many seconds, and makes no
  footage for a slot the cut never shows;
* UI story promo and App promo declare a 15 s and a 30 s cut: every kept line is spoken
  inside the cut, the end card's line is among them with time to finish, and no clip runs
  past the end.
"""
from __future__ import annotations

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
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from core.media_render_quota import declared_seconds  # noqa: E402
from core.social_cuts import (  # noqa: E402
    CUT_RUNTIME,
    HIDDEN_ATTRIBUTE,
    cut_audio_plan,
    cut_moments,
    cut_to_length,
    place,
    slots_cut_out,
)
from core.social_templates import (  # noqa: E402
    BLOCK_KEYS,
    SOCIAL_IMAGE,
    SOCIAL_VIDEO,
    SocialTemplateError,
    root_duration,
    validate_social_blocks,
    with_root_duration,
)
from modules.documents import social_starters  # noqa: E402
from modules.socials import render  # noqa: E402

HTML = (
    '<!doctype html><html><head><script src="assets/vendor/gsap.min.js"></script></head><body>'
    '<div id="root" data-composition-id="main" data-start="0" data-duration="40" data-width="1080" data-height="1920">'
    '<div id="bg" class="clip" data-start="0" data-duration="40" data-track-index="0"></div>'
    '<section id="a" class="clip" data-start="0" data-duration="10" data-track-index="1"><h1>{{ headline }}</h1></section>'
    '<section id="b" class="clip scene" data-start="10" data-duration="10" data-track-index="1"></section>'
    '<section id="c" class="clip" data-start="20" data-duration="20" data-track-index="1"></section>'
    '<video id="v" class="clip" data-slot="mid" src="assets/slots/mid.mp4" data-start="12" data-duration="6" '
    'data-track-index="2" muted></video>'
    '<audio id="mix" src="assets/audio/mix.wav" data-start="0" data-duration="40" data-track-index="10"></audio>'
    "</div><script>var s = '<i data-start=\"12\" data-duration=\"1\">'; window.__timelines = {};</script></body></html>"
)
STRETCHES = [[0, 5], [25, 35]]
VIDEO = {
    "html": HTML.replace('<video id="v" class="clip" data-slot="mid"', '<video id="v" class="clip"'),
    "css": "",
    "variables_schema": {"headline": {"type": "text"}},
    "sizes": ["1080x1920"],
    "durations": [15, 40],
    "cuts": {"15": STRETCHES},
}
BLOCKS = {**VIDEO, "html": HTML}


def _errors(blocks, fmt=SOCIAL_VIDEO):
    with pytest.raises(SocialTemplateError) as caught:
        validate_social_blocks(blocks, fmt)
    return {e["field"]: e["message"] for e in caught.value.errors}


def _tag(html, element_id):
    return re.search(rf'<[a-z]+ id="{element_id}"[^<>]*>', html).group(0)


def _attr(tag, name):
    found = re.search(rf'\s{name}="([^"]*)"', tag)
    return found.group(1) if found else None


class _StartTags(HTMLParser):
    """Every element's attributes, in order. A script's text is data to the parser, never markup."""

    def __init__(self):
        super().__init__()
        self.found = []

    def handle_starttag(self, tag, attrs):
        if tag != "script":
            self.found.append(dict(attrs))


def _clips(html):
    """(start, duration) of every timed element but the root, outside the scripts."""
    parser = _StartTags()
    parser.feed(html)
    parser.close()
    return [
        (float(attrs["data-start"]), float(attrs["data-duration"]))
        for attrs in parser.found
        if "data-composition-id" not in attrs and attrs.get("data-start") is not None
    ]


# ── the contract ───────────────────────────────────────────────────────────


def test_cuts_is_part_of_the_contract():
    assert "cuts" in BLOCK_KEYS
    assert validate_social_blocks(VIDEO, SOCIAL_VIDEO)["cuts"] == {"15": STRETCHES}


def test_a_length_shorter_than_the_timeline_needs_its_cut():
    message = _errors({key: value for key, value in VIDEO.items() if key != "cuts"})["cuts"]
    assert "15 s" in message and "40 s" in message
    # Within a second of the timeline, a length plays it as authored.
    near = {**VIDEO, "html": VIDEO["html"].replace('data-duration="40" data-width', 'data-duration="39.5" data-width'),
            "durations": [39, 40]}
    near.pop("cuts")
    assert validate_social_blocks(near, SOCIAL_VIDEO)["durations"] == [39, 40]


@pytest.mark.parametrize("cut, field", [
    ([[25, 35], [0, 5]], "cuts.15[1]"),          # out of order
    ([[0, 10], [5, 10]], "cuts.15[1]"),          # overlapping
    ([[30, 45]], "cuts.15[0]"),                  # past the timeline
    ([[5, 5], [20, 30]], "cuts.15[0]"),          # ends where it starts
    ([[0, 5, 6]], "cuts.15[0]"),
    ([["0", 15]], "cuts.15[0]"),
    ([[True, 15]], "cuts.15[0]"),
    ([], "cuts.15"),
    ([[i, i + 1] for i in range(13)], "cuts.15"),  # more than 12 stretches
    ([[0, 5], [25, 34]], "cuts.15"),             # 14 s, not 15
])
def test_a_malformed_cut_is_refused(cut, field):
    assert field in _errors({**VIDEO, "cuts": {"15": cut}})


def test_a_cut_names_only_declared_lengths_and_is_an_object():
    assert "cuts.20" in _errors({**VIDEO, "cuts": {"15": STRETCHES, "20": [[0, 20]]}})
    assert "cuts" in _errors({**VIDEO, "cuts": [[0, 15]]})


def test_an_image_has_no_cuts():
    html = ('<!doctype html><html><head></head><body><div id="root" data-composition-id="main" data-width="1080" '
            'data-height="1080"><h1>{{ headline }}</h1></div></body></html>')
    image = {"html": html, "css": "", "variables_schema": {"headline": {"type": "text"}}, "sizes": ["1080x1080"],
             "cuts": {"15": STRETCHES}}
    assert "cuts" in _errors(image, SOCIAL_IMAGE)


# ── applying a cut ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("second, landing", [
    (0.0, (0.0, True)), (4.5, (4.5, True)), (5.0, (5.0, False)), (24.0, (5.0, False)),
    (25.0, (5.0, True)), (34.0, (14.0, True)), (35.0, (15.0, False)), (39.0, (15.0, False)),
])
def test_place_moves_a_second_through_the_cut(second, landing):
    assert place(second, [(0, 5), (25, 35)]) == landing


def test_the_root_carries_the_length_and_the_stretches():
    html = cut_to_length(BLOCKS, 15)["html"]
    root = _tag(html, "root")
    assert _attr(root, "data-duration") == "15"
    assert json.loads(_attr(root, "data-cut")) == STRETCHES


def test_every_clip_moves_through_the_cut():
    html = cut_to_length(BLOCKS, 15)["html"]
    assert (_attr(_tag(html, "bg"), "data-start"), _attr(_tag(html, "bg"), "data-duration")) == ("0", "15")
    assert (_attr(_tag(html, "a"), "data-start"), _attr(_tag(html, "a"), "data-duration")) == ("0", "5")
    assert (_attr(_tag(html, "c"), "data-start"), _attr(_tag(html, "c"), "data-duration")) == ("5", "10")
    assert (_attr(_tag(html, "mix"), "data-start"), _attr(_tag(html, "mix"), "data-duration")) == ("0", "15")
    assert all(start + duration <= 15 for start, duration in _clips(html))


def test_a_clip_outside_the_cut_is_hidden_but_stays_in_the_page():
    cut = cut_to_length(BLOCKS, 15)
    for element_id in ("b", "v"):
        tag = _tag(cut["html"], element_id)
        assert HIDDEN_ATTRIBUTE in tag
        assert _attr(tag, "data-start") is None and _attr(tag, "data-duration") is None
        assert _attr(tag, "data-track-index") is None and "clip" not in (_attr(tag, "class") or "").split()
    assert _attr(_tag(cut["html"], "b"), "class") == "scene"
    assert f"[{HIDDEN_ATTRIBUTE}]" in cut["css"] and "display:none" in cut["css"]
    assert slots_cut_out(BLOCKS, 15) == {"mid"} and slots_cut_out(BLOCKS, 40) == set()


def test_scripts_are_left_alone_and_the_runtime_sits_before_the_pages_own():
    html = cut_to_length(BLOCKS, 15)["html"]
    assert "var s = '<i data-start=\"12\" data-duration=\"1\">'" in html
    assert html.index('src="assets/vendor/gsap.min.js"') < html.index(CUT_RUNTIME) < html.index("window.__timelines = {}")
    assert "{{" not in CUT_RUNTIME  # media-render's placeholder filling never touches it


def test_a_length_without_a_cut_only_sets_the_roots_duration():
    assert cut_to_length(BLOCKS, 40) == {**BLOCKS, "html": with_root_duration(HTML, 40)}


def test_voice_lines_and_sfx_move_with_the_cut_and_music_stays():
    plan = {
        "voice": {"voice": "af_heart", "lines": [{"id": "l1", "at": 0.3, "text": "a"}, {"id": "l2", "at": 12.0, "text": "b"},
                                                 {"id": "l3", "at": 26.5, "text": "c"}]},
        "music": {"track": "where-the-night-begins", "start": 190.1},
        "sfx": [{"path": "assets/sfx/x.ogg", "at": 30.0}, {"path": "assets/sfx/y.ogg", "at": 7.0}],
    }
    cut = cut_audio_plan(plan, [(0, 5), (25, 35)])
    assert [(line["id"], line["at"]) for line in cut["voice"]["lines"]] == [("l1", 0.3), ("l3", 6.5)]
    assert cut["sfx"] == [{"path": "assets/sfx/x.ogg", "at": 10.0}]
    assert cut["music"] == plan["music"] and plan["voice"]["lines"][2]["at"] == 26.5  # the plan itself is untouched
    assert "voice" not in cut_audio_plan({"voice": {"lines": [{"id": "l2", "at": 12.0, "text": "b"}]}}, [(0, 5)])


def test_snapshot_moments_move_with_the_cut():
    assert cut_moments(BLOCKS, 15, [1, 6, 26, 36]) == [1.0, 6.0]
    assert cut_moments(BLOCKS, 40, [1, 39, 41]) == [1.0, 39.0]


# ── a post that chose a length ─────────────────────────────────────────────


def _starter(slug):
    return next(s for s in social_starters.social_starters() if s["slug"] == slug)


def _post(starter, length):
    values = {name: {"value": value} for name, value in starter["sample_data"].items()}
    return SimpleNamespace(id=uuid.uuid4(), workspace_id=uuid.uuid4(), variables=values, length_seconds=length, footage=None)


@pytest.mark.parametrize("slug", ["ui-story-promo", "app-promo"])
@pytest.mark.parametrize("length", [15, 30])
def test_a_post_renders_and_reserves_its_cut(slug, length):
    starter = _starter(slug)
    template = SimpleNamespace(id=uuid.uuid4(), format=SOCIAL_VIDEO, blocks=starter["blocks"])
    post = _post(starter, length)
    bundle = render.bundle_for(post, template, {})
    html = bundle["composition"]["html"]
    assert root_duration(html) == float(length) and "data-cut=" in html
    assert all(start + duration <= length + 1e-6 for start, duration in _clips(html))
    assert all(line["at"] < length for line in bundle["audio"]["voice"]["lines"])
    assert declared_seconds(render.at_chosen_length(starter["blocks"], post), SOCIAL_VIDEO) == length


def test_no_footage_is_made_for_a_slot_the_cut_never_shows(monkeypatch):
    starter = _starter("app-promo")
    template = SimpleNamespace(id=uuid.uuid4(), format=SOCIAL_VIDEO, blocks=starter["blocks"])
    asked = []
    monkeypatch.setattr(render.footage_recipes, "plan_for", lambda footage, *a, **k: asked.append(dict(footage)))
    footage = {"focus": {"prompt": "a moment of concentration"}, "desk": {"prompt": "someone studying"}}
    post = SimpleNamespace(**{**vars(_post(starter, 15)), "footage": footage})
    render.footage_plan_for(post, template, caps=None)
    post.length_seconds = 30
    render.footage_plan_for(post, template, caps=None)
    assert asked == [{"desk": footage["desk"]}, footage]  # focus plays 17.1-21.1 s: outside the 15 s cut


# ── the seeded cuts ────────────────────────────────────────────────────────


@pytest.mark.parametrize("slug, authored", [("ui-story-promo", 40), ("app-promo", 38)])
def test_the_starters_declare_a_15_and_a_30_second_cut(slug, authored):
    blocks = _starter(slug)["blocks"]
    assert blocks["durations"] == [15, 30, authored] and set(blocks["cuts"]) == {"15", "30"}
    last_line = blocks["audio_plan"]["voice"]["lines"][-1]["id"]
    for length in (15, 30):
        cut = cut_to_length(blocks, length)
        lines = cut["audio_plan"]["voice"]["lines"]
        starts = [line["at"] for line in lines]
        assert starts == sorted(starts) and all(at < length for at in starts), (slug, length)
        assert all(b - a >= 1.0 for a, b in zip(starts, starts[1:])), (slug, length)  # one line after another
        assert lines[-1]["id"] == last_line and length - lines[-1]["at"] >= 3.0, (slug, length)  # the end card is said
        assert all(start + duration <= length + 1e-6 for start, duration in _clips(cut["html"])), (slug, length)
        assert cut_moments(blocks, length, _starter(slug)["preview"]["at"]), (slug, length)
    assert cut_to_length(blocks, authored)["html"] == with_root_duration(blocks["html"], authored)
    social_starters._starters.cache_clear()
