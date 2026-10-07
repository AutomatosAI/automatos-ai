"""F377 (night 11, 7 Oct) — every voice line a length keeps has room to be spoken.

Night 11: the UI story's 15 s retake failed "line l02 is still speaking at 4.19 s, when line
l05 starts … shorten the line": its cut gave l02 under two seconds, and the field's limit
allowed seven words. A field's limit was never tied to the time its line gets.

Pinned, for every seeded video and every length it offers (each cut and the authored timeline):

* a kept line's window runs from its start to the next kept line's start, after the cut (the
  last line's to the end), as media-render's fit reads it (``media_render.fit.window_ends``);
* the longest the line can be (its fields' ``max_chars`` at the composer's 2.5 words a second,
  sped up as far as media-render's voice fit allows, ``voice_max_tempo``, ending
  ``voice_fit_gap_seconds`` early) fits that window, so no field filled within its limit can
  fail the voice check;
* when the voice check does refuse a line, the owner is told which fields it speaks
  (``modules/socials/spoken_fields.py``), never its line id.
"""
from __future__ import annotations

import asyncio
import math
import os
import re
import sys
import uuid
from pathlib import Path

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
# media-render's settings and fit (standard library only), found LAST so no orchestrator module is shadowed.
if str(_MEDIA_RENDER) not in sys.path:
    sys.path.append(str(_MEDIA_RENDER))

from core.social_cuts import cut_to_length  # noqa: E402
from core.social_templates import NUMBER, is_bundle_variable, placeholders, voice_lines  # noqa: E402
from media_render.config import load_settings  # noqa: E402
from media_render.fit import Timed, window_ends  # noqa: E402
from modules.documents import social_starters  # noqa: E402
from modules.socials import render, spoken_fields  # noqa: E402
from modules.socials.compose import WORDS_PER_SECOND  # noqa: E402

SETTINGS = load_settings({})
# A spoken word with the space after it: English averages 4.7 letters a word.
CHARS_PER_WORD = 6
# A figure said aloud ("two hundred and twelve"), and a brand's name.
NUMBER_WORDS = 4
BRAND_WORDS = 4
_WORD = re.compile(r"\w")


def _videos():
    starters = social_starters.social_starters("social_video")
    social_starters._starters.cache_clear()
    return starters


def _lengths():
    return [(s["slug"], length) for s in _videos() for length in s["blocks"]["durations"]]


def _longest_words(text, schema):
    """The most words a line can say: each field at its limit, the brand's name, its own words."""
    words = 0
    for name in placeholders(text):
        if is_bundle_variable(name):
            words += BRAND_WORDS
        elif schema[name]["type"] == NUMBER:
            words += NUMBER_WORDS
        else:
            assert "max_chars" in schema[name], f"{name} is spoken: it needs a limit"
            words += math.ceil(schema[name]["max_chars"] / CHARS_PER_WORD)
    own = re.sub(r"\{\{[^{}]*\}\}", " ", text).split()
    return words + sum(1 for word in own if _WORD.search(word))


def _seconds_needed(words):
    """The shortest a line of ``words`` can be made: at the composer's pace, at the fit's top speed, with its gap."""
    return words / WORDS_PER_SECOND / SETTINGS.voice_max_tempo + SETTINGS.voice_fit_gap_seconds


@pytest.mark.parametrize("slug, length", _lengths())
def test_every_kept_line_has_room_for_its_longest_words(slug, length):
    starter = next(s for s in _videos() if s["slug"] == slug)
    blocks = cut_to_length(starter["blocks"], length)
    schema = starter["blocks"]["variables_schema"]
    lines = [line for line in voice_lines(blocks["audio_plan"]) if isinstance(line.get("text"), str)]
    assert lines, (slug, length)
    ends = window_ends([Timed(id=line["id"], at=float(line["at"]), seconds=None) for line in lines], float(length))
    short = {}
    for line in lines:
        window = ends[line["id"]] - float(line["at"])
        needed = _seconds_needed(_longest_words(line["text"], schema))
        if needed > window:
            short[line["id"]] = (round(window, 2), round(needed, 2))
    assert short == {}, f"{slug} at {length} s: line (window s, needed s) {short}"


def test_the_ui_storys_15_second_cut_gives_its_problem_line_room():
    """The night-11 failure: l02's window in the 15 s cut is past the 2 s it had."""
    starter = next(s for s in _videos() if s["slug"] == "ui-story-promo")
    lines = {line["id"]: line["at"] for line in cut_to_length(starter["blocks"], 15)["audio_plan"]["voice"]["lines"]}
    assert lines["l05"] - lines["l02"] > 2.3


# ── the owner's words for a refused line ──────────────────────────────────


OVERLAP = {
    "section": "audio", "severity": "error", "code": "voice_lines_overlap", "line": "l05",
    "message": "line l02 is still speaking at 4.19 s, when line l05 starts: it lasts 3.10 s and its window is 1.97 s, "
               "more than 1.25x speed can fit; shorten the line",
}


def _labels():
    starter = next(s for s in _videos() if s["slug"] == "ui-story-promo")
    return spoken_fields.spoken_labels(starter["blocks"])


def test_each_line_is_mapped_to_the_labels_of_the_fields_it_speaks():
    labels = _labels()
    schema = next(s for s in _videos() if s["slug"] == "ui-story-promo")["blocks"]["variables_schema"]
    assert labels["l02"] == (schema["problem_line"]["label"], schema["problem_accent"]["label"])
    assert labels["l10"] == (schema["end_line"]["label"], schema["end_accent"]["label"])  # the brand's name is no field
    assert spoken_fields.spoken_labels(None) == {}


def test_a_refused_line_is_told_by_its_fields_never_its_line_id():
    labels = {"l02": ("Problem: the line", "Problem: the accent words"), "l09": ("Team: the line about the team",)}
    failure = render.RenderFailure("check_failed", "The composition failed its check with 1 error(s): line l02 …",
                                   {"findings": [OVERLAP]})
    named = spoken_fields.named(failure, labels)
    assert isinstance(named, render.RenderFailure) and named.code == "check_failed"
    assert named.message == (
        "The spoken 'Problem: the line' and 'Problem: the accent words' are too long for their moment: shorten them, "
        "or choose a longer video. Nothing was rendered."
    )
    assert "l02" not in named.message and named.report["findings"] == [OVERLAP]
    overrun = {"code": "voice_line_overruns", "line": "l09", "message": "line l09 runs from 27.94 s to 41.10 s, past the end"}
    assert spoken_fields.owner_message({"findings": [overrun]}, labels) == (
        "The spoken 'Team: the line about the team' is too long for its moment: shorten it, "
        "or choose a longer video. Nothing was rendered."
    )
    other = render.RenderFailure("check_failed", "Two text blocks overlap.", {"findings": [{"code": "text_overlap"}]})
    assert spoken_fields.named(other, labels) is other


def test_the_render_tells_the_owner_the_fields(monkeypatch):
    finished = []

    async def refused(*_args, **_kwargs):
        raise render.RenderFailure("check_failed", "The composition failed its check with 1 error(s): line l02 …",
                                   {"findings": [OVERLAP]})

    monkeypatch.setattr(render, "_render_sizes", refused)
    monkeypatch.setattr(render, "_finish", lambda factory, job, **kwargs: finished.append(kwargs["failure"]))
    job = render.RenderJob(post_id=uuid.uuid4(), workspace_id=uuid.uuid4(), actor="owner", content_hash="h", title="T",
                           format="video", bundle={}, spoken_fields=_labels())
    assert asyncio.run(render._render(job, None, None, lambda: None)) is False
    (failure,) = finished
    assert failure.message.startswith("The spoken '") and "l02" not in failure.message
    assert render.RenderJob(post_id=job.post_id, workspace_id=job.workspace_id, actor="a", content_hash="h", title="T",
                            format=None, bundle={}).spoken_fields == {}
