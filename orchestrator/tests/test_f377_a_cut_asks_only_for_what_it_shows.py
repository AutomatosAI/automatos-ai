"""F377 (night 11, 7 Oct) — a video's shorter cut asks only for the fields it shows.

Night 11: every video draft from the composer failed "fill in … before rendering" (11 to 44
fields), and the UI story's 15 s cut still demanded the fields of the stretches it drops.

Pinned (``core.social_cuts``):

* ``shown_variables``: a variable shows at a length when its ``{{ name }}`` sits outside every
  clip the cut hides (scripts included), or in a voice line the cut keeps; the authored length
  shows them all (``None``);
* ``fields_cut_out`` names, per cut, the variables it never shows, and ``schema_at_length`` gives
  each of them a neutral default (an empty text, false, 0 inside the number's bounds), keeping
  every other spec as it is;
* the render (``cut_to_length`` through ``render.bundle_for``) takes a 15 s post with only its
  shown fields, and the bundle still sets every ``{{ name }}`` media-render fills; the authored
  length still refuses a field left empty;
* the composer's follow-up (``compose._missing``) and the agents' and Studio's requirements
  (``field_requirements.social_requirements``) ask the same: the cut's own fields;
* the seeded UI story and App promo need fewer fields at 15 s than at their full length, and a
  15 s post that fills only those renders.
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
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from core.social_cuts import (  # noqa: E402
    cut_to_length,
    fields_cut_out,
    neutral_value,
    schema_at_length,
    schema_for_length,
    shown_variables,
)
from core.social_templates import SOCIAL_VIDEO, placeholders, resolve_variables, validate_social_blocks  # noqa: E402
from modules.documents import social_starters  # noqa: E402
from modules.documents.field_requirements import social_requirements  # noqa: E402
from modules.socials import compose, render  # noqa: E402

HTML = (
    '<!doctype html><html><head><script src="assets/vendor/gsap.min.js"></script></head><body>'
    '<div id="root" data-composition-id="main" data-start="0" data-duration="40" data-width="1080" data-height="1920">'
    '<div id="bg" class="clip" data-start="0" data-duration="40" data-track-index="0"><p>{{ kicker }}</p></div>'
    '<section id="a" class="clip" data-start="0" data-duration="10" data-track-index="1"><h1>{{ headline }}</h1></section>'
    '<section id="b" class="clip" data-start="10" data-duration="15" data-track-index="1">'
    '<div><p title="{{ stat_label }}">{{ stat_value }}</p><br><span>{{ stat_note }}</span></div>'
    '</section>'
    '<section id="c" class="clip" data-start="25" data-duration="15" data-track-index="1"><p>{{ end_line }}</p></section>'
    '<audio id="mix" src="assets/audio/mix.wav" data-start="0" data-duration="40" data-track-index="10"></audio>'
    '</div><script>window.__timelines = {}; const BARS = {{ bars }};</script></body></html>'
)
SCHEMA = {
    "kicker": {"type": "text", "default": ""},
    "headline": {"type": "text"},
    "stat_value": {"type": "text"},
    "stat_label": {"type": "text"},
    "stat_note": {"type": "text"},
    "spoken": {"type": "text"},
    "said_late": {"type": "text"},
    "count": {"type": "number", "min": 2, "max": 9},
    "shown_flag": {"type": "boolean"},
    "bars": {"type": "number"},
    "end_line": {"type": "text"},
}
BLOCKS = {
    "html": HTML.replace("{{ end_line }}", "{{ end_line }}{{ count }}{{ shown_flag }}"),
    "css": "",
    "variables_schema": SCHEMA,
    "sizes": ["1080x1920"],
    "durations": [15, 40],
    "cuts": {"15": [[0, 5], [30, 40]]},
    "audio_plan": {"voice": {"lines": [
        {"id": "l01", "at": 0.5, "text": "{{ spoken }}"},
        {"id": "l02", "at": 12.0, "text": "{{ said_late }}"},
    ]}},
}
SHOWN_AT_15 = ["kicker", "headline", "spoken", "count", "shown_flag", "bars", "end_line"]
CUT_OUT_AT_15 = ["stat_value", "stat_label", "stat_note", "said_late"]
VALUES = {"headline": "A record month.", "spoken": "Here is our month.", "count": 4, "shown_flag": True, "bars": 3,
          "end_line": "See you soon."}


def _post(values, length):
    variables = {name: {"value": value} for name, value in values.items()}
    return SimpleNamespace(id=uuid.uuid4(), workspace_id=uuid.uuid4(), variables=variables, length_seconds=length,
                           footage=None, music=None, voice=None, targets=[])


def _template(blocks):
    return SimpleNamespace(id=uuid.uuid4(), format=SOCIAL_VIDEO, blocks=blocks)


# ── what a length shows ─────────────────────────────────────────────────────


def test_a_cut_shows_the_fields_outside_its_hidden_clips_and_in_its_kept_lines():
    validate_social_blocks(BLOCKS, SOCIAL_VIDEO)
    assert shown_variables(BLOCKS, 15) == SHOWN_AT_15
    assert shown_variables(BLOCKS, 40) is None  # the authored timeline shows everything
    assert fields_cut_out(BLOCKS) == {"15": CUT_OUT_AT_15}
    assert fields_cut_out({"durations": [40]}) == {} and fields_cut_out(None) == {}


def test_the_fields_a_cut_leaves_out_get_a_neutral_default_and_the_rest_stay_as_they_are():
    at_15 = schema_at_length(SCHEMA, CUT_OUT_AT_15)
    assert {name: spec.get("default") for name, spec in at_15.items() if name not in SHOWN_AT_15} == {
        "stat_value": "", "stat_label": "", "stat_note": "", "said_late": "",
    }
    assert all(at_15[name] == SCHEMA[name] for name in SHOWN_AT_15)
    assert schema_at_length(SCHEMA, ()) == SCHEMA
    assert neutral_value({"type": "number", "min": 2}) == 2 and neutral_value({"type": "number", "max": -1}) == -1
    assert neutral_value({"type": "number"}) == 0 and neutral_value({"type": "boolean"}) is False
    assert "default" not in SCHEMA["stat_value"]  # the template's own schema is untouched


# ── the render ──────────────────────────────────────────────────────────────


def test_a_15_second_post_renders_with_only_the_fields_its_cut_shows():
    bundle = render.bundle_for(_post(VALUES, 15), _template(BLOCKS), {})
    variables = bundle["variables"]
    # media-render fills every {{ name }} in the page, hidden clips included: each has a value.
    page = bundle["composition"]["html"]
    assert all(name in variables for name in placeholders(page) if not name.startswith(("brand.", "size.")))
    assert (variables["stat_value"], variables["stat_note"], variables["said_late"]) == ("", "", "")
    assert variables["headline"] == "A record month." and variables["bars"] == 3
    assert [line["id"] for line in bundle["audio"]["voice"]["lines"]] == ["l01"]


def test_the_full_length_still_asks_for_every_field():
    with pytest.raises(render.NotRenderable, match="fill in stat_value, stat_label, stat_note, said_late"):
        render.bundle_for(_post(VALUES, 40), _template(BLOCKS), {})
    with pytest.raises(render.NotRenderable, match="fill in headline"):
        render.bundle_for(_post({k: v for k, v in VALUES.items() if k != "headline"}, 15), _template(BLOCKS), {})


def test_cut_to_length_carries_the_schema_of_its_length():
    assert cut_to_length(BLOCKS, 15)["variables_schema"] == schema_at_length(SCHEMA, CUT_OUT_AT_15)
    assert cut_to_length(BLOCKS, 40)["variables_schema"] == SCHEMA


# ── the composer and the requirements ──────────────────────────────────────


def test_the_composer_asks_only_for_the_cuts_own_fields():
    entry = {"variables_schema": SCHEMA, "fields_cut_out": fields_cut_out(BLOCKS)}
    supplied = {"headline": {"value": "A record month."}}
    proposal = {"template": entry, "variables": supplied, "length_seconds": 15}
    assert compose._missing(proposal) == ["spoken", "count", "shown_flag", "bars", "end_line"]
    whole = compose._missing({**proposal, "length_seconds": None})
    assert whole == ["stat_value", "stat_label", "stat_note", "spoken", "said_late", "count", "shown_flag", "bars", "end_line"]
    assert schema_for_length(entry, 40) == SCHEMA  # a length that is not a cut asks for everything


def test_the_requirements_name_each_cuts_own_fields():
    answer = social_requirements(BLOCKS)
    assert "data.stat_value" in answer["required_fields"]
    assert answer["required_by_length"] == {
        "15": ["data.bars", "data.count", "data.end_line", "data.headline", "data.shown_flag", "data.spoken"],
    }
    assert "required_by_length" not in social_requirements({"variables_schema": SCHEMA})


# ── the seeded cuts ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("slug", ["ui-story-promo", "app-promo"])
def test_a_seeded_15_second_cut_needs_fewer_fields_and_renders_with_only_those(slug):
    starter = next(s for s in social_starters.social_starters() if s["slug"] == slug)
    blocks = starter["blocks"]
    every = resolve_variables(blocks["variables_schema"], {}).missing
    at_15 = resolve_variables(cut_to_length(blocks, 15)["variables_schema"], {}).missing
    assert at_15 and len(at_15) < len(every), (slug, len(at_15), len(every))
    filled = {name: value for name, value in starter["sample_data"].items() if name in at_15}
    bundle = render.bundle_for(_post(filled, 15), _template(blocks), {})
    assert bundle["composition"]["html"] and all(line["at"] < 15 for line in bundle["audio"]["voice"]["lines"])
    with pytest.raises(render.NotRenderable, match="fill in"):
        render.bundle_for(_post(filled, max(blocks["durations"])), _template(blocks), {})
    social_starters._starters.cache_clear()
