"""PRD-251D US-D302 (E11): the three software videos explain their fields, and every default is argued for.

The App promo, the Cinematic product promo and the UI story promo carry 87, 122 and 120 fields,
most of them once labelled in words only their author understood ("Phone: choice C", "Caption and
voice: beat 1") and with no description. Auto writes a field that has no default from the brief and
leaves a field with a default as it is, so a default prints the same words in every business's video.

Pinned, for each of the three:

* every field has a label and a description, and no label is the field's bare name;
* no two fields share a label: the owner's questions and the render's "fill in …" name a field by its
  label alone, and two fields with one label would collapse into one question;
* the fields with a default are exactly the ones listed here, each with one line on why its words are
  right for any business, so a new default has to be argued for in this file first;
* no field marked as a claim, and no field a voice line speaks, has words as its default (an empty
  default, for an optional part of a line, is not words);
* the count of required fields: App promo 68 before US-D302 and 67 after (the big moment's button),
  Cinematic product promo 69 and UI story promo 66, unchanged; on 19 Nov 2026 one fewer each (66, 68,
  65): the claim a line shows without, empty until a sourced figure fills it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from core.social_templates import is_bundle_variable, placeholders, resolve_variables, voice_lines  # noqa: E402
from modules.documents import social_starters  # noqa: E402

APP, CINEMATIC, UI_STORY = "app-promo", "cinematic-product-promo", "ui-story-promo"
SLUGS = (APP, CINEMATIC, UI_STORY)
FIELDS = {APP: 87, CINEMATIC: 122, UI_STORY: 120}
# 19 Nov 2026: one fewer each, the claim a line shows without (tracks_detail, outputs_count).
REQUIRED = {APP: 66, CINEMATIC: 68, UI_STORY: 65}

# Each default, and why its words are right for any business.
APP_DEFAULTS = {
    "hook_accent": "empty: the headline reads whole without accent words",
    "slogan_note": "empty: the reveal reads whole without a line under the slogan",
    "phone_time": "the clock in the phone's top bar: part of the phone, not a claim",
    "phone_time_later": "the same clock a minute on, on the second phone screen",
    "today_label": "a generic label for a progress card on any app's first screen",
    "question_label": "a generic instruction above a set of answers",
    "result_label": "a neutral word for a right answer picked",
    "tutor_button": "a generic button word",
    "tutor_header": "a generic header for a help screen",
    "tutor_title": "a generic title for a help screen",
    "tutor_scope": "empty: no note under the answer unless the brief gives one",
    "ask_placeholder": "the generic hint in an empty message box",
    "ask_speak": "a generic button word",
    "ask_send": "a generic button word",
    "dock_1": "a menu tab every app has",
    "dock_2": "a menu tab every app has",
    "dock_3": "a menu tab every app has",
    "dock_4": "a menu tab every app has",
    "mock_button": "US-D302: a generic button word that begins a step, pressed on screen",
    "end_note": "empty: small print only when the business's copy rules call for it",
    "tracks_detail": "empty: a claim, shown only with its sourced figure; the list reads whole without it",
}
# The chat, the board, the question card, the morning, the results and the team: the screens the
# Cinematic product promo and the UI story promo share, field for field.
SCREEN_DEFAULTS = {
    "hook_accent": "empty: the headline reads whole without accent words",
    "tagline_accent": "empty: the tagline reads whole without an accent word",
    "tagline_tail": "empty: the tagline can end with its first part",
    "chat_crumb": "a generic page name on the chat screen",
    "chat_counter_label": "a neutral status word in the chat screen's pill",
    "chat_counter": "the chat screen's count of open conversations, part of the screen, not the business",
    "chat_thinking": "a neutral status word while the reply loads",
    "chat_reply": "a generic acknowledgement before the steps; it names nothing of the business",
    "step_1_label": "a generic step word in the reply",
    "step_1_time": "a step's timing on the chat screen, part of the screen, not a claim",
    "step_2_label": "a generic step word in the reply",
    "step_2_time": "a step's timing on the chat screen, part of the screen, not a claim",
    "step_3_label": "a generic step word in the reply",
    "step_3_time": "a step's timing on the chat screen, part of the screen, not a claim",
    "chat_placeholder": "the generic hint in an empty message box",
    "board_title": "a generic title for a task board",
    "board_working_label": "a neutral status word over a counter whose digits the page sets itself",
    "board_agents_label": "a neutral word over the board's second counter",
    "board_agents": "matches the board's four task cards, each taken by one name",
    "board_queue_label": "a neutral status word over a counter whose digits the page sets itself",
    "board_attention_label": "a neutral status word over a counter whose digits the page sets itself",
    "board_column_1": "a column every task board has",
    "board_column_2": "a column every task board has",
    "board_column_3": "a column every task board has",
    "task_1_kind": "a generic card type",
    "task_1_flag": "empty: no priority mark unless the brief gives one",
    "task_2_kind": "a generic card type",
    "task_2_flag": "empty: no priority mark unless the brief gives one",
    "task_3_kind": "a generic card type",
    "task_3_flag": "empty: no priority mark unless the brief gives one",
    "task_4_kind": "a generic card type",
    "task_4_flag": "empty: no priority mark unless the brief gives one",
    "task_unassigned": "the generic label on a card nobody has taken yet",
    "ask_badge": "a generic badge word",
    "ask_own_words": "a generic link under two answer buttons",
    "ask_toast": "a neutral confirmation once an answer is picked",
    "night_hour": "a clock time: a late evening, which any overnight story fits",
    "night_minute": "a clock time: the late evening's minutes",
    "morning_hour": "a clock time: the next morning",
    "morning_minute": "a clock time: on the hour",
    "outputs_eyebrow": "a generic heading for a screen of finished work",
    "outputs_title": "a generic title for a screen of finished work",
    "outputs_pill_1": "a generic filter tab",
    "outputs_pill_2": "a generic filter tab",
    "outputs_pill_3": "a generic filter tab",
    "outputs_pill_4": "a generic filter tab",
    "outputs_count": "empty: a claim, shown only with its sourced count; the line reads whole without it",
    "team_line_2": "empty: the team's statement can be one line",
    "team_line_2_accent": "empty: the line reads whole without accent words",
    "team_line_3": "empty: the team's statement can be one line",
    "team_line_3_accent": "empty: the line reads whole without accent words",
    "team_idle": "a neutral status word on a name card",
    "team_busy": "a neutral status word on a name card",
    "end_note": "empty: no small print unless the brief gives some",
}
DEFAULTS = {
    APP: APP_DEFAULTS,
    CINEMATIC: SCREEN_DEFAULTS,
    UI_STORY: {**SCREEN_DEFAULTS, "problem_accent": "empty: the problem line reads whole without accent words"},
}


@pytest.fixture(scope="module")
def starters():
    found = {s["slug"]: s for s in social_starters.social_starters() if s["slug"] in SLUGS}
    social_starters._starters.cache_clear()
    return found


def _words(text):
    return " ".join(text.split()).lower() if isinstance(text, str) else ""


@pytest.mark.parametrize("slug", SLUGS)
def test_every_field_has_a_label_and_a_description(starters, slug):
    schema = starters[slug]["blocks"]["variables_schema"]
    assert len(schema) == FIELDS[slug]
    unexplained = [name for name, spec in schema.items() if not _words(spec.get("label")) or not _words(spec.get("description"))]
    assert unexplained == []


@pytest.mark.parametrize("slug", SLUGS)
def test_no_label_is_a_fields_bare_name_and_no_two_fields_share_one(starters, slug):
    schema = starters[slug]["blocks"]["variables_schema"]
    bare = [name for name, spec in schema.items() if _words(spec["label"]) in {name.lower(), _words(name.replace("_", " "))}]
    assert bare == []
    labels = [spec["label"] for spec in schema.values()]
    assert sorted({label for label in labels if labels.count(label) > 1}) == []


@pytest.mark.parametrize("slug", SLUGS)
def test_every_default_is_on_the_list_with_why_it_suits_any_business(starters, slug):
    schema = starters[slug]["blocks"]["variables_schema"]
    defaulted = {name for name, spec in schema.items() if "default" in spec}
    listed = set(DEFAULTS[slug])
    assert sorted(defaulted - listed) == [], "a default nobody argued for: add it to the list, with why"
    assert sorted(listed - defaulted) == []
    assert all(_words(why) for why in DEFAULTS[slug].values())
    assert len(resolve_variables(schema, {}).missing) == REQUIRED[slug] == FIELDS[slug] - len(listed)


@pytest.mark.parametrize("slug", SLUGS)
def test_no_claim_and_no_spoken_field_has_words_as_its_default(starters, slug):
    blocks = starters[slug]["blocks"]
    schema = blocks["variables_schema"]
    spoken = {
        name for line in voice_lines(blocks["audio_plan"])
        for name in placeholders(str(line.get("text") or "")) if not is_bundle_variable(name)
    }
    worded = {name: spec["default"] for name, spec in schema.items() if _words(str(spec.get("default", "")))}
    assert {name: text for name, text in worded.items() if schema[name].get("claim") is True} == {}
    assert {name: text for name, text in worded.items() if name in spoken} == {}
