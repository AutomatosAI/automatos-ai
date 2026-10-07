"""F377 (night 11, 7 Oct) — no seeded video's default copy is Automatos's.

Night 11: the Cinematic promo printed "Hire the agents. Equip them with skills." and a
"Command Centre / Deliverables" screen no field controlled; the UI story's board read
"WORKING · AGENTS 9 · QUEUE · ATTENTION" and "NEEDS YOU"; the App promo's dock had a
"Podcast" tab and its fields were an exam app's.

Pinned (UI story promo, Cinematic product promo, App promo, the Data story):

* no default names Automatos or its product: agents, the Command Centre, Deliverables,
  skills, platform tools, "NEEDS YOU", nor an exam, a quiz, a tutor or a podcast;
* no label or description talks of agents or exam material either, so the composer and the
  owner are asked for a product's own words;
* the team's statement is the post's own (required); its other lines are optional;
* the page itself prints no word of its own: every word on screen is a field.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from modules.documents import social_starters  # noqa: E402

VIDEOS = ("ui-story-promo", "cinematic-product-promo", "app-promo", "data-story")
NOT_A_DEFAULT = re.compile(
    r"\b(?:automatos|auto|agents?|command cent(?:re|er)|deliverables?|skills?|platform|needs you|"
    r"exams?|quiz(?:zes)?|tutor|podcast|mock|course)\b",
    re.IGNORECASE,
)
NOT_A_LABEL = re.compile(r"\b(?:agents?|exams?|quiz|tutor|podcast|mock|tracks?|course|grade|gate)\b", re.IGNORECASE)
# What night 11 saw on screen, by field: never a default again.
SEEN = {
    "board_title": "Command Centre", "outputs_title": "Deliverables", "ask_badge": "NEEDS YOU",
    "team_line_1": "Hire the agents.", "team_line_2_accent": "skills.", "step_3_label": "platform create task",
    "board_agents_label": "AGENTS", "chat_crumb": "OPERATIONS · Conversations ›", "dock_3": "Podcast",
}


def _starter(slug):
    starter = next(s for s in social_starters.social_starters() if s["slug"] == slug)
    social_starters._starters.cache_clear()
    return starter


@pytest.mark.parametrize("slug", VIDEOS)
def test_no_default_label_or_description_is_automatos_copy(slug):
    schema = _starter(slug)["blocks"]["variables_schema"]
    defaults = {name: spec["default"] for name, spec in schema.items() if isinstance(spec.get("default"), str)}
    assert {name: text for name, text in defaults.items() if NOT_A_DEFAULT.search(text)} == {}
    assert {name: text for name, text in defaults.items() if SEEN.get(name) == text} == {}
    worded = {name: f"{spec.get('label', '')} {spec.get('description', '')}" for name, spec in schema.items()}
    assert {name: text for name, text in worded.items() if NOT_A_LABEL.search(text)} == {}


@pytest.mark.parametrize("slug", ["ui-story-promo", "cinematic-product-promo"])
def test_the_teams_statement_is_the_posts_own_and_its_other_lines_are_optional(slug):
    schema = _starter(slug)["blocks"]["variables_schema"]
    assert "default" not in schema["team_line_1"]
    assert all(schema[name]["default"] == "" for name in ("team_line_2", "team_line_2_accent", "team_line_3", "team_line_3_accent"))


@pytest.mark.parametrize("slug", VIDEOS)
def test_the_page_prints_no_word_of_its_own(slug):
    html = _starter(slug)["blocks"]["html"]
    body = re.sub(r"<(script|style|title)\b.*?</\1>", " ", html, flags=re.S | re.I)
    text = re.sub(r"\{\{[^{}]*\}\}", " ", re.sub(r"<!--.*?-->|<[^<>]*>", " ", body, flags=re.S))
    assert re.findall(r"[A-Za-z]{2,}", text) == []
