"""F378 (night 11, 7 Oct): the composer fills every required field or asks the owner.

B2: all four video drafts from compose failed render ("fill in … before rendering", 11 to
44 fields), and the proposal said nothing about it beyond a warning. Pinned:

* the follow-ups go round again for what the last round left empty or got refused, a
  refused value asked for with the reason ("is longer than 60 characters (61 given)");
* at most ``FILL_ROUNDS`` rounds, and none once the proposal's time budget is spent;
* a field the model answers null for (the brief does not give it) is not asked again;
* every required field still empty comes back in ``questions`` by its label, the plain
  words the editor shows as "Auto needs: …"; a required field answered blank counts as empty.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from modules.socials import compose  # noqa: E402

SCHEMA = {
    "hook": {"type": "text", "label": "Hook: the headline", "max_chars": 60},
    "beat_1": {"type": "text", "label": "Caption and voice: beat 1"},
    "end_url": {"type": "text", "label": "End card: the address"},
    "note": {"type": "text", "label": "Small print", "default": ""},
}
STORY = {"id": "story", "name": "UI story promo", "format": "social_video", "sizes": ["1080x1920"], "variables_schema": SCHEMA}


def _ctx():
    return compose.ComposeContext(
        brief="A reel for Tide Café: open from 7 every morning.", format="video",
        channels=[{"toolkit": "instagram", "label": "Instagram"}], templates=[STORY], candidates=[],
    )


def _answer(**variables):
    return {"title": "Open from 7", "copy": {"base": "Open from 7.", "channels": {"instagram": "Open from 7."}},
            "format": "video", "template_id": "story", "variables": variables, "sources": {}}


class _Model:
    def __init__(self, answers):
        self.answers, self.asked = list(answers), []

    async def generate_response(self, messages, tools=None):
        self.asked.append(messages)
        answer = self.answers.pop(0)
        return SimpleNamespace(content=answer if isinstance(answer, str) else json.dumps(answer))


def _propose(model, timeout=5.0):
    return asyncio.run(compose.propose(_ctx(), lambda: model, timeout))


def _asked_for(messages):
    note, schema = messages[-1]["content"].split("\n", 1)
    assert note == compose.FILL_NOTE
    return json.loads(schema)


def _values(proposal):
    return {name: spec["value"] for name, spec in proposal["variables"].items()}


def test_a_refused_fill_is_asked_for_again_with_the_reason():
    model = _Model([
        _answer(beat_1="We open at 7.", end_url="tidecafe.co.uk"),
        {"variables": {"hook": "x" * 61}},
        {"variables": {"hook": "Coffee from 7, every morning."}},
    ])
    proposal = _propose(model)
    assert len(model.asked) == 3
    assert _asked_for(model.asked[1]) == {"hook": SCHEMA["hook"]}
    assert _asked_for(model.asked[2]) == {"hook": {**SCHEMA["hook"], "refused": "is longer than 60 characters (61 given)"}}
    assert _values(proposal)["hook"] == "Coffee from 7, every morning."
    assert proposal["questions"] == []


def test_the_rounds_are_bounded_and_what_is_still_empty_is_asked_of_the_owner():
    model = _Model([_answer(beat_1="We open at 7.", end_url="tidecafe.co.uk"),
                    *({"variables": {"hook": "x" * 61}} for _ in range(compose.FILL_ROUNDS + 2))])
    proposal = _propose(model)
    assert len(model.asked) == 1 + compose.FILL_ROUNDS
    assert "hook" not in proposal["variables"]
    assert proposal["questions"] == ["Hook: the headline"]


def test_a_field_answered_null_is_not_asked_again_and_is_the_owners_question():
    model = _Model([_answer(hook="Coffee from 7."), {"variables": {"beat_1": "We open at 7.", "end_url": None}}])
    proposal = _propose(model)
    assert len(model.asked) == 2
    assert _values(proposal) == {"hook": "Coffee from 7.", "beat_1": "We open at 7."}
    assert proposal["questions"] == ["End card: the address"]


def test_a_required_field_answered_blank_is_empty():
    model = _Model([_answer(hook="Coffee from 7.", beat_1="We open at 7.", end_url="  "), {"variables": {"end_url": ""}}])
    proposal = _propose(model)
    assert "end_url" not in proposal["variables"]
    assert _asked_for(model.asked[1]) == {"end_url": SCHEMA["end_url"]}
    assert proposal["questions"] == ["End card: the address"]


def test_no_follow_up_once_the_time_budget_is_spent(monkeypatch):
    clock = iter([0.0] + [1e9] * 10)  # the start, then long past the budget
    monkeypatch.setattr(compose, "_now", lambda: next(clock))
    model = _Model([_answer(hook="Coffee from 7.")])
    proposal = _propose(model)
    assert len(model.asked) == 1
    assert proposal["questions"] == ["Caption and voice: beat 1", "End card: the address"]
