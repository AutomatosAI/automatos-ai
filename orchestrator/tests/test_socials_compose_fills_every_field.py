"""F253 follow-up (3 Oct 2026): a post Auto writes is one the render can make.

Gerard's video posts left to "Let Auto pick" failed twice: the composer chose a video
starter (dozens of required fields), its one answer filled some of them, and the render
refused "fill in beat_1, beat_2, ... before rendering". The composer now follows up:

* an answer that fills every required variable is the only call;
* the required variables an answer left empty, or gave a value that does not hold, are
  asked for by name with their schema, after the answer, and merged in; a value the
  answer gave that holds is never replaced;
* many are asked for ``FILL_BATCH`` at a time, so each answer fits the output budget;
* a follow-up that is unusable or late leaves the proposal as it was, its warning saying
  which variables have no value yet, instead of failing the whole draft.
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
    "beat_2": {"type": "text", "label": "Caption and voice: beat 2"},
    "end_url": {"type": "text", "label": "End card: the address"},
    "note": {"type": "text", "label": "Small print", "default": ""},
}
STORY = {"id": "story", "name": "UI story promo", "format": "social_video", "sizes": ["1080x1920"], "variables_schema": SCHEMA}


def _ctx(template=STORY):
    return compose.ComposeContext(
        brief="40-second reel for Automatos: Auto runs the night.", format="video",
        channels=[{"toolkit": "twitter", "label": "X"}], templates=[template], candidates=[],
    )


def _answer(**variables):
    return {"title": "Auto runs the night", "copy": {"base": "Meet Auto.", "per_channel": {"twitter": "Meet Auto."}},
            "format": "video", "template_id": "story", "variables": variables, "sources": {}}


class _Model:
    """The workspace's model: answers from a queue (each with its own delay), records what it was asked."""

    def __init__(self, answers, delays=()):
        self.answers, self.delays, self.asked = list(answers), list(delays), []

    async def generate_response(self, messages, tools=None):
        self.asked.append(messages)
        delay = self.delays[len(self.asked) - 1] if len(self.asked) <= len(self.delays) else 0
        if delay:
            await asyncio.sleep(delay)
        answer = self.answers.pop(0)
        return SimpleNamespace(content=answer if isinstance(answer, str) else json.dumps(answer))


def _propose(model, ctx=None, timeout=5.0):
    return asyncio.run(compose.propose(ctx or _ctx(), lambda: model, timeout))


def _values(proposal):
    return {name: spec["value"] for name, spec in proposal["variables"].items()}


def _asked_for(messages):
    note, schema = messages[-1]["content"].split("\n", 1)
    assert note == compose.FILL_NOTE and messages[-2]["role"] == "assistant"
    return json.loads(schema)


def test_an_answer_that_fills_every_required_field_is_the_only_call():
    model = _Model([_answer(hook="Your business never sleeps.", beat_1="Work comes in.", beat_2="Auto takes it.", end_url="automatos.app")])
    proposal = _propose(model)
    assert len(model.asked) == 1
    assert not any(w.startswith("No value yet") for w in proposal["warnings"])


def test_the_fields_left_empty_are_asked_for_by_name_and_merged_without_replacing_the_rest():
    first = _answer(hook="Your business never sleeps.", beat_1="Work comes in.")
    model = _Model([first, {"variables": {"beat_2": "Auto takes it.", "end_url": "automatos.app", "hook": "Replaced?"}}])
    proposal = _propose(model)

    assert len(model.asked) == 2
    assert _asked_for(model.asked[1]) == {"beat_2": SCHEMA["beat_2"], "end_url": SCHEMA["end_url"]}
    assert json.loads(model.asked[1][-2]["content"]) == first  # the answer it follows up
    assert _values(proposal) == {"hook": "Your business never sleeps.", "beat_1": "Work comes in.",
                                 "beat_2": "Auto takes it.", "end_url": "automatos.app"}
    assert not any(w.startswith("No value yet") for w in proposal["warnings"])


def test_a_value_that_does_not_hold_is_asked_for_again():
    model = _Model([_answer(hook="x" * 61, beat_1="a", beat_2="b", end_url="automatos.app"),
                    {"variables": {"hook": "Your business never sleeps."}}])
    proposal = _propose(model)
    assert _asked_for(model.asked[1]) == {"hook": SCHEMA["hook"]}
    assert _values(proposal)["hook"] == "Your business never sleeps."


def test_many_empty_fields_are_asked_for_a_batch_at_a_time():
    schema = {f"field_{n:02d}": {"type": "text", "label": f"Field {n}"} for n in range(60)}
    template = {**STORY, "variables_schema": schema}
    batches = [list(schema)[i:i + compose.FILL_BATCH] for i in range(0, 60, compose.FILL_BATCH)]
    model = _Model([_answer(), *({"variables": {name: f"value of {name}" for name in batch}} for batch in batches)])
    proposal = _propose(model, _ctx(template))

    assert [len(_asked_for(messages)) for messages in model.asked[1:]] == [25, 25, 10]
    assert _values(proposal) == {name: f"value of {name}" for name in schema}


def test_a_follow_up_that_is_unusable_or_late_leaves_the_proposal_and_says_what_is_empty():
    first = _answer(hook="Your business never sleeps.", beat_1="Work comes in.")
    for model, timeout in ((_Model([first, "no idea"]), 5.0), (_Model([first, {"variables": {}}], delays=(0, 1.0)), 0.2)):
        proposal = _propose(model, timeout=timeout)
        assert _values(proposal) == {"hook": "Your business never sleeps.", "beat_1": "Work comes in."}
        assert "No value yet for: beat_2, end_url" in proposal["warnings"]
