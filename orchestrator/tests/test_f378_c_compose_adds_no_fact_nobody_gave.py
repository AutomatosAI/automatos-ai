"""F378 (night 11, 7 Oct): the composer adds no fact nobody gave.

"It invents things… Tasting notes I never gave, 'Worldwide shipping', 'Wholesale Growth', a
source report that doesn't exist… That's my name on a public post." Pinned:

* the prompt says plainly to use only the brief's facts, never to add product details,
  tasting notes, prices, offers, shipping, handles, growth claims, names, dates or sources,
  and to ask the owner for a missing fact instead; the fill follow-up says the same and
  lets the model answer null;
* a figure in the copy or the fields that the brief, the current take and the bound
  sources do not hold is a warning naming it; small counters and template defaults are not;
* the handle field takes the brand kit's handle only, and the kit's handles reach the
  prompt; an @mention nobody gave is a warning;
* the model's own questions for the owner are kept.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from modules.socials import compose, compose_checks, compose_given  # noqa: E402

BRIEF = "Harbour Blend is back: 412 bags roasted this week, £12 a bag."
SCHEMA = {
    "headline": {"type": "text", "label": "Headline", "max_chars": 80},
    "stat": {"type": "text", "label": "The figure", "claim": True, "default": ""},
    "counter": {"type": "text", "label": "Page counter", "default": "01 / 05"},
    "handle": {"type": "text", "label": "Handle", "default": ""},
}
CARD = {"id": "card", "name": "Announcement card", "format": "social_image", "sizes": ["1080x1350"], "variables_schema": SCHEMA}
REPORT = {"kind": "report", "ref": "r-1", "title": "Roastery week 40", "detail": "Wholesale up 18% on September"}


def _ctx(**overrides):
    base = dict(brief=BRIEF, format="image", channels=[{"toolkit": "instagram", "label": "Instagram"}],
                templates=[CARD], candidates=[REPORT], handles={"instagram": "@harbourline"})
    return compose.ComposeContext(**{**base, **overrides})


def _raw(copy="Harbour Blend is back.", **variables):
    return {"title": "Harbour Blend", "format": "image", "template_id": "card",
            "copy": {"base": copy, "channels": {"instagram": copy}}, "variables": {"headline": "Harbour Blend is back", **variables}}


def _numbers_warning(proposal):
    return next((w for w in proposal["warnings"] if w.startswith("Numbers nobody gave")), None)


def test_the_prompt_says_to_use_only_the_briefs_facts_and_ask_for_the_rest():
    system, material = compose.build_messages(_ctx())
    assert compose.FACTS_NOTE in system["content"] and compose.HANDLE_NOTE in system["content"]
    for word in ("tasting note", "price", "shipping", "handle", "growth claim", "name", "date", "source", "questions"):
        assert word in compose.FACTS_NOTE
    assert json.loads(material["content"])["brand_handles"] == {"instagram": "@harbourline"}
    assert "questions" in compose._ANSWER_SHAPE
    assert "answer null" in compose.FILL_NOTE and "use only the facts the brief gives" in compose.FILL_NOTE


def test_a_figure_nobody_gave_is_a_warning_naming_it():
    copy = "412 bags roasted, £12 a bag, worldwide shipping from £5.99 and 46% growth."
    proposal = compose_checks.checked_proposal(_raw(copy), _ctx())
    assert _numbers_warning(proposal) == (
        "Numbers nobody gave: 5.99, 46. Check each one, or take it out before approval"
    )


def test_the_briefs_figures_counters_and_defaults_pass():
    proposal = compose_checks.checked_proposal(_raw("412 bags, 1 of 3.", stat="1,200", counter="01 / 05"),
                                               _ctx(brief=BRIEF + " 1200 members."))
    assert _numbers_warning(proposal) is None


def test_a_figure_from_the_bound_source_passes_and_from_an_unbound_one_does_not():
    raw = _raw(stat="18%")
    unbound = compose_checks.checked_proposal(raw, _ctx())
    assert "18" in _numbers_warning(unbound)
    bound = compose_checks.checked_proposal({**raw, "sources": {"stat": {"kind": "report", "ref": "r-1"}}}, _ctx())
    assert bound["sources"]["stat"]["ref"] == "r-1" and _numbers_warning(bound) is None


def test_the_handle_is_the_brand_kits():
    invented = compose_checks.checked_proposal(_raw(handle="@harbourlinecoffee"), _ctx())
    assert invented["variables"]["handle"] == {"value": "@harbourline", "claim": False}
    assert "The handle @harbourlinecoffee is not one of your brand kit's; @harbourline is used" in invented["warnings"]
    kept = compose_checks.checked_proposal(_raw(handle="harbourline"), _ctx())
    assert kept["variables"]["handle"]["value"] == "harbourline"
    no_kit = compose_checks.checked_proposal(_raw(handle="@harbourlinecoffee"), _ctx(handles={}))
    assert "handle" not in no_kit["variables"]
    assert any(w.startswith("The handle @harbourlinecoffee is not in your brand kit") for w in no_kit["warnings"])


def test_a_mention_nobody_gave_is_a_warning():
    proposal = compose_checks.checked_proposal(_raw("Thanks @tidecafe and @harbourline. Mail hello@harbour.co"), _ctx())
    assert "The copy mentions @tidecafe, which neither the brief nor your brand kit gives: check it" in proposal["warnings"]
    assert compose_given.unknown_mentions(proposal, _ctx(brief=BRIEF + " With @tidecafe.")) == []


def test_the_models_questions_for_the_owner_are_kept():
    raw = {**_raw(), "questions": ["What does a bag cost for wholesale?", 7, "  What does a bag cost for wholesale? "]}
    assert compose_checks.checked_proposal(raw, _ctx())["questions"] == ["What does a bag cost for wholesale?"]
