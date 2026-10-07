"""F377 (night 11, 7 Oct) — a figure written as text fills a number field.

The Data story's bars are drawn from number fields. The composer (a model) writes figures as
text as often as not ("412", "1,240"), and a value that does not fit its field is dropped:
the field is then empty, and the render refuses "fill in … before rendering". A retake could
not get past it.

Pinned (``core.social_templates.resolve_variables``): a figure written as text (digits,
thousands grouped by commas, a decimal part) fills a number field as that number; anything
else ("lots", "1,24", "96 kg") is still refused; and a 15 s Data story whose values came as
text renders with them as numbers.
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

from core.social_templates import SOCIAL_VIDEO, resolve_variables  # noqa: E402
from modules.documents import social_starters  # noqa: E402
from modules.socials import render  # noqa: E402

SCHEMA = {"count": {"type": "number"}, "label": {"type": "text"}}


@pytest.mark.parametrize("written, number", [
    ("412", 412), ("1,240", 1240), (" 96.5 ", 96.5), ("-3", -3), ("12.0", 12.0), ("2,000,000", 2000000),
])
def test_a_figure_written_as_text_is_that_number(written, number):
    resolved = resolve_variables(SCHEMA, {"count": written, "label": written})
    assert resolved.values["count"] == number and type(resolved.values["count"]) is type(number)
    assert resolved.values["label"] == written  # a text field keeps the text as written


@pytest.mark.parametrize("written", ["lots", "1,24", "96 kg", "", "1.2.3"])
def test_anything_else_is_still_refused(written):
    resolved = resolve_variables(SCHEMA, {"count": written, "label": "x"})
    assert "count" not in resolved.values and resolved.invalid == ["count must be a number"]


def test_a_15_second_data_story_with_its_values_as_text_renders():
    starter = next(s for s in social_starters.social_starters() if s["slug"] == "data-story")
    social_starters._starters.cache_clear()
    values = {"headline": "Our busiest month yet.", "ranking_title": "Most ordered this month",
              "item_1_name": "Sourdough", "item_1_value": "412", "item_2_name": "Cinnamon buns", "item_2_value": "287",
              "item_3_name": "Rye loaf", "item_3_value": "1,064"}
    post = SimpleNamespace(id=uuid.uuid4(), workspace_id=uuid.uuid4(), length_seconds=15, footage=None, music=None,
                           voice=None, targets=[], variables={name: {"value": value} for name, value in values.items()})
    template = SimpleNamespace(id=uuid.uuid4(), format=SOCIAL_VIDEO, blocks=starter["blocks"])
    bundle = render.bundle_for(post, template, {})
    assert (bundle["variables"]["item_1_value"], bundle["variables"]["item_3_value"]) == (412, 1064)
