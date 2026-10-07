"""F377 (night 11, 7 Oct) — the videos told on a software product's screens say so.

Night 11: a coffee roaster's brief ("no software, no agents, no dashboards") still came back
as the UI story's task board, because nothing about a template said whom it was for.

Pinned:

* **The contract.** ``made_for`` is a block of the template contract: ``"software"``, or left
  out for a template any business can use; anything else is refused on save.
* **The seeds.** UI story promo, Cinematic product promo and App promo are made for software,
  and their descriptions say plainly what they show and that they suit a software brief only;
  the Data story and every image template are for any business.
* **Where templates are listed.** The editor's gallery entry and the composer's template entry
  carry ``made_for`` (the composer's with the description too); the agents' list ends a software
  template's row with whom it is for, and its schema answer carries ``made_for``.
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

from api import socials_compose  # noqa: E402
from core.social_templates import MADE_FOR_SOFTWARE, SocialTemplateError, made_for, validate_social_blocks  # noqa: E402
from modules.documents.social_starters import social_starters  # noqa: E402
from modules.socials import template_gallery  # noqa: E402
from modules.tools.discovery.template_tools import template_row  # noqa: E402

SOFTWARE = {"ui-story-promo", "cinematic-product-promo", "app-promo"}
PAGE = (
    '<!doctype html><html><head></head><body><div id="root" data-composition-id="main" data-duration="3">'
    '<p>{{ headline }}</p></div></body></html>'
)
BLOCKS = {"html": PAGE, "variables_schema": {"headline": {"type": "text"}}, "sizes": ["1080x1920"]}


class _Rows:
    """The query chain the composer's template listing runs, answering ``rows``."""

    def __init__(self, rows):
        self.rows = rows

    def query(self, *_columns):
        return self

    def filter(self, *_criteria):
        return self

    def order_by(self, *_columns):
        return self

    def all(self):
        return self.rows


def _row(starter):
    return SimpleNamespace(
        id=uuid.uuid4(), name=starter["name"], description=starter["description"], format=starter["format"],
        category=starter["category"], blocks=starter["blocks"], thumbnail_url=None, created_by="system",
        updated_at=None, sample_data=starter["sample_data"],
    )


def test_made_for_is_software_or_left_out():
    assert validate_social_blocks({**BLOCKS, "made_for": MADE_FOR_SOFTWARE}, "social_video")["made_for"] == "software"
    assert "made_for" not in validate_social_blocks(BLOCKS, "social_video")
    with pytest.raises(SocialTemplateError, match="made_for: must be one of"):
        validate_social_blocks({**BLOCKS, "made_for": "bakery"}, "social_video")
    assert made_for({"made_for": "bakery"}) is None and made_for(None) is None


def test_the_three_software_videos_say_whom_they_are_for_and_every_other_template_suits_any_business():
    for starter in social_starters():
        if starter["slug"] in SOFTWARE:
            assert made_for(starter["blocks"]) == MADE_FOR_SOFTWARE, starter["slug"]
            description = starter["description"]
            assert description.startswith(("For a software product only:", "For a phone app only:")), starter["slug"]
            assert "Pick it only for a brief about" in description, starter["slug"]
        else:
            assert "made_for" not in starter["blocks"], starter["slug"]


def test_the_gallery_the_composer_and_the_agents_see_it():
    starters = {s["slug"]: s for s in social_starters("social_video")}
    ui_story, data_story = _row(starters["ui-story-promo"]), _row(starters["data-story"])
    gallery = [template_gallery.entry(row, store=None) for row in (ui_story, data_story)]
    assert [entry["made_for"] for entry in gallery] == ["software", None]
    entries = socials_compose.social_templates(_Rows([ui_story, data_story]), uuid.uuid4(), "video")
    assert [(e["made_for"], e["description"]) for e in entries] == [
        ("software", ui_story.description), (None, data_story.description),
    ]
    assert entries[0]["fields_by_length"]["15"] and "headline" in entries[1]["fields_by_length"]["15"]
    assert template_row(ui_story).endswith("| made for software only: pick it only when the brief is about software")
    assert "made for" not in template_row(data_story)
