"""F378 (night 11, 7 Oct): the composer's copy is the shape a save takes.

The owner's compose → save flow broke on the first step: compose answered
``copy.per_channel`` and ``POST /posts`` refused it ("copy keys must be 'base' and
'channels', got ['per_channel']"). The retake, the plan maker and the frontend each
translated it by hand. Now the composer answers ``{"base", "channels"}`` and every
caller passes it through as it is:

* a checked proposal's copy is ``{"base", "channels"}``, each selected channel with
  its own text, and the post's copy validator takes it unchanged;
* the retake's and the plan maker's edits carry the composer's copy as it is.
"""
from __future__ import annotations

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

from api import socials_retake  # noqa: E402
from modules.socials import compose, compose_checks, service  # noqa: E402
from services import socials_plan_maker  # noqa: E402

CHANNELS = [{"toolkit": "twitter", "label": "X"}, {"toolkit": "linkedin", "label": "LinkedIn"}]


def _ctx():
    return compose.ComposeContext(
        brief="Harvest Club opens Friday.", format="text", channels=CHANNELS, templates=[], candidates=[],
    )


def _raw():
    return {"title": "Harvest Club", "format": "text",
            "copy": {"base": "Harvest Club opens Friday.", "channels": {"twitter": "Friday: Harvest Club."}}}


def test_the_answer_shape_the_model_is_taught_is_base_and_channels():
    assert set(compose._ANSWER_SHAPE["copy"]) == {"base", "channels"}


def test_a_checked_proposals_copy_is_base_and_channels_and_a_save_takes_it_unchanged():
    copy = compose_checks.checked_proposal(_raw(), _ctx())["copy"]
    assert copy == {"base": "Harvest Club opens Friday.",
                    "channels": {"twitter": "Friday: Harvest Club.", "linkedin": "Harvest Club opens Friday."}}
    assert service._validate_copy(copy) == copy


def test_the_retake_and_the_plan_maker_pass_the_composers_copy_through():
    proposal = compose_checks.checked_proposal(_raw(), _ctx())
    assert socials_retake.take_changes(proposal)["copy"] == proposal["copy"]
    slot = SimpleNamespace(template_id="kept")
    assert socials_plan_maker._changes(proposal, slot)["copy"] == proposal["copy"]
