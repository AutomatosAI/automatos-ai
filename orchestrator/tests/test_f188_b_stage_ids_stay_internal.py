"""F188 (night 6) — the onboarding stage ids are Auto's, never the owner's.

From 02:02:53 every turn's prompt carried the raw stage id, `boom`. Nothing
said it was internal, so the owner heard "You've successfully completed the
boom stage… next is powerup". The boom block also said "the setup checklist
card carries the remaining steps", but that card renders only at powerup, so
Auto invented a "Command Center… tab" for it. Now the prompt says the ids are
internal, and at boom it gives Auto the sentence for "is setup done?".
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from modules.context.sections.base import SectionContext
from modules.context.sections.onboarding import OnboardingSection

INTERNAL = "The ids are internal: never say one to the owner; say what is built and what is left, in plain words."
DONE_YET = ('If they ask whether setup is done: "Your team is built — the last step is seeing it answer or do one '
            'real thing for you." They see no checklist yet.')


def _render(stage):
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = type(
        "Ws", (), {"onboarding": {"stage": stage, "stages": {}, "segment": {}}})()
    ctx = SectionContext(agent=None, workspace_id="ws-1", db_session=db, messages=[])
    return " ".join(asyncio.run(OnboardingSection().render(ctx)).split())


@pytest.mark.parametrize("stage", ["questions", "teach", "proposal", "building", "boom", "powerup"])
def test_every_stage_says_the_ids_are_internal(stage):
    assert INTERNAL in _render(stage)


def test_boom_says_what_to_answer_and_promises_no_checklist_card():
    rendered = _render("boom")
    assert DONE_YET in rendered
    assert "checklist card carries the remaining steps" not in rendered
