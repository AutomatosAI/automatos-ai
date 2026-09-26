"""F165 (night 5, persona B28) — mission approve refuses what it cannot apply.

The mission page's Approve button sends `modifications` (plan edits) next to
max_concurrent_override; the server's approve body declared only
max_concurrent_override, token_budget_override and skip_verification, and
ignored unknown keys, so edits were dropped without a word. Plan edits go to
PATCH /api/missions/{id}/plan before approval (PRD-163 S4/Q57). The approve body
now refuses anything else, and names the route for edits.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.missions import MissionApproveRequest

EDITS = {"task_overrides": {"t1": {"title": "Draft the visit notes"}}, "agent_overrides": {"t1": "WRITER"},
         "notes": "WRITER drafts, COUNTINGHOUSE checks every number"}


def test_plan_edits_on_approve_are_refused_with_the_route_for_them():
    with pytest.raises(ValidationError) as refused:
        MissionApproveRequest.model_validate({"modifications": EDITS, "max_concurrent_override": 2})
    assert "PATCH /api/missions/{mission_id}/plan first, then approve" in str(refused.value)


def test_any_other_unknown_key_is_refused():
    with pytest.raises(ValidationError):
        MissionApproveRequest.model_validate({"token_budget": 50_000})


@pytest.mark.parametrize("body", [
    {},                                                           # the inbox and the chat widget
    {"max_concurrent_override": 2},                               # the mission page's Parallel setting
    {"max_concurrent_override": 3, "token_budget_override": 200_000, "skip_verification": False},
])
def test_what_the_ui_sends_today_is_accepted(body):
    assert MissionApproveRequest.model_validate(body).model_dump(exclude_none=True) == body
