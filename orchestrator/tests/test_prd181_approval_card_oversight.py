"""PRD-181 S5 — EU-AI-Act Art.14 oversight on the approval card (backend payload).

An ``ask`` verdict's approval card must carry the autonomy risk tier + the
oversight rationale (why a human is in the loop). This tests the backend half —
the ``mission_approval`` card payload the ``result_formatter`` emits — gains a
``risk_tier`` and ``oversight_rationale`` sourced from the S6 AI-Act mapping. The
frontend half (MissionApprovalWidget) is covered by a vitest test.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

for _k in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"):
    os.environ.setdefault(_k, "test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
sys.modules.setdefault("camelot", types.ModuleType("camelot"))

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))


def test_mission_approval_card_carries_risk_tier_and_rationale():
    from modules.tools.formatting.result_formatter import ToolResultFormatter

    result = {
        "success": True,
        "mission_id": "abc-123",
        "awaiting_approval": True,
        "goal": "refund the customer and email them",
        "state": "AWAITING_APPROVAL",
        "task_count": 2,
        "tasks": [{"title": "issue refund"}, {"title": "send email"}],
        # the mission carries a dominant risk signal from its planned actions
        "risk_class": "external_side_effect",
    }
    fd = ToolResultFormatter.format_for_frontend(result, "platform_create_mission")

    card = fd["mission_approval"]
    assert card["mission_id"] == "abc-123"
    # S5: the card shows the AI-Act oversight tier + rationale.
    assert card["risk_tier"] == "human_in_the_loop"
    assert card["risk_class"] == "external_side_effect"
    assert card["oversight_rationale"], "the card must explain why a human is in the loop"
    assert card["requires_approval"] is True


def test_mission_approval_defaults_to_human_in_the_loop_without_risk_signal():
    """A mission that reached AWAITING_APPROVAL is, by definition, human-in-the-
    loop — even if no explicit risk_class was attached, the card must not imply
    'no oversight'."""
    from modules.tools.formatting.result_formatter import ToolResultFormatter

    result = {
        "success": True,
        "mission_id": "def-456",
        "awaiting_approval": True,
        "goal": "do the thing",
        "task_count": 1,
    }
    fd = ToolResultFormatter.format_for_frontend(result, "platform_create_mission")
    card = fd["mission_approval"]
    assert card["risk_tier"] == "human_in_the_loop"
    assert card["requires_approval"] is True
    assert card["oversight_rationale"]


def test_non_approval_result_has_no_card():
    from modules.tools.formatting.result_formatter import ToolResultFormatter

    fd = ToolResultFormatter.format_for_frontend(
        {"success": True, "mission_id": "x", "awaiting_approval": False}, "platform_create_mission"
    )
    assert "mission_approval" not in fd
