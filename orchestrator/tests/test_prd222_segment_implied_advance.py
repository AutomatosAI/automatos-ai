"""PRD-222 — the recorded answers move the stage; the payoff gate cannot be skipped.

Live test 2026-09-02 (prod): the model saved answers through the tool twice
(success=True) but never sent advance_to — stage frozen at not_started while Auto
recited the questions. Now a segment-only write advances to what the answers
prove: any answer → questions; all three → teach. Explicit targets still win.
Also: advance_to="completed" from building walked around the boom gate.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from core.models.core import Agent
from core.models.orchestration import OrchestrationRun
from modules.tools.discovery.handlers_onboarding import update_onboarding
from services.onboarding_state import (
    InvalidStageTransition,
    advance_onboarding_stage,
    current_stage,
    implied_stage,
    set_segment,
)


class _Workspace:
    def __init__(self, stage="not_started", segment=None, funnel=None):
        self.id = uuid4()
        self.onboarding = {"stage": stage, "stages": {}, "segment": dict(segment or {})}
        if funnel:
            self.onboarding["funnel"] = funnel


def _db(agents=0, missions=0, workspace=None):
    counts = {Agent: agents, OrchestrationRun: missions}
    db = MagicMock()

    def _query(model):
        q = MagicMock()
        q.filter.return_value.count.return_value = counts.get(model, 0)
        q.filter.return_value.first.return_value = workspace
        return q

    db.query.side_effect = _query
    return db


# ── implied_stage (pure) ────────────────────────────────────────────────────

def test_any_answer_implies_questions_and_all_three_imply_teach():
    assert implied_stage("not_started", {}) is None
    assert implied_stage("not_started", {"business": "a barber"}) == "questions"
    assert implied_stage("questions", {"business": "a barber", "goal": "bookings"}) is None
    full = {"business": "a barber", "goal": "bookings", "comfort": "novice"}
    assert implied_stage("not_started", full) == "teach"
    assert implied_stage("questions", full) == "teach"


def test_later_stages_never_move_on_answers():
    full = {"business": "a", "goal": "b", "comfort": "c"}
    for stage in ("teach", "proposal", "building", "boom", "powerup"):
        assert implied_stage(stage, full) is None


# ── set_segment advances on what the answers prove ─────────────────────────

def test_first_answer_moves_not_started_to_questions():
    ws = _Workspace()
    set_segment(None, ws, {"business": "Snip & Fade, a barber shop"})
    assert current_stage(ws) == "questions"
    assert ws.onboarding["segment"]["business"] == "Snip & Fade, a barber shop"
    assert "questions" in ws.onboarding["stages"]  # funnel stamp, like any advance


def test_answers_accumulate_and_the_third_moves_to_teach():
    ws = _Workspace()
    set_segment(None, ws, {"business": "a barber"})
    set_segment(None, ws, {"goal": "bookings"})
    assert current_stage(ws) == "questions"
    set_segment(None, ws, {"comfort": "brand new"})
    assert current_stage(ws) == "teach"
    assert ws.onboarding["segment"] == {"business": "a barber", "goal": "bookings", "comfort": "brand new"}


def test_answers_at_a_later_stage_merge_without_moving():
    ws = _Workspace("proposal", segment={"business": "a", "goal": "b", "comfort": "c"})
    set_segment(None, ws, {"team_size": 4})
    assert current_stage(ws) == "proposal"
    assert ws.onboarding["segment"]["team_size"] == 4


def test_explicit_advance_target_is_never_overridden():
    ws = _Workspace()
    full = {"business": "a", "goal": "b", "comfort": "c"}
    advance_onboarding_stage(None, ws, "questions", segment=full)
    assert current_stage(ws) == "questions"  # the model asked for questions; it gets questions


# ── through the tool handler ───────────────────────────────────────────────

def test_handler_segment_only_calls_walk_the_stage_forward():
    ws = _Workspace()
    db = _db(workspace=ws)
    r1 = asyncio.run(update_onboarding(db, ws.id, {"segment": {"business": "a barber"}}))
    assert r1["success"] is True and r1["data"]["stage"] == "questions"
    asyncio.run(update_onboarding(db, ws.id, {"segment": {"goal": "bookings"}}))
    r3 = asyncio.run(update_onboarding(db, ws.id, {"segment": {"comfort": "novice"}}))
    assert r3["data"]["stage"] == "teach"


# ── the payoff gate cannot be skipped around ───────────────────────────────

def test_completed_from_building_needs_the_build_too():
    ws = _Workspace("building")
    with pytest.raises(InvalidStageTransition, match="completed needs a build"):
        advance_onboarding_stage(_db(0, 0), ws, "completed")
    with pytest.raises(InvalidStageTransition, match="powerup needs a build"):
        advance_onboarding_stage(_db(0, 0), ws, "powerup")
    assert current_stage(ws) == "building"


def test_past_the_payoff_the_gate_does_not_re_run():
    ws = _Workspace("boom")
    advance_onboarding_stage(_db(0, 0), ws, "powerup")   # already past boom: no re-check
    assert current_stage(ws) == "powerup"
    advance_onboarding_stage(_db(0, 0), ws, "completed")
    assert current_stage(ws) == "completed"


def test_skipped_is_untouched_by_the_gate():
    ws = _Workspace("building")
    advance_onboarding_stage(_db(0, 0), ws, "skipped")
    assert current_stage(ws) == "skipped"
