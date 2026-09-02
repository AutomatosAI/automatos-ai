"""PRD-222 honesty gate — ``boom`` requires build evidence.

Live test 2026-08-29: two personas (Saffron, Waggle) advanced to the payoff
stage having built NOTHING, because the stage machine validated ORDERING only.
``boom`` is "here is your team", so it now needs the workspace to actually hold
a build: the package funnel stamp, a built agent, or a mission.

Pure-logic tests against a MagicMock session — no DB. The "built agent"
definition is pinned to the purge's survivor predicate (a workspace-owned agent
that is neither a system agent nor a hidden onboarding-role agent).
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
    BUILD_EVIDENCE_STAGE,
    InvalidStageTransition,
    advance_onboarding_stage,
    build_evidence,
    current_stage,
)

_PACKAGE_STAMP = {"package_installed": {"slug": "shopify-management", "at": "2026-08-29T00:00:00Z"}}


class _Workspace:
    def __init__(self, stage: str = "building", funnel: dict | None = None):
        self.id = uuid4()
        self.onboarding = {"stage": stage, "stages": {}, "segment": {}}
        if funnel:
            self.onboarding["funnel"] = funnel


def _db(agents: int = 0, missions: int = 0, workspace=None):
    """MagicMock session with per-model ``.count()`` and a ``.first()`` for the
    handler's workspace load. ``made`` keeps each query object so a test can
    inspect the filter clauses it was given."""
    counts = {Agent: agents, OrchestrationRun: missions}
    db = MagicMock()
    db.made = {}

    def _query(model):
        q = MagicMock()
        q.filter.return_value.count.return_value = counts.get(model, 0)
        q.filter.return_value.first.return_value = workspace
        db.made[model] = q
        return q

    db.query.side_effect = _query
    return db


# --------------------------------------------------------------------------- #
# State layer — the single writer refuses an empty payoff
# --------------------------------------------------------------------------- #


def test_gate_is_on_boom():
    assert BUILD_EVIDENCE_STAGE == "boom"


def test_boom_refused_when_nothing_was_built():
    ws = _Workspace("building")
    with pytest.raises(InvalidStageTransition, match="needs a build"):
        advance_onboarding_stage(_db(0, 0), ws, "boom")
    assert current_stage(ws) == "building"  # nothing written


def test_boom_allowed_on_the_package_funnel_stamp():
    ws = _Workspace("building", funnel=_PACKAGE_STAMP)
    advance_onboarding_stage(_db(0, 0), ws, "boom")
    assert current_stage(ws) == "boom"


def test_boom_allowed_on_a_built_agent():
    ws = _Workspace("building")
    advance_onboarding_stage(_db(agents=1), ws, "boom")
    assert current_stage(ws) == "boom"


def test_boom_allowed_on_a_mission():
    ws = _Workspace("building")
    advance_onboarding_stage(_db(missions=1), ws, "boom")
    assert current_stage(ws) == "boom"


def test_other_stages_never_consult_evidence():
    ws = _Workspace("proposal")
    db = _db(0, 0)
    advance_onboarding_stage(db, ws, "building")
    assert current_stage(ws) == "building"
    assert Agent not in db.made and OrchestrationRun not in db.made


def test_pure_document_path_keeps_ordering_only():
    # db=None is the logic-test path: no session to count against, so the
    # ordering validator alone applies, exactly as before this gate.
    ws = _Workspace("building")
    advance_onboarding_stage(None, ws, "boom")
    assert current_stage(ws) == "boom"


def test_ordering_still_validated_before_evidence():
    # A backward move is refused for ordering, never reaches the evidence read.
    ws = _Workspace("boom")
    db = _db(agents=5)
    with pytest.raises(InvalidStageTransition, match="non-forward"):
        advance_onboarding_stage(db, ws, "building")
    assert Agent not in db.made


# --------------------------------------------------------------------------- #
# The evidence read — shape + the "built agent" definition
# --------------------------------------------------------------------------- #


def test_build_evidence_shape():
    ws = _Workspace("building")
    assert build_evidence(_db(agents=2, missions=1), ws) == {
        "package_installed": False,
        "agents_built": 2,
        "missions": 1,
        "any": True,
    }
    assert build_evidence(_db(0, 0), ws)["any"] is False
    assert build_evidence(None, _Workspace("building", funnel=_PACKAGE_STAMP)) == {
        "package_installed": True,
        "agents_built": 0,
        "missions": 0,
        "any": True,
    }


def test_built_agent_definition_matches_the_purge_survivor_predicate():
    """Scoped to the workspace; excludes system agents and onboarding-role agents
    (NULL-safe, like ``workspace_purge._AGENT_SURVIVOR_SQL``)."""
    ws = _Workspace("building")
    db = _db(agents=1)
    build_evidence(db, ws)
    clauses = " | ".join(str(c) for c in db.made[Agent].filter.call_args.args).lower()
    assert "agents.workspace_id" in clauses
    assert "agents.is_system_agent is not true" in clauses
    assert "agents.required_role is null" in clauses
    assert "agents.required_role !=" in clauses
    mission_clauses = " | ".join(
        str(c) for c in db.made[OrchestrationRun].filter.call_args.args
    ).lower()
    assert "orchestration_runs.workspace_id" in mission_clauses


# --------------------------------------------------------------------------- #
# Tool handler — the refusal reaches Auto as a clean, honest error
# --------------------------------------------------------------------------- #


def test_handler_returns_the_honest_error_and_moves_nothing():
    ws = _Workspace("building")
    db = _db(0, 0, workspace=ws)
    res = asyncio.run(update_onboarding(db, ws.id, {"advance_to": "boom"}))
    assert res["success"] is False
    assert "needs a build" in res["error"]
    assert "package" in res["error"] and "agents" in res["error"]
    assert current_stage(ws) == "building"
    db.rollback.assert_called_once()
    db.commit.assert_not_called()


def test_handler_advances_to_boom_once_a_package_is_installed():
    ws = _Workspace("building", funnel=_PACKAGE_STAMP)
    db = _db(0, 0, workspace=ws)
    res = asyncio.run(update_onboarding(db, ws.id, {"advance_to": "boom"}))
    assert res["success"] is True
    assert res["data"]["stage"] == "boom"
    db.commit.assert_called_once()


def test_handler_advances_to_boom_on_built_agents():
    ws = _Workspace("building")
    res = asyncio.run(
        update_onboarding(_db(agents=3, workspace=ws), ws.id, {"advance_to": "boom"})
    )
    assert res["success"] is True and res["data"]["stage"] == "boom"


# --------------------------------------------------------------------------- #
# Prompt — Auto is told the rule up front
# --------------------------------------------------------------------------- #


def test_building_prompt_states_the_refusal():
    from modules.context.sections.onboarding import _STAGE_BUILDING

    low = _STAGE_BUILDING.lower()
    assert "refused until the workspace" in low
    assert "never announce a team before it exists" in low
