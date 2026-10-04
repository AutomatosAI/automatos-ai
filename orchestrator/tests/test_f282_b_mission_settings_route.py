"""F282 (night 8) — PATCH /api/missions/{id}/settings: one setting, one switch.

Whether a mission checked each step with the owner depended on how Auto
spelled the setting (modules/coordination/owner_checks.py), and the New
mission form had no switch of its own — the owner could only ask Auto, and
Auto sometimes said the setting was on when it wasn't. This route is the one
way to change it once a mission exists: the mission page's switch calls it
directly, and it always leaves the setting under one name, check_each_step.
"""
from __future__ import annotations

from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from fastapi import HTTPException

from api.mission_settings import MissionSettingsRequest, update_mission_settings
from core.models.orchestration import OrchestrationRun
from core.models.orchestration_enums import RunState


def _owner(ws):
    return NS(workspace_id=ws, user_id="2", auth_type="anonymous", user=NS(id="2"))


def _mission(db, ws, state=RunState.RUNNING, config=None):
    run = OrchestrationRun(workspace_id=ws, goal="Ship the weekly newsletter", state=state.value,
                           created_by="user_test", config=config if config is not None else {})
    db.add(run)
    db.flush()
    return run


def test_turns_the_setting_on(db_session, seed_workspace):
    ws = UUID(seed_workspace())
    run = _mission(db_session, ws)

    out = update_mission_settings(run.id, MissionSettingsRequest(check_each_step=True), ctx=_owner(ws), db=db_session)

    db_session.refresh(run)
    assert out == {"id": str(run.id), "check_each_step": True}
    assert run.config == {"check_each_step": True}


def test_clears_whatever_spelling_auto_used(db_session, seed_workspace):
    ws = UUID(seed_workspace())
    run = _mission(db_session, ws, config={"wait_for_me": True, "tags": ["launch"]})

    out = update_mission_settings(run.id, MissionSettingsRequest(check_each_step=False), ctx=_owner(ws), db=db_session)

    db_session.refresh(run)
    assert out == {"id": str(run.id), "check_each_step": False}
    assert run.config == {"tags": ["launch"]}          # wait_for_me gone, nothing else disturbed


@pytest.mark.parametrize("state", [RunState.COMPLETED, RunState.FAILED, RunState.CANCELLED])
def test_refuses_a_mission_that_is_done(db_session, seed_workspace, state):
    ws = UUID(seed_workspace())
    run = _mission(db_session, ws, state=state)

    with pytest.raises(HTTPException) as refused:
        update_mission_settings(run.id, MissionSettingsRequest(check_each_step=True), ctx=_owner(ws), db=db_session)

    assert refused.value.status_code == 409
    assert state.value in refused.value.detail
    db_session.refresh(run)
    assert run.config == {}                             # refused before it touched anything


@pytest.mark.parametrize("state", [RunState.AWAITING_APPROVAL, RunState.PLANNING, RunState.RUNNING, RunState.PAUSED])
def test_a_mission_that_can_still_run_accepts_it(db_session, seed_workspace, state):
    ws = UUID(seed_workspace())
    run = _mission(db_session, ws, state=state)

    out = update_mission_settings(run.id, MissionSettingsRequest(check_each_step=True), ctx=_owner(ws), db=db_session)

    assert out["check_each_step"] is True


def test_scoped_to_the_workspace(db_session, seed_workspace):
    ws = UUID(seed_workspace())
    other_ws = UUID(seed_workspace())
    run = _mission(db_session, ws)

    with pytest.raises(HTTPException) as refused:
        update_mission_settings(run.id, MissionSettingsRequest(check_each_step=True), ctx=_owner(other_ws), db=db_session)

    assert refused.value.status_code == 404
