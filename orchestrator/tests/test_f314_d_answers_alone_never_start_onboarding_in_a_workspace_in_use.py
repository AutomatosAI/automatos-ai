"""F314 (night 9): saved answers alone never start onboarding in a workspace already in use.

Chat 9928b259: in a workspace holding 12 of the owner's documents, the owner answered
two of the quiz's questions and said "Now please just answer". Saving the answers moved
the stage on by itself (implied_stage), so the quiz's guidance and AutoBrain's Tier 0
pin rode every turn of the night. The answers are kept; only a started onboarding runs.
"""
from __future__ import annotations

from uuid import UUID

import pytest
from sqlalchemy import text

ANSWERS = {"business": "Harbourline Coffee Roasters in Bristol - roastery, club, 30 cafés wholesale",
           "comfort": "technical"}


@pytest.fixture
def workspace(db_session, seed_workspace):
    from core.models.workspaces import Workspace

    ws_id = UUID(seed_workspace())
    return db_session.query(Workspace).filter(Workspace.id == ws_id).one()


def _in_use(db_session, workspace):
    db_session.execute(text("INSERT INTO documents (workspace_id, filename, status, upload_date) "
                            "VALUES (CAST(:ws AS uuid), 'wholesale-terms-2026.md', 'completed', NOW())"),
                       {"ws": str(workspace.id)})
    db_session.flush()


def test_answers_in_a_workspace_in_use_are_kept_and_start_nothing(db_session, workspace):
    from services.onboarding_state import current_stage, get_onboarding, is_onboarding_active, set_segment

    _in_use(db_session, workspace)

    set_segment(db_session, workspace, ANSWERS)

    assert current_stage(workspace) == "not_started"
    assert get_onboarding(workspace)["segment"] == ANSWERS
    assert is_onboarding_active(workspace) is False


def test_answers_in_a_new_workspace_still_move_the_questions_on(db_session, workspace):
    from services.onboarding_state import current_stage, set_segment

    set_segment(db_session, workspace, ANSWERS)

    assert current_stage(workspace) == "questions"
    set_segment(db_session, workspace, {"goal": "Answer the cafés' questions"})
    assert current_stage(workspace) == "teach"


def test_onboarding_started_in_a_workspace_in_use_takes_its_answers(db_session, workspace):
    """The owner asked to set up, and Auto started it: answers move it on as before."""
    from services.onboarding_state import advance_onboarding_stage, current_stage, set_segment

    _in_use(db_session, workspace)
    advance_onboarding_stage(db_session, workspace, "questions")

    set_segment(db_session, workspace, {**ANSWERS, "goal": "Answer the cafés' questions"})

    assert current_stage(workspace) == "teach"


def test_the_tool_saves_answers_without_starting_it(db_session, workspace):
    import asyncio

    from modules.tools.discovery.handlers_onboarding import update_onboarding
    from services.onboarding_state import current_stage

    _in_use(db_session, workspace)

    out = asyncio.run(update_onboarding(db_session, workspace.id, {"segment": ANSWERS, "timezone": "Europe/London"}))

    assert out["success"] is True and out["data"]["stage"] == "not_started"
    db_session.refresh(workspace)
    assert current_stage(workspace) == "not_started"
