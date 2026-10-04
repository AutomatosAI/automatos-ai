"""F314 (night 9): Auto answered a plain question with its onboarding quiz in a workspace
that already held 12 of the owner's documents (chat 9928b259: "Before we dive into the
coffee order, could you tell me a little about your business? 1. What's your
business? …"). The stage stayed not_started, so the quiz's guidance rode every turn and
AutoBrain pinned every turn to Tier 0. Onboarding that has not started is no longer
offered to a workspace in use; started or asked for, it runs as before.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import UUID

import pytest
from sqlalchemy import text

QUESTION = "Hi Auto. A café wants 10 kg of coffee next week. What do we charge them for delivery?"
QUIZ = "What's your business?"


@pytest.fixture
def workspace(db_session, seed_workspace):
    from core.models.workspaces import Workspace

    ws_id = UUID(seed_workspace())
    return db_session.query(Workspace).filter(Workspace.id == ws_id).one()


def _add_document(db_session, ws_id):
    db_session.execute(text("INSERT INTO documents (workspace_id, filename, status, upload_date) "
                            "VALUES (CAST(:ws AS uuid), 'wholesale-terms-2026.md', 'completed', NOW())"),
                       {"ws": str(ws_id)})
    db_session.flush()


def _section(db_session, ws_id, said):
    from modules.context.sections.base import SectionContext
    from modules.context.sections.onboarding import OnboardingSection

    ctx = SectionContext(agent=None, workspace_id=str(ws_id), db_session=db_session,
                         messages=[{"role": "user", "content": said}])
    return asyncio.run(OnboardingSection().render(ctx))


def test_a_new_empty_workspace_still_gets_the_quiz(db_session, workspace):
    from services.onboarding_state import is_onboarding_active

    assert is_onboarding_active(workspace) is True
    assert QUIZ in _section(db_session, workspace.id, QUESTION)


def test_a_workspace_with_documents_gets_an_answer_not_the_quiz(db_session, workspace):
    from consumers.chatbot.auto import AutoBrain
    from services.onboarding_state import is_onboarding_active

    _add_document(db_session, workspace.id)

    assert is_onboarding_active(workspace) is False
    assert AutoBrain(db_session, str(workspace.id))._onboarding_active() is False     # no Tier-0 pin
    assert _section(db_session, workspace.id, QUESTION) == ""


def test_a_workspace_with_cards_on_its_board_is_in_use(db_session, workspace):
    from core.models.core import BoardTask
    from services.onboarding_state import is_onboarding_active

    db_session.add(BoardTask(workspace_id=workspace.id, title="Reply to Rosa", status="review"))
    db_session.flush()

    assert is_onboarding_active(workspace) is False


def test_onboarding_the_owner_started_or_asked_for_still_runs(db_session, workspace):
    from services.onboarding_state import is_onboarding_active

    _add_document(db_session, workspace.id)
    assert QUIZ in _section(db_session, workspace.id, "Please set up my workspace")      # asked for

    now = datetime.now(timezone.utc).isoformat()
    workspace.onboarding = {"stage": "questions", "stages": {"questions": now}, "segment": {}, "started_at": now}
    db_session.flush()
    assert is_onboarding_active(workspace) is True                                       # started
    assert QUIZ in _section(db_session, workspace.id, QUESTION)
