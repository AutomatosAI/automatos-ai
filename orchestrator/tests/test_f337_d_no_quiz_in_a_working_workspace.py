"""F337 (night 10): "Can you make me a wholesale price list for cafés, as a spreadsheet?"
got "What's your business? … How comfortable are you with AI?" and no tool (chat
d2564ecf), in a workspace with weeks of cards and documents. Its onboarding sat at
``questions`` (answers saved on night 9, before F314), so F314's not-started rule passed
it by. A quiz left idle at ``questions`` in a workspace in use is not offered now; one the
owner is answering, or one in a new workspace, runs as before; and the quiz's own
guidance puts a piece of work first.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from sqlalchemy import text

PRICE_LIST = "Can you make me a wholesale price list for cafés, as a spreadsheet?"
QUIZ = "What's your business?"


@pytest.fixture
def workspace(db_session, seed_workspace):
    from core.models.workspaces import Workspace

    ws_id = UUID(seed_workspace())
    return db_session.query(Workspace).filter(Workspace.id == ws_id).one()


def _at_the_quiz(db_session, workspace, hours_ago: float) -> None:
    when = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    workspace.onboarding = {"stage": "questions", "stages": {"questions": when}, "segment": {"business": "coffee"},
                            "started_at": when, "updated_at": when}
    db_session.flush()


def _add_card(db_session, ws_id) -> None:
    from core.models.core import BoardTask

    db_session.add(BoardTask(workspace_id=ws_id, title="Quote for Rosa", status="done"))
    db_session.flush()


def _add_document(db_session, ws_id) -> None:
    db_session.execute(text("INSERT INTO documents (workspace_id, filename, status, upload_date) "
                            "VALUES (CAST(:ws AS uuid), 'harbourline-key-facts.csv', 'completed', NOW())"),
                       {"ws": str(ws_id)})
    db_session.flush()


def _section(db_session, ws_id, said):
    from modules.context.sections.base import SectionContext
    from modules.context.sections.onboarding import OnboardingSection

    ctx = SectionContext(agent=None, workspace_id=str(ws_id), db_session=db_session,
                         messages=[{"role": "user", "content": said}])
    return asyncio.run(OnboardingSection().render(ctx))


def test_a_quiz_left_idle_in_a_workspace_in_use_is_not_offered(db_session, workspace):
    from consumers.chatbot.auto import AutoBrain
    from services.onboarding_state import is_onboarding_active

    _add_card(db_session, workspace.id)
    _at_the_quiz(db_session, workspace, hours_ago=20)

    assert is_onboarding_active(workspace) is False
    assert AutoBrain(db_session, str(workspace.id))._onboarding_active() is False     # no Tier-0 pin
    assert _section(db_session, workspace.id, PRICE_LIST) == ""


def test_documents_alone_make_a_workspace_in_use_too(db_session, workspace):
    from services.onboarding_state import is_onboarding_active

    _add_document(db_session, workspace.id)
    _at_the_quiz(db_session, workspace, hours_ago=20)

    assert is_onboarding_active(workspace) is False


def test_a_quiz_the_owner_is_answering_still_runs(db_session, workspace):
    from services.onboarding_state import is_onboarding_active

    _add_card(db_session, workspace.id)
    _at_the_quiz(db_session, workspace, hours_ago=0.25)

    assert is_onboarding_active(workspace) is True
    assert QUIZ in _section(db_session, workspace.id, "Coffee, a roastery in Bristol.")


def test_a_new_workspace_keeps_its_quiz_however_long_it_waits(db_session, workspace):
    from services.onboarding_state import is_onboarding_active

    _at_the_quiz(db_session, workspace, hours_ago=20)

    assert is_onboarding_active(workspace) is True
    assert QUIZ in _section(db_session, workspace.id, "Hi")


def test_asking_to_set_up_still_starts_it(db_session, workspace):
    _add_card(db_session, workspace.id)
    _at_the_quiz(db_session, workspace, hours_ago=20)

    assert QUIZ in _section(db_session, workspace.id, "Please set up my workspace")


def test_the_quiz_puts_a_piece_of_work_first(db_session, workspace):
    out = _section(db_session, workspace.id, PRICE_LIST)

    assert QUIZ in out
    assert "do that work first" in out
    assert out.index("do that work first") < out.index(QUIZ)


@pytest.mark.parametrize("doc, idle", [
    ({"stage": "questions"}, True),                                                    # no stamps: an old row
    ({"stage": "questions", "updated_at": "not a time"}, True),
    ({"stage": "questions", "updated_at": "2026-10-05T09:00:00+00:00"}, True),
    ({"stage": "questions", "updated_at": "2026-10-05T11:30:00+00:00"}, False),
    ({"stage": "questions", "stages": {"questions": "2026-10-05T11:30:00"}}, False),    # naive reads as UTC
])
def test_idle_is_read_from_the_latest_onboarding_write(doc, idle):
    from services.onboarding_content import quiz_left_idle

    now = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)
    assert quiz_left_idle(NS(onboarding=doc), now=now) is idle
