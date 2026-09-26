"""F188 (night 6) — onboarding moves past boom when its payoff happens.

The workspace reached boom at 02:02:53 and stayed there all night. At 02:08:39
search_knowledge answered from the owner's own documents (991, 992), and the
owner said "Spot on". The first mission ran at 02:53. Only Auto's
platform_update_onboarding(advance_to="powerup") could move the stage, and Auto
never made that call. Now the events move it: on SaaS to powerup, on local to
completed, since powerup has no local UI. Only from boom, only for the owner's
own documents (not the agents' reports, not a widget visitor's search), and a
mission's first start, not a resume.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from services import onboarding_state

NIGHT_6_HIT = [{"document_id": 991, "content": "Harbourline voice: warm, plain, no exclamation marks.",
                "similarity": 0.81},
               {"document_id": 992, "content": "Wholesale: Harbour Blend £21/kg, decaf £24/kg.", "similarity": 0.78}]


@pytest.fixture
def workspace(monkeypatch):
    """A workspace on a committing engine (the move is its own session's commit)."""
    from core.database import database
    from core.database.database import get_database_url

    engine = create_engine(get_database_url())
    Session = sessionmaker(bind=engine)
    monkeypatch.setattr(database, "SessionLocal", Session)
    ids = []

    def make(stage):
        ws = uuid.uuid4()
        ids.append(ws)
        doc = {"stage": stage, "stages": {stage: "2026-09-26T02:02:53+00:00"}, "segment": {}}
        with engine.begin() as conn:
            conn.execute(text("INSERT INTO workspaces (id, name, onboarding) "
                              "VALUES (CAST(:id AS uuid), 'f188-night-6', CAST(:doc AS jsonb))"),
                         {"id": str(ws), "doc": json.dumps(doc)})
        return ws

    def stage(ws):
        with engine.connect() as conn:
            return conn.execute(text("SELECT onboarding->>'stage' FROM workspaces WHERE id = CAST(:id AS uuid)"),
                                {"id": str(ws)}).scalar()

    yield NS(make=make, stage=stage, Session=Session)
    with engine.begin() as conn:
        for ws in ids:
            conn.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": str(ws)})
    engine.dispose()


@pytest.fixture
def edition(monkeypatch):
    from config import config

    def set_edition(local):
        monkeypatch.setattr(type(config), "IS_LOCAL_EDITION", local)
    set_edition(False)
    return set_edition


# ── search_knowledge ────────────────────────────────────────────────────────

class _Rows:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows


def _search(ws, *, source_type=None):
    from modules.agents.services.agent_platform_tools import AgentPlatformTools

    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = NS(workspace_id=ws, team=None)
    db.execute.return_value = _Rows([(991, "brand-voice.md", "/uploads/a.md", source_type),
                                     (992, "wholesale-prices.md", "/uploads/b.md", source_type)])
    tools = AgentPlatformTools.__new__(AgentPlatformTools)

    async def retrieve_context(**kwargs):
        return NS(chunks=[dict(chunk) for chunk in NIGHT_6_HIT])

    tools.db, tools.rag_config, tools.logger = db, None, MagicMock()
    tools.rag_service = NS(retrieve_context=retrieve_context)
    return asyncio.run(tools.execute_tool("search_knowledge", {"query": "What's our wholesale price?"}, 324))


def test_night_6s_answer_from_the_owners_documents_moves_boom_on(workspace, edition):
    ws = workspace.make("boom")
    result = _search(ws)

    assert result["success"] is True
    assert workspace.stage(ws) == "powerup"


def test_on_local_it_moves_to_completed(workspace, edition):
    edition(True)
    ws = workspace.make("boom")
    _search(ws)
    assert workspace.stage(ws) == "completed"


@pytest.mark.parametrize("case", ["agents-reports", "not-at-boom", "widget-turn"])
def test_nothing_else_moves_it(workspace, edition, monkeypatch, case):
    import core.security.surface as surface

    ws = workspace.make("teach" if case == "not-at-boom" else "boom")
    if case == "widget-turn":
        monkeypatch.setattr(surface, "widget_turn", lambda: True)
    _search(ws, source_type="agent_output" if case == "agents-reports" else None)
    assert workspace.stage(ws) == ("teach" if case == "not-at-boom" else "boom")


# ── a mission's first start ─────────────────────────────────────────────────

def test_a_missions_first_start_moves_boom_on(workspace, edition):
    from services.orchestration_state import _note_first_mission

    ws = workspace.make("boom")
    db = workspace.Session()
    try:
        _note_first_mission(db, NS(id=uuid.uuid4(), workspace_id=ws))
        db.commit()
    finally:
        db.close()
    assert workspace.stage(ws) == "powerup"


def test_only_the_first_start_is_the_payoff(monkeypatch):
    from core.models.orchestration_enums import ActorType, RunState
    from services import orchestration_state

    noted = []
    monkeypatch.setattr(orchestration_state, "_note_first_mission", lambda db, run: noted.append(run.id))
    monkeypatch.setattr("services.orchestration_board_bridge.sync_mission_board_status", lambda db, run: None)
    run = NS(id=uuid.uuid4(), workspace_id=uuid.uuid4(), state=RunState.AWAITING_APPROVAL.value, started_at=None,
             state_type=None, stop_reason=None, stop_detail=None)
    orchestration_state.transition_run(MagicMock(), run, RunState.RUNNING, ActorType.HUMAN, actor_id="user_2")
    orchestration_state.transition_run(MagicMock(), run, RunState.PAUSED, ActorType.HUMAN, actor_id="user_2")
    orchestration_state.transition_run(MagicMock(), run, RunState.RUNNING, ActorType.HUMAN, actor_id="user_2")
    assert noted == [run.id]                                   # the resume is not a first start
