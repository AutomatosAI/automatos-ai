"""PRD-256 N17-8 (F440): platform_get_agent by name reads the agent the board gives work to.

Night 16: the board's WRITER (#58, active) took the cards, while platform_get_agent('WRITER')
read #309: ``ilike '%WRITER%'`` and the first row, with no order, no whole-name and no active
preference. It now reads the board's agent (the active one carrying the whole name) and lists
every namesake with its status. The write resolvers that read a name the same way
(``_resolve_agent``, ``resolve_agent``) now refuse a name several agents carry, naming them.
These run against Postgres (rolled back).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

ROLE = "Writer"


@pytest.fixture
def board(db_session, seed_workspace):
    """Three WRITERs: two switched off, then the board's (active, with a skill)."""
    from core.models import Agent, Skill, agent_skills

    ws = UUID(seed_workspace())

    def agent(name, status="inactive"):
        made = Agent(name=name, agent_type="custom", description="Writes the newsletter.", status=status,
                     configuration={}, model_config={"provider": "openrouter", "model_id": "openai/gpt-4o"},
                     job_title=ROLE, workspace_id=ws, created_by="platform", owner_type="workspace",
                     owner_id=str(ws))
        db_session.add(made)
        db_session.flush()
        return made.id

    writers = [agent("WRITER"), agent("WRITER"), agent("WRITER", status="active")]
    voice = Skill(name="f440-newsletter-voice", skill_type="technical", description="The newsletter's voice")
    db_session.add(voice)
    db_session.flush()
    db_session.execute(agent_skills.insert().values(agent_id=writers[2], skill_id=voice.id))
    return NS(db=db_session, ws=ws, agent=agent, writers=writers)


def _by_name(board, said):
    from modules.tools.discovery.handlers_agents import get_agent

    return asyncio.run(get_agent(board.db, board.ws, {"agent_name": said}))


def _line(agent_id, status, name="WRITER"):
    return f"{agent_id} · {name} · {ROLE} · {status}"


# ── platform_get_agent by name ───────────────────────────────────────────────

def test_by_name_it_reads_the_boards_agent_and_lists_the_namesakes(board):
    from modules.tools.discovery.agent_refs import resolve_active_agent

    got = _by_name(board, "WRITER")
    first, second, active = board.writers

    assert got["agent"]["id"] == resolve_active_agent(board.db, board.ws, "WRITER")[0].id == active
    assert got["agent"]["assigned_skills_count"] == 1
    assert got["same_name"] == [_line(first, "inactive"), _line(second, "inactive"), _line(active, "active")]


def test_two_active_namesakes_give_the_lowest_id_and_a_contained_name_is_never_listed(board):
    newer = board.agent("WRITER", status="active")
    board.agent("NEWSLETTER WRITER", status="active")

    got = _by_name(board, "writer")

    assert got["agent"]["id"] == board.writers[2] < newer
    assert len(got["same_name"]) == 4 and not any("NEWSLETTER" in line for line in got["same_name"])
    assert _by_name(board, "writ")["agent"]["id"] == board.writers[0]   # a part of the name still reads


def test_one_agent_carrying_the_name_lists_no_namesakes_and_none_is_not_found(board):
    board.agent("EDITOR", status="active")

    assert "same_name" not in _by_name(board, "EDITOR")
    assert _by_name(board, "NOBODY") == {"success": False, "error": "Agent not found"}


def test_a_widget_turn_sees_only_who_the_agent_is(board):
    from core.security.surface import WIDGET, turn_surface

    with turn_surface(WIDGET, ("chat", "agents:read")):
        got = _by_name(board, "WRITER")

    assert got == {"success": True,
                   "agent": {"name": "WRITER", "description": "Writes the newsletter.", "status": "active"}}


# ── a write by a shared name changes nothing and names each ─────────────────

def _rows(board):
    from core.models import Agent

    board.db.expire_all()
    rows = board.db.query(Agent).filter(Agent.id.in_(board.writers)).order_by(Agent.id).all()
    return [(row.id, row.description, row.configuration, row.model_config) for row in rows]


@pytest.mark.parametrize("change", [{"description": "Writes the Friday newsletter."},
                                    {"runtime": "cli", "model": "opus"}], ids=["description", "runtime"])
def test_an_update_by_a_shared_name_changes_nothing_and_names_each(board, change, monkeypatch):
    from config import config
    from modules.tools.discovery.handlers_agents import update_agent
    from modules.tools.discovery.handlers_assignments import resolve_agent

    monkeypatch.setattr(config, "CLI_RUNTIME_ENABLED", True, raising=False)
    before = _rows(board)

    reply = asyncio.run(update_agent(board.db, board.ws, {"agent_name": "WRITER", **change}))

    assert reply["success"] is False and reply["error"].startswith("3 agents match agent_name 'WRITER'")
    assert all(f"{agent_id}:WRITER" in reply["error"] for agent_id in board.writers)
    assert _rows(board) == before
    assert resolve_agent(board.db, board.ws, {"agent_name": "WRITER"})[0] is None
