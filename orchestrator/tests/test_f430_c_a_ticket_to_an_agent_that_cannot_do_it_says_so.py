"""PRD-256 F430 (C): a ticket given to a CLI agent whose sessions cannot call what it names says so.

Socials day run (14 Oct): "make a quote card post" went to the Social Media Director, a CLI agent
whose sessions had no Socials tool; the assignment checked only its model (F141), the run refused
three times, and asked why, Auto guessed. Now every board path that gives a card to an agent
(platform_create_task, platform_assign_task, platform_update_task with an agent, and the REST
create and PATCH) answers with ``cannot_call`` (the platform tools the title and brief name that the
agent's sessions cannot call, exactly what its tools block would call missing) and ``warning``.
The assignment always stands: a check that fails leaves the answer as it was.

PURE: ``calls_it_cannot_make`` and ``assignment_warning``. On Postgres (the orchestrator-tests
job's, one rolled-back transaction): the tools and the routes, on real agents.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from sqlalchemy import text

from services import cli_session_prompt
from services.cli_session_prompt import calls_it_cannot_make
from services.session_capability_gaps import assignment_warning

POST_BRIEF = "Make a quote card post with platform_create_social_post, render true."
NO_SOCIALS = {"runtime": "cli", "session_tool_groups": ["documents"]}


def _cli(**config):
    return NS(id=348, name="Social Media Director", configuration={"runtime": "cli", **config})


# ── pure: what a CLI agent's sessions cannot call ───────────────────────────

def test_a_cli_agent_without_socials_cannot_make_the_post_the_brief_names():
    assert calls_it_cannot_make(_cli(session_tool_groups=["documents"]), POST_BRIEF) == ["platform_create_social_post"]


@pytest.mark.parametrize("agent", [
    _cli(),                                                     # the default groups carry Socials
    NS(id=268, name="Content Creator", configuration={"runtime": "api"}),
    NS(id=269, name="No configuration", configuration=None),
])
def test_an_agent_that_can_call_it_is_not_flagged(agent):
    assert calls_it_cannot_make(agent, POST_BRIEF) == []


@pytest.mark.parametrize("brief, flagged", [
    ("List the folder with workspace_list_dir first.", ["workspace_list_dir"]),   # no session has it
    ("Read it with workspace_read_file.", []),                                    # the CLI's own Read
    ("Don't use any platform_ tool for this.", []),                               # a family word, no tool
])
def test_only_tools_a_session_genuinely_lacks_are_flagged(brief, flagged):
    assert calls_it_cannot_make(_cli(), brief) == flagged


def test_the_warning_names_the_agent_what_it_cannot_call_and_the_switch():
    said = assignment_warning(_cli(session_tool_groups=["documents"]), "Quote card", POST_BRIEF)
    assert said["cannot_call"] == ["platform_create_social_post"]
    assert said["warning"].startswith("Social Media Director runs as a CLI session and cannot call")
    assert "Socials tool group" in said["warning"] and "Model tab" in said["warning"]
    assert assignment_warning(_cli(), "Quote card", POST_BRIEF) == {}


def test_a_check_that_fails_is_logged_and_warns_nothing(monkeypatch, caplog):
    def broken(agent, said):
        raise RuntimeError("the tool list could not be read")

    monkeypatch.setattr(cli_session_prompt, "calls_it_cannot_make", broken)
    with caplog.at_level("ERROR"):
        assert assignment_warning(_cli(session_tool_groups=["documents"]), "Quote card", POST_BRIEF) == {}
    assert any("[F430]" in record.getMessage() for record in caplog.records)


# ── on Postgres: every path that gives a card to an agent ───────────────────

@pytest.fixture
def board(db_session, seed_workspace, monkeypatch):
    import api.board_tasks as bt
    from modules.tools.discovery import handlers_board_tasks as handlers

    ws = UUID(seed_workspace())

    def agent(name, configuration):
        return db_session.execute(
            text("INSERT INTO agents (name, agent_type, workspace_id, status, configuration) "
                 "VALUES (:name, 'custom', CAST(:w AS uuid), 'active', CAST(:c AS json)) RETURNING id"),
            {"name": name, "w": str(ws), "c": json.dumps(configuration)}).scalar()

    agents = NS(director=agent("Social Media Director", NO_SOCIALS), writer=agent("WRITER", {"runtime": "cli"}),
                creator=agent("Content Creator", {}))
    monkeypatch.setattr(handlers, "_notify_dispatch_safe", lambda db, ws, task_id: None)
    monkeypatch.setattr(bt, "notify_task_available", lambda db, **kw: None)
    monkeypatch.setattr(bt, "_launch_task_execution", lambda **kw: None)
    return NS(db=db_session, ws=ws, agents=agents,
              ctx=NS(workspace_id=ws, user=NS(id="owner@cafe.test", email="owner@cafe.test", clerk_user_id=None)))


def _create(board, agent_name):
    from modules.tools.discovery.handlers_board_tasks import create_board_task

    return asyncio.run(create_board_task(board.db, board.ws, {
        "title": "Quote card for the Harvest Club", "description": POST_BRIEF, "assigned_agent_name": agent_name}))


def _card(board, **fields):
    from core.models.core import BoardTask

    task = BoardTask(**{"workspace_id": board.ws, "title": "Quote card for the Harvest Club", "description": POST_BRIEF,
                        "priority": "medium", "source_type": "user", "status": "inbox", "attempts": 0, **fields})
    board.db.add(task)
    board.db.flush()
    return task


def _warned(answer, board):
    assert answer["cannot_call"] == ["platform_create_social_post"], answer
    assert "Social Media Director" in answer["warning"] and "Socials tool group" in answer["warning"]


@pytest.mark.integration
def test_platform_create_task_files_the_card_and_says_what_the_agent_cannot_call(board):
    from core.models.core import BoardTask

    answer = _create(board, "Social Media Director")
    assert answer["success"] is True
    _warned(answer, board)
    assert answer["warning"].endswith("Tell the owner in one line.")              # Auto relays it
    task = board.db.query(BoardTask).filter(BoardTask.workspace_id == board.ws).one()
    assert task.assigned_agent_id == board.agents.director                       # the assignment stands


@pytest.mark.integration
@pytest.mark.parametrize("agent_name", ["WRITER", "Content Creator"])
def test_an_agent_that_can_make_the_post_gets_no_warning(board, agent_name):
    answer = _create(board, agent_name)
    assert answer["success"] is True and "cannot_call" not in answer and "warning" not in answer


@pytest.mark.integration
def test_platform_assign_task_carries_the_warning(board):
    from modules.tools.discovery.handlers_board_task_assign import assign_board_task

    task = _card(board)
    answer = asyncio.run(assign_board_task(board.db, board.ws, {"task_id": task.id,
                                                                 "agent_name": "Social Media Director"}))
    assert answer["success"] is True
    _warned(answer, board)


@pytest.mark.integration
def test_platform_update_task_with_an_agent_and_an_edit_keeps_the_warning(board):
    from modules.tools.discovery.handlers_board_tasks import update_board_task

    task = _card(board)
    answer = asyncio.run(update_board_task(board.db, board.ws, {
        "task_id": task.id, "agent_name": "Social Media Director", "title": "Quote card, October"}))
    assert answer["success"] is True and answer["assigned_agent"] == "Social Media Director"
    _warned(answer, board)


@pytest.mark.integration
def test_the_rest_create_and_patch_carry_the_warning(board):
    import api.board_tasks as bt

    from tests.test_1094_a_ticket_with_no_agent_is_never_in_progress import _Request

    created = asyncio.run(bt.create_task(_Request({"title": "Quote card", "description": POST_BRIEF,
                                                   "assigned_agent_id": board.agents.director}),
                                         ctx=board.ctx, db=board.db))
    _warned(created, board)
    assert not created["warning"].endswith("Tell the owner in one line.")       # the owner reads this one

    task = _card(board)
    patched = asyncio.run(bt.update_task(task.id, _Request({"assigned_agent_id": board.agents.director}),
                                         ctx=board.ctx, db=board.db))
    _warned(patched, board)
    retitled = asyncio.run(bt.update_task(task.id, _Request({"title": "Quote card, October"}),
                                          ctx=board.ctx, db=board.db))
    assert "cannot_call" not in retitled                                         # no assignment, no warning


@pytest.mark.integration
def test_a_check_that_fails_never_stops_the_assignment(board, monkeypatch):
    from core.models.core import BoardTask

    def broken(agent, said):
        raise RuntimeError("the tool list could not be read")

    monkeypatch.setattr(cli_session_prompt, "calls_it_cannot_make", broken)
    answer = _create(board, "Social Media Director")
    assert answer["success"] is True and "cannot_call" not in answer
    task = board.db.query(BoardTask).filter(BoardTask.workspace_id == board.ws).one()
    assert task.assigned_agent_id == board.agents.director
