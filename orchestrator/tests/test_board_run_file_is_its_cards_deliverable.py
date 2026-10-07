"""A file an agent writes during a board run is its ticket's Deliverable (7 Oct).

A board run names its card in the tool calls' context as ``board_task_id``
(agent_factory), but ``exec_workspace._derive_source`` read only heartbeat, mission,
task, Playbook and trigger ids: the file the agent wrote landed as a ``chat``
Deliverable with no source, off its card and without the card's tags. The card's id now
attributes it (``source_type='task'``), so DeliverableService.register adds the card's
owner tags. A context that names a mission, a Playbook or a trigger, and a chat turn,
keep the attribution they had.
"""
from __future__ import annotations

import contextlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

CARD = 612
WS = uuid4()


def _exec_workspace():
    """exec_workspace loaded by path: its own imports are stdlib-only, and the
    modules.tools package would pull the whole tool stack."""
    target = Path(__file__).resolve().parents[1] / "modules" / "tools" / "execution" / "exec_workspace.py"
    spec = importlib.util.spec_from_file_location("board_run_exec_workspace_under_test", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def db(monkeypatch):
    """The session the write's registration opens: the card's tags, and the row it writes."""
    import core.database.database as database_mod

    session = MagicMock()
    session.execute.return_value.fetchone.return_value = (uuid4(), True)
    session.query.return_value.filter.return_value.scalar.return_value = [
        "session", "mission", "sim-night-2026-10-07", "Harbourline",
    ]
    monkeypatch.setattr(database_mod, "SessionLocal", lambda: contextlib.nullcontext(session))
    return session


def _write(caller_context):
    _exec_workspace()._auto_register_deliverable(
        workspace_id=WS, file_path="plans/launch-board.png", write_result={"success": True, "size": 2048},
        agent_id=None, caller_context=caller_context, trace_id="t-1",
    )


def _registered(session) -> dict:
    return session.execute.call_args[0][1]


def test_a_board_run_write_is_its_cards_deliverable_with_the_cards_tags(db):
    _write({"board_task_id": CARD, "field_context": {"field_id": "f-1"}})

    params = _registered(db)
    assert (params["source_type"], params["source_id"]) == ("task", str(CARD))
    assert json.loads(params["extra"])["tags"] == ["sim-night-2026-10-07", "harbourline"]


def test_a_chat_write_is_unchanged(db):
    _write(None)

    params = _registered(db)
    assert (params["source_type"], params["source_id"]) == ("chat", None)
    assert "tags" not in json.loads(params["extra"])
    db.query.assert_not_called()


@pytest.mark.parametrize("context, source", [
    (None, ("chat", None)),
    ({}, ("chat", None)),
    ({"board_task_id": CARD}, ("task", str(CARD))),
    ({"mission_id": "m-1", "board_task_id": CARD}, ("mission", "m-1")),
    ({"playbook_id": "pb-1", "board_task_id": CARD}, ("playbook", "pb-1")),
    ({"trigger_id": "tr-1", "board_task_id": CARD}, ("trigger", "tr-1")),
    ({"source_type": "upload", "source_id": "u-1", "board_task_id": CARD}, ("upload", "u-1")),
])
def test_only_a_context_naming_nothing_else_takes_the_card(context, source):
    assert _exec_workspace()._derive_source(context) == source
