"""F263 (night 8): "What's waiting for me?" answers with what the board's Needs you shows.

Four times Auto said "5 failed tasks" when nothing on the board had failed, naming
last week's failures by title, and it gave totals ("2 in review") without saying
which. The board summary counted a card as failed when it had ever carried an
error, and never said which cards were the owner's. Now the failures are the cards
failed now, by number, and the summary and the snapshot carry what the board's own
Needs you counts and lists.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID

import pytest


@pytest.fixture
def board(db_session, seed_workspace):
    from core.models.core import BoardTask

    ws = UUID(seed_workspace())

    def card(title, status, error=None):
        # F293 (on main): a card in Review counts once someone worked on it, so it carries an answer.
        task = BoardTask(workspace_id=ws, title=title, status=status, description="…", error_message=error,
                         result="Draft ready." if status == "review" else None)
        db_session.add(task)
        db_session.flush()
        return f"#{task.workspace_seq:04d}"

    return NS(db=db_session, ws=ws,
              redone=card("Weekly numbers", "done", "Task execution failed after 2 attempts: Empty response"),
              let_go=card("Supplier check", "cancelled", "Stalled: no progress for 120s"),
              failed=card("Price card for the market stall", "failed", "Stalled: no progress for 120s"),
              review=card("Reply to Owen Price", "review"),
              running=card("Rota for October", "in_progress"))


def test_a_failure_is_a_card_failed_now_by_its_number(board):
    from modules.tools.discovery.handlers_analytics import board_summary

    summary = asyncio.run(board_summary(board.db, board.ws, {}))
    assert [(c["number"], c["title"]) for c in summary["failed_tasks"]] == [
        (board.failed, "Price card for the market stall")]            # not the redone or cancelled cards
    assert summary["by_status"]["failed"] == 1


def test_whats_waiting_is_the_boards_needs_you_by_number(board):
    from modules.tools.discovery.handlers_analytics import board_snapshot, board_summary
    from services.needs_you import needs_you_counts

    summary = asyncio.run(board_summary(board.db, board.ws, {}))
    waiting = summary["waiting_for_you"]
    assert waiting["total"] == needs_you_counts(board.db, board.ws)["total"] == 2
    assert sorted((c["kind"], c["number"], c["title"]) for c in waiting["cards"]) == [
        ("failed", board.failed, "Price card for the market stall"),
        ("review", board.review, "Reply to Owen Price")]

    snapshot = asyncio.run(board_snapshot(board.db, board.ws, {}))
    assert snapshot["waiting_for_you"] == waiting


def test_a_widget_turn_is_told_nothing_of_the_owners(board):
    from core.security.surface import WIDGET, turn_surface
    from modules.tools.discovery.handlers_analytics import board_summary

    with turn_surface(WIDGET, ("chat", "tasks:read"), None):
        summary = asyncio.run(board_summary(board.db, board.ws, {}))
    assert "waiting_for_you" not in summary and "failed_tasks" not in summary
