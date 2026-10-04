"""The board cards of a mission's redo (F267, F284, F286, night 8).

- F267: a step sent back can wait its turn before its redo runs (a mission that checks
  each step runs one step at a time), and the waiting step said "In progress": #0446.1
  for 3½ minutes while another step ran. Its card now says Assigned until its redo
  starts; the mission's sync moves it to In progress when the step is picked up
  (``orchestration_board_bridge.sync_board_status``). A Claude Code agent's card stays
  as it is: its host claims an Assigned card of its agent as a ticket of its own
  (``board_dispatcher.CLI_CLAIMABLE_MIRRORS``), and only the mission may run the step.
- F284, F286: a card whose work runs again (the card of a mission that opens again, of
  a step built from one that was sent back) keeps what it showed in its history
  (``planning_data.previous_runs``, as a Reject keeps the draft it sends back), and
  shows no result until the new one lands.
"""
from __future__ import annotations

from typing import Any

# The board status of a step whose redo waits to be picked up by its mission.
WAITING_ITS_TURN = "assigned"


def waits_its_turn(db: Any, card: Any) -> None:
    """``card`` (a mission step's) says Assigned until its redo starts, unless its agent
    is a Claude Code agent, whose host would claim it."""
    from services.cli_ticket_lane import is_cli_agent

    if card is not None and not is_cli_agent(db, card.assigned_agent_id):
        card.status = WAITING_ITS_TURN


def card_runs_again(card: Any, *, why: str, by: str) -> None:
    """``card``'s result goes into its history with ``why`` and ``by``, and off its face,
    with its finish time and failure: its work runs again."""
    from api.board_tasks import keep_previous_run

    keep_previous_run(card, why=why, by=by)
    card.result = None
    card.completed_at = None
    card.error_message = None


__all__ = ["WAITING_ITS_TURN", "card_runs_again", "waits_its_turn"]
