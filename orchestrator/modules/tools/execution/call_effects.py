"""What a call did, where its name alone does not say it (F261, night 8).

A reply's claim is backed by an action that succeeded this turn (F108,
``action_claims``), matched by name. Two of the board's actions do different
things by their arguments:

- platform_update_task_status moves a card. To "done" it approves the card, to
  "cancelled" it cancels it, to "assigned" it sends it back to its agent, to
  "in_progress" it starts it. Night 8: asked to cancel #0422, Auto moved it to
  done, then said "Task #0422 has been moved to 'cancelled'". By name, the move
  backed the cancel.
- platform_update_task changes a card's fields, and sends it back to its agent
  only with ``send_back``. Night 8: Auto changed #0451's brief, said it had sent
  it back, and #0451 stayed in Review.

Night 9 (F309): platform_update_task with a status moves the card the way
platform_update_task_status does (``ticket_edit_moves``), and is recorded as that move.

So a successful call of either is recorded with what it did too, as
``<action>:<status>`` or ``<action>:send_back``. The families back a claim with
those (``platform_update_task_status:done`` approves a card).

Stdlib only.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

STATUS_MOVE = "update_task_status"
CARD_EDIT = "update_task"
SENT_BACK = "send_back"
_TRUE = ("true", "yes", "1")


def _is_set(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in _TRUE
    return value is True


def call_effects(action: str, params: Any) -> Tuple[str, ...]:
    """What ``action`` did with ``params``, beside its name: ``("<action>:done",)``
    for a card moved to done, ``("<action>:send_back",)`` for a card sent back
    with its edit; () when the name says it all."""
    if not isinstance(params, dict):
        return ()
    if action.endswith(STATUS_MOVE):
        status = str(params.get("status") or "").strip().lower()
        return (f"{action}:{status}",) if status else ()
    if action.endswith(CARD_EDIT) and _is_set(params.get(SENT_BACK)):
        return (f"{action}:{SENT_BACK}",)
    status = str(params.get("status") or "").strip().lower() if action.endswith(CARD_EDIT) else ""
    return (f"{action}_status:{status}",) if status else ()   # F309 (9): its status is the card's move


# F308 (night 9): what a mission's answer said of its steps: #0033 was made and approved
# with no check of each step while the reply said "Each step will pause for your approval".
STEPS_CHECKED, STEPS_UNCHECKED = "mission_steps_checked", "mission_steps_unchecked"


def result_effects(result: Any) -> Tuple[str, ...]:
    """What a call's answer says it left a mission's steps doing: waiting for the
    owner's check, or running on unchecked; () when it says nothing of it."""
    if not isinstance(result, dict) or not isinstance(result.get("checks_each_step"), bool):
        return ()
    return (STEPS_CHECKED,) if result["checks_each_step"] else (STEPS_UNCHECKED,)


def call_params(tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
    """The parameters the action itself received: platform_execute's ``params``
    (or the keys beside ``action`` when it sent none), else the call's own."""
    if tool_name != "platform_execute" or not isinstance(tool_args, dict):
        return tool_args if isinstance(tool_args, dict) else {}
    inner = tool_args.get("params")
    if isinstance(inner, dict) and inner:
        return inner
    return {k: v for k, v in tool_args.items() if k not in ("action", "name", "params")}


__all__ = ["STEPS_CHECKED", "STEPS_UNCHECKED", "call_effects", "call_params", "result_effects"]
