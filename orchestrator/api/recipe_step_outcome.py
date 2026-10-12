"""PRD-256 P256-FIX-T6 (F401): a playbook step's call that waits on the owner's click reads as waiting.

A step's Composio action rides the spine (api/recipe_executor ``_execute_step``). On a run Auto
started, the owner's-click gate holds a send (``owner_only.asks_before_a_send``, FX-011): nothing
runs and the spine answers "Card raised: send …". The step wrapped every unsuccessful spine result
as "Error executing <action>: …", so the run recorded 'Error executing
LINKEDIN_CREATE_LINKED_IN_POST: Card raised: send …', F131's rule failed the step on its last call
and the run read as failed. Nothing was posted and the grant was right: the same shape FX-004
fixed for the chat.

Now such a call is recorded as the card's own words with :data:`WAITING` set (``card_raised.WAITING``,
the flag the chat's tool-end carries since RVW-22): not a success (nothing ran), not a failure. A
step whose last call waits is not failed (``step_failure``), its summary says "(waiting)", the
step's record lists what waits (:data:`WAITING_ON`), and the run's output ends with what waits for
the owner's click. A real failure, a refusal included, keeps "Error executing …".
"""
from __future__ import annotations

import functools
from typing import Any, Awaitable, Callable, Dict, List, Optional

from modules.tools.discovery.agent_sends import CARD_RAISED as SEND_CARD_RAISED
from modules.tools.execution.card_raised import CARD_RAISED, FOR_THE_MODEL, WAITING

# How ``_execute_step`` records a Composio action the spine did not run.
FAILED_CALL = "Error executing {action}: "
# The words every raised card starts with ("Card raised: "), from the card's own template.
CARD_TEXT = CARD_RAISED.split("{", 1)[0]
# What a card tells the model after its subject (". Nothing … clicks; end your turn."): the run's
# output speaks to the owner instead.
_FOR_THE_MODEL_TAILS = tuple(text[text.rindex("}") + 1:] for text in (SEND_CARD_RAISED, FOR_THE_MODEL))
# A step's record: the cards its calls raised, in its calls' order.
WAITING_ON = "waiting_on"
WAITING_HEADER = "Waiting on your click:"
WAITING_LINE = "- Step {order}: {card}. Nothing happens until you click."

Step = Callable[..., Awaitable[Dict[str, Any]]]


def card_text(call: Dict[str, Any]) -> Optional[str]:
    """The card's words when this unsuccessful call raised one (with or without the step's
    "Error executing <action>: " in front), else None."""
    if call.get("success") is not False:
        return None
    said = str(call.get("result") or "").removeprefix(FAILED_CALL.format(action=call.get("action") or ""))
    return said if said.startswith(CARD_TEXT) else None


def waits_for_the_click(call: Any) -> Any:
    """The call as the step records it: a raised card is its own words, waiting."""
    if not isinstance(call, dict) or call.get(WAITING) is True:
        return call
    said = card_text(call)
    return {**call, "result": said, WAITING: True} if said is not None else call


def records_the_wait(execute: Step) -> Step:
    """Wrap ``_execute_step``: every call of its result that raised a card is recorded as waiting.
    The step's own loop (over the length limit) is not touched."""
    @functools.wraps(execute)
    async def wrapped(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        result = await execute(*args, **kwargs)
        execution = result.get("execution") if isinstance(result, dict) else None
        calls = execution.get("tool_calls") if isinstance(execution, dict) else None
        if not isinstance(calls, list) or not calls:
            return result
        return {**result, "execution": {**execution, "tool_calls": [waits_for_the_click(c) for c in calls]}}
    return wrapped


def is_waiting(call: Dict[str, Any]) -> bool:
    """Whether the step recorded this call as waiting on the owner's click."""
    return call.get(WAITING) is True


def normalize_tool_calls(raw_calls: Any) -> List[Dict[str, Any]]:
    """A step's tool calls as the run keeps them; a waiting call keeps its flag."""
    if not isinstance(raw_calls, list):
        return []
    return [_normalized(call) for call in raw_calls if isinstance(call, dict)]


def _normalized(call: Dict[str, Any]) -> Dict[str, Any]:
    record = {
        "action": call.get("action") or call.get("function", {}).get("name", "unknown"),
        "params": call.get("params") or call.get("function", {}).get("arguments", {}),
        "result": call.get("result") or call.get("content", {}),
        "duration_ms": call.get("duration_ms", 0),
        "success": call.get("success"),
    }
    return {**record, WAITING: True} if is_waiting(call) else record


def call_summary(call: Dict[str, Any]) -> str:
    """"<action> (success|error|waiting)" for the step's record."""
    action = call.get("action", "unknown")
    if is_waiting(call):
        return f"{action} ({WAITING})"
    # F137: the call's own success flag. B22's "Tool composio_execute failed: …"
    # holds no "error", so the text guess showed four failed drafts as (success).
    worked = call.get("success")
    if worked is None:  # a record from before the flag: the old guess
        worked = "error" not in str(call.get("result", "")).lower()[:100]
    return f"{action} ({'success' if worked else 'error'})"


def waiting_on(tool_calls: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """``{waiting_on: [card, …]}`` for a step's record when its calls raised cards, else {}."""
    cards = [str(call.get("result") or "") for call in tool_calls if is_waiting(call)]
    return {WAITING_ON: cards} if cards else {}


def with_the_wait(final_output: Optional[str], step_results: List[Dict[str, Any]]) -> Optional[str]:
    """The run's output, then every card its steps left waiting on the owner's click."""
    lines = [WAITING_LINE.format(order=step.get("order"), card=_for_the_owner(card))
             for step in step_results for card in step.get(WAITING_ON) or []]
    if not lines:
        return final_output
    said = "\n".join([WAITING_HEADER, *lines])
    return f"{final_output}\n\n{said}" if final_output else said


def _for_the_owner(card: str) -> str:
    for tail in _FOR_THE_MODEL_TAILS:
        if card.endswith(tail):
            return card[: -len(tail)]
    return card.rstrip(".")


__all__ = ["CARD_TEXT", "FAILED_CALL", "WAITING", "WAITING_ON", "call_summary", "card_text", "is_waiting",
           "normalize_tool_calls", "records_the_wait", "waiting_on", "waits_for_the_click", "with_the_wait"]
