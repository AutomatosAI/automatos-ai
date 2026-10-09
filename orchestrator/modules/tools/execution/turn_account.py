"""F264 (night 8): when Auto's reply comes back empty, the owner is told what was done.

About 15 of night 8's replies ended "I apologize, but I encountered an issue
generating a response. Please try again.", most after the work had happened:
#0231 approved, #0205 sent back, #0220 made, missions #0282 and #0344 made. The
model's answer came back empty (most often the retry after F108's nudge), the
chat forced one more answer with no tools, that was empty too, and the
apology replaced it. On the board the work was done; in the chat it read as a
failure to retry, which would have done it twice.

Now a blank answer (no text, no call) in Auto's own turn becomes a plain account
of the turn's calls, from their results: each change that went through, by card
number and where it is now when the result says so; each card that waits for the
owner's click (an ask is neither a change nor a failure, P256-FIX-RVW-15); how many
did not go through; and, when nothing changed, that nothing did. Refusals are the model's to read and
are never quoted (they name calls on purpose). Agent runs keep their own
empty-answer handling. Stdlib only, beside the Composio slug reader (composio_action) and
the action registry (P256-FIX-RVW-36: a registered action's permission_level says whether it
only reads, so platform_load_skill is never a change).
"""
from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

Outcome = Tuple[str, Dict[str, Any], Any]          # (action, its params, its result)

ACCOUNT_HEADER = "My reply didn't come through, so here is what was done this turn:"
NOT_THROUGH = "- {count} other request{s} didn't go through, so {they} changed nothing."
NOTHING_CHANGED = "My reply didn't come through, and nothing was changed: {why} Please ask again."
WHY_NO_CALL = "no action ran this turn."
WHY_ONLY_READS = "I only looked things up."
WHY_ALL_FAILED = "what I tried didn't go through."
# Only asks for the owner's click (card_raised): nothing changed yet, and asking again would not help.
WAITING_FOR_YOU = "My reply didn't come through, and nothing has changed yet: {they} for your click."
# Calls that only look: they change nothing the owner would want told. A registered action's
# permission_level says so first; these stems read a call the registry does not hold.
READ_LEVEL = "read"
_READS = ("_list", "_get_", "search", "browse", "board_", "read", "grep", "query", "summary",
          "snapshot", "history", "fetch", "find", "_view", "recommend", "check_", "wait_for")
_WHERE = {"done": "Done", "cancelled": "Cancelled", "assigned": "with its agent", "review": "in Review",
          "in_progress": "in progress", "inbox": "in the Inbox", "blocked": "blocked",
          "awaiting_approval": "waiting for your approval", "running": "running", "paused": "paused",
          "completed": "finished"}
_PAST = {"create": "made", "update": "changed", "delete": "deleted", "cancel": "cancelled", "assign": "assigned",
         "unassign": "taken off", "approve": "approved", "reject": "sent back", "schedule": "scheduled",
         "execute": "started", "run": "started", "trigger": "started", "send": "sent", "submit": "submitted",
         "install": "installed", "pause": "paused", "resume": "resumed", "store": "saved", "upload": "uploaded",
         "add": "added", "remove": "removed", "publish": "published", "set": "set", "configure": "set up"}
_NOUNS = {"task": "card", "tasks": "cards", "recipe": "playbook"}
# A message is only passed on when it is plain words: no names of calls, no ids.
_INTERNAL = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b|platform|\b[0-9a-f]{8}-[0-9a-f]{4}-|exec-[0-9a-f]{6}", re.I)
_MESSAGE_CHARS = 160


def _payload(result: Any) -> Dict[str, Any]:
    """The action's own result: the chat wraps it as ``raw_result``."""
    if not isinstance(result, dict):
        return {}
    raw = result.get("raw_result")
    return raw if isinstance(raw, dict) else result


def _failed(result: Any) -> bool:
    return isinstance(result, dict) and (result.get("success") is False or result.get("successful") is False)


def _registered_level(action: str) -> Optional[str]:
    """The registered action's permission_level ("read", "write", "destructive"), None when unregistered."""
    from modules.tools.discovery.action_registry import get_action_registry

    definition = get_action_registry().get(action)
    return getattr(definition, "permission_level", None) if definition is not None else None


def is_read(action: str) -> bool:
    """A call that only looks; a Composio action by its slug's whole words (P256-FIX-RVW-5); a
    registered action by its permission_level (P256-FIX-RVW-36: platform_load_skill and
    platform_fleet_status read, platform_checkpoint_thread writes); any other by its name's stems."""
    from .composio_action import is_slug, slug_reads

    if is_slug(action):
        return slug_reads(action)
    level = _registered_level(action)
    if level is not None:
        return level == READ_LEVEL
    name = action.lower()
    return any(stem in name for stem in _READS)


def thing_of(action: str) -> str:
    """"card", "mission steps": what a call's name says it is about, in plain words."""
    words = action.lower().removeprefix("platform_").split("_")
    return " ".join(_NOUNS.get(w, w) for w in words[1:]) or "change"


def what_it_did(action: str) -> str:
    """"Mission made", "Playbook scheduled": what a call did, from its name, in plain words."""
    verb = action.lower().removeprefix("platform_").split("_")[0]
    return f"{thing_of(action).capitalize()} {_PAST.get(verb, 'done')}"


def _line(action: str, payload: Dict[str, Any]) -> str:
    """One change that went through: its card by number and where it is now, or the
    action's own plain message, or what the call's name says it did."""
    number, title = payload.get("number"), payload.get("title")
    where = payload.get("status") or payload.get("state")
    named = f"{number} ({title})" if number and isinstance(title, str) and title.strip() else number
    agent = payload.get("assigned_agent")
    if named and where == "assigned" and isinstance(agent, str) and agent:
        return f"- {named}: with {agent}"
    if named and isinstance(where, str):
        return f"- {named}: {_WHERE.get(where, where.replace('_', ' '))}"
    message = payload.get("message")
    if isinstance(message, str) and message.strip() and not _INTERNAL.search(message):
        return f"- {message.strip()[:_MESSAGE_CHARS]}"
    return f"- {what_it_did(action)}" + (f": {number}" if number else "")


def _not_through(failed: int) -> str:
    return NOT_THROUGH.format(count=failed, s="" if failed == 1 else "s", they="it" if failed == 1 else "they")


def _waiting(asks: Sequence[str], failed: int) -> str:
    """Nothing changed, and the turn's asks wait for the owner's click: never "Please ask again"."""
    cards = list(dict.fromkeys(asks))
    lines = [WAITING_FOR_YOU.format(they="it waits" if len(cards) == 1 else "they wait"), *cards]
    return "\n".join([*lines, _not_through(failed)] if failed else lines)


def account_of(outcomes: Sequence[Outcome]) -> str:
    """What the owner is told when the answer came back empty: the changes the
    turn's calls made and the cards that wait for the owner's click, or plainly
    that nothing changed."""
    from .card_raised import account_line

    asked = [account_line(result, action) for action, _params, result in outcomes]
    asks = [line for line in asked if line]
    settled = [outcome for outcome, line in zip(outcomes, asked) if line is None]
    changes = [_line(action, _payload(result)) for action, _params, result in settled
               if not _failed(result) and not is_read(action)]
    failed = sum(1 for _action, _params, result in settled if _failed(result))
    if not changes and asks:
        return _waiting(asks, failed)
    if not changes:
        why = WHY_ALL_FAILED if failed else (WHY_ONLY_READS if outcomes else WHY_NO_CALL)
        return NOTHING_CHANGED.format(why=why)
    lines = [ACCOUNT_HEADER, *dict.fromkeys(changes), *dict.fromkeys(asks)]
    return "\n".join([*lines, _not_through(failed)] if failed else lines)


def is_blank(response: Any) -> bool:
    """No text worth showing and no call: what the chat turned into its apology."""
    if response is None:
        return True
    if getattr(response, "tool_calls", None):
        return False
    return not str(getattr(response, "content", None) or "").strip()


def with_account(response: Any, outcomes: Sequence[Outcome]) -> Any:
    """A copy of ``response`` whose content is the turn's account (never mutated).
    It was never streamed, so the chat shows it."""
    kept = vars(response) if response is not None and hasattr(response, "__dict__") else {}
    return SimpleNamespace(**{"usage": None, "finish_reason": "stop", **kept, "content": account_of(outcomes),
                              "tool_calls": None, "streamed": False})


LLMCall = Callable[[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]], Awaitable[Any]]


def auto_speaks(promises: Optional[bool]) -> bool:
    """Auto's own turn: the executor says so (``promises``), or else the turn's lane."""
    if promises is not None:
        return promises
    from core.llm.usage_context import LANE_CHAT, current_usage_scope

    # the chat service books the whole turn to the chat lane; agent runs book theirs
    return current_usage_scope().get("request_type") == LANE_CHAT


def said_or_accounted(llm: LLMCall, outcomes: Callable[[], Sequence[Outcome]],
                      promises: Optional[bool] = None) -> LLMCall:
    """``llm``, whose blank answer in Auto's own turn is replaced by the account of
    ``outcomes()``, the turn's calls so far. An agent run's blank answer is left as
    it came (its run has its own handling)."""
    async def call(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]]) -> Any:
        response = await llm(messages, tools)
        if not is_blank(response) or not auto_speaks(promises):
            return response
        return with_account(response, outcomes())
    return call


__all__ = ["account_of", "auto_speaks", "is_blank", "is_read", "said_or_accounted", "thing_of", "what_it_did",
           "with_account"]
