"""PRD-256 FX-007: the receipts rule, as the claim tests read it.

The regex claim families are gone (D10); a reply's claim is matched to the turn's done writes
by the receipts (``consumers/chatbot/claims_backed.py``). The tests that pinned a family's
sentence read it the same two ways the turn does:

- ``nudged``: what the tool loop's one nudge (F108) says was claimed, from the calls so far
  (``receipts.unbacked_claim``), or None; in Auto's chat turn unless ``promises`` says it is an
  agent's run (False) or leaves it to the turn's lane (None), P256-FIX-RVW-7;
- ``line``: the not-done line above the answer (``receipts.honesty_lines``), or None.

A call is ``(action, params, result)`` or an action's name alone (it went through).
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from consumers.chatbot.claims_backed import is_not_done_line
from consumers.chatbot.receipts import build_receipts, honesty_lines, unbacked_claim
from modules.tools.execution.tool_execution_tracker import ToolExecutionTracker

OK = {"success": True}
Call = Union[str, Tuple[str, Dict[str, Any], Any]]


def call(action: str, params: Optional[Dict[str, Any]] = None, result: Any = None) -> Tuple[str, Dict[str, Any], Any]:
    """One platform call through the dispatcher, as the loop records it."""
    return ("platform_execute", {"action": action, "params": params or {}}, OK if result is None else result)


def tracker_of(calls: Iterable[Call]) -> ToolExecutionTracker:
    tracker = ToolExecutionTracker()
    for one in calls:
        tool, args, result = call(one) if isinstance(one, str) else one
        tracker.record_outcome(tool, args, result)
    return tracker


def nudged(text: str, *calls: Call, promises: Optional[bool] = True) -> Optional[str]:
    """What F108's nudge says the reply claimed with no done write of its kind behind it, else None."""
    return unbacked_claim(text, tracker_of(calls).outcomes, promises=promises)


def line(text: str, *calls: Call) -> Optional[str]:
    """The not-done line the receipts put above ``text``, else None."""
    lines: List[str] = honesty_lines(build_receipts(tracker_of(calls)), text)
    return next((said for said in lines if is_not_done_line(said)), None)


__all__ = ["OK", "call", "line", "nudged", "tracker_of"]
