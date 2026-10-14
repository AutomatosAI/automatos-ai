"""PRD-256 FX-007: the receipts rule, as the claim tests read it.

The regex claim families are gone (D10); a reply's claim is matched to the turn's done writes
by the receipts (``consumers/chatbot/claims_backed.py``). The tests that pinned a family's
sentence read it the same two ways the turn does:

- ``nudged``: what the tool loop's claim check (F108) says was claimed, from the calls so far
  (``receipts.unbacked_claim``), or None; in Auto's chat turn unless ``promises`` says it is an
  agent's run (False) or leaves it to the turn's lane (None), P256-FIX-RVW-7. Since P256-FIX-N15-1
  the check nudges nothing: it replaces the claim's sentence with the not-done line in place;
- ``line``: the not-done line the reply carries in place of the claims it took out
  (``claim_removed.without_unbacked_claims``), or None. Since P256-FIX-N15-2 that is the line's
  only place: nothing puts it above the answer;
- ``unbacked``: the verbs of every claim the receipts rule reads as unbacked, verbs in no family
  included (``claims_backed.unbacked_claims``). A verb in no family ("drafted", "rejected") is
  read but never taken out, so since N15-2 it draws no line.

A call is ``(action, params, result)`` or an action's name alone (it went through).
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from consumers.chatbot.claim_removed import not_done_in_place, without_unbacked_claims
from consumers.chatbot.claims_backed import says_not_done, unbacked_claims
from consumers.chatbot.receipts import DONE, READ, WAITING, WRITE, build_receipts, unbacked_claim, unbacked_sentences
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
    """What F108's check says the reply claimed with no done write of its kind behind it, else None."""
    return unbacked_claim(text, tracker_of(calls).outcomes, promises=promises)


def line(text: str, *calls: Call, promises: Optional[bool] = True) -> Optional[str]:
    """The not-done line the reply ``text`` carries in place of its unbacked claims, else None: taken
    from the reply as the owner gets it (``without_unbacked_claims``), so the replacement is checked too."""
    outcomes: List[Tuple[str, Dict[str, Any], Any]] = list(tracker_of(calls).outcomes)
    removed = without_unbacked_claims(text, outcomes, promises)
    if removed is None:
        return None
    found = unbacked_sentences(text, outcomes, promises)
    said = not_done_in_place((verb for verb, _sentence in found), outcomes)
    assert says_not_done(removed.text) and said in removed.text, removed.text   # the line is in the reply
    return said


def unbacked(text: str, *calls: Call) -> List[str]:
    """The verbs of the claims of ``text`` that no done write of the calls backs, by the receipts rule."""
    receipts = build_receipts(tracker_of(calls))
    done = [r for r in receipts if r["kind"] == WRITE and r["status"] == DONE]
    reads = [r for r in receipts if r["kind"] == READ and r["status"] == DONE]
    waiting = any(r["status"] == WAITING for r in receipts)
    return [verb for verb, _known in unbacked_claims(text, done, waiting=waiting, reads=reads)]


__all__ = ["OK", "call", "line", "nudged", "tracker_of", "unbacked"]
