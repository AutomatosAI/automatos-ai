"""PRD-256 FX-004: an ask for the owner's click is waiting, never a failure.

An owner-only call from a person's chat (``owner_only``), or a confirmation-gated one, does
not run: it returns an ask (``requires_confirmation: True``) and, when the grant was issued,
the approval card the chat renders (``grant_id``). Night 12 read that ask as a refusal: the
receipt said "I tried to change the agent and it didn't go through", the model was handed
"Tool X failed: Waiting for the owner's click…" and told the owner the change failed.

This module is the one wording of an ask, read four ways: the receipt (``waiting``, with
``receipt_effect``), the model's tool result (``for_the_model``, wired into the tool router
by ``the_model_reads_the_card``), the tool-end line the chat shows (``tool_end_summary``) and
the account a blank answer becomes (``account_line``, read by ``turn_account.account_of``).
What the card asks (``ACT``) is set by the ask itself (``owner_only._ask``: "change an agent
'Scout' (agent #12)"); an ask that does not say is named from the call's name ("delete the
document").
"""
from __future__ import annotations

import functools
from typing import Any, Callable, Dict, Optional

from .call_effects import answers_in
from .turn_account import thing_of

# The ask's key for what its card asks: the verb and the subject, in the owner's words.
ACT = "act"
CARD_RAISED = "Card raised: {act}. Nothing changes until the owner clicks"
FOR_THE_MODEL = CARD_RAISED + "; say so in one line and do not retry the call."
TOOL_END = CARD_RAISED + "."
RECEIPT_EFFECT = "card raised: {act}"
# An ask whose card could not be issued (tool_grants' fail-safe floor) is still waiting, but
# nothing claims a card that is not there: the model keeps the ask's own words, still not "failed".
ASKED_FOR_THE_MODEL = "Not run, and not refused: {said} Say so in one line and do not retry the call."
ASKED_TOOL_END = "Asked for the owner's OK: {act}."
ASKED_EFFECT = "asked for your OK: {act}"
# The blank-answer account (turn_account, F264) speaks to the owner: the card waits for them.
ACCOUNT_LINE = "- Card raised: {act}. Nothing changes until you click."
ASKED_ACCOUNT_LINE = "- Asked for your OK: {act}. Nothing changes until you say so."

Envelope = Dict[str, Any]
Backfill = Callable[[Envelope, Dict[str, Any]], Envelope]


def the_ask(result: Any) -> Optional[Dict[str, Any]]:
    """The ask in a call's result (the executor's own, or inside the chat's envelope), else None."""
    for answer in answers_in(result):
        if answer.get("requires_confirmation") is True:
            return answer
    return None


def is_waiting(result: Any) -> bool:
    """Whether the call asked for the owner's click instead of running."""
    return the_ask(result) is not None


def has_a_card(ask: Dict[str, Any]) -> bool:
    """Whether the ask's grant was issued, so the chat renders its approval card."""
    grant = ask.get("grant_id")
    return isinstance(grant, int) and not isinstance(grant, bool)


def act_of(ask: Dict[str, Any], action: str = "", subject: str = "") -> str:
    """What the card asks: the ask's own words, else "<verb> the <thing>" from the call's name."""
    said = ask.get(ACT)
    if isinstance(said, str) and said.strip():
        return said.strip()
    name = str(ask.get("action") or action or "").lower().removeprefix("platform_")
    words = [word for word in name.split("_") if word]
    what = f"{words[0]} the {thing_of(name)}" if len(words) > 1 else (name or "run the call")
    return f"{what} {subject}".strip()


def for_the_model(result: Any, action: str = "") -> Optional[str]:
    """The model's tool result for an ask ("Card raised: …"); None for any other result."""
    ask = the_ask(result)
    if ask is None:
        return None
    if has_a_card(ask):
        return FOR_THE_MODEL.format(act=act_of(ask, action))
    said = str(ask.get("message") or "").strip() or f"{act_of(ask, action)} waits for the owner's OK."
    return ASKED_FOR_THE_MODEL.format(said=said)


def tool_end_summary(result: Any) -> Optional[str]:
    """The chat's tool-end line for an ask, in the model's words; None for any other result."""
    ask = the_ask(result)
    if ask is None:
        return None
    return (TOOL_END if has_a_card(ask) else ASKED_TOOL_END).format(act=act_of(ask))


def account_line(result: Any, action: str = "") -> Optional[str]:
    """P256-FIX-RVW-15: the blank-answer account's line for an ask ("- Card raised: …"); None
    for any other result, so the account never counts an ask as a change or a failure."""
    ask = the_ask(result)
    if ask is None:
        return None
    return (ACCOUNT_LINE if has_a_card(ask) else ASKED_ACCOUNT_LINE).format(act=act_of(ask, action))


def the_model_reads_the_card(backfill: Backfill) -> Backfill:
    """Wrap ``ToolRouter._maybe_add_error_envelope`` (every failed call's envelope passes
    through it with the executor's own result): an ask whose card is raised reaches the model
    as "Card raised: …", never "Tool X failed: …" (FX-004). Any other envelope is unchanged."""
    @functools.wraps(backfill)
    def wrapped(envelope: Envelope, raw_result: Dict[str, Any]) -> Envelope:
        said = for_the_model(raw_result)
        return backfill({**envelope, "llm_context": said} if said else envelope, raw_result)
    return wrapped


def the_results_flag(event: Dict[str, Any]) -> Dict[str, Any]:
    """FX-005: a tool-end event's ``success`` is its result's own. The loop marks a call that did
    not raise as a success; a refusal (``success: False``) or an ask is not one, so the screen
    never shows a green tick on it. Any other event passes unchanged."""
    result = event.get("result")
    if event.get("type") != "tool-end" or not isinstance(result, dict):
        return event
    ok = bool(result.get("success", True)) and not is_waiting(result)
    return {**event, "success": bool(event.get("success")) and ok}


def receipt_effect(result: Any, action: str, subject: str = "") -> str:
    """A waiting receipt's effect: "card raised: change an agent 'Scout'"."""
    ask = the_ask(result) or {}
    template = RECEIPT_EFFECT if has_a_card(ask) else ASKED_EFFECT
    return template.format(act=act_of(ask, action, "" if ask.get(ACT) else subject))


__all__ = ["ACCOUNT_LINE", "ACT", "FOR_THE_MODEL", "RECEIPT_EFFECT", "TOOL_END", "account_line", "act_of",
           "for_the_model", "has_a_card", "is_waiting", "receipt_effect", "the_ask", "the_model_reads_the_card", "the_results_flag",
           "tool_end_summary"]
