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
document"). P256-FIX-N17-3 (F439): an ask that met a card raised on an earlier turn of the chat
(``already_waiting``) is never "Card raised": each reading says it still waits, raised at its time.

PRD-263 US-018: the routing graph does not read an ask as a failure either. The tool router
recorded every call's outcome, so each ask became a failure signal for its tool, which the
GraphRouter then penalised (``fails_for_intent``). ``holds_the_signal`` and
``signals_unless_it_asked`` hold a router call's signals until its answer is known and drop
them when it asked; the router's body (over the length limit) is not touched.
"""
from __future__ import annotations

import functools
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, Iterator, List, Optional

from .call_effects import answers_in
from .stale_cards import ALREADY_WAITING
from .turn_account import thing_of

# The ask's key for what its card asks: the verb and the subject, in the owner's words.
ACT = "act"
CARD_RAISED = "Card raised: {act}. Nothing changes until the owner clicks"
# P256-FIX-N17-3 (F439): night 16 told the owner to "click … on the board"; the card is in Approvals.
FOR_THE_MODEL = CARD_RAISED + ("; say so in one line, say its approval card is under Approvals (and beside this "
                               "reply in a chat), not a ticket on the board, and do not retry the call.")
TOOL_END = CARD_RAISED + "."
RECEIPT_EFFECT = "card raised: {act}"
# P256-FIX-N17-3: a chat turn that met its own card from an earlier turn (``ALREADY_WAITING`` on the ask)
# raised nothing: each reading names that card and when it was raised ("at 03:04 on 13 Oct").
ALREADY_FOR_THE_MODEL = ("Not run: this exact call already waits on its approval card, raised {at}: {act}. No new "
                         "card was raised: tell the owner it is that card (beside the reply that raised it and under "
                         "Approvals, not a ticket on the board), and do not retry the call.")
ALREADY_TOOL_END = "Still waiting on its card, raised {at}: {act}."
ALREADY_EFFECT = "still waiting on its card (raised {at}): {act}"
ALREADY_ACCOUNT_LINE = "- Still waiting on its card, raised {at}: {act}. Nothing changes until you click."
# An ask whose card could not be issued (tool_grants' fail-safe floor) is still waiting, but
# nothing claims a card that is not there: the model keeps the ask's own words, still not "failed".
ASKED_FOR_THE_MODEL = "Not run, and not refused: {said} Say so in one line and do not retry the call."
ASKED_TOOL_END = "Asked for the owner's OK: {act}."
ASKED_EFFECT = "asked for your OK: {act}"
# The blank-answer account (turn_account, F264) speaks to the owner: the card waits for them.
ACCOUNT_LINE = "- Card raised: {act}. Nothing changes until you click."
ASKED_ACCOUNT_LINE = "- Asked for your OK: {act}. Nothing changes until you say so."
# P256-FIX-RVW-22: an ask's tool-end event and frame carry this beside ``success: False``, so the
# activity trail draws it as waiting (an hourglass and the card's words), never "<tool> failed".
WAITING = "waiting"
# The flagged tool-end event the loop is handing to its callback, while it does (``emit_flagged``).
_STREAMING: ContextVar[Optional[Dict[str, Any]]] = ContextVar("card_raised_streaming", default=None)
# The routing signals the router call in progress has recorded, held until its answer is known.
_HELD_SIGNALS: ContextVar[Optional[List[Callable[[], None]]]] = ContextVar("card_raised_held_signals",
                                                                          default=None)

Envelope = Dict[str, Any]
Backfill = Callable[[Envelope, Dict[str, Any]], Envelope]
Routed = Callable[..., Awaitable[Envelope]]


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


def raised_at(ask: Dict[str, Any]) -> Optional[str]:
    """When the card this ask meets again was raised ("at 03:04 on 13 Oct"), when the ask is
    marked ``already_waiting``; None for a card this call raised."""
    from modules.tools.discovery.my_actions_reads import at_said

    waiting = ask.get(ALREADY_WAITING)
    since = waiting.get("since") if isinstance(waiting, dict) else None
    try:
        return at_said(datetime.fromisoformat(str(since))) if since else None
    except ValueError:
        return None


def _card_said(ask: Dict[str, Any], raised: str, already: str, act: str) -> str:
    """The card's line: the one met again with its time (``already``), else the one raised."""
    at = raised_at(ask)
    return already.format(at=at, act=act) if at else raised.format(act=act)


def for_the_model(result: Any, action: str = "") -> Optional[str]:
    """The model's tool result for an ask ("Card raised: …"); None for any other result."""
    ask = the_ask(result)
    if ask is None:
        return None
    if has_a_card(ask):
        return _card_said(ask, FOR_THE_MODEL, ALREADY_FOR_THE_MODEL, act_of(ask, action))
    said = str(ask.get("message") or "").strip() or f"{act_of(ask, action)} waits for the owner's OK."
    return ASKED_FOR_THE_MODEL.format(said=said)


def tool_end_summary(result: Any) -> Optional[str]:
    """The chat's tool-end line for an ask, in the model's words; None for any other result."""
    ask = the_ask(result)
    if ask is None:
        return None
    if has_a_card(ask):
        return _card_said(ask, TOOL_END, ALREADY_TOOL_END, act_of(ask))
    return ASKED_TOOL_END.format(act=act_of(ask))


def account_line(result: Any, action: str = "") -> Optional[str]:
    """P256-FIX-RVW-15: the blank-answer account's line for an ask ("- Card raised: …"); None
    for any other result, so the account never counts an ask as a change or a failure."""
    ask = the_ask(result)
    if ask is None:
        return None
    if has_a_card(ask):
        return _card_said(ask, ACCOUNT_LINE, ALREADY_ACCOUNT_LINE, act_of(ask, action))
    return ASKED_ACCOUNT_LINE.format(act=act_of(ask, action))


def the_model_reads_the_card(backfill: Backfill) -> Backfill:
    """Wrap ``ToolRouter._maybe_add_error_envelope`` (every failed call's envelope passes
    through it with the executor's own result): an ask whose card is raised reaches the model
    as "Card raised: …", never "Tool X failed: …" (FX-004). Any other envelope is unchanged."""
    @functools.wraps(backfill)
    def wrapped(envelope: Envelope, raw_result: Dict[str, Any]) -> Envelope:
        said = for_the_model(raw_result)
        return backfill({**envelope, "llm_context": said} if said else envelope, raw_result)
    return wrapped


def asked_instead(envelope: Any) -> bool:
    """Whether a router envelope is a call that did not run because it asked for the owner's
    click (its own result, ``raw_result``, is an ask). A call that succeeded never is."""
    return isinstance(envelope, dict) and not envelope.get("success") and is_waiting(envelope.get("raw_result"))


def holds_the_signal(record: Callable[..., None]) -> Callable[..., None]:
    """Wrap ``ToolRouter._record_tool_signal`` (PRD-263 US-018): inside a router call the
    signal waits for the call's answer (``signals_unless_it_asked``); outside one it is
    recorded at once, as before."""
    @functools.wraps(record)
    def wrapped(*args: Any, **kwargs: Any) -> None:
        held = _HELD_SIGNALS.get()
        if held is None:
            return record(*args, **kwargs)
        held.append(functools.partial(record, *args, **kwargs))
        return None
    return wrapped


def signals_unless_it_asked(execute_and_format: Routed) -> Routed:
    """Wrap ``ToolRouter.execute_and_format`` (PRD-263 US-018): the routing signals its call
    records go to the recorder when it returns, unless the call asked for the owner's click
    instead of running. An ask is not a failure, so no tool is penalised for waiting on the
    owner. Each call holds its own signals, so a nested call's never mix with its caller's."""
    @functools.wraps(execute_and_format)
    async def wrapped(*args: Any, **kwargs: Any) -> Envelope:
        held: List[Callable[[], None]] = []
        token = _HELD_SIGNALS.set(held)
        try:
            envelope = await execute_and_format(*args, **kwargs)
        finally:
            _HELD_SIGNALS.reset(token)
        if not asked_instead(envelope):
            for send in held:
                send()
        return envelope
    return wrapped


def the_results_flag(event: Dict[str, Any]) -> Dict[str, Any]:
    """FX-005: a tool-end event's ``success`` is its result's own. The loop marks a call that did
    not raise as a success; a refusal (``success: False``) or an ask is not one, so the screen
    never shows a green tick on it. Any other event passes unchanged."""
    result = event.get("result")
    if event.get("type") != "tool-end" or not isinstance(result, dict):
        return event
    waiting = is_waiting(result)
    ok = bool(result.get("success", True)) and not waiting
    flagged = {**event, "success": bool(event.get("success")) and ok}
    return {**flagged, WAITING: True} if waiting else flagged


@contextmanager
def _streaming(event: Dict[str, Any]) -> Iterator[None]:
    token = _STREAMING.set(event)
    try:
        yield
    finally:
        _STREAMING.reset(token)


async def emit_flagged(cb: Callable[[Dict[str, Any]], Awaitable[None]], event: Dict[str, Any]) -> None:
    """The tool loop's emit (P256-FIX-RVW-22): the event goes out with its result's own flag, and
    while the callback formats it, ``streams_the_wait`` can read whether that call is waiting."""
    flagged = the_results_flag(event)
    with _streaming(flagged):
        await cb(flagged)


def streams_the_wait(tool_end: Callable[..., str]) -> Callable[..., str]:
    """Wrap ``StreamingHandler.format_aisdk_tool_end``: the frame of the call the loop is emitting
    carries ``waiting`` when its event does (an ask), whichever caller formats it; the chat's
    callback (inside ``_stream_tool_loop``, over the length limit) is not touched. Every other
    frame (a refusal, a done call, a frame formatted outside the loop) is unchanged."""
    @functools.wraps(tool_end)
    def wrapped(handler: Any, *args: Any, waiting: bool = False, **kwargs: Any) -> str:
        event = _STREAMING.get() or {}
        call_id = kwargs.get("tool_call_id", args[0] if args else None)
        asked = event.get(WAITING) is True and event.get("tool_call_id") == call_id
        return tool_end(handler, *args, waiting=bool(waiting or asked), **kwargs)
    return wrapped


def receipt_effect(result: Any, action: str, subject: str = "") -> str:
    """A waiting receipt's effect: "card raised: change an agent 'Scout'"."""
    ask = the_ask(result) or {}
    act = act_of(ask, action, "" if ask.get(ACT) else subject)
    return _card_said(ask, RECEIPT_EFFECT, ALREADY_EFFECT, act) if has_a_card(ask) else ASKED_EFFECT.format(act=act)


def already_waiting(result: Any) -> bool:
    """Whether the call met its card from an earlier turn again (nothing was raised this turn)."""
    ask = the_ask(result)
    return ask is not None and has_a_card(ask) and raised_at(ask) is not None


__all__ = ["ACCOUNT_LINE", "ACT", "ALREADY_FOR_THE_MODEL", "ALREADY_WAITING", "FOR_THE_MODEL", "RECEIPT_EFFECT",
           "TOOL_END", "WAITING", "account_line", "act_of", "already_waiting", "asked_instead", "emit_flagged",
           "for_the_model", "has_a_card", "holds_the_signal", "is_waiting", "raised_at", "receipt_effect",
           "signals_unless_it_asked", "streams_the_wait", "the_ask", "the_model_reads_the_card", "the_results_flag",
           "tool_end_summary"]
