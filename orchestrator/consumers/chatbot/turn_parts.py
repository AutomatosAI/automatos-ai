"""PRD-268 US-C12c (§6.3 'In the thread', §6.6 'Storage', K12, K13): the turn's parts.

The reply a chat turn saves carries, beside its text, what the turn left the person to act on. The
parts are built once the tool loop has ended:

* ``{type: 'reply_chips@1', replies}``: the chips the loop's final response carries (US-C12b), held
  again to the tool's own check (US-C12a). There are none on a turn that called
  ``platform_get_brief`` (never on the brief);
* ``{type: 'follow_up@1', follow_up_id, at}``: one for each ``platform_schedule_follow_up`` that
  succeeded this turn, called directly or as ``platform_execute``'s action, and was not cancelled
  later in it. It is read from the turn's tracker: the row's id and its UTC time, never its reason.

They are message parts, never ``Message.source``, which is NULL for every in-turn message (K12).
The receipts decorators carry them (``consumers/chatbot/receipts.py``):
* the turn starts with none, and the loop's final chunk sets them;
* they go out once, on a ``tool-data`` frame under ``turn_parts`` beside the receipts frames, and
  never to a public widget visitor;
* the saved parts end with them, after the text part.

Nothing else reads them. The model's history, previews and the voice read text only, and a client
that does not know a part's type drops it.

The text, once. When a round's reply chips end the run (US-C12b: no model call after it), that
round is the answer. Its text streamed live as the round's and stays on the screen:
* its narration frame is held until the loop's answer is known (``on_screen``);
* the answer is marked streamed, so the turn sends no tail;
* ``narration.split_reply`` leaves the round out of the narration.
``is_its_round`` is the one test all three read.
"""
from __future__ import annotations

import copy
import dataclasses
import logging
from datetime import datetime, timezone
from typing import Any, Dict, FrozenSet, Optional, Sequence, Tuple

from modules.tools.execution.params_text import json_container
from modules.tools.execution.reply_chips_turn import (
    OFFER_REPLIES, REPLY_CHIPS, action_answer, brief_in, called_action,
)

logger = logging.getLogger(__name__)

Part = Dict[str, Any]

REPLY_CHIPS_PART = "reply_chips@1"
FOLLOW_UP_PART = "follow_up@1"
LIVE_KEY = "turn_parts"
# follow_up_context's SCHEDULE_FOLLOW_UP and CANCEL_FOLLOW_UP, written here so the chat's seams stay
# light to import (the test pins them to the tool's), as reply_chips_turn writes the names it reads.
SCHEDULE_FOLLOW_UP = "platform_schedule_follow_up"
CANCEL_FOLLOW_UP = "platform_cancel_follow_up"
# LLMResponse.streamed: the response's text went out live, so the turn sends no tail.
STREAMED = "streamed"

__all__ = ["CANCEL_FOLLOW_UP", "FOLLOW_UP_PART", "LIVE_KEY", "REPLY_CHIPS_PART", "SCHEDULE_FOLLOW_UP", "as_streamed",
           "chips_part",
           "follow_up_part", "is_its_round", "offers_replies", "turn_parts", "turn_parts_frames"]


def chips_part(final: Any, outcomes: Sequence[Any]) -> Optional[Part]:
    """The ``reply_chips@1`` part of the chips ``final`` carries, held to the tool's own check. None
    when it carries none, when they fail the check, or when the turn called the brief."""
    chips = getattr(final, REPLY_CHIPS, None)
    if not chips or brief_in(outcomes):
        return None
    from modules.tools.discovery.handlers_replies import replies_refusal

    shown, refusal = replies_refusal(list(chips) if isinstance(chips, (list, tuple)) else chips)
    if refusal is not None:
        logger.warning("[turn-parts] the answer's chips fail the tool's check: the reply is saved without them")
        return None
    return {"type": REPLY_CHIPS_PART, "replies": list(shown)}


def _utc_time(value: Any) -> Optional[str]:
    """``value`` as the ISO time it names, in UTC, when it is one with its offset; None otherwise."""
    if not isinstance(value, str):
        return None
    try:
        at = datetime.fromisoformat(value)
    except ValueError:
        return None
    return at.astimezone(timezone.utc).isoformat() if at.tzinfo is not None else None


def _row_id(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def follow_up_part(action: Any, result: Any) -> Optional[Part]:
    """The ``follow_up@1`` part of one call the tracker recorded: a ``platform_schedule_follow_up``
    that succeeded, with the row id and the time its answer gives. None for any other call, for a
    refused one, and for an answer whose id or time is not one."""
    if str(action).strip() != SCHEDULE_FOLLOW_UP:
        return None
    answer = action_answer(result)
    if answer is None:
        return None
    follow_up_id, at = answer.get("follow_up_id"), _utc_time(answer.get("at"))
    if not _row_id(follow_up_id) or at is None:
        logger.warning("[turn-parts] a follow-up's answer has no row id or no time: the reply is saved without its line")
        return None
    return {"type": FOLLOW_UP_PART, "follow_up_id": follow_up_id, "at": at}


def _cancelled(outcomes: Sequence[Any]) -> FrozenSet[int]:
    """The follow-ups a ``platform_cancel_follow_up`` of the turn cancelled (its answer's ids)."""
    answers = [action_answer(result) for action, _params, result in outcomes if str(action).strip() == CANCEL_FOLLOW_UP]
    return frozenset(answer["follow_up_id"] for answer in answers
                     if answer is not None and _row_id(answer.get("follow_up_id")))


def turn_parts(final: Any, tracker: Any) -> Tuple[Part, ...]:
    """The parts the turn's saved reply carries after its text: its reply chips, then one for each
    follow-up it scheduled and did not cancel, once each, in order. ``final`` is the loop's answer
    and ``tracker`` its ToolExecutionTracker (None when no loop ran). A turn with neither gets
    ``()``, so its parts are exactly as before."""
    outcomes = list(getattr(tracker, "outcomes", None) or ())
    chips, gone = chips_part(final, outcomes), _cancelled(outcomes)
    lines = [part for part in (follow_up_part(action, result) for action, _params, result in outcomes)
             if part and part["follow_up_id"] not in gone]
    follow_ups = {part["follow_up_id"]: part for part in lines}  # once each, in the order first scheduled
    return (*((chips,) if chips else ()), *follow_ups.values())


def turn_parts_frames(handler: Any, parts: Sequence[Part]) -> Tuple[str, ...]:
    """The turn's parts on one ``tool-data`` frame under ``LIVE_KEY``. The web chat's data callback
    reads it, as it reads the receipts. A turn with no parts gets no frame."""
    return (handler.format_aisdk_tool_data({LIVE_KEY: [dict(part) for part in parts]}),) if parts else ()


# ── the text, once ──────────────────────────────────────────────────────────

def offers_replies(response: Any) -> bool:
    """Whether ``response`` calls ``platform_offer_replies``, directly or as ``platform_execute``'s
    action. The run may end on such a round's text (US-C12b), so its narration waits for the
    loop's answer."""
    for call in getattr(response, "tool_calls", None) or ():
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict):
            continue
        if called_action(str(function.get("name") or ""), json_container(function.get("arguments"))) == OFFER_REPLIES:
            return True
    return False


def is_its_round(answer: Any, round_text: str) -> bool:
    """Whether ``answer`` is a round's own text, the run having ended on it (US-C12b: the round's
    reply chips ended the turn with no model call). It is when it carries chips and its words are
    that round's (spacing aside, as FX-017's held rounds are read)."""
    text = getattr(answer, "content", None)
    if not getattr(answer, REPLY_CHIPS, None) or not isinstance(text, str) or not (round_text or "").strip():
        return False
    return text.split() == round_text.split()


def as_streamed(response: Any) -> Any:
    """A copy of ``response`` marked streamed: its text is on the screen already, as its round's, so
    the turn sends no tail. ``response`` is never changed. One that cannot be marked comes back as
    it was, and that is logged."""
    if dataclasses.is_dataclass(response) and STREAMED in {f.name for f in dataclasses.fields(response)}:
        return dataclasses.replace(response, **{STREAMED: True})
    fresh = copy.copy(response)
    try:
        setattr(fresh, STREAMED, True)
    except AttributeError:
        logger.warning("[turn-parts] a %s answer cannot be marked streamed: its round is narration, as before",
                       type(response).__name__)
        return response
    return fresh
