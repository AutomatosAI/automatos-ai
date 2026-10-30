"""PRD-268 US-C12b (§6.6, FR-12, K12): a reply with chips ends the turn, with no further model call.

Auto calls ``platform_offer_replies`` (US-C12a) last, after writing its question, when the person
must choose: the round's own text is the reply and the chips are its likely answers. Without this,
the loop calls the model once more after that round: a round trip, and its cost, for a reply that is
already written. ``ends_on_reply_chips`` wraps ``ToolLoopExecutor.run`` beside
``answers_at_the_cap`` (``run`` and ``_execute_round`` stay byte-identical, K4), and for that run
wraps the caller's ``on_tool_result`` and ``on_round_end``:

* after a round in which the tool succeeded (called directly, or as ``platform_execute``'s action),
  ``on_round_end`` answers ``force_final`` with the round's own text, read from the assistant message
  the loop just appended, and ``_force_synthesize`` returns that text with no model call. That text
  is the model's own answer, not a forced synthesis, so the result is not marked ``forced_final``:
  the chat's answer checks (F187's invented ids, F205's internal words) read it as any answer;
* the run's response carries ``reply_chips`` (the tool's replies);
* a round with no text is not ended early: the loop goes on as it always has (the chat would force
  one more synthesis for an empty answer anyway), and the chips ride the answer the next model call
  writes. A later round of other work leaves them off: they were offered for an earlier question;
* a turn that called ``platform_get_brief`` is never ended early and carries no chips (never on the
  brief); a refused call changes nothing; a caller hook's own ``force_final`` wins. A caller that
  ends the run on its own words (the chat's fatal-error line) gets no chips under them; one that
  asks the model for the answer does.

The agent lanes share ``run``: there the chips have nowhere to go and change only the early end.
The names are written here, as ``tool_loop`` writes the actions it reads, so the loop stays light to
import (the test pins each to its handler's); the handler's own check is imported only once a chips
call has run.
"""
from __future__ import annotations

import copy
import dataclasses
import functools
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

Run = Callable[..., Awaitable[Any]]
ToolResultHook = Callable[[str, Dict[str, Any], Any], Awaitable[Any]]
RoundEndHook = Callable[[Any], Awaitable[Any]]

# The response's attribute (core.llm.clients.base.LLMResponse.reply_chips).
REPLY_CHIPS = "reply_chips"
# handlers_replies.OFFER_REPLIES and REPLIES_PARAM (its answer's key too), handlers_brief.GET_BRIEF,
# and the dispatcher whose ``action`` names the action it runs (unified_executor's PRD-64 route).
OFFER_REPLIES = "platform_offer_replies"
REPLIES = "replies"
GET_BRIEF = "platform_get_brief"
DISPATCHER = "platform_execute"
ACTION_KEY = "action"

__all__ = ["GET_BRIEF", "OFFER_REPLIES", "REPLIES", "REPLY_CHIPS", "brief_called", "called_action",
           "ends_on_reply_chips", "offered_replies", "round_text", "says_its_own", "with_reply_chips"]


def called_action(name: str, args: Any) -> str:
    """The action a call ran: ``platform_execute``'s ``action``, else the call's own name."""
    if name == DISPATCHER and isinstance(args, dict):
        return str(args.get(ACTION_KEY) or "").strip()
    return name


def _answer(result: Any) -> Optional[Dict[str, Any]]:
    """The action's own answer when the call succeeded (the chat hands it back as ``raw_result``),
    else None."""
    if not isinstance(result, dict) or result.get("success") is False:
        return None
    raw = result.get("raw_result")
    answer = raw if isinstance(raw, dict) else result
    return answer if answer.get("success") is True else None


def offered_replies(name: str, args: Any, result: Any) -> Optional[List[str]]:
    """The replies a ``platform_offer_replies`` call that succeeded offered, held to the handler's
    own check (US-C12a); None for any other call, a refused one, or replies the check refuses."""
    if called_action(name, args) != OFFER_REPLIES:
        return None
    answer = _answer(result)
    if answer is None:
        return None
    from modules.tools.discovery.handlers_replies import replies_refusal

    shown, refusal = replies_refusal(answer.get(REPLIES))
    return None if refusal is not None else shown


def round_text(messages: Any) -> str:
    """What the model wrote in the round the loop just appended (the last assistant message that
    carries tool calls), exactly as written; '' when it wrote nothing."""
    for message in reversed(messages if isinstance(messages, list) else []):
        if isinstance(message, dict) and message.get("role") == "assistant" and "tool_calls" in message:
            content = message.get("content")
            return content if isinstance(content, str) and content.strip() else ""
    return ""


def brief_called(loop: Any) -> bool:
    """Whether the turn called ``platform_get_brief``, directly or through ``platform_execute``: the
    tracker records each call under the action asked for, succeeded or not, as the model wrote it
    (the executor strips ``action`` before it runs it, so this does too)."""
    outcomes = getattr(getattr(loop, "tracker", None), "outcomes", None) or ()
    return any(str(action).strip() == GET_BRIEF for action, _params, _result in outcomes)


def says_its_own(post: Any) -> bool:
    """A caller hook ending the run on its own words (a fixed line), not on the model's."""
    return post is not None and bool(post.force_final) and post.final_content is not None


def with_reply_chips(response: Any, chips: List[str]) -> Any:
    """A copy of ``response`` carrying ``chips``; ``response`` is never changed. One that cannot
    carry them (a frozen or slotted object) comes back as it was, and that is logged."""
    if dataclasses.is_dataclass(response) and REPLY_CHIPS in {f.name for f in dataclasses.fields(response)}:
        return dataclasses.replace(response, **{REPLY_CHIPS: list(chips)})
    fresh = copy.copy(response)
    try:
        setattr(fresh, REPLY_CHIPS, list(chips))
    except AttributeError:
        logger.warning("[reply-chips] a %s response cannot carry its chips — it is sent without them",
                       type(response).__name__)
        return response
    return fresh


class _ChipsRun:
    """One run's chips: the caller's hooks, wrapped, and the response the run ends on."""

    def __init__(self, loop: Any, messages: Any, on_tool_result: Optional[ToolResultHook],
                 on_round_end: Optional[RoundEndHook]) -> None:
        self._loop, self._messages = loop, messages
        self._on_tool_result, self._on_round_end = on_tool_result, on_round_end
        self.chips: Optional[List[str]] = None
        self._round = 0  # the rounds ended so far: the index of the round a call runs in
        self._offered_in: Optional[int] = None  # the round whose call offered ``chips``
        self._ended_here = False  # this run ended on the round's own text
        self._caller_spoke = False  # a caller hook ended the run on its own words

    async def on_tool_result(self, name: str, args: Dict[str, Any], result: Any) -> Any:
        """Note the chips a call offered, then hand the call to the caller's hook as it came."""
        offered = offered_replies(name, args, result)
        if offered is not None:
            self.chips, self._offered_in = offered, self._round
        post = await self._on_tool_result(name, args, result) if self._on_tool_result is not None else None
        self._caller_spoke = self._caller_spoke or says_its_own(post)
        return post

    async def on_round_end(self, state: Any) -> Any:
        """The caller's hook first (its own force_final wins); then, after a round that offered
        chips and wrote text, that text as the final answer, with no model call."""
        post = await self._on_round_end(state) if self._on_round_end is not None else None
        self._caller_spoke = self._caller_spoke or says_its_own(post)
        offered, self._round = self._offered_in == self._round, self._round + 1
        if not offered or (post is not None and post.force_final) or brief_called(self._loop):
            return post
        text = round_text(self._messages)
        if not text:
            logger.info("[reply-chips] the round that offered chips wrote no text — the loop asks for its answer")
            return post
        from .tool_loop import ToolPostResult  # here: tool_loop imports this module

        logger.info("[reply-chips] %d chips offered — the run ends on the round's own text, no model call",
                    len(self.chips or ()))
        self._ended_here = True
        return ToolPostResult(force_final=True, final_content=text)

    def _ride(self) -> bool:
        """Whether the chips ride the run's answer: offered in the round just before it (or in the
        round it ended in), on no brief, and under no caller's own words."""
        return (self.chips is not None and self._offered_in is not None and self._offered_in >= self._round - 1
                and not self._caller_spoke and not brief_called(self._loop))

    def carried(self, result: Any) -> Any:
        """``result`` with its response carrying the chips, and not marked ``forced_final`` when the
        run ended on the round's own text; as it came otherwise."""
        if not dataclasses.is_dataclass(result) or getattr(result, "response", None) is None:
            return result
        changes: Dict[str, Any] = {}
        if self._ended_here:
            changes["forced_final"] = False  # the model's own answer: the chat's answer checks read it
        if self._ride():
            changes["response"] = with_reply_chips(result.response, self.chips or [])
        return dataclasses.replace(result, **changes) if changes else result


def ends_on_reply_chips(run: Run) -> Run:
    """Wrap ``ToolLoopExecutor.run`` (its arguments are keyword-only): a round that offered reply
    chips and wrote text ends the run on that text, with no model call, and the run's answer
    carries the chips. A run that offered none comes back exactly as it would have."""
    @functools.wraps(run)
    async def wrapped(self: Any, **kwargs: Any) -> Any:
        chips = _ChipsRun(self, kwargs.get("messages"), kwargs.get("on_tool_result"), kwargs.get("on_round_end"))
        result = await run(self, **{**kwargs, "on_tool_result": chips.on_tool_result,
                                    "on_round_end": chips.on_round_end})
        return chips.carried(result)
    return wrapped
