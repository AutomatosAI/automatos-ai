"""F328 (night 9b): a run that reaches its round cap still ends on its answer.

Long cards stopped at a step they announced: #1970 (BA, 16:07:10Z) "Let me check Quay
Bakehouse's recent orders through September:", #1994 (Ops) twice, 16:43:41Z "Let me
check the payment terms report:" and 16:45:21Z "Now let me get supplier information:".
F306 sent each to review, but none was the one-nudge budget or the model announcing
again: every one logged "[tool-loop] iteration 10: 1 tool call(s)" and then "Hit max
tool iterations (10)" (a board card's ``execute_with_prompt`` runs at
``max_tool_iterations=10``). Eleven sonnet calls per run in llm_usage, one new tool a
round (platform_execute, search_knowledge, platform_get_deliverable): the runs were
working, not looping. The 11th reply asked for one more call; the loop stops at its cap
with that reply as its response, the agent path keeps only the reply's words, and the
words before a tool call are the announcement. The call itself is dropped.

The chat lane already answers at the cap (``consumers.chatbot.service``: one more call,
tools off). The agent lanes now do the same: the announcement stays in the history,
the loop asks once, in the user's turn, for the answer from the results it has, with
no tools, so nothing more can run. Bound: one model call per run, never a tool call.
A reply that is still empty or still asks for a tool leaves the run where it stopped,
and F306 says so on the card.
"""
from __future__ import annotations

import dataclasses
import functools
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .nudges import announced_step, blank, nudge

logger = logging.getLogger(__name__)

Messages = List[Dict[str, Any]]
LLMCall = Callable[[Messages, Optional[List[Dict[str, Any]]]], Awaitable[Any]]
Run = Callable[..., Awaitable[Any]]

CAP_ANSWER_MSG = (
    "This run has used every round of tool calls it is allowed, so {what} will not run. Write "
    "your answer now from the results above: the finished work itself (the figures, the table, "
    "the text), not what you did. Say plainly anything you could not check."
)
NEXT_STEP = "the step you just announced (\"{step}\")"
NEXT_CALL = "the tool call you just asked for"


def ran_out_of_rounds(result: Any, max_iterations: int) -> bool:
    """The loop stopped at its cap with a tool call still asked for, so its response is
    not an answer (F328: #1970, #1994 twice)."""
    response = getattr(result, "response", None)
    return (bool(getattr(result, "max_iterations_reached", False))
            and getattr(result, "iterations", 0) >= max_iterations
            and bool(getattr(response, "tool_calls", None)))


def _chat_turn() -> bool:
    """A chat turn answers at the cap itself (its limit notice, then a call with no
    tools); the agent lanes book their own lane (core.llm.usage_context)."""
    from core.llm.usage_context import LANE_CHAT, current_usage_scope

    return current_usage_scope().get("request_type") == LANE_CHAT


async def answer_at_the_cap(llm: LLMCall, pending: Any, messages: Messages) -> Optional[Any]:
    """Ask once, with tools off, for the answer from the results already in
    ``messages``. ``pending`` is the reply that asked for a call past the cap: its words
    stay in the history and the nudge names its step. None when the reply is still
    empty or still asks for a tool."""
    text = getattr(pending, "content", "") or ""
    step = announced_step(text)
    what = NEXT_STEP.format(step=step) if step else NEXT_CALL
    if text.strip():
        messages.append({"role": "assistant", "content": text})
    messages.append(nudge(CAP_ANSWER_MSG.format(what=what)))
    answer = await llm(messages, None)
    if blank(answer) or getattr(answer, "tool_calls", None):
        return None
    return answer


def answers_at_the_cap(run: Run) -> Run:
    """Wrap ``ToolLoopExecutor.run``: when an agent run stops at its round cap with a
    call still asked for, its response becomes the answer it gives with tools off.
    Any other result, and every chat turn, comes back as it was."""
    @functools.wraps(run)
    async def wrapped(self: Any, **kwargs: Any) -> Any:
        result = await run(self, **kwargs)
        if not ran_out_of_rounds(result, self.max_iterations) or _chat_turn():
            return result
        logger.warning("[F328] the run reached its cap of %d rounds with a call still asked for — "
                       "asking once for its answer, tools off", self.max_iterations)
        try:
            answer = await answer_at_the_cap(self._llm, result.response, kwargs["messages"])
        except Exception:
            logger.exception("[F328] the call for the answer at the cap failed — the run ends where it stopped")
            return result
        if answer is None:
            logger.warning("[F328] the answer at the cap came back empty or asked for a tool — the run "
                           "ends where it stopped")
            return result
        return dataclasses.replace(result, response=answer)
    return wrapped


__all__ = ["CAP_ANSWER_MSG", "answer_at_the_cap", "answers_at_the_cap", "ran_out_of_rounds"]
