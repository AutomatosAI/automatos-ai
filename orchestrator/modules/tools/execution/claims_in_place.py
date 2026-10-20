"""P256-FIX-N15-2 (F423, night 15): the not-done line is said only in place of a claim, never above.

"Just to be clear: I haven't done that yet, and nothing has changed. Ask me again if you want it
done." sat above 148 of night 15's 458 replies, most with no write call at all, pure questions among
them ("why does a natural Ethiopian taste different", "what's on my board"). It came from the receipts'
second path (``receipts.honesty_lines``), which read every past participle as a claim ("the agent field
was cut off in this pull", "I've used 'Harbour Blend'", "It's been run 18 times") and put the line above
the text. N15-1 already takes an unbacked claim out of the reply and puts the line in its place; that is
now the line's one producer, and the receipts put only a refused write above the text.

The loop's final-reply check (``ToolLoopExecutor._recover_claimed_action``) runs only when the loop ends
on a reply with no tool call and no nudge spent. A run that ends another way (a forced synthesis, the
round cap's answer, a retry after the narration nudge) was answered by the line above; ``its_answer_is_checked``
gives that answer the same check, so a claim is taken out whichever way the run ended. Taking out twice
changes nothing: the line in place denies, and a denial is no claim. As with the loop's own check, a run
offered no tools is not checked (F108-b).

N15-CI (F264-b): in Auto's own turn, a reply that was nothing but claims, one of them a write said wrong,
is answered by the account of the turn's calls (``turn_account.account_of``): what changed, never "Ask
me again" over work that happened. In an agent's run nothing is put in a claim's place (its text may be
a customer's); the sentences taken out are kept for the run (core/llm/taken_out.py), which puts them
in its result's ``execution`` for the card's note (services/claims_taken_out.py).
"""
from __future__ import annotations

import dataclasses
import functools
import logging
from contextvars import ContextVar
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

Run = Callable[..., Awaitable[Any]]

UNBACKED_LOGGED = "[tool-loop] unbacked claim family=%s sentence=%r — replaced by the not-done line"
# P256-FIX-N17-5 (F446): whether the chat turn read before its loop (retrieval first, the board): recorded with
# record_execution, never as an outcome, so its loop's check saw a turn that read nothing. Set for the loop by
# receipts.the_loop_writes_receipts, taken by the first executor made while it runs (the chat's own, ``reads_ahead``).
READ_AHEAD: ContextVar[bool] = ContextVar("tool_loop_read_ahead", default=False)


def reads_ahead() -> bool:
    """Whether the chat loop running now read before it, said once: the executor made first takes it, so an
    agent's run made later in the turn keeps its own default."""
    ahead = READ_AHEAD.get()
    READ_AHEAD.set(False)
    return ahead


def in_place(loop: Any, reply: Any) -> Optional[Any]:
    """A copy of ``reply`` with its unbacked claims taken out and the not-done line in the first one's
    place (``claim_removed``), logged with the claim and its sentence; None when it calls a tool, says
    nothing or claims nothing the run's calls did not do (``loop.promises``: whose voice it is, RVW-7;
    ``loop.read_ahead``: the turn read before the loop, N17-5)."""
    text = getattr(reply, "content", None)
    if not isinstance(text, str) or not text.strip() or getattr(reply, "tool_calls", None):
        return None
    from consumers.chatbot.claim_removed import in_place_of, without_unbacked_claims
    from core.llm.taken_out import kept_for_the_run

    removed = without_unbacked_claims(text, loop.tracker.outcomes, promises=loop.promises,
                                      read_ahead=getattr(loop, "read_ahead", False))
    if removed is None:
        return None
    logger.warning(UNBACKED_LOGGED, removed.family, removed.sentence)
    kept_for_the_run(removed.taken_out)
    return in_place_of(reply, _said(loop, removed))


def _said(loop: Any, removed: Any) -> str:
    """What the owner gets in place of the reply: the reply without its unbacked claims; in Auto's own
    turn, the account of the run's calls when nothing else was left and the line would deny a write
    the run did (``Removed.denies_done``, F264-b)."""
    from .turn_account import account_of, auto_speaks

    if removed.denies_done and auto_speaks(loop.promises):
        return account_of(loop.tracker.outcomes)
    return removed.text


def its_answer_is_checked(run: Run) -> Run:
    """Wraps ``ToolLoopExecutor.run``: the response it ends on, however it ended, holds no unbacked claim
    (a run offered tools; ``run`` takes its arguments by keyword)."""
    @functools.wraps(run)
    async def checked(loop: Any, *args: Any, **kwargs: Any) -> Any:
        result = await run(loop, *args, **kwargs)
        if not kwargs.get("tools"):
            return result
        fresh = in_place(loop, getattr(result, "response", None))
        return result if fresh is None else dataclasses.replace(result, response=fresh)
    return checked


__all__ = ["READ_AHEAD", "UNBACKED_LOGGED", "in_place", "its_answer_is_checked", "reads_ahead"]
