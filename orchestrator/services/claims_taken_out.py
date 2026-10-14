"""P256-FIX-N15-CI: a card's answer says what the claim check took out of it, in its note.

The tool loop takes an unbacked claim out of an agent's text with nothing in its place (Auto's
"Just to be clear… Ask me again" line never goes into text that may be for a customer: #0245's email
to Nadia). The run's result carries the sentences taken out (core/llm/taken_out.py); the card's answer
gets a note for the owner, beside F201's "Check before sending" (services/draft_guides.py), and goes
to review: the agent said it had done something its run did not do.

The note is added as F380's social check adds its own: ``finalize_board_task_run``'s keywords with the
note on the answer, before the writer's own checks read it.
"""
from __future__ import annotations

import functools
from typing import Any, Awaitable, Callable, Dict, Optional

from core.llm.taken_out import TAKEN_OUT_KEY

Async = Callable[..., Awaitable[Any]]

TAKEN_OUT_NOTE = ("Check before sending: nothing in this run did what the answer said here, so it was taken "
                  "out of the answer: {quoted}.")
QUOTED_CHARS = 200   # of each sentence quoted


def taken_out_note(exec_result: Any) -> Optional[str]:
    """The owner's note for a successful run whose text had claims taken out, else None."""
    if not isinstance(exec_result, dict) or exec_result.get("status") != "success":
        return None
    taken = (exec_result.get("execution") or {}).get(TAKEN_OUT_KEY) or ()
    quoted = "; ".join(f'"{str(sentence)[:QUOTED_CHARS]}"' for sentence in taken if str(sentence).strip())
    return TAKEN_OUT_NOTE.format(quoted=quoted) if quoted else None


def with_the_note(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """``finalize_board_task_run``'s keywords with the note on the answer and ``force_review``, when
    the run took a claim out; else as they came. A new dict: ``kwargs`` is never changed."""
    exec_result = kwargs.get("exec_result")
    note = taken_out_note(exec_result)
    if note is None:
        return kwargs
    answer = f"{exec_result.get('result') or ''}\n\n{note}".strip()
    return {**kwargs, "exec_result": {**exec_result, "result": answer}, "force_review": True}


def a_cards_claims_taken_out_are_noted(finalize: Async) -> Async:
    """Wrap ``api.board_tasks.finalize_board_task_run`` (keywords after ``db``): an answer the claim
    check took a sentence out of says so in its note, and goes to review."""
    @functools.wraps(finalize)
    async def wrapped(db: Any, *args: Any, **kwargs: Any) -> Any:
        return await finalize(db, *args, **with_the_note(kwargs))
    return wrapped


__all__ = ["TAKEN_OUT_NOTE", "a_cards_claims_taken_out_are_noted", "taken_out_note", "with_the_note"]
