"""PRD-256 US-003: memory takes receipts, what the turn did, told apart from what it said.

Every chat exchange is distilled into durable facts for long-term memory (``SmartMemoryManager``,
``_build_distill_prompt``). The distiller read the reply alone, so "I've created the ticket"
after a call that failed, or after no call at all, could be remembered next turn as a ticket
that exists. A turn now carries its receipts (``consumers/chatbot/receipts.py``: the platform's
own account of each call, built from the tool tracker after the loop; the model never writes
them), and memory reads them as the record of actions, the reply as what was said:

- ``remembers_the_receipts`` wraps ``store_conversation``: it takes the turn's receipts
  (``receipts=``; None for a store that is not a chat turn's), which the distiller reads while
  that store runs.
- ``the_record_beside_the_reply`` wraps the distiller's prompt: with receipts, ``RECORD_RULE``
  and the record of actions go in before the exchange and the reply is labelled as what was
  said. A fact that work was done comes from the record alone: a claim the receipts do not back
  is never remembered as done, and a card the turn did make is remembered by its number.

A receipt is the dict ``receipts.receipt`` builds: ``{action, kind, status, subject, effect,
link, reason}``. This module reads those keys only, so the memory stack imports no tool code.
"""
from __future__ import annotations

import functools
import logging
from contextvars import ContextVar
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

Receipt = Dict[str, Any]
Store = Callable[..., Awaitable[bool]]
Prompt = Callable[[str, str], str]

RECORD_HEADING = ("Record of actions (written by the platform from the calls that ran this turn; the assistant "
                  "cannot write it):")
NO_ACTIONS = "- No action ran in this turn."
RECORD_RULE = (
    "The exchange comes with the platform's record of the actions its turn ran. The record is the only account "
    "of what was done; the assistant's reply is only what it said. A fact that something was done (created, "
    "moved, sent, posted, saved, changed) comes from the record alone: when the reply says work was done and the "
    "record has no done action for it, it was NOT done, so store no fact that it was. A refused or skipped action "
    "was not done. A done write is worth keeping when it matters later: keep its subject as the record gives it "
    "(a card's number, a document's title)."
)
REPLY_LABEL = "Assistant's reply (what it said, not a record of what it did): "

# The receipts of the store that is running, set by ``remembers_the_receipts`` for that call only.
_REMEMBERED: ContextVar[Optional[List[Receipt]]] = ContextVar("remembered_receipts", default=None)


def _record_line(r: Receipt) -> str:
    """One receipt as a line of the record: "- platform_create_task, write done: #0422: task created"."""
    effect = r.get("effect") or ""
    what = f"{r['subject']}: {effect}" if r.get("subject") else effect
    why = f" ({r['reason']})" if r.get("reason") else ""
    return f"- {r.get('action')}, {r.get('kind')} {r.get('status')}: {what}{why}"


def record_of_actions(receipts: Sequence[Receipt]) -> str:
    """The record of actions the distiller reads: one line per receipt, or that none ran."""
    return "\n".join([RECORD_HEADING, *([_record_line(r) for r in receipts] or [NO_ACTIONS])])


def with_the_record(prompt: str, receipts: Sequence[Receipt], user_message: str, assistant_response: str) -> str:
    """The distill prompt with the record of actions before its exchange and the reply labelled as
    what was said; the prompt as it is (and a warning) when its exchange cannot be found."""
    exchange = f"User: {user_message}\nAssistant: {assistant_response}"
    at = prompt.rfind(exchange)
    if at < 0:
        logger.warning("[PRD-256] the distill prompt has no exchange to put the record of actions beside")
        return prompt
    told = (f"{RECORD_RULE}\n\n{record_of_actions(receipts)}\n\n"
            f"User: {user_message}\n{REPLY_LABEL}{assistant_response}")
    return prompt[:at] + told + prompt[at + len(exchange):]


def remembers_the_receipts(store: Store) -> Store:
    """Wrap ``SmartMemoryManager.store_conversation``: it takes the turn's receipts (``receipts=``;
    None for a store that is not a chat turn's), which its distiller reads while it runs."""
    @functools.wraps(store)
    async def wrapped(*args: Any, receipts: Optional[Sequence[Receipt]] = None, **kwargs: Any) -> bool:
        token = _REMEMBERED.set(None if receipts is None else list(receipts))
        try:
            return await store(*args, **kwargs)
        finally:
            _REMEMBERED.reset(token)
    return wrapped


def the_record_beside_the_reply(build_prompt: Prompt) -> Prompt:
    """Wrap the distiller's prompt builder: with the receipts of the store that runs it, the record
    of actions goes in before the exchange, told apart from the reply."""
    @functools.wraps(build_prompt)
    def wrapped(user_message: str, assistant_response: str) -> str:
        prompt = build_prompt(user_message, assistant_response)
        receipts = _REMEMBERED.get()
        return prompt if receipts is None else with_the_record(prompt, receipts, user_message, assistant_response)
    return wrapped


__all__ = ["NO_ACTIONS", "RECORD_HEADING", "RECORD_RULE", "REPLY_LABEL", "record_of_actions",
           "remembers_the_receipts", "the_record_beside_the_reply", "with_the_record"]
