"""F316 (night 9b, build 15): a figure remembered from an earlier chat was given as today's.

"How many cancelled April to September?" got "2", and "Where did that come from?" got "I
retrieved that information from our previous conversation" (7c726e13); "Were any boxes late?"
got order 5207 from search_knowledge alone, then "I found that information in my memory …
stored on October 4th" (11b9a74a). The figures are Auto's own earlier answers. Every chat
exchange is distilled into durable facts for long-term memory, and the distiller is told to
"preserve specifics (names, standards, numbers, ids)", so "2 Harvest Club members cancelled"
became a business fact. The next chat's prompt carries it under "What You Know About This
User", undated, beside what the owner really told Auto, with nothing to say a counted figure
is old.

Now:

- ``labels_what_is_remembered`` wraps the memory block's builders (the chat's
  ``MemorySection.render`` and ``atom_memory_block``): the block opens with
  ``REMEMBERED_RULE``. What the owner told Auto stands until they change it; a figure counted
  or read from a database, the board or a document is not today's figure, is counted again
  before it is given, and is called remembered if it is mentioned.
- ``leaves_counted_figures_out`` wraps the distiller's prompt: a figure the assistant counted
  or read is not a durable fact and is left out; what the owner states (prices, terms, dates, a
  change or a correction) is kept.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

REMEMBERED_RULE = (
    "Remembered from earlier chats and work: each was true when it was stored, not necessarily now. What the "
    "owner told you (a change, a correction, a decision) stands until they change it: apply it. A figure that "
    "was counted or read from a database, the board or a document (a count, a total, stock, takings) is not "
    "today's figure: count it again before you give it, and if you mention a remembered figure, say it is "
    "remembered from an earlier chat. Never give your memory or an earlier conversation as the source of a "
    "current figure."
)
MEMORY_HEADINGS = ("## What You Know About This User", "## What you remember about this user:")
DISTILL_ANCHOR = "Return ONLY a JSON array"
DISTILL_RULE = (
    "A figure the assistant counted or read from a database, the board or a document (a count, a total, a stock "
    "level, takings, an order's status) is not a durable fact: it changes and is counted again when asked, so "
    "leave it out. Keep what the user states as a standing fact, a change or a correction (prices, terms, dates, "
    "who supplies what)."
)


def with_the_rule(block: str) -> str:
    """``block`` with ``REMEMBERED_RULE`` under its heading (or first); '' stays ''."""
    if not block or not block.strip():
        return block
    for heading in MEMORY_HEADINGS:
        if heading in block:
            return block.replace(heading, f"{heading}\n{REMEMBERED_RULE}\n", 1)
    return f"{REMEMBERED_RULE}\n\n{block}"


Builder = Callable[..., Awaitable[str]]


def labels_what_is_remembered(build: Builder) -> Builder:
    """Wrap an async builder of the memory block: what it returns opens with the rule."""
    @functools.wraps(build)
    async def wrapped(*args: Any, **kwargs: Any) -> str:
        return with_the_rule(await build(*args, **kwargs))
    return wrapped


Prompt = Callable[[str, str], str]


def leaves_counted_figures_out(build_prompt: Prompt) -> Prompt:
    """Wrap the distiller's prompt builder: ``DISTILL_RULE`` goes in before its output rule."""
    @functools.wraps(build_prompt)
    def wrapped(user_message: str, assistant_response: str) -> str:
        prompt = build_prompt(user_message, assistant_response)
        if DISTILL_ANCHOR not in prompt:
            logger.warning("[F316] the distill prompt has no '%s': counted figures are not ruled out",
                           DISTILL_ANCHOR)
            return prompt
        return prompt.replace(DISTILL_ANCHOR, f"{DISTILL_RULE}\n\n{DISTILL_ANCHOR}", 1)
    return wrapped


__all__ = ["DISTILL_RULE", "REMEMBERED_RULE", "labels_what_is_remembered", "leaves_counted_figures_out",
           "with_the_rule"]
