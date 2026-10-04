"""F324 (night 9b, build 15): a correction told to Auto didn't reach the team, and Auto said it had.

"My Support Agent just told me nothing is changing for Quay Coffee House's terms. I told you
this afternoon they move to 30-day terms from November. Can you make sure the whole team knows
that, so I don't have to tell each of them?" got "I've stored the information … in my memory.
This will ensure that all agents, including the Support Agent, are aware of this change going
forward" (5e404e17). Its one call was platform_store_memory. A minute later Support #1962 said
"no documents indicating any planned changes" and the Analyst #1963 found an older memory
("Memory 5": Quay on 14-day terms), not the new one. Auto's memory is recalled by a search per
run, so an agent may or may not get a fact from it, and may get an older one beside it. When the
owner wrote the change down as a document (doc 1560), Support, the Watchdog and the Analyst had
it within minutes: the owner's documents are what every agent reads.

The honest, simplest fix is the one the owner found: a correction for the whole team goes into
the owner's documents, in their words and with the date it starts, when they say so (F305: the
owner decides what enters their knowledge). Not a new shared memory block in every agent's
prompt: that would be a second place for the owner's facts beside their documents, with no
page they can read or edit, and F249's lessons already show how a note carried into every run
gets read as "your corrections" on cards it was never about (F315, F318).

So a message asking Auto to make sure the team knows something gets ``TELL_THE_TEAM_NOTE``
after the document passages: Auto says it remembers it itself, and asks whether to write it
into a note in the owner's documents (platform_upload_document) so every agent reads it, and
does that only on a yes. And a reply that says the agents know, are aware or have been told,
with no call this turn that put it where they read it, is nudged once and then corrected
(``modules/tools/execution/shop_and_team_claims.py``, "told to the whole team").
"""
from __future__ import annotations

import functools
import logging
import re
from typing import Any, AsyncGenerator, Callable, Dict, List

logger = logging.getLogger(__name__)

SYSTEM_ROLE = "system"
TELL_THE_TEAM_NOTE = (
    "The owner wants the whole team to know something. Your memory (platform_store_memory) is only yours: the "
    "agents don't reliably get it on their cards. What every agent reads is the owner's documents. Say plainly "
    "that you'll remember it yourself, then ask whether to write it into a short note in their documents "
    "(platform_upload_document, in the owner's words, with the date it starts) so every agent reads it, and do "
    "that only when they say yes. Never say the agents know, are aware, or have been told unless a call in this "
    "reply put it where they read it."
)

_GROUP = (r"(?:(?:the|my|our) (?:whole |entire )?team|(?:all|every|each) (?:of )?(?:the |my |our )?"
          r"(?:agents?|helpers?|team)|everyone|everybody|(?:the|my|our) agents|each of them|all of them)")
_ASKS = re.compile(
    r"\b(?:make sure|ensure|let|tell|inform|update|brief|remind|warn)\b(?! (?:me|us)\b)[^.?!\n]{0,40}\b" + _GROUP + r"\b"
    r"|\b(?:whole|entire) team (?:knows?|is aware)\b|\bso i don't have to tell (?:each|every|all)\b",
    re.IGNORECASE)


def asks_to_tell_the_team(text: object) -> bool:
    """Whether the owner asks Auto to make sure the team (every agent) knows something."""
    return bool(_ASKS.search(str(text or "").replace("’", "'")))


Turn = Callable[..., AsyncGenerator[Any, None]]


def tells_the_team_honestly(retrieval_first: Turn) -> Turn:
    """Wrap ``StreamingChatService._retrieval_first``: a message asking Auto to tell the team
    gets ``TELL_THE_TEAM_NOTE`` last, after the document passages."""
    @functools.wraps(retrieval_first)
    async def wrapped(chat: Any, latest_text: str, llm_messages: List[Dict[str, Any]], *args: Any,
                      **kwargs: Any) -> AsyncGenerator[Any, None]:
        async for frame in retrieval_first(chat, latest_text, llm_messages, *args, **kwargs):
            yield frame
        if not getattr(chat, "widget_mode", False) and asks_to_tell_the_team(latest_text):
            logger.info("[F324] the owner asks Auto to tell the team: memory is Auto's own, documents reach them")
            llm_messages.append({"role": SYSTEM_ROLE, "content": TELL_THE_TEAM_NOTE})
    return wrapped


__all__ = ["TELL_THE_TEAM_NOTE", "asks_to_tell_the_team", "tells_the_team_honestly"]
