"""F303 (night 9): Auto gave way on a figure without checking it again.

"The Analyst counted 11 club cancellations April to September on card 1886. You told
me 4 a minute ago. Which is right, and why did you say 4?" got "The Analyst's count of
11 … is correct. My previous answer of 4 was based on an incomplete query of the
database." with no tool call at all (chat ec665ae6, L106): Auto agreed because it was
told to, then made up why. Earlier, "Where did 87 come from? Two cards this afternoon
said 63 active." got "I incorrectly stated 87 members based on an internal system
check" (chat be7d1ae6), also unchecked.

Two pieces, the cheapest that hold:

- This note. When the owner's message sets a figure from another source against one
  of Auto's, or asks which is right, the turn says, before the first model call, that
  Auto checks again with a tool (a count, a query, the source read) before it agrees or
  disagrees, says so if it can't, and gives a reason for a difference only if a tool
  showed one.
- The claim check (modules/tools/execution/action_claims.py, "re-checked"): a reply in
  which Auto says a figure is right or wrong, or what an earlier one "was based on",
  with no count, query or read behind it, is nudged once to check, and corrected in
  plain words if it still hasn't.

A widget visitor's turn gets no note: the board and its figures are the owner's.
"""
from __future__ import annotations

import functools
import logging
import re
from typing import Any, AsyncGenerator, Callable, Dict, List

logger = logging.getLogger(__name__)

SYSTEM_ROLE = "system"
RECHECK_NOTE = (
    "The owner's message sets a figure from another source against one of yours, or asks which figure is "
    "right. Before you agree or disagree, check again in this reply: run the count or the query again, or "
    "read the source, with a tool, and say which figure is right and why from what the tool returned. If "
    "you can't check it, say so plainly. Never agree only because the owner or an agent said so, and never "
    "guess why an earlier figure was different: give a reason only if a tool showed it."
)

_FIGURE = re.compile(r"\d")
# The owner pointing at another figure, or at Auto's: "the Analyst counted 11", "two cards
# said 63", "you told me 4", "which is right?", "where did 87 come from?", "that's wrong".
_DISPUTE = re.compile(
    r"\bwhich (?:one |figure |number |count |total )?is (?:right|correct)\b"
    r"|\byou (?:said|told me|gave me|got|counted|reported|had)\b"
    r"|\bwhere did\b[^.?!\n]{0,40}\bcome from\b"
    r"|\b(?:counted|said|says|got|gets|found|finds|showed|shows|reported|reports|came to|comes to)\s+"
    r"(?:it (?:at|as)\s+|only\s+|just\s+|about\s+|around\s+)?[£$€]?\d"
    r"|\b(?:that's|that is|this is|it's|it is) (?:wrong|not right|incorrect|off)\b",
    re.IGNORECASE,
)


def disputes_a_figure(text: object) -> bool:
    """Whether the owner's message sets another figure against Auto's, or asks which is right."""
    said = str(text or "")
    return bool(_FIGURE.search(said)) and bool(_DISPUTE.search(said))


Turn = Callable[..., AsyncGenerator[Any, None]]


def rechecks_disputed_figures(retrieval_first: Turn) -> Turn:
    """Wrap ``StreamingChatService._retrieval_first``: a message that disputes a figure
    gets ``RECHECK_NOTE`` last, before the turn's first model call."""
    @functools.wraps(retrieval_first)
    async def wrapped(chat: Any, latest_text: str, llm_messages: List[Dict[str, Any]], *args: Any,
                      **kwargs: Any) -> AsyncGenerator[Any, None]:
        if not getattr(chat, "widget_mode", False) and disputes_a_figure(latest_text):
            logger.info("[F303] the owner disputes a figure: the turn is told to check again first")
            llm_messages.append({"role": SYSTEM_ROLE, "content": RECHECK_NOTE})
        async for frame in retrieval_first(chat, latest_text, llm_messages, *args, **kwargs):
            yield frame
    return wrapped


__all__ = ["RECHECK_NOTE", "disputes_a_figure", "rechecks_disputed_figures"]
