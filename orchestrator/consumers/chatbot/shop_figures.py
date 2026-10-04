"""F316 (night 9b, build 15): Auto's shop counts came from a document or from memory, not the shop.

Night 9b ran on a coffee roaster's workspace with its shop database (harbourline_shop)
connected. "How many Harvest Club boxes go out on Monday 5 October?" was answered from the
October club box note ("doesn't specify the total number") in every fresh chat, with only the
automatic document search behind it; pushed, it came back 5,957 (2fe5384a), 87 (e77f789a) and
63. "How many cancelled April to September?" gave 2, then 118, then "12" with a list adding to
11 (170ac093). "Where did that come from?" got "I retrieved that information from our previous
conversation" (7c726e13), "from my memory" (50aac6a9, 11b9a74a) and "I seem to have misplaced
the source" (cb5a62e6). The board gave 63 and 11 every time. F302 put platform_query_data on
every turn, and the document passages already said a count belongs to the database; neither
made the turn count from the shop, and nothing checked the reply.

The remembered figures are Auto's own earlier answers: every chat exchange is distilled into
durable facts ("preserve specifics … numbers"), and the next chat's prompt carries them under
"What You Know About This User" with nothing to say they are old (see
``modules/context/remembered_figures.py``). A follow-up turn can't see the call behind the
previous reply either, so "where did that come from?" reached for memory.

Now, in a workspace with a connected database, a turn that asks a figure from the shop (a
count, a total, stock, takings, whether there were any orders or cancellations) is told, after
the document passages, to count it from the shop with platform_query_data in this reply: never
from an earlier answer, a remembered figure or a dated document, and to say where the figure
came from. "Where did that come from?" after such a question is told to count again. The turn
is marked (``mark_shop_figure_turn``), so a reply that gives a figure, says it isn't there, or
names memory as its source with no call to the shop this turn is nudged once and then
corrected (``modules/tools/execution/shop_and_team_claims.py``). A widget visitor's turn gets
none of it: the shop is the owner's.
"""
from __future__ import annotations

import functools
import logging
import re
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from modules.tools.execution.shop_and_team_claims import mark_shop_figure_turn

logger = logging.getLogger(__name__)

SYSTEM_ROLE = "system"
SHOP_NOTE = (
    "The owner is asking for a figure from their shop (a count, a total, stock, takings or orders). Count it from "
    "the shop system in this reply: call platform_query_data now with the owner's own words, even if you gave a "
    "figure before, remember one, or a document or a card gives one. A dated document (a note, a list, a review) "
    "is not today's figure, and a figure remembered from an earlier chat is not current: never give either as "
    "the answer. Say in the reply that the figure is from the shop system and what it counted. If the shop "
    "system can't give it, say so plainly; a document that doesn't give the number is no answer."
)
WHERE_FROM_NOTE = (
    "The owner is asking where your last figure came from. You can't see the calls behind your earlier replies, "
    "and your memory or an earlier chat is no source. Count it again now from the shop system with "
    "platform_query_data, give what it returns and what it counted, and say plainly whether it matches the "
    "figure you gave."
)

_COUNT = re.compile(r"\bhow many\b|\bnumber of\b|\bcount(?:s|ed|ing)?\b(?! on\b)|\btotals?\b|\btally\b", re.I)
_HOW_MUCH = re.compile(r"\bhow much\b", re.I)
# "How much" for a price or a margin is the owner's terms or margin sheet, not a count.
_PRICE = re.compile(r"\b(?:charge[sd]?|costs?|prices?|priced|fees?|pay|paid|spend|budget|margins?|profit)\b", re.I)
_STOCK = re.compile(
    r"\b(?:in stock|stock (?:levels?|of|for|left)|green stock|on hand|reorder|"
    r"run(?:s|ning)? (?:out|short|low)|takings|revenue|turnover)\b"
    r"|\b(?:got|have|has|is there|are there)\s+enough\b|\benough\s+\w+(?:\s+\w+)?\s+(?:for|to cover|left)\b"
    r"|\bwhat did (?:the shop|we|i|the business) (?:take|make|sell|turn over)\b",
    re.I)
_ANY_RECORDS = re.compile(
    r"\b(?:were|was|are|is|did|have|has)\s+(?:there\s+)?any\b[^?.!\n]{0,60}\b(?:orders?|box(?:es)?|subscri\w*|"
    r"members?|customers?|caf[ée]s?|accounts?|shipments?|deliveries|cancell\w*|refunds?|returns?|sales)\b",
    re.I)
# A count of the platform's own things, not the shop's: cards, agents, documents, words.
_NOT_THE_SHOP = re.compile(
    r"\bhow many\s+(?:\w+\s+){0,2}?(?:cards?|tasks?|tickets?|agents?|documents?|papers?|files?|playbooks?|"
    r"missions?|words?|characters?|credits?|tokens?|tools?|skills?|steps?|runs?)\b",
    re.I)
_WHERE_FROM = re.compile(
    r"\bwhere (?:did|does|do|has) (?:that|this|it|these|those|the (?:figure|number|count|total)s?)\s+"
    r"(?:come|came) from\b|\bwhat(?:'s| is| was) (?:the|your) source\b"
    r"|\bhow did you (?:get|count|work out|calculate|come up with)\b",
    re.I)


def asks_a_shop_figure(text: object) -> bool:
    """Whether the owner's message asks a figure the shop holds: a count, a total, stock,
    takings, or whether there were any orders, boxes, cancellations."""
    said = _NOT_THE_SHOP.sub(" ", str(text or "").replace("’", "'"))   # "how many cards …" is the board's
    how_much = bool(_HOW_MUCH.search(said)) and not _PRICE.search(said)
    return bool(_COUNT.search(said) or _STOCK.search(said) or _ANY_RECORDS.search(said)) or how_much


def asks_where_a_figure_came_from(text: object) -> bool:
    """Whether the owner asks where Auto's last answer came from."""
    return bool(_WHERE_FROM.search(str(text or "").replace("’", "'")))


def _text(content: Any) -> str:
    if isinstance(content, list):
        return " ".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return str(content or "")


def previous_owner_message(llm_messages: List[Dict[str, Any]], latest_text: str) -> str:
    """The owner's message before this one in the chat, or ''."""
    said = [_text(m.get("content")) for m in llm_messages if isinstance(m, dict) and m.get("role") == "user"]
    if said and said[-1].strip() == str(latest_text or "").strip():
        said = said[:-1]
    return said[-1] if said else ""


def _has_database(chat: Any) -> bool:
    from consumers.chatbot.knowledge_prefetch import _has_database as has_database

    return has_database(chat.db, chat.workspace_id)


def shop_note(chat: Any, latest_text: str, llm_messages: List[Dict[str, Any]]) -> Optional[str]:
    """The note for a turn that asks a shop figure (or where one came from), else None.
    Marks the turn either way, so the claim check knows."""
    mark_shop_figure_turn(False)
    if getattr(chat, "widget_mode", False):
        return None
    asks = asks_a_shop_figure(latest_text)
    where = (not asks and asks_where_a_figure_came_from(latest_text)
             and asks_a_shop_figure(previous_owner_message(llm_messages, latest_text)))
    if not (asks or where) or not _has_database(chat):
        return None
    mark_shop_figure_turn(True)
    logger.info("[F316] a shop figure is asked: the turn counts it from the shop (where-from=%s)", where)
    return WHERE_FROM_NOTE if where else SHOP_NOTE


Turn = Callable[..., AsyncGenerator[Any, None]]


def counts_from_the_shop(retrieval_first: Turn) -> Turn:
    """Wrap ``StreamingChatService._retrieval_first``: a turn that asks a shop figure gets
    ``SHOP_NOTE`` (or ``WHERE_FROM_NOTE``) last, after the document passages."""
    @functools.wraps(retrieval_first)
    async def wrapped(chat: Any, latest_text: str, llm_messages: List[Dict[str, Any]], *args: Any,
                      **kwargs: Any) -> AsyncGenerator[Any, None]:
        note = shop_note(chat, latest_text, llm_messages)
        async for frame in retrieval_first(chat, latest_text, llm_messages, *args, **kwargs):
            yield frame
        if note:
            llm_messages.append({"role": SYSTEM_ROLE, "content": note})
    return wrapped


__all__ = ["SHOP_NOTE", "WHERE_FROM_NOTE", "asks_a_shop_figure", "asks_where_a_figure_came_from",
           "counts_from_the_shop", "previous_owner_message", "shop_note"]
