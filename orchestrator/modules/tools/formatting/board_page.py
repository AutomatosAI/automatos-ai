"""F442 (nights 16-17): a page of the board reaches the model whole.

"List them all, every single card in review" read 19 of 54 cards in eleven calls, then
"the results were cut off". Each platform_list_tasks page went to the model as indent=2
JSON carrying every card's full brief, and a chat turn keeps 2,000 tokens of one tool's
answer (consumers/chatbot/service.py _TOOL_RESULT_TOKEN_BUDGET; the tool loop's
content_truncate_tokens). A card with a 1,500-character brief is ~485 tokens, so about
four fit; the cut took the last visible card's agent and number and the page's total and
next_offset with it, and the model worked out offsets ahead that skipped cards.

Now a board page over that budget reaches the model as one line per card (number, title,
status, priority, agent, dates, tags, an error when there is one), with no briefs, as many
whole lines as fit, and a header and a trailer that say where the page sits and which
offset reads the rows it did not show. A page that fits is unchanged.
"""
from __future__ import annotations

import functools
from typing import Any, Callable, Dict, List, Optional

from core.context_guard import count_tokens
from modules.tools.formatting.generated_document_summary import DEFAULT_MAX_CHARS
from modules.tools.formatting.tickets_by_number import PLATFORM_PREFIX

# What a chat turn keeps of one tool's answer is 2,000 tokens; a compact page stays under it.
PAGE_TOKENS = 1900
TITLE_CHARS = 100
TAGS_CHARS = 60
ERROR_CHARS = 80
DAY_CHARS = 10            # "2026-10-15"
CELL = " | "
NOTHING = "-"
NULL = "null"
HEAD = "Tool: {tool}\nStatus: success\n\n"
WHERE = ("total={total} offset={offset} limit={limit} returned={returned} shown={shown} "
         "next_offset={next_offset}")
OWNER_LEGEND = "number | title | status | priority | agent | created | finished | tags | err"
VISITOR_LEGEND = "id | title | status | created | finished"
SKIPPED = "Rows {first}..{last} of this page were not shown: call again with offset={first} for them."
MORE = "More match: call again with offset={next_offset}, same filters."
EVERY = "That is every match."
BRIEFS = "Briefs are left out here: platform_get_task #NNNN reads one."

Row = Dict[str, Any]


def is_a_board_page(result: Any, tool_name: str) -> bool:
    """A successful platform answer shaped as a board page (only board_task_list._envelope
    builds ``tasks`` beside ``total`` and ``next_offset``)."""
    if not (str(tool_name or "").startswith(PLATFORM_PREFIX) and isinstance(result, dict)):
        return False
    rows = result.get("tasks")
    return (bool(result.get("success")) and isinstance(rows, list) and all(isinstance(r, dict) for r in rows)
            and "total" in result and "next_offset" in result)


def over_budget(text: str, max_chars: int) -> bool:
    """Whether the chat would cut ``text``: past PAGE_TOKENS, or already cut at ``max_chars``."""
    return len(text) >= max_chars or count_tokens(text) > PAGE_TOKENS


def _words(value: Any, chars: int) -> str:
    """``value`` on one line, at most ``chars`` long."""
    return " ".join(str(value or "").split())[:chars]


def _day(value: Any) -> str:
    return str(value)[:DAY_CHARS] if value else NOTHING


def _owner_line(row: Row) -> str:
    """One card as its owner sees it: by its board number, with its agent; never its brief."""
    tags = _words(", ".join(str(t) for t in row.get("tags") or []), TAGS_CHARS) or NOTHING
    cells = [row.get("number") or f"id {row.get('id')}", _words(row.get("title"), TITLE_CHARS),
             row.get("status") or NOTHING, row.get("priority") or NOTHING, row.get("assigned_agent") or NOTHING,
             f"created {_day(row.get('created_at'))}", f"finished {_day(row.get('completed_at'))}", tags]
    if row.get("error_message"):
        cells.append(f"err {_words(row['error_message'], ERROR_CHARS)}")
    return CELL.join(str(cell) for cell in cells)


def _visitor_line(row: Row) -> str:
    """One card as a widget visitor sees it (F155): what it is and where it stands."""
    return CELL.join([f"id {row.get('id')}", _words(row.get("title"), TITLE_CHARS), str(row.get("status") or NOTHING),
                      f"created {_day(row.get('created_at'))}", f"finished {_day(row.get('completed_at'))}"])


def _trailer(page: Row, shown: int, returned: int) -> str:
    offset = int(page.get("offset") or 0)
    if shown < returned:
        return SKIPPED.format(first=offset + shown, last=offset + returned - 1)
    if page.get("next_offset") is not None:
        return MORE.format(next_offset=page["next_offset"])
    return EVERY


def _page_text(page: Row, tool_name: str, legend: str, lines: List[str], shown: int) -> str:
    """The page with its first ``shown`` lines, the header and the trailer."""
    offset, returned = int(page.get("offset") or 0), len(lines)
    after: Optional[int] = offset + shown if shown < returned else page.get("next_offset")
    where = WHERE.format(total=page.get("total"), offset=offset, limit=page.get("limit"), returned=returned,
                         shown=shown, next_offset=NULL if after is None else after)
    note = [str(page["note"])] if page.get("note") else []
    body = [where, *note, legend, *lines[:shown], _trailer(page, shown, returned), BRIEFS]
    return HEAD.format(tool=tool_name) + "\n".join(body)


def _fits(text: str, max_chars: int) -> bool:
    return len(text) <= max_chars and count_tokens(text) <= PAGE_TOKENS


def board_page(page: Row, tool_name: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """The page as one line per card: whole lines added while the whole text fits, never one cut part-way."""
    rows = page["tasks"]
    visitor = all("assigned_agent" not in row for row in rows)
    legend = VISITOR_LEGEND if visitor else OWNER_LEGEND
    lines = [_visitor_line(row) if visitor else _owner_line(row) for row in rows]
    shown = 0
    while shown < len(lines) and _fits(_page_text(page, tool_name, legend, lines, shown + 1), max_chars):
        shown += 1
    return _page_text(page, tool_name, legend, lines, shown)


def reads_a_board_page_whole(format_for_llm: Callable[..., str]) -> Callable[..., str]:
    """Wrap ``ToolResultFormatter.format_for_llm`` (under ``names_tickets_by_number``, so rows come
    named by their numbers): a board page the chat would cut reaches the model as whole lines."""
    @functools.wraps(format_for_llm)
    def wrapped(result: Dict[str, Any], tool_name: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
        text = format_for_llm(result, tool_name, max_chars)
        if not is_a_board_page(result, tool_name) or not over_budget(text, max_chars):
            return text
        return board_page(result, tool_name, max_chars)
    return wrapped


__all__ = ["PAGE_TOKENS", "board_page", "is_a_board_page", "over_budget", "reads_a_board_page_whole"]
