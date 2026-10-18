"""F442 (nights 16-17): "List them all, every single card in review" read 19 of 54 cards in eleven
calls, then "the results were cut off". Each page reached the model as indent=2 JSON with every
card's full brief, cut to the 2,000 tokens a chat turn keeps of one tool's answer: about four cards
fit, the last one lost its agent and number, and the page lost its total and next_offset.

Now a page over that budget reaches the model as one whole line per card, with its number and its
agent and no brief, and says which offset reads what it did not show. The cards here are shaped like
the night's: 55-character titles, 1,500-character briefs, an agent on each, every fifth one tagged.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import re
from datetime import datetime, timezone
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from sqlalchemy import text

from core.context_guard import count_tokens, truncate_to_token_budget
from modules.tools.execution.tool_loop import ToolLoopExecutor
from modules.tools.formatting.board_page import OWNER_LEGEND, PAGE_TOKENS, VISITOR_LEGEND
from modules.tools.formatting.result_formatter import ToolResultFormatter
from modules.tools.formatting.tickets_by_number import by_number_only

REVIEW = 120
FIRST_NUMBER = 1600
TAGGED_EVERY = 5
TAGS = ["harness", "org-review", "risk-4", "rx:1"]
AGENT = "Scout"
BRIEF_WORDS = "Check the supplier's risk register line by line. "
BRIEF = (BRIEF_WORDS * 40)[:1500]
TOOL = "platform_list_tasks"
FINISHED = datetime(2026, 10, 15, 23, 1, tzinfo=timezone.utc)
CHAT_TOOL_TOKENS = inspect.signature(ToolLoopExecutor.__init__).parameters["content_truncate_tokens"].default
HEADER = re.compile(r"^total=(?P<total>\d+) offset=(?P<offset>\d+) limit=(?P<limit>\d+) returned=(?P<returned>\d+) "
                    r"shown=(?P<shown>\d+) next_offset=(?P<next_offset>\d+|null)$", re.MULTILINE)
SKIPPED = re.compile(r"^Rows (?P<first>\d+)\.\.(?P<last>\d+) of this page were not shown: call again with "
                     r"offset=(?P<offset>\d+) for them\.$", re.MULTILINE)
EVERY_NUMBER = set(range(FIRST_NUMBER, FIRST_NUMBER + REVIEW))
MOST_CALLS = 10


def _title(i):
    return f"Supplier risk review {i:03d} for the Q3 controls audit plan"        # 55 characters


@pytest.fixture
def board(db_session, seed_workspace):
    """120 review cards (#1600..#1719) on Scout, 5 in the inbox, 4 in review in another workspace."""
    from core.models.core import BoardTask

    ws, other = UUID(seed_workspace()), UUID(seed_workspace())
    agent_id = db_session.execute(text(
        "INSERT INTO agents (name, agent_type, workspace_id, status, configuration) "
        "VALUES (:name, 'custom', CAST(:w AS uuid), 'active', CAST('{}' AS json)) RETURNING id"),
        {"name": AGENT, "w": str(ws)}).scalar()
    review = [BoardTask(workspace_id=ws, workspace_seq=FIRST_NUMBER + i, title=_title(i), description=BRIEF,
                        status="review", priority="medium", assigned_agent_id=agent_id, completed_at=FINISHED,
                        tags=TAGS if i % TAGGED_EVERY == 0 else []) for i in range(REVIEW)]
    inbox = [BoardTask(workspace_id=ws, title=f"Inbox card {i}", status="inbox") for i in range(5)]
    elsewhere = [BoardTask(workspace_id=other, title=f"Another shop's card {i}", status="review") for i in range(4)]
    db_session.add_all([*review, *inbox, *elsewhere])
    db_session.flush()
    assert all(len(card.title) == 55 for card in review)
    return NS(db=db_session, ws=ws)


def _page(board, **params):
    from modules.tools.discovery.board_task_list import list_board_tasks

    return asyncio.run(list_board_tasks(board.db, board.ws, {"status": "review", **params}))


def _seen(page, tool=TOOL, **kwargs):
    """What the model reads of ``page``."""
    return ToolResultFormatter.format_for_llm(page, tool, **kwargs)


def _read(said):
    """The header's numbers and the board numbers of the rows shown."""
    head = HEADER.search(said).groupdict()
    numbers = [int(line[1:].split(" ", 1)[0]) for line in said.splitlines() if line.startswith("#")]
    return head, numbers


def _as_today(page, tool=TOOL):
    """The indent=2 JSON the model read before F442 (and still reads of a page that fits)."""
    rest = {key: value for key, value in by_number_only(page).items() if key != "success"}
    return f"Tool: {tool}\nStatus: success\n\n{json.dumps(rest, default=str, indent=2)}"[:20000]


def test_a_hundred_card_page_fits_what_the_chat_keeps(board):
    from consumers.chatbot import service

    said = _seen(_page(board, limit=100))

    assert count_tokens(said) <= PAGE_TOKENS < min(service._TOOL_RESULT_TOKEN_BUDGET, CHAT_TOOL_TOKENS)
    assert truncate_to_token_budget(said, CHAT_TOOL_TOKENS) == said
    assert "(truncated)" not in said and BRIEF_WORDS.strip() not in said


def test_each_row_shown_has_its_number_and_agent_and_the_page_says_where_it_sits(board):
    said = _seen(_page(board, limit=100))
    head, numbers = _read(said)

    shown = int(head["shown"])
    assert head["total"] == "120" and head["offset"] == "0" and head["returned"] == "100"
    assert 30 <= shown < 100 and int(head["next_offset"]) == shown and len(numbers) == shown
    assert OWNER_LEGEND in said
    rows = [line for line in said.splitlines() if line.startswith("#")]
    for line, number in zip(rows, numbers):
        assert line.startswith(("#16", "#17")) and f" | {AGENT} | " in line
        assert ("risk-4" in line) is ((number - FIRST_NUMBER) % TAGGED_EVERY == 0)
        assert "finished 20" in line


def test_the_rows_it_did_not_show_are_named_with_the_offset_that_reads_them(board):
    said = _seen(_page(board, limit=100))
    shown = int(_read(said)[0]["shown"])

    skipped = SKIPPED.search(said)
    assert skipped and skipped.group("first") == skipped.group("offset") == str(shown)
    assert skipped.group("last") == "99"
    assert said.rstrip().endswith("Briefs are left out here: platform_get_task #NNNN reads one.")


def test_following_next_offset_reads_every_card_once_in_four_calls(board):
    seen, offset, calls = [], 0, 0
    while offset is not None and calls < MOST_CALLS:
        head, numbers = _read(_seen(_page(board, limit=100, offset=offset)))
        seen += numbers
        calls += 1
        offset = None if head["next_offset"] == "null" else int(head["next_offset"])

    assert calls <= 4
    assert len(seen) == len(set(seen)) == REVIEW and set(seen) == EVERY_NUMBER


def test_offsets_worked_out_ahead_still_reach_every_card_through_the_skipped_rows_line(board):
    offsets, seen, calls = [0, 100], set(), 0
    while offsets and calls < MOST_CALLS:
        said = _seen(_page(board, limit=100, offset=offsets.pop(0)))
        seen |= set(_read(said)[1])
        calls += 1
        skipped = SKIPPED.search(said)
        if skipped:
            offsets.append(int(skipped.group("offset")))

    assert seen == EVERY_NUMBER


def test_a_page_that_fits_is_the_same_json_as_before(board):
    page = _page(board, status="inbox")

    assert page["returned"] == 5
    assert _seen(page) == _as_today(page)


def test_a_board_summary_and_a_list_with_no_next_offset_are_unchanged(board):
    from modules.tools.discovery.handlers_analytics import board_summary

    summary = asyncio.run(board_summary(board.db, board.ws, {}))
    assert _seen(summary, "platform_board_summary") == _as_today(summary, "platform_board_summary")

    page = _page(board, limit=100)
    no_paging = {key: value for key, value in page.items() if key != "next_offset"}
    assert _seen(no_paging) == _as_today(no_paging)


def test_a_widget_visitors_page_over_budget_shows_only_what_a_visitor_may_see(board):
    from core.security.surface import WIDGET, turn_surface

    with turn_surface(WIDGET, ("chat", "tasks:read"), None):
        page = _page(board, limit=200)
    said = _seen(page)

    rows = [line for line in said.splitlines() if line.startswith("id ")]
    assert VISITOR_LEGEND in said and rows and HEADER.search(said)
    assert count_tokens(said) <= PAGE_TOKENS
    for absent in (AGENT, "medium", "risk-4", "#16", "#17", BRIEF_WORDS.strip()):
        assert absent not in said
    assert all(line.count(" | ") == 4 and "finished 20" in line for line in rows)


def test_a_smaller_max_chars_is_honoured(board):
    said = _seen(_page(board, limit=100), max_chars=1500)

    assert len(said) <= 1500
    head, numbers = _read(said)
    assert int(head["shown"]) == len(numbers) > 0 and SKIPPED.search(said)
    assert BRIEF_WORDS.strip() not in said
