"""PRD-256 P256-FIX-N15-4 (night 15, F425): "what have you done for me tonight?" lists the whole night,
and chat-history search finds what the owner said in another chat.

* 12 of 12 "what have you done tonight" asks stopped between 23:50 and 23:55. ``platform_my_actions``
  read the whole night, but each item reached the model as an indented JSON object, oldest first,
  and a chat turn keeps about 2,000 tokens of a tool's answer: the night was cut after its first
  couple of dozen items. The answer's ``record`` is now one short line each, fitted to that cut;
  what does not fit is counted, with the ``since`` that reads on. The reads take only the turns
  that hold a write, oldest first, so a cap leaves out the newest and the answer says from when.
* ``platform_search_chat_history`` returned nothing for any term all night: it read the chats of
  the workspace's active members, through ``workspace_members``, and the local edition's operator
  has no such row. It reads the asking person's own chats in this workspace now.
"""
from __future__ import annotations

import asyncio
import inspect
import re
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest

from consumers.chatbot.narration import reply_parts
from consumers.chatbot.receipts import receipt
from core.context_guard import truncate_to_token_budget
from modules.tools.discovery import handlers_my_actions, my_actions_reads
from modules.tools.discovery.handlers_search import NO_PERSON_CHATS, chat_search_terms
from modules.tools.discovery.my_actions_window import window_of
from modules.tools.execution.tool_loop import ToolLoopExecutor
from modules.tools.formatting.result_formatter import ToolResultFormatter
from tests.test_prd256_fix_what_auto_did import _chat, _person

ACTION = "platform_my_actions"
SEARCH = "platform_search_chat_history"
MOVE = "platform_update_task_status"
TURNS = 40                     # three writes each: 120 receipts
LINES = 3 * TURNS + 2          # and the owner's two clicks
DAY_HEADING = re.compile(r"^\d{1,2} [A-Z][a-z]{2}:$")
# What a chat turn keeps of one tool's answer (consumers/chatbot/service.py runs the loop with it).
CHAT_TOOL_TOKENS = inspect.signature(ToolLoopExecutor.__init__).parameters["content_truncate_tokens"].default
SAID_IN_STOCK = "Can you ask Quay Coffee House about the 30-day terms from November?"


def _turn_writes(index: int) -> list:
    """One turn's three writes: a card made, the card moved to Done, then an agent or a memory."""
    number = f"#{1000 + index}"
    third = (receipt("platform_create_agent", {"name": f"Scout {index}"},
                     {"success": True, "agent_id": 500 + index, "name": f"Scout {index}"}) if index % 2 == 0 else
             receipt("platform_store_memory", {"content": "Prices go up on 1 Nov"}, {"success": True, "message": "Stored"}))
    return [receipt("platform_create_task", {"title": f"Restock {index}"},
                    {"success": True, "number": number, "title": f"Restock {index}"}),
            receipt(MOVE, {"task_id": number, "status": "done"}, {"success": True, "number": number}),
            third]


def _message(db, ws, chat, at, parts, role="assistant"):
    from core.models.core import Message

    db.add(Message(chat_id=chat.id, workspace_id=ws, role=role, created_at=at.replace(tzinfo=None), parts=parts))
    db.flush()


@pytest.fixture
def owner(db_session, seed_workspace):
    """A workspace with its owner (no workspace_members row, as on the local edition) and Auto."""
    from core.models.core import Agent
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    db, ws = db_session, UUID(seed_workspace())
    auto = Agent(name="Auto", agent_type="system", description="", status="active", configuration={},
                 workspace_id=ws, created_by="test", owner_type="workspace", owner_id=str(ws), is_system_agent=True)
    db.add(auto)
    db.flush()
    executor = PlatformActionExecutor(db, ws)
    executor._full_autonomy = lambda: True
    return NS(db=db, ws=ws, owner=_person(db, "owner"), auto=auto.id, executor=executor)


def _night(owner, start):
    """40 turns of three writes, evenly from ``start`` to now, in two chats, then two of the owner's clicks."""
    from core.services.approval_grants import grant_grant
    from modules.tools.execution.tool_grants import issue_tool_grant

    step = (datetime.now(timezone.utc) - start) / (TURNS + 2)
    chats = [_chat(owner.db, owner.ws, owner.owner, title) for title in ("Stock", "Wholesale")]
    times = [start + step * (index + 1) for index in range(TURNS)]
    for index, at in enumerate(times):
        _message(owner.db, owner.ws, chats[index % 2], at, reply_parts("", "", "Done.", _turn_writes(index)))
    clicked_by = {"conversation_id": str(chats[0].id), "turn_id": "t-1", "driving_user_id": str(owner.owner)}
    for number in ("#1003", "#1031"):
        grant = issue_tool_grant(owner.db, owner.ws, action=MOVE, params={"task_id": number, "status": "done"},
                                 caller_context=clicked_by, question_md=f"Approve (move to Done) {number}")
        grant_grant(grant, granted_by=f"user:{owner.owner}")
    owner.db.flush()
    return NS(**vars(owner), times=times, start=start)


@pytest.fixture
def tonight(owner):
    """Tonight so far, from the window's start."""
    window, _ = window_of({"window": "tonight"})
    return _night(owner, window.start)


@pytest.fixture
def evening(owner):
    """The last 200 minutes, a few minutes between turns (so each line has its own time)."""
    return _night(owner, datetime.now(timezone.utc) - timedelta(minutes=200))


def _ask(who, params, action=ACTION, driven=True):
    """The call as Auto's chat turn makes it: the person in the chat is the server's, never a param."""
    context = {"driving_user_id": str(who.owner), "user_id": "user_owner", "conversation_id": str(uuid4()),
               "turn_id": "t-2"} if driven else {}
    with patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)):
        return asyncio.run(who.executor.execute(action, {**params, "_agent_id": who.auto}, context))


def _what_chat_sees(out: dict) -> str:
    """The answer as Auto reads it: formatted for the model, then cut to the chat's token budget."""
    return truncate_to_token_budget(ToolResultFormatter.format_for_llm(out, ACTION), CHAT_TOOL_TOKENS)


def _lines(out):
    """The record's lines, its day headings left out."""
    return [line for line in out["record"].splitlines() if line and not DAY_HEADING.match(line)]


def _from_the_start(night):
    return (night.start - timedelta(minutes=1)).isoformat()


# ── The whole night, in what the chat keeps of the answer ─────────────────────────────────

def test_a_nights_120_writes_are_all_in_what_the_chat_keeps_oldest_and_newest(tonight):
    out = _ask(tonight, {"window": "tonight"})
    seen = _what_chat_sees(out)

    assert out["success"] is True, out
    assert out["counts"] == {"done": 3 * TURNS, "waiting": 0, "decided": 0, "clicks": 2}
    assert out["truncated"] is False and out["left_out"] is None, out["message"]
    assert "(truncated)" not in seen and out["record"].replace("\n", "\\n") in seen   # all of it, as written
    assert len(_lines(out)) == LINES
    assert _lines(out)[0].endswith(" #1000")                           # the oldest first
    assert f"'Scout {TURNS - 2}'" in seen and seen.count("#1039") == 2   # the newest: made, then moved
    assert seen.count("you approved the card: ") == 2                  # with the owner's clicks


def test_the_record_is_oldest_first_one_change_a_line_each_at_its_time(evening):
    lines = _lines(_ask(evening, {"since": _from_the_start(evening)}))

    assert [line.split(" ", 1)[0] for line in lines[:3]] == [evening.times[0].strftime("%H:%M")] * 3
    assert lines[1] == f"{evening.times[0]:%H:%M} moved to Done #1000"
    assert lines[2].endswith("'Scout 0'") and lines[3].endswith(" #1001")
    assert all(" you approved the card: Approve (move to Done) #10" in line for line in lines[-2:])
    assert all(re.match(r"^\d\d:\d\d ", line) for line in lines)


# ── What does not fit is counted, with the since that reads on; nothing repeats ───────────

def test_what_does_not_fit_is_counted_and_since_reads_the_rest_with_nothing_repeated(evening, monkeypatch):
    full = _lines(_ask(evening, {"since": _from_the_start(evening)}))
    monkeypatch.setattr(handlers_my_actions, "ANSWER_TOKENS", 900)

    pages, since = [], _from_the_start(evening)
    for _page in range(LINES):
        out = _ask(evening, {"since": since})
        pages.append(out)
        if out["left_out"] is None:
            break
        since = out["left_out"]["since"]

    first = pages[0]
    assert len(pages) > 1 and first["truncated"] is True
    assert first["counts"]["done"] == 3 * TURNS                        # the counts are the whole window's
    assert f"{first['left_out']['count']} more, from " in first["message"]
    assert f"since={first['left_out']['since']}" in first["message"]
    assert first["left_out"]["count"] == LINES - len(_lines(first))
    assert [line for page in pages for line in _lines(page)] == full   # every line once, in order


def test_a_cap_on_the_turns_read_leaves_out_the_newest_and_says_from_when(evening, monkeypatch):
    monkeypatch.setattr(my_actions_reads, "MAX_MESSAGES", 10)

    out = _ask(evening, {"since": _from_the_start(evening)})

    assert out["truncated"] is True and out["left_out"]["count"] == 0
    assert datetime.fromisoformat(out["left_out"]["since"]) == evening.times[10]   # the first turn not read
    assert _lines(out)[0].endswith(" #1000") and len(_lines(out)) == 30           # the oldest ten turns
    assert "more, from " in out["message"] and "you approved" not in out["record"]  # the clicks come after
    rest = _ask(evening, {"since": out["left_out"]["since"]})
    assert _lines(rest)[0].endswith(" #1010")


def test_turns_with_no_write_are_not_read_against_the_cap(owner, monkeypatch):
    monkeypatch.setattr(my_actions_reads, "MAX_MESSAGES", 5)
    chat, start = _chat(owner.db, owner.ws, owner.owner, "Board"), datetime.now(timezone.utc) - timedelta(hours=1)
    looked = [receipt("platform_list_tasks", {}, {"success": True, "tasks": []})]
    for minute in range(10):
        _message(owner.db, owner.ws, chat, start + timedelta(minutes=minute), reply_parts("", "", "Here.", looked))
    for minute in range(10, 13):
        _message(owner.db, owner.ws, chat, start + timedelta(minutes=minute),
                 reply_parts("", "", "Done.", _turn_writes(minute)[:1]))

    out = _ask(owner, {"since": (start - timedelta(minutes=1)).isoformat()})

    assert out["counts"]["done"] == 3 and out["left_out"] is None
    lines = _lines(out)                                               # one run of the same change, one line
    assert len(lines) == 1 and all(number in lines[0] for number in ("#1010", "#1011", "#1012")), lines
    assert lines[0].count(", ") == 2


# ── platform_search_chat_history finds what the owner said in another chat ────────────────

@pytest.fixture
def said(owner, seed_workspace):
    """The owner's words in their 'Stock' chat, and the same words in a stranger's chat and in the
    owner's chat in another workspace, none of which the search may return."""
    from core.models.core import Chat

    earlier = datetime.now(timezone.utc) - timedelta(hours=3)
    stock = _chat(owner.db, owner.ws, owner.owner, "Stock")
    _chat(owner.db, owner.ws, owner.owner, "Wholesale")
    _message(owner.db, owner.ws, stock, earlier, [{"type": "text", "text": SAID_IN_STOCK}], role="user")
    theirs = _chat(owner.db, owner.ws, _person(owner.db, "stranger"), "Theirs")
    _message(owner.db, owner.ws, theirs, earlier, [{"type": "text", "text": SAID_IN_STOCK}], role="user")
    other_ws = UUID(seed_workspace())
    elsewhere = Chat(id=uuid4(), user_id=owner.owner, workspace_id=other_ws, title="Elsewhere", visibility="private")
    owner.db.add(elsewhere)
    owner.db.flush()
    _message(owner.db, other_ws, elsewhere, earlier, [{"type": "text", "text": SAID_IN_STOCK}], role="user")
    return owner


@pytest.mark.parametrize("query", ["Quay", "quay terms", "'Quay Coffee House'", "november TERMS quay"])
def test_a_term_the_owner_used_in_another_chat_is_found(said, query):
    out = _ask(said, {"query": query}, action=SEARCH)

    assert out["success"] is True, out
    assert [(hit["chat_title"], hit["role"]) for hit in out["results"]] == [("Stock", "user")]
    assert out["results"][0]["content"] == SAID_IN_STOCK and out["total"] == 1


def test_a_word_the_message_does_not_hold_finds_nothing(said):
    out = _ask(said, {"query": "Quay pricing"}, action=SEARCH)

    assert out["success"] is True and out["results"] == []


def test_a_search_with_no_person_behind_it_reads_nothing(said):
    out = _ask(said, {"query": "Quay", "_driving_user_id": said.owner}, action=SEARCH, driven=False)

    assert out["success"] is False and out["error"] == NO_PERSON_CHATS   # never a caller-supplied person


def test_the_querys_words_are_matched_literally():
    assert chat_search_terms("50% off_peak 'Quay' \"terms,\"") == ["50\\%", "off\\_peak", "Quay", "terms"]
    assert chat_search_terms(" '' ") == []
