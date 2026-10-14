"""PRD-256 P256-FIX-N14-3 (night 14, F417): Auto has a record of what it did.

Night 14, 04:43: "Tonight I've only read things. I haven't sent, created, changed or scheduled
anything", after Auto had made an agent, a skill, five cards, a Gmail draft, a memory and a
playbook run that night; "what did I decide about #…" found 1 click of 8. The receipts were saved
with each message and each click on its grant, and nothing read them across chats.

``platform_my_actions`` reads them: the asking person's chats' done and waiting writes and their
clicks, each card by its board number, for a window or one card (``ticket``). A click's memory
names its card's number even when the card's words do not. P256-FIX-N15-4: the answer's ``record``
gives them one line each, oldest first ("23:01 card made #1588"), under its day's heading.
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

from consumers.chatbot.narration import reply_parts
from consumers.chatbot.receipts import receipt
from modules.tools.discovery.handlers_my_actions import NO_PERSON
from modules.tools.discovery.my_actions_window import (
    LAST_DAY, MAX_LOOKBACK_DAYS, TONIGHT, TODAY, YESTERDAY, window_of,
)
from services.click_memory import APPROVED, decision_memory, remember_the_click

ACTION = "platform_my_actions"
DAY_HEADING = re.compile(r"^\d{1,2} [A-Z][a-z]{2}:$")
MOVE = "platform_update_task_status"
NOW = datetime.now(timezone.utc)


class _Memory:
    """The long-term store the store-memory handler writes to: each write recorded."""

    is_durable_configured = True

    def __init__(self):
        self.stored = []

    async def store_long_term(self, workspace_id, content, agent_id=None, metadata=None, **_):
        self.stored.append(NS(content=content, metadata=metadata or {}))
        return {"facts_extracted": 1}

    def namespace(self, workspace_id):
        return NS(resolve=lambda agent_id: f"ws_{workspace_id}")


def _person(db, name):
    return db.execute(text("INSERT INTO users (email, username) VALUES (:e, :u) RETURNING id"),
                      {"e": f"{name}-{uuid4().hex[:8]}@harbourline.test", "u": f"{name}-{uuid4().hex[:8]}"}).scalar()


def _chat(db, ws, user_id, title):
    from core.models.core import Chat

    row = Chat(id=uuid4(), user_id=user_id, workspace_id=ws, title=title, visibility="private")
    db.add(row)
    db.flush()
    return row


def _turn(db, ws, chat, receipts, minutes_ago):
    """One assistant message as a chat turn saves it: its receipts part, then its answer."""
    from core.models.core import Message

    at = (NOW - timedelta(minutes=minutes_ago)).replace(tzinfo=None)
    db.add(Message(chat_id=chat.id, workspace_id=ws, role="assistant", created_at=at,
                   parts=reply_parts("", "", "Done.", receipts)))
    db.flush()


@pytest.fixture
def night(db_session, seed_workspace):
    """Two of the owner's chats with writes (a card by number, a card by its link, an agent, a
    memory; a read and a refused write beside them), a stranger's chat with a write, the owner's
    click on a third card, the stranger's click and a pending card."""
    from core.models.core import Agent, BoardTask
    from core.services.approval_grants import grant_grant
    from modules.tools.discovery.platform_executor import PlatformActionExecutor
    from modules.tools.execution.tool_grants import issue_tool_grant

    db, ws = db_session, UUID(seed_workspace())
    owner, stranger = _person(db, "owner"), _person(db, "stranger")
    auto = Agent(name="Auto", agent_type="system", description="", status="active", configuration={},
                 workspace_id=ws, created_by="test", owner_type="workspace", owner_id=str(ws), is_system_agent=True)
    made, linked, clicked = (BoardTask(workspace_id=ws, title=t, status="inbox", source_type="user")
                             for t in ("Autumn tier launch copy", "Supplier follow-up", "Spring menu"))
    db.add_all([auto, made, linked, clicked])
    db.flush()
    number = {card.id: f"#{card.workspace_seq:04d}" for card in (made, linked, clicked)}
    first, second = _chat(db, ws, owner, "Launch"), _chat(db, ws, owner, "Notes")
    _turn(db, ws, first, [
        receipt("platform_create_task", {"title": made.title},
                {"success": True, "task_id": made.id, "number": number[made.id], "title": made.title}),
        receipt("platform_create_agent", {"name": "Scout"}, {"success": True, "agent_id": 12, "name": "Scout"}),
        receipt("platform_list_tasks", {}, {"success": True, "tasks": []}),
        receipt("platform_create_skill", {"name": "Refunds"}, {"success": False, "error": "Missing required params"}),
    ], minutes_ago=50)
    _turn(db, ws, second, [
        receipt("platform_store_memory", {"content": "Prices go up on 1 Nov"}, {"success": True, "message": "Stored"}),
        receipt("platform_update_task", {"title": linked.title}, {"success": True, "task_id": linked.id}),
    ], minutes_ago=20)
    _turn(db, ws, _chat(db, ws, stranger, "Theirs"), [
        receipt("platform_create_agent", {"name": "Ghost"}, {"success": True, "agent_id": 13, "name": "Ghost"}),
    ], minutes_ago=10)
    clicked_by = {"conversation_id": str(first.id), "turn_id": "t-1", "driving_user_id": str(owner)}
    grant = issue_tool_grant(db, ws, action=MOVE, params={"task_id": number[clicked.id], "status": "done"},
                             caller_context=clicked_by, question_md="Approve (move to Done) 'Spring menu':\n- note: ship")
    grant_grant(grant, granted_by=f"user:{owner}")
    theirs = issue_tool_grant(db, ws, action=MOVE, params={"task_id": number[made.id], "status": "done"},
                              caller_context=clicked_by, question_md="Approve (move to Done) 'Autumn tier launch copy'")
    grant_grant(theirs, granted_by=f"user:{stranger}")
    issue_tool_grant(db, ws, action=MOVE, params={"task_id": number[linked.id], "status": "done"},
                     caller_context=clicked_by, question_md="Approve (move to Done) 'Supplier follow-up'")
    db.flush()
    executor = PlatformActionExecutor(db, ws)
    executor._full_autonomy = lambda: True
    return NS(db=db, ws=ws, owner=owner, auto=auto.id, executor=executor, grant=grant, number=number,
              made=made, linked=linked, clicked=clicked)


def _ask(night, params, driver="owner"):
    """The read as Auto's chat turn makes it: the person in the chat is the server's, never a param."""
    context = {"driving_user_id": str(night.owner), "user_id": "user_owner", "conversation_id": str(uuid4()),
               "turn_id": "t-2"} if driver == "owner" else {}
    with patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)):
        return asyncio.run(night.executor.execute(ACTION, {**params, "_agent_id": night.auto}, context))


def _since(hours=2):
    return (NOW - timedelta(hours=hours)).isoformat()


def _lines(out):
    """The record's lines, its day headings left out ("13 Oct:")."""
    return [line for line in out["record"].splitlines() if line and not DAY_HEADING.match(line)]


# ── Two chats with writes and one click: all four, each card by its number ───────────────

def test_the_read_lists_the_writes_of_both_chats_and_the_click_with_board_numbers(night):
    out = _ask(night, {"since": _since()})

    assert out["success"] is True, out
    lines = _lines(out)                                               # oldest first: both chats, then the click
    assert len(lines) == 5, out["record"]
    assert lines[0].endswith(night.number[night.made.id])             # by the number its answer gave
    assert lines[1].endswith("'Scout'")
    assert lines[3].endswith(night.number[night.linked.id])           # by the card its link opens
    assert "you approved the card: Approve (move to Done) 'Spring menu'" in lines[4]
    assert lines[4].count(night.number[night.clicked.id]) == 1         # the click's card, by its number
    assert all(re.match(r"^\d\d:\d\d ", line) for line in lines)      # each at its time
    assert "task_id" not in str(out) and f"#{night.linked.id} " not in out["record"]   # a row id is never shown
    assert out["counts"] == {"done": 4, "waiting": 0, "decided": 0, "clicks": 1}
    assert out["truncated"] is False and out["left_out"] is None
    assert out["message"].endswith("4 done, 0 waiting for your click, 0 asked and since decided, 1 decided by you.")
    assert "Ghost" not in str(out) and "Autumn tier launch copy" not in out["record"]  # not the stranger's


def _asked_to_delete(night, minutes_ago=5):
    """A turn in the owner's 'Notes' chat that raised a card to delete Scout (its grant, pending)."""
    from core.models.core import Chat
    from modules.tools.execution.tool_grants import issue_tool_grant

    asked = {"success": False, "requires_confirmation": True, "owner_only": True, "action": "platform_delete_agent",
             "act": "delete an agent Scout", "message": "This waits for the owner's click."}
    chat = night.db.query(Chat).filter(Chat.user_id == night.owner, Chat.title == "Notes").one()
    _turn(night.db, night.ws, chat, [receipt("platform_delete_agent", {"name": "Scout"}, asked)], minutes_ago)
    grant = issue_tool_grant(night.db, night.ws, action="platform_delete_agent", params={"agent_id": 12},
                             caller_context={"conversation_id": str(chat.id), "driving_user_id": str(night.owner)},
                             question_md="Delete an agent 'Scout' (agent #12)")
    grant.requested_at = NOW - timedelta(minutes=minutes_ago + 1)   # asked in the turn, before it was saved
    night.db.flush()
    return grant


def test_a_write_waiting_for_the_click_is_listed_as_waiting(night):
    _asked_to_delete(night)

    out = _ask(night, {"since": _since()})

    assert len([line for line in _lines(out) if "waiting for your click: " in line]) == 1
    assert out["counts"]["waiting"] == 1 and out["counts"]["decided"] == 0


@pytest.mark.parametrize("decide, decision", [("grant", "approved"), ("deny", "declined")])
def test_a_waiting_write_whose_card_was_since_decided_no_longer_waits(night, decide, decision):
    from core.services.approval_grants import deny_grant, grant_grant

    grant = _asked_to_delete(night)
    (grant_grant(grant, granted_by=f"user:{night.owner}") if decide == "grant"
     else deny_grant(grant, revoked_by=f"user:{night.owner}"))
    night.db.flush()

    out = _ask(night, {"since": _since()})

    assert out["counts"]["waiting"] == 0 and out["counts"]["decided"] == 1
    assert "waiting for your click" not in out["record"]
    assert [line for line in _lines(out) if line[6:].startswith(f"asked your click, you {decision}: ")]


def test_a_decision_on_a_board_task_is_no_click_on_a_call_card(night):
    from core.models.approval_grants import SUBJECT_BOARD_TASK
    from core.services.approval_grants import create_grant, grant_grant

    other = create_grant(night.db, night.ws, subject_type=SUBJECT_BOARD_TASK, subject_id=str(night.made.id),
                         tool_name="platform_update_task_status")
    grant_grant(other, granted_by=f"user:{night.owner}")
    night.db.flush()

    out = _ask(night, {"since": _since()})

    assert out["counts"]["clicks"] == 1 and len([line for line in _lines(out) if " you approved " in line]) == 1


# ── 'What did I decide about #<n>' finds the click ───────────────────────────────────────

def test_what_did_i_decide_about_a_card_finds_the_click_by_its_number(night):
    said = night.number[night.clicked.id]

    out = _ask(night, {"ticket": said})                                # no window: the lookback

    assert out["counts"] == {"done": 0, "waiting": 0, "decided": 0, "clicks": 1}
    assert _lines(out) and " you approved the card: " in _lines(out)[0] and said in _lines(out)[0]
    assert out["message"].startswith("In the last 30 days") and f"about card {said}" in out["message"]
    assert _ask(night, {"ticket": said.lstrip("#").lstrip("0")})["record"] == out["record"]  # "1437" is #1437


def test_a_ticket_narrows_the_writes_to_that_card(night):
    out = _ask(night, {"since": _since(), "ticket": night.number[night.linked.id]})

    assert _lines(out) == [line for line in _lines(out) if line.endswith(night.number[night.linked.id])]
    assert out["counts"] == {"done": 1, "waiting": 0, "decided": 0, "clicks": 0}


def test_the_clicks_memory_names_the_cards_number(night, monkeypatch):
    import modules.memory.unified_memory_service as memory_service

    memory = _Memory()
    monkeypatch.setattr(memory_service, "get_unified_memory_service", lambda: memory)

    stored = asyncio.run(remember_the_click(night.db, NS(workspace_id=night.ws), night.grant.id, APPROVED))

    assert stored and stored.get("success"), stored
    said = memory.stored[0].content
    assert said.startswith('The owner approved the card "Approve (move to Done) \'Spring menu\'" (card ')
    assert f"(card {night.number[night.clicked.id]})" in said


def test_a_number_the_cards_words_show_is_not_said_twice():
    grant = NS(id=3, question_md="Approve (move to Done) #1403 'Autumn tier launch copy':", tool_name=MOVE,
               details={}, granted_at=datetime(2026, 10, 12, 9, tzinfo=timezone.utc), revoked_at=None)

    assert decision_memory(grant, APPROVED, ["#1403"]).count("#1403") == 1
    assert '" (card #140) on' in decision_memory(grant, APPROVED, ["#140"])     # #1403 is not #140
    assert '" (card #1437) on 12 Oct 2026' in decision_memory(grant, APPROVED, ["#1437"])


# ── A window with no writes says so; no person, no read ──────────────────────────────────

def test_a_window_with_no_writes_says_so(night):
    # The real clock, not NOW (import time): the fixture's click is stamped when it is made,
    # which on a slow shard is minutes after the module was imported.
    out = _ask(night, {"since": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()})

    assert out["success"] is True and out["record"] == "" and not any(out["counts"].values())
    assert out["message"].startswith("Nothing since ") and "no writes of mine" in out["message"]
    assert out["note"] is None


def test_a_call_with_no_person_behind_it_reads_nothing(night):
    out = _ask(night, {"since": _since(), "_driving_user_id": night.owner}, driver=None)

    assert out["success"] is False and out["error"] == NO_PERSON           # never a caller-supplied person


def test_a_ticket_that_is_no_number_is_refused(night):
    out = _ask(night, {"ticket": "the spring one"})

    assert out["success"] is False and "board number" in out["error"]


# ── The windows, as the owner says them ───────────────────────────────────────────────────

@pytest.mark.parametrize("now, named, start, end", [
    (datetime(2026, 10, 13, 4, 43, tzinfo=timezone.utc), TONIGHT, datetime(2026, 10, 12, 18), datetime(2026, 10, 13, 4, 43)),
    (datetime(2026, 10, 12, 21, 0, tzinfo=timezone.utc), TONIGHT, datetime(2026, 10, 12, 18), datetime(2026, 10, 12, 21)),
    (datetime(2026, 10, 13, 4, 43, tzinfo=timezone.utc), TODAY, datetime(2026, 10, 13), datetime(2026, 10, 13, 4, 43)),
    (datetime(2026, 10, 13, 4, 43, tzinfo=timezone.utc), YESTERDAY, datetime(2026, 10, 12), datetime(2026, 10, 13)),
    (datetime(2026, 10, 13, 4, 43, tzinfo=timezone.utc), LAST_DAY, datetime(2026, 10, 12, 4, 43), datetime(2026, 10, 13, 4, 43)),
])
def test_each_window_is_read_as_the_owner_means_it(now, named, start, end):
    window, wrong = window_of({"window": named}, now)

    assert wrong is None
    assert (window.start, window.end) == (start.replace(tzinfo=timezone.utc), end.replace(tzinfo=timezone.utc))


def test_a_window_or_since_it_cannot_read_is_refused():
    assert window_of({"window": "fortnight"})[0] is None
    assert window_of({"since": "last tuesday"})[1].startswith("since must be a time")
    assert window_of({"since": "0001-01-01T00:00:00+05:00"})[0] is None      # out of range, not a crash
    far, _ = window_of({"since": "2001-01-01T00:00:00Z"}, NOW)
    assert far.start == NOW - timedelta(days=MAX_LOOKBACK_DAYS)              # never further back than that


# ── Auto finds it, and the reply's receipts show the read ─────────────────────────────────

def test_the_read_is_registered_for_the_questions_it_answers():
    from modules.tools.discovery.action_registry import get_action_registry

    action = get_action_registry().get(ACTION)

    assert action is not None and action.permission_level == "read"
    assert "what have you done tonight?" in action.examples and "what did I decide about #1437?" in action.examples


def test_a_recap_given_with_each_items_time_is_not_taken_for_this_replys_claims(night):
    from consumers.chatbot.claims_backed import unbacked_claims

    out = _ask(night, {"since": _since()})
    made = _lines(out)[1]                                             # "23:01 agent made 'Scout'"
    recap = f"At {made.split()[0]} I created the agent Scout."

    assert made.endswith("'Scout'") and unbacked_claims(recap, []) == []
    assert unbacked_claims("I created the agent Scout.", []) != []           # with no time, still a claim


def test_the_replys_receipts_show_the_read():
    out = {"success": True, "title": "what I did and what you decided, tonight", "done": [], "clicks": []}

    said = receipt(ACTION, {"window": TONIGHT}, out)

    assert (said["kind"], said["status"]) == ("read", "done")
    assert said["subject"] == "what I did and what you decided, tonight"
