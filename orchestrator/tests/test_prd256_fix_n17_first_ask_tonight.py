"""P256-FIX-N17-11 (F443, nights 16-17): "what did I ask you first tonight?" reads the night.

Night 16 asked it three times and night 17 four; each answer named a recent ask (night 17: 18:27,
then only the repeats, 18:31) while the night's first was 17:03, after a two-hour gap. No tool
listed the person's own messages oldest first: platform_search_chat_history required a query and
gave the newest hits of both roles, plus a second ``formatted`` copy, cut to the chat's 2,000
tokens from the head.

It now reads a window with no query (``window``, or ``since``, which overrides it), oldest or
newest first, one role or both; a message after 30 minutes or more with nothing carries the gap;
the answer fits what the chat keeps, and what it leaves out is counted with the since that reads
on. No query and no window is refused in words. Each read runs through the executor's whole stack,
as Auto's chat turn calls it (the person in the chat is the server's, never a param).
"""
from __future__ import annotations

import asyncio
import inspect
from datetime import datetime, time, timedelta, timezone
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest

from core.context_guard import truncate_to_token_budget
from modules.tools.discovery import chat_history_read as reads
from modules.tools.execution.tool_loop import ToolLoopExecutor
from modules.tools.formatting.result_formatter import ToolResultFormatter
from tests.test_prd256_fix_what_auto_did import _chat, _person

SEARCH = "platform_search_chat_history"
CHAT_TOOL_TOKENS = inspect.signature(ToolLoopExecutor.__init__).parameters["content_truncate_tokens"].default
FIRST = "Morning stock check: how much Guji is left?"
HELLO = "Hello Auto."
ASKED = "What did I ask you first tonight?"
# Two days back, so every time below is in the past whenever CI runs.
DAY = datetime.combine((datetime.now(timezone.utc) - timedelta(days=2)).date(), time(0), tzinfo=timezone.utc)


def _at(hour, minute=0):
    return DAY + timedelta(hours=hour, minutes=minute)


def _message(owner, chat, at, said, role="user"):
    from core.models.core import Message

    owner.db.add(Message(chat_id=chat.id, workspace_id=owner.ws, role=role, created_at=at.replace(tzinfo=None),
                         parts=[{"type": "text", "text": said}]))
    owner.db.flush()


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
    person = _person(db, "owner")
    return NS(db=db, ws=ws, owner=person, auto=auto.id, executor=executor,
              chats=[_chat(db, ws, person, title) for title in ("Stock", "Wholesale")])


@pytest.fixture
def night16(owner):
    """Night 16's shape: one ask before 18:00, then 18:29, 18:50, 19:23, every five minutes from 19:30
    to 22:00, and 30 repeats after; a reply after each of the first three."""
    _message(owner, owner.chats[0], _at(17, 40), "Before the night: is the roaster booked?")
    for at, said in ((_at(18, 29), FIRST), (_at(18, 50), "Draft the Quay quote."), (_at(19, 23), "Any new cards?")):
        _message(owner, owner.chats[0], at, said)
        _message(owner, owner.chats[0], at + timedelta(minutes=1), "Done.", role="assistant")
    for step in range(31):
        _message(owner, owner.chats[step % 2], _at(19, 30) + timedelta(minutes=5 * step), f"Ask {step}")
    for step in range(30):
        _message(owner, owner.chats[1], _at(22, 1) + timedelta(minutes=step), ASKED)
    return owner


def _read(who, params, driven=True):
    """The call as Auto's chat turn makes it."""
    context = {"driving_user_id": str(who.owner), "user_id": "user_owner", "conversation_id": str(uuid4()),
               "turn_id": "t-1"} if driven else {}
    with patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)):
        return asyncio.run(who.executor.execute(SEARCH, {**params, "_agent_id": who.auto}, context))


def _times(out):
    return [hit["date"][-9:-4] for hit in out["results"]]


# ── The first ask of the night, oldest first, the person's own ──────────────────────────

def test_the_first_ask_after_six_is_read_oldest_first(night16):
    out = _read(night16, {"since": _at(18).isoformat(), "order": "oldest", "role": "user", "limit": 3})

    assert out["success"] is True, out
    assert _times(out) == ["18:29", "18:50", "19:23"] and out["results"][0]["content"] == FIRST
    assert {hit["role"] for hit in out["results"]} == {"user"}
    assert "gap" not in out["results"][0] and "gap" not in out["results"][1]          # 21 min: no gap line
    assert out["results"][2]["gap"] == "after 33 min with nothing"
    assert out["left_out"] == {"count": 3 + 31 + 30 - 3, "since": _at(19, 30).isoformat()}
    assert f"since={_at(19, 30).isoformat()}" in out["message"] and "formatted" not in out
    rest = _read(night16, {"since": out["left_out"]["since"], "order": "oldest", "role": "user", "limit": 2})
    assert _times(rest) == ["19:30", "19:35"] and "gap" not in rest["results"][1]      # 5 min apart


def test_newest_first_is_the_default_and_counts_what_is_before(night16):
    out = _read(night16, {"since": _at(18).isoformat(), "role": "user", "limit": 2})

    assert out["order"] == "newest" and _times(out) == ["22:30", "22:29"]
    assert out["left_out"] == {"count": 62, "before": _at(22, 29).isoformat()}


def test_a_gap_of_thirty_minutes_is_marked(owner):
    for at, said in ((_at(14), "Afternoon: what's on the board?"), (_at(14, 10), "Thanks."),
                     (_at(15), "One more thing."), (_at(17, 3) + timedelta(seconds=28), HELLO),
                     (_at(17, 5), "What needs me?")):
        _message(owner, owner.chats[0], at, said)

    out = _read(owner, {"since": _at(12).isoformat(), "order": "oldest", "role": "user"})

    hits = out["results"]
    assert [hit["content"] for hit in hits][2:4] == ["One more thing.", HELLO]
    assert hits[2]["gap"] == "after 50 min with nothing" and hits[3]["gap"] == "after 2 h 3 min with nothing"
    assert list(hits[3])[0] == "gap" and "gap" not in hits[4]                      # the gap comes first
    assert out["left_out"] is None and out["total"] == 5


def test_tonight_starts_at_six_utc_and_says_so():
    read, wrong = reads.read_asked({"window": "tonight", "role": "user", "order": "oldest"}, now=_at(22, 14))

    assert wrong is None and read.window.start == _at(18) and read.window.end == _at(22, 14)
    said = reads.answer(read, [], 0)["message"]
    assert said.startswith("No messages you sent tonight (from 18:00 on ") and "UTC" in said
    early = reads.read_asked({"window": "tonight"}, now=_at(1, 5))[0]
    assert early.window.start == _at(18) - timedelta(days=1)                       # the small hours: last evening


# ── What cannot be read is refused in words ─────────────────────────────────────────────

def test_no_query_and_no_window_is_refused_in_words(owner):
    out = _read(owner, {})

    assert out["success"] is False and out["error"].startswith("Say what to read: a query")
    assert "tonight" in out["error"] and out["error"].endswith("Nothing was read.")


@pytest.mark.parametrize("params, said", [
    ({"window": "tonight", "role": "owner"}, "role must be user or assistant; got 'owner'. Nothing was read."),
    ({"window": "tonight", "order": "first"}, "order must be newest or oldest; got 'first'. Nothing was read."),
    ({"window": "fortnight"}, "window must be one of: tonight, today, yesterday, last_24_hours, this_week"),
    ({"since": "last Tuesday"}, "since must be a time like 2026-10-12T18:00:00Z; got 'last Tuesday'"),
])
def test_a_bad_role_or_order_is_refused(owner, params, said):
    out = _read(owner, params)

    assert out["success"] is False and out["error"] == said


def test_a_read_with_no_person_behind_it_reads_nothing(night16):
    out = _read(night16, {"window": "last_24_hours", "_driving_user_id": night16.owner}, driven=False)

    assert out["success"] is False and out["error"] == reads.NO_PERSON_CHATS


# ── The answer fits what the chat keeps ─────────────────────────────────────────────────

def test_the_answer_fits_the_chat_budget(owner):
    start = _at(9)
    for minute in range(240):
        _message(owner, owner.chats[minute % 2], start + timedelta(minutes=minute),
                 f"Ask {minute:03d}: " + "please check the wholesale terms and the Guji stock again " * 6)

    out = _read(owner, {"since": start.isoformat(), "order": "oldest", "limit": 100})
    seen = truncate_to_token_budget(ToolResultFormatter.format_for_llm(out, SEARCH), CHAT_TOOL_TOKENS)

    assert out["success"] is True and "formatted" not in out
    assert "(truncated)" not in seen and out["results"][0]["content"].startswith("Ask 000")   # the oldest is in
    assert 0 < len(out["results"]) < 100 and out["left_out"]["count"] == 240 - len(out["results"])
    assert all(hit["content"][:8] in seen for hit in out["results"])                           # every one shown
    assert out["left_out"]["since"] == (start + timedelta(minutes=len(out["results"]))).isoformat()


# ── The dispatcher runs a windowed read with no query, direct or through platform_execute ─

@pytest.fixture
def dispatcher(monkeypatch):
    from modules.tools.execution import unified_executor
    from modules.tools.execution.unified_executor import UnifiedToolExecutor

    ran = []

    async def _run(self, action_name, params, **kwargs):
        ran.append((action_name, params))
        return {"success": True}

    monkeypatch.setattr(UnifiedToolExecutor, "_policy_gate_check", lambda self, *a, **k: None)
    monkeypatch.setattr(UnifiedToolExecutor, "_execute_platform_action", _run)
    monkeypatch.setattr(unified_executor, "fire_telemetry", lambda **kwargs: None)
    executor = UnifiedToolExecutor.__new__(UnifiedToolExecutor)
    executor.composio_actions, executor.db = {}, None

    def call(tool, params):
        return asyncio.run(executor.execute_tool(tool, params, agent_id=322, workspace_id=str(uuid4()),
                                                 trace_id="t-f443"))
    return NS(call=call, ran=ran)


@pytest.mark.parametrize("via", ["direct", "platform_execute"])
def test_the_dispatcher_runs_a_windowed_read_without_a_query(dispatcher, via):
    asked = {"window": "last_24_hours", "since": _at(17).isoformat(), "order": "oldest", "role": "user", "limit": 5}
    call = (SEARCH, asked) if via == "direct" else ("platform_execute", {"action": SEARCH, "params": asked})

    result = dispatcher.call(*call)

    assert result["success"] is True, result
    ((ran_as, params),) = dispatcher.ran
    assert ran_as == SEARCH and {k: v for k, v in params.items() if not k.startswith("_")} == asked
