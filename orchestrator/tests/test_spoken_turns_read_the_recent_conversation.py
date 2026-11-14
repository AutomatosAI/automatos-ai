"""Spoken turns read the recent conversation (voice latency C-4).

A turn sent with ``spokenMode`` read its whole chat twice before its first model call: once in
the route, for the lane, and once in the detached turn. A quick-ask lands in the person's typed
chat, which can be long. With ``SPOKEN_MODE_HISTORY_MESSAGES`` at 4 or more it reads that many
of the chat's newest messages instead.

* ``spoken_window`` is the count for a spoken turn and None (the whole chat) for a typed turn,
  the setting at 0, or a value under ``MIN_SPOKEN_WINDOW``.
* ``_recent_rows`` asks for the newest rows, newest first with a limit, and hands them back
  oldest first.
* The turn's read (it has its saved message's id) starts at a person's message and ends with
  its own; when its message is not in the window it reads the whole chat, as before.
* The route's read (no id yet) is the window as it is.
* Through the real route: a spoken turn for a chosen agent makes one read (the lane of a chosen
  agent uses none of it); a typed turn, and any turn with the setting off, reads twice with
  ``recent=None``, as before.

DB-free: the session is a fake that keeps what it was asked.
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from api import chat, chat_turn_history
from api.chat_models import ChatMessageRequest, ChatRequest, MessagePart
from config import config
from consumers.chatbot.model_history import FAILED_TURN_MARKER
from consumers.chatbot.streaming import StreamingHandler
from consumers.chatbot.turn_errors import TURN_ERROR_ORIGIN
from core.models import Message

ORCH = Path(__file__).resolve().parents[1]
WS = "6a0f3c1e-2b4d-4e8f-9a1b-c40000000001"
CHAT_ID = "6a0f3c1e-2b4d-4e8f-9a1b-c4000000c4a7"
USER_PK = 7
AUTO = 11
CHOSEN = 23
WINDOW = 12
NEWEST = "What did we decide about pricing?"
SAVED_ID = "the-turns-own-message"
NEWEST_FIRST = str(Message.created_at.desc())


# --------------------------------------------------------------------------- a saved chat


class _Query:
    """One ``db.query(Message)``: what it was filtered, ordered and limited by is kept."""

    def __init__(self, db: "_Db") -> None:
        self.db, self.order, self.count = db, "", None

    def filter(self, *criteria):
        self.db.filters.append(criteria)
        return self

    def order_by(self, *clauses):
        self.order = ", ".join(str(clause) for clause in clauses)
        self.db.orders.append(self.order)
        return self

    def limit(self, count):
        self.count = count
        self.db.limits.append(count)
        return self

    def all(self):
        rows = list(reversed(self.db.chat.rows)) if self.order == NEWEST_FIRST else list(self.db.chat.rows)
        return rows if self.count is None else rows[:self.count]


class _Db:
    def __init__(self, chat: "_SavedChat") -> None:
        self.chat, self.queried, self.filters, self.orders, self.limits = chat, [], [], [], []

    def query(self, model):
        self.queried.append(model)
        return _Query(self)


class _SavedChat:
    """A chat's saved messages, oldest first, behind the two reads a turn can make: the bounded
    one on its session (``db``) and the whole chat (``get_messages_by_chat_id``)."""

    def __init__(self) -> None:
        self.rows, self.whole_reads = [], 0
        self.db = _Db(self)

    def add(self, role: str, text: str, source=None, id=None):
        row = NS(id=id or f"m{len(self.rows)}", role=role, parts=[{"type": "text", "text": text}], source=source)
        self.rows.append(row)
        return row

    def get_messages_by_chat_id(self, chat_id, limit=None):
        self.whole_reads += 1
        return list(self.rows)


def _exchanges(chat: _SavedChat, count: int) -> _SavedChat:
    """``count`` answered asks: person, Auto, person, Auto ..."""
    for index in range(count):
        chat.add("user", f"ask {index}")
        chat.add("assistant", f"reply {index}")
    return chat


def _long_chat() -> _SavedChat:
    """Twenty answered asks, then the turn's own message: 41 rows."""
    chat = _exchanges(_SavedChat(), 20)
    chat.add("user", NEWEST, id=SAVED_ID)
    return chat


def _read(chat: _SavedChat, message_id=None, recent=None):
    newest = ChatMessageRequest(parts=[MessagePart(type="text", text=NEWEST)])
    return chat_turn_history.turn_history(chat, CHAT_ID, newest, {}, message_id, recent=recent)


def _said(history) -> list:
    return [(entry["role"], entry["parts"][0]["text"]) for entry in history]


# --------------------------------------------------------------------------- the setting


def test_the_setting_is_zero_unless_the_environment_sets_it():
    source = (ORCH / "config.py").read_text()
    surface = json.loads((ORCH / "reports" / "config-surface.json").read_text())

    assert 'int(os.getenv("SPOKEN_MODE_HISTORY_MESSAGES", "0"))' in source
    assert isinstance(config.SPOKEN_MODE_HISTORY_MESSAGES, int)
    assert "SPOKEN_MODE_HISTORY_MESSAGES" in json.dumps(surface)


@pytest.mark.parametrize("setting,spoken,window", [
    (12, True, 12),
    (4, True, 4),  # MIN_SPOKEN_WINDOW itself
    (12, False, None),  # typed
    (0, True, None),  # off
    (3, True, None),  # under MIN_SPOKEN_WINDOW: the lane reads the message before this one
    (0, False, None),
])
def test_the_window_is_the_setting_only_for_a_spoken_turn(monkeypatch, setting, spoken, window):
    monkeypatch.setattr(config, "SPOKEN_MODE_HISTORY_MESSAGES", setting)

    assert chat_turn_history.spoken_window(spoken) == window


def test_the_route_reaches_the_window_through_the_history_module():
    assert chat.spoken_window is chat_turn_history.spoken_window
    assert "spoken_window" in chat_turn_history.__all__
    assert chat_turn_history.MIN_SPOKEN_WINDOW == 4


# --------------------------------------------------------------------------- the bounded read


def test_the_bounded_read_asks_for_the_newest_rows_and_hands_them_back_oldest_first():
    chat = _exchanges(_SavedChat(), 20)  # 40 rows

    rows = chat_turn_history._recent_rows(chat.db, CHAT_ID, WINDOW)

    assert rows == chat.rows[-WINDOW:]
    assert chat.db.queried == [Message]
    assert chat.db.orders == [NEWEST_FIRST] and NEWEST_FIRST.endswith("created_at DESC")
    assert chat.db.limits == [WINDOW]
    assert len(chat.db.filters) == 1
    assert chat.whole_reads == 0


def test_the_bounded_read_of_a_short_chat_is_the_whole_chat():
    chat = _exchanges(_SavedChat(), 2)

    assert chat_turn_history._recent_rows(chat.db, CHAT_ID, WINDOW) == chat.rows


def test_an_id_that_is_no_uuid_reads_nothing_and_asks_nothing():
    chat = _exchanges(_SavedChat(), 2)

    assert chat_turn_history._recent_rows(chat.db, "not-a-chat-id", WINDOW) == []
    assert chat.db.queried == []


def test_leading_rows_that_are_not_a_persons_are_dropped_into_a_new_list():
    chat = _SavedChat()
    rows = [chat.add("assistant", "a post"), chat.add("assistant", "another"),
            chat.add("user", "an ask"), chat.add("assistant", "a reply")]
    given = list(rows)

    kept = chat_turn_history._from_a_persons_message(rows)

    assert kept == rows[2:] and kept is not rows
    assert rows == given
    assert chat_turn_history._from_a_persons_message(rows[:2]) == []
    assert chat_turn_history._from_a_persons_message([]) == []


# --------------------------------------------------------------------------- the turn's read


def test_the_turns_read_is_the_window_from_a_persons_message_to_its_own():
    chat = _long_chat()

    history = _read(chat, SAVED_ID, recent=WINDOW)

    # The window is rows 29..40; row 29 is Auto's reply to ask 14, so the read starts at ask 15.
    assert len(history) == WINDOW - 1
    assert _said(history)[0] == ("user", "ask 15")
    assert _said(history)[-1] == ("user", NEWEST)
    assert chat.db.limits == [WINDOW]
    assert chat.whole_reads == 0


def test_a_failed_turns_note_inside_the_window_is_the_marker():
    chat = _exchanges(_SavedChat(), 10)
    chat.add("user", "Research agentfactory")
    chat.add("assistant", "⚠️ Auto could not finish this reply.", source={"origin": TURN_ERROR_ORIGIN})
    chat.add("user", NEWEST, id=SAVED_ID)

    history = _read(chat, SAVED_ID, recent=WINDOW)

    assert _said(history)[-3:] == [
        ("user", "Research agentfactory"), ("assistant", FAILED_TURN_MARKER), ("user", NEWEST),
    ]
    assert chat.whole_reads == 0


def test_a_turn_whose_message_fell_out_of_the_window_reads_the_whole_chat():
    chat = _exchanges(_SavedChat(), 1)
    chat.add("user", NEWEST, id=SAVED_ID)
    _exchanges(chat, 6)  # twelve rows saved since: the turns queued behind it, and their replies

    history = _read(chat, SAVED_ID, recent=WINDOW)

    assert chat.db.limits == [WINDOW] and chat.whole_reads == 1
    assert _said(history) == [
        ("user", "ask 0"), ("assistant", "reply 0"),
        *[("assistant", f"reply {index}") for index in range(6)],
        ("user", NEWEST),
    ]


def test_a_later_message_of_the_persons_inside_the_window_is_left_to_its_own_turn():
    chat = _exchanges(_SavedChat(), 1)
    chat.add("user", NEWEST, id=SAVED_ID)
    chat.add("user", "And one more thing")

    history = _read(chat, SAVED_ID, recent=WINDOW)

    assert _said(history) == [("user", "ask 0"), ("assistant", "reply 0"), ("user", NEWEST)]
    assert chat.whole_reads == 0


# --------------------------------------------------------------------------- the route's read


def test_the_routes_read_is_the_window_as_it_is():
    chat = _long_chat()

    history = _read(chat, recent=WINDOW)

    assert len(history) == WINDOW
    assert _said(history)[0] == ("assistant", "reply 14")  # no leading row is dropped
    assert _said(history)[-1] == ("user", NEWEST)
    assert chat.whole_reads == 0


def test_the_routes_read_of_one_post_and_one_message_has_both():
    chat = _SavedChat()
    chat.add("assistant", "Your morning brief is ready.")
    chat.add("user", NEWEST, id=SAVED_ID)

    assert len(_read(chat, recent=WINDOW)) == 2


# --------------------------------------------------------------------------- without a window


@pytest.mark.parametrize("message_id", [None, SAVED_ID])
@pytest.mark.parametrize("recent", [None, 0])
def test_without_a_window_the_whole_chat_is_read_and_the_session_is_never_asked(message_id, recent):
    chat = _long_chat()

    history = _read(chat, message_id, recent=recent)

    assert chat.db.queried == [] and chat.whole_reads == 1
    assert len(history) == len(chat.rows)
    assert _said(history)[0] == ("user", "ask 0") and _said(history)[-1] == ("user", NEWEST)


def test_a_caller_that_names_no_window_reads_the_whole_chat():
    chat = _long_chat()
    newest = ChatMessageRequest(parts=[MessagePart(type="text", text=NEWEST)])

    history = chat_turn_history.turn_history(chat, CHAT_ID, newest, {})

    assert chat.db.queried == [] and chat.whole_reads == 1
    assert len(history) == len(chat.rows)


# --------------------------------------------------------------------------- through the route


@pytest.fixture
def route(monkeypatch):
    """``POST /api/chat`` with its collaborators patched (the fixture of
    ``test_spoken_turns_skip_the_apps_search.py``); every ``turn_history`` call is kept, and the
    lane keeps the history length it was given."""
    reads = []
    lane_lengths = []
    handler = StreamingHandler()

    class _ChatService:
        def __init__(self, db):
            self.db = db

        def save_message(self, **kwargs):
            return NS(id=SAVED_ID)

    class _StreamingService:
        def __init__(self, db, workspace_id=None):
            self.streaming_handler = handler

        async def stream_response_with_agent(self, **kwargs):
            yield handler.format_aisdk_text("Twelve pounds.")
            yield handler.format_aisdk_finish()

    class _Queue:
        @asynccontextmanager
        async def acquire(self, key):
            yield

    def _history(*args, **kwargs):
        reads.append((args, kwargs))
        return [{"role": "user", "parts": [{"type": "text", "text": NEWEST}]}]

    async def _the_lane(db, ctx, request, message_text, history_length, before=""):
        lane_lengths.append(history_length)
        return NS(agent_id=request.agentId or AUTO, assessment=None, session_agent=None,
                  suggest_mission=False, skip_composio=False)

    async def _inline(*, chat_id, produce, on_complete=None):
        async for chunk in produce():
            yield chunk

    monkeypatch.setattr(chat, "ChatService", _ChatService)
    monkeypatch.setattr(chat, "StreamingChatService", _StreamingService)
    monkeypatch.setattr(chat, "SessionLocal", lambda: NS(close=lambda: None))
    monkeypatch.setattr(chat, "get_session_queue", lambda: _Queue())
    monkeypatch.setattr(chat, "get_user_id", lambda db, ctx=None: USER_PK)
    monkeypatch.setattr(chat, "_open_chat", lambda *args, **kwargs: CHAT_ID)
    monkeypatch.setattr(chat, "turn_history", _history)
    monkeypatch.setattr(chat, "_turn_lane", _the_lane)
    monkeypatch.setattr(chat, "run_detached_turn", _inline)
    return NS(reads=reads, lane_lengths=lane_lengths)


def _ask(body: dict) -> list:
    """The frames ``POST /api/chat`` streams for ``body``."""
    ctx = NS(workspace_id=WS, user=NS(id="owner@c4.test", system_role="user"))

    async def _run() -> list:
        response = await chat.stream_chat(request=ChatRequest(**body), ctx=ctx, db=NS(rollback=lambda: None))
        return [chunk async for chunk in response.body_iterator]

    return asyncio.run(_run())


def _body(**over) -> dict:
    return {"message": {"role": "user", "parts": [{"type": "text", "text": NEWEST}]}, **over}


def _is_the_turns_read(read) -> bool:
    """The turn's read names its saved message (the fifth argument); the route's has none yet."""
    args, _ = read
    return len(args) == 5 and args[4] == SAVED_ID


def test_a_spoken_turn_for_a_chosen_agent_reads_once_in_the_turn(route, monkeypatch):
    monkeypatch.setattr(config, "SPOKEN_MODE_HISTORY_MESSAGES", WINDOW)

    frames = _ask(_body(spokenMode=True, agentId=CHOSEN))

    (read,) = route.reads
    assert _is_the_turns_read(read) and read[1] == {"recent": WINDOW}
    assert route.lane_lengths == [0]
    assert frames


def test_a_spoken_turn_for_auto_reads_twice_both_within_the_window(route, monkeypatch):
    monkeypatch.setattr(config, "SPOKEN_MODE_HISTORY_MESSAGES", WINDOW)

    _ask(_body(spokenMode=True))

    lanes_read, turns_read = route.reads
    assert len(lanes_read[0]) == 4 and lanes_read[1] == {"recent": WINDOW}
    assert _is_the_turns_read(turns_read) and turns_read[1] == {"recent": WINDOW}
    assert route.lane_lengths == [1]


@pytest.mark.parametrize("agent", [None, CHOSEN])
def test_a_typed_turn_still_reads_the_whole_chat_twice(route, monkeypatch, agent):
    monkeypatch.setattr(config, "SPOKEN_MODE_HISTORY_MESSAGES", WINDOW)

    _ask(_body(spokenMode=False, **({"agentId": agent} if agent else {})))

    lanes_read, turns_read = route.reads
    assert len(lanes_read[0]) == 4 and lanes_read[1] == {"recent": None}
    assert _is_the_turns_read(turns_read) and turns_read[1] == {"recent": None}
    assert route.lane_lengths == [1]


@pytest.mark.parametrize("setting", [0, 3])
@pytest.mark.parametrize("spoken", [False, True])
@pytest.mark.parametrize("agent", [None, CHOSEN])
def test_setting_off_every_turn_reads_the_whole_chat_twice(route, monkeypatch, setting, spoken, agent):
    monkeypatch.setattr(config, "SPOKEN_MODE_HISTORY_MESSAGES", setting)

    _ask(_body(spokenMode=spoken, **({"agentId": agent} if agent else {})))

    lanes_read, turns_read = route.reads
    assert len(lanes_read[0]) == 4 and _is_the_turns_read(turns_read)
    assert [kwargs for _, kwargs in route.reads] == [{"recent": None}, {"recent": None}]
    assert route.lane_lengths == [1]
