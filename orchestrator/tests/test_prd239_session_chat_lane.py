"""PRD-239 S2 — chat with a session agent: one ticket per message, the
previous session resumed on the host that ran it, an honest turn line with the
card, and the session's ending delivered back into the conversation.

Pure units: fake rows, fake session, the stream handler and the message store
stubbed at the import boundary."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from services import cli_ticket_lane as lane  # noqa: E402
from services import session_agent_chat as chat  # noqa: E402

WS = uuid4()


def _msg(role, text):
    return {"role": role, "parts": [{"type": "text", "text": text}]}


# ── ids and context ──────────────────────────────────────────────────────────

def test_chat_source_id_round_trips_to_the_conversation():
    task = SimpleNamespace(source_type="chat", source_id=lane.chat_source_id("c1", "m1"))
    assert lane.chat_origin_of(task) == "c1"
    assert lane.chat_origin_of(SimpleNamespace(source_type="heartbeat", source_id="agent:15")) is None
    assert lane.chat_origin_of(SimpleNamespace(source_type="chat", source_id="chat:")) is None


def test_conversation_context_excludes_the_message_itself_and_caps_each_turn():
    history = [_msg("user", "hi"), _msg("assistant", "hello"), _msg("user", "z" * 900), _msg("user", "the new one")]
    text = chat.conversation_context(history, "Bob")
    assert text.startswith("## Conversation so far")
    assert "- **Operator:** hi" in text and "- **Bob:** hello" in text
    assert "the new one" not in text and "…" in text and "z" * 700 not in text
    assert chat.conversation_context([_msg("user", "only me")], "Bob") == ""


def test_ticket_prompt_carries_context_only_for_a_fresh_session():
    history = [_msg("user", "hi"), _msg("assistant", "hello"), _msg("user", "now")]
    fresh = chat.ticket_prompt("Bob", "now", history, continuing=False)
    resumed = chat.ticket_prompt("Bob", "now", history, continuing=True)
    assert "## Conversation so far" in fresh and fresh.endswith("## Message\nnow")
    assert "## Conversation so far" not in resumed and resumed.endswith("## Message\nnow")
    assert chat.ticket_title("Bob", "Hey Bob,\ncan you…") == "Chat with Bob: Hey Bob,"


# ── filing ───────────────────────────────────────────────────────────────────

def test_file_chat_ticket_asks_the_previous_host_to_resume_and_carries_the_operator(monkeypatch):
    seen = {}

    def fake_file(db, **kw):
        seen.update(kw)
        return SimpleNamespace(id=9, status="assigned", blocked_reason=None)

    monkeypatch.setattr(chat, "previous_session_of", lambda db, ws, cid, aid: ("sess-1", "host-1"))
    monkeypatch.setattr(chat, "file_cli_ticket", fake_file)
    agent = SimpleNamespace(id=15, name="Bob")
    task, continuing = chat.file_chat_ticket(
        None, workspace_id=WS, chat_id="c1", agent=agent, user_text="Hey Bob", history=[_msg("user", "Hey Bob")], user_id=2,
    )
    assert task.id == 9 and continuing is True
    assert seen["source_type"] == "chat" and seen["source_id"].startswith("chat:c1:")
    assert seen["resume_session_id"] == "sess-1" and seen["resume_host_id"] == "host-1"
    assert seen["actor"] == "user:2" and seen["created_by_type"] == "user" and seen["created_by_id"] == "2"
    assert seen["title"] == "Chat with Bob: Hey Bob" and "## Conversation so far" not in seen["prompt"]


def test_a_fresh_conversation_files_without_resume_and_with_context(monkeypatch):
    seen = {}
    monkeypatch.setattr(chat, "previous_session_of", lambda db, ws, cid, aid: None)
    monkeypatch.setattr(chat, "file_cli_ticket", lambda db, **kw: seen.update(kw) or SimpleNamespace(id=1))
    history = [_msg("user", "earlier"), _msg("assistant", "reply"), _msg("user", "Hey")]
    _, continuing = chat.file_chat_ticket(None, workspace_id=WS, chat_id="c1", agent=SimpleNamespace(id=15, name="Bob"),
                                          user_text="Hey", history=history, user_id=2)
    assert continuing is False and seen["resume_session_id"] is None
    assert "- **Operator:** earlier" in seen["prompt"]


# ── what the turn says ───────────────────────────────────────────────────────

def test_turn_line_is_honest_about_host_approval_and_continuity():
    t = SimpleNamespace(id=9, status="assigned", blocked_reason=None)
    line = chat.turn_line("Bob", t, continuing=True)
    assert "filed ticket #9" in line and "continues where your last exchange left off" in line
    assert "reply lands here when the session ends" in line
    assert "No CLI host" in chat.turn_line("Bob", SimpleNamespace(id=9, status="assigned", blocked_reason="none online"), False)
    assert "waiting for your approval" in chat.turn_line("Bob", SimpleNamespace(id=9, status="blocked", blocked_reason=None), False)


# ── the reply ────────────────────────────────────────────────────────────────

def _task(status, **ref):
    return SimpleNamespace(id=9, status=status, workspace_id=WS, assigned_agent_id=15,
                           source_type="chat", source_id="chat:c1:m1", runtime_ref=ref or {})


def test_reply_text_for_each_ending():
    assert chat.reply_text("Bob", _task("done"), {"result": "Here you go."}, "done") == "Here you go."
    review = chat.reply_text("Bob", _task("review", denials=2), {"result": "Partly."}, "review")
    assert review.startswith("Partly.") and "2 tool calls refused" in review
    assert "ended without a written reply" in chat.reply_text("Bob", _task("done", exit_reason="completed"), {"result": ""}, "done")
    assert chat.reply_text("Bob", _task("failed"), {"error": "boom"}, "failed") == "Bob could not finish ticket #9: boom"
    assert "cancelled before it answered" in chat.reply_text("Bob", _task("cancelled"), {}, "cancelled")


def test_deliver_session_reply_posts_into_the_origin_chat(monkeypatch):
    import services.chat_messenger as messenger

    seen = {}
    monkeypatch.setattr(messenger, "deliver_background_message", lambda db, **kw: seen.update(kw) or "posted")
    assert chat.deliver_session_reply(None, _task("done"), {"result": "Done."}, "done", agent_name="Bob") == "posted"
    assert seen["chat_id"] == "c1" and seen["text"] == "Done." and seen["link_id"] == "9"
    assert seen["source"]["label"] == "Bob · Claude Code session" and seen["source"]["origin"] == "session_agent"
    seen.clear()
    heartbeat = SimpleNamespace(id=3, source_type="heartbeat", source_id="agent:15", workspace_id=WS, assigned_agent_id=15)
    assert chat.deliver_session_reply(None, heartbeat, {}, "done") is None and not seen


# ── lane helpers ─────────────────────────────────────────────────────────────

def test_exec_result_for_each_board_ending():
    done = lane.exec_result_for(SimpleNamespace(id=1, status="done", result="out", error_message=None,
                                                runtime_ref={"usage": {"input_tokens": 10, "output_tokens": 5}}))
    assert done["status"] == "success" and done["result"] == "out" and done["tokens_used"] == 15
    assert done["runtime"] == "cli" and done["execution"]["tokens_used"] == 15
    review = lane.exec_result_for(SimpleNamespace(id=1, status="review", result="out", error_message=None, runtime_ref=None))
    assert review["status"] == "success" and "held for review" in review["result"]
    failed = lane.exec_result_for(SimpleNamespace(id=1, status="failed", result=None, error_message="bad", runtime_ref={}))
    assert failed["status"] == "error" and failed["error"] == "bad"
    assert lane.exec_result_for(SimpleNamespace(id=1, status="cancelled", result=None, error_message=None, runtime_ref={}))["status"] == "cancelled"


def test_running_predecessor_is_an_older_running_ticket_of_the_same_conversation(monkeypatch):
    monkeypatch.setattr(lane, "chat_tickets", lambda db, ws, cid, aid, statuses: [SimpleNamespace(id=9), SimpleNamespace(id=3)])
    newer = SimpleNamespace(id=9, workspace_id=WS, assigned_agent_id=15, source_type="chat", source_id="chat:c1:m2")
    assert lane.running_predecessor_of(None, newer).id == 3
    oldest = SimpleNamespace(id=3, workspace_id=WS, assigned_agent_id=15, source_type="chat", source_id="chat:c1:m1")
    assert lane.running_predecessor_of(None, oldest) is None
    assert lane.running_predecessor_of(None, SimpleNamespace(id=5, source_type="heartbeat", source_id="agent:15", assigned_agent_id=15, workspace_id=WS)) is None


def test_previous_session_prefers_the_id_the_hooks_reported(monkeypatch):
    ended = [
        SimpleNamespace(runtime_ref={"session_id": "pre", "cli_session_id": "real", "host_id": "h1"}),
    ]
    monkeypatch.setattr(lane, "chat_tickets", lambda db, ws, cid, aid, statuses: ended)
    assert lane.previous_session_of(None, WS, "c1", 15) == ("real", "h1")
    monkeypatch.setattr(lane, "chat_tickets", lambda db, ws, cid, aid, statuses: [SimpleNamespace(runtime_ref=None)])
    assert lane.previous_session_of(None, WS, "c1", 15) is None


# ── the turn ─────────────────────────────────────────────────────────────────

class _Query:
    def __init__(self, result):
        self._result = result

    def filter(self, *a, **k):
        return self

    def first(self):
        return self._result


def _stub_module(monkeypatch, name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


def _ensure_parents(monkeypatch, dotted):
    parts = dotted.split(".")
    for i in range(1, len(parts)):
        name = ".".join(parts[:i])
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = []  # a package, so submodule imports resolve through sys.modules
            monkeypatch.setitem(sys.modules, name, pkg)


async def _collect(gen):
    return [chunk async for chunk in gen]


def test_the_turn_streams_the_line_and_the_card_and_persists_both(monkeypatch):
    saved = []
    handler = SimpleNamespace(
        format_aisdk_chat_id=lambda c: f"id:{c}\n",
        format_aisdk_text=lambda t: f"text:{t}\n",
        format_aisdk_tool_data=lambda d: f"data:{json.dumps(d)}\n",
        format_aisdk_finish=lambda: "finish\n",
    )

    class ChatService:
        def __init__(self, db):
            self.db = db

        def save_message(self, **kw):
            saved.append(kw)

    _ensure_parents(monkeypatch, "consumers.chatbot.streaming")
    _stub_module(monkeypatch, "consumers.chatbot", ChatService=ChatService)
    _stub_module(monkeypatch, "consumers.chatbot.streaming", get_streaming_handler=lambda: handler)
    _ensure_parents(monkeypatch, "modules.tools.discovery.handlers_board_tasks")
    _stub_module(
        monkeypatch, "modules.tools.discovery.handlers_board_tasks",
        task_card=lambda task, name: {"id": task.id, "title": task.title, "status": task.status, "assigned_agent": name},
    )
    ticket = SimpleNamespace(id=9, title="Chat with Bob: Hey", status="assigned", blocked_reason=None, runtime_ref=None)
    monkeypatch.setattr(chat, "file_chat_ticket", lambda db, **kw: (ticket, False))
    db = SimpleNamespace(query=lambda model: _Query(SimpleNamespace(id=15, name="Bob")))

    chunks = asyncio.run(_collect(chat.produce_session_agent_turn(
        db=db, workspace_id=WS, chat_id="c1", agent_id=15, message_history=[], user_text="Hey", user_id=2,
    )))
    assert chunks[0] == "id:c1\n"
    assert chunks[1].startswith("text:Bob runs as a Claude Code session") and "filed ticket #9" in chunks[1]
    assert json.loads(chunks[2][5:])["task_card"]["id"] == 9
    assert chunks[-1] == "finish\n"
    assert saved and saved[0]["role"] == "assistant" and saved[0]["chat_id"] == "c1"
    assert [p["type"] for p in saved[0]["parts"]] == ["text", "task_card"]


def test_an_unknown_agent_ends_the_turn_honestly(monkeypatch):
    handler = SimpleNamespace(
        format_aisdk_chat_id=lambda c: "id\n", format_aisdk_text=lambda t: f"text:{t}\n",
        format_aisdk_tool_data=lambda d: "data\n", format_aisdk_finish=lambda: "finish\n",
    )
    _ensure_parents(monkeypatch, "consumers.chatbot.streaming")
    _stub_module(monkeypatch, "consumers.chatbot", ChatService=lambda db: None)
    _stub_module(monkeypatch, "consumers.chatbot.streaming", get_streaming_handler=lambda: handler)
    _ensure_parents(monkeypatch, "modules.tools.discovery.handlers_board_tasks")
    _stub_module(monkeypatch, "modules.tools.discovery.handlers_board_tasks", task_card=lambda t, n: {})
    db = SimpleNamespace(query=lambda model: _Query(None))
    chunks = asyncio.run(_collect(chat.produce_session_agent_turn(
        db=db, workspace_id=WS, chat_id="c1", agent_id=404, message_history=[], user_text="Hey", user_id=2,
    )))
    assert chunks == ["id\n", "text:Agent 404 is not in this workspace.\n", "finish\n"]
