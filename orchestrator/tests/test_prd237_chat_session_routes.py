"""PRD-237 S6/S7 — chat session routes + cancel, unit-level (db mocked).

1. Route order — ``/session`` MUST resolve before ``/{chat_id}`` (the PRD-220
   ``/search`` failure mode), and the cancel route exists.
2. The session document is validated: chat ids are UUIDs (canonicalised,
   de-duplicated), at most 8 open, ``activeChatId`` ∈ ``openChatIds``,
   read-stamps numeric and capped to the newest 50.
3. GET returns the empty doc when nothing is stored; PUT rebuilds the JSONB
   (never mutates) under the workspace key and stamps ``updatedAt`` in UTC.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import HTTPException


def _api_chat():
    try:
        import api.chat as chat_module
    except Exception as e:  # env without the heavy router deps
        pytest.skip(f"api.chat not importable in this env: {e}")
    return chat_module


def _get_paths(chat_module, method: str):
    return [r.path for r in chat_module.router.routes if method in (getattr(r, "methods", None) or set())]


# ---------------------------------------------------------------------------
# 1. route order
# ---------------------------------------------------------------------------

def test_session_routes_registered_before_chat_id_param_route():
    m = _api_chat()
    gets = _get_paths(m, "GET")
    assert "/api/chat/session" in gets, gets
    assert gets.index("/api/chat/session") < gets.index("/api/chat/{chat_id}"), gets
    assert "/api/chat/session" in _get_paths(m, "PUT")


def test_cancel_route_registered():
    m = _api_chat()
    assert "/api/chat/{chat_id}/cancel" in _get_paths(m, "POST")


# ---------------------------------------------------------------------------
# 2. validation
# ---------------------------------------------------------------------------

def _req(m, **kwargs):
    return m.ChatSessionRequest(**kwargs)


def test_session_doc_canonicalises_and_dedupes_ids():
    m = _api_chat()
    a, b = str(uuid4()), str(uuid4())
    doc = m._session_doc_from_request(_req(m, openChatIds=[a.upper(), b, a], activeChatId=b, draftOpen=True))
    assert doc["openChatIds"] == [a, b]
    assert doc["activeChatId"] == b
    assert doc["draftOpen"] is True


def test_session_doc_rejects_non_uuid_ids():
    m = _api_chat()
    with pytest.raises(HTTPException) as exc:
        m._session_doc_from_request(_req(m, openChatIds=["not-a-chat"]))
    assert exc.value.status_code == 422
    assert "openChatIds" in exc.value.detail


def test_session_doc_caps_open_tabs():
    m = _api_chat()
    ids = [str(uuid4()) for _ in range(m.MAX_OPEN_CHATS + 1)]
    with pytest.raises(HTTPException) as exc:
        m._session_doc_from_request(_req(m, openChatIds=ids))
    assert exc.value.status_code == 422


def test_session_doc_requires_active_among_open():
    m = _api_chat()
    with pytest.raises(HTTPException) as exc:
        m._session_doc_from_request(_req(m, openChatIds=[str(uuid4())], activeChatId=str(uuid4())))
    assert exc.value.status_code == 422


def test_session_doc_read_stamps_numeric_and_capped():
    m = _api_chat()
    stamps = {str(uuid4()): 1_000 + i for i in range(m.MAX_TRACKED_THREADS + 5)}
    doc = m._session_doc_from_request(_req(m, lastReadAt=stamps))
    assert len(doc["lastReadAt"]) == m.MAX_TRACKED_THREADS
    assert min(doc["lastReadAt"].values()) == 1_005  # the oldest five were dropped
    with pytest.raises(HTTPException):
        m._session_doc_from_request(_req(m, lastReadAt={str(uuid4()): "yesterday"}))
    with pytest.raises(HTTPException):
        m._session_doc_from_request(_req(m, lastReadAt={str(uuid4()): True}))


# ---------------------------------------------------------------------------
# 3. handlers
# ---------------------------------------------------------------------------

def _ctx(ws):
    return SimpleNamespace(user=SimpleNamespace(id=42), workspace_id=ws)


def _db_with_user(row):
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = row
    return db


def test_get_session_returns_empty_doc_when_nothing_stored():
    m = _api_chat()
    ws = uuid4()
    out = asyncio.run(m.get_chat_session(ctx=_ctx(ws), db=_db_with_user(SimpleNamespace(chat_sessions=None))))
    assert out == m._EMPTY_SESSION_DOC
    out = asyncio.run(m.get_chat_session(ctx=_ctx(ws), db=_db_with_user(None)))
    assert out == m._EMPTY_SESSION_DOC


def test_get_session_returns_this_workspace_doc_only():
    m = _api_chat()
    ws, other = uuid4(), uuid4()
    mine = {"activeChatId": None, "draftOpen": True, "openChatIds": [], "lastReadAt": {}, "updatedAt": "x"}
    row = SimpleNamespace(chat_sessions={str(ws): mine, str(other): {"draftOpen": False}})
    out = asyncio.run(m.get_chat_session(ctx=_ctx(ws), db=_db_with_user(row)))
    assert out == mine


def test_put_session_rebuilds_jsonb_under_workspace_key():
    m = _api_chat()
    ws, other = uuid4(), uuid4()
    original = {str(other): {"draftOpen": True}}
    row = SimpleNamespace(chat_sessions=original)
    db = _db_with_user(row)
    cid = str(uuid4())
    out = asyncio.run(m.put_chat_session(_req(m, openChatIds=[cid], activeChatId=cid), ctx=_ctx(ws), db=db))
    assert out["openChatIds"] == [cid] and out["activeChatId"] == cid
    assert out["updatedAt"].endswith("+00:00"), out["updatedAt"]  # UTC, parseable by Date.parse
    assert row.chat_sessions[str(ws)] == out
    assert row.chat_sessions[str(other)] == {"draftOpen": True}  # other workspaces untouched
    assert row.chat_sessions is not original  # rebuilt, not mutated in place
    db.commit.assert_called_once()


def test_put_session_404s_without_a_user_row():
    m = _api_chat()
    with pytest.raises(HTTPException) as exc:
        asyncio.run(m.put_chat_session(_req(m), ctx=_ctx(uuid4()), db=_db_with_user(None)))
    assert exc.value.status_code == 404
