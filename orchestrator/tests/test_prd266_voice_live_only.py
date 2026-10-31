"""PRD-266 US-008b — the custom-LLM socket serves live calls only.

* ``LIVE_CALL_STATUSES`` is the one constant of the live statuses
  (``modules/voice/live_socket.py``); the mint supersedes by the same one.
* ``WS /api/voice/retell/llm-websocket/{call_id}`` refuses with 4409, before
  accept, every call whose status is not live (ended, failed, superseded,
  anything unknown); minted and started calls are accepted; an unminted call
  stays 4401 and a disarmed lane 4403.
* ``call_ended`` stamps the call ended (failed when it never started), and the
  socket then refuses that call.
* Mid-call: a turn after the status left the live set closes the socket
  without reaching Auto.

No database: the route's session and the call row are faked; vendor HTTP is
never reached.
"""
from __future__ import annotations

import asyncio
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

_orchestrator_root = Path(__file__).resolve().parent.parent
if str(_orchestrator_root) not in sys.path:
    sys.path.insert(0, str(_orchestrator_root))

NOT_LIVE_STATUSES = ("ended", "failed", "superseded", "on_hold")
START_MS, END_MS = 1_700_000_000_000, 1_700_000_060_000  # a 60s call


class _FakeWebSocket:
    """A scripted Retell-side socket; receive_json yields a loop tick first so a
    just-created turn task runs before the next message."""

    def __init__(self, incoming=()):
        self.incoming = list(incoming)
        self.sent: list = []
        self.accepted = False
        self.close_code = None

    async def accept(self):
        self.accepted = True

    async def close(self, code=1000):
        self.close_code = code

    async def send_json(self, data):
        self.sent.append(data)

    async def receive_json(self):
        from fastapi import WebSocketDisconnect

        await asyncio.sleep(0)
        if not self.incoming:
            raise WebSocketDisconnect(1000)
        return self.incoming.pop(0)


def _route_with_status(monkeypatch, status, *, armed: bool = True):
    """Arm (or disarm) the lane and fake the call's mint row (``status`` None = no row)."""
    import api.voice_retell as vr

    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None if status is None else (status,)
    db.query.return_value.filter.return_value.scalar.return_value = status

    @contextmanager
    def fake_session():
        yield db

    monkeypatch.setattr(vr.live_settings, "voice_live_enabled", lambda: armed)
    monkeypatch.setattr(vr, "get_db_session", fake_session)
    return vr


def _open(monkeypatch, status, *, armed: bool = True) -> _FakeWebSocket:
    vr = _route_with_status(monkeypatch, status, armed=armed)
    ws = _FakeWebSocket()
    asyncio.run(vr.retell_llm_websocket(ws, "call_live_only"))
    return ws


# ---------------------------------------------------------------------------
# The one constant
# ---------------------------------------------------------------------------

def test_live_statuses_are_one_constant_beside_superseded():
    from modules.voice import live_socket, mint

    assert live_socket.LIVE_CALL_STATUSES == ("minted", "started")
    assert live_socket.CALL_STATUS_SUPERSEDED not in live_socket.LIVE_CALL_STATUSES
    # The mint supersedes the caller's live calls by the same constant, not a copy.
    assert mint.LIVE_CALL_STATUSES is live_socket.LIVE_CALL_STATUSES
    assert mint.CALL_STATUS_SUPERSEDED is live_socket.CALL_STATUS_SUPERSEDED
    assert not hasattr(mint, "ACTIVE_CALL_STATUSES")


@pytest.mark.parametrize("status", (*NOT_LIVE_STATUSES, None, ""))
def test_is_live_refuses_every_status_but_minted_and_started(status):
    from modules.voice.live_socket import is_live

    assert is_live(status) is False


@pytest.mark.parametrize("status", ("minted", "started"))
def test_is_live_accepts_minted_and_started(status):
    from modules.voice.live_socket import is_live

    assert is_live(status) is True


# ---------------------------------------------------------------------------
# The route: 4409 before accept for every call that is not live
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status", NOT_LIVE_STATUSES)
def test_socket_refuses_a_call_that_is_not_live_before_accept(monkeypatch, status):
    ws = _open(monkeypatch, status)
    assert ws.close_code == 4409
    assert ws.accepted is False
    assert ws.sent == []  # not even the config frame


@pytest.mark.parametrize("status", ("minted", "started"))
def test_socket_accepts_a_live_call(monkeypatch, status):
    ws = _open(monkeypatch, status)
    assert ws.accepted is True
    assert ws.close_code is None
    assert ws.sent and ws.sent[0]["response_type"] == "config"


def test_unminted_call_stays_4401(monkeypatch):
    ws = _open(monkeypatch, None)
    assert ws.close_code == 4401
    assert ws.accepted is False


@pytest.mark.parametrize("status", ("started", "ended"))
def test_disarmed_lane_stays_4403(monkeypatch, status):
    ws = _open(monkeypatch, status, armed=False)
    assert ws.close_code == 4403
    assert ws.accepted is False


# ---------------------------------------------------------------------------
# call_ended stamps the call ended — and the socket then refuses it
# ---------------------------------------------------------------------------

def _call_row(status: str, started_at=None) -> SimpleNamespace:
    return SimpleNamespace(
        status=status, started_at=started_at, ended_at=None,
        duration_seconds=None, disconnect_reason=None, vendor_latency=None,
        workspace_id=None,  # call_ended charges the call's workspace (PRD-265 D-6); none here
    )


def _apply(event: str, row: SimpleNamespace, call: dict) -> str:
    from modules.voice.call_events import apply_call_event

    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = row
    return apply_call_event(db, event, {"call_id": "call_live_only", **call})


def test_call_ended_webhook_sets_ended(monkeypatch):
    row = _call_row("started", started_at=datetime(2023, 11, 14, 22, 13, 20))
    disposition = _apply(
        "call_ended", row,
        {"start_timestamp": START_MS, "end_timestamp": END_MS, "disconnection_reason": "user_hangup"},
    )
    assert disposition == "updated"
    assert row.status == "ended"
    assert row.duration_seconds == 60
    # the call that just ended can no longer open the socket
    assert _open(monkeypatch, row.status).close_code == 4409


def test_call_ended_that_never_started_sets_failed(monkeypatch):
    row = _call_row("minted")
    _apply("call_ended", row, {"disconnection_reason": "dial_failed"})
    assert row.status == "failed"
    assert _open(monkeypatch, row.status).close_code == 4409


def test_a_replayed_call_started_never_revives_an_ended_call():
    row = _call_row("minted")
    _apply("call_started", row, {"start_timestamp": START_MS})
    assert row.status == "started"
    _apply("call_ended", row, {"start_timestamp": START_MS, "end_timestamp": END_MS})
    _apply("call_started", row, {"start_timestamp": START_MS})  # a late retry
    assert row.status == "ended"


# ---------------------------------------------------------------------------
# Mid-call: a turn after the status left the live set is refused
# ---------------------------------------------------------------------------

def _turn(rid: int) -> dict:
    return {
        "interaction_type": "response_required", "response_id": rid,
        "transcript": [{"role": "user", "content": f"turn {rid}"}],
    }


async def _first_socket(call_id: str) -> bool:
    return False  # the call's first socket, not an automatic reconnect


def _run_turns(monkeypatch, statuses: list) -> tuple[_FakeWebSocket, list]:
    """Two turns on one accepted socket; the call's status is re-read once per turn."""
    from config import config as app_config
    from modules.voice import live_socket

    monkeypatch.setattr(app_config, "VOICE_LIVE_FIRST_FRAME_ACK_SECONDS", 0)
    reads: list = []
    streamed: list = []

    def call_status(cid: str):
        reads.append(cid)
        return statuses[len(reads) - 1]

    async def stream_turn(req):
        streamed.append(req.response_id)
        yield {"response_id": req.response_id, "content": f"answer {req.response_id}", "content_complete": True}

    ports = live_socket.SocketPorts(
        stream_turn=stream_turn, call_status=call_status,
        ai_greeting="Hi, this is Auto, an AI assistant.", ai_reminder="I'm an AI assistant.",
        socket_reconnected=_first_socket, clock=lambda: 0.0,
    )
    ws = _FakeWebSocket([{"interaction_type": "call_details", "call": {}}, _turn(1), _turn(2)])
    asyncio.run(live_socket.LiveCallSocket(ws, "call_live_only", ports).run())
    assert len(reads) == len(statuses), "the status is read once per turn"
    return ws, streamed


@pytest.mark.parametrize("status", (*NOT_LIVE_STATUSES, None))
def test_a_turn_after_the_status_left_the_live_set_is_refused(monkeypatch, status):
    ws, streamed = _run_turns(monkeypatch, ["started", status])
    assert streamed == [1]  # the live turn spoke; the stale one never reached Auto
    assert ws.close_code == 1000
    assert not any(m.get("response_id") == 2 for m in ws.sent)


def test_turns_on_a_live_call_keep_speaking(monkeypatch):
    ws, streamed = _run_turns(monkeypatch, ["minted", "started"])
    assert streamed == [1, 2]
    assert ws.close_code is None
