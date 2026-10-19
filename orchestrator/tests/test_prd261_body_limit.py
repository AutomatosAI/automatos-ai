"""PRD-261 W-C1 (M4) — upload caps counted on the wire (``core/security/body_limit.py``).

The middleware is driven directly over ASGI with a fake app that records each body message it
read, so each case is exact: the app reads the body message by message (nothing buffered in
between, so no upload is held in memory before auth); a chunked body past the cap is 413, the
app never sees a byte past it, and the response it then starts is replaced; one within it reaches
the app whole; a body with no length and no chunking is 411; other paths and body-less methods
pass through untouched. ``main.py`` wiring is pinned in ``test_prd261_c1_wire.py`` (the live app).
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

from core.security.body_limit import CappedBodyMiddleware  # noqa: E402

CAPPED = "/api/themes/import"
LIMIT = 100
LIMITS = {CAPPED: LIMIT, "/api/marketplace/themes/submit": LIMIT}


class _App:
    """Reads the body as a route would, one message at a time, and answers 200 with its length.

    On ``http.disconnect`` it does what FastAPI does with a body it couldn't read: answers 400
    (``on_disconnect="400"``), raises like Starlette's ``ClientDisconnect`` (``"raise"``) or
    returns without answering (``"leave"``).
    """

    def __init__(self, on_disconnect: str = "400") -> None:
        self.ran = False
        self.body = b""
        self.seen: List[int] = []  # the size of each body message, in the order the app read them
        self.on_disconnect = on_disconnect

    async def __call__(self, scope, receive, send) -> None:
        self.ran = True
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                await self._gone(send)
                return
            self.seen.append(len(message.get("body", b"")))
            self.body += message.get("body", b"")
            if not message.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": str(len(self.body)).encode()})

    async def _gone(self, send) -> None:
        if self.on_disconnect == "raise":
            raise RuntimeError("client disconnected")
        if self.on_disconnect == "400":
            await send({"type": "http.response.start", "status": 400, "headers": []})
            await send({"type": "http.response.body", "body": b'{"detail": "There was an error parsing the body"}'})


def _scope(path: str = CAPPED, method: str = "POST", headers: Optional[dict] = None):
    raw = [(name.encode("latin-1"), value.encode("latin-1")) for name, value in (headers or {}).items()]
    return {"type": "http", "method": method, "path": path, "headers": raw}


def _run(scope, parts: List[bytes], app: Optional[_App] = None):
    """Send ``parts`` as body messages; returns (app, status, json-or-text body, messages read)."""
    app = app or _App()
    queue = [{"type": "http.request", "body": part, "more_body": n < len(parts) - 1} for n, part in enumerate(parts)]
    read = []
    sent = []

    async def receive():
        if queue:
            read.append(queue[0])
            return queue.pop(0)
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    asyncio.run(CappedBodyMiddleware(app, LIMITS)(scope, receive, send))
    status = sent[0]["status"]
    body = b"".join(message.get("body", b"") for message in sent[1:])
    return app, status, body, read


CHUNKED = {"transfer-encoding": "chunked"}


@pytest.mark.parametrize("on_disconnect", ["400", "raise", "leave"])
def test_a_chunked_body_past_the_cap_is_413_and_the_app_never_sees_a_byte_past_it(on_disconnect):
    parts = [b"x" * 40] * 10  # 400 bytes, no Content-Length
    app, status, body, read = _run(_scope(headers=CHUNKED), parts, _App(on_disconnect))

    assert status == 413 and json.loads(body) == {"detail": "Payload too large"}
    assert app.seen == [40, 40]  # the third message passed 100 bytes: the app got a disconnect instead
    assert len(read) == 3  # stopped as soon as the count passed 100 bytes, not at the end


def test_the_body_reaches_the_app_message_by_message_never_buffered_first():
    order = []
    parts = [b"a" * 30, b"b" * 30, b"c" * 40]
    queue = [{"type": "http.request", "body": part, "more_body": n < 2} for n, part in enumerate(parts)]

    async def receive():
        order.append("server")
        return queue.pop(0)

    async def app(scope, receive_, send):
        while True:
            message = await receive_()
            order.append(f"app:{len(message['body'])}")
            if not message["more_body"]:
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})

    async def send(message):
        pass

    asyncio.run(CappedBodyMiddleware(app, LIMITS)(_scope(headers=CHUNKED), receive, send))
    assert order == ["server", "app:30", "server", "app:30", "server", "app:40"]


def test_a_chunked_body_within_the_cap_reaches_the_app_whole():
    parts = [b"a" * 30, b"b" * 30, b"c" * 40]  # exactly the cap
    app, status, body, _ = _run(_scope(headers=CHUNKED), parts)

    assert status == 200 and app.ran and app.body == b"".join(parts)


def test_no_content_length_and_no_chunking_is_411():
    app, status, body, read = _run(_scope(headers={}), [b"x"])
    assert status == 411 and "Content-Length" in json.loads(body)["detail"]
    assert app.ran is False and read == []


@pytest.mark.parametrize("length, expected", [("101", 413), ("abc", 400), ("-1", 400)])
def test_a_content_length_over_the_cap_or_not_a_number_is_refused_before_reading(length, expected):
    app, status, _, read = _run(_scope(headers={"content-length": length}), [b"x"])
    assert status == expected and app.ran is False and read == []


def test_a_body_longer_than_its_content_length_claims_is_still_counted():
    # The server frames by Content-Length, but the count never trusts the header alone.
    app, status, _, _ = _run(_scope(headers={"content-length": "10"}), [b"x" * 60, b"x" * 60])
    assert status == 413 and app.seen == [60]


def test_the_submit_path_is_capped_too():
    app, status, _, _ = _run(_scope("/api/marketplace/themes/submit", headers=CHUNKED), [b"x" * 200])
    assert status == 413 and app.seen == []


@pytest.mark.parametrize("path, method", [("/api/other/upload", "POST"), (CAPPED, "GET"), (CAPPED, "DELETE")])
def test_other_paths_and_bodyless_methods_pass_through_untouched(path, method):
    parts = [b"x" * 60] * 5  # past the cap, chunked: still not this middleware's business
    app, status, _, _ = _run(_scope(path, method, headers=CHUNKED), parts)
    assert status == 200 and app.ran and len(app.body) == 300


def test_a_client_that_leaves_mid_body_reaches_the_app_as_a_disconnect_and_gets_no_413():
    app = _App(on_disconnect="leave")
    messages = [{"type": "http.request", "body": b"x" * 10, "more_body": True}, {"type": "http.disconnect"}]
    sent = []

    async def receive():
        return messages.pop(0)

    async def send(message):
        sent.append(message)

    asyncio.run(CappedBodyMiddleware(app, LIMITS)(_scope(headers=CHUNKED), receive, send))
    assert sent == [] and app.seen == [10]
