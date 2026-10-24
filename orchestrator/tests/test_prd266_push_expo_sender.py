"""PRD-266 US-010a (§4.4, §4.10 exception 2) — the Expo sender behind the ``PushSender`` protocol.

``ExpoPushSender`` over ``httpx.MockTransport`` (every request recorded; ``sleep`` recorded, so
the one retry costs no time):

* 250 messages are 3 requests to Expo's send URL (100, 100, 50) and 250 tickets in order;
  2,500 receipt ids are 3 getReceipts calls (1,000, 1,000, 500).
* Expo's headers always; ``Authorization: Bearer`` only when ``EXPO_PUSH_ACCESS_TOKEN`` is set.
* A ``DeviceNotRegistered`` ticket or receipt is parsed to its code; an odd code is not kept.
* A 503 (or a 429) then a 200 is one retry after ``EXPO_RETRY_DELAY_SECONDS``; a second 503 is
  not retried again; a 400 is never retried and every ticket of the batch is an error.
* A body that is not JSON, a request-level ``errors`` array, a short ticket list, data that is
  not JSON (a UUID, a NaN) and a transport error give error tickets (or no receipts) without
  raising; one ticket not in Expo's shape is an error for its message only, and one such receipt
  is left out while the others are kept.
* The log holds no push token, access token, title, body or data, nor Expo's own message.
* ``send`` and ``receipts`` take their argument positionally (PRD-264 EUAI-03's guardrail).

DB-free: no database, no network.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Callable, Dict, List

import httpx
import pytest

from config import config
from services.push import expo
from services.push.expo import (
    EXPO_RECEIPTS_BATCH,
    EXPO_RECEIPTS_URL,
    EXPO_RETRY_DELAY_SECONDS,
    EXPO_SEND_BATCH,
    EXPO_SEND_URL,
    MALFORMED_ANSWER,
    REQUEST_FAILED,
    UNKNOWN_ERROR,
    ExpoPushSender,
)
from services.push.sender import PushMessage, PushReceipt, PushSender, PushTicket

FAKE_ACCESS = "xp-token1"
TITLE = "Ledger · AI"
BODY = "Ship the Lisbon deck?"
DATA_MARK = "prd266-data-mark"
# Expo's own error message echoes the token; the sender must never log or keep it.
EXPO_MESSAGE = "is not a registered push notification recipient"

Handler = Callable[[httpx.Request], httpx.Response]


def _token(index: int) -> str:
    return f"ExponentPushToken[prd266-{index:04d}]"


def _messages(count: int) -> List[PushMessage]:
    return [
        PushMessage(to=_token(index), title=TITLE, body=BODY, data={"mark": DATA_MARK, "n": index})
        for index in range(count)
    ]


class _Recorder:
    """A MockTransport handler that records each request and answers from ``answer``."""

    def __init__(self, answer: Callable[[httpx.Request, int], httpx.Response]) -> None:
        self.requests: List[httpx.Request] = []
        self._answer = answer

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._answer(request, len(self.requests))

    def bodies(self) -> List[Any]:
        return [json.loads(request.content) for request in self.requests]


def _sender(recorder: Handler, sleeps: List[float] | None = None) -> ExpoPushSender:
    recorded = sleeps if sleeps is not None else []
    return ExpoPushSender(
        client_factory=lambda: httpx.Client(transport=httpx.MockTransport(recorder)),
        sleep=recorded.append,
    )


def _all_ok(request: httpx.Request, _attempt: int) -> httpx.Response:
    sent = json.loads(request.content)
    return httpx.Response(200, json={"data": [{"status": "ok", "id": f"ticket-{m['to']}"} for m in sent]})


def _receipts_ok(request: httpx.Request, _attempt: int) -> httpx.Response:
    ids = json.loads(request.content)["ids"]
    return httpx.Response(200, json={"data": {ticket_id: {"status": "ok"} for ticket_id in ids}})


def _both_ok(request: httpx.Request, attempt: int) -> httpx.Response:
    answer = _receipts_ok if str(request.url) == EXPO_RECEIPTS_URL else _all_ok
    return answer(request, attempt)


@pytest.fixture(autouse=True)
def _no_access_token(monkeypatch):
    monkeypatch.setattr(config, "EXPO_PUSH_ACCESS_TOKEN", "", raising=False)


def test_the_constants_name_expo_and_its_batch_sizes():
    assert EXPO_SEND_URL == "https://exp.host/--/api/v2/push/send"
    assert EXPO_RECEIPTS_URL == "https://exp.host/--/api/v2/push/getReceipts"
    assert (EXPO_SEND_BATCH, EXPO_RECEIPTS_BATCH, EXPO_RETRY_DELAY_SECONDS) == (100, 1000, 1.0)


def test_the_settings_default_to_no_token_and_ten_seconds():
    from config import Config

    assert Config.EXPO_PUSH_ACCESS_TOKEN == ""
    assert Config.EXPO_PUSH_TIMEOUT_SECONDS == 10


def test_the_default_client_verifies_tls_and_times_out_from_config(monkeypatch):
    seen: Dict[str, Any] = {}
    monkeypatch.setattr(config, "EXPO_PUSH_TIMEOUT_SECONDS", 7)
    monkeypatch.setattr(expo.httpx, "Client", lambda **kwargs: seen.update(kwargs) or "client")
    assert expo._default_client() == "client"
    assert seen == {"timeout": 7, "verify": True, "follow_redirects": False}


def test_the_expo_sender_is_a_push_sender_and_takes_its_argument_positionally():
    sender = _sender(_Recorder(_all_ok))
    assert isinstance(sender, PushSender)
    with pytest.raises(TypeError):
        sender.send(messages=_messages(1))  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        sender.receipts(ticket_ids=["a"])  # type: ignore[call-arg]


def test_250_messages_are_three_requests_and_250_tickets_in_order():
    recorder = _Recorder(_all_ok)
    messages = _messages(250)
    tickets = _sender(recorder).send(messages)
    assert [str(request.url) for request in recorder.requests] == [EXPO_SEND_URL] * 3
    assert [len(body) for body in recorder.bodies()] == [100, 100, 50]
    assert [request.method for request in recorder.requests] == ["POST"] * 3
    assert len(tickets) == 250
    assert [ticket.to for ticket in tickets] == [message.to for message in messages]
    assert all(ticket.status == "ok" and ticket.ticket_id == f"ticket-{ticket.to}" for ticket in tickets)


def test_no_messages_make_no_request():
    recorder = _Recorder(_all_ok)
    assert _sender(recorder).send([]) == []
    assert _sender(recorder).receipts([]) == {}
    assert recorder.requests == []


def test_a_message_goes_in_expos_shape_and_unset_fields_are_left_out():
    recorder = _Recorder(_all_ok)
    full = PushMessage(to=_token(1), title=TITLE, body=BODY, data={"v": 2}, category_id="automatos.question.2",
                       channel_id="questions", priority="high", sound="default")
    bare = PushMessage(to=_token(2), title=TITLE, body=BODY, sound=None)
    _sender(recorder).send([full, bare])
    first, second = recorder.bodies()[0]
    assert first == {"to": _token(1), "title": TITLE, "body": BODY, "data": {"v": 2}, "priority": "high",
                     "categoryId": "automatos.question.2", "channelId": "questions", "sound": "default"}
    assert second == {"to": _token(2), "title": TITLE, "body": BODY, "data": {}, "priority": "high"}


def test_a_message_never_changes_after_it_is_built():
    data: Dict[str, Any] = {"v": 2}
    message = PushMessage(to=_token(1), title=TITLE, body=BODY, data=data)
    data["v"] = 3
    assert message.data["v"] == 2
    with pytest.raises(TypeError):
        message.data["v"] = 4  # type: ignore[index]


def test_2500_receipt_ids_are_three_getreceipts_calls():
    recorder = _Recorder(_receipts_ok)
    ids = [f"ticket-{index}" for index in range(2500)]
    receipts = _sender(recorder).receipts(ids)
    assert [str(request.url) for request in recorder.requests] == [EXPO_RECEIPTS_URL] * 3
    assert [len(body["ids"]) for body in recorder.bodies()] == [1000, 1000, 500]
    assert set(receipts) == set(ids)
    assert receipts["ticket-7"] == PushReceipt(ticket_id="ticket-7", status="ok", error=None)


def test_expos_headers_are_sent_and_authorization_only_when_the_token_is_set(monkeypatch):
    recorder = _Recorder(_both_ok)
    _sender(recorder).send(_messages(1))
    plain = recorder.requests[-1].headers
    assert plain["accept"] == "application/json"
    assert plain["content-type"] == "application/json"
    assert plain["accept-encoding"] == "gzip"
    assert "authorization" not in plain

    monkeypatch.setattr(config, "EXPO_PUSH_ACCESS_TOKEN", FAKE_ACCESS)
    _sender(recorder).send(_messages(1))
    _sender(recorder).receipts(["ticket-1"])
    assert recorder.requests[-2].headers["authorization"] == f"Bearer {FAKE_ACCESS}"
    assert recorder.requests[-1].headers["authorization"] == f"Bearer {FAKE_ACCESS}"


def test_device_not_registered_is_parsed_from_a_ticket_and_a_receipt():
    def answer(request: httpx.Request, _attempt: int) -> httpx.Response:
        if str(request.url) == EXPO_RECEIPTS_URL:
            return httpx.Response(200, json={"data": {
                "t-1": {"status": "error", "message": f'"{_token(0)}" {EXPO_MESSAGE}',
                        "details": {"error": "DeviceNotRegistered"}},
                "t-2": {"status": "ok"},
                "t-unasked": {"status": "ok"},
            }})
        return httpx.Response(200, json={"data": [
            {"status": "ok", "id": "t-1"},
            {"status": "error", "message": f'"{_token(1)}" {EXPO_MESSAGE}', "details": {"error": "DeviceNotRegistered"}},
        ]})

    sender = _sender(_Recorder(answer))
    tickets = sender.send(_messages(2))
    assert tickets == [
        PushTicket(to=_token(0), status="ok", ticket_id="t-1"),
        PushTicket(to=_token(1), status="error", error="DeviceNotRegistered"),
    ]
    receipts = sender.receipts(["t-1", "t-2"])
    assert receipts == {
        "t-1": PushReceipt(ticket_id="t-1", status="error", error="DeviceNotRegistered"),
        "t-2": PushReceipt(ticket_id="t-2", status="ok", error=None),
    }


def test_an_error_ticket_without_a_plain_code_keeps_no_text_of_expos():
    def answer(_request: httpx.Request, _attempt: int) -> httpx.Response:
        return httpx.Response(200, json={"data": [
            {"status": "error", "message": EXPO_MESSAGE},
            {"status": "error", "details": {"error": f"{_token(1)} {EXPO_MESSAGE}"}},
            {"status": "error", "details": {"error": "DeviceNotRegistered\n"}},
        ]})

    tickets = _sender(_Recorder(answer)).send(_messages(3))
    assert [ticket.error for ticket in tickets] == [UNKNOWN_ERROR] * 3


def test_one_ticket_not_in_expos_shape_is_an_error_for_its_message_only():
    def answer(_request: httpx.Request, _attempt: int) -> httpx.Response:
        return httpx.Response(200, json={"data": [
            {"status": "ok"},
            {"status": "ok", "id": "t-2"},
            {"status": "maybe", "id": "t-3"},
            {"status": "ok", "id": "t 4\n"},
            "nope",
        ]})

    messages = _messages(5)
    tickets = _sender(_Recorder(answer)).send(messages)
    assert tickets == [
        PushTicket(to=messages[0].to, status="error", error=MALFORMED_ANSWER),
        PushTicket(to=messages[1].to, status="ok", ticket_id="t-2"),
        PushTicket(to=messages[2].to, status="error", error=MALFORMED_ANSWER),
        PushTicket(to=messages[3].to, status="error", error=MALFORMED_ANSWER),
        PushTicket(to=messages[4].to, status="error", error=MALFORMED_ANSWER),
    ]


def test_one_receipt_not_in_expos_shape_is_left_out_and_the_others_are_kept():
    def answer(_request: httpx.Request, _attempt: int) -> httpx.Response:
        return httpx.Response(200, json={"data": {
            "t-1": {"status": "error", "details": {"error": "DeviceNotRegistered"}},
            "t-2": {"status": "later"},
            "t-3": "nope",
            "t-4": {"status": "ok"},
        }})

    receipts = _sender(_Recorder(answer)).receipts(["t-1", "t-2", "t-3", "t-4"])
    assert receipts == {
        "t-1": PushReceipt(ticket_id="t-1", status="error", error="DeviceNotRegistered"),
        "t-4": PushReceipt(ticket_id="t-4", status="ok", error=None),
    }


@pytest.mark.parametrize("busy", [503, 429])
def test_a_503_or_429_then_a_200_is_one_retry_after_the_delay(busy):
    def answer(request: httpx.Request, attempt: int) -> httpx.Response:
        return httpx.Response(busy) if attempt == 1 else _all_ok(request, attempt)

    recorder, sleeps = _Recorder(answer), []
    tickets = _sender(recorder, sleeps).send(_messages(3))
    assert len(recorder.requests) == 2
    assert sleeps == [EXPO_RETRY_DELAY_SECONDS]
    assert [ticket.status for ticket in tickets] == ["ok"] * 3


def test_a_second_503_is_not_retried_again_and_fails_the_batch():
    recorder, sleeps = _Recorder(lambda _request, _attempt: httpx.Response(503)), []
    tickets = _sender(recorder, sleeps).send(_messages(2))
    assert len(recorder.requests) == 2
    assert sleeps == [EXPO_RETRY_DELAY_SECONDS]
    assert [(ticket.status, ticket.error) for ticket in tickets] == [("error", REQUEST_FAILED)] * 2


def test_a_400_is_not_retried_and_every_ticket_is_an_error():
    recorder, sleeps = _Recorder(lambda _request, _attempt: httpx.Response(400, json={"errors": [{"code": "X"}]})), []
    messages = _messages(3)
    tickets = _sender(recorder, sleeps).send(messages)
    assert len(recorder.requests) == 1
    assert sleeps == []
    assert tickets == [PushTicket(to=m.to, status="error", error=REQUEST_FAILED) for m in messages]


def test_only_the_failed_batch_is_errors():
    def answer(request: httpx.Request, attempt: int) -> httpx.Response:
        return httpx.Response(400) if attempt == 2 else _all_ok(request, attempt)

    tickets = _sender(_Recorder(answer)).send(_messages(250))
    statuses = [ticket.status for ticket in tickets]
    assert statuses == ["ok"] * 100 + ["error"] * 100 + ["ok"] * 50


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param(lambda: httpx.Response(200, content=b"<html>not json</html>"), id="not-json"),
        pytest.param(lambda: httpx.Response(200, json={"errors": [{"code": "PUSH_TOO_MANY_EXPERIENCE_IDS"}]}),
                     id="request-errors"),
        pytest.param(lambda: httpx.Response(200, json={"data": [{"status": "ok", "id": "t-1"}]}), id="short-list"),
        pytest.param(lambda: httpx.Response(200, json={"data": "nope"}), id="not-a-list"),
        pytest.param(lambda: httpx.Response(200, json=[1, 2]), id="not-an-object"),
    ],
)
def test_a_malformed_answer_gives_error_tickets_without_raising(answer):
    tickets = _sender(_Recorder(lambda _request, _attempt: answer())).send(_messages(2))
    assert [(ticket.status, ticket.error) for ticket in tickets] == [("error", REQUEST_FAILED)] * 2


@pytest.mark.parametrize("data", [{"id": uuid.uuid4()}, {"score": float("nan")}], ids=["uuid", "nan"])
def test_data_that_is_not_json_fails_its_batch_without_raising_or_sending(data):
    recorder = _Recorder(_all_ok)
    good = _messages(EXPO_SEND_BATCH)
    bad = PushMessage(to=_token(999), title=TITLE, body=BODY, data=data)
    tickets = _sender(recorder).send([*good, bad])
    assert len(recorder.requests) == 1
    assert [ticket.status for ticket in tickets[:EXPO_SEND_BATCH]] == ["ok"] * EXPO_SEND_BATCH
    assert tickets[-1] == PushTicket(to=bad.to, status="error", error=REQUEST_FAILED)


def test_a_transport_error_gives_error_tickets_and_no_receipts():
    def refuse(request: httpx.Request, _attempt: int) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    sender = _sender(_Recorder(refuse))
    tickets = sender.send(_messages(2))
    assert [(ticket.status, ticket.error) for ticket in tickets] == [("error", REQUEST_FAILED)] * 2
    assert sender.receipts(["t-1"]) == {}


def test_a_failed_receipts_batch_is_absent_and_the_others_are_kept():
    def answer(request: httpx.Request, attempt: int) -> httpx.Response:
        return httpx.Response(200, content=b"not json") if attempt == 2 else _receipts_ok(request, attempt)

    ids = [f"ticket-{index}" for index in range(2500)]
    receipts = _sender(_Recorder(answer)).receipts(ids)
    assert set(receipts) == set(ids[:1000] + ids[2000:])


def test_the_log_holds_no_token_title_body_or_data(monkeypatch, caplog):
    monkeypatch.setattr(config, "EXPO_PUSH_ACCESS_TOKEN", FAKE_ACCESS)

    def answer(request: httpx.Request, attempt: int) -> httpx.Response:
        if attempt == 1:
            return httpx.Response(200, json={"data": [
                {"status": "error", "message": f'"{_token(0)}" {EXPO_MESSAGE}', "details": {"error": "DeviceNotRegistered"}},
                {"status": "ok", "id": "t-2"},
            ]})
        if attempt == 2:
            return httpx.Response(200, json={"data": [{"status": "ok", "message": f'"{_token(2)}" {EXPO_MESSAGE}'}]})
        raise httpx.ConnectError(f"refused {FAKE_ACCESS}", request=request)

    caplog.set_level(logging.DEBUG, logger=expo.__name__)
    sender = _sender(_Recorder(answer))
    sender.send(_messages(2))
    sender.send(_messages(3)[2:])
    sender.send(_messages(1))
    sender.receipts(["t-2"])
    text = caplog.text
    assert "DeviceNotRegistered" in text
    for secret in (FAKE_ACCESS, "ExponentPushToken", "prd266-", TITLE, BODY, DATA_MARK, EXPO_MESSAGE):
        assert secret not in text
