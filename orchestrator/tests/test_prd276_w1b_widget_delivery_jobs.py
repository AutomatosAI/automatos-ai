"""PRD-276 Wave 1b: a shopper's callback is delivered by durable jobs, one per destination.

The contact details travel sealed in the job and are removed from the row the moment it ends, done
or dead; a dead delivery retried after that sends nothing and says why. Each attempt is logged,
a retryable failure re-queues the job, and the last one records the error. A send happens only
under the job's fence. A webhook gets ``X-Automatos-Delivery``, the job's id, on every attempt;
Telegram and Slack never see it. Real database; the channels are fakes.
"""
from __future__ import annotations

import asyncio
import uuid
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from sqlalchemy import text

from channels.drivers import SendResult
from core.jobs import queue
from core.jobs.registry import kind_of, registered
from services.destinations import delivery_jobs as dj
from services.destinations import dispatcher
from services.destinations.base import CallbackPayload

PHONE = "+447700900123"
DESTINATIONS = [{"platform": "telegram"}, {"platform": "webhook", "webhook_url": "https://hooks.example.com/leads"}]


@pytest.fixture(scope="module")
def engine():  # conftest's new_session binds to it
    from sqlalchemy import create_engine

    from core.database.database import get_database_url
    from core.models.jobs import Job
    from core.models.sites import Site
    from core.models.widget_event_log import WidgetEventLog

    eng = create_engine(get_database_url(), pool_pre_ping=True)
    for model in (Job, Site, WidgetEventLog):  # a test database built before the tables
        model.__table__.create(eng, checkfirst=True)
    yield eng
    eng.dispose()


@pytest.fixture
def site(new_session):
    ws, site_id = str(uuid.uuid4()), str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'w1b-widget')"), {"id": ws})
    s.execute(text("INSERT INTO sites (id, workspace_id, type, display_name) VALUES "
                   "(CAST(:id AS uuid), CAST(:w AS uuid), 'custom', 'Acme Fire Doors')"), {"id": site_id, "w": ws})
    s.commit()
    s.close()
    yield SimpleNamespace(id=uuid.UUID(site_id), workspace_id=uuid.UUID(ws))
    s = new_session.sweep()
    s.execute(text("DELETE FROM widget_event_log WHERE site_id = CAST(:id AS uuid)"), {"id": site_id})
    s.execute(text("DELETE FROM jobs.jobs WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws})
    s.execute(text("DELETE FROM sites WHERE id = CAST(:id AS uuid)"), {"id": site_id})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:w AS uuid)"), {"w": ws})
    s.commit()
    s.close()


@pytest.fixture
def channel(monkeypatch):
    """A fake ``send_to_channel``: answers each send with the next result given (``answer``), and
    keeps what it was sent."""
    sent, answers = [], [SendResult(ok=True, latency_ms=40)]

    async def send(*, db, workspace_id, platform, text, target=None, delivery_id=None):
        sent.append(SimpleNamespace(platform=platform, text=text, target=target, delivery_id=delivery_id))
        return answers.pop(0) if len(answers) > 1 else answers[0]

    monkeypatch.setattr("channels.sender.send_to_channel", send)
    errors = MagicMock()
    monkeypatch.setattr(dispatcher, "record_error", errors)
    return SimpleNamespace(sent=sent, errors=errors, answer=lambda *r: answers.__setitem__(slice(None), r))


def _do(new_session, work, *args, **kwargs):
    s = new_session()
    try:
        out = work(s, *args, **kwargs)
        s.commit()
        return out
    finally:
        s.close()


def _request(new_session, site, destinations=DESTINATIONS, request_id="cb_acme01"):
    contact = CallbackPayload(request_id=request_id, name="Jordan Lee", phone=PHONE, product_context="FD30 door set",
                              urgency=None, preferred_time="this afternoon", site_display_name="Acme Fire Doors",
                              site_external_id=None)
    return _do(new_session, dj.queue_deliveries, site=site, session_id="sess-1", request_id=request_id,
               payload=contact, destinations=destinations)


def _claim(new_session):
    return _do(new_session, queue.claim, kind_of(dj.DELIVER_KIND), "w-test", 10)


def _rows(new_session, site):
    return _do(new_session, lambda s: s.execute(text(
        "SELECT idempotency_key, status, payload, payload::text FROM jobs.jobs WHERE workspace_id = CAST(:w AS uuid) "
        "ORDER BY idempotency_key"), {"w": str(site.workspace_id)}).all())


def _events(new_session, site):
    return _do(new_session, lambda s: s.execute(text(
        "SELECT event_type, event_data FROM widget_event_log WHERE site_id = CAST(:id AS uuid) ORDER BY id"),
        {"id": str(site.id)}).all())


def test_one_job_per_destination_with_the_contact_sealed_and_queued_once(new_session, site):
    assert _request(new_session, site) == 2 and _request(new_session, site) == 0
    rows = _rows(new_session, site)
    assert [r[0] for r in rows] == ["cb_acme01:0", "cb_acme01:1"]
    assert all(PHONE not in r[3] and "Jordan Lee" not in r[3] and r[2][dj.CONTACT_KEY] for r in rows)
    assert dj._contact(rows[0][2]).phone == PHONE


def test_a_delivery_is_sent_logged_and_its_contact_wiped(new_session, site, channel):
    _request(new_session, site, [DESTINATIONS[0]])
    (job,) = _claim(new_session)
    assert asyncio.run(dj.deliver_widget_callback(job)) is True
    (sent,) = channel.sent
    assert sent.platform == "telegram" and PHONE in sent.text and sent.delivery_id == str(job.id)
    ((_, status, payload, _),) = _rows(new_session, site)
    assert status == "done" and dj.CONTACT_KEY not in payload and payload["request_id"] == "cb_acme01"
    ((event, data),) = _events(new_session, site)
    assert event == "callback_delivered" and data["attempt"] == 1 and not channel.errors.called


def test_a_retryable_failure_requeues_keeping_the_contact_and_the_last_one_records_the_error(new_session, site,
                                                                                               channel):
    channel.answer(SendResult(ok=False, latency_ms=5, error="telegram 502", retryable=True))
    _request(new_session, site, [DESTINATIONS[0]])
    due = text("UPDATE jobs.jobs SET run_at = clock_timestamp() WHERE workspace_id = CAST(:w AS uuid)")
    for attempt in range(1, dj.DELIVERY_ATTEMPTS + 1):
        _do(new_session, lambda s: s.execute(due, {"w": str(site.workspace_id)}))
        (job,) = _claim(new_session)
        with pytest.raises(dj.NotYetDelivered) if attempt < dj.DELIVERY_ATTEMPTS else nullcontext():
            asyncio.run(dj.deliver_widget_callback(job))
        if attempt < dj.DELIVERY_ATTEMPTS:
            _do(new_session, queue.fail, job, "telegram 502")
            assert dj.CONTACT_KEY in _rows(new_session, site)[0][2]
    ((_, status, payload, _),) = _rows(new_session, site)
    assert status == "done" and dj.CONTACT_KEY not in payload
    events = _events(new_session, site)
    assert [e[1]["permanent"] for e in events] == [False] * (dj.DELIVERY_ATTEMPTS - 1) + [True]
    assert channel.errors.call_count == 1 and channel.errors.call_args.kwargs["workspace_id"] == site.workspace_id


def test_a_dead_job_loses_its_contact_and_a_retry_sends_nothing(new_session, site, channel):
    _request(new_session, site, [DESTINATIONS[0]])
    _do(new_session, lambda s: s.execute(text("UPDATE jobs.jobs SET attempts = max_attempts - 1 "
                                              "WHERE workspace_id = CAST(:w AS uuid)"), {"w": str(site.workspace_id)}))
    (job,) = _claim(new_session)  # its last attempt
    _do(new_session, queue.fail, job, "worker lost")
    ((_, status, payload, _),) = _rows(new_session, site)
    assert status == "dead" and dj.CONTACT_KEY not in payload
    _do(new_session, lambda s: s.execute(text("UPDATE jobs.jobs SET status = 'queued', attempts = 0 WHERE id = :id"),
                                         {"id": job.id}))  # the dead-job route's retry
    (again,) = _claim(new_session)
    assert asyncio.run(dj.deliver_widget_callback(again)) is False and channel.sent == []
    ((event, data),) = _events(new_session, site)
    assert event == "callback_failed" and data["error"] == dj.CONTACT_GONE and data["permanent"] is True


def test_a_closed_workspace_sends_nothing(new_session, site, channel):
    _request(new_session, site, [DESTINATIONS[0]])
    (job,) = _claim(new_session)
    _do(new_session, lambda s: s.execute(text("UPDATE workspaces SET deleted_at = now() WHERE id = CAST(:w AS uuid)"),
                                         {"w": str(site.workspace_id)}))
    assert asyncio.run(dj.deliver_widget_callback(job)) is None and channel.sent == []


def test_the_dashboard_test_send_makes_one_attempt_per_destination(site, channel, monkeypatch):
    channel.answer(SendResult(ok=False, latency_ms=5, error="telegram 502", retryable=True),
                   SendResult(ok=True, latency_ms=9))
    contact = CallbackPayload(request_id="cb_test", name="Test", phone=PHONE, product_context=None, urgency=None,
                              preferred_time=None, site_display_name="Acme Fire Doors", site_external_id=None)
    results = asyncio.run(dispatcher.dispatch_callback_for_site(
        site_id=site.id, workspace_id=site.workspace_id, session_id="dashboard-test", request_id="cb_test",
        payload=contact, destinations=DESTINATIONS))
    assert [r.success for r in results] == [False, True] and len(channel.sent) == 2
    assert all(s.delivery_id is None for s in channel.sent)


def test_a_webhook_gets_the_delivery_id_and_a_chat_driver_never_does(monkeypatch):
    from channels import sender
    from channels.drivers import webhook

    got = []
    transport = httpx.MockTransport(lambda request: got.append(request) or httpx.Response(204))
    real = httpx.AsyncClient
    monkeypatch.setattr(webhook.httpx, "AsyncClient", lambda **kw: real(transport=transport, **kw))
    asyncio.run(webhook.WebhookDriver().send(workspace_id="w", config={}, target="https://hooks.example.com/leads",
                                             text="Callback for Acme", delivery_id="job-7"))
    asyncio.run(webhook.post_webhook("https://hooks.example.com/reports", {"text": "moderation"}))
    assert got[0].headers[webhook.DELIVERY_HEADER] == "job-7" and webhook.DELIVERY_HEADER not in got[1].headers

    route, notice = SimpleNamespace(config={}, target="123"), SimpleNamespace(apply=lambda t: t, text=None)
    chat = SimpleNamespace(AI_NOTICE_IN_PAYLOAD=False, ACCEPTS_DELIVERY_ID=False)
    hook = SimpleNamespace(AI_NOTICE_IN_PAYLOAD=True, ACCEPTS_DELIVERY_ID=True)
    assert "delivery_id" not in sender._driver_kwargs(chat, route, "w", "hi", notice, "job-7")
    assert sender._driver_kwargs(hook, route, "w", "hi", notice, "job-7")["delivery_id"] == "job-7"
    assert "delivery_id" not in sender._driver_kwargs(hook, route, "w", "hi", notice)


def test_the_callback_endpoint_queues_the_jobs():
    from api.widgets import callback

    assert callback.enqueue_callback_dispatch is dj.enqueue_callback_dispatch
    assert dj.DELIVER_KIND in registered()


def test_a_callback_that_cant_be_queued_is_recorded_and_never_fails_the_request(monkeypatch):
    errors = MagicMock()
    monkeypatch.setattr("core.utils.exception_telemetry.record_error", errors)

    def refuses(*args, **kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(dj, "queue_deliveries", refuses)
    site = SimpleNamespace(id=uuid.uuid4(), workspace_id=uuid.uuid4())
    contact = CallbackPayload(request_id="cb_down", name="Jordan Lee", phone=PHONE, product_context=None, urgency=None,
                              preferred_time=None, site_display_name="Acme Fire Doors", site_external_id=None)
    assert dj.enqueue_callback_dispatch(site=site, session_id="s", request_id="cb_down", payload=contact,
                                        destinations=DESTINATIONS) == 0
    assert errors.call_args.kwargs["operation"] == "queue_callback"
