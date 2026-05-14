"""
PRD-008-A Phase 6 — Destination dispatchers + orchestrator tests
====================================================================

Pure-Python unit tests with mocked HTTP clients / SMTP. Verifies:
- Each dispatcher returns DispatchResult (never raises)
- Success / failure / timeout paths
- Retry behaviour with exponential backoff
- Telemetry events written for every attempt
- Permanent vs retryable failure classification
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

import config  # noqa: E402,F401


def _await(coro):
    return asyncio.run(coro)


def _payload(**overrides) -> "CallbackPayload":
    from services.destinations.base import CallbackPayload
    defaults = dict(
        request_id="cb_test123",
        name="James Smith",
        phone="+447700900123",
        product_context="EN 12101-9 panel",
        urgency=None,
        preferred_time=None,
        site_display_name="INBUILD UK",
        site_external_id="inbuilduk.myshopify.com",
    )
    defaults.update(overrides)
    return CallbackPayload(**defaults)


# ===========================================================================
# Email dispatcher
# ===========================================================================

def test_email_dispatch_fails_when_destination_lacks_address():
    from services.destinations.email import dispatch_email

    result = _await(dispatch_email(
        destination={},
        payload=_payload(),
        smtp_send_func=lambda **kw: None,
    ))
    assert result.success is False
    assert result.retryable is False
    assert "address" in result.error.lower()


def test_email_dispatch_fails_when_smtp_not_configured(monkeypatch):
    from services.destinations.email import dispatch_email
    import config as config_mod

    monkeypatch.setattr(config_mod.config, "SMTP_HOST", "", raising=False)

    result = _await(dispatch_email(
        destination={"address": "sales@example.com"},
        payload=_payload(),
        smtp_send_func=lambda **kw: None,
    ))
    assert result.success is False
    assert result.retryable is False
    assert "SMTP not configured" in result.error


def test_email_dispatch_success(monkeypatch):
    from services.destinations.email import dispatch_email
    import config as config_mod

    monkeypatch.setattr(config_mod.config, "SMTP_HOST", "smtp.test.com", raising=False)
    monkeypatch.setattr(config_mod.config, "SMTP_PORT", 587, raising=False)
    monkeypatch.setattr(config_mod.config, "SMTP_USER", "u", raising=False)
    monkeypatch.setattr(config_mod.config, "SMTP_PASSWORD", "p", raising=False)
    monkeypatch.setattr(config_mod.config, "SMTP_FROM", "from@test", raising=False)

    captured = {}
    def fake_send(**kw):
        captured.update(kw)

    result = _await(dispatch_email(
        destination={"address": "sales@example.com"},
        payload=_payload(),
        smtp_send_func=fake_send,
    ))
    assert result.success is True
    assert result.destination_type == "email"
    assert captured["recipient"] == "sales@example.com"
    assert "+447700900123" in captured["body"]
    assert "EN 12101-9 panel" in captured["body"]
    assert "cb_test123" in captured["body"]


def test_email_dispatch_marks_smtp_failure_retryable(monkeypatch):
    from services.destinations.email import dispatch_email
    import config as config_mod

    monkeypatch.setattr(config_mod.config, "SMTP_HOST", "smtp.test.com", raising=False)
    monkeypatch.setattr(config_mod.config, "SMTP_PORT", 587, raising=False)

    def fake_send(**kw):
        raise ConnectionError("SMTP server unreachable")

    result = _await(dispatch_email(
        destination={"address": "sales@example.com"},
        payload=_payload(),
        smtp_send_func=fake_send,
    ))
    assert result.success is False
    assert result.retryable is True
    assert "ConnectionError" in result.error


# ===========================================================================
# Slack webhook dispatcher
# ===========================================================================

def _mock_http_client(*, status_code: int = 200, text: str = "ok", raise_exc=None):
    """Build an AsyncMock httpx.AsyncClient that returns the given response
    (or raises) on .post()/.get()/.put()."""
    response = MagicMock(status_code=status_code, text=text)
    response.json.return_value = {"customers": []}

    client = MagicMock()
    if raise_exc:
        client.post = AsyncMock(side_effect=raise_exc)
        client.get = AsyncMock(side_effect=raise_exc)
        client.put = AsyncMock(side_effect=raise_exc)
    else:
        client.post = AsyncMock(return_value=response)
        client.get = AsyncMock(return_value=response)
        client.put = AsyncMock(return_value=response)
    client.aclose = AsyncMock(return_value=None)
    return client


def test_slack_dispatch_rejects_non_https_url():
    from services.destinations.slack import dispatch_slack_webhook

    result = _await(dispatch_slack_webhook(
        destination={"url": "http://insecure"}, payload=_payload(),
    ))
    assert result.success is False
    assert result.retryable is False


def test_slack_dispatch_success_on_200():
    from services.destinations.slack import dispatch_slack_webhook

    client = _mock_http_client(status_code=200, text="ok")
    result = _await(dispatch_slack_webhook(
        destination={"url": "https://hooks.slack.com/abc"},
        payload=_payload(),
        http_client=client,
    ))
    assert result.success is True
    # Verify the payload included the right fields
    posted = client.post.await_args.kwargs["json"]
    assert "INBUILD UK" in posted["text"]
    field_titles = {f["title"] for f in posted["attachments"][0]["fields"]}
    assert {"Name", "Phone", "Product"}.issubset(field_titles)


def test_slack_dispatch_marks_4xx_permanent():
    from services.destinations.slack import dispatch_slack_webhook

    client = _mock_http_client(status_code=404, text="not found")
    result = _await(dispatch_slack_webhook(
        destination={"url": "https://hooks.slack.com/abc"},
        payload=_payload(),
        http_client=client,
    ))
    assert result.success is False
    assert result.retryable is False


def test_slack_dispatch_marks_5xx_retryable():
    from services.destinations.slack import dispatch_slack_webhook

    client = _mock_http_client(status_code=503, text="service unavailable")
    result = _await(dispatch_slack_webhook(
        destination={"url": "https://hooks.slack.com/abc"},
        payload=_payload(),
        http_client=client,
    ))
    assert result.success is False
    assert result.retryable is True


def test_slack_dispatch_handles_network_exception():
    from services.destinations.slack import dispatch_slack_webhook
    import httpx

    client = _mock_http_client(raise_exc=httpx.ConnectError("dns fail"))
    result = _await(dispatch_slack_webhook(
        destination={"url": "https://hooks.slack.com/abc"},
        payload=_payload(),
        http_client=client,
    ))
    assert result.success is False
    assert result.retryable is True


# ===========================================================================
# CRM webhook dispatcher
# ===========================================================================

def test_crm_dispatch_includes_auth_header_when_configured():
    from services.destinations.crm import dispatch_crm_webhook

    client = _mock_http_client(status_code=200)
    _await(dispatch_crm_webhook(
        destination={
            "url": "https://api.crm.example.com/leads",
            "auth_header": "Bearer abc123",
        },
        payload=_payload(),
        http_client=client,
    ))
    headers = client.post.await_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer abc123"


def test_crm_dispatch_emits_stable_contract_shape():
    from services.destinations.crm import dispatch_crm_webhook

    client = _mock_http_client(status_code=200)
    _await(dispatch_crm_webhook(
        destination={"url": "https://api.crm.example.com/leads"},
        payload=_payload(),
        http_client=client,
    ))
    body = client.post.await_args.kwargs["json"]
    assert body["event"] == "automatos.callback_requested"
    assert body["version"] == "1"
    assert body["request_id"] == "cb_test123"
    assert body["lead"]["name"] == "James Smith"
    assert body["lead"]["phone"] == "+447700900123"
    assert body["site"]["external_id"] == "inbuilduk.myshopify.com"


def test_crm_dispatch_marks_401_permanent():
    """Bad auth header — retrying won't help. Dashboard flags + merchant fixes."""
    from services.destinations.crm import dispatch_crm_webhook

    client = _mock_http_client(status_code=401, text="unauthorized")
    result = _await(dispatch_crm_webhook(
        destination={"url": "https://api.crm.example.com", "auth_header": "Bearer bad"},
        payload=_payload(),
        http_client=client,
    ))
    assert result.success is False
    assert result.retryable is False


# ===========================================================================
# Shopify customer-note dispatcher
# ===========================================================================

def test_shopify_note_fails_without_token():
    from services.destinations.shopify_note import dispatch_shopify_customer_note

    result = _await(dispatch_shopify_customer_note(
        destination={"type": "shopify_customer_note"},
        payload=_payload(),
        shop_domain="x.myshopify.com",
        access_token="",
    ))
    assert result.success is False
    assert result.retryable is False
    assert "access_token" in result.error or "shop_domain" in result.error


def test_shopify_note_updates_existing_customer():
    """Customer found → PUT note onto existing customer."""
    from services.destinations.shopify_note import dispatch_shopify_customer_note

    response = MagicMock(status_code=200, text="ok")
    response.json.return_value = {
        "customers": [{"id": 12345, "note": "previous note"}]
    }
    update_response = MagicMock(status_code=200, text="ok")
    update_response.json.return_value = {"customer": {"id": 12345}}

    client = MagicMock()
    client.get = AsyncMock(return_value=response)
    client.put = AsyncMock(return_value=update_response)
    client.post = AsyncMock()
    client.aclose = AsyncMock(return_value=None)

    result = _await(dispatch_shopify_customer_note(
        destination={"type": "shopify_customer_note"},
        payload=_payload(),
        shop_domain="x.myshopify.com",
        access_token="shpat_xxx",
        http_client=client,
    ))
    assert result.success is True
    assert result.extra["customer_op"] == "updated"
    # Note appended, not replaced
    put_body = client.put.await_args.kwargs["json"]
    assert "previous note" in put_body["customer"]["note"]
    assert "cb_test123" in put_body["customer"]["note"]


def test_shopify_note_creates_customer_when_none_found():
    from services.destinations.shopify_note import dispatch_shopify_customer_note

    search_response = MagicMock(status_code=200, text="ok")
    search_response.json.return_value = {"customers": []}  # no match
    create_response = MagicMock(status_code=201, text="created")
    create_response.json.return_value = {"customer": {"id": 99}}

    client = MagicMock()
    client.get = AsyncMock(return_value=search_response)
    client.post = AsyncMock(return_value=create_response)
    client.put = AsyncMock()
    client.aclose = AsyncMock(return_value=None)

    result = _await(dispatch_shopify_customer_note(
        destination={"type": "shopify_customer_note"},
        payload=_payload(),
        shop_domain="x.myshopify.com",
        access_token="shpat_xxx",
        http_client=client,
    ))
    assert result.success is True
    assert result.extra["customer_op"] == "created"
    post_body = client.post.await_args.kwargs["json"]
    assert post_body["customer"]["phone"] == "+447700900123"
    assert post_body["customer"]["first_name"] == "James"
    assert post_body["customer"]["last_name"] == "Smith"


def test_shopify_note_401_is_permanent():
    """Bad access token — never recoverable without merchant action."""
    from services.destinations.shopify_note import dispatch_shopify_customer_note

    response = MagicMock(status_code=401, text="unauthorized")
    response.json.return_value = {}
    client = MagicMock()
    client.get = AsyncMock(return_value=response)
    client.aclose = AsyncMock(return_value=None)

    result = _await(dispatch_shopify_customer_note(
        destination={"type": "shopify_customer_note"},
        payload=_payload(),
        shop_domain="x.myshopify.com",
        access_token="bad_token",
        http_client=client,
    ))
    assert result.success is False
    assert result.retryable is False
    assert "401" in result.error


# ===========================================================================
# Orchestrator — dispatch_one_destination retry logic
# ===========================================================================

def test_orchestrator_unknown_destination_type_logs_permanent_failure():
    from services.destinations.dispatcher import dispatch_one_destination

    db = MagicMock()

    async def run():
        return await dispatch_one_destination(
            db=db,
            site_id=uuid4(),
            session_id="s",
            request_id="cb_x",
            destination={"type": "carrier_pigeon"},
            payload=_payload(),
        )

    result = _await(run())
    assert result.success is False
    assert result.retryable is False
    # Telemetry was attempted (one row)
    db.add.assert_called()


def test_orchestrator_succeeds_on_first_attempt():
    from services.destinations import dispatcher as disp_mod
    from services.destinations.base import DispatchResult

    success = DispatchResult(success=True, destination_type="slack_webhook", latency_ms=42)
    db = MagicMock()

    async def fake_dispatcher(*, destination, payload):
        return success

    with patch.object(disp_mod, "_resolve_dispatcher", return_value=fake_dispatcher):
        result = _await(disp_mod.dispatch_one_destination(
            db=db,
            site_id=uuid4(),
            session_id="s",
            request_id="cb_x",
            destination={"type": "slack_webhook", "url": "https://x"},
            payload=_payload(),
        ))

    assert result.success is True
    # Telemetry: callback_delivered written
    row = db.add.call_args[0][0]
    assert row.event_type == "callback_delivered"


def test_orchestrator_retries_on_retryable_failure(monkeypatch):
    """Retryable failure → orchestrator retries up to MAX_ATTEMPTS."""
    from services.destinations import dispatcher as disp_mod
    from services.destinations.base import DispatchResult

    # Make backoff instant for the test
    monkeypatch.setattr(disp_mod, "BACKOFF_SECONDS", (0, 0, 0))

    calls = []
    db = MagicMock()

    async def flaky(*, destination, payload):
        calls.append(1)
        # First two attempts fail, third succeeds
        if len(calls) < 3:
            return DispatchResult(
                success=False, destination_type="x", latency_ms=10,
                error="temporary", retryable=True,
            )
        return DispatchResult(success=True, destination_type="x", latency_ms=10)

    with patch.object(disp_mod, "_resolve_dispatcher", return_value=flaky):
        result = _await(disp_mod.dispatch_one_destination(
            db=db,
            site_id=uuid4(),
            session_id="s",
            request_id="cb_x",
            destination={"type": "x"},
            payload=_payload(),
        ))

    assert result.success is True
    assert len(calls) == 3
    # Telemetry: 2 failed + 1 delivered = 3 rows
    assert db.add.call_count == 3


def test_orchestrator_bails_on_permanent_failure(monkeypatch):
    """Permanent failure → no retry, single attempt logged."""
    from services.destinations import dispatcher as disp_mod
    from services.destinations.base import DispatchResult

    monkeypatch.setattr(disp_mod, "BACKOFF_SECONDS", (0, 0, 0))

    calls = []
    db = MagicMock()

    async def always_permanent(*, destination, payload):
        calls.append(1)
        return DispatchResult(
            success=False, destination_type="x", latency_ms=10,
            error="bad auth", retryable=False,
        )

    with patch.object(disp_mod, "_resolve_dispatcher", return_value=always_permanent):
        _await(disp_mod.dispatch_one_destination(
            db=db,
            site_id=uuid4(),
            session_id="s",
            request_id="cb_x",
            destination={"type": "x"},
            payload=_payload(),
        ))

    assert len(calls) == 1
    row = db.add.call_args[0][0]
    assert row.event_type == "callback_failed"
    assert row.event_data["permanent"] is True


def test_orchestrator_exhausts_retries_on_persistent_retryable_failure(monkeypatch):
    """Always-failing retryable → MAX_ATTEMPTS calls, then gives up."""
    from services.destinations import dispatcher as disp_mod
    from services.destinations.base import DispatchResult

    monkeypatch.setattr(disp_mod, "BACKOFF_SECONDS", (0, 0, 0))
    monkeypatch.setattr(disp_mod, "MAX_ATTEMPTS", 3)

    calls = []
    db = MagicMock()

    async def always_fails(*, destination, payload):
        calls.append(1)
        return DispatchResult(
            success=False, destination_type="x", latency_ms=10,
            error="500", retryable=True,
        )

    with patch.object(disp_mod, "_resolve_dispatcher", return_value=always_fails):
        result = _await(disp_mod.dispatch_one_destination(
            db=db,
            site_id=uuid4(),
            session_id="s",
            request_id="cb_x",
            destination={"type": "x"},
            payload=_payload(),
        ))

    assert result.success is False
    assert len(calls) == 3  # MAX_ATTEMPTS


# ===========================================================================
# Orchestrator — dispatch_callback_for_site fans out
# ===========================================================================

def test_fan_out_dispatches_each_destination_in_parallel(monkeypatch):
    from services.destinations import dispatcher as disp_mod
    from services.destinations.base import DispatchResult

    monkeypatch.setattr(disp_mod, "BACKOFF_SECONDS", (0, 0, 0))

    seen_destinations = []

    async def fake_one(*, db, site_id, session_id, request_id, destination, payload, **kw):
        seen_destinations.append(destination["type"])
        return DispatchResult(success=True, destination_type=destination["type"], latency_ms=1)

    with patch.object(disp_mod, "dispatch_one_destination", new=fake_one), \
         patch.object(disp_mod, "SessionLocal", return_value=MagicMock()):
        results = _await(disp_mod.dispatch_callback_for_site(
            site_id=uuid4(),
            session_id="s",
            request_id="cb_x",
            payload=_payload(),
            destinations=[
                {"type": "email", "address": "a@b"},
                {"type": "slack_webhook", "url": "https://x"},
                {"type": "crm_webhook", "url": "https://y"},
            ],
        ))

    assert len(results) == 3
    assert set(seen_destinations) == {"email", "slack_webhook", "crm_webhook"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
