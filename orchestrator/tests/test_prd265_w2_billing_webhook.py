"""PRD-265 W2 — Stripe's webhook route (``api/billing_webhooks.py``).

No Clerk auth: the credential is the ``Stripe-Signature`` header. 404 in the
local edition; 503 while billing isn't configured (no secret key, or no
webhook signing secret); 400 for a signature that doesn't hold; 200 once
``services.billing.events.receive`` has settled the event; a processing
exception propagates as a 500 so Stripe redelivers.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.billing_webhooks import router
from config import config

PATH = "/api/v1/billing/webhooks/stripe"


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def hosted(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")


@pytest.fixture(autouse=True)
def _no_real_db(monkeypatch):
    # The route opens a session directly (no FastAPI dependency); keep it off any real DB.
    monkeypatch.setattr("core.database.database.SessionLocal", lambda: MagicMock())


def test_webhook_is_hosted_only(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    res = TestClient(_app()).post(PATH, content=b"{}", headers={"stripe-signature": "t=1,v1=x"})
    assert res.status_code == 404


def test_webhook_503_without_a_stripe_secret_key(hosted, monkeypatch):
    monkeypatch.setattr(config, "STRIPE_SECRET_KEY", "")
    res = TestClient(_app()).post(PATH, content=b"{}", headers={"stripe-signature": "t=1,v1=x"})
    assert res.status_code == 503


def test_webhook_503_without_a_webhook_signing_secret(hosted, monkeypatch):
    monkeypatch.setattr(config, "STRIPE_SECRET_KEY", "sk_test_123")
    monkeypatch.setattr(config, "STRIPE_WEBHOOK_SECRET", "")
    res = TestClient(_app()).post(PATH, content=b"{}", headers={"stripe-signature": "t=1,v1=x"})
    assert res.status_code == 503


def test_webhook_400_on_a_bad_signature(hosted, monkeypatch):
    monkeypatch.setattr(config, "STRIPE_SECRET_KEY", "sk_test_123")
    monkeypatch.setattr(config, "STRIPE_WEBHOOK_SECRET", "whsec_test")
    res = TestClient(_app()).post(PATH, content=b'{"id": "evt_1"}', headers={"stripe-signature": "t=1,v1=deadbeef"})
    assert res.status_code == 400
    assert res.json()["detail"] == "Invalid signature"


def test_webhook_200_when_receive_settles_the_event(hosted, monkeypatch):
    seen = {}

    def _receive(db, provider, payload, signature):
        seen["payload"] = payload
        seen["signature"] = signature
        return "processed"

    monkeypatch.setattr(config, "STRIPE_SECRET_KEY", "sk_test_123")
    monkeypatch.setattr(config, "STRIPE_WEBHOOK_SECRET", "whsec_test")
    monkeypatch.setattr("services.billing.events.receive", _receive)
    body = b'{"id": "evt_1", "type": "invoice.paid"}'
    res = TestClient(_app()).post(PATH, content=body, headers={"stripe-signature": "t=1,v1=whatever"})
    assert res.status_code == 200
    assert res.json() == {"received": True, "status": "processed"}
    assert seen == {"payload": body, "signature": "t=1,v1=whatever"}


def test_webhook_processing_exception_propagates_as_500(hosted, monkeypatch):
    def _blow_up(db, provider, payload, signature):
        raise RuntimeError("the ledger write failed")

    monkeypatch.setattr(config, "STRIPE_SECRET_KEY", "sk_test_123")
    monkeypatch.setattr(config, "STRIPE_WEBHOOK_SECRET", "whsec_test")
    monkeypatch.setattr("services.billing.events.receive", _blow_up)
    client = TestClient(_app(), raise_server_exceptions=False)
    res = client.post(PATH, content=b'{"id": "evt_1"}', headers={"stripe-signature": "t=1,v1=whatever"})
    assert res.status_code == 500
