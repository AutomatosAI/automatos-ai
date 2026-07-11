"""PRD-194 S1 (P2-13, security §1.1 CRITICAL) — webhook reject-on-signature-mismatch.

Until this change, the external webhook lanes failed OPEN on signature
problems: the Composio V3 block logged "allowing through for debugging" on a
mismatch or a verification error, and both the workspace and the playbook
(recipe) lanes silently skipped verification when the signature header was
absent — so a configured secret bought nothing against an attacker who simply
omitted the header. Slack inbound was never verifiable at all: no X-Slack-
Signature scheme existed.

These tests pin the fail-closed contract (the GitHub lane's semantics,
`github_webhooks.py` — the in-tree template): **when a secret is configured,
a valid signature is mandatory — mismatch, verification error, or missing
header ⇒ 401 and nothing dispatches.** No secret configured keeps the
URL-as-secret floor (unchanged posture).

Pure: requests are hand-built Starlette Requests, the DB is a stub (poisoned
where the handler must reject before ever touching it); no network, no real
Composio, no live Slack.
"""

from __future__ import annotations

import os

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import asyncio  # noqa: E402
import base64  # noqa: E402
import hashlib  # noqa: E402
import hmac  # noqa: E402
import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from uuid import uuid4  # noqa: E402

import pytest  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from starlette.requests import Request  # noqa: E402

import tests.conftest as _conftest  # noqa: E402

_conftest._restore_real_app_modules()

from api import composio as composio_api  # noqa: E402
from api import webhooks as webhooks_api  # noqa: E402
from api import workflow_recipes as recipes_api  # noqa: E402


# ---------------------------------------------------------------- helpers

def _make_request(headers: dict, body: bytes = b"{}") -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/webhooks/test",
        "headers": [(k.lower().encode(), str(v).encode()) for k, v in headers.items()],
        "query_string": b"",
    }

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


def _run(coro):
    return asyncio.run(coro)


class _PoisonedDb:
    """A DB the handler must never reach — rejection happens first."""

    def query(self, *args, **kwargs):
        raise AssertionError("db must not be touched when the signature is rejected")


class _StubQuery:
    def __init__(self, single=None, many=None):
        self._single = single
        self._many = many or []

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self._single

    def all(self):
        return self._many


class _RoutingDb:
    """query(Model) routed by model name — workspace lookup + channel rows."""

    def __init__(self, workspace=None, channel_rows=None):
        self._workspace = workspace
        self._channel_rows = channel_rows or []

    def query(self, model, *args, **kwargs):
        if model.__name__ == "ChannelConnection":
            return _StubQuery(many=self._channel_rows)
        return _StubQuery(single=self._workspace)


def _fake_workspace(settings=None):
    return SimpleNamespace(id=uuid4(), is_active=True, settings=settings or {})


# ---------------------------------------------------------------- Composio V3

_COMPOSIO_SECRET_B64 = base64.b64encode(b"composio-test-secret").decode()


def _v3_signature(body: bytes, webhook_id: str, webhook_ts: str, secret_b64: str) -> str:
    signed_content = f"{webhook_id}.{webhook_ts}.".encode() + body
    digest = hmac.new(base64.b64decode(secret_b64), signed_content, hashlib.sha256).digest()
    return "v1," + base64.b64encode(digest).decode()


def test_composio_webhook_mismatch_rejected(monkeypatch):
    """A V3 signature mismatch is a 401 and dispatches nothing (was: 200 + dispatch)."""
    monkeypatch.setattr(composio_api.config, "COMPOSIO_WEBHOOK_SECRET", _COMPOSIO_SECRET_B64)
    req = _make_request(
        {
            "webhook-signature": "v1,definitely-not-the-signature",
            "webhook-id": "wh_1",
            "webhook-timestamp": "1700000000",
        },
        body=b'{"type": "composio.trigger.message", "data": {}}',
    )
    with pytest.raises(HTTPException) as ei:
        _run(composio_api.handle_webhook(request=req, x_composio_signature=None, db=_PoisonedDb()))
    assert ei.value.status_code == 401


def test_composio_webhook_verification_error_rejected(monkeypatch):
    """A verification *error* (garbage headers) rejects too — no fail-open on exception."""
    monkeypatch.setattr(composio_api.config, "COMPOSIO_WEBHOOK_SECRET", "%%% not base64 %%%")
    req = _make_request(
        {"webhook-signature": "v1,whatever", "webhook-id": "wh_1", "webhook-timestamp": "1"},
    )
    with pytest.raises(HTTPException) as ei:
        _run(composio_api.handle_webhook(request=req, x_composio_signature=None, db=_PoisonedDb()))
    assert ei.value.status_code == 401


def test_composio_webhook_missing_signature_rejected_when_secret_set(monkeypatch):
    """Secret configured + no signature header at all ⇒ 401 (was: no verification ran)."""
    monkeypatch.setattr(composio_api.config, "COMPOSIO_WEBHOOK_SECRET", _COMPOSIO_SECRET_B64)
    req = _make_request({}, body=b"{}")
    with pytest.raises(HTTPException) as ei:
        _run(composio_api.handle_webhook(request=req, x_composio_signature=None, db=_PoisonedDb()))
    assert ei.value.status_code == 401


def test_composio_webhook_valid_v3_signature_accepted(monkeypatch):
    """A correctly signed request passes verification (fails later on 400, not 401).

    Timestamp is *fresh*: since PRD-194 S2 a stale provider timestamp is
    rejected by the replay guard even when the signature verifies.
    """
    monkeypatch.setattr(composio_api.config, "COMPOSIO_WEBHOOK_SECRET", _COMPOSIO_SECRET_B64)
    body = b"{}"  # no trigger_name → the handler 400s AFTER the signature gate
    ts = str(int(time.time()))
    sig = _v3_signature(body, "wh_1", ts, _COMPOSIO_SECRET_B64)
    req = _make_request(
        {"webhook-signature": sig, "webhook-id": "wh_1", "webhook-timestamp": ts},
        body=body,
    )
    with pytest.raises(HTTPException) as ei:
        _run(composio_api.handle_webhook(request=req, x_composio_signature=None, db=_PoisonedDb()))
    assert ei.value.status_code == 400  # missing trigger_name — proof the 401 gate passed


# ---------------------------------------------------------------- workspace lane

def test_workspace_webhook_hmac_mandatory_when_secret():
    """Secret-configured workspace lane 401s when the signature header is absent."""
    ws = _fake_workspace(settings={"webhook_secret": "topsecret"})
    req = _make_request({"content-type": "application/json"}, body=b'{"message": "hi"}')
    with pytest.raises(HTTPException) as ei:
        _run(webhooks_api.general_workspace_webhook(workspace_key="k", request=req, db=_RoutingDb(ws)))
    assert ei.value.status_code == 401


def test_verify_webhook_signature_paths():
    """Pure contract of the shared verifier: mandatory-when-secret, floor otherwise."""
    secret = "topsecret"
    body = b'{"x": 1}'
    good = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

    # correct signature (with and without the GitHub sha256= prefix) passes
    _run(webhooks_api._verify_webhook_signature(
        _make_request({"x-webhook-signature": good}, body), secret))
    _run(webhooks_api._verify_webhook_signature(
        _make_request({"x-hub-signature-256": f"sha256={good}"}, body), secret))

    # mismatch ⇒ 401
    with pytest.raises(HTTPException):
        _run(webhooks_api._verify_webhook_signature(
            _make_request({"x-webhook-signature": "0" * 64}, body), secret))

    # secret set + missing header ⇒ 401 (the deleted silent skip)
    with pytest.raises(HTTPException):
        _run(webhooks_api._verify_webhook_signature(_make_request({}, body), secret))

    # no secret ⇒ no-op (URL-as-secret floor unchanged)
    _run(webhooks_api._verify_webhook_signature(_make_request({}, body), None))


# ---------------------------------------------------------------- Slack v0

def _slack_headers(secret: str, body: bytes, ts: int) -> dict:
    base = f"v0:{ts}:".encode() + body
    sig = "v0=" + hmac.new(secret.encode(), base, hashlib.sha256).hexdigest()
    return {"x-slack-signature": sig, "x-slack-request-timestamp": str(ts)}


def test_slack_signature_verified():
    """A valid X-Slack-Signature passes; an invalid one 401s."""
    secret, body = "slack-signing-secret", b'{"type": "event_callback"}'
    now = int(time.time())

    _run(webhooks_api._verify_slack_signature(
        _make_request(_slack_headers(secret, body, now), body), secret))

    with pytest.raises(HTTPException) as ei:
        _run(webhooks_api._verify_slack_signature(
            _make_request(_slack_headers("wrong-secret", body, now), body), secret))
    assert ei.value.status_code == 401


def test_slack_signature_stale_or_garbage_timestamp_rejected():
    secret, body = "slack-signing-secret", b"{}"
    stale = int(time.time()) - 4000
    with pytest.raises(HTTPException):
        _run(webhooks_api._verify_slack_signature(
            _make_request(_slack_headers(secret, body, stale), body), secret))

    headers = _slack_headers(secret, body, int(time.time()))
    headers["x-slack-request-timestamp"] = "not-a-number"
    with pytest.raises(HTTPException):
        _run(webhooks_api._verify_slack_signature(_make_request(headers, body), secret))


def test_slack_lane_enforces_collected_signing_secret():
    """A Slack-signed request to the workspace lane verifies against the
    channel's collected signing secret — a mismatch 401s instead of skipping."""
    ws = _fake_workspace()
    row = SimpleNamespace(config={"signing_secret": "the-real-secret"}, status="active")
    headers = _slack_headers("attacker-secret", b'{"message": "hi"}', int(time.time()))
    headers["content-type"] = "application/json"
    req = _make_request(headers, body=b'{"message": "hi"}')
    with pytest.raises(HTTPException) as ei:
        _run(webhooks_api.general_workspace_webhook(
            workspace_key="k", request=req, db=_RoutingDb(ws, channel_rows=[row])))
    assert ei.value.status_code == 401


def test_slack_header_does_not_bypass_generic_hmac():
    """S1 gap (P2-13): an x-slack-signature header with NO collected Slack
    signing secret must not bypass the mandatory generic HMAC. Before this
    fix the lane branched on the header alone — any caller could skip a
    configured webhook_secret by adding a garbage Slack header."""
    ws = _fake_workspace(settings={"webhook_secret": "topsecret"})
    headers = {"content-type": "application/json", "x-slack-signature": "v0=garbage"}
    req = _make_request(headers, body=b'{"message": "hi"}')
    with pytest.raises(HTTPException) as ei:
        _run(webhooks_api.general_workspace_webhook(
            workspace_key="k", request=req, db=_RoutingDb(ws, channel_rows=[])))
    assert ei.value.status_code == 401


def test_resolve_slack_signing_secret_prefers_active_row():
    ws = _fake_workspace()
    inactive = SimpleNamespace(config={"signing_secret": "old"}, status="inactive")
    active = SimpleNamespace(config={"signing_secret": "new"}, status="active")
    db = _RoutingDb(ws, channel_rows=[inactive, active])
    assert webhooks_api._resolve_slack_signing_secret(db, ws) == "new"
    assert webhooks_api._resolve_slack_signing_secret(_RoutingDb(ws), ws) is None


# ---------------------------------------------------------------- recipe lane

def _fake_recipe(secret: str | None):
    cfg = {"webhook_id": "wh1"}
    if secret:
        cfg["webhook_secret"] = secret
    return SimpleNamespace(schedule_config=cfg, steps=[], name="r", workspace_id=uuid4())


class _RecipeDb:
    def __init__(self, recipe):
        self._recipe = recipe

    def query(self, *args, **kwargs):
        return _StubQuery(single=self._recipe)


def test_recipe_webhook_hmac_mandatory_when_secret():
    """Secret-configured playbook lane 401s on a missing or invalid signature."""
    recipe = _fake_recipe("recipe-secret")

    req = _make_request({"content-type": "application/json"}, body=b"{}")
    with pytest.raises(HTTPException) as ei:
        _run(recipes_api.recipe_webhook(webhook_id="wh1", request=req, db=_RecipeDb(recipe)))
    assert ei.value.status_code == 401

    req = _make_request(
        {"content-type": "application/json", "x-webhook-signature": "0" * 64}, body=b"{}")
    with pytest.raises(HTTPException) as ei:
        _run(recipes_api.recipe_webhook(webhook_id="wh1", request=req, db=_RecipeDb(recipe)))
    assert ei.value.status_code == 401


def test_recipe_webhook_valid_signature_accepted():
    """A correctly signed request passes the gate (fails later on 400 no-steps, not 401)."""
    recipe = _fake_recipe("recipe-secret")
    body = b"{}"
    good = hmac.new(b"recipe-secret", body, hashlib.sha256).hexdigest()
    req = _make_request(
        {"content-type": "application/json", "x-webhook-signature": good}, body=body)
    with pytest.raises(HTTPException) as ei:
        _run(recipes_api.recipe_webhook(webhook_id="wh1", request=req, db=_RecipeDb(recipe)))
    assert ei.value.status_code == 400  # "Recipe has no steps" — past the signature gate
