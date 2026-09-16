"""The heartbeat webhook sink refuses blocked ranges and pins the checked address (2026-09-16).

``HeartbeatService._route_heartbeat_to_webhook`` POSTs a heartbeat result to
whatever URL a workspace editor typed into the agent's heartbeat form, from
the API worker's own network position. Unlocking the heartbeat routes for
workspace roles made that form work for everyone but the platform super
admin, so the sink now goes through the PRD-240 seam
(``core/security/web_access.py``): resolved once, every answer checked against
the blocked ranges and the operator denylist, the POST sent to the address that
was checked with the hostname as ``Host`` and SNI, redirects never followed.
The ``WEB_ACCESS`` switch does not govern it — an operator's destination is
not agent web access.

No network: DNS is faked, the client is an ``httpx.MockTransport`` that
records what was actually SENT (the assertion that matters for SSRF).
"""
from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import socket
from typing import List

import httpx
import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import core.security.web_access as wa  # noqa: E402
from services.heartbeat_service import HeartbeatService  # noqa: E402

PUBLIC_IP = "93.184.216.34"
PRIVATE_IP = "10.0.0.5"
_DNS = {"hooks.example": PUBLIC_IP, "evil.example": PRIVATE_IP}


def _fake_getaddrinfo(host, port, proto=None):
    ip = _DNS.get(host)
    if ip is None:
        try:
            ip = str(ipaddress.ip_address(host))
        except ValueError:
            raise socket.gaierror(f"no fake DNS for {host}")
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]


@pytest.fixture(autouse=True)
def fake_dns(monkeypatch):
    monkeypatch.setattr(wa, "_getaddrinfo", _fake_getaddrinfo)


def _stub_client(monkeypatch, handler) -> List[httpx.Request]:
    """Route the sink's client through a MockTransport; return the requests SENT."""
    sent: List[httpx.Request] = []

    def _recording(req: httpx.Request):
        sent.append(req)
        return handler(req)

    transport = httpx.MockTransport(_recording)
    monkeypatch.setattr(
        HeartbeatService,
        "_webhook_client",
        staticmethod(lambda: httpx.AsyncClient(transport=transport, follow_redirects=False)),
    )
    return sent


def _deliver(url: str) -> None:
    asyncio.run(
        HeartbeatService()._route_heartbeat_to_webhook(
            webhook_url=url,
            title="WATCHTOWER Heartbeat",
            message="ok",
            agent_id=578,
            agent_name="WATCHTOWER",
            link_id=None,
            status="success",
        )
    )


def test_public_webhook_is_posted_to_the_checked_address_with_the_name(monkeypatch):
    sent = _stub_client(monkeypatch, lambda req: httpx.Response(200, text="ok"))
    _deliver("https://hooks.example/heartbeat")
    (req,) = sent
    assert req.method == "POST"
    assert req.url.host == PUBLIC_IP  # connected to the address that was checked
    assert req.url.path == "/heartbeat"
    assert req.headers["host"] == "hooks.example"
    assert req.extensions["sni_hostname"] == "hooks.example"
    body = json.loads(req.content)
    assert body["event"] == "heartbeat_complete"
    assert body["agent_id"] == 578 and body["agent_name"] == "WATCHTOWER"


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.example/hook",              # a name that resolves inside
        "http://169.254.169.254/latest/meta-data",  # the metadata service
        "http://127.0.0.1:5432/",                 # loopback
        "http://postgres.internal.test:5432/",    # NXDOMAIN — never sent either
        "ftp://hooks.example/hook",               # wrong scheme
    ],
)
def test_a_webhook_that_points_inside_is_never_sent(url, monkeypatch):
    sent = _stub_client(monkeypatch, lambda req: httpx.Response(200, text="ok"))
    _deliver(url)
    assert sent == []


def test_a_redirect_is_reported_not_followed(monkeypatch):
    sent = _stub_client(
        monkeypatch,
        lambda req: httpx.Response(302, headers={"location": "http://169.254.169.254/latest/"}),
    )
    _deliver("https://hooks.example/heartbeat")
    assert len(sent) == 1  # the hop into the metadata range was never requested


def test_switch_off_does_not_govern_webhooks_but_the_denylist_does(monkeypatch):
    sent = _stub_client(monkeypatch, lambda req: httpx.Response(200, text="ok"))
    monkeypatch.setattr(wa.config, "WEB_ACCESS", False)
    _deliver("https://hooks.example/heartbeat")
    assert len(sent) == 1

    monkeypatch.setattr(wa.config, "WEB_ACCESS_DENY", ("hooks.example",))
    _deliver("https://hooks.example/heartbeat")
    assert len(sent) == 1  # refused: nothing new was sent


def test_a_resolver_that_never_answers_is_a_refusal(monkeypatch):
    import time

    sent = _stub_client(monkeypatch, lambda req: httpx.Response(200, text="ok"))
    monkeypatch.setattr(wa, "RESOLVE_TIMEOUT_SECONDS", 0.05)

    def _slow(host, port, proto=None):
        time.sleep(0.3)
        return _fake_getaddrinfo(host, port, proto)

    monkeypatch.setattr(wa, "_getaddrinfo", _slow)
    _deliver("https://hooks.example/heartbeat")
    assert sent == []  # refused, not stalled


def test_the_real_client_never_follows_redirects():
    client = HeartbeatService._webhook_client()
    try:
        assert client.follow_redirects is False
    finally:
        asyncio.run(client.aclose())


def test_delivery_failure_is_logged_never_raised(monkeypatch):
    def _boom(req):
        raise httpx.ConnectError("refused")

    _stub_client(monkeypatch, _boom)
    _deliver("https://hooks.example/heartbeat")  # no exception escapes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
