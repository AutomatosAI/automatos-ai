"""Destinations a workspace user saves go through the outbound guard on a shared server.

A channel webhook (``channels/drivers/webhook.post_webhook``, also the moderation
post) and a credential test (``core/credentials/tester.CredentialTester``: Postgres,
Redis, SSH and the HTTP testers) reach an address a workspace user typed, from the
server's own network position. Where several people share the server
(``editions.current().multi_person``: saas, enterprise), every address the name
resolves to is checked against the private, loopback, link-local and metadata ranges
(``core/security/web_access.py``), nothing is sent to a refused one, and an allowed
one is connected to at the checked address with the name kept as ``Host`` / SNI. On
the single-operator local install the destination is used as given, as before.

No network: DNS is faked through ``web_access._getaddrinfo`` and every client is a
recorder or an ``httpx.MockTransport``.
"""
from __future__ import annotations

import asyncio
import os
import socket
import ssl
from typing import Any, Dict, List

import httpx
import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import core.security.web_access as wa  # noqa: E402
from channels.drivers import webhook  # noqa: E402
from config import config  # noqa: E402
from core.credentials import destinations, tester as tester_mod  # noqa: E402
from core.credentials.service import CredentialStore  # noqa: E402
from core.credentials.tester import CredentialTester  # noqa: E402
from tests.helpers_outbound_dns import PUBLIC_IP, fake_dns  # noqa: E402

DNS = {
    "hooks.example.com": PUBLIC_IP,
    "db.example.com": PUBLIC_IP,
    "internal.example.com": "10.0.0.5",
    "loopback.example.com": "127.0.0.1",
    "metadata.example.com": "169.254.169.254",
}
BLOCKED_HOSTS = ("internal.example.com", "loopback.example.com", "metadata.example.com")
SHARED_EDITIONS = ("saas", "enterprise")


def _no_dns(*_args, **_kwargs):
    raise AssertionError("a single-operator install must not resolve the destination")


@pytest.fixture(autouse=True)
def dns(monkeypatch):
    fake_dns(monkeypatch, DNS)
    monkeypatch.setattr(config, "WEB_ACCESS_DENY", (), raising=False)


@pytest.fixture
def edition(monkeypatch):
    return lambda name: monkeypatch.setattr(config, "AUTH_EDITION", name)


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- the rule and the host check


@pytest.mark.parametrize("name,guarded", [("local", False), ("saas", True), ("enterprise", True)])
def test_destinations_are_guarded_wherever_several_people_share_the_server(edition, name, guarded):
    edition(name)
    assert wa.destinations_guarded() is guarded


def test_resolve_host_refuses_each_blocked_range_and_pins_a_public_address():
    for host in BLOCKED_HOSTS:
        target = wa.resolve_host(host, 5432)
        assert not target.ok and "blocked range" in target.reason
    target = wa.resolve_host("DB.example.com.", 5432)
    assert (target.ok, target.host, target.ip, target.port, target.pinned) == (
        True, "db.example.com", PUBLIC_IP, 5432, True)


def test_resolve_host_checks_every_answer_not_just_the_first(monkeypatch):
    monkeypatch.setattr(wa, "_getaddrinfo", lambda host, port, proto=None: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_IP, port)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.1.2.3", port)),
    ])
    assert not wa.resolve_host("mixed.example.com", 6379).ok


def test_resolve_host_refuses_a_denied_host_a_bad_port_and_no_host(monkeypatch):
    monkeypatch.setattr(config, "WEB_ACCESS_DENY", ("example.com",), raising=False)
    assert "WEB_ACCESS_DENY" in wa.resolve_host("db.example.com", 5432).reason
    monkeypatch.setattr(config, "WEB_ACCESS_DENY", (), raising=False)
    assert wa.resolve_host("db.example.com", 0).reason == "Malformed port"
    assert wa.resolve_host("db.example.com", 70000).reason == "Malformed port"
    assert wa.resolve_host("  ", 5432).reason == "No hostname given"


def test_a_single_operator_install_uses_the_destination_as_given(edition, monkeypatch):
    edition("local")
    monkeypatch.setattr(wa, "_getaddrinfo", _no_dns)
    url_target = _run(wa.operator_destination_async("http://127.0.0.1:5678/hook"))
    host_target = _run(wa.operator_host_async("localhost", 5432))
    assert url_target.ok and not url_target.pinned
    assert host_target.ok and not host_target.pinned and host_target.host == "localhost"


# --------------------------------------------------------------------------- the channel webhook


def _webhook_client(monkeypatch, status: int = 204) -> List[httpx.Request]:
    sent: List[httpx.Request] = []
    transport = httpx.MockTransport(lambda request: sent.append(request) or httpx.Response(status))
    real = httpx.AsyncClient
    monkeypatch.setattr(webhook.httpx, "AsyncClient", lambda **kw: real(transport=transport, **kw))
    return sent


def _no_client(**_kwargs):
    raise AssertionError("no HTTP client may be built for a refused destination")


@pytest.mark.parametrize("name", SHARED_EDITIONS)
@pytest.mark.parametrize("host", BLOCKED_HOSTS)
def test_a_webhook_into_a_blocked_range_is_refused_without_a_request(edition, monkeypatch, name, host):
    edition(name)
    monkeypatch.setattr(webhook.httpx, "AsyncClient", _no_client)
    result = _run(webhook.post_webhook(f"https://{host}/hook", {"text": "hi"}))
    assert result.ok is False and result.retryable is False
    assert "blocked range" in result.error


def test_the_webhook_driver_send_is_refused_too(edition, monkeypatch):
    edition("saas")
    monkeypatch.setattr(webhook.httpx, "AsyncClient", _no_client)
    result = _run(webhook.WebhookDriver().send(
        workspace_id="w", config={"webhook_url": "http://metadata.example.com/latest"}, target=None, text="hi"))
    assert result.ok is False and result.retryable is False


def test_a_public_webhook_is_sent_to_the_checked_address_and_no_redirect_is_followed(edition, monkeypatch):
    edition("saas")
    sent = _webhook_client(monkeypatch, status=302)
    result = _run(webhook.post_webhook("https://hooks.example.com/leads?k=1", {"text": "hi"}, delivery_id="d-1"))
    assert result.ok is True and len(sent) == 1
    request = sent[0]
    assert request.url.host == PUBLIC_IP and request.url.path == "/leads"
    assert request.headers["Host"] == "hooks.example.com"
    assert request.extensions["sni_hostname"] == "hooks.example.com"
    assert request.headers[webhook.DELIVERY_HEADER] == "d-1"


def test_a_local_install_still_posts_to_a_private_webhook(edition, monkeypatch):
    edition("local")
    monkeypatch.setattr(wa, "_getaddrinfo", _no_dns)
    sent = _webhook_client(monkeypatch)
    result = _run(webhook.post_webhook("http://127.0.0.1:5678/webhook/n8n", {"text": "hi"}))
    assert result.ok is True
    assert str(sent[0].url) == "http://127.0.0.1:5678/webhook/n8n"


# --------------------------------------------------------------------------- the credential testers


class _Conn:
    async def fetchval(self, _query):
        return "PostgreSQL 16"

    async def close(self):
        return None


def _record_postgres(monkeypatch) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []

    async def connect(**kwargs):
        calls.append(kwargs)
        return _Conn()

    monkeypatch.setattr(tester_mod.asyncpg, "connect", connect)
    return calls


def _postgres(host: str, **extra) -> Dict[str, Any]:
    return {"host": host, "port": 5432, "database": "d", "user": "u", "password": "p", **extra}


def _test(credential_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
    return _run(CredentialTester().test_credential(credential_type, data))


@pytest.mark.parametrize("host", BLOCKED_HOSTS)
def test_the_postgres_test_refuses_a_private_host_without_connecting(edition, monkeypatch, host):
    edition("saas")
    calls = _record_postgres(monkeypatch)
    result = _test("postgres", _postgres(host))
    assert result["success"] is False and "blocked range" in result["message"]
    assert calls == []


def test_the_postgres_test_connects_to_the_checked_address_with_tls_named_for_the_host(edition, monkeypatch):
    edition("saas")
    calls = _record_postgres(monkeypatch)
    result = _test("postgres", _postgres("db.example.com", ssl="verify-full"))
    assert result["success"] is True
    assert calls[0]["host"] == PUBLIC_IP
    context = calls[0]["ssl"]
    assert isinstance(context, ssl.SSLContext) and context.check_hostname
    sslobj = context.wrap_bio(ssl.MemoryBIO(), ssl.MemoryBIO(), server_hostname=PUBLIC_IP)
    assert sslobj.server_hostname == "db.example.com"


def test_postgres_tls_keeps_each_mode():
    assert destinations.postgres_tls("disable", False, "h") is False
    assert destinations.postgres_tls("prefer", False, "h") is None
    loose = destinations.postgres_tls("require", False, "h")
    assert loose.verify_mode == ssl.CERT_NONE and not loose.check_hostname
    assert destinations.postgres_tls("verify-ca", False, "h").verify_mode == ssl.CERT_REQUIRED


def test_a_local_install_still_tests_its_own_postgres(edition, monkeypatch):
    edition("local")
    monkeypatch.setattr(wa, "_getaddrinfo", _no_dns)
    calls = _record_postgres(monkeypatch)
    assert _test("postgres", _postgres("localhost"))["success"] is True
    assert calls[0]["host"] == "localhost"


class _Redis:
    built: List[Dict[str, Any]] = []

    def __init__(self, **kwargs):
        _Redis.built.append(kwargs)

    def ping(self):
        return True

    def info(self):
        return {"redis_version": "7"}


@pytest.fixture
def redis_recorder(monkeypatch):
    _Redis.built = []
    monkeypatch.setattr(tester_mod.redis, "Redis", _Redis)
    return _Redis.built


@pytest.mark.parametrize("host", BLOCKED_HOSTS)
def test_the_redis_test_refuses_a_private_host_without_connecting(edition, redis_recorder, host):
    edition("enterprise")
    result = _test("redis", {"host": host, "port": 6379})
    assert result["success"] is False and "blocked range" in result["message"]
    assert redis_recorder == []


def test_the_redis_test_connects_to_the_checked_address(edition, redis_recorder):
    edition("saas")
    assert _test("redis", {"host": "db.example.com", "port": 6379})["success"] is True
    assert redis_recorder[0]["host"] == PUBLIC_IP


def test_a_local_install_still_tests_its_own_redis(edition, redis_recorder, monkeypatch):
    edition("local")
    monkeypatch.setattr(wa, "_getaddrinfo", _no_dns)
    assert _test("redis", {"host": "127.0.0.1", "port": 6379})["success"] is True
    assert redis_recorder[0]["host"] == "127.0.0.1"


def test_the_ssh_test_checks_the_resolved_address_not_the_name(edition, monkeypatch):
    """``internal.example.com`` looks public as a string; it resolves to 10.0.0.5."""
    edition("saas")

    def no_ssh():
        raise AssertionError("no SSH client may be built for a refused host")

    monkeypatch.setattr(tester_mod.paramiko, "SSHClient", no_ssh)
    result = _test("ssh_credentials", {"host": "internal.example.com", "username": "u", "password": "p"})
    assert result["success"] is False and "blocked range" in result["message"]


class _Response:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def json(self):
        return {"data": []}

    async def text(self):
        return ""


class _Session:
    def __init__(self):
        self.calls: List[tuple] = []

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return _Response()

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return _Response()


HTTP_TESTS = [
    ("openai_api", lambda base: {"api_key": "k", "base_url": base}),
    ("anthropic_api", lambda base: {"api_key": "k", "base_url": base}),
    ("generic_api", lambda base: {"api_key": "k", "base_url": base}),
    ("discord_webhook", lambda base: {"webhook_url": base + "/api/webhooks/1/x"}),
]


def _http_test(credential_type: str, data: Dict[str, Any]):
    tester = CredentialTester()
    tester.session = _Session()
    return _run(tester.test_credential(credential_type, data)), tester.session.calls


@pytest.mark.parametrize("credential_type,data", HTTP_TESTS)
@pytest.mark.parametrize("host", BLOCKED_HOSTS)
def test_an_http_tester_refuses_a_private_destination_without_a_request(edition, credential_type, data, host):
    edition("saas")
    result, calls = _http_test(credential_type, data(f"https://{host}"))
    assert result["success"] is False and "blocked range" in result["message"]
    assert calls == []


@pytest.mark.parametrize("credential_type,data", HTTP_TESTS)
def test_an_http_tester_sends_to_the_checked_address_and_follows_no_redirect(edition, credential_type, data):
    edition("saas")
    _result, calls = _http_test(credential_type, data("https://hooks.example.com"))
    _method, url, kwargs = calls[0]
    assert httpx.URL(url).host == PUBLIC_IP
    assert kwargs["headers"]["Host"] == "hooks.example.com"
    assert kwargs["server_hostname"] == "hooks.example.com"
    assert kwargs["allow_redirects"] is False


def test_a_local_install_still_tests_a_private_api(edition, monkeypatch):
    edition("local")
    monkeypatch.setattr(wa, "_getaddrinfo", _no_dns)
    result, calls = _http_test("openai_api", {"api_key": "k", "base_url": "http://127.0.0.1:11434/v1"})
    assert result["success"] is True
    assert calls[0][1] == "http://127.0.0.1:11434/v1/models" and "server_hostname" not in calls[0][2]


def test_the_store_has_no_unguarded_test_paths_left():
    """``_test_api_call`` and ``_test_database_connection`` had no caller and reached a
    user's address with no check; credential tests go through ``CredentialTester``."""
    assert not hasattr(CredentialStore, "_test_api_call")
    assert not hasattr(CredentialStore, "_test_database_connection")
