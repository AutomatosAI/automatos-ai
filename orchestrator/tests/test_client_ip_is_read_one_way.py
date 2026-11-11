"""The caller's address is read one way everywhere (``core.security.client_ip``).

With ``CLIENT_IP_HEADER`` set (hosted on Railway: ``X-Real-IP``), every per-IP limit reads that
header and a client's own ``X-Forwarded-For`` is ignored, so it can't rotate past a limit or aim
one at someone else. Unset, each caller reads the address as it always has: a route the first
X-Forwarded-For hop, the widget middleware the socket peer.
"""
from __future__ import annotations

import pytest
from starlette.requests import Request

from core.security.client_ip import TrustedClientAddressMiddleware, client_ip

FORGED = "198.51.100.66, 203.0.113.9"


def _scope(headers=(), peer="10.0.0.5"):
    return {"type": "http", "method": "GET", "path": "/", "query_string": b"",
            "headers": [(k.lower().encode(), v.encode()) for k, v in headers],
            "client": (peer, 4321) if peer else None}


@pytest.fixture
def railway(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "CLIENT_IP_HEADER", "X-Real-IP")


@pytest.fixture
def unset(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "CLIENT_IP_HEADER", "")


def test_the_configured_header_is_the_address_and_forwarded_for_is_ignored(railway):
    scope = _scope([("X-Forwarded-For", FORGED), ("X-Real-IP", "203.0.113.20")])
    assert client_ip(Request(scope)) == "203.0.113.20"


def test_with_the_configured_header_missing_the_address_is_the_peer_never_forwarded_for(railway):
    scope = _scope([("X-Forwarded-For", FORGED)])
    assert client_ip(Request(scope)) == "10.0.0.5"


def test_unset_a_route_reads_the_first_forwarded_hop_as_before(unset):
    scope = _scope([("X-Forwarded-For", FORGED)])
    assert client_ip(Request(scope)) == "198.51.100.66"
    assert client_ip(Request(_scope())) == "10.0.0.5"
    assert client_ip(Request(_scope(peer=None))) == "unknown"


def test_every_limit_reads_through_it(railway):
    from api.credentials import get_client_ip
    from main import _get_real_client_ip

    request = Request(_scope([("X-Forwarded-For", FORGED), ("X-Real-IP", "203.0.113.20")]))
    assert get_client_ip(request) == _get_real_client_ip(request) == "203.0.113.20"


def _through_the_middleware(scope):
    import asyncio

    seen = {}

    async def app(inner, receive, send):
        seen["client"] = inner.get("client")

    asyncio.run(TrustedClientAddressMiddleware(app)(scope, None, None))
    return seen["client"]


def test_the_middleware_gives_the_scope_the_callers_address(railway):
    scope = _scope([("X-Forwarded-For", FORGED), ("X-Real-IP", "203.0.113.20")])
    assert _through_the_middleware(scope) == ("203.0.113.20", 4321)
    assert _through_the_middleware(_scope([("X-Forwarded-For", FORGED)])) == ("10.0.0.5", 4321)


@pytest.mark.parametrize("junk", ["garbage", "1.2.3.4:80", "203.0.113.20\r\nX-Evil: 1", "999.1.1.1"])
def test_a_configured_header_that_is_not_an_ip_falls_back_to_the_peer(railway, junk):
    scope = _scope([("X-Forwarded-For", FORGED), ("X-Real-IP", junk)])
    assert client_ip(Request(scope)) == "10.0.0.5"
    assert _through_the_middleware(scope) == ("10.0.0.5", 4321)


def test_unset_the_middleware_leaves_the_scope_as_it_is(unset):
    assert _through_the_middleware(_scope([("X-Real-IP", "203.0.113.20")])) == ("10.0.0.5", 4321)


def test_the_widget_limiter_sees_the_caller_through_it(railway, monkeypatch):
    import asyncio

    from api.widgets import rate_limit

    seen = []

    class _Store:
        def check(self, key_id, limit, window=None, **kwargs):
            seen.append(key_id)
            return (True, limit, limit - 1, 60)

    async def app(scope, receive, send):
        return None

    limiter = rate_limit.WidgetRateLimitMiddleware(app)
    monkeypatch.setattr(limiter, "_store", _Store())
    scope = _scope([("X-Forwarded-For", FORGED), ("X-Real-IP", "203.0.113.20")])
    scope["path"] = "/api/widgets/blog"
    asyncio.run(TrustedClientAddressMiddleware(limiter)(scope, None, None))
    assert seen and all("203.0.113.20" in key for key in seen) and not any("198.51.100.66" in key for key in seen)


def test_main_puts_it_outside_the_widget_limiter():
    from main import app

    names = [m.cls.__name__ for m in app.user_middleware]
    assert names.index("TrustedClientAddressMiddleware") < names.index("WidgetRateLimitMiddleware")
