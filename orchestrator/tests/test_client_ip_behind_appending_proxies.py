"""The caller's address behind proxies that append to the header (``CLIENT_IP_PROXY_COUNT``).

With ``CLIENT_IP_HEADER`` set to the header the deployment's proxies append to and the count stated,
the address is the value that many from the header's end, read the same way by a route
(``client_ip``) and by the middleware that rewrites the scope's ``client``:

* the header missing, fewer values than the count, or the value at that position not an address:
  the socket peer, and never any other value of the header;
* the header's lines are one list, in order; empty items and spaces are skipped;
* a port, and an IPv6 address in brackets, are read as an AWS load balancer writes them; a zone is
  not an address;
* the work is bounded: a value further from the end than ``ITEMS_EXAMINED_MAX`` items is not read;
* a malformed count is never read as another count: the address is the peer, and the middleware
  logs the count's problem, never its value, when it is built.

Addresses are from the documentation ranges (RFC 5737, RFC 3849).
"""
from __future__ import annotations

import asyncio
import logging

import pytest
from starlette.requests import Request

from core.security.client_ip import ITEMS_EXAMINED_MAX, TrustedClientAddressMiddleware, client_ip

PEER = "10.0.0.5"
CALLER = "203.0.113.7"
FORGED = "198.51.100.66"
EDGE = "192.0.2.10"


def _scope(lines=(), header="X-Forwarded-For", peer=PEER):
    return {"type": "http", "method": "GET", "path": "/", "query_string": b"",
            "headers": [(header.lower().encode(), line.encode()) for line in lines],
            "client": (peer, 4321) if peer else None}


def _through_the_middleware(scope):
    seen = {}

    async def app(inner, receive, send):
        seen["client"] = inner.get("client")

    asyncio.run(TrustedClientAddressMiddleware(app)(scope, None, None))
    return seen["client"][0] if seen["client"] else None


def _both(scope):
    """What a route reads and what the middleware puts in the scope: always the same."""
    route = client_ip(Request(scope))
    assert _through_the_middleware(scope) == route
    return route


@pytest.fixture
def proxies(monkeypatch):
    """Set the header and the count as an operator would (the count as text, as config.py reads it)."""
    from config import config

    def configure(count, header="X-Forwarded-For"):
        monkeypatch.setattr(config, "CLIENT_IP_HEADER", header)
        monkeypatch.setattr(config, "CLIENT_IP_PROXY_COUNT", count)

    return configure


def test_one_load_balancer_reads_the_value_it_appended_never_the_first(proxies):
    proxies("1")
    assert _both(_scope([f"{FORGED}, {CALLER}"])) == CALLER
    assert _both(_scope([CALLER])) == CALLER


def test_a_content_network_in_front_reads_the_value_the_network_appended(proxies):
    proxies("2")
    assert _both(_scope([f"{FORGED}, {CALLER}, {EDGE}"])) == CALLER


def test_the_first_value_is_read_only_when_it_is_at_the_stated_position(proxies):
    proxies("2")
    assert _both(_scope([f"{CALLER}, {EDGE}"])) == CALLER


def test_without_the_header_the_address_is_the_peer(proxies):
    proxies("1")
    assert _both(_scope()) == PEER
    assert _both(_scope([CALLER], header="X-Real-IP")) == PEER


def test_fewer_values_than_the_count_is_the_peer(proxies):
    proxies("2")
    assert _both(_scope([CALLER])) == PEER
    proxies("3")
    assert _both(_scope([f"{FORGED}, {CALLER}"])) == PEER


def test_no_peer_and_nothing_usable_is_unknown(proxies):
    proxies("2")
    assert client_ip(Request(_scope([CALLER], peer=None))) == "unknown"


@pytest.mark.parametrize("junk", ["garbage", "999.1.1.1", "203.0.113.7 extra", "unknown", "[192.0.2.1]",
                                  "203.0.113.7:99999", "[2001:db8::1]:99999", "fe80::1%eth0", "[fe80::1%eth0]"])
def test_a_value_at_the_position_that_is_not_an_address_is_the_peer_never_another_value(proxies, junk):
    proxies("1")
    assert _both(_scope([f"{CALLER}, {junk}"])) == PEER


@pytest.mark.parametrize(("written", "address"), [
    # The examples on AWS's X-Forwarded-For page, read in Python's canonical form (RFC 5952).
    ("2001:DB8::21f:5bff:febf:ce22:8a2e", "2001:db8:0:21f:5bff:febf:ce22:8a2e"),
    ("[2001:db8::1]", "2001:db8::1"),
    ("[2001:db8:85a3:8d3:1319:8a2e:370:7348]:8080", "2001:db8:85a3:8d3:1319:8a2e:370:7348"),
    ("203.0.113.7:8080", CALLER),
    (" [2001:db8::1]:443 ", "2001:db8::1"),
])
def test_ipv6_and_ports_are_read_as_a_load_balancer_writes_them(proxies, written, address):
    proxies("1")
    assert _both(_scope([f"{FORGED}, {written}"])) == address


def test_spaces_and_empty_items_are_skipped(proxies):
    proxies("1")
    assert _both(_scope([f" {FORGED} ,, , {CALLER} , ,"])) == CALLER
    proxies("2")
    assert _both(_scope([f" {FORGED} ,, , {CALLER} , ,"])) == FORGED


def test_repeated_lines_are_one_list_in_order(proxies):
    proxies("1")
    assert _both(_scope([FORGED, CALLER])) == CALLER
    assert _both(_scope([f"{FORGED}, {CALLER}", ""])) == CALLER
    proxies("2")
    assert _both(_scope([f"{FORGED}, {CALLER}", EDGE])) == CALLER
    assert _both(_scope([FORGED, CALLER, EDGE])) == CALLER


def test_a_long_header_is_read_from_its_end_only(proxies):
    proxies("1")
    junk = ", ".join(["198.51.100.1"] * 50_000)
    assert _both(_scope([f"{junk}, {CALLER}"])) == CALLER
    assert _both(_scope(["x" * 200_000 + f", {CALLER}"])) == CALLER


def test_a_value_further_than_the_items_examined_is_not_read(proxies):
    proxies("2")
    within = CALLER + "," * (ITEMS_EXAMINED_MAX - 2) + EDGE
    beyond = CALLER + "," * ITEMS_EXAMINED_MAX + EDGE
    assert _both(_scope([within])) == CALLER
    assert _both(_scope([beyond])) == PEER
    assert _both(_scope([CALLER] + [""] * ITEMS_EXAMINED_MAX + [EDGE])) == PEER


@pytest.mark.parametrize("count", ["0", "9", "-1", "1.5", "two", "+1", "١", "1 2", "1000", 0, True])
def test_a_malformed_count_is_the_peer_never_another_count(proxies, count):
    proxies(count)
    assert _both(_scope([f"{FORGED}, {CALLER}, {EDGE}"])) == PEER


def test_with_the_count_unset_the_header_reads_as_before(proxies):
    proxies("")
    assert _both(_scope([f"{CALLER}, {EDGE}"])) == CALLER
    assert _both(_scope([f"{CALLER}:8080"])) == PEER


def test_a_count_without_a_header_changes_nothing(proxies):
    proxies("1", header="")
    assert client_ip(Request(_scope([f"{FORGED}, {CALLER}"]))) == FORGED
    assert _through_the_middleware(_scope([f"{FORGED}, {CALLER}"])) == PEER


def test_the_count_may_be_given_as_a_number(proxies):
    proxies(2)
    assert _both(_scope([f"{FORGED}, {CALLER}, {EDGE}"])) == CALLER


@pytest.mark.parametrize(("count", "header", "named"), [
    ("9", "X-Forwarded-For", "must be a whole number from 1 to 8"),
    ("lots", "X-Real-IP", "must be a whole number from 1 to 8"),
    ("1", "", "needs CLIENT_IP_HEADER"),
])
def test_the_middleware_logs_an_unusable_count_when_built_never_its_value(proxies, caplog, count, header, named):
    proxies(count, header=header)
    with caplog.at_level(logging.ERROR, logger="core.security.client_ip"):
        TrustedClientAddressMiddleware(lambda *args: None)
    [record] = [r for r in caplog.records if r.name == "core.security.client_ip"]
    assert record.getMessage().startswith(f"CLIENT_IP_PROXY_COUNT {named}")
    assert f" {count} " not in record.getMessage() and not record.getMessage().endswith(count)


@pytest.mark.parametrize(("count", "header"), [("1", "X-Forwarded-For"), ("", "X-Real-IP"), ("", "")])
def test_the_middleware_logs_nothing_for_a_usable_count(proxies, caplog, count, header):
    proxies(count, header=header)
    with caplog.at_level(logging.ERROR, logger="core.security.client_ip"):
        TrustedClientAddressMiddleware(lambda *args: None)
    assert not [r for r in caplog.records if r.name == "core.security.client_ip"]


def test_the_widget_limiter_sees_the_counted_address(proxies, monkeypatch):
    from api.widgets import rate_limit

    proxies("1")
    seen = []

    class _Store:
        def check(self, key_id, limit, window=None, **kwargs):
            seen.append(key_id)
            return (True, limit, limit - 1, 60)

    async def app(scope, receive, send):
        return None

    limiter = rate_limit.WidgetRateLimitMiddleware(app)
    monkeypatch.setattr(limiter, "_store", _Store())
    scope = {**_scope([f"{FORGED}, {CALLER}"]), "path": "/api/widgets/blog"}
    asyncio.run(TrustedClientAddressMiddleware(limiter)(scope, None, None))
    assert seen and all(CALLER in key for key in seen) and not any(FORGED in key for key in seen)
