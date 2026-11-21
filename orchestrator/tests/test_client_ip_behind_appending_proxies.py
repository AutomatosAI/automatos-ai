"""The caller's address behind proxies that append to the header (``CLIENT_IP_PROXY_COUNT``).

With ``CLIENT_IP_HEADER`` set to the header the deployment's proxies append to and the count stated,
the address is the value that many from the header's end, read the same way by a route
(``client_ip``) and by the middleware that rewrites the scope's ``client``:

* the header missing, fewer values than the count, or the value at that position not an address:
  the socket peer, and never any other value of the header. A header that is there with fewer
  values than the count logs one warning for the process, naming the settings and no address;
* the header's lines are one list, in order; empty items, spaces and tabs are skipped, and header
  names are matched in any case;
* a port, and an IPv6 address in brackets, are read as an AWS load balancer writes them; a zone is
  not an address; an IPv4-mapped IPv6 address is read as its IPv4 address;
* the work is bounded: a value further from the end than ``ITEMS_EXAMINED_MAX`` items is not read;
* outside ``enterprise`` a malformed count is ignored, the address read as if it were unset, and
  the middleware logs the count's problem once per process, never its value.

Addresses are from the documentation ranges (RFC 5737, RFC 3849).
"""
from __future__ import annotations

import asyncio
import logging

import pytest
from starlette.requests import Request

from core.security import client_ip as client_ip_module
from core.security.client_ip import ITEMS_EXAMINED_MAX, TrustedClientAddressMiddleware, client_ip

PEER = "10.0.0.5"
CALLER = "203.0.113.7"
# A value the caller sent in the header itself, ahead of what the deployment's proxies appended.
CALLER_SENT = "198.51.100.66"
EDGE = "192.0.2.10"
LOGGER = "core.security.client_ip"


@pytest.fixture(autouse=True)
def _fresh_once_per_process_logs(monkeypatch):
    """Each test starts as a new process would, with no once-per-process line written yet."""
    monkeypatch.setattr(client_ip_module, "_logged_once", set())


def _scope(lines=(), header="x-forwarded-for", peer=PEER):
    """A request scope; ``header`` is sent in the case given, as a client may send it."""
    return {"type": "http", "method": "GET", "path": "/", "query_string": b"",
            "headers": [(header.encode(), line.encode()) for line in lines],
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


def _records(caplog, level):
    return [r for r in caplog.records if r.name == LOGGER and r.levelno == level]


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
    assert _both(_scope([f"{CALLER_SENT}, {CALLER}"])) == CALLER
    assert _both(_scope([CALLER])) == CALLER


def test_a_content_network_in_front_reads_the_value_the_network_appended(proxies):
    proxies("2")
    assert _both(_scope([f"{CALLER_SENT}, {CALLER}, {EDGE}"])) == CALLER


def test_the_first_value_is_read_only_when_it_is_at_the_stated_position(proxies):
    proxies("2")
    assert _both(_scope([f"{CALLER}, {EDGE}"])) == CALLER


def test_without_the_header_the_address_is_the_peer(proxies):
    proxies("1")
    assert _both(_scope()) == PEER
    assert _both(_scope([CALLER], header="x-real-ip")) == PEER


def test_fewer_values_than_the_count_is_the_peer(proxies):
    proxies("2")
    assert _both(_scope([CALLER])) == PEER
    proxies("3")
    assert _both(_scope([f"{CALLER_SENT}, {CALLER}"])) == PEER


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
    assert _both(_scope([f"{CALLER_SENT}, {written}"])) == address


@pytest.mark.parametrize("written", ["::ffff:203.0.113.7", "::FFFF:cb00:7107", "[::ffff:203.0.113.7]",
                                     "[::ffff:203.0.113.7]:8080"])
def test_an_ipv4_mapped_address_is_read_as_its_ipv4_address_with_the_count_set(proxies, written):
    proxies("1")
    assert _both(_scope([f"{CALLER_SENT}, {written}"])) == CALLER


def test_an_ipv4_mapped_address_is_read_as_its_ipv4_address_with_the_count_unset(proxies):
    # Before: the IPv6 text (on Python 3.11, ::ffff:cb00:7107). Now the IPv4 address. A value that was
    # refused before is still refused.
    proxies("", header="X-Real-IP")
    assert _both(_scope(["::ffff:203.0.113.7"], header="x-real-ip")) == CALLER
    assert _both(_scope(["::ffff:203.0.113.7:80"], header="x-real-ip")) == PEER


def test_with_no_header_named_the_forwarded_hop_is_still_its_text(proxies):
    proxies("", header="")
    assert client_ip(Request(_scope(["::ffff:203.0.113.7"]))) == "::ffff:203.0.113.7"


def test_spaces_tabs_and_empty_items_are_skipped(proxies):
    proxies("1")
    assert _both(_scope([f" {CALLER_SENT} ,, , {CALLER} , ,"])) == CALLER
    assert _both(_scope([f"{CALLER_SENT},\t{CALLER}\t"])) == CALLER
    proxies("2")
    assert _both(_scope([f" {CALLER_SENT} ,, , {CALLER} , ,"])) == CALLER_SENT
    assert _both(_scope([f"{CALLER_SENT},\t{CALLER}"])) == CALLER_SENT


def test_a_leading_comma_is_an_empty_item(proxies):
    proxies("1")
    assert _both(_scope([f", {CALLER_SENT}, {CALLER}"])) == CALLER
    proxies("2")
    assert _both(_scope([f", {CALLER_SENT}, {CALLER}"])) == CALLER_SENT
    proxies("3")
    assert _both(_scope([f", {CALLER_SENT}, {CALLER}"])) == PEER


def test_a_quoted_item_holding_a_comma_before_the_proxies_values_moves_nothing(proxies):
    proxies("1")
    assert _both(_scope([f'"{CALLER_SENT},{EDGE}", {CALLER}'])) == CALLER
    proxies("2")
    assert _both(_scope([f'"a,b", {CALLER}, {EDGE}'])) == CALLER


def test_repeated_lines_are_one_list_in_order(proxies):
    proxies("1")
    assert _both(_scope([CALLER_SENT, CALLER])) == CALLER
    assert _both(_scope([f"{CALLER_SENT}, {CALLER}", ""])) == CALLER
    proxies("2")
    assert _both(_scope([f"{CALLER_SENT}, {CALLER}", EDGE])) == CALLER
    assert _both(_scope([CALLER_SENT, CALLER, EDGE])) == CALLER


@pytest.mark.parametrize("sent_as", ["X-Forwarded-For", "X-FORWARDED-FOR", "x-Forwarded-for"])
def test_header_names_are_matched_in_any_case_by_both_readers(proxies, sent_as):
    proxies("1", header="x-forwarded-FOR")
    assert _both(_scope([f"{CALLER_SENT}, {CALLER}"], header=sent_as)) == CALLER
    proxies("", header="X-Real-Ip")
    assert _both(_scope([CALLER], header="X-REAL-IP")) == CALLER


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


@pytest.mark.parametrize("count", ["0", "9", "-1", "1.5", "two", "+1", "01", "١", "1 2", "1000", 0, True])
def test_a_malformed_count_is_ignored_as_if_unset_never_read_as_another_count(proxies, count):
    proxies(count)
    assert _both(_scope([f"{CALLER_SENT}, {CALLER}, {EDGE}"])) == CALLER_SENT
    proxies(count, header="X-Real-IP")
    assert _both(_scope([CALLER], header="x-real-ip")) == CALLER


def test_with_the_count_unset_the_header_reads_as_before(proxies):
    proxies("")
    assert _both(_scope([f"{CALLER}, {EDGE}"])) == CALLER
    assert _both(_scope([f"{CALLER}:8080"])) == PEER


def test_a_count_without_a_header_changes_nothing(proxies):
    proxies("1", header="")
    assert client_ip(Request(_scope([f"{CALLER_SENT}, {CALLER}"]))) == CALLER_SENT
    assert _through_the_middleware(_scope([f"{CALLER_SENT}, {CALLER}"])) == PEER


def test_the_count_may_be_given_as_a_number(proxies):
    proxies(2)
    assert _both(_scope([f"{CALLER_SENT}, {CALLER}, {EDGE}"])) == CALLER


# ── the once-per-process log lines ──────────────────────────────────────────


def _assert_names_no_address(record):
    message = record.getMessage()
    assert all(address not in message for address in (PEER, CALLER, CALLER_SENT, EDGE, "203.0.113", "10.0.0"))


@pytest.mark.parametrize("route_first", [True, False])
def test_fewer_values_than_the_count_is_warned_once_through_both_readers(proxies, caplog, route_first):
    proxies("2", header="X-Real-IP")
    scope = _scope([CALLER], header="x-real-ip")
    readers = [lambda: client_ip(Request(scope)), lambda: _through_the_middleware(scope)]
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        for read in (readers if route_first else readers[::-1]) * 2:
            assert read() == PEER
    [record] = _records(caplog, logging.WARNING)
    assert record.getMessage().startswith("CLIENT_IP_HEADER carried fewer values than CLIENT_IP_PROXY_COUNT")
    assert "read from the connection" in record.getMessage()
    _assert_names_no_address(record)


def test_no_warning_without_the_header_or_with_enough_values(proxies, caplog):
    proxies("2")
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert _both(_scope()) == PEER
        assert _both(_scope([f"{CALLER}, {EDGE}"])) == CALLER
    assert not _records(caplog, logging.WARNING)


@pytest.mark.parametrize(("count", "header", "named"), [
    ("9", "X-Forwarded-For", "must be a plain whole number from 1 to 8"),
    ("lots", "X-Real-IP", "must be a plain whole number from 1 to 8"),
    ("01", "X-Real-IP", "must be a plain whole number from 1 to 8"),
    ("1", "", "needs CLIENT_IP_HEADER"),
])
def test_an_unusable_count_is_logged_once_per_process_never_its_value(proxies, caplog, count, header, named):
    proxies(count, header=header)
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        TrustedClientAddressMiddleware(lambda *args: None)
        TrustedClientAddressMiddleware(lambda *args: None)
    [record] = _records(caplog, logging.ERROR)
    message = record.getMessage()
    assert message.startswith(f"CLIENT_IP_PROXY_COUNT {named}")
    assert "ignored" in message and "as if it were unset" in message
    assert f" {count} " not in message and not message.endswith(count)


@pytest.mark.parametrize(("count", "header"), [("1", "X-Forwarded-For"), ("", "X-Real-IP"), ("", "")])
def test_the_middleware_logs_nothing_for_a_usable_count(proxies, caplog, count, header):
    proxies(count, header=header)
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        TrustedClientAddressMiddleware(lambda *args: None)
    assert not _records(caplog, logging.ERROR)


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
    scope = {**_scope([f"{CALLER_SENT}, {CALLER}"]), "path": "/api/widgets/blog"}
    asyncio.run(TrustedClientAddressMiddleware(limiter)(scope, None, None))
    assert seen and all(CALLER in key for key in seen) and not any(CALLER_SENT in key for key in seen)
