"""Blocked-range coverage of the shared SSRF seam (2026-09-16, from the review
of the heartbeat webhook sink — the same list guards ``web_fetch``).

Two gaps closed: the CGNAT block ``100.64.0.0/10`` (cloud private networking,
some providers' metadata endpoints) was not listed, and an IPv4-mapped IPv6
literal (``::ffff:10.0.0.1``) slipped past every IPv4 range because stdlib
containment never crosses address families.

F076 (night-3 fixer flag): the other IPv6 forms that carry a v4 address got
the same pass — IPv4-compatible ``::127.0.0.1`` (deprecated; the whole
``::/96`` is now refused), 6to4 ``2002:7f00:1::`` and NAT64 ``64:ff9b::7f00:1``
(judged by the v4 address they carry).
"""
from __future__ import annotations

import pytest

from core.security.url_validator import blocked_network_for


@pytest.mark.parametrize(
    "ip, expected",
    [
        ("100.64.0.1", "100.64.0.0/10"),
        ("100.127.255.254", "100.64.0.0/10"),
        ("100.128.0.1", None),  # first address past the CGNAT block
        ("::ffff:10.0.0.1", "10.0.0.0/8"),
        ("::ffff:169.254.169.254", "169.254.0.0/16"),
        ("::ffff:127.0.0.1", "127.0.0.0/8"),
        ("::ffff:93.184.216.34", None),  # a mapped public address stays public
        ("fd12::1", "fc00::/7"),
        ("::1", "::1/128"),
        ("::", "::/128"),
        # F076 — IPv4-compatible: refused whatever it carries
        ("::127.0.0.1", "::/96"),
        ("::10.0.0.1", "::/96"),
        ("::93.184.216.34", "::/96"),
        # F076 — 6to4 and NAT64 name the v4 address they carry
        ("2002:7f00:1::1", "127.0.0.0/8"),
        ("2002:a9fe:a9fe::", "169.254.0.0/16"),
        ("2002:5db8:d822::1", None),  # 6to4 of a public address stays public
        ("64:ff9b::127.0.0.1", "127.0.0.0/8"),
        ("64:ff9b::a9fe:a9fe", "169.254.0.0/16"),
        ("64:ff9b::93.184.216.34", None),  # NAT64 of a public address stays public
        ("2606:4700::6810:84e5", None),  # an ordinary public v6 address
        ("93.184.216.34", None),
        ("not-an-ip", "invalid"),
    ],
)
def test_blocked_network_for(ip, expected):
    assert blocked_network_for(ip) == expected


def test_a_hostname_resolving_to_an_ipv4_compatible_loopback_is_refused(monkeypatch):
    import socket

    from core.security import url_validator

    answer = [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::127.0.0.1", 443, 0, 0))]
    monkeypatch.setattr(url_validator.socket, "getaddrinfo", lambda *a, **k: answer)
    assert url_validator.validate_webhook_url("https://hook.example.com/x") == (
        False, "Resolved to blocked range (::/96)")
