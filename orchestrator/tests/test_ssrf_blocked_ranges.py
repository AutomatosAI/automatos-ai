"""Blocked-range coverage of the shared SSRF seam (2026-09-16, from the review
of the heartbeat webhook sink — the same list guards ``web_fetch``).

Two gaps closed: the CGNAT block ``100.64.0.0/10`` (cloud private networking,
some providers' metadata endpoints) was not listed, and an IPv4-mapped IPv6
literal (``::ffff:10.0.0.1``) slipped past every IPv4 range because stdlib
containment never crosses address families.
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
        ("93.184.216.34", None),
        ("not-an-ip", "invalid"),
    ],
)
def test_blocked_network_for(ip, expected):
    assert blocked_network_for(ip) == expected
