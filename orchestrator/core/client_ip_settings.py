"""The two settings that say where a caller's address is, and the rules each edition holds them to.

``CLIENT_IP_HEADER`` names the header the deployment's proxy writes the caller's address to.
``CLIENT_IP_PROXY_COUNT`` says how many of the deployment's own proxies append to that header: 1
for one load balancer, 2 for a content network in front of one. The address is then the value that
many from the header's end (``core.security.client_ip``). Unset, it is the header's first value.

Pure standard library, and reads no setting: ``core.security.client_ip`` reads the pair by these
rules, and ``core.edition_start_up`` checks it at start with the values ``config.py`` hands in.
"""
from __future__ import annotations

import re
from typing import Any, Iterator, Optional, Tuple

HEADER_SETTING = "CLIENT_IP_HEADER"
COUNT_SETTING = "CLIENT_IP_PROXY_COUNT"
# The header proxies append to (an AWS application load balancer's default mode), lower-case.
FORWARDED_FOR = "x-forwarded-for"
# A content network, a firewall, a load balancer and an ingress controller make four.
PROXY_COUNT_MAX = 8
# ASCII digits only (str.isdigit would take other scripts' digits), and few enough to read.
_COUNT_TEXT = re.compile(r"[0-9]{1,3}")

COUNT_RANGE_PROBLEM = (
    f"must be a whole number from 1 to {PROXY_COUNT_MAX}: how many of your own proxies append the "
    f"caller's address to {HEADER_SETTING}"
)
COUNT_NEEDS_HEADER_PROBLEM = (
    f"needs {HEADER_SETTING}: name the header your own proxies append the caller's address to"
)
COUNT_REQUIRED_PROBLEM = (
    f"must be set: {HEADER_SETTING} names the header proxies append to, so say how many of your own "
    "proxies append to it (1 for one load balancer, 2 for a content network in front of one)"
)


class MalformedProxyCount(ValueError):
    """``CLIENT_IP_PROXY_COUNT`` is set, but not to a whole number from 1 to ``PROXY_COUNT_MAX``."""


def header_name(value: Any) -> Optional[str]:
    """The configured header's name, trimmed and lower-case; None when unset or blank."""
    name = str(value or "").strip().lower()
    return name or None


def proxy_count(value: Any) -> Optional[int]:
    """``CLIENT_IP_PROXY_COUNT`` as a count: None when unset or blank.

    Raises ``MalformedProxyCount`` on anything but a whole number from 1 to ``PROXY_COUNT_MAX``, so a
    malformed value is never read as some other count.
    """
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    if not _COUNT_TEXT.fullmatch(text):
        raise MalformedProxyCount(COUNT_RANGE_PROBLEM)
    count = int(text)
    if not 1 <= count <= PROXY_COUNT_MAX:
        raise MalformedProxyCount(COUNT_RANGE_PROBLEM)
    return count


def count_problem(header: Any, count: Any) -> Optional[str]:
    """What stops ``CLIENT_IP_PROXY_COUNT`` being used in any edition, or None: not a count in
    range, or a count with no header to count in. Never quotes a value."""
    try:
        stated = proxy_count(count)
    except MalformedProxyCount:
        return COUNT_RANGE_PROBLEM
    if stated is not None and header_name(header) is None:
        return COUNT_NEEDS_HEADER_PROBLEM
    return None


def start_up_problems(header: Any, count: Any) -> Iterator[Tuple[str, str]]:
    """Each ``(name, problem)`` where the address must be the proxy's to write (``enterprise``).

    On top of ``count_problem``: the count must be stated when the header is the one proxies append
    to, since its first value is then the caller's own. Never quotes a value.
    """
    problem = count_problem(header, count)
    if problem:
        yield COUNT_SETTING, problem
    elif header_name(header) == FORWARDED_FOR and proxy_count(count) is None:
        yield COUNT_SETTING, COUNT_REQUIRED_PROBLEM


__all__ = [
    "COUNT_NEEDS_HEADER_PROBLEM",
    "COUNT_RANGE_PROBLEM",
    "COUNT_REQUIRED_PROBLEM",
    "COUNT_SETTING",
    "FORWARDED_FOR",
    "HEADER_SETTING",
    "PROXY_COUNT_MAX",
    "MalformedProxyCount",
    "count_problem",
    "header_name",
    "proxy_count",
    "start_up_problems",
]
