"""PRD-277 US-401, FR-9, FR-10: the one way sign-in calls the organisation's identity provider.

Sign-in makes three calls, and none goes to an address a request supplies:

* the metadata document, at the address built from ``OIDC_ISSUER`` (``core.auth.oidc_provider``);
* the signing keys (``jwks_uri``) and the token endpoint (``token_endpoint``), both named by that
  metadata, which itself came from the issuer over verified TLS.

Every call:

* is ``https``, with a host and no whitespace or control character; anything else is refused
  before any connection, and an address the HTTP client cannot parse is a refusal too;
* resolves the host and connects to exactly the address it checked, the hostname kept as ``Host``
  and as the TLS name (``core.security.web_access.build_pinned_request``), so a DNS answer that
  changes between the check and the connection changes nothing;
* may reach a private, loopback, link-local or shared address (``url_validator.BLOCKED_NETWORKS``)
  only when its host and port are the configured issuer's own: an identity provider inside the
  customer's network (FR-9). Every other host, and every other port of that host, must resolve to
  public addresses only, so the metadata can never turn sign-in into a way to reach anything else
  inside the network;
* never reaches the reserved ranges in ``SIGN_IN_RESERVED_NETWORKS`` (IETF protocol assignments,
  benchmarking, multicast and the old site-local block), on any host, also when an IPv6 address
  carries one (mapped, 6to4 or NAT64);
* verifies TLS, hostname included: against httpx's default authorities, or, when
  ``OIDC_CA_BUNDLE_FILE`` is set, against the operating system's authorities and the ones in that
  file (FR-10: an identity provider whose certificate a private authority signed);
* follows no redirect (a 3xx is a failure), ignores proxy settings in the environment (a proxy
  would connect to an address this module never checked), waits at most ``CONNECT_TIMEOUT_SECONDS``
  to connect and ``READ_TIMEOUT_SECONDS`` for each read, and reads at most ``MAX_RESPONSE_BYTES``.

A failure is ``ProviderUnreachable``, whose message names no address, code, token or secret. It is
raised outside the handler that caught the HTTP client's own exception (whose text can carry a URL),
so that exception is neither its cause nor its context; the log names the call and the failure's
type, never a value.
"""
from __future__ import annotations

import ipaddress
import json
import logging
import socket
import ssl
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Tuple
from urllib.parse import urlsplit

import httpx

from config import config
from core.security.url_validator import blocked_network_for
from core.security.web_access import OutboundTarget, build_pinned_request

logger = logging.getLogger(__name__)

HTTPS = "https"
HTTPS_PORT = 443
CONNECT_TIMEOUT_SECONDS = 5.0
READ_TIMEOUT_SECONDS = 10.0
MAX_RESPONSE_BYTES = 1024 * 1024
HTTP_OK = 200
JSON_ACCEPT = {"Accept": "application/json"}
UNREACHABLE_MESSAGE = "The organisation's identity provider could not be reached."

# Ranges no identity provider is on, refused on top of url_validator's, on every host.
SIGN_IN_RESERVED_NETWORKS = tuple(ipaddress.ip_network(net) for net in (
    "192.0.0.0/24", "198.18.0.0/15", "224.0.0.0/4", "ff00::/8", "fec0::/10",
))
# NAT64's well-known prefix (RFC 6052): ``64:ff9b::a.b.c.d`` is a.b.c.d.
NAT64_PREFIX = ipaddress.ip_network("64:ff9b::/96")

# The resolver seam: tests replace it; production is the system resolver.
_getaddrinfo = socket.getaddrinfo


class ProviderUnreachable(RuntimeError):
    """A call to the identity provider failed or was refused here. Its text never holds a value."""


@dataclass(frozen=True)
class ProviderAnswer:
    """What the provider answered: its status and its body as JSON (``None`` when not JSON)."""

    status: int
    body: Any


def host_of(url: Optional[str]) -> str:
    """``url``'s host, lower-case, with no trailing dot; empty when it has none."""
    try:
        return (urlsplit(url or "").hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def _port_of(url: Optional[str]) -> Optional[int]:
    try:
        return urlsplit(url or "").port or HTTPS_PORT
    except ValueError:
        return None


def issuer_host() -> str:
    """The configured issuer's host, the one host allowed to resolve to a private address."""
    return host_of(config.OIDC_ISSUER)


def is_https_address(url: Any) -> bool:
    """Whether ``url`` is an ``https`` address with a host, a valid port and no space or control character.

    The character check comes first: ``urlsplit`` quietly drops a tab, CR or LF, so the address it
    reads would not be the one the HTTP client is handed.
    """
    if not isinstance(url, str) or not url:
        return False
    if any(ch.isspace() or not ch.isprintable() for ch in url):
        return False
    try:
        parts = urlsplit(url)
        _ = parts.port  # a malformed port raises here
    except ValueError:
        return False
    return parts.scheme == HTTPS and bool(parts.hostname) and parts.username is None


def _refused(purpose: str, why: str) -> ProviderUnreachable:
    logger.warning("Identity provider %s call refused: %s", purpose, why)
    return ProviderUnreachable(UNREACHABLE_MESSAGE)


def _addresses(host: str, port: int) -> List[str]:
    """Every address ``host`` resolves to (empty when it does not resolve)."""
    try:
        infos = _getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError, OSError, ValueError):
        return []
    found: List[str] = []
    for _family, _type, _proto, _canon, sockaddr in infos:
        if str(sockaddr[0]) not in found:
            found.append(str(sockaddr[0]))
    return found


def _carried_v4(addr: ipaddress.IPv6Address) -> Optional[ipaddress.IPv4Address]:
    """The v4 address an IPv6 address carries: IPv4-mapped, 6to4 or NAT64."""
    if addr.ipv4_mapped is not None:
        return addr.ipv4_mapped
    if addr.sixtofour is not None:
        return addr.sixtofour
    return ipaddress.IPv4Address(int(addr) & 0xFFFFFFFF) if addr in NAT64_PREFIX else None


def _reserved_for_sign_in(ip: str) -> bool:
    """Whether ``ip``, or the v4 address it carries, is in ``SIGN_IN_RESERVED_NETWORKS`` (junk counts)."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return True
    candidates = [addr] + ([_carried_v4(addr)] if addr.version == 6 else [])
    return any(candidate is not None and candidate.version == network.version and candidate in network
               for candidate in candidates for network in SIGN_IN_RESERVED_NETWORKS)


def pin(url: str, purpose: str) -> OutboundTarget:
    """The checked address to connect to for ``url``, or ``ProviderUnreachable``.

    ``https`` only. The issuer's own host may resolve to a private address; any other host must
    resolve to public addresses only, every answer checked. ``SIGN_IN_RESERVED_NETWORKS`` never.
    """
    if not is_https_address(url):
        raise _refused(purpose, "not an https address")
    parts = urlsplit(url)
    host, port = host_of(url), parts.port or HTTPS_PORT
    addresses = _addresses(host, port)
    if not addresses:
        raise _refused(purpose, "the host does not resolve")
    private_allowed = (host, port) == (issuer_host(), _port_of(config.OIDC_ISSUER))
    for ip in addresses:
        blocked = blocked_network_for(ip)
        if blocked == "invalid" or (blocked and not private_allowed):
            raise _refused(purpose, "the host resolves to an address sign-in may not reach")
    if any(_reserved_for_sign_in(ip) for ip in addresses):
        raise _refused(purpose, "the host resolves to a reserved address")
    pinned = next((ip for ip in addresses if ":" not in ip), addresses[0])
    return OutboundTarget(True, "OK", host, pinned, port, HTTPS)


def _tls_verify() -> Any:
    """httpx's default authorities; with ``OIDC_CA_BUNDLE_FILE``, the system's plus that file's."""
    bundle = (config.OIDC_CA_BUNDLE_FILE or "").strip()
    if not bundle:
        return True
    context = ssl.create_default_context()
    context.load_verify_locations(cafile=bundle)
    return context


def _client() -> httpx.Client:
    timeout = httpx.Timeout(READ_TIMEOUT_SECONDS, connect=CONNECT_TIMEOUT_SECONDS)
    return httpx.Client(verify=_tls_verify(), follow_redirects=False, timeout=timeout, trust_env=False)


def _read_capped(response: httpx.Response) -> Optional[bytes]:
    """The body, or ``None`` when it is longer than ``MAX_RESPONSE_BYTES``."""
    chunks, size = [], 0
    for chunk in response.iter_bytes():
        size += len(chunk)
        if size > MAX_RESPONSE_BYTES:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def _as_json(body: bytes) -> Any:
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None


def _send(method: str, url: str, purpose: str, **kwargs: Any) -> Tuple[int, Optional[bytes]]:
    """``(status, body)`` of one pinned call; ``(status, None)`` when the body is too long."""
    target = pin(url, purpose)
    name = f"[{target.host}]" if ":" in target.host else target.host
    host = name if target.port == HTTPS_PORT else f"{name}:{target.port}"
    headers = {**dict(kwargs.pop("headers", None) or {}), "Host": host}
    with _client() as client:
        request = build_pinned_request(client, method, url, target, headers=headers, **kwargs)
        response = client.send(request, stream=True)
        try:
            return response.status_code, _read_capped(response)
        finally:
            response.close()


def call(method: str, url: str, purpose: str, *, data: Optional[Mapping[str, str]] = None,
         headers: Optional[Mapping[str, str]] = None) -> ProviderAnswer:
    """One call to the provider; ``ProviderUnreachable`` when it could not be made or read.

    A status other than 200 is still an answer: the caller decides what it means.
    """
    failure: Optional[str] = None
    try:
        status, body = _send(method, url, purpose, data=dict(data) if data else None,
                             headers={**JSON_ACCEPT, **dict(headers or {})})
    except (httpx.HTTPError, httpx.InvalidURL, ssl.SSLError, OSError, ValueError) as error:
        failure = type(error).__name__
    if failure is not None:
        raise _refused(purpose, failure)
    if body is None:
        raise _refused(purpose, "the answer is too long")
    return ProviderAnswer(status=status, body=_as_json(body))


def get_json(url: str, purpose: str) -> Any:
    """The JSON a ``GET`` answers with 200; ``ProviderUnreachable`` for anything else."""
    answer = call("GET", url, purpose)
    if answer.status != HTTP_OK:
        raise _refused(purpose, f"it answered {answer.status}")
    if answer.body is None:
        raise _refused(purpose, "the answer is not JSON")
    return answer.body


__all__ = [
    "CONNECT_TIMEOUT_SECONDS",
    "HTTPS",
    "MAX_RESPONSE_BYTES",
    "READ_TIMEOUT_SECONDS",
    "SIGN_IN_RESERVED_NETWORKS",
    "UNREACHABLE_MESSAGE",
    "ProviderAnswer",
    "ProviderUnreachable",
    "call",
    "get_json",
    "host_of",
    "is_https_address",
    "issuer_host",
    "pin",
]
