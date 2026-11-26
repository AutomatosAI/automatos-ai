"""Where a credential test may connect: the outbound guard, applied to ``CredentialTester``.

A credential's host, port and URLs are whatever a workspace user typed, and the test
runs from the server's own network position. Where several people share the server
(``web_access.destinations_guarded``), every address the name resolves to is checked
against the private, loopback, link-local and metadata ranges and ``WEB_ACCESS_DENY``,
and the test connects to the address that was checked, with the name kept for TLS (SNI
and the certificate check) and, for SSH, the host-key lookup. An HTTP test follows no
redirect, since a hop could land in a blocked range. On a single-operator install the
destination is used as given: testing your own local database is the point there.
"""

from __future__ import annotations

import socket
import ssl
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Union

from core.security.web_access import (
    OutboundTarget,
    operator_destination_async,
    pinned_request_parts,
)

#: The Postgres ``ssl`` modes that build a TLS context here; the others leave it to asyncpg.
POSTGRES_TLS_MODES = ("require", "verify-ca", "verify-full")
POSTGRES_TLS_OFF = "disable"


@dataclass(frozen=True)
class HttpDestination:
    """An aiohttp request's URL and options, or the guard's ``refusal``."""

    refusal: str = ""
    url: str = ""
    options: Mapping[str, Any] = field(default_factory=dict)


async def http_destination(url: str, headers: Optional[Mapping[str, str]] = None) -> HttpDestination:
    """The request for ``url`` with ``headers``: refused, pinned (the checked address,
    ``Host`` and ``server_hostname`` = the name, no redirects), or, unguarded, as given."""
    target = await operator_destination_async(url)
    if not target.ok:
        return HttpDestination(refusal=target.reason)
    parts = pinned_request_parts(url, target, headers)
    if not target.pinned:
        return HttpDestination(url=url, options={"headers": parts["headers"]})
    options = {"headers": parts["headers"], "server_hostname": target.host, "allow_redirects": False}
    return HttpDestination(url=parts["url"], options=options)


def connect_address(target: OutboundTarget) -> str:
    """The address to open: the checked IP when pinned, else the host as given."""
    return target.ip if target.pinned else target.host


class _NamedTLSContext(ssl.SSLContext):
    """A TLS context that sends ``server_name`` as SNI and checks the certificate
    against it, whatever address the socket was opened to (asyncpg passes the
    address it connected to as the server name)."""

    server_name: str = ""

    def wrap_bio(self, incoming, outgoing, server_side=False, server_hostname=None, session=None):
        return super().wrap_bio(incoming, outgoing, server_side, self.server_name or server_hostname, session)


def postgres_tls(ssl_mode: str, allow_unauthorized: bool, server_name: str) -> Union[ssl.SSLContext, bool, None]:
    """asyncpg's ``ssl`` argument for ``ssl_mode``: False to disable, None to let
    asyncpg decide (``allow``, ``prefer``), else a context named for ``server_name``
    that checks the certificate unless ``require`` or ``allow_unauthorized``."""
    if ssl_mode == POSTGRES_TLS_OFF:
        return False
    if ssl_mode not in POSTGRES_TLS_MODES:
        return None
    context = _NamedTLSContext(ssl.PROTOCOL_TLS_CLIENT)
    context.load_default_certs(ssl.Purpose.SERVER_AUTH)
    context.server_name = server_name
    if allow_unauthorized or ssl_mode == "require":
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    return context


def ssh_socket(target: OutboundTarget) -> Optional[socket.socket]:
    """For paramiko's ``sock=``: a connection to the checked address when pinned, so
    the host key is still looked up under the name; None to let paramiko connect."""
    if not target.pinned:
        return None
    return socket.create_connection((target.ip, target.port))


__all__ = [
    "HttpDestination",
    "connect_address",
    "http_destination",
    "postgres_tls",
    "ssh_socket",
]
