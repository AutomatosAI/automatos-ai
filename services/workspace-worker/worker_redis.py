"""The worker's Redis address, with TLS checked as the platform checks it (PRD-278 US-201).

The worker is its own image and cannot import the orchestrator's
``core.redis.connection``; this is the worker's equivalent of that builder's TLS rule,
for the one client the worker opens (``main.WorkspaceWorker.start``):

* ``redis://`` (and any address that is not ``rediss://``) goes to
  ``redis.asyncio.from_url`` exactly as given, with no argument added.
* ``rediss://`` is TLS with the server's certificate checked: it must chain to a CA in
  the image's trust store and name the host that was dialled. ``ssl_cert_reqs`` and
  ``ssl_check_hostname`` are passed explicitly, and the address's query string is
  dropped before redis-py reads it (redis-py lets a query-string option override a
  keyword argument), so nothing in the address can change the check. The host, port,
  database (the path), user name and password are read from the address as before.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple
from urllib.parse import urlsplit, urlunsplit

TLS_SCHEME = "rediss"
#: The server's certificate is required and verified. There is deliberately no
#: setting for anything weaker.
TLS_CERT_REQS = "required"


def redis_address_and_tls(url: str) -> Tuple[str, Dict[str, Any]]:
    """The address to hand ``redis.asyncio.from_url`` and the TLS arguments to add.

    An address that cannot be split is returned unchanged, so ``from_url`` refuses it
    exactly as it did before.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return url, {}
    if parts.scheme != TLS_SCHEME:
        return url, {}
    address = urlunsplit(parts._replace(query="", fragment=""))
    return address, {"ssl_cert_reqs": TLS_CERT_REQS, "ssl_check_hostname": True}
