"""One way to build a connection to the platform's Redis (PRD-278 US-201).

Every Redis client the orchestrator opens to its own Redis takes its connection
arguments from :func:`connection_kwargs` (code outside ``core/redis/`` through
:func:`new_sync_client` and :func:`new_async_client`), so whether a connection
uses TLS is decided here and nowhere else:

* ``redis://`` is a plain connection, as before.
* ``rediss://`` is TLS with the server's certificate checked: it must chain to a
  CA in the system trust store and name the host that was dialled. Nothing here
  turns the check off, and the address's query string is not read, so an address
  cannot turn it off either.
* A user name in the address (a Redis ACL user) is kept.

Where the address comes from differs by client, each as it was before this
module (``REDIS_TLS`` applies wherever the host settings are used):

* :func:`redis_target_from_config` (the main client): ``config.REDIS_URL``, the
  ``REDIS_URL`` setting or one built from ``REDIS_HOST`` and ``REDIS_PORT``
  (both needed) and ``REDIS_PASSWORD``, database 0; ``None`` when neither.
* :func:`redis_target_from_url_or_host_settings` (the task queue, the shared
  context): the ``REDIS_URL`` setting when it is set, otherwise the host settings
  with 127.0.0.1 and 6379 for a missing host or port, and the caller's database.

A parse error never quotes the address: part of it may be a password.

Connections that test a user's own saved Redis credential (``core/credentials/``)
go to the user's Redis, not the platform's, and do not come through here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional
from urllib.parse import ParseResult, unquote, urlparse

import redis
import redis.asyncio as aioredis

PLAIN_SCHEME = "redis"
TLS_SCHEME = "rediss"
DEFAULT_PORT = 6379
DEFAULT_DB = 0
#: The server's certificate is required and verified. There is deliberately no
#: setting for anything weaker.
TLS_CERT_REQS = "required"


class ConnectionKind(str, Enum):
    """What the arguments are for: the TLS arguments differ for a bare pool."""

    SYNC = "sync"    # redis.Redis(...)
    ASYNC = "async"  # redis.asyncio.Redis(...)
    POOL = "pool"    # redis.ConnectionPool(...)


@dataclass(frozen=True)
class RedisTarget:
    """Where the platform's Redis is and how to reach it.

    The password is kept out of ``repr`` so a target can be logged.
    """

    host: str
    port: int = DEFAULT_PORT
    db: int = DEFAULT_DB
    username: Optional[str] = None
    password: Optional[str] = field(default=None, repr=False)
    tls: bool = False

    def describe(self) -> str:
        """``host:port db=N tls=on|off``, with no credential in it, for logs."""
        return f"{self.host}:{self.port} db={self.db} tls={'on' if self.tls else 'off'}"


#: The host the task queue and the shared context dial when ``REDIS_HOST`` is unset.
LOCAL_HOST = "127.0.0.1"

_UNPARSEABLE = "the Redis address cannot be parsed"
_BAD_SCHEME = "a Redis address must start with redis:// or rediss://"
_NO_HOST = "the Redis address names no host"
_BAD_PORT = "the Redis address's port is not a valid port"
_BAD_DB = "the Redis address's database is not a number"


def parse_redis_url(url: str) -> RedisTarget:
    """Turn a ``redis://`` or ``rediss://`` address into a :class:`RedisTarget`.

    The user name and password are percent-decoded, as redis-py's own
    ``from_url`` does. The path is the database number (0 when absent).

    Raises:
        ValueError: another scheme, no host, a bad port or a bad database number.
            The message is one of a fixed few and never quotes the address.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        raise ValueError(_UNPARSEABLE) from None
    if parsed.scheme not in (PLAIN_SCHEME, TLS_SCHEME):
        raise ValueError(_BAD_SCHEME)
    if not parsed.hostname:
        raise ValueError(_NO_HOST)
    return RedisTarget(
        host=parsed.hostname,
        port=_port(parsed),
        db=_db(parsed.path.lstrip("/")),
        username=unquote(parsed.username) if parsed.username else None,
        password=unquote(parsed.password) if parsed.password else None,
        tls=parsed.scheme == TLS_SCHEME,
    )


def _port(parsed: ParseResult) -> int:
    """The port, or 6379; the library's own error would quote part of the address."""
    try:
        return parsed.port or DEFAULT_PORT
    except ValueError:
        raise ValueError(_BAD_PORT) from None


def _db(path: str) -> int:
    """The database number from the path, or 0; never quoting the path."""
    if not path:
        return DEFAULT_DB
    try:
        return int(path)
    except ValueError:
        raise ValueError(_BAD_DB) from None


def redis_target_from_config() -> Optional[RedisTarget]:
    """The main client's Redis: ``config.REDIS_URL``, or ``None`` when it is unset.

    ``config.REDIS_URL`` is the ``REDIS_URL`` setting, or one built from
    ``REDIS_HOST`` and ``REDIS_PORT`` when both are set (database 0).

    Raises:
        ValueError: the address is not usable (message never quotes it).
    """
    # Imported here, as the Redis client always has: a module-level import would
    # pin whichever ``config`` happened to be loaded first (tests swap it).
    from config import config

    url = config.REDIS_URL
    return parse_redis_url(url) if url else None


def redis_target_from_host_settings(db: int) -> RedisTarget:
    """``REDIS_HOST`` (127.0.0.1 when unset), ``REDIS_PORT`` (6379 when unset),
    ``REDIS_PASSWORD``, ``REDIS_TLS`` and the caller's database number, read the
    way the task queue and the shared context always read them."""
    from config import config

    return RedisTarget(
        host=config.REDIS_HOST or LOCAL_HOST,
        port=int(config.REDIS_PORT or DEFAULT_PORT),
        db=db,
        password=config.REDIS_PASSWORD or None,
        tls=config.REDIS_TLS,
    )


def redis_target_from_url_or_host_settings(db: int) -> RedisTarget:
    """The ``REDIS_URL`` setting when it is set, otherwise the host settings with
    database ``db`` (see :func:`redis_target_from_host_settings`).

    Raises:
        ValueError: ``REDIS_URL`` is set but not usable (message never quotes it).
    """
    from config import config

    url = config.redis_url_as_set()
    return parse_redis_url(url) if url else redis_target_from_host_settings(db)


def connection_kwargs(target: RedisTarget, kind: ConnectionKind) -> Dict[str, Any]:
    """The connection arguments for one kind of client, TLS included.

    Callers add their own timeouts, pool size and ``decode_responses``; only the
    address, the credentials and TLS come from here. A user name is passed only
    when there is one, so a plain address gives exactly the arguments it did
    before this module.
    """
    kwargs: Dict[str, Any] = {
        "host": target.host,
        "port": target.port,
        "db": target.db,
        "password": target.password,
    }
    if target.username:
        kwargs["username"] = target.username
    if target.tls:
        kwargs.update(_tls_kwargs(kind))
    return kwargs


def new_sync_client(target: RedisTarget, **options: Any) -> redis.Redis:
    """A ``redis.Redis`` for ``target``; ``options`` are the caller's own
    (timeouts, retries, ``decode_responses``). Code outside ``core/redis/`` builds
    its clients through this or :func:`new_async_client`, never directly."""
    return redis.Redis(**connection_kwargs(target, ConnectionKind.SYNC), **options)


def new_async_client(target: RedisTarget, **options: Any) -> aioredis.Redis:
    """A ``redis.asyncio.Redis`` for ``target``, with the caller's own ``options``."""
    return aioredis.Redis(**connection_kwargs(target, ConnectionKind.ASYNC), **options)


def _tls_kwargs(kind: ConnectionKind) -> Dict[str, Any]:
    """TLS with the certificate and the host name checked.

    A client (sync or async) takes ``ssl=True``; a bare pool takes the TLS
    connection class instead. Both are given the check explicitly rather than
    relying on the library's defaults.
    """
    checked: Dict[str, Any] = {"ssl_cert_reqs": TLS_CERT_REQS, "ssl_check_hostname": True}
    if kind is ConnectionKind.POOL:
        return {"connection_class": redis.SSLConnection, **checked}
    return {"ssl": True, **checked}
