"""TLS on every connection to the application's own database (PRD-278 §5.3, Wave 3).

Two settings, read through ``config``:

* ``DATABASE_SSLMODE``: libpq's ``sslmode`` (disable, allow, prefer, require, verify-ca,
  verify-full). ``verify-ca`` checks the server's certificate against a CA; ``verify-full``
  also checks that the certificate names the host connected to.
* ``DATABASE_SSLROOTCERT``: the CA file that check uses. Only with ``verify-ca`` or
  ``verify-full``.

``database_tls()`` turns them into what each driver takes:

* **libpq**, which ``psycopg2.connect`` and SQLAlchemy's psycopg2 dialect both use: the
  keywords ``sslmode`` and ``sslrootcert`` (``libpq_params``; SQLAlchemy gets them in
  ``connect_args``, which win over the same keys in the address). libpq does the check.
* **asyncpg** (``asyncpg_kwargs``): the ``ssl`` argument, which wins over an ``sslmode`` in
  the address. asyncpg takes no ``sslrootcert`` keyword, so with a CA file it is given an
  ``ssl.SSLContext`` that trusts only that file, requires the certificate, and checks the
  host name for ``verify-full``. Without a CA file it is given the mode as text, which
  asyncpg applies the way libpq does, from ``~/.postgresql/root.crt``.
* **libpq's tools** (``pg_isready``, ``psql``): ``PGSSLMODE`` and ``PGSSLROOTCERT``, which
  ``docker-entrypoint.sh`` sets from the same two settings on those commands only.

With neither setting, nothing is added to any connection, so each connects as it did
before the settings existed. The main engine keeps its own default: in production,
``sslmode=require`` when its address names no ``sslmode`` (``main_engine_url``).

No message here quotes an address, a password or the CA file's path.
"""
from __future__ import annotations

import ssl
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Mapping, Optional, Union
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from config import config

SSLMODES = ("disable", "allow", "prefer", "require", "verify-ca", "verify-full")
CERTIFICATE_CHECKING_MODES = frozenset({"verify-ca", "verify-full"})
SERVER_VERIFYING_MODE = "verify-full"
# The main engine's own default, unchanged: in production, encrypted when the address names
# no sslmode. It applies only while DATABASE_SSLMODE is unset.
PRODUCTION_ENVIRONMENT = "production"
PRODUCTION_DEFAULT_SSLMODE = "require"
DEFAULT_ENVIRONMENT = "development"
# One SSLContext per CA file and host-name rule, reused by every asyncpg connection.
_CONTEXT_CACHE_SIZE = 4

AsyncpgSSL = Union[str, ssl.SSLContext]


class DatabaseTLSConfigError(ValueError):
    """``DATABASE_SSLMODE`` and ``DATABASE_SSLROOTCERT`` can't be applied as given."""


@dataclass(frozen=True)
class DatabaseTLS:
    """How a connection to the application's database is secured. Both unset: as before."""

    sslmode: Optional[str] = None
    sslrootcert: Optional[str] = None

    @property
    def configured(self) -> bool:
        """True when ``DATABASE_SSLMODE`` is set, so every connection path applies it."""
        return self.sslmode is not None

    @property
    def checks_certificate(self) -> bool:
        """True when the server's certificate is checked against a CA (verify-ca, verify-full)."""
        return self.sslmode in CERTIFICATE_CHECKING_MODES

    @property
    def verifies_server(self) -> bool:
        """True when both the server's certificate and its host name are checked (verify-full).

        The predicate an install that must refuse an unchecked connection can require
        (PRD-278 D-6). Nothing requires it yet.
        """
        return self.sslmode == SERVER_VERIFYING_MODE

    def libpq_params(self) -> Dict[str, str]:
        """The keywords for ``psycopg2.connect`` and for SQLAlchemy's ``connect_args``."""
        params: Dict[str, str] = {}
        if self.sslmode:
            params["sslmode"] = self.sslmode
        if self.sslrootcert:
            params["sslrootcert"] = self.sslrootcert
        return params

    def with_libpq_params(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        """A copy of a psycopg2 keyword dict with these settings over it."""
        return {**params, **self.libpq_params()}

    def asyncpg_kwargs(self) -> Dict[str, AsyncpgSSL]:
        """The keywords for ``asyncpg.connect``: none, or ``ssl`` (see the module docstring)."""
        if self.sslmode is None:
            return {}
        if self.sslrootcert:
            return {"ssl": _verifying_context(self.sslrootcert, self.verifies_server)}
        return {"ssl": self.sslmode}


@lru_cache(maxsize=_CONTEXT_CACHE_SIZE)
def _verifying_context(cafile: str, check_hostname: bool) -> ssl.SSLContext:
    """A client context that trusts only ``cafile`` and requires the server's certificate.

    ``create_default_context`` with a ``cafile`` loads that file and not the system's CAs,
    as libpq's ``sslrootcert`` does, and starts with ``CERT_REQUIRED`` and host-name checks
    on. ``verify-ca`` turns the host-name check off; the certificate stays required.
    """
    context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=cafile)
    context.check_hostname = check_hostname
    return context


def parse_database_tls(sslmode: Optional[str], sslrootcert: Optional[str]) -> DatabaseTLS:
    """The two settings, checked. Blank counts as unset. Raises ``DatabaseTLSConfigError``."""
    mode = (sslmode or "").strip().lower() or None
    rootcert = (sslrootcert or "").strip() or None
    if mode is not None and mode not in SSLMODES:
        raise DatabaseTLSConfigError(
            f"DATABASE_SSLMODE must be one of {', '.join(SSLMODES)}; got {mode!r}"
        )
    if rootcert is not None and mode not in CERTIFICATE_CHECKING_MODES:
        raise DatabaseTLSConfigError(
            "DATABASE_SSLROOTCERT is used only with DATABASE_SSLMODE verify-ca or verify-full: "
            "set DATABASE_SSLMODE=verify-full to check the server's certificate and host name"
        )
    return DatabaseTLS(sslmode=mode, sslrootcert=rootcert)


def database_tls() -> DatabaseTLS:
    """``DATABASE_SSLMODE`` and ``DATABASE_SSLROOTCERT`` from config, checked."""
    return parse_database_tls(config.DATABASE_SSLMODE, config.DATABASE_SSLROOTCERT)


def main_engine_url(url: str, environment: Optional[str], tls: DatabaseTLS) -> str:
    """The main engine's address.

    With ``DATABASE_SSLMODE`` unset, production adds ``sslmode=require`` when the address
    names no ``sslmode``, exactly as the engine always has. With it set, the address is
    used as given and the setting reaches the driver through ``connect_args``.
    """
    if tls.configured or (environment or DEFAULT_ENVIRONMENT).lower() != PRODUCTION_ENVIRONMENT:
        return url
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    if "sslmode" in params:
        return url
    params["sslmode"] = [PRODUCTION_DEFAULT_SSLMODE]
    return urlunparse(parsed._replace(query=urlencode(params, doseq=True)))


def create_app_engine(url: str, *, connect_args: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> Engine:
    """A SQLAlchemy engine on the application's own database, secured by ``database_tls()``.

    The settings' libpq keywords go in ``connect_args``, over any the caller passes.
    """
    return create_engine(url, connect_args=database_tls().with_libpq_params(connect_args or {}), **kwargs)
