"""PRD-278 US-201: the workspace worker's Redis client checks TLS the way the platform's does.

The worker (services/workspace-worker) cannot import ``core.redis.connection``; its own
``worker_redis.redis_address_and_tls`` gives ``main.WorkspaceWorker.start`` the address
and the arguments for ``redis.asyncio.from_url``:

* any address that is not ``rediss://`` goes to ``from_url`` exactly as given, with
  exactly the arguments the worker passed before;
* a ``rediss://`` address loses its query string and gets ``ssl_cert_reqs="required"``
  and ``ssl_check_hostname=True``, so an option in the address cannot change the check
  (redis-py lets a query-string option override a keyword argument, shown below);
* the host, port, database, user name and password still come from the address.
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest
import redis.asyncio as aioredis
import redis.exceptions
from redis.asyncio.connection import SSLConnection as AsyncSSLConnection

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from helpers_workspace_worker import WORKER_DIR, load_worker_main  # noqa: E402

CHECKED = {"ssl_cert_reqs": "required", "ssl_check_hostname": True}
LOOSENING_QUERY = "?ssl_cert_reqs=none&ssl_check_hostname=false&db=9"
# Joined from parts, as test_prd278_w2_redis_connection does: no credential-shaped literal.
PASSWORD = "-".join(("not", "a", "real", "value"))
TLS_ADDRESS = "".join(("rediss://", "acl-user:", PASSWORD, "%40x", "@cache.example.test:6380/2"))
PLAIN_ADDRESSES = (
    "redis://localhost:6379/0",
    "".join(("redis://", ":", PASSWORD, "@redis.example.test:6379/3?socket_timeout=5")),
    "unix:///var/run/redis.sock?db=1",
)


def _worker_redis():
    spec = importlib.util.spec_from_file_location("worker_redis_under_test", WORKER_DIR / "worker_redis.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("address", PLAIN_ADDRESSES)
def test_an_address_that_is_not_rediss_is_passed_as_given_with_nothing_added(address):
    assert _worker_redis().redis_address_and_tls(address) == (address, {})


def test_a_rediss_address_loses_its_query_string_and_gets_the_checked_tls_arguments():
    module = _worker_redis()
    assert module.redis_address_and_tls(TLS_ADDRESS + LOOSENING_QUERY) == (TLS_ADDRESS, CHECKED)
    assert module.redis_address_and_tls(TLS_ADDRESS + "#frag") == (TLS_ADDRESS, CHECKED)
    assert module.redis_address_and_tls(TLS_ADDRESS) == (TLS_ADDRESS, CHECKED)
    assert module.TLS_CERT_REQS == "required"


def test_from_url_would_honour_the_query_string_without_the_worker_s_address():
    """Non-vacuous: redis-py lets the address's options override the keyword arguments."""
    pool = aioredis.from_url(TLS_ADDRESS + LOOSENING_QUERY, **CHECKED).connection_pool
    assert pool.connection_kwargs["ssl_cert_reqs"] == "none"
    assert pool.connection_kwargs["ssl_check_hostname"] is False


def test_the_worker_s_tls_client_checks_the_certificate_and_keeps_the_address_parts():
    address, tls = _worker_redis().redis_address_and_tls(TLS_ADDRESS + LOOSENING_QUERY)
    pool = aioredis.from_url(address, decode_responses=True, **tls).connection_pool
    assert issubclass(pool.connection_class, AsyncSSLConnection)
    kwargs = pool.connection_kwargs
    assert (kwargs["ssl_cert_reqs"], kwargs["ssl_check_hostname"]) == ("required", True)
    assert (kwargs["host"], kwargs["port"], kwargs["db"]) == ("cache.example.test", 6380, 2)
    assert (kwargs["username"], kwargs["password"]) == ("acl-user", PASSWORD + "@x")


class _Built(Exception):
    """Raised by the recorded ``from_url`` so ``start`` stops before it dials Redis."""


def _start_records(monkeypatch, tmp_path, address: str):
    monkeypatch.setenv("REDIS_URL", address)
    worker = load_worker_main(monkeypatch, tmp_path).WorkspaceWorker()
    calls = []

    def recorded(*args, **kwargs):
        calls.append((args, kwargs))
        raise _Built

    monkeypatch.setattr(aioredis, "from_url", recorded)
    with pytest.raises(_Built):
        asyncio.run(worker.start())
    return calls


@pytest.mark.parametrize("address", PLAIN_ADDRESSES)
def test_start_builds_a_plain_client_with_exactly_the_arguments_it_always_had(monkeypatch, tmp_path, address):
    assert _start_records(monkeypatch, tmp_path, address) == [
        ((address,), {"decode_responses": True, "retry_on_error": [redis.exceptions.BusyLoadingError]}),
    ]


def test_start_builds_a_tls_client_with_the_check_and_without_the_query_string(monkeypatch, tmp_path):
    assert _start_records(monkeypatch, tmp_path, TLS_ADDRESS + LOOSENING_QUERY) == [
        ((TLS_ADDRESS,), {"decode_responses": True, "retry_on_error": [redis.exceptions.BusyLoadingError],
                          **CHECKED}),
    ]
