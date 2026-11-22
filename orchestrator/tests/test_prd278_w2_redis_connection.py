"""PRD-278 US-201: one way to build a connection to the platform's Redis.

``core.redis.connection`` turns the settings into connection arguments for a sync
client, an async client and a pool. ``rediss://`` is TLS with the certificate and
the host name checked; ``redis://`` is plain, with exactly the arguments the
clients were given before; a user name is kept; ``REDIS_TLS`` chooses the scheme
only for an address built from ``REDIS_HOST``/``REDIS_PORT``/``REDIS_PASSWORD``.
A parse error never quotes the address.

Every credential and credentialed address here is put together at run time, so no
single literal looks like one to the secret scanner. Builds real redis-py clients
and connection objects; none of them connects.
"""
from __future__ import annotations

import os
import ssl
import subprocess
import sys
from pathlib import Path

import pytest
import redis
import redis.asyncio as aioredis
from redis.asyncio.connection import SSLConnection as AsyncSSLConnection

from config import config
from core.redis.connection import (
    ConnectionKind,
    RedisTarget,
    connection_kwargs,
    parse_redis_url,
    redis_target_from_config,
    redis_target_from_host_settings,
    redis_target_from_url_or_host_settings,
)

ORCHESTRATOR = Path(__file__).resolve().parents[1]
HOST = "cache.example"
PASSWORD = "-".join(("not", "a", "real", "value"))
TLS_KEYS = {"ssl", "ssl_cert_reqs", "ssl_check_hostname", "connection_class"}


def address(scheme: str, user: str = "", password=None, host: str = HOST, rest: str = ":6380/2") -> str:
    """A Redis address joined from its parts (scheme, user, password, host, rest)."""
    userinfo = user + ("" if password is None else ":" + password)
    return "".join((scheme, "://", userinfo, "@" if userinfo else "", host, rest))


# --------------------------------------------------------------------------- the address


@pytest.mark.parametrize("scheme, tls", [("redis", False), ("rediss", True)])
@pytest.mark.parametrize(
    "user, password, username",
    [("", None, None), ("", PASSWORD, None), ("app", PASSWORD, "app"), ("app", None, "app")],
    ids=["no-credentials", "password-only", "user-and-password", "user-only"],
)
def test_each_scheme_with_and_without_a_user_and_a_password(scheme, tls, user, password, username):
    target = parse_redis_url(address(scheme, user, password))
    assert target == RedisTarget(host=HOST, port=6380, db=2, username=username, password=password, tls=tls)


def test_port_and_database_default_to_6379_and_0():
    assert parse_redis_url(address("rediss", rest="")) == RedisTarget(host=HOST, port=6379, db=0, tls=True)
    assert parse_redis_url(address("redis", rest="/")) == RedisTarget(host=HOST)


def test_credentials_are_percent_decoded_as_redis_py_does():
    encoded_user, encoded_password = "ops%2F" + "team", "p%40ss" + "%2Fw%25rd"
    target = parse_redis_url(address("rediss", encoded_user, encoded_password))
    assert (target.username, target.password) == ("ops/team", "".join(("p", "@ss", "/w", "%rd")))


@pytest.mark.parametrize(
    "url",
    [address("unix"), address("http"), "redis:///0", address("redis", rest=":6379/not-a-number"),
     address("redis", rest=":not-a-port/0"), "redis://[unclosed:6379/0"],
)
def test_an_unusable_address_is_an_error(url):
    with pytest.raises(ValueError):
        parse_redis_url(url)


SECRET_HEAD = "".join(("leak", "me", "42"))


@pytest.mark.parametrize(
    "url",
    [
        # An unencoded "/" in the password: the library reads the head as a port.
        address("redis", "default", SECRET_HEAD + "/tail", rest=":6379/0"),
        # The same address with the head where the database number goes.
        address("redis", "default", None, rest=":6379/" + SECRET_HEAD),
    ],
    ids=["password-read-as-port", "password-read-as-database"],
)
def test_a_parse_error_never_quotes_the_address(url):
    with pytest.raises(ValueError) as caught:
        parse_redis_url(url)
    assert SECRET_HEAD not in str(caught.value)
    assert caught.value.__cause__ is None and caught.value.__suppress_context__


def test_the_password_never_reaches_a_log_line():
    target = RedisTarget(host=HOST, username="app", password=PASSWORD, tls=True)
    assert PASSWORD not in repr(target)
    assert target.describe() == f"{HOST}:6379 db=0 tls=on"


# --------------------------------------------------------------------------- the settings


@pytest.fixture
def no_redis_settings(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    for name in ("REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD"):
        monkeypatch.setattr(config, name, None, raising=False)
    monkeypatch.setattr(config, "REDIS_TLS", False, raising=False)
    monkeypatch.setattr(config, "REDIS_DB", "0", raising=False)
    return monkeypatch


def _tls_parsed_with(value: str) -> tuple:
    """REDIS_TLS and SQL_DEBUG, both set to ``value``, as a fresh process parses them."""
    env = {**os.environ, "REDIS_TLS": value, "SQL_DEBUG": value}
    out = subprocess.run(
        [sys.executable, "-c", "from config import Config; print(Config.REDIS_TLS, Config.SQL_DEBUG)"],
        cwd=ORCHESTRATOR, env=env, capture_output=True, text=True, timeout=60, check=True,
    ).stdout.split()
    return tuple(word == "True" for word in out[-2:])


@pytest.mark.parametrize(
    "value, on",
    [("true", True), ("TRUE", True), ("True", True), ("false", False), ("", False), ("1", False),
     ("yes", False), ("on", False)],
)
def test_redis_tls_reads_like_the_other_on_off_settings(value, on):
    """``os.getenv(NAME, "false").lower() == "true"``, the form most of config.py's
    on/off settings use (SQL_DEBUG among them): only "true", in any case, is on."""
    assert _tls_parsed_with(value) == (on, on)


def test_redis_tls_is_off_by_default():
    """The test environment sets no REDIS_TLS, so this is the default."""
    assert type(config).__dict__["REDIS_TLS"] is False


def test_nothing_configured_is_none_for_the_main_client(no_redis_settings):
    assert redis_target_from_config() is None


def test_a_host_without_a_port_is_not_configured_for_the_main_client(no_redis_settings):
    """The main client's old rule: host or port missing means Redis is disabled."""
    no_redis_settings.setattr(config, "REDIS_HOST", HOST, raising=False)
    assert redis_target_from_config() is None


def test_a_host_without_a_port_is_that_host_on_6379_for_the_host_settings(no_redis_settings):
    """The task queue's and the shared context's old rule: ``REDIS_PORT or 6379``."""
    no_redis_settings.setattr(config, "REDIS_HOST", HOST, raising=False)
    no_redis_settings.setattr(config, "REDIS_PASSWORD", PASSWORD, raising=False)
    assert redis_target_from_url_or_host_settings(4) == RedisTarget(host=HOST, port=6379, db=4, password=PASSWORD)


def test_no_host_is_this_machine_for_the_host_settings(no_redis_settings):
    assert redis_target_from_host_settings(0) == RedisTarget(host="127.0.0.1", port=6379, db=0)


def test_redis_tls_applies_to_the_host_settings(no_redis_settings):
    no_redis_settings.setattr(config, "REDIS_TLS", True, raising=False)
    assert redis_target_from_host_settings(0).tls is True


def test_redis_url_is_used_as_it_is(no_redis_settings):
    no_redis_settings.setenv("REDIS_URL", address("rediss", "app", PASSWORD, rest=":6380/4"))
    no_redis_settings.setattr(config, "REDIS_HOST", "parts.example", raising=False)
    no_redis_settings.setattr(config, "REDIS_PORT", "6379", raising=False)
    expected = RedisTarget(host=HOST, port=6380, db=4, username="app", password=PASSWORD, tls=True)
    assert redis_target_from_config() == expected
    assert redis_target_from_url_or_host_settings(7) == expected, "the address's database, not the caller's"


def test_redis_tls_does_not_override_the_scheme_of_a_redis_url(no_redis_settings):
    no_redis_settings.setenv("REDIS_URL", address("redis", rest=":6379/0"))
    no_redis_settings.setattr(config, "REDIS_TLS", True, raising=False)
    assert redis_target_from_config().tls is False


def test_an_address_built_from_the_parts_is_not_a_redis_url_setting(no_redis_settings):
    no_redis_settings.setattr(config, "REDIS_HOST", HOST, raising=False)
    no_redis_settings.setattr(config, "REDIS_PORT", "6380", raising=False)
    assert config.REDIS_URL and config.redis_url_as_set() is None
    assert redis_target_from_url_or_host_settings(5).db == 5


@pytest.mark.parametrize("tls", [False, True])
@pytest.mark.parametrize(
    "password", [None, PASSWORD, "".join(("s/cr", "@t", "%41#?"))], ids=["none", "plain", "url-special"]
)
def test_redis_tls_chooses_the_scheme_for_an_address_built_from_its_parts(no_redis_settings, tls, password):
    no_redis_settings.setattr(config, "REDIS_HOST", HOST, raising=False)
    no_redis_settings.setattr(config, "REDIS_PORT", "6380", raising=False)
    no_redis_settings.setattr(config, "REDIS_PASSWORD", password, raising=False)
    no_redis_settings.setattr(config, "REDIS_TLS", tls, raising=False)
    no_redis_settings.setattr(config, "REDIS_DB", "5", raising=False)  # not part of the address, as before
    assert config.REDIS_URL.startswith("rediss://" if tls else "redis://")
    assert redis_target_from_config() == RedisTarget(host=HOST, port=6380, db=0, password=password, tls=tls)


# --------------------------------------------------------------------------- the arguments


@pytest.mark.parametrize("kind", list(ConnectionKind))
def test_a_plain_address_gives_exactly_the_arguments_it_did_before(kind):
    kwargs = connection_kwargs(RedisTarget(host=HOST, port=6380, db=2, password=PASSWORD), kind)
    assert kwargs == {"host": HOST, "port": 6380, "db": 2, "password": PASSWORD}


@pytest.mark.parametrize("kind", list(ConnectionKind))
def test_a_user_name_is_passed_on(kind):
    kwargs = connection_kwargs(RedisTarget(host=HOST, username="app", password=PASSWORD), kind)
    assert kwargs["username"] == "app" and kwargs["password"] == PASSWORD
    assert not TLS_KEYS & set(kwargs)


@pytest.mark.parametrize("kind", [ConnectionKind.SYNC, ConnectionKind.ASYNC])
def test_tls_for_a_client_checks_the_certificate_and_the_host_name(kind):
    kwargs = connection_kwargs(RedisTarget(host=HOST, tls=True), kind)
    assert kwargs["ssl"] is True
    assert kwargs["ssl_cert_reqs"] == "required"
    assert kwargs["ssl_check_hostname"] is True
    assert "connection_class" not in kwargs


def test_tls_for_a_pool_uses_the_tls_connection_and_checks_the_certificate():
    kwargs = connection_kwargs(RedisTarget(host=HOST, tls=True), ConnectionKind.POOL)
    assert kwargs["connection_class"] is redis.SSLConnection
    assert kwargs["ssl_cert_reqs"] == "required"
    assert kwargs["ssl_check_hostname"] is True
    assert "ssl" not in kwargs, "a pool takes the connection class, not ssl="


def _connection_of(pool):
    """The connection object a pool would hand out (built, never connected)."""
    return pool.connection_class(**pool.connection_kwargs)


def test_real_clients_and_pool_verify_the_server_for_rediss():
    target = parse_redis_url(address("rediss", "app", PASSWORD, rest=":6380/1"))
    sync_client = redis.Redis(**connection_kwargs(target, ConnectionKind.SYNC))
    async_client = aioredis.Redis(**connection_kwargs(target, ConnectionKind.ASYNC))
    pool = redis.ConnectionPool(**connection_kwargs(target, ConnectionKind.POOL))

    assert sync_client.connection_pool.connection_class is redis.SSLConnection
    assert pool.connection_class is redis.SSLConnection
    assert async_client.connection_pool.connection_class is AsyncSSLConnection
    for conn_pool in (sync_client.connection_pool, pool, async_client.connection_pool):
        conn = _connection_of(conn_pool)
        assert conn.cert_reqs == ssl.CERT_REQUIRED
        assert conn.check_hostname is True
        assert (conn.host, conn.port, conn.db, conn.username, conn.password) == (HOST, 6380, 1, "app", PASSWORD)


def test_real_clients_and_pool_stay_plain_for_redis():
    target = parse_redis_url(address("redis", "", PASSWORD, rest=":6379/0"))
    assert redis.Redis(**connection_kwargs(target, ConnectionKind.SYNC)).connection_pool.connection_class is redis.Connection
    assert redis.ConnectionPool(**connection_kwargs(target, ConnectionKind.POOL)).connection_class is redis.Connection
    async_pool = aioredis.Redis(**connection_kwargs(target, ConnectionKind.ASYNC)).connection_pool
    assert async_pool.connection_class is aioredis.Connection
