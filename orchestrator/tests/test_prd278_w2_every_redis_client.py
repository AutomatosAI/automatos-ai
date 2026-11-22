"""PRD-278 US-202: every connection to the platform's Redis goes through one function.

The main pool, the async publish/subscribe client, the chat-turn registry, both
clients of the task queue, the shared-context adapter and the intake progress
client take their arguments from ``core.redis.connection.connection_kwargs``, so
``rediss://`` gives each of them TLS with the certificate checked and a user name
reaches each of them. Their own timeouts, retries and pool size are unchanged, and
so is where each one looked for its Redis (a host with no port, ``REDIS_DB``, a
malformed ``REDIS_URL``), except that the task queue now follows ``REDIS_URL``.

The source guard fails when a file in ``orchestrator/`` outside ``core/redis/``
builds a Redis client or pool itself. Credentials and credentialed addresses are
joined at run time, so no literal looks like one to the secret scanner. Builds real
clients; none of them connects.
"""
from __future__ import annotations

import ast
import asyncio
import logging
import ssl
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
import redis
from redis.asyncio.connection import SSLConnection as AsyncSSLConnection

from config import config
from core.redis import client as redis_client_module
from core.redis.connection import RedisTarget

ORCHESTRATOR = Path(__file__).resolve().parents[1]
HOST = "cache.example"
PASSWORD = "-".join(("not", "a", "real", "value"))
#: Would appear in an error or a log line if a parse error quoted the address.
SECRET_HEAD = "".join(("leak", "me", "42"))


def address(scheme: str, user: str = "", password=None, host: str = HOST, rest: str = ":6380/3") -> str:
    """A Redis address joined from its parts (scheme, user, password, host, rest)."""
    userinfo = user + ("" if password is None else ":" + password)
    return "".join((scheme, "://", userinfo, "@" if userinfo else "", host, rest))


TLS_URL = address("rediss", "app", PASSWORD)
TLS_TARGET = RedisTarget(host=HOST, port=6380, db=3, username="app", password=PASSWORD, tls=True)
#: An unencoded "/" in the password: the URL parser reads its head as a port.
MALFORMED_URL = address("redis", "default", SECRET_HEAD + "/tail", rest=":6379/0")


@pytest.fixture
def redis_settings(monkeypatch):
    """No Redis configured, and the global client unbuilt; tests add what they need."""
    monkeypatch.delenv("REDIS_URL", raising=False)
    for name in ("REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD"):
        monkeypatch.setattr(config, name, None, raising=False)
    monkeypatch.setattr(config, "REDIS_TLS", False, raising=False)
    monkeypatch.setattr(config, "REDIS_DB", "0", raising=False)
    monkeypatch.setattr(redis_client_module, "_redis_client", None)
    monkeypatch.setattr(redis_client_module.RedisClient, "test_connection", lambda self: True)
    return monkeypatch


def _host_settings(monkeypatch, *, port="6380", db="0") -> None:
    monkeypatch.setattr(config, "REDIS_HOST", HOST, raising=False)
    monkeypatch.setattr(config, "REDIS_PORT", port, raising=False)
    monkeypatch.setattr(config, "REDIS_PASSWORD", PASSWORD, raising=False)
    monkeypatch.setattr(config, "REDIS_DB", db, raising=False)


def _assert_checked_tls(conn_pool, ssl_class) -> None:
    assert conn_pool.connection_class is ssl_class
    conn = conn_pool.connection_class(**conn_pool.connection_kwargs)  # built, never connected
    assert conn.cert_reqs == ssl.CERT_REQUIRED
    assert conn.check_hostname is True
    assert (conn.host, conn.port, conn.db, conn.username, conn.password) == (HOST, 6380, 3, "app", PASSWORD)


def _where(client) -> tuple:
    kwargs = client.connection_pool.connection_kwargs
    return kwargs["host"], kwargs["port"], kwargs["db"], kwargs["password"]


def _assert_logged_without_the_address(caplog) -> None:
    text = caplog.text
    assert "REDIS_URL is not a usable Redis address (ValueError)" in text
    assert SECRET_HEAD not in text
    assert all(record.exc_info is None for record in caplog.records)


# --------------------------------------------------------------------------- the main pool


def test_the_main_pool_follows_rediss_and_keeps_the_user(redis_settings):
    redis_settings.setenv("REDIS_URL", TLS_URL)
    client = redis_client_module.get_redis_client()
    assert client is not None and client.target == TLS_TARGET
    _assert_checked_tls(client.pool, redis.SSLConnection)
    kwargs = client.pool.connection_kwargs
    assert kwargs["socket_timeout"] == redis_client_module.REDIS_SOCKET_TIMEOUT_S
    assert client.pool.max_connections == 50


def test_the_main_pool_for_a_plain_address_is_as_before(redis_settings):
    redis_settings.setenv("REDIS_URL", address("redis", "", PASSWORD, rest=":6379/2"))
    client = redis_client_module.get_redis_client()
    assert client.pool.connection_class is redis.Connection
    expected = {
        "host": HOST, "port": 6379, "db": 2, "password": PASSWORD,
        "decode_responses": True,
        "socket_timeout": redis_client_module.REDIS_SOCKET_TIMEOUT_S,
        "socket_connect_timeout": redis_client_module.REDIS_CONNECT_TIMEOUT_S,
        "socket_keepalive": True,
        "health_check_interval": redis_client_module.REDIS_HEALTH_CHECK_INTERVAL_S,
    }
    kwargs = client.pool.connection_kwargs
    assert {key: kwargs.get(key) for key in expected} == expected
    assert not {"username", "ssl_cert_reqs", "ssl_check_hostname"} & set(kwargs)
    assert client.pool.max_connections == 50


def test_the_main_pool_with_redis_tls_on_its_parts(redis_settings):
    _host_settings(redis_settings, db="5")
    redis_settings.setattr(config, "REDIS_TLS", True, raising=False)
    client = redis_client_module.get_redis_client()
    assert client.target == RedisTarget(host=HOST, port=6380, db=0, password=PASSWORD, tls=True), "REDIS_DB unread, as before"
    assert client.pool.connection_class is redis.SSLConnection


def test_nothing_configured_still_disables_redis(redis_settings):
    assert redis_client_module.get_redis_client() is None


def test_a_host_without_a_port_still_disables_the_main_client(redis_settings):
    _host_settings(redis_settings, port=None)
    assert redis_client_module.get_redis_client() is None


def test_a_malformed_redis_url_disables_the_main_client_and_is_logged_without_the_address(redis_settings, caplog):
    redis_settings.setenv("REDIS_URL", MALFORMED_URL)
    caplog.set_level(logging.ERROR, logger="core.redis.client")
    assert redis_client_module.get_redis_client() is None
    _assert_logged_without_the_address(caplog)


# --------------------------------------------------------------------------- publish/subscribe


def test_the_async_pubsub_client_gets_tls_and_still_no_read_timeout(monkeypatch):
    captured: dict = {}

    class _FakePubSub:
        async def subscribe(self, channel):
            captured["channel"] = channel

    class _FakeRedis:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def pubsub(self):
            return _FakePubSub()

    monkeypatch.setattr(redis_client_module.aioredis, "Redis", _FakeRedis)
    asyncio.run(redis_client_module.RedisClient(TLS_TARGET).get_async_pubsub("events"))
    assert (captured["ssl"], captured["ssl_cert_reqs"], captured["ssl_check_hostname"]) == (True, "required", True)
    assert (captured["host"], captured["username"], captured["password"], captured["db"]) == (HOST, "app", PASSWORD, 3)
    assert captured["socket_connect_timeout"] == redis_client_module.REDIS_CONNECT_TIMEOUT_S
    assert "socket_timeout" not in captured


# --------------------------------------------------------------------------- chat turns


def test_the_chat_turn_registry_gets_tls_and_keeps_its_fast_no_retry_shape(monkeypatch):
    from services.chat_turns import TurnRegistry

    monkeypatch.setattr(redis_client_module, "get_redis_client", lambda: SimpleNamespace(target=TLS_TARGET))
    client = TurnRegistry()._client()
    _assert_checked_tls(client.connection_pool, AsyncSSLConnection)
    kwargs = client.connection_pool.connection_kwargs
    assert kwargs["socket_timeout"] == 1 and kwargs["socket_connect_timeout"] == 1
    assert client.get_retry()._retries == 0


# --------------------------------------------------------------------------- the task queue


def _queue():
    from core.task_runner.queued import QueuedTaskRunner

    return QueuedTaskRunner()


def test_the_task_queue_follows_redis_url_over_the_host_settings(redis_settings):
    redis_settings.setenv("REDIS_URL", TLS_URL)
    _host_settings(redis_settings, port="6379", db="7")
    redis_settings.setattr(config, "REDIS_HOST", "parts.example", raising=False)
    runner = _queue()
    _assert_checked_tls(runner._redis.connection_pool, redis.SSLConnection)
    _assert_checked_tls(runner._aredis.connection_pool, AsyncSSLConnection)
    assert runner._redis.connection_pool.connection_kwargs["socket_timeout"] == 5


def test_the_task_queue_on_host_settings_reads_them_as_before(redis_settings):
    _host_settings(redis_settings)
    runner = _queue()
    assert _where(runner._redis) == _where(runner._aredis) == (HOST, 6380, 0, PASSWORD)


def test_the_task_queue_on_host_settings_uses_redis_db(redis_settings):
    """REDIS_DB is the queue's database on the host settings, as before (the main
    pool has always used 0 there)."""
    _host_settings(redis_settings, db="7")
    runner = _queue()
    assert _where(runner._redis) == _where(runner._aredis) == (HOST, 6380, 7, PASSWORD)
    assert redis_client_module.get_redis_client().target.db == 0


def test_the_task_queue_on_a_host_without_a_port_dials_that_host_on_6379(redis_settings):
    _host_settings(redis_settings, port=None, db="2")
    assert _where(_queue()._aredis) == (HOST, 6379, 2, PASSWORD)


def test_the_task_queue_with_nothing_configured_still_dials_this_machine(redis_settings):
    redis_settings.setattr(config, "REDIS_DB", "3", raising=False)
    runner = _queue()
    assert _where(runner._aredis) == ("127.0.0.1", 6379, 3, None)
    assert runner._aredis.connection_pool.connection_class is not AsyncSSLConnection


def test_a_malformed_redis_url_never_stops_the_task_queue(redis_settings, caplog):
    """It is logged without the address, and the queue builds itself from the host
    settings exactly as it did before it read REDIS_URL at all."""
    redis_settings.setenv("REDIS_URL", MALFORMED_URL)
    _host_settings(redis_settings, db="4")
    caplog.set_level(logging.ERROR, logger="core.task_runner.queued")
    runner = _queue()
    assert _where(runner._redis) == _where(runner._aredis) == (HOST, 6380, 4, PASSWORD)
    _assert_logged_without_the_address(caplog)


def test_the_task_queue_reads_no_host_setting_itself():
    source = (ORCHESTRATOR / "core" / "task_runner" / "queued.py").read_text(encoding="utf-8")
    for name in ("REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD"):
        assert f"config.{name}" not in source


# --------------------------------------------------------------------------- the other clients


def test_the_shared_context_adapter_gets_tls(redis_settings):
    from modules.context.adapters import redis_context

    redis_settings.setenv("REDIS_URL", TLS_URL)
    client = asyncio.run(redis_context._get_async_redis())
    _assert_checked_tls(client.connection_pool, AsyncSSLConnection)


@pytest.mark.parametrize(
    "port, expected_port, host, expected_host",
    [("6380", 6380, HOST, HOST), (None, 6379, HOST, HOST), (None, 6379, None, "127.0.0.1")],
    ids=["host-and-port", "host-without-port", "nothing"],
)
def test_the_shared_context_adapter_reads_the_host_settings_as_before(redis_settings, port, expected_port, host, expected_host):
    from modules.context.adapters import redis_context

    _host_settings(redis_settings, port=port, db="6")  # REDIS_DB was never read here
    redis_settings.setattr(config, "REDIS_HOST", host, raising=False)
    client = asyncio.run(redis_context._get_async_redis())
    assert _where(client) == (expected_host, expected_port, 0, PASSWORD)


def test_a_malformed_redis_url_raises_from_the_shared_context_adapter_without_the_address(redis_settings):
    from modules.context.adapters import redis_context

    redis_settings.setenv("REDIS_URL", MALFORMED_URL)
    with pytest.raises(ValueError) as caught:
        asyncio.run(redis_context._get_async_redis())
    assert SECRET_HEAD not in str(caught.value)


def _fake_async_redis(captured: dict):
    class _FakeRedis:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def ping(self):
            return True

    return _FakeRedis


def test_the_intake_progress_client_gets_tls(redis_settings):
    from modules.intake import progress

    captured: dict = {}
    fake = _fake_async_redis(captured)
    redis_settings.setenv("REDIS_URL", TLS_URL)
    redis_settings.setattr(progress, "_async_redis", None)
    redis_settings.setattr("redis.asyncio.Redis", fake)
    assert isinstance(asyncio.run(progress._get_async_redis()), fake)
    assert (captured["ssl"], captured["ssl_cert_reqs"], captured["ssl_check_hostname"]) == (True, "required", True)
    assert (captured["username"], captured["db"], captured["decode_responses"]) == ("app", 3, True)


def test_a_malformed_redis_url_disables_intake_progress_and_is_logged_without_the_address(redis_settings, caplog):
    from modules.intake import progress

    redis_settings.setenv("REDIS_URL", MALFORMED_URL)
    redis_settings.setattr(progress, "_async_redis", None)
    caplog.set_level(logging.ERROR, logger="modules.intake.progress")
    assert asyncio.run(progress._get_async_redis()) is None
    _assert_logged_without_the_address(caplog)


# --------------------------------------------------------------------------- the source guard

#: Files that may open a Redis connection without core.redis.connection, and why.
EXEMPT = {
    "core/credentials/tester.py": "tests a user's own saved Redis credential: the user's Redis, not the platform's",
    "core/credentials/service.py": "tests a user's own saved Redis credential: the user's Redis, not the platform's",
}
SKIP_DIRS = frozenset({"tests", "__pycache__", "node_modules", ".venv", "venv"})
CONNECTION_MODULE = "core/redis/"
CONSTRUCTORS = frozenset({
    "Redis", "StrictRedis", "ConnectionPool", "BlockingConnectionPool", "from_url",
    "RedisCluster", "Sentinel",
})
#: APScheduler's job store builds its own redis.Redis from the arguments it is given.
JOB_STORE_MODULE = "apscheduler.jobstores.redis"


def _redis_names(tree: ast.AST):
    """The names a file binds to redis modules, and to redis constructors."""
    modules, constructors = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "redis" or alias.name.startswith("redis."):
                    modules.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "redis" or node.module.startswith("redis."):
                for alias in node.names:
                    (constructors if alias.name in CONSTRUCTORS else modules).add(alias.asname or alias.name)
            elif node.module == JOB_STORE_MODULE:
                constructors.update(alias.asname or alias.name for alias in node.names)
    return modules, constructors


def _root(node: ast.AST):
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def direct_constructions(source: str) -> list:
    """Line numbers where the source builds a Redis client or pool itself."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # another file's invalid escape sequence is not this test's concern
        tree = ast.parse(source)
    modules, constructors = _redis_names(tree)
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in constructors:
            lines.append(node.lineno)
        elif isinstance(func, ast.Attribute) and func.attr in CONSTRUCTORS and _root(func) in modules | constructors:
            lines.append(node.lineno)
    return sorted(lines)


def _sources():
    for path in ORCHESTRATOR.rglob("*.py"):
        rel = path.relative_to(ORCHESTRATOR).as_posix()
        if SKIP_DIRS.intersection(path.relative_to(ORCHESTRATOR).parts) or rel.startswith(CONNECTION_MODULE):
            continue
        yield rel, path.read_text(encoding="utf-8", errors="ignore")


def test_no_redis_client_or_pool_is_built_outside_core_redis():
    offenders = {
        rel: lines for rel, source in _sources()
        if rel not in EXEMPT and (lines := direct_constructions(source))
    }
    assert offenders == {}, (
        "Build Redis connections from core.redis.connection.connection_kwargs, so TLS and the "
        f"user name are honoured. Built directly in: {offenders}"
    )


def test_each_exemption_still_builds_a_client():
    """A stale exemption silently widens the guard."""
    for rel in EXEMPT:
        assert direct_constructions((ORCHESTRATOR / rel).read_text(encoding="utf-8")), rel


@pytest.mark.parametrize(
    "source",
    [
        "import redis\nredis.Redis(host='h')",
        "import redis\nredis.ConnectionPool(host='h')",
        "import redis.asyncio as aioredis\naioredis.from_url('redis://h')",
        "import redis.asyncio as aioredis\naioredis.Redis.from_url('redis://h')",
        "import redis.asyncio\nredis.asyncio.Redis(host='h')",
        "from redis import asyncio as aioredis\naioredis.Redis(host='h')",
        "from redis import Redis\nRedis(host='h')",
        "from redis.asyncio import Redis as R\nR.from_url('redis://h')",
        "from redis.cluster import RedisCluster\nRedisCluster(host='h')",
        "def f():\n    import redis as _r\n    return _r.StrictRedis()",
        "from apscheduler.jobstores.redis import RedisJobStore\nRedisJobStore(host='h')",
    ],
)
def test_the_guard_catches_each_way_of_building_one(source):
    assert direct_constructions(source)


@pytest.mark.parametrize(
    "source",
    [
        "from redis.exceptions import RedisError\ntry:\n    pass\nexcept RedisError:\n    pass",
        "import redis\ndef f(client: redis.Redis = None):\n    return client",
        "from core.redis.client import get_redis_client\nget_redis_client()",
        "FileUploadable.from_url('https://example.invalid/x')",
    ],
)
def test_the_guard_leaves_other_code_alone(source):
    assert direct_constructions(source) == []
