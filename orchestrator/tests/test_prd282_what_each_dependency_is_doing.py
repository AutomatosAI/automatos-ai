"""PRD-282 US-202, US-203: GET /health reports what each dependency is doing, from
background results, and /metrics carries one series per dependency.

* Redis, object storage, the search index, the memory store and the model provider each
  get an entry beside the database: up, down or not configured, with when it was checked
  (core/health_dependency_checks.py, run by core/health_dependencies.py).
* The route only reads the last results: with every dependency hanging it answers at once.
* The model provider is never called: its entry is how the last real call ended
  (core/llm/last_call.py, noted by LLMManager._track_usage).
* A dependency down never changes /health's status code; no entry holds an address, a
  credential or error text.
* /metrics: automatos_dependency_up, 1 up, 0 down, no series when not configured.
* The configuration line is healthy for an install on POSTGRES_* with no DATABASE_URL.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError
from redis.exceptions import ConnectionError as RedisConnectionError

from config import config
from core import health_dependencies, health_dependency_checks as checks, health_probe, health_readiness
from core.health_watch import Reading
from core.llm import last_call
from core.qdrant.errors import ResponseHandlingException

NAMES = [name for name, _ in health_dependencies.DEPENDENCIES]
AT_ONCE_S = 0.5
WAIT_S = 5
STATES = {checks.UP, checks.DOWN, checks.NOT_CONFIGURED, health_readiness.UNCHECKED}


def _now():
    return datetime.now(timezone.utc)


def _health(monkeypatch, state):
    """GET /health through main's handler, on ``state`` as the app's state."""
    import main

    monkeypatch.setattr(main.app, "state", state)
    monkeypatch.setattr(health_probe, "database_status", lambda: health_probe.HEALTHY)
    answer = main.health_check()
    if hasattr(answer, "status_code"):
        return answer.status_code, json.loads(answer.body)
    return 200, answer


# --------------------------------------------------------------------------- not configured, with no call


def _refuse(*_args, **_kwargs):
    raise AssertionError("a check called a dependency the install does not have")


def test_each_check_says_not_configured_without_a_call_when_the_dependency_is_absent(monkeypatch):
    """The local edition with every dependency left out."""
    monkeypatch.setattr(type(config), "REDIS_URL", None)
    monkeypatch.setattr("core.redis.client.get_redis_client", _refuse)
    monkeypatch.setattr("core.storage.s3.is_storage_configured", lambda: False)
    monkeypatch.setattr("core.storage.s3.get_s3_client", _refuse)
    monkeypatch.setattr(config, "S3_VECTORS_ENABLED", False)
    monkeypatch.setattr("modules.search.vector_store.backends.s3_vectors_backend.S3VectorsBackend", _refuse)
    monkeypatch.setattr(config, "QDRANT_URL", "")
    monkeypatch.setattr("core.qdrant.client.QdrantRestClient", _refuse)
    monkeypatch.setattr(last_call, "_last", None)
    monkeypatch.setattr("core.llm.providers.env_api_key", lambda slug: None)
    monkeypatch.setattr("core.llm.providers.runs_on_workload_identity", lambda slug: False)

    for _, attr in health_dependencies.DEPENDENCIES:
        assert getattr(checks, attr)() == checks.NOT_CONFIGURED, attr


# --------------------------------------------------------------------------- each check, up and down


class _Redis:
    def __init__(self, error=None):
        self.error = error

    def get_redis(self):
        return self

    def ping(self):
        if self.error is not None:
            raise self.error
        return True


@pytest.mark.parametrize("client,expected", [
    (_Redis(), checks.UP),
    (_Redis(RedisConnectionError("refused")), checks.DOWN),
    (None, checks.DOWN),  # configured, but its client could not be set up
])
def test_redis_is_a_ping_on_the_shared_client(monkeypatch, client, expected):
    monkeypatch.setattr(type(config), "REDIS_URL", property(lambda self: "redis://" + "health-test:6379/0"))
    monkeypatch.setattr("core.redis.client.get_redis_client", lambda: client)
    assert checks.redis_state() == expected


class _Bucket:
    def __init__(self, error=None):
        self.error, self.asked = error, []

    def head_bucket(self, Bucket):
        self.asked.append(Bucket)
        if self.error is not None:
            raise self.error


@pytest.mark.parametrize("error,expected", [
    (None, checks.UP),
    (ClientError({"Error": {"Code": "403"}}, "HeadBucket"), checks.DOWN),
    (EndpointConnectionError(endpoint_url="http://storage-test:9000"), checks.DOWN),
])
def test_object_storage_is_head_bucket_on_the_fast_fail_client(monkeypatch, error, expected):
    from core.storage import s3

    bucket, profiles = _Bucket(error), []
    monkeypatch.setattr(s3, "is_storage_configured", lambda: True)
    monkeypatch.setattr(s3, "get_s3_client", lambda profile: profiles.append(profile) or bucket)
    assert checks.object_storage_state() == expected
    assert bucket.asked == [config.S3_DOCUMENTS_BUCKET] and profiles == [s3.FAST_FAIL]


def _vector_backend(calls, error=None):
    class Backend:
        def __init__(self, workspace_id):
            self.bucket_name = config.S3_VECTORS_BUCKET.replace("{workspace_id}", workspace_id)
            self.index_name = "health-test-index"
            self.client = self

        def get_index(self, **kwargs):
            calls.append(("get_index", kwargs))
            if error is not None:
                raise error

        def list_vector_buckets(self, **kwargs):
            calls.append(("list_vector_buckets", kwargs))

    return Backend


@pytest.mark.parametrize("bucket,call", [("shared-health-test", "get_index"),
                                         ("health-test-{workspace_id}", "list_vector_buckets")])
def test_the_search_index_is_a_metadata_call_through_the_backends_client(monkeypatch, bucket, call):
    calls = []
    monkeypatch.setattr(config, "S3_VECTORS_ENABLED", True)
    monkeypatch.setattr(config, "S3_VECTORS_BUCKET", bucket)
    monkeypatch.setattr("modules.search.vector_store.backends.s3_vectors_backend.S3VectorsBackend",
                        _vector_backend(calls))
    assert checks.search_index_state() == checks.UP
    assert [name for name, _ in calls] == [call]


def test_a_search_index_that_refuses_is_down(monkeypatch):
    monkeypatch.setattr(config, "S3_VECTORS_ENABLED", True)
    monkeypatch.setattr(config, "S3_VECTORS_BUCKET", "shared-health-test")
    monkeypatch.setattr("modules.search.vector_store.backends.s3_vectors_backend.S3VectorsBackend",
                        _vector_backend([], ClientError({"Error": {"Code": "NotFoundException"}}, "GetIndex")))
    assert checks.search_index_state() == checks.DOWN


def _memory_client(seen, error=None):
    class Client:
        def __init__(self, url, api_key=None, timeout=None):
            seen.append(("built", timeout))

        async def get_collections(self):
            seen.append(("listed", None))
            if error is not None:
                raise error

        async def aclose(self):
            seen.append(("closed", None))

    return Client


@pytest.mark.parametrize("error,expected", [
    (None, checks.UP),
    (ResponseHandlingException(OSError("no answer")), checks.DOWN),
])
def test_the_memory_store_lists_its_collections_and_closes_its_client(monkeypatch, error, expected):
    seen = []
    monkeypatch.setattr(config, "QDRANT_URL", "http://" + "memory-test:6333")
    monkeypatch.setattr("core.qdrant.client.QdrantRestClient", _memory_client(seen, error))
    assert checks.memory_store_state() == expected
    assert seen == [("built", config.HEALTH_DEPENDENCY_CHECK_TIMEOUT_SECONDS), ("listed", None), ("closed", None)]


def test_the_model_provider_is_how_the_last_real_call_ended_and_never_a_call(monkeypatch):
    monkeypatch.setattr(last_call, "_last", None)
    monkeypatch.setattr("core.llm.providers.runs_on_workload_identity", lambda slug: False)
    monkeypatch.setattr("core.llm.providers.env_api_key", lambda slug: "set" if slug == "openai" else None)
    assert checks.model_provider_state() == health_readiness.UNCHECKED  # named, no call has ended yet
    last_call.note("success")
    assert checks.model_provider_state() == checks.UP
    last_call.note("error")
    assert checks.model_provider_state() == checks.DOWN


def test_every_real_call_through_the_manager_notes_how_it_ended(monkeypatch):
    """LLMManager._track_usage runs for every call, success or failure, async or sync."""
    from core.llm.manager import LLMManager

    monkeypatch.setattr(last_call, "_last", None)
    manager = SimpleNamespace(_log_cost_audit=lambda **kwargs: None, _record_usage=lambda *args: None,
                              _note_ask_cost=lambda counts: None)
    LLMManager._track_usage(manager, None, time.monotonic(), status="error")
    assert last_call.last() is not None and last_call.last().succeeded is False
    LLMManager._track_usage(manager, None, time.monotonic(), status="success")
    assert last_call.last().succeeded is True


# --------------------------------------------------------------------------- the route only reads


def test_with_every_dependency_hanging_the_route_answers_at_once(monkeypatch):
    release = threading.Event()

    def hangs():
        release.wait(WAIT_S)
        return checks.UP

    for _, attr in health_dependencies.DEPENDENCIES:
        monkeypatch.setattr(checks, attr, hangs)
    monkeypatch.setattr(health_readiness, "_database_status", hangs)
    monkeypatch.setattr(config, "HEALTH_DEPENDENCY_CHECK_TIMEOUT_SECONDS", WAIT_S)
    state = SimpleNamespace()

    async def scenario():
        health_dependencies.start(state)
        await asyncio.sleep(0.05)  # every check is on its thread now, hanging
        started = time.monotonic()
        answered = _health(monkeypatch, state)
        took = time.monotonic() - started
        await health_dependencies.stop(state)
        return answered, took

    try:
        (status, body), took = asyncio.run(scenario())
    finally:
        release.set()
    assert status == 200 and took < AT_ONCE_S
    for name in NAMES:
        assert body["components"][name] == {"state": health_readiness.UNCHECKED, "checked_at": None}, name


def _state_with(readings):
    watches = health_dependencies.dependency_watches()
    for watch in watches:
        state = readings.get(watch.name)
        watch.last = Reading(state, _now()) if state else None
    return SimpleNamespace(dependency_watches=watches)


def test_with_redis_down_health_still_answers_200_and_says_so(monkeypatch):
    state = _state_with({name: checks.DOWN for name in NAMES})
    status, body = _health(monkeypatch, state)
    assert status == 200
    assert body["components"]["redis"]["state"] == checks.DOWN and body["components"]["redis"]["checked_at"]
    assert body["status"] == health_probe.overall_status(
        {key: value for key, value in body["components"].items() if isinstance(value, str)})


def test_the_database_alone_still_decides_the_status_code(monkeypatch):
    import main

    monkeypatch.setattr(main.app, "state", _state_with({name: checks.UP for name in NAMES}))
    monkeypatch.setattr(health_probe, "database_status", lambda: health_probe.DATABASE_STUCK)
    assert main.health_check().status_code == 503


def test_no_entry_holds_an_address_a_credential_or_error_text(monkeypatch, caplog):
    secret = uuid.uuid4().hex
    address = f"https://{uuid.uuid4().hex}.example:6379"

    def fails():
        raise RuntimeError(f"cannot reach {address} with {secret}")

    for _, attr in health_dependencies.DEPENDENCIES:
        monkeypatch.setattr(checks, attr, fails)
    monkeypatch.setattr(config, "HEALTH_DEPENDENCY_CHECK_TIMEOUT_SECONDS", WAIT_S)
    watches = health_dependencies.dependency_watches()

    async def scenario():
        for watch in watches:
            await watch.check_now()

    with caplog.at_level("ERROR", logger="core.health_watch"):
        asyncio.run(scenario())
    status, body = _health(monkeypatch, SimpleNamespace(dependency_watches=watches))
    entries = {name: body["components"][name] for name in NAMES}
    text = json.dumps(entries)
    assert secret not in text and address not in text and "cannot reach" not in text
    for entry in entries.values():
        assert set(entry) == {"state", "checked_at"} and entry["state"] in STATES
    assert all(entry["state"] == checks.DOWN for entry in entries.values())


def test_before_any_check_every_dependency_is_listed_unchecked():
    assert health_dependencies.components(SimpleNamespace()) == {
        name: {"state": health_readiness.UNCHECKED, "checked_at": None} for name in NAMES}


# --------------------------------------------------------------------------- /metrics


def _series(name):
    from prometheus_client import REGISTRY

    return REGISTRY.get_sample_value("automatos_dependency_up", {"dependency": name})


def test_metrics_carry_one_series_per_dependency_and_none_when_not_configured():
    health_dependencies.record_gauge("redis", Reading(checks.UP, _now()))
    health_dependencies.record_gauge("memory_store", Reading(checks.DOWN, _now()))
    assert (_series("redis"), _series("memory_store")) == (1.0, 0.0)
    health_dependencies.record_gauge("redis", Reading(checks.NOT_CONFIGURED, _now()))
    health_dependencies.record_gauge("memory_store", Reading(health_readiness.UNCHECKED, _now()))
    assert (_series("redis"), _series("memory_store")) == (None, None)


def test_each_watch_sets_its_series_as_it_reads_and_the_database_has_one_too(monkeypatch):
    monkeypatch.setattr(checks, "redis_state", lambda: checks.UP)
    monkeypatch.setattr(health_readiness, "_database_status", lambda: health_probe.DATABASE_BUSY)
    redis = next(watch for watch in health_dependencies.dependency_watches() if watch.name == "redis")

    async def scenario():
        await redis.check_now()
        await health_readiness.database_watch().check_now()

    asyncio.run(scenario())
    assert (_series("redis"), _series("database")) == (1.0, 1.0)  # busy is ready


# --------------------------------------------------------------------------- the configuration line


@pytest.mark.parametrize("missing,expected", [(None, health_probe.HEALTHY), ("POSTGRES_HOST", health_probe.DEGRADED)])
def test_the_configuration_line_counts_postgres_settings(monkeypatch, missing, expected):
    monkeypatch.setattr(config, "DATABASE_URL", None)
    for name in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB"):
        monkeypatch.setattr(config, name, None if name == missing else uuid.uuid4().hex)
    assert health_probe.config_status() == expected
    status, body = _health(monkeypatch, SimpleNamespace())
    assert body["components"]["config"] == expected
