"""PRD-282 US-202, US-203: GET /health reports what each dependency is doing, from
background results, and /metrics carries one series per dependency.

* Redis, object storage, the search index, the memory store and the model provider each
  get an entry beside the database: up, down or not configured, with when it was checked,
  or unchecked (core/health_dependency_checks.py, run by core/health_dependencies.py).
* A missing bucket or index is up (the store answered), unless core may not create the
  index; a refusal for permissions is neither up nor down; anything else is down.
* The model provider is never called: its entry is how the last provider call through
  the manager ended (core/llm/last_call.py), and only reach or acceptance failures count.
* The route only reads the current readings: with every dependency hanging it answers at
  once, and a stale reading reads unchecked, with no series.
* A dependency down never changes /health's status code; no entry holds an address, a
  credential, error text or a state outside the fixed set.
* The configuration line is healthy for an install on POSTGRES_* with no DATABASE_URL.
"""
from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest
from botocore.exceptions import ClientError, ConnectTimeoutError, EndpointConnectionError, ReadTimeoutError
from redis.exceptions import ConnectionError as RedisConnectionError

from config import QDRANT_URL_DEFAULT, config
from core import health_dependencies, health_dependency_checks as checks, health_probe, health_readiness
from core.health_watch import Reading
from core.llm import last_call
from core.monitoring.automatos_metrics import set_dependency_up
from core.qdrant.errors import ResponseHandlingException, UnexpectedResponse
from tests.test_prd282_ready_means_the_database_answers import _until

NAMES = [name for name, _ in health_dependencies.DEPENDENCIES]
AT_ONCE_S = 0.5
WAIT_S = 5
SHORT_LIMIT_S = 0.05
STATES = {checks.UP, checks.DOWN, checks.NOT_CONFIGURED, health_readiness.UNCHECKED}
UNCHECKED_ENTRY = {"state": health_readiness.UNCHECKED, "checked_at": None}
VECTORS_MODULE = "modules.search.vector_store.backends.s3_vectors_backend"


@pytest.fixture(autouse=True)
def _no_series_left_behind():
    """Every series a test here sets is removed after it."""
    yield
    for name in [*NAMES, "database"]:
        set_dependency_up(name, None)


@pytest.fixture(autouse=True)
def _fresh_redis_clients(monkeypatch):
    monkeypatch.setattr(checks, "_redis_clients", {})


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


def _series(name):
    from prometheus_client import REGISTRY

    return REGISTRY.get_sample_value("automatos_dependency_up", {"dependency": name})


def _watch(name):
    return next(watch for watch in health_dependencies.dependency_watches() if watch.name == name)


def _client_error(code, status, operation):
    return ClientError({"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}}, operation)


# --------------------------------------------------------------------------- not configured, with no call


def _refuse(*_args, **_kwargs):
    raise AssertionError("a check called a dependency the install does not have")


def test_each_check_says_not_configured_without_a_call_when_the_dependency_is_absent(monkeypatch):
    """The local edition with every dependency left out: no client is even built."""
    monkeypatch.setattr(type(config), "REDIS_URL", None)
    monkeypatch.setattr(checks, "_new_redis_client", _refuse)
    monkeypatch.setattr("core.storage.s3.is_storage_configured", lambda: False)
    monkeypatch.setattr("core.storage.s3.get_s3_client", _refuse)
    monkeypatch.setattr(config, "S3_VECTORS_ENABLED", False)
    monkeypatch.setattr(f"{VECTORS_MODULE}.metadata_client", _refuse)
    monkeypatch.setattr(config, "QDRANT_URL", "")
    monkeypatch.setattr("core.qdrant.client.QdrantRestClient", _refuse)

    for attr in ("redis_state", "object_storage_state", "search_index_state", "memory_store_state"):
        assert getattr(checks, attr)() == checks.NOT_CONFIGURED, attr


# --------------------------------------------------------------------------- Redis


class _Redis:
    def __init__(self, error=None):
        self.error, self.pings = error, 0

    def ping(self):
        self.pings += 1
        if self.error is not None:
            raise self.error
        return True


def _redis_at(monkeypatch):
    monkeypatch.setattr(type(config), "REDIS_URL", property(lambda self: "redis://" + "health-test:6379/0"))
    monkeypatch.setattr("core.redis.client.get_redis_client", _refuse)  # the shared client is never touched


@pytest.mark.parametrize("error,expected", [(None, checks.UP), (RedisConnectionError("refused"), checks.DOWN),
                                            (TimeoutError("slow"), checks.DOWN)])
def test_redis_is_a_ping_on_a_client_of_its_own_built_once(monkeypatch, error, expected):
    _redis_at(monkeypatch)
    client, built = _Redis(error), []
    monkeypatch.setattr(checks, "_new_redis_client", lambda target: built.append(target) or client)
    assert (checks.redis_state(), checks.redis_state()) == (expected, expected)
    assert len(built) == 1 and client.pings == 2


def test_the_redis_client_is_built_through_the_factory_with_short_timeouts_and_no_retries():
    from core.redis.client import REDIS_CONNECT_TIMEOUT_S, REDIS_SOCKET_TIMEOUT_S
    from core.redis.connection import parse_redis_url

    client = checks._new_redis_client(parse_redis_url("redis://" + "health-test:6379/0"))  # builds, never dials
    kwargs = client.connection_pool.connection_kwargs
    assert (kwargs["socket_timeout"], kwargs["socket_connect_timeout"]) == (REDIS_SOCKET_TIMEOUT_S,
                                                                            REDIS_CONNECT_TIMEOUT_S)
    assert client.get_retry()._retries == 0


def test_redis_checks_at_once_on_several_threads_build_one_client(monkeypatch):
    _redis_at(monkeypatch)
    built, start = [], threading.Barrier(4)

    def slow_build(target):
        built.append(target)
        time.sleep(0.05)  # widens the window two builders would race in
        return _Redis()

    monkeypatch.setattr(checks, "_new_redis_client", slow_build)
    answers = []
    threads = [threading.Thread(target=lambda: (start.wait(), answers.append(checks.redis_state())))
               for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(WAIT_S)
    assert answers == [checks.UP] * 4 and len(built) == 1


def test_a_redis_address_that_cannot_be_used_is_down_without_a_client(monkeypatch):
    monkeypatch.setattr(type(config), "REDIS_URL", property(lambda self: "http://" + "health-test:6379"))
    monkeypatch.setattr(checks, "_new_redis_client", _refuse)
    assert checks.redis_state() == checks.DOWN


# --------------------------------------------------------------------------- object storage


class _Bucket:
    def __init__(self, error=None):
        self.error, self.asked = error, []

    def head_bucket(self, Bucket):
        self.asked.append(Bucket)
        if self.error is not None:
            raise self.error


@pytest.mark.parametrize("error,expected", [
    (None, checks.UP),
    (_client_error("404", 404, "HeadBucket"), checks.UP),          # the store answered: no such bucket
    (_client_error("NoSuchBucket", 404, "HeadBucket"), checks.UP),
    (_client_error("403", 403, "HeadBucket"), checks.NOT_PERMITTED),
    (_client_error("AccessDenied", 403, "HeadBucket"), checks.NOT_PERMITTED),
    (_client_error("500", 500, "HeadBucket"), checks.DOWN),
    (_client_error("503", 503, "HeadBucket"), checks.DOWN),
    (EndpointConnectionError(endpoint_url="http://" + "storage-test:9000"), checks.DOWN),
    (ConnectTimeoutError(endpoint_url="http://" + "storage-test:9000"), checks.DOWN),
    (ReadTimeoutError(endpoint_url="http://" + "storage-test:9000"), checks.DOWN),
])
def test_object_storage_is_head_bucket_on_the_fast_fail_client(monkeypatch, error, expected):
    from core.storage import s3

    bucket, profiles = _Bucket(error), []
    monkeypatch.setattr(s3, "is_storage_configured", lambda: True)
    monkeypatch.setattr(s3, "get_s3_client", lambda profile: profiles.append(profile) or bucket)
    assert checks.object_storage_state() == expected
    assert bucket.asked == [config.S3_DOCUMENTS_BUCKET] and profiles == [s3.FAST_FAIL]


# --------------------------------------------------------------------------- the search index


class _Vectors:
    def __init__(self, error=None):
        self.error, self.calls = error, []

    def get_index(self, **kwargs):
        self.calls.append(("get_index", kwargs))
        if self.error is not None:
            raise self.error

    def list_vector_buckets(self, **kwargs):
        self.calls.append(("list_vector_buckets", kwargs))
        if self.error is not None:
            raise self.error


def _vectors_on(monkeypatch, bucket, client, create_if_missing=True):
    monkeypatch.setattr(config, "S3_VECTORS_ENABLED", True)
    monkeypatch.setattr(config, "S3_VECTORS_BUCKET", bucket)
    monkeypatch.setattr(config, "S3_VECTORS_CREATE_IF_MISSING", create_if_missing)
    monkeypatch.setattr(f"{VECTORS_MODULE}.metadata_client", lambda: client)


@pytest.mark.parametrize("bucket,call", [("shared-health-test", "get_index"),
                                         ("health-test-{workspace_id}", "list_vector_buckets")])
def test_the_search_index_is_a_metadata_call_on_the_kept_client(monkeypatch, bucket, call):
    client = _Vectors()
    _vectors_on(monkeypatch, bucket, client)
    assert checks.search_index_state() == checks.UP
    assert [name for name, _ in client.calls] == [call]


@pytest.mark.parametrize("error,create_if_missing,expected", [
    (_client_error("NotFoundException", 404, "GetIndex"), True, checks.UP),     # core makes it on first use
    (_client_error("NotFoundException", 404, "GetIndex"), False, checks.DOWN),  # core will not make it
    (_client_error("ResourceNotFoundException", 404, "GetIndex"), False, checks.DOWN),
    (_client_error("AccessDeniedException", 403, "GetIndex"), True, checks.NOT_PERMITTED),
    (_client_error("AccessDeniedException", 403, "GetIndex"), False, checks.NOT_PERMITTED),
    (_client_error("ServiceUnavailableException", 503, "GetIndex"), True, checks.DOWN),
    (EndpointConnectionError(endpoint_url="https://" + "vectors-test"), True, checks.DOWN),
    (ReadTimeoutError(endpoint_url="https://" + "vectors-test"), True, checks.DOWN),
])
def test_what_each_search_index_answer_means(monkeypatch, error, create_if_missing, expected):
    _vectors_on(monkeypatch, "shared-health-test", _Vectors(error), create_if_missing)
    assert checks.search_index_state() == expected


def test_a_per_workspace_layout_refused_for_permissions_is_not_permitted(monkeypatch):
    _vectors_on(monkeypatch, "health-test-{workspace_id}",
                _Vectors(_client_error("AccessDeniedException", 403, "ListVectorBuckets")))
    assert checks.search_index_state() == checks.NOT_PERMITTED


def test_the_metadata_client_is_built_once_on_its_own_session_with_the_fast_fail_profile(monkeypatch):
    import boto3.session

    from core.storage.s3 import FAST_FAIL

    vectors = importlib.import_module(VECTORS_MODULE)
    sessions, start = [], threading.Barrier(4)

    class Session:
        def __init__(self):
            sessions.append(self)

        def client(self, service, **kwargs):
            time.sleep(0.05)  # widens the window two builders would race in
            self.service, self.kwargs = service, kwargs
            return self

    monkeypatch.setattr(vectors, "_metadata_clients", {})
    monkeypatch.setattr(boto3.session, "Session", Session)
    clients = []
    threads = [threading.Thread(target=lambda: (start.wait(), clients.append(vectors.metadata_client())))
               for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(WAIT_S)
    assert len(sessions) == 1 and len(clients) == 4 and all(client is sessions[0] for client in clients)
    built = sessions[0]
    assert built.service == "s3vectors" and built.kwargs["region_name"] == config.AWS_REGION
    boto_config = built.kwargs["config"]
    assert boto_config.retries == {"max_attempts": FAST_FAIL.max_attempts}
    assert (boto_config.connect_timeout, boto_config.read_timeout) == (FAST_FAIL.connect_timeout,
                                                                       FAST_FAIL.read_timeout)
    assert vectors.metadata_client() is built


# --------------------------------------------------------------------------- the memory store


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


NO_ANSWER = ResponseHandlingException(OSError("no answer"))
OTHER_ADDRESS = "http://" + "memory-test:6333"


@pytest.mark.parametrize("url,error,expected", [
    (QDRANT_URL_DEFAULT, None, checks.UP),                   # the default address, and a store there
    (QDRANT_URL_DEFAULT, NO_ANSWER, checks.NOT_CONFIGURED),  # the default address, and nothing there
    (OTHER_ADDRESS, NO_ANSWER, checks.DOWN),                 # an address the operator set
    (OTHER_ADDRESS, None, checks.UP),
    (QDRANT_URL_DEFAULT, UnexpectedResponse(401, "Unauthorized", b"", httpx.Headers()), checks.DOWN),
])
def test_the_memory_store_lists_its_collections_and_closes_its_client(monkeypatch, url, error, expected):
    seen = []
    monkeypatch.setattr(config, "QDRANT_URL", url)
    monkeypatch.setattr("core.qdrant.client.QdrantRestClient", _memory_client(seen, error))
    assert checks.memory_store_state() == expected
    assert seen == [("built", config.HEALTH_DEPENDENCY_CHECK_TIMEOUT_SECONDS), ("listed", None), ("closed", None)]


def test_the_memory_store_default_is_taken_from_config_alone():
    import config as config_module

    assert "QDRANT_URL_DEFAULT)" in inspect.getsource(config_module.Config)
    assert inspect.getsource(checks).count(QDRANT_URL_DEFAULT) == 0


# --------------------------------------------------------------------------- the model provider


class _Status(Exception):
    def __init__(self, status_code, text="refused"):
        super().__init__(text)
        self.status_code = status_code


class APIConnectionError(Exception):
    """Named as the OpenAI and Anthropic SDKs name it."""


@pytest.mark.parametrize("error,own_key,noted_down", [
    (APIConnectionError("no route"), False, True),
    (httpx.ConnectError("refused"), False, True),
    (ConnectionRefusedError("refused"), False, True),
    (httpx.ReadTimeout("slow"), False, True),
    (TimeoutError(), False, True),
    (_Status(500), False, True),
    (_Status(529), False, True),
    (RuntimeError("Error code: 503 - {}"), False, True),
    (_Status(401), False, True),   # the install's own key refused
    (_Status(403), False, True),
    (_Status(401), True, False),   # a workspace's own key: not the install's provider
    (_Status(403), True, False),
    (_Status(400), False, False),  # one request refused
    (_Status(404), False, False),
    (_Status(413), False, False),
    (_Status(429), False, False),
    (RuntimeError("Error code: 402 - {}"), False, False),  # out of credit
    (ValueError("the model declined the request"), False, False),
])
def test_only_reach_or_acceptance_failures_note_the_provider_down(monkeypatch, error, own_key, noted_down):
    before = last_call.LastCall(succeeded=True, ended_at=_now())
    monkeypatch.setattr(last_call, "_last", before)
    last_call.note_failure(error, own_key=own_key)
    assert (last_call.last().succeeded is False) if noted_down else (last_call.last() is before)


def test_an_empty_answer_leaves_the_note_as_it_was(monkeypatch):
    monkeypatch.setattr(last_call, "_last", None)
    last_call.note_answer(SimpleNamespace(content="  ", tool_calls=None))
    assert last_call.last() is None
    last_call.note_answer(SimpleNamespace(content=None, tool_calls=[{"name": "a_tool"}]))
    assert last_call.last().succeeded is True


def test_the_model_provider_is_unchecked_before_a_call_and_stamped_when_the_call_ended(monkeypatch):
    monkeypatch.setattr(last_call, "_last", None)
    assert checks.model_provider_state() == health_readiness.UNCHECKED  # never "not configured"
    ended = _now() - timedelta(minutes=1)
    monkeypatch.setattr(last_call, "_last", last_call.LastCall(succeeded=True, ended_at=ended))
    watch = _watch("model_provider")
    asyncio.run(watch.check_now())
    state = SimpleNamespace(dependency_watches=(watch,))
    assert health_dependencies.components(state)["model_provider"] == {"state": checks.UP,
                                                                      "checked_at": ended.isoformat()}
    assert _series("model_provider") == 1.0


def test_a_model_provider_reading_older_than_its_bound_is_unchecked_with_no_series(monkeypatch):
    ended = _now() - timedelta(seconds=health_dependencies.MODEL_PROVIDER_STALE_AFTER_S + 1)
    monkeypatch.setattr(last_call, "_last", last_call.LastCall(succeeded=True, ended_at=ended))
    watch = _watch("model_provider")
    assert watch.stale_after_s == health_dependencies.MODEL_PROVIDER_STALE_AFTER_S
    asyncio.run(watch.check_now())
    assert health_dependencies.components(SimpleNamespace(dependency_watches=(watch,)))["model_provider"] == \
        UNCHECKED_ENTRY
    assert _series("model_provider") is None


# The manager's call methods, driven with a stand-in provider (no network, no database).


class _Answer:
    content, tool_calls = "an answer", None


def _manager(provider, track_usage=lambda *args, **kwargs: None, own_key=False):
    failed = []
    manager = SimpleNamespace(
        provider=provider, config=SimpleNamespace(model="health-test-model"), _tracking_ctx={"is_byok": own_key},
        _credit_gate=lambda: None, _ensure_provider_initialized=lambda: None, _call_budget=lambda: None,
        _track_usage=track_usage, _note_success=lambda budget=None: None, _note_cut=lambda response, budget: None,
        _call_failed=lambda exc, start, budget: failed.append(exc), _raise_unavailable=lambda exc: None)
    return manager, failed


def _provider(error=None):
    async def generate_response(messages, tools=None):
        if error is not None:
            raise error
        return _Answer()

    def generate_response_sync(messages):
        if error is not None:
            raise error
        return _Answer()

    return SimpleNamespace(generate_response=generate_response, generate_response_sync=generate_response_sync)


def _call(manager, run_async):
    from core.llm.manager import LLMManager

    if run_async:
        return asyncio.run(inspect.unwrap(LLMManager.generate_response)(manager, [{"role": "user", "content": "hi"}]))
    return inspect.unwrap(LLMManager.generate_response_sync)(manager, [{"role": "user", "content": "hi"}])


@pytest.mark.parametrize("run_async", [True, False])
def test_the_managers_call_methods_note_an_answer_and_an_unreachable_provider(monkeypatch, run_async):
    monkeypatch.setattr(last_call, "_last", None)
    _call(_manager(_provider())[0], run_async)
    assert last_call.last().succeeded is True
    manager, failed = _manager(_provider(httpx.ConnectError("refused")))
    with pytest.raises(httpx.ConnectError):
        _call(manager, run_async)
    assert last_call.last().succeeded is False and len(failed) == 1  # the manager's own failure path still ran


@pytest.mark.parametrize("run_async", [True, False])
def test_a_refused_request_or_a_workspaces_own_key_leaves_the_note_as_it_was(monkeypatch, run_async):
    before = last_call.LastCall(succeeded=True, ended_at=_now())
    monkeypatch.setattr(last_call, "_last", before)
    for error, own_key in ((_Status(400), False), (_Status(401), True)):
        with pytest.raises(_Status):
            _call(_manager(_provider(error), own_key=own_key)[0], run_async)
    assert last_call.last() is before


@pytest.mark.parametrize("run_async", [True, False])
def test_an_answer_followed_by_a_bookkeeping_error_stays_noted_as_an_answer(monkeypatch, run_async):
    monkeypatch.setattr(last_call, "_last", None)

    def usage_row_fails(*args, **kwargs):
        raise RuntimeError("the usage row could not be written")

    with pytest.raises(RuntimeError):
        _call(_manager(_provider(), track_usage=usage_row_fails)[0], run_async)
    assert last_call.last().succeeded is True


# --------------------------------------------------------------------------- a check the role may not make


def test_a_check_the_role_may_not_make_is_unchecked_with_no_series_and_logged_once(monkeypatch, caplog):
    monkeypatch.setattr(checks, "object_storage_state", lambda: checks.NOT_PERMITTED)
    watch = _watch("object_storage")
    set_dependency_up("object_storage", True)

    async def scenario():
        for _ in range(3):
            await watch.check_now()

    with caplog.at_level("INFO", logger="core.health_watch"):
        asyncio.run(scenario())
    lines = [record.getMessage() for record in caplog.records if record.name == "core.health_watch"]
    assert lines == [f"Health: object_storage is {checks.NOT_PERMITTED}: {checks.PERMISSIONS['object_storage']}"]
    assert "s3:ListBucket" in lines[0]
    assert health_dependencies.components(SimpleNamespace(dependency_watches=(watch,)))["object_storage"] == \
        UNCHECKED_ENTRY
    assert _series("object_storage") is None


def test_a_dependency_down_is_logged_when_it_changes_and_when_it_recovers_never_every_tick(monkeypatch, caplog):
    answers = iter([checks.DOWN, checks.DOWN, checks.DOWN, checks.UP, checks.UP])
    monkeypatch.setattr(checks, "redis_state", lambda: next(answers))
    watch = _watch("redis")

    async def scenario():
        for _ in range(5):
            await watch.check_now()

    with caplog.at_level("INFO", logger="core.health_watch"):
        asyncio.run(scenario())
    lines = [(record.levelname, record.getMessage()) for record in caplog.records if record.name == "core.health_watch"]
    assert lines == [("WARNING", "Health: redis is down"), ("INFO", "Health: redis is up")]


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
        await _until(lambda: all(watch._running is not None for watch in state.dependency_watches))
        started = time.monotonic()
        answered = _health(monkeypatch, state)
        took = time.monotonic() - started
        release.set()  # every thread ends, and its late answer lands while the loop still runs
        running = [watch._running for watch in (*state.dependency_watches, state.database_watch)]
        await _until(lambda: all(future.done() for future in running))
        await health_dependencies.stop(state)
        return answered, took

    try:
        (status, body), took = asyncio.run(scenario())
    finally:
        release.set()
    assert status == 200 and took < AT_ONCE_S
    for name in NAMES:
        assert body["components"][name] == UNCHECKED_ENTRY, name


def test_a_hung_checks_reading_goes_stale_and_takes_its_series_with_it(monkeypatch):
    release = threading.Event()
    monkeypatch.setattr(config, "HEALTH_DEPENDENCY_CHECK_TIMEOUT_SECONDS", SHORT_LIMIT_S)
    monkeypatch.setattr(checks, "redis_state", lambda: release.wait(WAIT_S) and checks.UP)
    watch = _watch("redis")
    state = SimpleNamespace(dependency_watches=(watch,))

    async def scenario():
        await watch.check_now()  # no answer within the limit: down, and its thread still runs
        assert (health_dependencies.components(state)["redis"]["state"], _series("redis")) == (checks.DOWN, 0.0)
        later = _now() + timedelta(seconds=watch.stale_after_s + 1)
        monkeypatch.setattr("core.health_watch._now", lambda: later)
        await watch.check_now()  # the thread still runs: no new reading, and this one is stale now
        found = health_dependencies.components(state)["redis"], _series("redis")
        release.set()
        await _until(watch._running.done)  # its late answer lands while the loop still runs
        return found

    try:
        entry, series = asyncio.run(scenario())
    finally:
        release.set()
    assert entry == UNCHECKED_ENTRY and series is None


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


def test_no_entry_holds_an_address_a_credential_error_text_or_a_state_outside_the_set(monkeypatch):
    secret = uuid.uuid4().hex
    address = f"https://{uuid.uuid4().hex}.example:6379"

    def fails():
        raise RuntimeError(f"cannot reach {address} with {secret}")

    returned = {"redis": fails, "object_storage": lambda: f"up at {address}",
                "search_index": lambda: f"down: {secret}", "memory_store": lambda: "sideways",
                "model_provider": lambda: Reading(f"up with {secret}", _now())}
    for name, attr in health_dependencies.DEPENDENCIES:
        monkeypatch.setattr(checks, attr, returned[name])
    monkeypatch.setattr(config, "HEALTH_DEPENDENCY_CHECK_TIMEOUT_SECONDS", WAIT_S)
    watches = health_dependencies.dependency_watches()

    async def scenario():
        for watch in watches:
            await watch.check_now()

    asyncio.run(scenario())
    status, body = _health(monkeypatch, SimpleNamespace(dependency_watches=watches))
    entries = {name: body["components"][name] for name in NAMES}
    text = json.dumps(entries)
    assert secret not in text and address not in text and "cannot reach" not in text and "sideways" not in text
    for entry in entries.values():
        assert set(entry) == {"state", "checked_at"} and entry["state"] in STATES
    assert entries["redis"]["state"] == checks.DOWN  # it raised: down, as any check that raises
    for name in ("object_storage", "search_index", "memory_store", "model_provider"):
        assert entries[name] == UNCHECKED_ENTRY and _series(name) is None, name


def test_before_any_check_every_dependency_is_listed_unchecked():
    assert health_dependencies.components(SimpleNamespace()) == {name: UNCHECKED_ENTRY for name in NAMES}


# --------------------------------------------------------------------------- start-up never fails on them


def test_a_watch_that_cannot_be_built_is_left_out_and_the_others_run(monkeypatch, caplog):
    real = health_dependencies._watch

    def build(name, attr):
        if name == "redis":
            raise RuntimeError("cannot build")
        return real(name, attr)

    monkeypatch.setattr(health_dependencies, "_watch", build)
    with caplog.at_level("ERROR", logger="core.health_dependencies"):
        watches = health_dependencies.dependency_watches()
    assert [watch.name for watch in watches] == [name for name in NAMES if name != "redis"]
    assert health_dependencies.components(SimpleNamespace(dependency_watches=watches))["redis"] == UNCHECKED_ENTRY
    assert any("redis check could not be set up" in record.getMessage() for record in caplog.records)


def test_dependency_checks_that_cannot_start_leave_readiness_running_and_the_entries_unchecked(monkeypatch, caplog):
    def cannot_start():
        raise RuntimeError("cannot start")

    monkeypatch.setattr(health_dependencies, "dependency_watches", cannot_start)
    monkeypatch.setattr(health_readiness, "_database_status", lambda: health_probe.HEALTHY)
    state = SimpleNamespace()

    async def scenario():
        with caplog.at_level("ERROR", logger="core.health_dependencies"):
            health_dependencies.start(state)
        await state.database_watch.first_reading()
        answered = health_readiness.answer(state)
        await health_dependencies.stop(state)
        return answered

    state.ready = True
    status_code, _ = asyncio.run(scenario())
    assert status_code == health_probe.HTTP_OK  # readiness kept its watch
    assert health_dependencies.components(state) == {name: UNCHECKED_ENTRY for name in NAMES}
    assert any("could not start" in record.getMessage() for record in caplog.records)


# --------------------------------------------------------------------------- /metrics


def test_metrics_carry_one_series_per_dependency_and_none_when_not_configured():
    health_dependencies.record_gauge("redis", Reading(checks.UP, _now()))
    health_dependencies.record_gauge("memory_store", Reading(checks.DOWN, _now()))
    assert (_series("redis"), _series("memory_store")) == (1.0, 0.0)
    health_dependencies.record_gauge("redis", Reading(checks.NOT_CONFIGURED, _now()))
    health_dependencies.record_gauge("memory_store", None)
    assert (_series("redis"), _series("memory_store")) == (None, None)


def test_each_watch_sets_its_series_as_it_reads_and_the_database_has_one_too(monkeypatch):
    monkeypatch.setattr(checks, "redis_state", lambda: checks.UP)
    monkeypatch.setattr(health_readiness, "_database_status", lambda: health_probe.DATABASE_BUSY)
    redis = _watch("redis")

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
