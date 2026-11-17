"""One Anthropic SDK client per key (voice latency D-4).

A provider object is built per ``LLMManager``, and a spoken turn's manager is new every
turn, so every turn built an SDK client and an HTTP pool (and, on a saved key, ran the
key's fingerprint) before its first model call. With ``LLM_CLIENT_REUSE`` on, providers
on the same key share the client kept in ``core/llm/clients/shared_clients.py``.

What is pinned here:

* switch off (the default): a build per provider, exactly as before, and an empty store;
* switch on: one client per (SDK class, key, base URL, keep-alive), the workspace header
  decided once per key when the client is built;
* the store NEVER closes a client: one handed out before ``forget_clients()``, before its
  time ran out or before the size cap pushed it out still completes a call;
* every call drops every entry past its time, and the store is bounded;
* a build runs outside the lock: threads asking for one key all get one object, and a
  build that is stuck for one key does not hold up another key;
* ``LLM_CLIENT_KEEPALIVE_SECONDS`` builds the client with the SDK's pool numbers and that
  idle time, and is read only while reuse is on.

Pure tests: the Anthropic SDK and the workspace lookup are stubbed. No network, no database.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

import core.llm.anthropic_workspace as aw
import core.llm.clients.anthropic_client as client_module
import core.llm.clients.shared_clients as sc
from config import config
from core.llm.clients.base import LLMConfig, LLMProvider

WORKSPACE = "wrkspc_01JwQvzr7rXLA5AGx3HKfFUJ"
ORG_KEY = "sk-test-org-level-key"
SCOPED_KEY = "sk-test-scoped-key"
OTHER_KEY = "sk-test-other-key"
ANSWER = "answered"


class _FakeClient:
    """Stands in for ``anthropic.Anthropic``: records how it was built and whether it was closed."""

    built: list = []

    def __init__(self, **kwargs):
        type(self).built.append(kwargs)
        self.kwargs = kwargs
        self.closed = False
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **_kwargs):
        if self.closed:
            raise RuntimeError("the client was closed")
        return ANSWER

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    """Reuse on, the knob at 0, no env key, and no client left over from another test."""
    sc.forget_clients()
    monkeypatch.setattr(config, "LLM_CLIENT_REUSE", True)
    monkeypatch.setattr(config, "LLM_CLIENT_KEEPALIVE_SECONDS", 0.0)
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", None)
    monkeypatch.setattr(config, "ANTHROPIC_WORKSPACE_ID", None)
    yield
    sc.forget_clients()


@pytest.fixture
def sdk(monkeypatch):
    """A fake SDK of this test's own (its class is part of the store's key)."""
    fake = type("_Sdk", (_FakeClient,), {"built": []})
    monkeypatch.setattr(client_module, "anthropic", SimpleNamespace(
        Anthropic=fake, DefaultHttpxClient=lambda **kwargs: SimpleNamespace(**kwargs)))
    return fake


@pytest.fixture
def lookups(monkeypatch):
    """The keys ``workspace_for_key`` was asked about; only ORG_KEY has a saved workspace."""
    asked = []

    def _workspace(key):
        asked.append(key)
        return WORKSPACE if key == ORG_KEY else None

    monkeypatch.setattr(client_module, "workspace_for_key", _workspace)
    return asked


@pytest.fixture
def clock(monkeypatch):
    """The store's clock, moved by hand. Only shared_clients reads it."""
    now = SimpleNamespace(value=1000.0)
    monkeypatch.setattr(sc, "time", SimpleNamespace(monotonic=lambda: now.value))
    return now


def _provider(api_key=SCOPED_KEY, base_url=None):
    return client_module.AnthropicProvider(
        LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-opus-5", api_key=api_key, base_url=base_url))


def _kept_clients():
    return [kept.client for kept in sc._clients.values()]


# ── switch off: as before ──────────────────────────────────────────────────


def test_the_switch_is_off_by_default():
    # The class holds what the environment gave at import; the fixture patches the instance.
    assert type(config).LLM_CLIENT_REUSE is False
    assert type(config).LLM_CLIENT_KEEPALIVE_SECONDS == 0


def test_with_the_switch_off_every_provider_builds_its_own_client(monkeypatch, sdk, lookups):
    monkeypatch.setattr(config, "LLM_CLIENT_REUSE", False)

    a, b = _provider(ORG_KEY), _provider(ORG_KEY)

    assert a.client is not b.client
    assert sdk.built == [{"api_key": ORG_KEY, "default_headers": {"anthropic-workspace-id": WORKSPACE}}] * 2
    assert lookups == [ORG_KEY, ORG_KEY]
    assert sc._clients == {}


def test_with_the_switch_off_the_keep_alive_knob_is_not_read(monkeypatch, sdk, lookups):
    monkeypatch.setattr(config, "LLM_CLIENT_REUSE", False)
    monkeypatch.setattr(config, "LLM_CLIENT_KEEPALIVE_SECONDS", 60.0)

    _provider()

    assert sdk.built == [{"api_key": SCOPED_KEY}]
    assert sc._clients == {}


# ── switch on: one client per key ──────────────────────────────────────────


def test_two_providers_on_one_key_share_one_client(sdk, lookups):
    a, b = _provider(ORG_KEY), _provider(ORG_KEY)

    assert a.client is b.client
    assert len(sdk.built) == 1
    assert lookups == [ORG_KEY]
    assert _kept_clients() == [a.client]


def test_another_key_or_another_base_url_gets_another_client(sdk, lookups):
    first = _provider(SCOPED_KEY)
    other_key = _provider(OTHER_KEY)
    other_url = _provider(SCOPED_KEY, base_url="https://models.example")

    assert len({id(first.client), id(other_key.client), id(other_url.client)}) == 3
    assert sdk.built == [
        {"api_key": SCOPED_KEY},
        {"api_key": OTHER_KEY},
        {"api_key": SCOPED_KEY, "base_url": "https://models.example"},
    ]
    assert _provider(SCOPED_KEY, base_url="https://models.example").client is other_url.client
    assert len(sdk.built) == 3


def test_the_workspace_header_is_decided_per_key(sdk, lookups):
    with_workspace = _provider(ORG_KEY)
    scoped = _provider(SCOPED_KEY)

    assert with_workspace.client.kwargs["default_headers"] == {"anthropic-workspace-id": WORKSPACE}
    assert "default_headers" not in scoped.client.kwargs
    assert _provider(ORG_KEY).client is with_workspace.client
    assert _provider(SCOPED_KEY).client is scoped.client


def test_another_sdk_class_never_gets_a_client_the_first_one_built(monkeypatch, sdk, lookups):
    first = _provider()
    other_sdk = type("_OtherSdk", (_FakeClient,), {"built": []})
    monkeypatch.setattr(client_module, "anthropic", SimpleNamespace(Anthropic=other_sdk))

    second = _provider()

    assert isinstance(second.client, other_sdk)
    assert second.client is not first.client


def test_no_key_means_no_client_and_nothing_stored(sdk, lookups):
    provider = _provider(api_key=None)

    assert provider.client is None
    assert sdk.built == [] and lookups == []
    assert sc._clients == {}


# ── when a kept client is dropped ──────────────────────────────────────────


def test_clearing_the_workspace_cache_forgets_the_clients(sdk, lookups):
    before = _provider()

    aw.clear_cache()

    assert sc._clients == {}
    after = _provider()
    assert after.client is not before.client
    assert len(sdk.built) == 2


def test_a_client_is_kept_for_the_cache_window_and_no_longer(sdk, lookups, clock):
    first = _provider(SCOPED_KEY)
    _provider(OTHER_KEY)

    clock.value += aw.CACHE_SECONDS - 1
    assert _provider(SCOPED_KEY).client is first.client
    assert len(sc._clients) == 2

    clock.value += 2
    again = _provider(SCOPED_KEY)

    assert again.client is not first.client
    assert len(sdk.built) == 3
    # The other key's entry was past its time too: any call drops it.
    assert _kept_clients() == [again.client]


def test_the_store_is_bounded_and_drops_the_entry_nearest_its_end(clock):
    for i in range(sc.MAX_CLIENTS + 1):
        clock.value += 1
        sc.shared_client(object, "sdk", f"key-{i}", keep_seconds=aw.CACHE_SECONDS)

    assert len(sc._clients) == sc.MAX_CLIENTS
    assert ("sdk", "key-0") not in sc._clients
    assert ("sdk", "key-1") in sc._clients and ("sdk", f"key-{sc.MAX_CLIENTS}") in sc._clients


def _drop_by_forgetting(_clock):
    sc.forget_clients()


def _drop_by_time(clock):
    clock.value += aw.CACHE_SECONDS + 1
    _provider(OTHER_KEY)


def _drop_by_the_cap(clock):
    for i in range(sc.MAX_CLIENTS):
        clock.value += 1
        _provider(f"sk-test-filler-{i}")


@pytest.mark.parametrize("drop", [_drop_by_forgetting, _drop_by_time, _drop_by_the_cap])
def test_a_client_handed_out_is_never_closed_by_the_store(sdk, lookups, clock, drop):
    provider = _provider(SCOPED_KEY)
    handed_out = provider.client

    drop(clock)

    assert all(kept is not handed_out for kept in _kept_clients())
    assert handed_out.closed is False
    assert provider.client.messages.create(model="claude-opus-5", max_tokens=1, messages=[]) == ANSWER
    assert _provider(SCOPED_KEY).client is not handed_out


# ── the keep-alive knob ────────────────────────────────────────────────────


def test_the_keep_alive_knob_builds_the_pool_with_that_idle_time(monkeypatch, sdk, lookups):
    monkeypatch.setattr(config, "LLM_CLIENT_KEEPALIVE_SECONDS", 60.0)

    provider = _provider()

    limits = provider.client.kwargs["http_client"].limits
    assert limits.keepalive_expiry == 60
    assert (limits.max_connections, limits.max_keepalive_connections) == (1000, 100)


def test_without_the_knob_the_sdk_builds_its_own_pool(sdk, lookups):
    assert "http_client" not in _provider().client.kwargs


def test_a_changed_knob_gets_a_client_of_its_own(monkeypatch, sdk, lookups):
    plain = _provider()
    monkeypatch.setattr(config, "LLM_CLIENT_KEEPALIVE_SECONDS", 60.0)

    kept_alive = _provider()

    assert kept_alive.client is not plain.client
    assert "http_client" in kept_alive.client.kwargs


# ── threads ────────────────────────────────────────────────────────────────


def test_ten_threads_asking_for_one_key_get_one_client(monkeypatch, sdk):
    threads = 10
    together = threading.Barrier(threads)

    def _slow_lookup(_key):
        time.sleep(0.05)
        return None

    monkeypatch.setattr(client_module, "workspace_for_key", _slow_lookup)

    def _ask(_n):
        together.wait(timeout=5)
        return _provider(SCOPED_KEY).client

    with ThreadPoolExecutor(max_workers=threads) as pool:
        clients = list(pool.map(_ask, range(threads)))

    assert len({id(client) for client in clients}) == 1
    assert _kept_clients() == [clients[0]]
    assert 1 <= len(sdk.built) <= threads  # a double build is allowed; one client is kept


def test_a_build_stuck_on_one_key_does_not_hold_up_another_key(monkeypatch, sdk):
    started, release = threading.Event(), threading.Event()
    got = {}

    def _lookup(key):
        if key == ORG_KEY:
            started.set()
            release.wait(timeout=5)
        return None

    monkeypatch.setattr(client_module, "workspace_for_key", _lookup)
    stuck = threading.Thread(target=lambda: got.update(stuck=_provider(ORG_KEY).client))
    free = threading.Thread(target=lambda: got.update(free=_provider(OTHER_KEY).client))

    stuck.start()
    try:
        assert started.wait(timeout=5)
        free.start()
        free.join(timeout=2)
        finished_while_stuck = not free.is_alive() and "stuck" not in got
    finally:
        release.set()
        stuck.join(timeout=5)
        if free.ident is not None:
            free.join(timeout=5)

    assert finished_while_stuck
    assert got["free"] is not got["stuck"]
    assert len(sc._clients) == 2
