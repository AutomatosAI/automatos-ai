"""F105 (wave 2): a model's context window is read from the registry once, then kept.

``core.context_guard.get_context_window`` read the model registry on the caller's session,
on the event loop, for every chat turn: for the context budget (modules/context/service.py)
and again in ``ContextGuard.check_and_compact`` before each model call of the turn. A
window is global and changes only when the registry is synced or an admin edits a model.
The registry's answer is now kept per model for ``CONTEXT_WINDOW_CACHE_TTL_SECONDS`` in a
bounded LRU (``CONTEXT_WINDOW_CACHE_SIZE``) shared by every thread; a failed read is never
kept. The answers are the registry's, as before.

DB-free: a fake registry counts its reads.
"""
from __future__ import annotations

import threading

import pytest

from core import context_guard as cg

DEFAULT = 128_000


class _Lookup:
    def __init__(self, registry, model_id):
        self._registry, self._model_id = registry, model_id

    def first(self):
        self._registry.reads.append(self._model_id)
        if self._registry.fails:
            raise RuntimeError("the pool is exhausted")
        window = self._registry.windows.get(self._model_id)
        return (window,) if window else None


class _Registry:
    """``db.query(LLMModel.context_window).filter(LLMModel.model_id == x).first()``."""

    def __init__(self, **windows):
        self.windows, self.reads, self.fails = windows, [], False

    def query(self, column):
        return self

    def filter(self, condition):
        return _Lookup(self, condition.right.value)


@pytest.fixture(autouse=True)
def empty_cache(monkeypatch):
    monkeypatch.setattr(cg, "_context_windows", type(cg._context_windows)())


def test_a_models_window_is_read_once_and_then_kept():
    registry = _Registry(**{"anthropic/claude-haiku-5-5": 200_000})

    windows = [cg.get_context_window("anthropic/claude-haiku-5-5", registry) for _ in range(3)]

    assert windows == [200_000] * 3
    assert registry.reads == ["anthropic/claude-haiku-5-5"]


def test_the_bare_name_fallback_answers_as_before_and_is_kept():
    registry = _Registry(**{"gpt-5.4": 400_000})

    assert cg.get_context_window("openai/gpt-5.4", registry) == 400_000
    assert cg.get_context_window("openai/gpt-5.4", registry) == 400_000
    assert registry.reads == ["openai/gpt-5.4", "gpt-5.4"]


def test_an_unregistered_model_gets_the_default_and_is_read_once():
    registry = _Registry()

    assert [cg.get_context_window("acme/unknown", registry) for _ in range(2)] == [DEFAULT, DEFAULT]
    assert registry.reads == ["acme/unknown", "unknown"]


def test_a_failed_read_gives_the_default_and_is_never_kept():
    registry = _Registry(**{"gpt-4o": 128_000 * 2})
    registry.fails = True

    assert cg.get_context_window("gpt-4o", registry) == DEFAULT
    registry.fails = False
    assert cg.get_context_window("gpt-4o", registry) == 256_000
    assert registry.reads == ["gpt-4o", "gpt-4o"]


def test_a_kept_window_is_read_again_once_it_is_stale(monkeypatch):
    registry = _Registry(**{"gpt-4o": 256_000})
    now = [1_000.0]
    monkeypatch.setattr(cg.time, "monotonic", lambda: now[0])

    cg.get_context_window("gpt-4o", registry)
    now[0] += cg.CONTEXT_WINDOW_CACHE_TTL_SECONDS - 1
    cg.get_context_window("gpt-4o", registry)
    assert registry.reads == ["gpt-4o"]

    registry.windows["gpt-4o"] = 1_000_000  # the registry was synced
    now[0] += 2
    assert cg.get_context_window("gpt-4o", registry) == 1_000_000
    assert registry.reads == ["gpt-4o", "gpt-4o"]


def test_the_cache_is_bounded_and_drops_the_least_recently_used(monkeypatch):
    monkeypatch.setattr(cg, "CONTEXT_WINDOW_CACHE_SIZE", 2)
    registry = _Registry(a=1_000, b=2_000, c=3_000)

    for name in ("a", "b", "a", "c"):
        cg.get_context_window(name, registry)

    assert list(cg._context_windows) == ["a", "c"]
    cg.get_context_window("b", registry)
    assert registry.reads == ["a", "b", "c", "b"]


def test_no_session_or_no_name_gives_the_default_and_keeps_nothing():
    assert cg.get_context_window("gpt-4o", None) == DEFAULT
    assert cg.get_context_window("", _Registry(gpt_4o=1)) == DEFAULT
    assert len(cg._context_windows) == 0


def test_threads_reading_one_model_agree():
    registry = _Registry(**{"gpt-4o": 256_000})
    answers = []

    def read():
        answers.append(cg.get_context_window("gpt-4o", registry))

    threads = [threading.Thread(target=read) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert answers == [256_000] * 8 and 1 <= len(registry.reads) <= 8
    assert cg.get_context_window("gpt-4o", registry) == 256_000
