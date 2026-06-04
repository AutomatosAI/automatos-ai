"""Tests for ActionSemanticIndex (PRD-138 US-003).

Pure unit tests — no Redis, no OpenRouter. EmbeddingManager, CacheService and
ActionRegistry are all replaced with deterministic fakes so similarity
ordering is fully predictable.
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock

import pytest

# Load action_semantic_index directly without importing the parent package
# (avoids pulling in the live platform_actions registrar / DB code).
_THIS = Path(__file__).resolve()
_DISCOVERY = _THIS.parents[1] / "modules" / "tools" / "discovery"

# Pre-load action_registry under the same name the index expects.
_ar_spec = importlib.util.spec_from_file_location(
    "action_registry_under_test", _DISCOVERY / "action_registry.py"
)
action_registry_mod = importlib.util.module_from_spec(_ar_spec)
sys.modules["action_registry_under_test"] = action_registry_mod
_ar_spec.loader.exec_module(action_registry_mod)
ActionDefinition = action_registry_mod.ActionDefinition
ActionRegistry = action_registry_mod.ActionRegistry

# Stub the dependencies the index tries to import lazily so __init__ does not
# touch real Redis / DB-backed embedding settings.
_fake_em_module = type(sys)("core.llm")
_fake_cache_module = type(sys)("core.cache.service")


class _FakeEmbeddingManager:
    """Deterministic embedding provider.

    Maps known keywords to one-hot-ish vectors so cosine similarity is
    predictable. Anything else gets a uniform vector.
    """

    DIM = 4

    def __init__(self) -> None:
        self.provider = MagicMock()
        self.provider.config = MagicMock()
        self.provider.config.model = "fake-model"
        self.batch_calls: List[List[str]] = []

    def get_provider_info(self) -> dict:
        return {"provider": "fake", "model": "fake-model", "dimension": self.DIM, "status": "active"}

    def get_dimension(self) -> int:
        return self.DIM

    @staticmethod
    def _vec(text: str) -> List[float]:
        text_l = text.lower()
        # axis 0=agents, 1=missions, 2=admin, 3=other
        if "agent" in text_l:
            return [1.0, 0.0, 0.0, 0.0]
        if "mission" in text_l:
            return [0.0, 1.0, 0.0, 0.0]
        if "admin" in text_l:
            return [0.0, 0.0, 1.0, 0.0]
        return [0.25, 0.25, 0.25, 0.25]

    async def generate_embedding(self, text: str) -> List[float]:
        return self._vec(text)

    async def generate_embeddings_batch(self, texts: List[str], max_concurrent: int = 5) -> List[List[float]]:
        self.batch_calls.append(list(texts))
        return [self._vec(t) for t in texts]


class _FakeCache:
    """In-memory stand-in for CacheService keyed by (model_key, text)."""

    def __init__(self) -> None:
        self.store: Dict[str, Dict[str, List[float]]] = {}
        self.get_calls: List[tuple] = []
        self.set_calls: List[tuple] = []

    def get_embeddings_batch(self, texts: List[str], model: str = "default") -> Dict[str, Optional[List[float]]]:
        self.get_calls.append((model, list(texts)))
        bucket = self.store.get(model, {})
        return {t: bucket.get(t) for t in texts}

    def set_embeddings_batch(self, embeddings: Dict[str, List[float]], model: str = "default") -> None:
        self.set_calls.append((model, dict(embeddings)))
        self.store.setdefault(model, {}).update(embeddings)


# Wire fake modules so the index's lazy imports resolve to fakes.
_fake_em = _FakeEmbeddingManager()
_fake_cache = _FakeCache()
_fake_em_module.create_embedding_manager = lambda: _fake_em  # type: ignore[attr-defined]
_fake_cache_module.get_cache_service = lambda: _fake_cache  # type: ignore[attr-defined]

# ActionSemanticIndex.__init__ imports core.cache.service / core.llm LAZILY, so
# the fakes must be live while THIS module's tests run. Installing them at import
# time, however, leaks a *pathless* fake ``core`` into the collection of sibling
# test modules — breaking every ``core.*`` import they make. Install in
# setup_module and restore in teardown_module so the fakes stay scoped to this
# file's test phase and never touch collection. (PRD-142 W2-S2b.)
_CORE_FAKE_KEYS = ("core", "core.llm", "core.cache", "core.cache.service")
_saved_core_modules: Dict[str, object] = {}


def setup_module(module):
    for _k in _CORE_FAKE_KEYS:
        _saved_core_modules[_k] = sys.modules.get(_k)
    sys.modules.setdefault("core", type(sys)("core"))
    sys.modules["core.llm"] = _fake_em_module
    sys.modules["core.cache"] = type(sys)("core.cache")
    sys.modules["core.cache.service"] = _fake_cache_module


def teardown_module(module):
    for _k, _v in _saved_core_modules.items():
        if _v is None:
            sys.modules.pop(_k, None)
        else:
            sys.modules[_k] = _v

# Patch the import the index uses for ActionDefinition + get_action_registry
# so we control the registry per-test. We rebuild a fresh registry per test.
_test_registry = ActionRegistry()
_test_registry._initialized = True


def _set_registry(actions: List[ActionDefinition]) -> ActionRegistry:
    reg = ActionRegistry()
    reg._initialized = True
    for a in actions:
        reg.register(a)
    return reg


# Now load the module under test. Patch its `.action_registry` import path to
# our pre-loaded module by giving it the same submodule name.
asi_spec = importlib.util.spec_from_file_location(
    "automatos.action_semantic_index_under_test",
    _DISCOVERY / "action_semantic_index.py",
)
# Inject a fake parent package with .action_registry already populated.
parent_pkg = type(sys)("automatos")
parent_pkg.__path__ = []  # mark as package
sub_pkg = type(sys)("automatos.action_registry")
sub_pkg.ActionDefinition = ActionDefinition
sub_pkg.get_action_registry = lambda: _test_registry
# The index does `from .action_registry import ...` — emulate that by giving
# the spec a package context.
asi_spec = importlib.util.spec_from_file_location(
    "automatos.tools_discovery.action_semantic_index",
    _DISCOVERY / "action_semantic_index.py",
    submodule_search_locations=[str(_DISCOVERY)],
)
# Build an artificial package layout so relative `.action_registry` resolves.
pkg_name = "automatos_tools_discovery_pkg"
pkg = type(sys)(pkg_name)
pkg.__path__ = [str(_DISCOVERY)]
sys.modules[pkg_name] = pkg
sys.modules[f"{pkg_name}.action_registry"] = action_registry_mod
asi_spec = importlib.util.spec_from_file_location(
    f"{pkg_name}.action_semantic_index",
    _DISCOVERY / "action_semantic_index.py",
)
action_semantic_index_mod = importlib.util.module_from_spec(asi_spec)
action_semantic_index_mod.__package__ = pkg_name
sys.modules[f"{pkg_name}.action_semantic_index"] = action_semantic_index_mod
asi_spec.loader.exec_module(action_semantic_index_mod)
ActionSemanticIndex = action_semantic_index_mod.ActionSemanticIndex
get_action_semantic_index = action_semantic_index_mod.get_action_semantic_index


# ---- Helpers ----

def _make(name: str, *, category: str = "agents", description: str = "", admin_only: bool = False, promoted: bool = False, tags=None, examples=None) -> ActionDefinition:
    return ActionDefinition(
        name=name,
        description=description or f"{name} description",
        category=category,
        parameters={"type": "object", "properties": {}, "required": []},
        admin_only=admin_only,
        promoted=promoted,
        tags=list(tags or []),
        examples=list(examples or []),
    )


def _make_index(actions: List[ActionDefinition]) -> ActionSemanticIndex:
    """Build an ActionSemanticIndex pinned to a fresh registry of `actions`."""
    idx = ActionSemanticIndex()
    idx._registry = _set_registry(actions)
    # Reset shared fakes so calls counts don't leak between tests.
    _fake_em.batch_calls.clear()
    _fake_cache.store.clear()
    _fake_cache.get_calls.clear()
    _fake_cache.set_calls.clear()
    return idx


def _run(coro):
    return asyncio.run(coro)


# ---- AC 12: rank_actions returns at most top_k, ordered desc ----

def test_rank_actions_returns_top_k_descending():
    actions = [_make(f"platform_agent_{i}", category="agents") for i in range(10)]
    idx = _make_index(actions)
    results = _run(idx.rank_actions(query="agent management", top_k=5))
    assert len(results) <= 5
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
    # All returned names should be from our registry
    names = {a.name for a in actions}
    assert all(n in names for n in (n for n, _ in results))


# ---- AC 13: cache key format ----

def test_cache_key_format_uses_provider_info():
    idx = _make_index([_make("platform_a", category="agents")])
    key = idx._cache_model_key()
    assert key == "fake:fake-model:4"
    assert "default" not in key


# ---- AC 14: exclude_admin hides admin-tagged actions ----

def test_exclude_admin_hides_admin_actions():
    actions = [
        _make("platform_normal_agent", category="agents"),
        _make("platform_admin_thing", category="admin", admin_only=True),
    ]
    idx = _make_index(actions)
    results = _run(idx.rank_actions(query="admin", top_k=10, exclude_admin=True))
    names = [n for n, _ in results]
    assert "platform_admin_thing" not in names
    assert "platform_normal_agent" in names


def test_admin_visible_when_not_excluded():
    actions = [
        _make("platform_admin_thing", category="admin", admin_only=True),
    ]
    idx = _make_index(actions)
    results = _run(idx.rank_actions(query="admin", top_k=10, exclude_admin=False))
    names = [n for n, _ in results]
    assert "platform_admin_thing" in names


# ---- AC 15: registry smaller than top_k returns all eligible ----

def test_fewer_actions_than_top_k_returns_all():
    actions = [
        _make("platform_a1", category="agents"),
        _make("platform_a2", category="agents"),
    ]
    idx = _make_index(actions)
    results = _run(idx.rank_actions(query="agent", top_k=15))
    assert len(results) == 2


# ---- Cache reuse: re-indexing does not re-embed ----

def test_ensure_indexed_reuses_in_memory_embeddings():
    actions = [_make("platform_a1", category="agents"), _make("platform_a2", category="agents")]
    idx = _make_index(actions)

    async def _double():
        await idx.ensure_indexed()
        first = len(_fake_em.batch_calls)
        await idx.ensure_indexed()
        return first, len(_fake_em.batch_calls)

    first, second = _run(_double())
    assert first == second  # second pass did no new embedding work


def test_promoted_excluded_by_default():
    actions = [
        _make("platform_normal", category="agents"),
        _make("platform_promoted_one", category="agents", promoted=True),
    ]
    idx = _make_index(actions)
    results = _run(idx.rank_actions(query="platform"))
    names = [n for n, _ in results]
    assert "platform_promoted_one" not in names
    assert "platform_normal" in names


def test_build_embedding_text_format():
    action = _make(
        "platform_x",
        category="agents",
        description="Do the thing",
        tags=["t1", "t2"],
        examples=["ex1", "ex2"],
    )
    text = ActionSemanticIndex._build_embedding_text(action)
    assert text == "platform_x: Do the thing | Tags: t1, t2 | Examples: ex1; ex2 | Category: agents"


def test_factory_returns_singleton():
    a = get_action_semantic_index()
    b = get_action_semantic_index()
    assert a is b


def test_cache_round_trip_writes_and_reads():
    actions = [_make("platform_a1", category="agents")]
    idx = _make_index(actions)
    _run(idx.ensure_indexed())
    # First build wrote to cache
    assert _fake_cache.set_calls, "expected cache set on first build"
    # New index instance, same fake cache → no embedding generation needed
    idx2 = ActionSemanticIndex()
    idx2._registry = idx._registry
    _fake_em.batch_calls.clear()
    _run(idx2.ensure_indexed())
    assert _fake_em.batch_calls == [], "cache hit should skip generate_embeddings_batch"
