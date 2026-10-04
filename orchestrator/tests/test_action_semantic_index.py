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


def test_promoted_action_boosted_over_equal_cosine_unpromoted():
    """PRD-232 US-014 (promotion-as-prior): a promoted action outranks an
    EQUAL-cosine unpromoted one. Both actions embed to the same 'agent' vector
    (identical cosine to the query), so only the promotion boost can separate
    them — the promoted one must come first."""
    actions = [
        _make("agent_plain", description="agent thing", category="agents"),
        _make("agent_promoted", description="agent thing", category="agents", promoted=True),
    ]
    idx = _make_index(actions)
    # exclude_promoted=False so the promoted action is eligible in the surface.
    results = _run(idx.rank_actions(query="agent", exclude_promoted=False, top_k=5))
    names = [n for n, _ in results]
    assert names[:2] == ["agent_promoted", "agent_plain"], (
        f"promoted action must outrank the equal-cosine unpromoted one; got {results}"
    )
    scores = {n: s for n, s in results}
    # The boost is exactly the configured prior on top of the identical cosine.
    from config import config
    assert scores["agent_promoted"] > scores["agent_plain"]
    assert scores["agent_promoted"] - scores["agent_plain"] == pytest.approx(
        config.TOOL_ROUTING_PROMOTION_BOOST, abs=1e-9
    )


def test_build_embedding_text_format():
    # PRD-232 US-006: the embedded text now also carries parameter enum values
    # (Options:, from the action's own schema) and the seeded utterance corpus
    # (Utterances:). "platform_x" is a test-only name with no corpus entry, so its
    # utterance section is empty here; the populated path is covered by
    # tests/test_prd232_us006_corpus_embeddings.py.
    action = _make(
        "platform_x",
        category="agents",
        description="Do the thing",
        tags=["t1", "t2"],
        examples=["ex1", "ex2"],
    )
    action.parameters["properties"]["mode"] = {"type": "string", "enum": ["fast", "slow"]}
    text = ActionSemanticIndex._build_embedding_text(action)
    assert text == (
        "platform_x: Do the thing | Tags: t1, t2 | Examples: ex1; ex2 | "
        "Category: agents | Options: fast, slow | Utterances: "
    )


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


# ---- Query-embed cache + timeout (never block the turn on a slow upstream) ----

_MODEL_KEY = "fake:fake-model:4"


def test_rank_query_embed_cache_hit_skips_live_embed():
    """A Redis-cached query vector is used verbatim — no live embed call."""
    actions = [_make("platform_agent_thing", category="agents")]
    idx = _make_index(actions)
    _fake_cache.set_embeddings_batch({"agent stuff": [1.0, 0.0, 0.0, 0.0]}, model=_MODEL_KEY)

    orig = _fake_em.generate_embedding

    async def _must_not_embed(text):
        raise AssertionError("live embed must not run on a query-cache hit")

    _fake_em.generate_embedding = _must_not_embed
    try:
        results = _run(idx.rank_actions(query="agent stuff", top_k=5))
    finally:
        _fake_em.generate_embedding = orig
    assert results, "cached query vector should still produce a ranking"
    assert results[0][0] == "platform_agent_thing"


def test_rank_query_embed_success_writes_cache():
    """A live query embed lands in the cache so the next identical query is free."""
    actions = [_make("platform_agent_thing", category="agents")]
    idx = _make_index(actions)
    results = _run(idx.rank_actions(query="agent stuff", top_k=5))
    assert results
    assert _fake_cache.store.get(_MODEL_KEY, {}).get("agent stuff") is not None


def test_rank_query_embed_timeout_falls_back_and_warms_cache():
    """When the live embed exceeds the budget, rank_actions returns [] fast
    (caller falls back to the full enum) and the embed finishes in the
    background, writing the cache for the next turn."""
    actions = [_make("platform_agent_thing", category="agents")]
    idx = _make_index(actions)

    orig = _fake_em.generate_embedding

    async def _slow_embed(text):
        await asyncio.sleep(0.2)
        return [1.0, 0.0, 0.0, 0.0]

    _fake_em.generate_embedding = _slow_embed
    try:
        async def _scenario():
            results = await idx.rank_actions(
                query="agent stuff", top_k=5, embed_timeout_s=0.05
            )
            assert results == [], "timed-out embed must fall back to []"
            assert _fake_cache.store.get(_MODEL_KEY, {}).get("agent stuff") is None
            # Let the abandoned embed finish inside the same loop.
            await asyncio.sleep(0.3)
            assert _fake_cache.store.get(_MODEL_KEY, {}).get("agent stuff") is not None

        _run(_scenario())
    finally:
        _fake_em.generate_embedding = orig


def test_rank_query_embed_timeout_disabled_with_nonpositive_budget():
    """embed_timeout_s <= 0 disables the bound — the call simply waits."""
    actions = [_make("platform_agent_thing", category="agents")]
    idx = _make_index(actions)

    orig = _fake_em.generate_embedding

    async def _slowish_embed(text):
        await asyncio.sleep(0.05)
        return [1.0, 0.0, 0.0, 0.0]

    _fake_em.generate_embedding = _slowish_embed
    try:
        results = _run(idx.rank_actions(query="agent stuff", top_k=5, embed_timeout_s=0))
    finally:
        _fake_em.generate_embedding = orig
    assert results and results[0][0] == "platform_agent_thing"


# ---- #927: a cold index never holds a turn ----
# The first turn after an embedding key is added (or after an upgrade that rewords
# actions, or an evicted Redis) used to embed the whole catalogue inside the turn —
# 4½ minutes on a slow upstream. The build now waits within the query-embed budget,
# runs once in the background, and fills the cache for the next turn.


def _slow_batch(delay: float, calls: List[List[str]]):
    async def _batch(texts, max_concurrent: int = 5):
        calls.append(list(texts))
        await asyncio.sleep(delay)
        return [_FakeEmbeddingManager._vec(t) for t in texts]

    return _batch


def test_a_cold_index_ranks_nothing_within_the_budget_and_warms_in_the_background():
    actions = [_make("platform_agent_thing", category="agents")]
    idx = _make_index(actions)
    calls: List[List[str]] = []
    orig = _fake_em.generate_embeddings_batch
    _fake_em.generate_embeddings_batch = _slow_batch(0.3, calls)
    try:
        async def _scenario():
            started = asyncio.get_running_loop().time()
            first = await idx.rank_actions(query="agent stuff", top_k=5, embed_timeout_s=0.05)
            waited = asyncio.get_running_loop().time() - started
            assert first == [], "a cold index must rank nothing, not hold the turn"
            assert waited < 0.25, f"the turn waited {waited:.2f}s for the index build"
            await asyncio.sleep(0.4)  # the abandoned build finishes on this loop
            assert "platform_agent_thing" in idx._action_embeddings
            assert any(
                v for v in _fake_cache.store.get(_MODEL_KEY, {}).values()
            ), "the background build must fill the cache"
            second = await idx.rank_actions(query="agent stuff", top_k=5, embed_timeout_s=0.05)
            assert second and second[0][0] == "platform_agent_thing"

        _run(_scenario())
    finally:
        _fake_em.generate_embeddings_batch = orig
    assert len(calls) == 1, "the catalogue is embedded once, not once per turn"


def test_turns_that_arrive_during_a_build_share_it():
    actions = [_make("platform_agent_thing", category="agents"), _make("platform_mission_thing", category="missions")]
    idx = _make_index(actions)
    calls: List[List[str]] = []
    orig = _fake_em.generate_embeddings_batch
    _fake_em.generate_embeddings_batch = _slow_batch(0.2, calls)
    try:
        async def _scenario():
            results = await asyncio.gather(
                idx.rank_actions(query="agent stuff", top_k=5, embed_timeout_s=0.05),
                idx.rank_actions(query="mission stuff", top_k=5, embed_timeout_s=0.05),
                idx.rank_actions(query="anything else", top_k=5, embed_timeout_s=0.05),
            )
            assert results == [[], [], []]
            await asyncio.sleep(0.3)

        _run(_scenario())
    finally:
        _fake_em.generate_embeddings_batch = orig
    assert len(calls) == 1, f"one upstream build expected, got {len(calls)}"


def test_with_the_budget_disabled_a_turn_still_waits_for_the_build():
    actions = [_make("platform_agent_thing", category="agents")]
    idx = _make_index(actions)
    calls: List[List[str]] = []
    orig = _fake_em.generate_embeddings_batch
    _fake_em.generate_embeddings_batch = _slow_batch(0.05, calls)
    try:
        results = _run(idx.rank_actions(query="agent stuff", top_k=5, embed_timeout_s=0))
    finally:
        _fake_em.generate_embeddings_batch = orig
    assert results and results[0][0] == "platform_agent_thing"


def test_a_build_that_fails_while_the_turn_waits_reaches_the_turn():
    actions = [_make("platform_agent_thing", category="agents")]
    idx = _make_index(actions)
    orig = _fake_em.generate_embeddings_batch

    async def _broken(texts, max_concurrent: int = 5):
        raise RuntimeError("upstream said no")

    _fake_em.generate_embeddings_batch = _broken
    try:
        with pytest.raises(RuntimeError, match="upstream said no"):
            _run(idx.rank_actions(query="agent stuff", top_k=5, embed_timeout_s=1.0))
    finally:
        _fake_em.generate_embeddings_batch = orig


def test_a_build_that_fails_after_the_turn_moved_on_is_logged(caplog):
    actions = [_make("platform_agent_thing", category="agents")]
    idx = _make_index(actions)
    orig = _fake_em.generate_embeddings_batch

    async def _slow_then_broken(texts, max_concurrent: int = 5):
        await asyncio.sleep(0.1)
        raise RuntimeError("upstream timed out late")

    _fake_em.generate_embeddings_batch = _slow_then_broken
    try:
        async def _scenario():
            assert await idx.rank_actions(query="agent stuff", top_k=5, embed_timeout_s=0.02) == []
            await asyncio.sleep(0.2)

        with caplog.at_level("WARNING"):
            _run(_scenario())
    finally:
        _fake_em.generate_embeddings_batch = orig
    assert "background index build failed: upstream timed out late" in caplog.text
    assert "never retrieved" not in caplog.text
    assert not getattr(idx, "_index_builds", {}), "a finished build is forgotten, so the next turn retries"


def test_warm_indexes_the_widest_view_ahead_of_the_first_turn():
    actions = [
        _make("platform_agent_thing", category="agents"),
        _make("platform_admin_thing", category="admin", admin_only=True),
        _make("platform_promoted_thing", category="agents", promoted=True),
    ]
    idx = _make_index(actions)
    _run(idx.warm())
    assert set(idx._action_embeddings) == {
        "platform_agent_thing",
        "platform_admin_thing",
        "platform_promoted_thing",
    }


def test_a_slow_build_is_waited_on_once_not_once_per_ranking():
    """A turn ranks several times (narrowing, the shadow surface, the prompt catalog),
    and not all inside one rank scope. Once one waiter has spent the budget on a build,
    later rankings, in that turn or another, return at once until the build ends."""
    actions = [_make("platform_agent_thing", category="agents")]
    idx = _make_index(actions)
    calls: List[List[str]] = []
    orig = _fake_em.generate_embeddings_batch
    _fake_em.generate_embeddings_batch = _slow_batch(0.6, calls)
    try:
        async def _scenario():
            loop = asyncio.get_running_loop()
            started = loop.time()
            for query in ("agent stuff", "mission stuff", "anything else"):
                assert await idx.rank_actions(query=query, top_k=5, embed_timeout_s=0.1) == []
            waited = loop.time() - started
            assert waited < 0.25, f"three rankings waited {waited:.2f}s; one budget is 0.1s"
            await asyncio.sleep(0.7)  # the build ends and is forgotten
            assert not idx._overdue_builds(), "an ended build is no longer overdue"
            ranked = await idx.rank_actions(query="agent stuff", top_k=5, embed_timeout_s=0.1)
            assert ranked and ranked[0][0] == "platform_agent_thing"

        _run(_scenario())
    finally:
        _fake_em.generate_embeddings_batch = orig
    assert len(calls) == 1


def test_one_build_serves_every_view_and_ranking_still_gates_super_admin_actions():
    """The boot warm-up and a turn without super-admin share one build of the widest
    view; who may see a super-admin-only action is still decided when ranking."""
    su_only = _make("platform_agent_su_thing", category="agents")
    su_only.super_admin_only = True
    actions = [_make("platform_agent_thing", category="agents"), su_only]
    idx = _make_index(actions)
    calls: List[List[str]] = []
    orig = _fake_em.generate_embeddings_batch
    _fake_em.generate_embeddings_batch = _slow_batch(0.1, calls)
    try:
        async def _scenario():
            warm = asyncio.ensure_future(idx.warm())
            plain = await idx.rank_actions(query="agent stuff", top_k=5, embed_timeout_s=1.0)
            await warm
            su = await idx.rank_actions(
                query="agent stuff", top_k=5, embed_timeout_s=1.0, include_super_admin=True
            )
            return plain, su

        plain, su = _run(_scenario())
    finally:
        _fake_em.generate_embeddings_batch = orig
    assert len(calls) == 1, "one build for the warm-up and both views"
    assert [n for n, _ in plain] == ["platform_agent_thing"]
    assert {n for n, _ in su} == {"platform_agent_thing", "platform_agent_su_thing"}
