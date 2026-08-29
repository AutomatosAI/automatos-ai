"""
PRD-232 US-003 — one rank_actions pass per turn.
================================================

C5/§4: ``ActionSemanticIndex.rank_actions`` ran 4× on the same turn query —
dispatcher narrowing, the shadow surface, the prompt catalog, and the graph
entry nodes each embedded + cosine-ranked independently.

US-003 makes the expensive work (index + query embed + cosine over the su-gated
candidate set) compute ONCE per turn inside ``rank_actions_scope()`` and every
surface slice its own view: narrowing drops promoted, the shadow keeps promoted,
the graph takes top-5. Outside the scope, behaviour is unchanged.

These tests build a real ``ActionSemanticIndex`` via ``__new__`` with light
in-memory fakes (no Redis / OpenRouter / DB) — the pattern the module documents
— and spy on ``_compute_full_ranking`` (the single embedding-ranking
computation) to count how many times a turn actually ranks.
"""
from __future__ import annotations

import asyncio
import types

import pytest

from modules.tools.discovery.action_semantic_index import (
    ActionSemanticIndex,
    rank_actions_scope,
)

QUERY = "close the blocked tickets from vector"
WS = "workspace-1"


# ---------------------------------------------------------------------------
# Light fakes (deterministic — no Redis / OpenRouter / DB)
# ---------------------------------------------------------------------------

def _vec(text: str):
    """Keyword-axed one-hot-ish vector; a small base component keeps norm != 0."""
    t = text.lower()
    return [
        1.0 if "agent" in t else 0.0,
        1.0 if any(w in t for w in ("task", "ticket", "board", "close", "blocked")) else 0.0,
        1.0 if "admin" in t else 0.0,
        0.1,
    ]


class _FakeEM:
    def __init__(self):
        self.provider = types.SimpleNamespace(config=types.SimpleNamespace(model="fake-model"))

    def get_provider_info(self):
        return {"provider": "fake", "model": "fake-model", "dimension": 4}

    def get_dimension(self):
        return 4

    async def generate_embedding(self, text):
        return _vec(text)

    async def generate_embeddings_batch(self, texts, max_concurrent=5):
        return [_vec(t) for t in texts]


class _FakeCache:
    def __init__(self):
        self.store = {}

    def get_embeddings_batch(self, texts, model="default"):
        bucket = self.store.get(model, {})
        return {t: bucket.get(t) for t in texts}

    def set_embeddings_batch(self, embeddings, model="default"):
        self.store.setdefault(model, {}).update(embeddings)


def _action(name, description, *, admin_only=False, promoted=False):
    return types.SimpleNamespace(
        name=name, description=description, category="agents",
        tags=[], examples=[], admin_only=admin_only, promoted=promoted,
        super_admin_only=False,
    )


class _FakeRegistry:
    def __init__(self, actions):
        self._actions = list(actions)

    def get_all(self):
        return list(self._actions)


_ACTIONS = [
    _action("platform_update_task_status", "close or update a task/ticket on the board"),
    _action("platform_list_agents", "list the agents in the workspace"),
    _action("platform_find_tools", "find tools for the task", promoted=True),
    _action("platform_admin_purge", "admin only board purge task", admin_only=True),
]


def _make_index():
    idx = ActionSemanticIndex.__new__(ActionSemanticIndex)
    idx._embedding_manager = _FakeEM()
    idx._cache = _FakeCache()
    idx._registry = _FakeRegistry(_ACTIONS)
    idx._action_embeddings = {}
    idx._indexed = False
    idx._lock = None
    idx._inflight = {}
    idx._rank_inflight = {}
    return idx


def _spy_compute(idx):
    """Wrap _compute_full_ranking to count how many times it actually computes."""
    count = {"n": 0}
    orig = idx._compute_full_ranking

    async def spy(*a, **k):
        count["n"] += 1
        return await orig(*a, **k)

    idx._compute_full_ranking = spy
    return count


def _names(ranked):
    return [n for n, _ in ranked]


# ---------------------------------------------------------------------------
# AC1 — exactly 1 computation for a full-path turn (narrow + shadow + catalog + graph)
# ---------------------------------------------------------------------------

def test_one_computation_per_turn_across_all_four_surfaces():
    idx = _make_index()
    count = _spy_compute(idx)

    async def _turn():
        with rank_actions_scope():
            # 1) dispatcher narrowing — drop promoted
            narrow = await idx.rank_actions(
                QUERY, top_k=15, exclude_admin=True, exclude_promoted=True, workspace_id=WS
            )
            # 2) shadow surface — keep promoted
            shadow = await idx.rank_actions(
                QUERY, top_k=15, exclude_admin=True, exclude_promoted=False, workspace_id=WS
            )
            # 3) prompt catalog — same gate as narrowing
            catalog = await idx.rank_actions(
                QUERY, top_k=15, exclude_admin=True, exclude_promoted=True, workspace_id=WS
            )
            # 4) graph entry nodes — top-5 slice
            graph = await idx.rank_actions(
                QUERY, top_k=5, exclude_admin=True, exclude_promoted=True, workspace_id=WS
            )
        return narrow, shadow, catalog, graph

    narrow, shadow, catalog, graph = asyncio.run(_turn())

    assert count["n"] == 1, f"expected 1 embedding-ranking computation, got {count['n']}"
    # The board-write action is the top hit for the VECTOR query.
    assert narrow[0][0] == "platform_update_task_status"
    # narrowing/catalog share the exact same sliced view.
    assert _names(narrow) == _names(catalog)


# ---------------------------------------------------------------------------
# AC2 — graph + shadow consume the shared result by SLICING, not recomputing
# ---------------------------------------------------------------------------

def test_shadow_and_graph_slice_the_shared_result():
    idx = _make_index()
    count = _spy_compute(idx)

    async def _turn():
        with rank_actions_scope():
            narrow = await idx.rank_actions(
                QUERY, top_k=15, exclude_admin=True, exclude_promoted=True, workspace_id=WS
            )
            shadow = await idx.rank_actions(
                QUERY, top_k=15, exclude_admin=True, exclude_promoted=False, workspace_id=WS
            )
            graph = await idx.rank_actions(
                QUERY, top_k=1, exclude_admin=True, exclude_promoted=True, workspace_id=WS
            )
        return narrow, shadow, graph

    narrow, shadow, graph = asyncio.run(_turn())

    assert count["n"] == 1
    # Shadow keeps the promoted action; narrowing dropped it — both from ONE ranking.
    assert "platform_find_tools" in _names(shadow)
    assert "platform_find_tools" not in _names(narrow)
    # Admin action never surfaces on either (fail-closed filter is per-call).
    assert "platform_admin_purge" not in _names(shadow)
    # Graph keeps its own smaller cap by slicing.
    assert len(graph) == 1 and graph[0][0] == "platform_update_task_status"


# ---------------------------------------------------------------------------
# AC3 — tenant isolation: workspace_id is in the memo key
# ---------------------------------------------------------------------------

def test_workspace_in_key_no_cross_tenant_reuse():
    idx = _make_index()
    count = _spy_compute(idx)

    async def _scenario():
        with rank_actions_scope():
            await idx.rank_actions(QUERY, workspace_id="tenant-A")
            after_a = count["n"]
            # Same query, DIFFERENT workspace → must NOT reuse A's ranking.
            await idx.rank_actions(QUERY, workspace_id="tenant-B")
            after_b = count["n"]
            # Repeat tenant A → served from the memo, no new computation.
            await idx.rank_actions(QUERY, workspace_id="tenant-A")
            after_a2 = count["n"]
        return after_a, after_b, after_a2

    after_a, after_b, after_a2 = asyncio.run(_scenario())
    assert after_a == 1
    assert after_b == 2, "different workspace shared a cache entry (key not tenant-scoped)"
    assert after_a2 == 2, "repeat of tenant A recomputed (memo miss)"


# ---------------------------------------------------------------------------
# Control — outside a scope, behaviour is unchanged (each call recomputes)
# ---------------------------------------------------------------------------

def test_no_scope_each_call_recomputes():
    idx = _make_index()
    count = _spy_compute(idx)

    async def _seq():
        await idx.rank_actions(QUERY, workspace_id=WS)
        await idx.rank_actions(QUERY, workspace_id=WS)
        return count["n"]

    n = asyncio.run(_seq())
    assert n == 2, "without a scope the per-turn memo must not persist across calls"


def test_scope_result_does_not_leak_across_turns():
    """A second turn (new scope) recomputes — the memo dies with its turn, so a
    completed ranking never survives the request that produced it."""
    idx = _make_index()
    count = _spy_compute(idx)

    async def _two_turns():
        with rank_actions_scope():
            await idx.rank_actions(QUERY, workspace_id=WS)
        first = count["n"]
        with rank_actions_scope():
            await idx.rank_actions(QUERY, workspace_id=WS)
        return first, count["n"]

    first, second = asyncio.run(_two_turns())
    assert first == 1
    assert second == 2, "a new turn's scope reused the previous turn's memo"
