"""Tests for PlatformActionsSection graph routing integration (PRD-139 US-005).

These tests verify:
1. Graph path is taken when TOOL_ROUTING_GRAPH=True and edges are populated
2. Graph path exceptions fall back to the existing embedding path
3. Graph path is never reached when TOOL_ROUTING_GRAPH=False
4. Chain hints are rendered for multi-action sequences

Collection-pollution discipline (PRD-142 W2-S2b): the section is loaded with
import-time stubs (estimator / base) that are restored immediately, so this
module leaves sys.modules untouched at collection time. The runtime stubs the
section imports lazily inside ``render()`` (config / action_registry /
action_semantic_index / graph_router) are (re)installed per-test by the autouse
fixture and torn down after each test. Each platform-actions test file is fully
self-contained — no cross-file ``setdefault`` coupling.
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

_THIS = Path(__file__).resolve()
_ORCH = _THIS.parents[1]
_SECTIONS = _ORCH / "modules" / "context" / "sections"
_DISCOVERY = _ORCH / "modules" / "tools" / "discovery"


# ---------------------------------------------------------------------------
# Watched sys.modules keys — snapshot now, restore after the import-time block.
# ---------------------------------------------------------------------------
_IMPORT_KEYS = (
    "modules.context.estimator",
    "modules.context.sections",
    "modules.context.sections.base",
    "modules.context.sections.platform_actions",
)
# Read by the section's LAZY imports at render() time — live only during tests.
_RUNTIME_KEYS = (
    "config",
    "modules.tools.discovery.action_registry",
    "modules.tools.discovery.action_semantic_index",
    "modules.tools.discovery.graph_router",
)
_GUARD_KEYS = (
    "modules",
    "modules.context",
    "modules.tools",
    "modules.tools.discovery",
)
_PRE_IMPORT_SNAPSHOT = {
    k: sys.modules.get(k) for k in (_IMPORT_KEYS + _RUNTIME_KEYS + _GUARD_KEYS)
}


def _restore(snapshot: dict) -> None:
    """Restore sys.modules to a captured snapshot (None ⇒ delete the key)."""
    for key, original in snapshot.items():
        if original is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = original


# ---------------------------------------------------------------------------
# Pre-load the real ActionRegistry under a private name (no pollution).
# ---------------------------------------------------------------------------
_ar_spec = importlib.util.spec_from_file_location(
    "action_registry_gt", _DISCOVERY / "action_registry.py"
)
_ar_mod = importlib.util.module_from_spec(_ar_spec)
sys.modules["action_registry_gt"] = _ar_mod
_ar_spec.loader.exec_module(_ar_mod)
ActionDefinition = _ar_mod.ActionDefinition
ActionRegistry = _ar_mod.ActionRegistry


# ---------------------------------------------------------------------------
# Build stub module objects (installed transiently for import, then per-test).
# ---------------------------------------------------------------------------
class _StubConfig:
    SEMANTIC_TOOL_ROUTING = True
    SEMANTIC_TOOL_ROUTING_TOP_K = 3
    TOOL_ROUTING_GRAPH = False
    PLATFORM_ACTIONS_MAX_TOKENS = 4000


_stub_config_module = ModuleType("config")
_stub_config_module.config = _StubConfig()


class _NoopEstimator:
    def estimate(self, c: str) -> int:
        return len(c) // 4 + 1


_estimator_mod = ModuleType("modules.context.estimator")
_estimator_mod.TokenEstimator = _NoopEstimator


_ar_module = ModuleType("modules.tools.discovery.action_registry")
_ar_module.ActionDefinition = ActionDefinition
_ar_module.ActionRegistry = ActionRegistry
_ar_module.get_action_registry = lambda: None  # replaced per-test via _install_registry()


class _FakeSemanticIndex:
    def __init__(self):
        self.calls: List[dict] = []
        self.next_result: List[Tuple[str, float]] = []
        self.exception: Optional[Exception] = None

    async def rank_actions(self, query, top_k=15, exclude_admin=True, exclude_promoted=True, include_super_admin=False):
        self.calls.append({"query": query, "top_k": top_k, "exclude_admin": exclude_admin, "exclude_promoted": exclude_promoted, "include_super_admin": include_super_admin})
        if self.exception:
            raise self.exception
        return list(self.next_result)


_asi_module = ModuleType("modules.tools.discovery.action_semantic_index")
_asi_module._fake_singleton = _FakeSemanticIndex()
_asi_module.get_action_semantic_index = lambda: _asi_module._fake_singleton


class _FakeGraphRouter:
    """Controllable fake for GraphRouter used by all tests in this file."""

    def __init__(self):
        self.calls: List[dict] = []
        self.next_result: List[Tuple[str, float, List[str]]] = []
        self.exception: Optional[Exception] = None

    async def rank_chains(self, query, *, workspace_id, agent_id=None, top_k=15, exclude_admin=True, exclude_promoted=True):
        # PRD-177 S5: workspace_id is now a required keyword — the section must
        # thread the tenant's workspace so the per-tenant graph is read.
        self.calls.append({
            "query": query, "workspace_id": workspace_id,
            "agent_id": agent_id, "top_k": top_k,
        })
        if self.exception:
            raise self.exception
        return list(self.next_result)


_fake_graph_router = _FakeGraphRouter()
_grm = ModuleType("modules.tools.discovery.graph_router")
_grm.get_graph_router = lambda: _fake_graph_router
_grm.GraphRouter = _FakeGraphRouter


# Runtime stubs the section's lazy imports resolve at render() time.
_RUNTIME_STUBS = {
    "config": _stub_config_module,
    "modules.tools.discovery.action_registry": _ar_module,
    "modules.tools.discovery.action_semantic_index": _asi_module,
    "modules.tools.discovery.graph_router": _grm,
}
_runtime_live_snapshot: dict = {}


def _install_runtime_stubs() -> None:
    """Install runtime stubs, remembering whatever was there to restore later."""
    global _runtime_live_snapshot
    _runtime_live_snapshot = {k: sys.modules.get(k) for k in _RUNTIME_STUBS}
    sys.modules.update(_RUNTIME_STUBS)


def _restore_runtime_stubs() -> None:
    _restore(_runtime_live_snapshot)


# ---------------------------------------------------------------------------
# Import-time block: bind SectionContext / PlatformActionsSection from disk,
# then restore sys.modules (collection-safe). config / registry / index /
# graph_router are render()-time only, so they stay out of the import.
# ---------------------------------------------------------------------------
sys.modules["modules.context.estimator"] = _estimator_mod

_sections_pkg = ModuleType("modules.context.sections")
_sections_pkg.__path__ = [str(_SECTIONS)]
sys.modules["modules.context.sections"] = _sections_pkg

_base_spec = importlib.util.spec_from_file_location(
    "modules.context.sections.base", _SECTIONS / "base.py"
)
_base_mod = importlib.util.module_from_spec(_base_spec)
sys.modules["modules.context.sections.base"] = _base_mod
_base_spec.loader.exec_module(_base_mod)
SectionContext = _base_mod.SectionContext

_sec_spec = importlib.util.spec_from_file_location(
    "modules.context.sections.platform_actions", _SECTIONS / "platform_actions.py"
)
_sec_mod = importlib.util.module_from_spec(_sec_spec)
sys.modules["modules.context.sections.platform_actions"] = _sec_mod
_sec_spec.loader.exec_module(_sec_mod)
PlatformActionsSection = _sec_mod.PlatformActionsSection

_restore(_PRE_IMPORT_SNAPSHOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_action(name: str, *, category: str = "agents", description: str = "") -> ActionDefinition:
    return ActionDefinition(
        name=name,
        description=description or f"{name} description",
        category=category,
        parameters={"type": "object", "properties": {}, "required": []},
        admin_only=False,
        promoted=False,
    )


def _install_registry(actions: List[ActionDefinition]) -> ActionRegistry:
    reg = ActionRegistry()
    reg._initialized = True
    for a in actions:
        reg.register(a)
    _ar_module.get_action_registry = lambda: reg
    return reg


def _ctx(query: str = "test query", agent_id: Optional[int] = None) -> SectionContext:
    kwargs: dict = {"query": query}
    if agent_id is not None:
        kwargs["agent_id"] = agent_id
    return SectionContext(agent=MagicMock(), workspace_id="ws-1", kwargs=kwargs)


def _set_flags(semantic: bool = True, graph: bool = False, top_k: int = 3) -> None:
    cfg = _stub_config_module.config
    cfg.SEMANTIC_TOOL_ROUTING = semantic
    cfg.SEMANTIC_TOOL_ROUTING_TOP_K = top_k
    cfg.TOOL_ROUTING_GRAPH = graph


def _reset_graph_router() -> None:
    _fake_graph_router.calls.clear()
    _fake_graph_router.next_result = []
    _fake_graph_router.exception = None


def _reset_semantic_index() -> None:
    fake = _asi_module._fake_singleton
    fake.calls.clear()
    fake.next_result = []
    fake.exception = None


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _reset_state():
    _install_runtime_stubs()
    _set_flags(semantic=True, graph=False, top_k=3)
    _reset_graph_router()
    _reset_semantic_index()
    try:
        yield
    finally:
        _reset_graph_router()
        _reset_semantic_index()
        _restore_runtime_stubs()


# ---------------------------------------------------------------------------
# AC #8: flag on + edges populated -> graph path called, chains rendered
# ---------------------------------------------------------------------------


def test_graph_path_called_when_flag_on_and_chains_populated():
    """When TOOL_ROUTING_GRAPH=True and graph returns chains, graph path is used."""
    actions = [
        _make_action("platform_get_latest_report", category="reports"),
        _make_action("platform_submit_report", category="reports"),
        _make_action("platform_list_agents", category="agents"),
    ]
    _install_registry(actions)
    _set_flags(semantic=True, graph=True, top_k=5)

    _fake_graph_router.next_result = [
        ("platform_get_latest_report", 0.92, ["platform_get_latest_report", "platform_submit_report"]),
        ("platform_list_agents", 0.85, ["platform_list_agents"]),
    ]

    section = PlatformActionsSection()
    out = _run(section.render(_ctx(query="submit a report")))

    # Graph router was called
    assert _fake_graph_router.calls, "GraphRouter.rank_chains must be called"
    assert _fake_graph_router.calls[0]["query"] == "submit a report"

    # Chain hints rendered for the multi-action chain
    assert "## Likely Platform Action Chains" in out
    assert "`platform_get_latest_report` then `platform_submit_report`" in out

    # Actions appear in filtered summary
    assert "platform_get_latest_report" in out
    assert "platform_submit_report" in out
    assert "platform_list_agents" in out

    # Semantic index was NOT called (graph handled it)
    assert _asi_module._fake_singleton.calls == []


def test_graph_path_passes_agent_id_from_context():
    """agent_id from ctx.kwargs is passed to rank_chains."""
    actions = [_make_action("platform_list_agents", category="agents")]
    _install_registry(actions)
    _set_flags(semantic=True, graph=True)

    _fake_graph_router.next_result = [
        ("platform_list_agents", 0.9, ["platform_list_agents"]),
    ]

    section = PlatformActionsSection()
    _run(section.render(_ctx(query="list agents", agent_id=42)))

    assert _fake_graph_router.calls[0]["agent_id"] == 42
    # PRD-177 S5: the section threads ctx.workspace_id so the per-tenant graph
    # is read for THIS workspace, not globally.
    assert _fake_graph_router.calls[0]["workspace_id"] == "ws-1"


def test_graph_path_no_agent_id_passes_none():
    """When no agent_id in kwargs, None is passed to rank_chains."""
    actions = [_make_action("platform_list_agents", category="agents")]
    _install_registry(actions)
    _set_flags(semantic=True, graph=True)

    _fake_graph_router.next_result = [
        ("platform_list_agents", 0.9, ["platform_list_agents"]),
    ]

    section = PlatformActionsSection()
    _run(section.render(_ctx(query="list agents")))

    assert _fake_graph_router.calls[0]["agent_id"] is None


def test_graph_single_action_chains_skip_hints():
    """When all chains are single-action, no hint block is rendered."""
    actions = [
        _make_action("platform_list_agents", category="agents"),
        _make_action("platform_create_agent", category="agents"),
    ]
    _install_registry(actions)
    _set_flags(semantic=True, graph=True)

    _fake_graph_router.next_result = [
        ("platform_list_agents", 0.9, ["platform_list_agents"]),
        ("platform_create_agent", 0.8, ["platform_create_agent"]),
    ]

    section = PlatformActionsSection()
    out = _run(section.render(_ctx(query="agent stuff")))

    assert "## Likely Platform Action Chains" not in out
    assert "platform_list_agents" in out
    assert "platform_create_agent" in out


# ---------------------------------------------------------------------------
# AC #9: flag on + GraphRouter raises -> embedding fallback works
# ---------------------------------------------------------------------------


def test_graph_raises_falls_back_to_embedding():
    """When GraphRouter.rank_chains raises, fall through to embedding path."""
    actions = [
        _make_action("platform_list_agents", category="agents"),
        _make_action("platform_create_agent", category="agents"),
    ]
    _install_registry(actions)
    _set_flags(semantic=True, graph=True, top_k=3)

    # Graph raises
    _fake_graph_router.exception = RuntimeError("graph DB unavailable")

    # Embedding fallback returns results
    _asi_module._fake_singleton.next_result = [
        ("platform_list_agents", 0.95),
    ]

    section = PlatformActionsSection()
    out = _run(section.render(_ctx(query="list my agents")))

    # Graph was attempted
    assert _fake_graph_router.calls, "Graph must be attempted before fallback"

    # Embedding fallback was used
    assert _asi_module._fake_singleton.calls, "Embedding fallback must be called"
    assert "platform_list_agents" in out


def test_graph_returns_empty_falls_back_to_embedding():
    """When GraphRouter returns no chains, fall through to embedding path."""
    actions = [
        _make_action("platform_list_agents", category="agents"),
    ]
    _install_registry(actions)
    _set_flags(semantic=True, graph=True, top_k=3)

    # Graph returns empty
    _fake_graph_router.next_result = []

    # Embedding provides results
    _asi_module._fake_singleton.next_result = [
        ("platform_list_agents", 0.88),
    ]

    section = PlatformActionsSection()
    out = _run(section.render(_ctx(query="agents")))

    assert _fake_graph_router.calls, "Graph must be attempted"
    assert _asi_module._fake_singleton.calls, "Embedding must be called as fallback"
    assert "platform_list_agents" in out


# ---------------------------------------------------------------------------
# AC #10: flag off -> graph code never reached
# ---------------------------------------------------------------------------


def test_graph_never_called_when_flag_off():
    """When TOOL_ROUTING_GRAPH=False, GraphRouter is never invoked."""
    actions = [_make_action("platform_list_agents", category="agents")]
    _install_registry(actions)
    _set_flags(semantic=True, graph=False, top_k=3)

    # Even if graph is configured with results, it shouldn't be called
    _fake_graph_router.next_result = [
        ("platform_list_agents", 0.99, ["platform_list_agents"]),
    ]
    _asi_module._fake_singleton.next_result = [
        ("platform_list_agents", 0.9),
    ]

    section = PlatformActionsSection()
    out = _run(section.render(_ctx(query="list agents")))

    assert _fake_graph_router.calls == [], "Graph must NOT be called when flag is off"
    assert _asi_module._fake_singleton.calls, "Embedding path must be used instead"
    assert "platform_list_agents" in out


def test_graph_never_called_when_semantic_routing_off():
    """When SEMANTIC_TOOL_ROUTING=False, neither graph nor embedding is called."""
    actions = [_make_action("platform_list_agents", category="agents")]
    _install_registry(actions)
    _set_flags(semantic=False, graph=True, top_k=3)

    section = PlatformActionsSection()
    out = _run(section.render(_ctx(query="list agents")))

    assert _fake_graph_router.calls == [], "Graph must not run when semantic off"
    assert _asi_module._fake_singleton.calls == [], "Index must not run when semantic off"
    # Full dump returned
    assert "platform_list_agents" in out
