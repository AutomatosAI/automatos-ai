"""Tests for PlatformActionsSection graph routing integration (PRD-139 US-005).

These tests verify:
1. Graph path is taken when TOOL_ROUTING_GRAPH=True and edges are populated
2. Graph path exceptions fall back to the existing embedding path
3. Graph path is never reached when TOOL_ROUTING_GRAPH=False
4. Chain hints are rendered for multi-action sequences

Strategy: imports the SAME PlatformActionsSection that the existing PRD-138
tests set up. Monkeypatches config and graph_router at the pytest-function level
so there are ZERO sys.modules conflicts regardless of collection order.
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

# ---------------------------------------------------------------------------
# The existing test file (test_platform_actions_section.py) may or may not
# have run first. We need the same PlatformActionsSection class and stubs.
# Strategy: replicate the EXACT module-level setup from that file, but guard
# with setdefault so whichever loads first wins — they're identical.
# ---------------------------------------------------------------------------

_THIS = Path(__file__).resolve()
_ORCH = _THIS.parents[1]
_SECTIONS = _ORCH / "modules" / "context" / "sections"
_DISCOVERY = _ORCH / "modules" / "tools" / "discovery"

# Load ActionRegistry class (private, doesn't interfere)
_ar_spec = importlib.util.spec_from_file_location(
    "action_registry_gt", _DISCOVERY / "action_registry.py"
)
_ar_mod = importlib.util.module_from_spec(_ar_spec)
sys.modules.setdefault("action_registry_gt", _ar_mod)
if not hasattr(_ar_mod, "ActionDefinition"):
    _ar_spec.loader.exec_module(_ar_mod)
ActionDefinition = _ar_mod.ActionDefinition
ActionRegistry = _ar_mod.ActionRegistry

# Ensure package stubs
for pkg in ("modules", "modules.context", "modules.tools", "modules.tools.discovery"):
    if pkg not in sys.modules:
        m = ModuleType(pkg)
        m.__path__ = []
        sys.modules[pkg] = m

# TokenEstimator
if "modules.context.estimator" not in sys.modules:
    _est = ModuleType("modules.context.estimator")
    class _E:
        def estimate(self, c: str) -> int:
            return len(c) // 4 + 1
    _est.TokenEstimator = _E
    sys.modules["modules.context.estimator"] = _est

# Config stub — must have all attrs both test files need
if "config" not in sys.modules:
    _cm = ModuleType("config")
    class _C:
        SEMANTIC_TOOL_ROUTING = True
        SEMANTIC_TOOL_ROUTING_TOP_K = 3
        TOOL_ROUTING_GRAPH = False
    _cm.config = _C()
    sys.modules["config"] = _cm
else:
    # Ensure TOOL_ROUTING_GRAPH exists on whatever config is already loaded
    _existing_cfg = sys.modules["config"].config
    if not hasattr(_existing_cfg, "TOOL_ROUTING_GRAPH"):
        _existing_cfg.TOOL_ROUTING_GRAPH = False

# BaseSection / SectionContext
if "modules.context.sections" not in sys.modules:
    _sp = ModuleType("modules.context.sections")
    _sp.__path__ = [str(_SECTIONS)]
    sys.modules["modules.context.sections"] = _sp
if "modules.context.sections.base" not in sys.modules:
    _bs = importlib.util.spec_from_file_location(
        "modules.context.sections.base", _SECTIONS / "base.py"
    )
    _bm = importlib.util.module_from_spec(_bs)
    sys.modules["modules.context.sections.base"] = _bm
    _bs.loader.exec_module(_bm)

SectionContext = sys.modules["modules.context.sections.base"].SectionContext

# Action registry stub
if "modules.tools.discovery.action_registry" not in sys.modules:
    _arm = ModuleType("modules.tools.discovery.action_registry")
    _arm.ActionDefinition = ActionDefinition
    _arm.ActionRegistry = ActionRegistry
    _arm.get_action_registry = lambda: None
    sys.modules["modules.tools.discovery.action_registry"] = _arm

_ar_module = sys.modules["modules.tools.discovery.action_registry"]
# Ensure our classes are there
if not hasattr(_ar_module, "get_action_registry"):
    _ar_module.get_action_registry = lambda: None

# Action semantic index stub
if "modules.tools.discovery.action_semantic_index" not in sys.modules:
    _asim = ModuleType("modules.tools.discovery.action_semantic_index")

    class _FSI:
        def __init__(self):
            self.calls: List[dict] = []
            self.next_result: List[Tuple[str, float]] = []
            self.exception: Optional[Exception] = None

        async def rank_actions(self, query, top_k=15, exclude_admin=True, exclude_promoted=True):
            self.calls.append({"query": query, "top_k": top_k, "exclude_admin": exclude_admin, "exclude_promoted": exclude_promoted})
            if self.exception:
                raise self.exception
            return list(self.next_result)

    _asim._fake_singleton = _FSI()
    _asim.get_action_semantic_index = lambda: _asim._fake_singleton
    sys.modules["modules.tools.discovery.action_semantic_index"] = _asim

_asi_module = sys.modules["modules.tools.discovery.action_semantic_index"]

# Graph router stub — this is the NEW module for PRD-139
class _FakeGraphRouter:
    """Controllable fake for GraphRouter used by all tests in this file."""

    def __init__(self):
        self.calls: List[dict] = []
        self.next_result: List[Tuple[str, float, List[str]]] = []
        self.exception: Optional[Exception] = None

    async def rank_chains(self, query, agent_id=None, top_k=15, exclude_admin=True, exclude_promoted=True):
        self.calls.append({"query": query, "agent_id": agent_id, "top_k": top_k})
        if self.exception:
            raise self.exception
        return list(self.next_result)


_fake_graph_router = _FakeGraphRouter()

_grm = ModuleType("modules.tools.discovery.graph_router")
_grm.get_graph_router = lambda: _fake_graph_router
_grm.GraphRouter = _FakeGraphRouter
sys.modules["modules.tools.discovery.graph_router"] = _grm

# Load the section module (or reuse existing). The section uses lazy imports
# so whatever is in sys.modules at CALL TIME is what gets resolved.
if "modules.context.sections.platform_actions" not in sys.modules:
    _ss = importlib.util.spec_from_file_location(
        "modules.context.sections.platform_actions", _SECTIONS / "platform_actions.py"
    )
    _sm = importlib.util.module_from_spec(_ss)
    sys.modules["modules.context.sections.platform_actions"] = _sm
    _ss.loader.exec_module(_sm)

PlatformActionsSection = sys.modules["modules.context.sections.platform_actions"].PlatformActionsSection


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
    cfg = sys.modules["config"].config
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
    _set_flags(semantic=True, graph=False, top_k=3)
    _reset_graph_router()
    _reset_semantic_index()
    yield
    _set_flags(semantic=True, graph=False, top_k=3)
    _reset_graph_router()
    _reset_semantic_index()


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
