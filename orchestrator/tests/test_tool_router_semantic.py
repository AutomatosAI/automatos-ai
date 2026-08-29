"""Tests for PRD-138 US-009 — semantic narrowing of platform_execute via
tool_router.get_tools_for_agent(query=...).

Covers:
- _semantic_routing_enabled / _semantic_routing_top_k config helpers.
- _run_coroutine_blocking from sync and from inside a running event loop.
- _rank_actions_for_dispatcher happy-path, raise-path, empty-path.
- get_tools_for_agent: query narrows the dispatcher enum; absence of query
  leaves the enum unchanged; rank_actions raising still yields a callable
  (full-enum) schema.

Heavy modules (DB, ToolRegistry, EmbeddingManager) are stubbed so this stays
a pure unit test.

Test isolation note: this file shares ``modules.tools.discovery.action_registry``
and ``modules.tools.discovery.action_semantic_index`` with
``test_platform_actions_section.py``. To avoid cross-file pollution (pytest
imports both test files at collection time before running any tests), this
file:
  1. Snapshots whatever was at those sys.modules keys BEFORE we stub.
  2. Installs its own stubs only long enough to import tool_router.
  3. Immediately restores the snapshot at module level.
  4. Re-installs stubs per-test via an autouse fixture so each test sees its
     own controllable singletons, and restores after.
"""
from __future__ import annotations

import asyncio
import importlib
import importlib.util
import logging
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Locate the worktree's orchestrator/ root so we can import the real modules.
# ---------------------------------------------------------------------------
_THIS = Path(__file__).resolve()
_ORCH = _THIS.parents[1]


# ---------------------------------------------------------------------------
# Snapshot sys.modules state we are about to clobber, so other test files
# (notably test_platform_actions_section.py) keep working when pytest runs us
# alongside them.
# ---------------------------------------------------------------------------
_DISCOVERY_KEYS = [
    "modules.tools.discovery.action_registry",
    "modules.tools.discovery.action_semantic_index",
]
_SNAPSHOT: Dict[str, Optional[types.ModuleType]] = {
    k: sys.modules.get(k) for k in _DISCOVERY_KEYS
}

# The low-level stubs below SHADOW real importable packages (core,
# core.database.database, modules.tools.*, config). Left in sys.modules after
# this module imports, they poison the collection of every sibling test that
# imports the real ``core.*`` / ``modules.tools.*`` tree. We snapshot them
# before install and restore at module level right after tool_router is loaded
# (its top-level imports are already bound by then). ``config`` is read lazily
# at runtime, so the autouse fixture re-installs it per-test. (PRD-142 W2-S2b.)
_LOW_LEVEL_KEYS = (
    "core",
    "core.database",
    "core.database.database",
    "modules",
    "modules.tools",
    # _install_discovery_stubs() creates this package object via _ensure_pkg;
    # snapshot+restore it here too so a pathless stub never leaks to siblings.
    "modules.tools.discovery",
    "modules.tools.registry",
    "modules.tools.execution",
    "modules.tools.formatting",
    "modules.tools.formatting.result_formatter",
    "config",
)
_LOW_LEVEL_SNAPSHOT: Dict[str, Optional[types.ModuleType]] = {}
_FAKE_CONFIG_MOD: Optional[types.ModuleType] = None


# ---------------------------------------------------------------------------
# Fakes wired into sys.modules BEFORE we import tool_router.
# tool_router.py does top-level imports of modules.tools.registry,
# modules.tools.execution, modules.tools.formatting.result_formatter and
# core.database.database — we don't need their real behaviour for these
# tests, so swap in stubs that satisfy the import.
# ---------------------------------------------------------------------------


def _ensure_pkg(name: str) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    mod.__path__ = []  # mark as package
    sys.modules[name] = mod
    return mod


def _install_low_level_stubs():
    # Snapshot every key we are about to shadow BEFORE touching it, so the
    # module-level restore can return sys.modules to its real state.
    for _k in _LOW_LEVEL_KEYS:
        _LOW_LEVEL_SNAPSHOT.setdefault(_k, sys.modules.get(_k))

    # core.database.database.SessionLocal — sync no-op factory
    _ensure_pkg("core")
    _ensure_pkg("core.database")
    if "core.database.database" not in sys.modules:
        db_mod = types.ModuleType("core.database.database")
        db_mod.SessionLocal = MagicMock(return_value=MagicMock())
        sys.modules["core.database.database"] = db_mod

    # modules.tools.registry — ToolCategory + get_tool_registry
    _ensure_pkg("modules")
    _ensure_pkg("modules.tools")
    if "modules.tools.registry" not in sys.modules:
        reg_mod = types.ModuleType("modules.tools.registry")

        class _FakeToolCategory:
            AGENTS = "agents"

        fake_registry = MagicMock()
        fake_registry.get_all_tools.return_value = []  # empty so no per-tool work
        fake_registry.validate_tool_access.return_value = (True, "")

        reg_mod.ToolCategory = _FakeToolCategory
        reg_mod.get_tool_registry = lambda db_session=None: fake_registry
        sys.modules["modules.tools.registry"] = reg_mod

    # Always shadow with our complete stub — the snapshot above preserves any
    # prior entry for restore. A `not in sys.modules` guard here would inherit
    # a half-initialised modules.tools.execution left by an earlier-collected
    # test (its real __init__ pulls a long exec_* chain and can land in
    # sys.modules without UnifiedToolExecutor bound), which is the
    # "(unknown location)" collection error this suite hit in CI.
    exec_mod = types.ModuleType("modules.tools.execution")
    exec_mod.UnifiedToolExecutor = MagicMock()
    sys.modules["modules.tools.execution"] = exec_mod

    if "modules.tools.formatting.result_formatter" not in sys.modules:
        _ensure_pkg("modules.tools.formatting")
        fmt_mod = types.ModuleType("modules.tools.formatting.result_formatter")
        fmt_mod.ToolResultFormatter = MagicMock()
        sys.modules["modules.tools.formatting.result_formatter"] = fmt_mod

    # config — singleton with the PRD-138 flags. tool_router reads via
    # `from config import config` lazily, so a module called "config"
    # exposing a `config` attribute is enough.
    #
    # Always shadow with OUR fake — the snapshot above preserves any prior (real)
    # config for restore. A `not in sys.modules` guard here would bind
    # _FAKE_CONFIG_CLS to the REAL Config class whenever an earlier-collected
    # test already imported config (e.g. anything importing a context section);
    # a later `_FAKE_CONFIG_CLS.SEMANTIC_TOOL_ROUTING = ...` toggle would then set
    # an attribute that `from config import config` never reads back, and the
    # flag-off tests would see stale values. Same class of leak the
    # modules.tools.execution stub above documents; fixed the same way.
    config_mod = types.ModuleType("config")

    class _FakeConfig:
        SEMANTIC_TOOL_ROUTING = False
        SEMANTIC_TOOL_ROUTING_TOP_K = 15

    config_mod.config = _FakeConfig()
    sys.modules["config"] = config_mod
    global _FAKE_CONFIG_MOD
    _FAKE_CONFIG_MOD = sys.modules["config"]
    fake_config_cls = type(sys.modules["config"].config)
    return fake_config_cls


def _restore_low_level_snapshot():
    """Undo _install_low_level_stubs at module level so the leaked stubs never
    reach the collection of sibling test modules."""
    for _k, _prior in _LOW_LEVEL_SNAPSHOT.items():
        if _prior is None:
            sys.modules.pop(_k, None)
        else:
            sys.modules[_k] = _prior


_FAKE_CONFIG_CLS = _install_low_level_stubs()


def _load_action_registry():
    """Load action_registry.py under a UNIQUE module name so we never replace
    the canonical ``modules.tools.discovery.action_registry`` entry that other
    test files (test_platform_actions_section.py) own."""
    spec = importlib.util.spec_from_file_location(
        "tr_test_action_registry",
        _ORCH / "modules" / "tools" / "discovery" / "action_registry.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tr_test_action_registry"] = mod
    spec.loader.exec_module(mod)
    return mod


_AR_MOD = _load_action_registry()
ActionDefinition = _AR_MOD.ActionDefinition
ActionRegistry = _AR_MOD.ActionRegistry


def _make_action(
    name: str,
    *,
    category: str = "agents",
    admin_only: bool = False,
    promoted: bool = False,
) -> ActionDefinition:
    return ActionDefinition(
        name=name,
        description=f"{name} description",
        category=category,
        parameters={"type": "object", "properties": {}, "required": []},
        admin_only=admin_only,
        promoted=promoted,
    )


def _build_test_registry() -> ActionRegistry:
    reg = ActionRegistry()
    reg._initialized = True
    reg.register(_make_action("platform_list_agents"))
    reg.register(_make_action("platform_create_agent"))
    reg.register(_make_action("platform_list_missions", category="missions"))
    reg.register(_make_action("platform_get_workspace_info", category="workspace"))
    reg.register(_make_action("platform_admin_only_action", category="admin", admin_only=True))
    reg.register(_make_action("platform_promoted_thing", category="promoted", promoted=True))
    return reg


_test_registry = _build_test_registry()
_dummy_index = MagicMock()


# Build the temporary stub modules used to satisfy tool_router's lazy imports.
def _build_discovery_stub_modules() -> Tuple[types.ModuleType, types.ModuleType]:
    ar_alias = types.ModuleType("modules.tools.discovery.action_registry")
    ar_alias.ActionDefinition = ActionDefinition
    ar_alias.ActionRegistry = ActionRegistry
    ar_alias.get_action_registry = lambda: _test_registry

    asi_alias = types.ModuleType("modules.tools.discovery.action_semantic_index")
    asi_alias.get_action_semantic_index = lambda: _dummy_index
    return ar_alias, asi_alias


_AR_STUB, _ASI_STUB = _build_discovery_stub_modules()


_DISCOVERY_PKG_ATTR = "get_action_registry"
_PKG_ATTR_SENTINEL = object()
_PKG_ATTR_SNAPSHOT: Any = _PKG_ATTR_SENTINEL


def _install_discovery_stubs():
    """Install our stubs at the canonical sys.modules paths and the package
    attribute that ``tool_router`` reads via
    ``from modules.tools.discovery import get_action_registry``."""
    pkg = _ensure_pkg("modules.tools.discovery")
    # tool_router.py top-level-imports modules.tools.discovery.signal_recorder
    # (NOT stubbed; it's leaf-loadable — stdlib-only top imports). Point the
    # (possibly pathless) discovery package at the real dir so that real
    # submodule resolves, while our action_registry / action_semantic_index
    # stubs below still shadow via their explicit sys.modules entries.
    _real_discovery_dir = str(_ORCH / "modules" / "tools" / "discovery")
    if _real_discovery_dir not in getattr(pkg, "__path__", []):
        pkg.__path__ = [_real_discovery_dir]
    global _PKG_ATTR_SNAPSHOT
    _PKG_ATTR_SNAPSHOT = (
        getattr(pkg, _DISCOVERY_PKG_ATTR)
        if hasattr(pkg, _DISCOVERY_PKG_ATTR)
        else _PKG_ATTR_SENTINEL
    )
    setattr(pkg, _DISCOVERY_PKG_ATTR, lambda: _test_registry)
    sys.modules["modules.tools.discovery.action_registry"] = _AR_STUB
    sys.modules["modules.tools.discovery.action_semantic_index"] = _ASI_STUB


def _restore_discovery_snapshot():
    """Put back whatever was at those keys before we stubbed (or remove the
    key if there was nothing). Other test files own those stubs; we must not
    keep ours in place after our module finishes loading."""
    for key, prior in _SNAPSHOT.items():
        if prior is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = prior
    pkg = sys.modules.get("modules.tools.discovery")
    if pkg is not None:
        if _PKG_ATTR_SNAPSHOT is _PKG_ATTR_SENTINEL:
            if hasattr(pkg, _DISCOVERY_PKG_ATTR):
                delattr(pkg, _DISCOVERY_PKG_ATTR)
        else:
            setattr(pkg, _DISCOVERY_PKG_ATTR, _PKG_ATTR_SNAPSHOT)


# Install stubs ONLY long enough to import tool_router; restore immediately
# afterwards so other test files' module-level stubbing wins again.
_install_discovery_stubs()


def _load_tool_router():
    spec = importlib.util.spec_from_file_location(
        "tr_under_test",
        _ORCH / "modules" / "tools" / "tool_router.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tr_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


tool_router = _load_tool_router()
_restore_discovery_snapshot()
_restore_low_level_snapshot()


# ---------------------------------------------------------------------------
# Per-test fixture: re-install our stubs for the duration of one test, then
# restore. This way each individual test in this file sees a clean
# ``_test_registry`` and ``_dummy_index``, but the surrounding test session
# sees whatever stubs other files installed.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _swap_in_router_stubs():
    """Swap our discovery stubs in for the duration of one test."""
    prior_config = sys.modules.get("config")
    if _FAKE_CONFIG_MOD is not None:
        sys.modules["config"] = _FAKE_CONFIG_MOD
    prior_ar = sys.modules.get("modules.tools.discovery.action_registry")
    prior_asi = sys.modules.get("modules.tools.discovery.action_semantic_index")
    pkg = _ensure_pkg("modules.tools.discovery")
    prior_pkg_attr = (
        getattr(pkg, _DISCOVERY_PKG_ATTR)
        if hasattr(pkg, _DISCOVERY_PKG_ATTR)
        else _PKG_ATTR_SENTINEL
    )
    sys.modules["modules.tools.discovery.action_registry"] = _AR_STUB
    sys.modules["modules.tools.discovery.action_semantic_index"] = _ASI_STUB
    setattr(pkg, _DISCOVERY_PKG_ATTR, lambda: _test_registry)
    # Reset config flags to defaults at the start of every test.
    _FAKE_CONFIG_CLS.SEMANTIC_TOOL_ROUTING = False
    _FAKE_CONFIG_CLS.SEMANTIC_TOOL_ROUTING_TOP_K = 15
    try:
        yield
    finally:
        if prior_ar is None:
            sys.modules.pop("modules.tools.discovery.action_registry", None)
        else:
            sys.modules["modules.tools.discovery.action_registry"] = prior_ar
        if prior_asi is None:
            sys.modules.pop("modules.tools.discovery.action_semantic_index", None)
        else:
            sys.modules["modules.tools.discovery.action_semantic_index"] = prior_asi
        if prior_pkg_attr is _PKG_ATTR_SENTINEL:
            if hasattr(pkg, _DISCOVERY_PKG_ATTR):
                delattr(pkg, _DISCOVERY_PKG_ATTR)
        else:
            setattr(pkg, _DISCOVERY_PKG_ATTR, prior_pkg_attr)
        if prior_config is None:
            sys.modules.pop("config", None)
        else:
            sys.modules["config"] = prior_config
        _FAKE_CONFIG_CLS.SEMANTIC_TOOL_ROUTING = False
        _FAKE_CONFIG_CLS.SEMANTIC_TOOL_ROUTING_TOP_K = 15


# ===========================================================================
# Test: config helpers
# ===========================================================================


def test_semantic_routing_disabled_by_default():
    _FAKE_CONFIG_CLS.SEMANTIC_TOOL_ROUTING = False
    assert tool_router._semantic_routing_enabled() is False


def test_semantic_routing_enabled_when_flag_true():
    _FAKE_CONFIG_CLS.SEMANTIC_TOOL_ROUTING = True
    assert tool_router._semantic_routing_enabled() is True


def test_semantic_routing_top_k_default():
    _FAKE_CONFIG_CLS.SEMANTIC_TOOL_ROUTING_TOP_K = 15
    assert tool_router._semantic_routing_top_k() == 15


def test_semantic_routing_top_k_custom():
    _FAKE_CONFIG_CLS.SEMANTIC_TOOL_ROUTING_TOP_K = 20
    assert tool_router._semantic_routing_top_k() == 20


def test_semantic_routing_top_k_invalid_value_falls_back():
    _FAKE_CONFIG_CLS.SEMANTIC_TOOL_ROUTING_TOP_K = "not-a-number"
    assert tool_router._semantic_routing_top_k() == 15


# ===========================================================================
# Test: _run_coroutine_blocking
# ===========================================================================


def test_run_coroutine_blocking_from_sync_context():
    async def _coro():
        return 42

    assert tool_router._run_coroutine_blocking(_coro()) == 42


def test_run_coroutine_blocking_inside_running_loop():
    """Inside a running loop, asyncio.run is illegal — the helper must
    delegate to a worker thread."""

    async def _outer():
        async def _inner():
            return "inner-result"

        # We're inside a running loop here. The helper must not raise.
        return tool_router._run_coroutine_blocking(_inner())

    assert asyncio.run(_outer()) == "inner-result"


# ===========================================================================
# Test: _rank_actions_for_dispatcher
# ===========================================================================


def test_rank_actions_for_dispatcher_happy_path():
    async def _fake_rank(query, top_k, exclude_admin, exclude_promoted, include_super_admin=False, workspace_id=None, **kwargs):
        return [
            ("platform_list_agents", 0.91),
            ("platform_create_agent", 0.83),
        ]

    _dummy_index.rank_actions = _fake_rank
    names = tool_router._rank_actions_for_dispatcher(
        query="list all agents",
        top_k=5,
        exclude_admin=True,
        exclude_promoted=True,
    )
    assert names == ["platform_list_agents", "platform_create_agent"]


def test_rank_actions_for_dispatcher_returns_none_on_raise(caplog):
    async def _raise(*args, **kwargs):
        raise RuntimeError("synthetic-failure")

    _dummy_index.rank_actions = _raise
    with caplog.at_level(logging.WARNING):
        names = tool_router._rank_actions_for_dispatcher(
            query="anything", top_k=5, exclude_admin=True, exclude_promoted=True,
        )
    assert names is None
    assert any("_rank_actions_for_dispatcher failed" in r.message for r in caplog.records)


def test_rank_actions_for_dispatcher_returns_none_on_empty():
    async def _empty(*args, **kwargs):
        return []

    _dummy_index.rank_actions = _empty
    assert tool_router._rank_actions_for_dispatcher(
        query="x", top_k=5, exclude_admin=True, exclude_promoted=True
    ) is None


# ===========================================================================
# Test: get_tools_for_agent — schema narrowing end-to-end
# ===========================================================================


def _dispatcher_from(tools: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pull the platform_execute schema out of a tools list."""
    for t in tools:
        if t.get("function", {}).get("name") == "platform_execute":
            return t
    raise AssertionError(f"platform_execute not in tools list: {tools}")


def _enum_of_tool(tool: Dict[str, Any]) -> Optional[List[str]]:
    return tool["function"]["parameters"]["properties"]["action"].get("enum")


def test_get_tools_for_agent_no_query_returns_full_enum(caplog):
    """AC: query=None → dispatcher enum is the full eligible set."""
    _FAKE_CONFIG_CLS.SEMANTIC_TOOL_ROUTING = True
    with caplog.at_level(logging.INFO):
        tools = tool_router.get_tools_for_agent(
            agent_id=None,
            workspace_id=None,
            is_admin=True,  # avoid workspace-admin DB lookup
        )
    dispatcher = _dispatcher_from(tools)
    enum = _enum_of_tool(dispatcher)
    # Full enum: all eligible actions (admin included since is_admin=True)
    assert "platform_list_agents" in enum
    assert "platform_get_workspace_info" in enum
    assert "platform_admin_only_action" in enum  # is_admin=True → admin allowed
    # PRD-232 US-014: a promoted action that is NOT a config pin and did NOT rank
    # (no query) is no longer attached first-class — it is reachable via the
    # dispatcher enum like any action (promotion-as-prior, not unconditional attach).
    assert "platform_promoted_thing" in enum
    # Trace log says NOT narrowed
    assert any(
        "dispatcher enum NOT narrowed" in r.message
        and "no query supplied" in r.message
        for r in caplog.records
    )


def test_get_tools_for_agent_with_query_narrows_enum(caplog):
    """AC: query + flag on → dispatcher enum is the ranked subset."""
    _FAKE_CONFIG_CLS.SEMANTIC_TOOL_ROUTING = True

    async def _rank(query, top_k, exclude_admin, exclude_promoted, include_super_admin=False, workspace_id=None, **kwargs):
        return [
            ("platform_list_agents", 0.95),
            ("platform_create_agent", 0.80),
        ]

    _dummy_index.rank_actions = _rank
    with caplog.at_level(logging.INFO):
        tools = tool_router.get_tools_for_agent(
            agent_id=None,
            workspace_id=None,
            is_admin=True,
            query="list all agents",
        )
    enum = _enum_of_tool(_dispatcher_from(tools))
    assert sorted(enum) == ["platform_create_agent", "platform_list_agents"]
    assert "platform_get_workspace_info" not in enum
    assert any(
        "dispatcher enum narrowed to 2 actions" in r.message
        for r in caplog.records
    )


def test_get_tools_for_agent_query_with_flag_off_does_not_narrow(caplog):
    """SEMANTIC_TOOL_ROUTING=False → query is ignored, full enum returned."""
    _FAKE_CONFIG_CLS.SEMANTIC_TOOL_ROUTING = False

    async def _should_not_be_called(*args, **kwargs):
        raise AssertionError("rank_actions should NOT be called when flag is off")

    _dummy_index.rank_actions = _should_not_be_called

    with caplog.at_level(logging.INFO):
        tools = tool_router.get_tools_for_agent(
            agent_id=None,
            workspace_id=None,
            is_admin=True,
            query="list all agents",
        )
    enum = _enum_of_tool(_dispatcher_from(tools))
    assert "platform_list_agents" in enum
    assert "platform_get_workspace_info" in enum  # full enum
    assert any(
        "flag SEMANTIC_TOOL_ROUTING=False" in r.message for r in caplog.records
    )


def test_get_tools_for_agent_rank_failure_falls_back_to_full(caplog):
    """AC: when ActionSemanticIndex.rank_actions raises, the returned tools
    list still contains a working platform_execute schema with the full enum
    (fallback proven)."""
    _FAKE_CONFIG_CLS.SEMANTIC_TOOL_ROUTING = True

    async def _boom(*args, **kwargs):
        raise RuntimeError("ranker exploded")

    _dummy_index.rank_actions = _boom
    with caplog.at_level(logging.INFO):
        tools = tool_router.get_tools_for_agent(
            agent_id=None,
            workspace_id=None,
            is_admin=True,
            query="list all agents",
        )
    enum = _enum_of_tool(_dispatcher_from(tools))
    # Full enum (admin included since is_admin=True, promoted excluded)
    assert "platform_list_agents" in enum
    assert "platform_get_workspace_info" in enum
    assert "platform_admin_only_action" in enum
    # Trace log distinguishes this case
    assert any(
        "rank_actions returned empty or raised" in r.message
        for r in caplog.records
    )


def test_get_tools_for_agent_excludes_admin_when_not_admin():
    """Permission gating: non-admin caller never gets admin actions in enum,
    regardless of query / ranking output."""
    _FAKE_CONFIG_CLS.SEMANTIC_TOOL_ROUTING = True

    async def _rank(query, top_k, exclude_admin, exclude_promoted, include_super_admin=False, workspace_id=None, **kwargs):
        # Even if the ranker tried to surface an admin action, the schema
        # builder should drop it.
        return [
            ("platform_list_agents", 0.9),
            ("platform_admin_only_action", 0.85),
        ]

    _dummy_index.rank_actions = _rank
    tools = tool_router.get_tools_for_agent(
        agent_id=None,
        workspace_id=None,
        is_admin=False,
        query="anything",
    )
    enum = _enum_of_tool(_dispatcher_from(tools))
    assert "platform_list_agents" in enum
    assert "platform_admin_only_action" not in enum


# ===========================================================================
# Test: get_tools_for_agent_async — async-native narrowing (no thread bridge)
# ===========================================================================


def test_get_tools_for_agent_async_narrows_enum_without_bridge(caplog):
    """The async entry awaits ranking on the caller's loop — narrowed enum,
    and the thread-bridge helper is never engaged."""
    _FAKE_CONFIG_CLS.SEMANTIC_TOOL_ROUTING = True

    async def _rank(query, top_k, exclude_admin, exclude_promoted, include_super_admin=False, workspace_id=None, **kwargs):
        return [
            ("platform_list_agents", 0.95),
            ("platform_create_agent", 0.80),
        ]

    _dummy_index.rank_actions = _rank
    with caplog.at_level(logging.INFO):
        tools = asyncio.run(
            tool_router.get_tools_for_agent_async(
                agent_id=None,
                workspace_id=None,
                is_admin=True,
                query="list all agents",
            )
        )
    enum = _enum_of_tool(_dispatcher_from(tools))
    assert sorted(enum) == ["platform_create_agent", "platform_list_agents"]
    assert any(
        "dispatcher enum narrowed to 2 actions" in r.message for r in caplog.records
    )
    assert not any(
        "_run_coroutine_blocking" in r.message for r in caplog.records
    ), "async entry must not pay the thread bridge"


def test_get_tools_for_agent_async_rank_failure_falls_back(caplog):
    """Async entry: a raising ranker still yields the full-enum dispatcher."""
    _FAKE_CONFIG_CLS.SEMANTIC_TOOL_ROUTING = True

    async def _boom(*args, **kwargs):
        raise RuntimeError("ranker exploded")

    _dummy_index.rank_actions = _boom
    with caplog.at_level(logging.INFO):
        tools = asyncio.run(
            tool_router.get_tools_for_agent_async(
                agent_id=None,
                workspace_id=None,
                is_admin=True,
                query="list all agents",
            )
        )
    enum = _enum_of_tool(_dispatcher_from(tools))
    assert "platform_list_agents" in enum
    assert "platform_get_workspace_info" in enum
    assert any(
        "rank_actions returned empty or raised" in r.message for r in caplog.records
    )


def test_get_tools_for_agent_async_no_query_full_enum():
    """Async entry mirrors the sync no-query behavior (full enum)."""
    _FAKE_CONFIG_CLS.SEMANTIC_TOOL_ROUTING = True
    tools = asyncio.run(
        tool_router.get_tools_for_agent_async(
            agent_id=None,
            workspace_id=None,
            is_admin=True,
        )
    )
    enum = _enum_of_tool(_dispatcher_from(tools))
    assert "platform_list_agents" in enum
    assert "platform_get_workspace_info" in enum


def test_rank_actions_for_dispatcher_async_happy_path():
    async def _fake_rank(query, top_k, exclude_admin, exclude_promoted, include_super_admin=False, workspace_id=None, **kwargs):
        return [
            ("platform_list_agents", 0.91),
            ("platform_create_agent", 0.83),
        ]

    _dummy_index.rank_actions = _fake_rank
    names = asyncio.run(
        tool_router._rank_actions_for_dispatcher_async(
            query="list all agents",
            top_k=5,
            exclude_admin=True,
            exclude_promoted=True,
        )
    )
    assert names == ["platform_list_agents", "platform_create_agent"]
