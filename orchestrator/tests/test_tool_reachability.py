"""PRD-155 S4 — tool reachability: every registered action resolves a handler.

The route-contract net (S1/S2) protects frontend→backend HTTP calls. THIS suite
protects the *agent* tool surface: every action Auto is told it can call must
resolve to a live dispatch handler — no LLM, no DB, no network.

It kills the stale-tool-name drift class permanently. The canonical bug is
``workspace_file_read`` vs ``workspace_read_file``: a registry action whose name
drifted from its dispatch branch, so the LLM is offered a tool that always
returns "Unknown ... tool". Same class on the platform side: a
``platform_*`` action in the registry with no key in
``PlatformActionExecutor._handlers``.

Sources of truth (all introspected, never hardcoded so a future action/handler
is swept automatically):

  * the ACTION set  — the LIVE ``ActionRegistry`` (148 actions today:
    140 ``platform_*`` + 8 ``workspace_*``);
  * platform HANDLERS — keys of ``PlatformActionExecutor._handlers``
    (a plain dict resolved at dispatch via ``.get(action_name)``);
  * workspace HANDLERS — the ``tool_name == "..."`` branches inside
    ``exec_workspace.execute_workspace_action`` (an if/elif chain ending in
    "Unknown workspace tool"), extracted by AST so a renamed/removed branch
    surfaces as registry drift.

Idiom: the prd143 sweep's closed-port POSTGRES preamble + conftest real-module
restore. Nothing here touches a DB.
"""
from __future__ import annotations

import ast
import difflib
import inspect
import os
import sys
import types
import importlib.util as _ilu
from typing import Dict, List, Optional, Set, Tuple

import pytest

# --- Closed-port DB preamble (prd143 idiom) -------------------------------
# The modules.tools import chain fail-soft-connects to Postgres at import; point
# it at a dead port so it refuses instantly instead of hanging. CI exports real
# POSTGRES_* so these setdefaults no-op there. Nothing in this file uses a DB.
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")


def _camelot_unlocatable() -> bool:  # pragma: no cover - env-dependent
    try:
        return _ilu.find_spec("camelot") is None
    except ValueError:
        return False


if _camelot_unlocatable():  # pragma: no cover - env-dependent
    sys.modules.setdefault("camelot", types.ModuleType("camelot"))

# CI collection-order safety net: sibling tests collected earlier may have
# stubbed modules.*/consumers.* into sys.modules. Restore the real chain before
# importing the app modules below (no-op once conftest has run, which is always
# under pytest). See tests/conftest.py::_restore_real_app_modules.
import tests.conftest as _conftest  # noqa: E402

_conftest._restore_real_app_modules()

from modules.tools.discovery.action_registry import ActionRegistry  # noqa: E402
from modules.tools.discovery.platform_actions import register_all_actions  # noqa: E402
from modules.tools.discovery.platform_executor import PlatformActionExecutor  # noqa: E402
from modules.tools.execution import exec_workspace  # noqa: E402
from modules.tools.execution.unified_executor import UnifiedToolExecutor  # noqa: E402


# ===========================================================================
# Sources of truth — built at import time so parametrization can use them
# ===========================================================================


def _build_real_registry() -> ActionRegistry:
    """The REAL catalogue, registered directly (not the singleton) so
    ``_ensure_initialized`` cannot trigger a second full init, and so the
    enumerated set is independent of whatever a sibling test left in the
    global singleton."""
    reg = ActionRegistry()
    register_all_actions(reg)
    reg._initialized = True
    return reg


_REGISTRY = _build_real_registry()
_ACTION_NAMES: List[str] = sorted(a.name for a in _REGISTRY.get_all())


def _platform_handler_names() -> Set[str]:
    """Keys of ``PlatformActionExecutor._handlers`` — the dict dispatch consults
    via ``.get(action_name)``. ``__init__`` only builds the dict (no DB use), so
    a dummy db/workspace is safe."""
    return set(PlatformActionExecutor(db=None, workspace_id=None)._handlers.keys())


def _workspace_dispatch_names() -> Set[str]:
    """The ``tool_name == "..."`` (and ``in (...)``) literals inside
    ``execute_workspace_action`` — i.e. every workspace tool the if/elif chain
    actually dispatches. Extracted by AST so a renamed/deleted branch is caught
    as registry drift instead of silently passing."""
    src = inspect.getsource(exec_workspace.execute_workspace_action)
    tree = ast.parse(src.lstrip())
    names: Set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Name)
            and node.left.id == "tool_name"
        ):
            for comp in node.comparators:
                if isinstance(comp, ast.Constant) and isinstance(comp.value, str):
                    names.add(comp.value)
                elif isinstance(comp, (ast.Tuple, ast.List)):
                    for elt in comp.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            names.add(elt.value)
    return names


def _reachable_handlers() -> Set[str]:
    """Union of every name the dispatch layer can resolve a handler for."""
    return _platform_handler_names() | _workspace_dispatch_names()


# ===========================================================================
# Drift detector (pure — fed the live sets here, fabricated sets in the
# negative test, so the net is proven to bite without mutating global state)
# ===========================================================================


def find_unreachable(
    action_names: List[str], reachable: Set[str]
) -> List[Tuple[str, Optional[str]]]:
    """Return ``(action, nearest_reachable_name | None)`` for every action that
    has NO dispatch handler. The suggestion is a difflib near-match against the
    reachable set — that is what turns ``workspace_file_read`` into a pointer at
    ``workspace_read_file``."""
    out: List[Tuple[str, Optional[str]]] = []
    pool = sorted(reachable)
    for name in sorted(action_names):
        if name not in reachable:
            close = difflib.get_close_matches(name, pool, n=1, cutoff=0.6)
            out.append((name, close[0] if close else None))
    return out


def format_drift(unreachable: List[Tuple[str, Optional[str]]]) -> str:
    lines = [
        "Registry actions with NO reachable dispatch handler "
        "(route-contract drift — the LLM is offered a tool that always 404s):"
    ]
    for name, suggestion in unreachable:
        hint = f"  — did you mean '{suggestion}'?" if suggestion else ""
        lines.append(f"  - {name}{hint}")
    return "\n".join(lines)


# ===========================================================================
# Tests
# ===========================================================================


def test_registry_is_non_vacuous():
    """Anchor: a registry that silently returns [] must not let the reachability
    test pass trivially. 148 actions today; floor well below that catches a
    collapse without churning on every new action."""
    assert len(_ACTION_NAMES) >= 140, (
        f"Live registry has only {len(_ACTION_NAMES)} actions — expected ≥140. "
        "Registry init likely broke; the reachability assertion would be vacuous."
    )
    # Namespace invariant the resolver relies on: every action is platform_* or
    # workspace_*. A third prefix would route to neither dispatch path.
    stray = [
        n
        for n in _ACTION_NAMES
        if not n.startswith("platform_") and not n.startswith("workspace_")
    ]
    assert not stray, f"Actions with no platform_/workspace_ prefix: {stray}"


def test_every_registry_action_resolves_a_handler():
    """THE invariant: every action in the live registry resolves a dispatch
    handler. Green on the chain tip; goes red the moment an action's name drifts
    from its handler (the stale-tool-name class)."""
    reachable = _reachable_handlers()
    unreachable = find_unreachable(_ACTION_NAMES, reachable)
    assert not unreachable, format_drift(unreachable)


def test_injected_stale_name_fails_with_diff_message():
    """The net bites: a fabricated stale name (the canonical ``workspace_file_read``
    drift, plus a platform typo) is flagged AND points at the canonical name via
    a difflib suggestion. Proven without touching the global registry."""
    reachable = _reachable_handlers()
    fabricated = ["workspace_file_read", "platform_lst_agents"]
    unreachable = find_unreachable(_ACTION_NAMES + fabricated, reachable)

    flagged = {name: suggestion for name, suggestion in unreachable}
    assert "workspace_file_read" in flagged, "stale workspace name not detected"
    assert "platform_lst_agents" in flagged, "stale platform name not detected"
    # The suggestions are the real canonical names — the diff is actionable.
    assert flagged["workspace_file_read"] == "workspace_read_file"
    assert flagged["platform_lst_agents"] == "platform_list_agents"

    message = format_drift(unreachable)
    assert "did you mean" in message
    assert "workspace_read_file" in message


def test_workspace_actions_all_dispatch():
    """Focused workspace-family check (the family the canonical drift lives in):
    every ``workspace_*`` registry action has an if/elif branch in
    ``execute_workspace_action``."""
    ws_actions = [n for n in _ACTION_NAMES if n.startswith("workspace_")]
    dispatch = _workspace_dispatch_names()
    missing = sorted(n for n in ws_actions if n not in dispatch)
    assert not missing, (
        "workspace_* registry actions with no dispatch branch in "
        f"execute_workspace_action: {missing} (have branches for {sorted(dispatch)})"
    )


def test_platform_actions_all_have_handlers():
    """Focused platform-family check: every ``platform_*`` registry action has a
    key in ``PlatformActionExecutor._handlers``."""
    plat_actions = [n for n in _ACTION_NAMES if n.startswith("platform_")]
    handlers = _platform_handler_names()
    missing = sorted(n for n in plat_actions if n not in handlers)
    assert not missing, (
        "platform_* registry actions with no key in "
        f"PlatformActionExecutor._handlers: {missing}"
    )


def test_unified_tool_routes_are_callable_coroutines():
    """The built-in (non-registry) tool surface — ``UnifiedToolExecutor.tool_routes``
    — maps each tool to a bound dispatch method. Guards against a future entry
    wired to a non-callable or sync function (every executor here is async).
    ``__init__`` is DB-free, so a dummy session is safe."""
    execu = UnifiedToolExecutor(db_session=None)
    assert execu.tool_routes, "tool_routes is empty — built-in dispatch lost"
    bad = [
        name
        for name, fn in execu.tool_routes.items()
        if not (callable(fn) and inspect.iscoroutinefunction(fn))
    ]
    assert not bad, f"tool_routes entries not async-callable: {bad}"
