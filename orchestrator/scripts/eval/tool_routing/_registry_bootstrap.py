"""
Lightweight bootstrap for ActionRegistry that bypasses the heavy
`modules.tools.__init__` package chain (which transitively imports
DB, memory, executor modules — none of which the eval needs).

The registrar files (`actions_*.py`) only register `ActionDefinition`
dataclass instances — they don't touch the DB. By short-circuiting
the parent package's `__init__.py` files with empty stub modules,
relative imports inside the registrar files still resolve, but no
heavy side-effects fire.

Used by `seed_eval_set.py` and `run_eval.py`.
"""

from __future__ import annotations

import sys
import types
from typing import Any


_STUBBED_PACKAGES = (
    "modules",
    "modules.tools",
    "modules.tools.discovery",
)


def _stub_package(name: str) -> None:
    """Insert an empty package module into sys.modules if not already present."""
    if name in sys.modules:
        return
    pkg = types.ModuleType(name)
    pkg.__path__ = []  # mark as package so submodule imports work
    sys.modules[name] = pkg


def load_registry() -> Any:
    """
    Stub the heavy parent packages, then import action_registry directly.

    Returns the initialized singleton with all ActionDefinitions registered.
    """
    # Resolve the discovery dir on disk so we can attach __path__ correctly.
    import pathlib

    here = pathlib.Path(__file__).resolve()
    orchestrator_root = here.parents[3]  # scripts/eval/tool_routing/_X.py → orchestrator/
    discovery_dir = orchestrator_root / "modules" / "tools" / "discovery"
    if not discovery_dir.is_dir():
        raise RuntimeError(f"Could not find discovery dir: {discovery_dir}")

    for name in _STUBBED_PACKAGES:
        _stub_package(name)

    # Point the stubbed packages at the real on-disk dirs so submodule
    # imports (`from .actions_X import ...`) can find the source files.
    sys.modules["modules.tools.discovery"].__path__ = [str(discovery_dir)]
    sys.modules["modules.tools"].__path__ = [str(discovery_dir.parent)]
    sys.modules["modules"].__path__ = [str(discovery_dir.parent.parent)]

    # Now the real import works without triggering modules/tools/__init__.py
    # because we already populated sys.modules with stub modules.
    import importlib

    action_registry_mod = importlib.import_module("modules.tools.discovery.action_registry")
    return action_registry_mod.get_action_registry()
