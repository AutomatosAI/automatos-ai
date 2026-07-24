"""PRD-184 US-004 — the dead tool-runtime concurrency helper + ToolService are deleted.

DEFINITIVE (acceptance gate): ``modules/tools/execution/concurrency.py`` (~166 LOC)
exported ``is_read_safe`` + ``partition_tool_batch`` and had ZERO callers — no
source file imported the module path and neither symbol appeared anywhere else in
the tree. It is gone.

CONDITIONAL (grep-proven dead → deleted): ``modules/tools/service.py`` ``ToolService``
/ ``ToolServiceConfig``. Its only references were the owning barrel
``modules/tools/__init__.py`` (one live ``from .service import`` line + two
``__all__`` entries + a docstring ``Usage:`` example) and a docstring bullet in
``modules/__init__.py``. Live tool execution uses the plural-package
``ComposioToolService`` / ``ComposioToolRouter`` — NOT this singular ToolService.
The barrel is trimmed in the same commit so the symbol cannot silently return.

BOUNDARY (must survive): ``modules/tools/composio_tool_router.py`` is LIVE
(``get_tool_router_for_agent`` is imported by ``composio_router_executor.py`` via
``exec_composio`` → ``unified_executor``). It has NO dead ``db_session`` delegate
method to excise — every method is reachable from the live factory — so the file
is left untouched.

Pure/static — file reads only, imports no app package.
"""
from __future__ import annotations

import pathlib
import sys

_ORCH = pathlib.Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

_CONCURRENCY = _ORCH / "modules" / "tools" / "execution" / "concurrency.py"
_SERVICE = _ORCH / "modules" / "tools" / "service.py"
_TOOLS_BARREL = _ORCH / "modules" / "tools" / "__init__.py"
_ROUTER = _ORCH / "modules" / "tools" / "composio_tool_router.py"

_SCAN_DIRS = ("modules", "services", "core", "api", "consumers", "evals")

# Unambiguous concurrency-only tokens — the module path and its two exported
# symbols. All three were grep-proven to have ZERO hits tree-wide before the cut
# (``services.concurrency_guard`` / ``starlette.concurrency`` are unrelated and do
# NOT contain these substrings).
_CONCURRENCY_TOKENS = (
    "tools.execution.concurrency",
    "from .concurrency",
    "is_read_safe",
    "partition_tool_batch",
)


def _py_files():
    for d in _SCAN_DIRS:
        root = _ORCH / d
        if not root.exists():
            continue
        yield from root.rglob("*.py")


def test_no_tools_concurrency_import():
    """concurrency.py is gone and no source references its path or symbols."""
    assert not _CONCURRENCY.exists(), (
        "modules/tools/execution/concurrency.py must stay deleted (PRD-184 US-004)"
    )
    offenders = {}
    for path in _py_files():
        text = path.read_text(errors="ignore")
        hits = [t for t in _CONCURRENCY_TOKENS if t in text]
        if hits:
            offenders[str(path.relative_to(_ORCH))] = hits
    assert not offenders, f"dangling concurrency helper references: {offenders}"


def test_tool_service_deleted_and_barrel_trimmed():
    """ToolService/service.py is gone and the owning barrel no longer wires it."""
    assert not _SERVICE.exists(), (
        "modules/tools/service.py (ToolService) must stay deleted (PRD-184 US-004)"
    )
    barrel = _TOOLS_BARREL.read_text()
    assert "from .service import" not in barrel, (
        "modules/tools/__init__.py must not re-import the deleted .service module"
    )
    assert "ToolService" not in barrel, (
        "modules/tools/__init__.py must not reference the deleted ToolService symbol"
    )
    # No source file anywhere imports the deleted singular ToolService.
    offenders = []
    for path in _py_files():
        if "from modules.tools import ToolService" in path.read_text(errors="ignore"):
            offenders.append(str(path.relative_to(_ORCH)))
    assert not offenders, f"dangling ToolService importers: {offenders}"


def test_composio_tool_router_stays_live():
    """Boundary proof: the LIVE Composio router file is untouched (no dead delegate
    was excised because none exists — every method is reachable from the factory)."""
    assert _ROUTER.exists(), "composio_tool_router.py is live and must survive"
    router = _ROUTER.read_text()
    for keep in (
        "def get_tool_router_for_agent(",
        "class ComposioToolRouter",
        "def search_tools(",
        "def execute_tool(",
    ):
        assert keep in router, f"live Composio router surface must survive: {keep}"
