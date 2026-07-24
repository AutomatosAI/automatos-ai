"""PRD-184 US-003 — the exec_planning stub vertical is deleted AND de-routed.

``modules/tools/execution/exec_planning.py`` (~339 LOC) was 3 template-writer
executors (planning/writing/analysis) fronting 10 hardcoded "tool" names that
just wrote canned Markdown to the workspace — 0 LLM, exposed to NO agent. The
dispatch dict in ``unified_executor.py`` was the ONLY thing that named them
(PRD-22 Expansion block); no agent toolset, registry, or skill referenced any of
the 10 names (``security_scan`` / ``run_tests`` / ``analyze_data`` hits were all
substrings of unrelated live code — ``plugin_security_scanner``, a playbook
``step_id``, ``analyze_database``).

Deletion + de-route ship together: the module is gone, and its import + dispatch
entries + 3 handler methods are removed from ``unified_executor.py`` — so the
dead tool names cannot be dispatched.

Pure/static — file reads only, imports no app package.
"""
from __future__ import annotations

import pathlib
import sys

_ORCH = pathlib.Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

_EXEC_PLANNING = _ORCH / "modules" / "tools" / "execution" / "exec_planning.py"
_UNIFIED = _ORCH / "modules" / "tools" / "execution" / "unified_executor.py"

# Unambiguous exec_planning-only tokens (these appear NOWHERE else as live code;
# the generic tool words security_scan/run_tests/analyze_data are intentionally
# NOT asserted repo-wide because they substring unrelated live subsystems).
_UNROUTED_TOKENS = (
    "exec_planning",
    "_execute_planning_tool",
    "_execute_writing_tool",
    "_execute_analysis_tool",
    "create_implementation_plan",
    "write_technical_content",
    "'refine_content'",
)


def test_exec_planning_deleted_and_unrouted():
    # 1. the module file is gone (no _legacy shim)
    assert not _EXEC_PLANNING.exists(), (
        "modules/tools/execution/exec_planning.py must stay deleted (PRD-184 US-003)"
    )
    # 2. unified_executor.py no longer imports, dispatches to, or defines the
    #    exec_planning wiring
    unified = _UNIFIED.read_text()
    present = [t for t in _UNROUTED_TOKENS if t in unified]
    assert not present, (
        f"unified_executor.py still wires the deleted exec_planning surface: {present}"
    )


def test_exec_planning_module_gone_tree_wide():
    """No source file anywhere imports the deleted exec_planning module."""
    offenders = []
    for d in ("modules", "services", "core", "api", "consumers", "evals"):
        root = _ORCH / d
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if "exec_planning" in path.read_text(errors="ignore"):
                offenders.append(str(path.relative_to(_ORCH)))
    assert not offenders, f"dangling exec_planning references: {offenders}"


def test_live_dispatch_neighbours_survive():
    """Boundary proof: only the exec_planning entries were cut — the sibling
    document/composio/widget dispatch entries in the same dict are intact."""
    unified = _UNIFIED.read_text()
    for keep in (
        "'create_pdf': self._execute_document_tool",
        "'composio_execute': self._execute_composio_execute",
        "'generate_document': self._execute_generate_document",
        "'widget_open_callback_form': self._execute_widget_callback",
    ):
        assert keep in unified, f"unrelated live dispatch entry must survive: {keep}"
