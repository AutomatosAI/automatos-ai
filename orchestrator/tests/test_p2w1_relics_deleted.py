"""PRD-187 S5 — the dead rival memory stacks are deleted, not orphaned.

Two stacks were mounted and serving nothing: ``AdvancedMemoryManager``
(687 lines behind the /api/v1/memory router — F039's cross-tenant-shaped
relic) and ``MemoryKnowledgeSystem`` (memory_items / knowledge_nodes /
knowledge_edges / learning_outcomes / harness_prescriptions — all 0 rows in
prod, lifetime). These tests prove the deletion is total:

1. The relic modules are unimportable (deleted, not ``_legacy``-suffixed).
2. No live source file still references them (no dangling imports — the
   import-regression shape PRD-185 S5 established).
3. The DROP migration exists and chains off the single head.
4. The consumers were REPOINTED, not deleted: memory stats + the workspace
   knowledge section now read the real L2 store.

Pure/static — file reads only.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

_ORCH = pathlib.Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

_SOURCE_DIRS = ("modules", "services", "core", "api", "consumers", "evals")

_RELIC_MODULES = (
    "api.memory",
    "modules.memory.storage",
    "modules.memory.storage.manager",
    "modules.memory.storage.knowledge_system",
    "modules.memory.types",
    "modules.memory.operations.augmentation",
    "modules.memory.operations.access_patterns",
    "modules.memory.operations.consolidation",
    "modules.memory.operations.execution_history",
    "modules.memory.operations.prompt_injection",
)

_RELIC_TOKENS = (
    "AdvancedMemoryManager",
    "MemoryKnowledgeSystem",
    "HarnessPrescription",
    "modules.memory.storage",
    "modules.memory.types",
    "HierarchicalMemoryManager",
)


def _spec_is_gone(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is None
    except ModuleNotFoundError:
        # a missing PARENT package raises instead of returning None — equally gone
        return True


def test_removed_memory_relics_unimportable():
    for mod in _RELIC_MODULES:
        assert _spec_is_gone(mod), (
            f"{mod} must stay deleted (PRD-187 S5) — no backward-compat shims"
        )


def test_no_dangling_relic_imports():
    offenders = []
    for d in _SOURCE_DIRS:
        root = _ORCH / d
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            text = path.read_text(errors="ignore")
            for token in _RELIC_TOKENS:
                if token in text:
                    offenders.append(f"{path.relative_to(_ORCH)}: {token}")
    for extra in ("main.py", "config.py"):
        text = (_ORCH / extra).read_text(errors="ignore")
        for token in _RELIC_TOKENS:
            if token in text:
                offenders.append(f"{extra}: {token}")
    assert not offenders, f"dangling relic references: {offenders}"


def test_live_operations_survive():
    # The one live operation (PRD-159 S4 consolidation) must NOT have been
    # swept up with the manager's satellites.
    assert importlib.util.find_spec("modules.memory.operations.contradiction") is not None


def test_drop_migration_exists_and_chains_off_head():
    mig = _ORCH / "alembic" / "versions" / "prd187_s5_drop_memory_relics.py"
    assert mig.exists()
    src = mig.read_text()
    assert 'down_revision = "prd185_s7_msg_retrieval_ctx"' in src
    for table in (
        "memory_items", "knowledge_nodes", "knowledge_edges",
        "learning_outcomes", "harness_prescriptions",
    ):
        assert table in src, f"DROP migration must cover {table}"


def test_consumers_repointed_to_real_store():
    stats = (_ORCH / "api" / "memory_stats.py").read_text()
    assert "MemoryShortTerm" in stats and "MemoryItem" not in stats
    workspaces = (_ORCH / "api" / "workspaces.py").read_text()
    assert "MemoryShortTerm" in workspaces and "knowledge_system" not in workspaces
