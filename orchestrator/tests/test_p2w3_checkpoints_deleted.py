"""PRD-200 S2 (P2-19) — the decorative checkpoint apparatus is deleted, not orphaned.

The session-checkpoint stack shipped fully (service, S3-blob dataclass, endpoint,
counter column) but did NO work at runtime: ``write_checkpoint`` had ZERO callers,
so ``GET /{id}/checkpoints`` always returned ``[]`` plus a never-incremented
``checkpoint_count``. Its per-verified-task S3-snapshot shape only ever resumes
from the last COMPLETED task — which the in-DB stall-recovery already does for
free — so wiring it bought almost nothing while advertising crash-recovery it
never performed. It is removed rather than wired (honest-OFF over silent placebo;
true in-flight resume is a distinct executor-touching build — PRD-200 Q1).

These tests prove the deletion is total (the import-regression shape PRD-185 S5
established):

1. The checkpoint service module is unimportable (deleted, not ``_legacy``-suffixed).
2. No live source file references the deleted checkpoint surface (no dangling imports).
3. The ``GET /{id}/checkpoints`` route is gone from the missions router and the
   committed route-manifest (the frontend route-contract lane reads it).
4. The DROP migration exists and chains off the current head.

Pure/static — file reads only. (The token set is deliberately specific: the bare
word "checkpoint" also names PRD-164's unrelated joiner checkpoint and skill-seed
config, which must survive.)
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

_ORCH = pathlib.Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

_SOURCE_DIRS = ("modules", "services", "core", "api", "consumers", "evals")

_GONE_MODULE = "services.checkpoint_service"

# Specific deleted symbols only — NOT the bare word "checkpoint" (PRD-164's
# joiner "checkpoint" and seed_skills "checkpoint_enabled" are unrelated live code).
_GONE_TOKENS = (
    "checkpoint_service",
    "SessionCheckpoint",
    "write_checkpoint",
    "list_checkpoints",
    "read_checkpoint",
    "checkpoint_count",
)


def _spec_is_gone(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is None
    except ModuleNotFoundError:
        # A missing PARENT package raises instead of returning None — equally gone.
        return True


def test_checkpoint_service_unimportable():
    assert _spec_is_gone(_GONE_MODULE), (
        "services.checkpoint_service must stay deleted (PRD-200 S2) — no "
        "backward-compat shim"
    )


def test_no_dangling_checkpoint_imports():
    offenders = []
    for d in _SOURCE_DIRS:
        root = _ORCH / d
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            text = path.read_text(errors="ignore")
            for token in _GONE_TOKENS:
                if token in text:
                    offenders.append(f"{path.relative_to(_ORCH)}: {token}")
    for extra in ("main.py", "config.py"):
        text = (_ORCH / extra).read_text(errors="ignore")
        for token in _GONE_TOKENS:
            if token in text:
                offenders.append(f"{extra}: {token}")
    assert not offenders, f"dangling checkpoint references: {offenders}"


def test_checkpoints_route_removed_from_missions_router():
    src = (_ORCH / "api" / "missions.py").read_text()
    assert "/checkpoints" not in src, (
        "GET /{mission_id}/checkpoints must be removed — it always returned an "
        "empty list (PRD-200 S2)"
    )
    assert "list_mission_checkpoints" not in src


def test_checkpoints_route_absent_from_committed_manifest():
    manifest = (_ORCH / "reports" / "route-manifest.json").read_text()
    assert "/checkpoints" not in manifest, (
        "route-manifest.json (read by the frontend route-contract lane) must "
        "not advertise the removed checkpoints route"
    )


def test_drop_migration_exists_and_chains_off_head():
    mig = _ORCH / "alembic" / "versions" / "prd200_s2_drop_checkpoint_count.py"
    assert mig.exists()
    src = mig.read_text()
    assert 'down_revision = "prd196_audit_logs_ws_created_idx"' in src
    assert "checkpoint_count" in src
    assert "orchestration_runs" in src
