"""PRD-170 S7 — one exec surface: the duplicate router is deleted (Q85).

Contract + reachability, DB-free (drives the same subprocess route-manifest
generator as PRD-155, pointed at an unreachable Postgres):

  * the never-mounted ``api/workspace_exec.py`` router MODULE is gone from the
    tree (deletion gate — a reintroduction fails here);
  * POST ``/api/workspaces/{workspace_id}/exec`` is served EXACTLY ONCE in the
    live app (the ``workspace_files.py`` surface). No second exec surface, no
    dead duplicate route.

This proves the Q85 resolution held: ``workspace_files`` POST /exec + the canvas
session shell cover execution; the terminal targets the session, not a separate
router.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ORCH_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ORCH_ROOT / "reports" / "route-manifest.json"
_EXEC_PATH = "/api/workspaces/{workspace_id}/exec"


def _run_dump() -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update(
        {
            "POSTGRES_USER": "test",
            "POSTGRES_PASSWORD": "test",
            "POSTGRES_HOST": "127.0.0.1",
            "POSTGRES_PORT": "59432",
            "POSTGRES_DB": "test",
            "DATABASE_URL": "postgresql://test:test@127.0.0.1:59432/test",
        }
    )
    return subprocess.run(
        [sys.executable, "-m", "scripts.dump_routes"],
        cwd=str(ORCH_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )


def test_workspace_exec_router_module_is_deleted():
    """The duplicate exec router file must not exist (Q85 — DELETE, not mount)."""
    dup = ORCH_ROOT / "api" / "workspace_exec.py"
    assert not dup.exists(), (
        "api/workspace_exec.py must be DELETED (Q85): the one exec surface is "
        "workspace_files.py POST /exec; the terminal uses the session shell."
    )


def test_workspace_exec_module_not_importable():
    """A stale import of the deleted router must fail (reachability guard)."""
    proc = subprocess.run(
        [sys.executable, "-c", "import api.workspace_exec"],
        cwd=str(ORCH_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode != 0, "api.workspace_exec should no longer be importable"


def test_exec_route_served_exactly_once():
    """POST /exec is served once — by workspace_files — with no dead duplicate."""
    proc = _run_dump()
    assert proc.returncode == 0, (
        f"dump_routes failed (rc={proc.returncode})\nSTDERR:\n{proc.stderr[-3000:]}"
    )
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    post_exec = [
        r for r in data["routes"]
        if r["path"] == _EXEC_PATH and r["method"] == "POST"
    ]
    assert len(post_exec) == 1, (
        f"POST {_EXEC_PATH} must be served exactly once, got {len(post_exec)}: "
        f"{post_exec}"
    )
