"""PRD-155 S1 — backend route manifest generator: deterministic and DB-free.

This file imports NO app modules at module level — it drives the generator as a
subprocess — so it needs no collection-order ``_sys_guard`` block. The subprocess
runs in a clean interpreter pointed at an UNREACHABLE Postgres (closed port
59432, the blessed fake-POSTGRES preamble), which proves the manifest generates
with no database available. ``create_engine`` is lazy and the lifespan that
calls ``init_database`` runs only when the app serves, so ``import main`` never
dials the database. This is exactly how CI and ``acceptance-prd155.sh`` invoke
it: ``python3 -m scripts.dump_routes``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ORCH_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ORCH_ROOT / "reports" / "route-manifest.json"

# A path the app must always serve — a non-vacuity anchor so a manifest that
# silently collapses to "[]" cannot pass.
_ANCHOR_PATH = "/api/agents/"


def _run_dump(extra_env: dict | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    # Unreachable Postgres: a DB connection attempt would refuse instantly
    # rather than hang. Generation that succeeds here is DB-free by definition.
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
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-m", "scripts.dump_routes"],
        cwd=str(ORCH_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )


def _load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_dump_routes_generates_without_db():
    proc = _run_dump()
    assert proc.returncode == 0, (
        f"dump_routes failed (rc={proc.returncode})\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr[-3000:]}"
    )
    assert MANIFEST_PATH.exists(), "manifest file was not written"
    data = _load_manifest()
    routes = data["routes"]
    assert routes, "manifest produced zero routes"
    assert data["route_count"] == len(routes)
    paths = {r["path"] for r in routes}
    assert _ANCHOR_PATH in paths, f"{_ANCHOR_PATH} missing from manifest"


def test_dump_routes_deterministic_two_runs():
    p1 = _run_dump()
    assert p1.returncode == 0, p1.stderr[-3000:]
    first = MANIFEST_PATH.read_text(encoding="utf-8")
    p2 = _run_dump()
    assert p2.returncode == 0, p2.stderr[-3000:]
    second = MANIFEST_PATH.read_text(encoding="utf-8")
    assert first == second, "manifest is not byte-identical across two runs"


def test_manifest_is_sorted_and_well_shaped():
    proc = _run_dump()
    assert proc.returncode == 0, proc.stderr[-3000:]
    routes = _load_manifest()["routes"]
    for r in routes:
        assert {"method", "path"} <= set(r), f"route entry missing keys: {r}"
        assert r["path"].startswith("/"), f"path not absolute: {r}"
    ordering = [(r["path"], r["method"]) for r in routes]
    assert ordering == sorted(ordering), "routes are not sorted by (path, method)"
