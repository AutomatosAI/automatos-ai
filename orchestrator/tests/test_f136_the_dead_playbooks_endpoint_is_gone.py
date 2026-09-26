"""F136 (night 4, B1) — the mined-playbooks endpoint is retired, not left broken.

GET /api/playbooks answered 500 for every workspace. It passed a bare SQL string
to db.execute, which SQLAlchemy 2.0 refuses, and the table it read, `playbooks`,
had been dropped by prd135_drop_bucket_6: its raw-SQL readers were invisible to
the dead-code scan. POST /api/playbooks/mine wrote to the same table. Gerard,
2026-09-25: delete the endpoint and the miner (the empty marketplace is a content
item). test_prd184_us001 guards the files; this guards the served surface.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ORCH = Path(__file__).resolve().parents[1]
RETIRED = {("GET", "/api/playbooks"), ("POST", "/api/playbooks/mine")}


def test_the_route_manifest_no_longer_serves_them():
    manifest = json.loads((ORCH / "reports" / "route-manifest.json").read_text())
    served = {(r.get("method"), r["path"]) for r in manifest["routes"]}
    assert not RETIRED & served
    assert manifest["route_count"] == len(manifest["routes"])


def test_no_frontend_call_reaches_for_them():
    calls = re.compile(r"/api/playbooks(?:[\"'`?/]|$)")
    frontend = ORCH.parent / "frontend"
    callers = [
        str(path.relative_to(frontend))
        for folder in ("app", "components", "hooks", "lib")
        for path in (frontend / folder).rglob("*.ts*")
        if "node_modules" not in path.parts and calls.search(path.read_text(errors="ignore"))
    ]
    assert callers == []
