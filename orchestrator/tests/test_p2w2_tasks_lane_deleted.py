"""PRD-192 S6 (P2-11) — the `/api/tasks` direct-step lane is DELETED.

Gerard's locked #5: the unattended shell/git ingress (`submit_task` enqueued
concrete steps to the worker with auth-only deps — no gate, no budget, no
audit) had ZERO product callers (grep-proven across backend, frontend, SDK,
tests), so the lane is removed rather than gated. P2-25's overlapping
sub-item ("/api/tasks under the policy gate") retires by removal.

Pins:
- the router file is gone and main.py no longer imports/registers it;
- the COMMITTED route-manifest carries no /api/tasks routes and its count is
  consistent (the frontend route-contract CI reads this file);
- the dead frontend api-client methods went with it;
- the workspace worker + its exec sandbox STAY (the github-clone lane still
  enqueues background jobs) — this story governed admission, not execution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ORCH = Path(__file__).resolve().parents[1]
_REPO = _ORCH.parent
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))


def test_tasks_router_file_deleted():
    assert not (_ORCH / "api" / "tasks.py").exists(), (
        "/api/tasks router resurfaced — PRD-192 S6 deleted the ungoverned "
        "direct-step ingress (locked #5); gate it through PolicyGate if it "
        "ever returns"
    )


def test_main_no_longer_registers_tasks_router():
    main_src = (_ORCH / "main.py").read_text()
    assert "from api.tasks import" not in main_src
    assert "tasks_router" not in main_src


def test_route_manifest_has_no_tasks_routes_and_consistent_count():
    """The frontend route-contract CI reads the COMMITTED manifest — the
    deleted routes must be gone from it and the count must match."""
    manifest = json.loads((_ORCH / "reports" / "route-manifest.json").read_text())
    paths = [r["path"] for r in manifest["routes"]]
    offenders = [
        p for p in paths if p == "/api/tasks" or p.startswith("/api/tasks/")
    ]
    assert offenders == [], f"/api/tasks routes still in the manifest: {offenders}"
    assert manifest["route_count"] == len(manifest["routes"])


def test_frontend_client_has_no_tasks_calls():
    api_client = _REPO / "frontend" / "lib" / "api-client.ts"
    if not api_client.exists():  # frontend not present in a trimmed checkout
        return
    src = api_client.read_text()
    assert "/api/tasks" not in src
    for dead in (
        "submitWorkspaceTask",
        "listWorkspaceTasks",
        "getWorkspaceTask(",
        "cancelWorkspaceTask",
    ):
        assert dead not in src, f"dead client method resurfaced: {dead}"


def test_worker_exec_sandbox_stays():
    """Deleting the INGRESS must not touch the worker: the github-clone lane
    still enqueues background jobs and the worker sandbox remains."""
    github_src = (_ORCH / "api" / "workspace_github.py").read_text()
    assert "workspace:tasks:normal" in github_src  # its own enqueue lane stays
    assert "/api/tasks" not in github_src          # but no dangling events_url

    worker_main = _REPO / "services" / "workspace-worker" / "main.py"
    if worker_main.exists():  # repo-root services/ present in full checkouts
        assert "background_job" in worker_main.read_text(errors="ignore")
