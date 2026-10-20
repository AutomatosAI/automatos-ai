"""The HARNESS HTTP API (``api/harness.py``): the Self-Learning tile's endpoint only.

PRD-263 US-047 deleted the approve and reject routes for queued HARNESS
prescriptions (maintainer issue #1149) with ``api/harness_commands.py``; their
cases went with them, mapped in the PR to ``test_prd263_harness_grant_flow.py``
(US-054), and the principal-resolver cases moved to
``test_auth_actor_resolver.py``, where ``core.auth.actor`` defines the resolver.
Dummy POSTGRES_* and the apscheduler stub let the harness_service import chain
load without a real DB or the prod-only scheduler dependency.
"""
import os
import sys
import types
from uuid import UUID

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_DB", "test")


def _install_fake_apscheduler():
    if "apscheduler" in sys.modules:
        return
    aps = types.ModuleType("apscheduler")
    schedulers = types.ModuleType("apscheduler.schedulers")
    asyncio_mod = types.ModuleType("apscheduler.schedulers.asyncio")
    asyncio_mod.AsyncIOScheduler = type("AsyncIOScheduler", (), {})
    jobstores = types.ModuleType("apscheduler.jobstores")
    memory_mod = types.ModuleType("apscheduler.jobstores.memory")
    memory_mod.MemoryJobStore = type("MemoryJobStore", (), {})
    aps.schedulers = schedulers
    aps.jobstores = jobstores
    schedulers.asyncio = asyncio_mod
    jobstores.memory = memory_mod
    sys.modules.update({
        "apscheduler": aps,
        "apscheduler.schedulers": schedulers,
        "apscheduler.schedulers.asyncio": asyncio_mod,
        "apscheduler.jobstores": jobstores,
        "apscheduler.jobstores.memory": memory_mod,
    })


_install_fake_apscheduler()

import api.harness as h  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402

_WS_ID = UUID("00000000-0000-0000-0000-000000000001")


def _ctx(*, clerk=None, email=None):
    return RequestContext(
        workspace_id=_WS_ID,
        user=UserContext(id=clerk or email, email=email, clerk_user_id=clerk),
    )


# --- PRD-263 US-047: the approve and reject routes are gone ------------------

def test_the_router_serves_only_the_self_learning_tile():
    """No HARNESS route applies or rejects a change: a change applies only on the
    owner's click on its Approvals card (US-047, #1149)."""
    served = {(method, route.path) for route in h.router.routes for method in route.methods}
    assert served == {("GET", "/api/harness/self-learning")}


# --- PRD-263 US-021: self-learning health endpoint ---------------------------
# The endpoint is a plain def, so these call it directly (no event loop). The real-DB
# reads (harness_runs, the workspace-scoped signal count) are covered by
# test_prd263_self_learning_endpoint.py; here the three reads are faked.

class _RollbackDB:
    """The endpoint touches the session itself to read the workspace's settings and
    to roll back a failed section."""

    def __init__(self, settings=None):
        self._settings = settings
        self.rollbacks = 0

    def get(self, _model, _id):
        return types.SimpleNamespace(settings=self._settings)

    def rollback(self):
        self.rollbacks += 1


def test_self_learning_health_combines_sections(monkeypatch):
    import services.harness_runs as runs
    import services.harness_service as hsvc

    status = {"status": "completed", "iteration_count": 3, "coverage": 0.75, "convergence": "watching",
              "cur_window": "2026-W42", "active_agents": 4, "artifacts": {"audit_report": "ok"}}
    latest = types.SimpleNamespace(
        iteration=3, coverage=0.75, verdicts={"convergence": "watching", "agents": {}},
        counts={"active_agents": 4, "judged_agents": 3, "insufficient_data": 1, "watch": 1, "regression": 0,
                "prescriptions": 2})
    monkeypatch.setattr(hsvc, "get_harness_service",
                        lambda: types.SimpleNamespace(get_status=lambda ws, db=None: dict(status)))
    monkeypatch.setattr(runs, "latest_run", lambda db, ws: latest)
    monkeypatch.setattr(h, "_signals_7d", lambda db, ws: 12)
    # A mode stored before PRD-263 reads through harness_mode(): full_auto is report_only.
    db = _RollbackDB(settings={"orchestrator": {"harness": {"mode": "full_auto"}}})

    out = h.self_learning_health(ctx=_ctx(clerk="admin"), db=db)

    assert out == {
        "workspace_id": str(_WS_ID), "status": "completed", "iteration_count": 3, "coverage": 0.75,
        "insufficient_data": 1, "watch": 1, "regression": 0, "mode": "report_only", "signals_7d": 12,
        "unavailable": [],
    }
    # Counts, coverage and mode only (#1148): no prescriptions, report, artifacts or
    # process-global tool-routing stats.
    assert not {"prescriptions", "harness", "tool_routing", "artifacts", "convergence"} & set(out)
    assert db.rollbacks == 0


def test_self_learning_health_degrades_gracefully(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("section down: secret-internal-detail")

    monkeypatch.setattr(h, "_harness_fields", _boom)
    monkeypatch.setattr(h, "_signals_7d", _boom)
    db = _RollbackDB()

    # Each section degrades to None figures and is named in `unavailable`; the endpoint
    # never 500s, never answers 0 for a figure it could not read, and leaks no error text.
    out = h.self_learning_health(ctx=_ctx(clerk="admin"), db=db)

    assert out["workspace_id"] == str(_WS_ID)
    assert all(out[field] is None for field in h.HARNESS_FIELDS + h.SIGNALS_FIELDS)
    assert out["unavailable"] == [h.SECTION_HARNESS, h.SECTION_SIGNALS]
    assert "secret-internal-detail" not in repr(out)
    assert db.rollbacks == 2
