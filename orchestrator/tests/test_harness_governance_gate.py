"""PRD-142 Wave 4 (§12.3): autonomy → HARNESS auto-apply ceiling coupling.

A workspace at autonomy=full gets the higher auto-apply risk ceiling
(HARNESS_AUTO_APPLY_MAX_RISK_FULL, default 3); everyone else gets the standard
ceiling (default 2). Both ceilings are Railway-overridable config. These tests
pin: (1) `_auto_apply_max_risk` selects full vs standard via auto_autonomy, and
(2) the ceiling actually flips a risk-3 prescription between auto-apply and queue
in `_phase_apply`.

Dummy POSTGRES_* + the apscheduler stub let the harness_service import chain load.
"""
import asyncio
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

from config import config  # noqa: E402
from services.harness_service import get_harness_service  # noqa: E402

_WS = UUID("00000000-0000-0000-0000-000000000001")


# --- the ceiling selector --------------------------------------------------

def test_auto_apply_max_risk_full_vs_standard(monkeypatch):
    import core.services.auto_autonomy as aa
    svc = get_harness_service()

    monkeypatch.setattr(aa, "is_full_autonomy", lambda db, ws: True)
    assert svc._auto_apply_max_risk(None, _WS) == config.HARNESS_AUTO_APPLY_MAX_RISK_FULL

    monkeypatch.setattr(aa, "is_full_autonomy", lambda db, ws: False)
    assert svc._auto_apply_max_risk(None, _WS) == config.HARNESS_AUTO_APPLY_MAX_RISK_STANDARD


def test_auto_apply_max_risk_default_full_is_higher_than_standard():
    # The shipped defaults must actually widen at full, else the coupling is a no-op.
    assert config.HARNESS_AUTO_APPLY_MAX_RISK_FULL > config.HARNESS_AUTO_APPLY_MAX_RISK_STANDARD


# --- the ceiling flips auto-apply vs queue in _phase_apply -----------------

def _patch_phase_apply_side_effects(monkeypatch, svc, applied, executed):
    async def fake_auto_apply(executor, rx):
        applied.append(rx["prescription_id"])
        return {"success": True}

    async def fake_escalate(db, ws, rx, changelog):
        return None

    async def fake_apply_approved(executor, ws, changelog):
        return None

    monkeypatch.setattr(svc, "_auto_apply_prescription", fake_auto_apply)
    monkeypatch.setattr(svc, "_snapshot_current_value", lambda rx: {})
    monkeypatch.setattr(svc, "_maybe_escalate", fake_escalate)
    monkeypatch.setattr(svc, "_apply_approved_board_tasks", fake_apply_approved)

    class _FakeExec:
        def __init__(self, *a, **k):
            pass

        async def execute(self, name, params):
            executed.append(name)
            return {"success": True, "data": {"id": 1}}

    import modules.tools.discovery.platform_executor as pe
    monkeypatch.setattr(pe, "PlatformActionExecutor", _FakeExec)


def _rx3():
    return {"prescription_id": "rx-3", "risk_score": 3, "change_type": "tag_update", "target_name": "A"}


def test_full_ceiling_auto_applies_risk_3(monkeypatch):
    svc = get_harness_service()
    applied, executed = [], []
    _patch_phase_apply_side_effects(monkeypatch, svc, applied, executed)

    asyncio.run(svc._phase_apply(_WS, [_rx3()], None, allow_auto_apply=True, max_risk=3))

    assert applied == ["rx-3"]                       # auto-applied
    assert "platform_create_task" not in executed    # not queued


def test_standard_ceiling_queues_risk_3(monkeypatch):
    svc = get_harness_service()
    applied, executed = [], []
    _patch_phase_apply_side_effects(monkeypatch, svc, applied, executed)

    asyncio.run(svc._phase_apply(_WS, [_rx3()], None, allow_auto_apply=True, max_risk=2))

    assert applied == []                             # NOT auto-applied
    assert "platform_create_task" in executed        # queued for review


def test_default_max_risk_uses_standard_ceiling(monkeypatch):
    # A direct caller that omits max_risk falls back to the standard config ceiling
    # (2 by default) — so a risk-3 rx queues, never silently auto-applies.
    svc = get_harness_service()
    applied, executed = [], []
    _patch_phase_apply_side_effects(monkeypatch, svc, applied, executed)

    asyncio.run(svc._phase_apply(_WS, [_rx3()], None, allow_auto_apply=True))

    assert applied == []
    assert "platform_create_task" in executed


# --- HIGH-2: the setter records who flipped the dial + from what -----------

def test_set_autonomy_level_audits_previous_level(monkeypatch):
    import core.services.auto_autonomy as aa
    from modules.tools.discovery.handlers_autonomy import set_autonomy_level

    monkeypatch.setattr(aa, "get_autonomy_level", lambda db, ws: "standard")
    monkeypatch.setattr(aa, "set_autonomy_level", lambda db, ws, level: {"level": level})

    class _DB:
        def commit(self):
            pass

        def rollback(self):
            pass

    out = asyncio.run(set_autonomy_level(_DB(), _WS, {"level": "full", "_agent_id": 7}))
    assert out["success"] is True
    assert out["data"]["level"] == "full"
    # The prior level is captured (returned + logged) so the change is auditable.
    assert out["data"]["previous_level"] == "standard"
