"""PRD-142 Wave 4 (W4-S12): HARNESS dual-writes prescriptions to the DB store.

Strangler step 1: each tick's prescriptions are written to harness_prescriptions
(status derived from the apply changelog) IN ADDITION to the baseline JSON, via an
ISOLATED session that never touches the tick's transaction and never raises —
degrading to JSON-only if the migration isn't applied. Reads stay on the JSON
(authoritative) until the human cutover. Mocked DB — no real tables required.

Dummy POSTGRES_* + the apscheduler stub let the harness_service import chain load.
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

from services.harness_service import get_harness_service  # noqa: E402

_WS = UUID("00000000-0000-0000-0000-000000000001")


class _FakeSession:
    def __init__(self):
        self.added = []
        self.flushed = False

    def add(self, obj):
        self.added.append(obj)

    def flush(self):
        self.flushed = True


class _FakeCtx:
    def __init__(self, sess):
        self._sess = sess

    def __enter__(self):
        return self._sess

    def __exit__(self, *a):
        return False


def test_dual_write_persists_prescriptions_with_status(monkeypatch):
    import core.database.database as dbmod
    sess = _FakeSession()
    monkeypatch.setattr(dbmod, "get_db_session", lambda: _FakeCtx(sess))

    prescriptions = [
        {"prescription_id": "rx-1", "change_type": "heartbeat_tune", "target_id": 7,
         "target_name": "SCOUT", "risk_score": 2, "proposed_value": {"interval_minutes": 90}},
        {"prescription_id": "rx-2", "change_type": "model_change_same_tier", "target_id": 8,
         "risk_score": 3, "proposed_value": {"model": "x"}},
        {"prescription_id": "rx-3", "change_type": "temperature_adjust", "target_id": 9, "risk_score": 1},
    ]
    changelog = {"applied": [{"prescription_id": "rx-1"}], "queued": [{"prescription_id": "rx-2"}]}

    get_harness_service()._persist_prescriptions_to_db(_WS, 5, prescriptions, changelog)

    assert sess.flushed is True
    assert len(sess.added) == 3
    by_id = {r.prescription_id: r for r in sess.added}
    assert by_id["rx-1"].status == "applied"          # in changelog.applied
    assert by_id["rx-1"].workspace_id == _WS          # workspace-scoped
    assert by_id["rx-1"].run_id == "5"                # the tick iteration
    assert by_id["rx-1"].change_type == "heartbeat_tune"
    assert by_id["rx-1"].proposed_value == {"interval_minutes": 90}
    assert by_id["rx-2"].status == "queued"           # in changelog.queued
    assert by_id["rx-3"].status == "proposed"         # neither applied nor queued


def test_dual_write_is_best_effort_never_raises(monkeypatch):
    import core.database.database as dbmod

    def _boom():
        raise RuntimeError("db down / table missing")

    monkeypatch.setattr(dbmod, "get_db_session", _boom)
    # Must NOT raise even if the DB is unavailable / un-migrated — degrades to JSON.
    get_harness_service()._persist_prescriptions_to_db(
        _WS, 5, [{"prescription_id": "rx-1", "change_type": "x"}], {}
    )


def test_dual_write_noop_on_empty_prescriptions(monkeypatch):
    import core.database.database as dbmod
    opened = []
    monkeypatch.setattr(dbmod, "get_db_session", lambda: opened.append(1) or _FakeCtx(_FakeSession()))
    get_harness_service()._persist_prescriptions_to_db(_WS, 5, [], {})
    assert opened == []   # no session opened when there's nothing to write
