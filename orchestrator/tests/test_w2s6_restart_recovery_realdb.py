"""PRD-142 Wave 2 · WS-G · W2-S6 — G4: real-DB restart-recovery durability.

When the orchestrator restarts, any in-flight row whose background ``asyncio``
executor died with the old process is stranded forever — a board task stuck
``in_progress``, a wizard profile stuck ``scraping``, a workflow execution stuck
``running``. W1-S6 added ``reap_orphaned_runs`` to sweep those three surfaces on
boot and move each stale orphan to its surface's terminal state.

W1-S6's own tests prove the *contract* against a fake Session — query returns
seeded rows, staleness is filtered in Python, mutations are asserted on
``SimpleNamespace`` objects. They never touch Postgres, so they cannot prove the
mutations actually **persist**: that a real backend ends up terminal after the
reaper's single end-of-sweep ``commit``, and that a freshly-started job survives.
**This is that missing real-DB durability regression.** It commits genuine
orphan rows (the state a crash leaves behind), runs the real reaper, and re-reads
the rows from Postgres to confirm the terminal transition stuck.

Revert guard (the AC): ``test_reaper_marks_stale_orphans_terminal`` **fails** if
``reap_orphaned_runs`` is reverted to a no-op or dropped from boot — the rows
stay in-flight and every terminal assertion trips.

SAFETY — this is the rare test that WRITES AND COMMITS. The transactional
rollback fixture (root ``conftest.py``) cannot undo a commit, so this test
deletes exactly the rows it created. To make a stray commit against production
impossible, the module *skips* unless the engine points at a local, disposable
test database (CI's Postgres service, or a developer's local ``test_db``); it
never runs against Railway. Marked ``integration`` so it joins the live-DB job.
"""
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

# Importing core.database.database builds the engine at module load, which needs
# POSTGRES_* creds. setdefault means a real .env / CI service env still wins;
# this only covers a bare shell so the import doesn't crash before the
# ephemeral-DB guard can skip. The defaults are themselves ephemeral-safe.
for _k, _v in {
    "POSTGRES_USER": "test",
    "POSTGRES_PASSWORD": "test",
    "POSTGRES_HOST": "127.0.0.1",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "test_db",
}.items():
    os.environ.setdefault(_k, _v)

import core.boot.reaper as reaper  # noqa: E402
from core.database.database import engine  # noqa: E402
from core.models.business_profiles import BusinessProfile  # noqa: E402
from core.models.core import BoardTask, WorkflowExecution  # noqa: E402
from core.models.workspaces import Workspace  # noqa: E402

pytestmark = pytest.mark.integration

_ORPHAN = "orphaned_on_restart"  # core.boot.reaper._ORPHAN_REASON
_EPHEMERAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
_EPHEMERAL_DBS = {"test_db", "test"}


def _is_ephemeral(url) -> bool:
    """True only for a local, disposable test database — never production.

    create_engine is lazy, so reading ``engine.url`` resolves the configured
    target WITHOUT connecting. We gate on it before opening any connection, so a
    Railway URL is rejected without a single byte sent to it.
    """
    return (
        (url.host or "").lower() in _EPHEMERAL_HOSTS
        and (url.database or "").lower() in _EPHEMERAL_DBS
    )


@pytest.fixture(scope="module")
def safe_engine():
    """Refuse to run this WRITE+COMMIT test against anything but an ephemeral DB."""
    if not _is_ephemeral(engine.url):
        pytest.skip(
            "refusing to run the destructive reaper test against a non-ephemeral "
            f"database ({engine.url.host}/{engine.url.database}); needs a local "
            "test_db / test"
        )
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # OperationalError and friends
        pytest.skip(f"no reachable Postgres for integration test: {exc}")
    return engine


@pytest.fixture
def seed(safe_engine, monkeypatch):
    """Commit one stale + one fresh orphan on each of the reaper's three surfaces.

    Stale rows are timestamped well past the staleness cutoff (driven by the live
    ``BOOT_REAPER_STALE_MINUTES`` so the test tracks the real boundary); fresh
    rows sit just inside it. Yields a namespace of row ids; tears down by deleting
    exactly those rows (a commit can't be rolled back).
    """
    # Telemetry isn't what G4 proves — W1-S6 covers the record_error fan-out — and
    # letting it write would couple this test to the error_events schema. Stub it.
    monkeypatch.setattr(reaper, "record_error", MagicMock())
    # The sweep must run regardless of the ambient BOOT_REAPER_ENABLED env; the
    # disabled short-circuit is covered by W1-S6's mock suite.
    monkeypatch.setattr(reaper.config, "BOOT_REAPER_ENABLED", True)

    stale_min = reaper.config.BOOT_REAPER_STALE_MINUTES
    now = datetime.now(timezone.utc)
    old = now - timedelta(minutes=stale_min + 60)   # comfortably past the cutoff
    fresh = now - timedelta(seconds=1)              # comfortably inside the window
    old_naive = old.replace(tzinfo=None)            # workflow_executions.started_at is tz-naive
    fresh_naive = fresh.replace(tzinfo=None)

    db = sessionmaker(bind=safe_engine)()
    ws = Workspace(id=uuid4(), name="w2s6-reaper-test")
    db.add(ws)
    db.flush()  # materialise ws.id for the child FKs

    rows = {
        "board_stale": BoardTask(workspace_id=ws.id, title="stale", status="in_progress", started_at=old),
        "board_fresh": BoardTask(workspace_id=ws.id, title="fresh", status="in_progress", started_at=fresh),
        "prof_stale": BusinessProfile(workspace_id=ws.id, domain="stale.example", status="scraping", updated_at=old),
        "prof_fresh": BusinessProfile(workspace_id=ws.id, domain="fresh.example", status="scraping", updated_at=fresh),
        "wfe_stale": WorkflowExecution(workspace_id=ws.id, status="running", started_at=old_naive),
        "wfe_fresh": WorkflowExecution(workspace_id=ws.id, status="running", started_at=fresh_naive),
    }
    db.add_all(rows.values())
    db.commit()  # the durable state a crash leaves behind

    ns = SimpleNamespace(db=db, now=now, ws_id=ws.id, **{k: v.id for k, v in rows.items()})
    try:
        yield ns
    finally:
        db.rollback()  # drop any partial tx before the deletes
        for model, ids in (
            (BoardTask, [ns.board_stale, ns.board_fresh]),
            (BusinessProfile, [ns.prof_stale, ns.prof_fresh]),
            (WorkflowExecution, [ns.wfe_stale, ns.wfe_fresh]),
        ):
            db.query(model).filter(model.id.in_(ids)).delete(synchronize_session=False)
        db.query(Workspace).filter(Workspace.id == ns.ws_id).delete(synchronize_session=False)
        db.commit()
        db.close()


# ── The leak precondition (proves the regression has teeth) ──────────────────

def test_orphaned_rows_persist_in_flight_until_reaped(seed):
    """A crash leaves these rows committed and in-flight — the state the reaper
    exists to clean. Asserting it on real Postgres stops the durability test below
    from passing vacuously if seeding ever breaks."""
    db = seed.db
    assert db.get(BoardTask, seed.board_stale).status == "in_progress"
    assert db.get(BusinessProfile, seed.prof_stale).status == "scraping"
    assert db.get(WorkflowExecution, seed.wfe_stale).status == "running"


# ── The fix moves stale orphans terminal, and it persists (the revert guard) ──

def test_reaper_marks_stale_orphans_terminal(seed):
    """Each stale orphan must be committed to its surface's terminal state.

    Revert guard: if ``reap_orphaned_runs`` is reverted to a no-op or removed from
    boot, the rows stay in_progress/scraping/running and every assertion trips.
    Re-reading via ``db.get`` after the reaper's commit proves the write persisted,
    not merely that an in-memory attribute was set — the real-DB proof W1-S6 lacked.
    """
    db = seed.db
    reaped = reaper.reap_orphaned_runs(db, now=seed.now)
    assert reaped >= 3  # at least our three stale rows (other test data may add more)

    board = db.get(BoardTask, seed.board_stale)
    assert board.status == "done"  # the board's own failure convention (no 'failed' column)
    assert board.completed_at is not None
    assert _ORPHAN in (board.error_message or "")

    prof = db.get(BusinessProfile, seed.prof_stale)
    assert prof.status == "failed"
    assert _ORPHAN in str(prof.quality_findings)

    wfe = db.get(WorkflowExecution, seed.wfe_stale)
    assert wfe.status == "failed"
    assert wfe.completed_at is not None
    assert _ORPHAN in (wfe.error_message or "")


# ── The staleness gate holds on real Postgres (no premature reaping) ──────────

def test_reaper_leaves_fresh_runs_untouched(seed):
    """A job that started moments ago is still legitimately running and must
    survive the sweep — the reaper must not kill live work."""
    db = seed.db
    reaper.reap_orphaned_runs(db, now=seed.now)

    assert db.get(BoardTask, seed.board_fresh).status == "in_progress"
    assert db.get(BusinessProfile, seed.prof_fresh).status == "scraping"
    assert db.get(WorkflowExecution, seed.wfe_fresh).status == "running"
