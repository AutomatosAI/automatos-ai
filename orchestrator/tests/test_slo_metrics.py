"""PRD-180 S5 — the three tracked SLOs compute from real telemetry.

Two layers, mirroring the dispatch suite:

* **DB-backed (real Postgres):** seed ``ToolExecutionLog`` + ``BoardTask`` rows and
  assert each SLI computes the expected value + target from that telemetry. Uses
  committed sessions and skips cleanly when no Postgres is reachable.
* **Pure (no DB):** the target/comparator judgement (``_meets``) and the empty-data
  honesty (a ``None`` value is never judged as pass/fail), runnable with no DB.
"""
from __future__ import annotations

import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# ``slo_metrics`` imports ``config`` + models but opens no DB connection at import;
# ``get_database_url`` is imported lazily in the engine fixture so the pure tests
# run with no DB. Mirrors test_board_sse_listen_notify.py.
_ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(_ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_ORCH_ROOT))

from services import slo_metrics  # noqa: E402
from core.models.composio_cache import ToolExecutionLog  # noqa: E402
from core.models.core import BoardTask  # noqa: E402


# ─────────────────────────── DB-backed (real Postgres) ──────────────────────

@pytest.fixture(scope="module")
def engine():
    """Real Postgres engine; skip the DB-backed tests cleanly when none is up."""
    try:
        from core.database.database import get_database_url

        eng = create_engine(get_database_url(), pool_pre_ping=True, pool_size=6, max_overflow=4)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"SLO suite needs a reachable Postgres: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def new_session(engine):
    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture
def workspace(engine, new_session):
    """Throwaway workspace; cleans up its telemetry + itself on teardown."""
    ws_id = str(uuid.uuid4())
    s = new_session()
    # Matches the proven test_board_dispatch.py seed: the model declares owner_id
    # nullable with no FK, so CI's create_all schema (the test target) accepts an
    # (id, name) workspace. ON CONFLICT keeps re-runs idempotent.
    s.execute(
        text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), :n) ON CONFLICT (id) DO NOTHING"),
        {"id": ws_id, "n": "prd180-slo"},
    )
    s.commit()
    s.close()

    yield ws_id

    s = new_session()
    s.execute(text("DELETE FROM tool_execution_logs WHERE workspace_id = CAST(:id AS uuid)"), {"id": ws_id})
    s.execute(text("DELETE FROM board_tasks WHERE workspace_id = CAST(:id AS uuid)"), {"id": ws_id})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": ws_id})
    s.commit()
    s.close()


def test_slo_metrics(workspace, new_session):
    """Each SLI computes a value + target from seeded telemetry."""
    ws_id = workspace
    now = datetime.now(timezone.utc)

    # SLI 1 telemetry: 8 success + 2 error = 80% success rate. Plus one 'eval'
    # row that must be EXCLUDED (only production counts).
    s = new_session()
    for i in range(8):
        s.add(ToolExecutionLog(
            app_name="gmail", action_name="SEND", status="success",
            workspace_id=ws_id, executed_at=now - timedelta(minutes=5),
            telemetry_source="production",
        ))
    for i in range(2):
        s.add(ToolExecutionLog(
            app_name="gmail", action_name="SEND", status="error",
            workspace_id=ws_id, executed_at=now - timedelta(minutes=5),
            telemetry_source="production",
        ))
    # This eval row would drag the rate down if counted — it must not be.
    s.add(ToolExecutionLog(
        app_name="gmail", action_name="SEND", status="error",
        workspace_id=ws_id, executed_at=now - timedelta(minutes=5),
        telemetry_source="eval",
    ))

    # SLI 2 telemetry: three dispatched tasks with known created→started lags of
    # 2s, 4s, 10s → p95 ≈ 9.4s (interpolated), sample_size 3. Plus one task that
    # never started (must be ignored).
    for lag in (2, 4, 10):
        created = now - timedelta(minutes=10)
        s.add(BoardTask(
            workspace_id=ws_id, title=f"lag-{lag}", status="in_progress",
            priority="medium", source_type="user",
            created_at=created, started_at=created + timedelta(seconds=lag),
        ))
    s.add(BoardTask(
        workspace_id=ws_id, title="never-started", status="assigned",
        priority="medium", source_type="user", created_at=now - timedelta(minutes=1),
    ))
    s.commit()
    s.close()

    s = new_session()
    try:
        # SLI 1 — success rate = 8/10 = 80.0%, production only, target 99% (fails).
        sli1 = slo_metrics.tool_call_success_rate(s, workspace_id=ws_id)
        assert sli1["sli"] == "tool_call_success_rate"
        assert sli1["value"] == 80.0, sli1
        assert sli1["sample_size"] == 10, "eval row must be excluded"
        assert sli1["target"] == slo_metrics.SUCCESS_RATE_TARGET
        assert sli1["meets_target"] is False  # 80% < 99%

        # SLI 2 — p95 dispatch latency over {2,4,10}s. Only started tasks count.
        sli2 = slo_metrics.board_dispatch_latency_p95_seconds(s, workspace_id=ws_id)
        assert sli2["sli"] == "board_dispatch_latency_p95_seconds"
        assert sli2["sample_size"] == 3, "the never-started task must be ignored"
        assert 9.0 <= sli2["value"] <= 10.0, sli2  # p95 of {2,4,10} ≈ 9.4
        assert sli2["target"] == slo_metrics.DISPATCH_LATENCY_P95_TARGET

        # SLI 3 — freshness: newest board mutation is seconds old → well under 30s.
        sli3 = slo_metrics.board_event_freshness_seconds(s, workspace_id=ws_id)
        assert sli3["sli"] == "board_event_freshness_seconds"
        assert sli3["value"] is not None
        assert sli3["value"] >= 0.0
        assert sli3["target"] == slo_metrics.EVENT_FRESHNESS_TARGET

        # The envelope wraps all three with a stable shape.
        env = slo_metrics.compute_slos(s, workspace_id=ws_id)
        assert len(env["slos"]) == 3
        assert {x["sli"] for x in env["slos"]} == {
            "tool_call_success_rate",
            "board_dispatch_latency_p95_seconds",
            "board_event_freshness_seconds",
        }
        assert "generated_at" in env
    finally:
        s.close()


def test_slis_report_none_on_empty_telemetry(workspace, new_session):
    """With no telemetry, each windowed SLI reports value=None + meets_target=None
    (honest empty measurement — never a fabricated 0)."""
    ws_id = workspace  # freshly created, no rows seeded
    s = new_session()
    try:
        sli1 = slo_metrics.tool_call_success_rate(s, workspace_id=ws_id)
        assert sli1["value"] is None and sli1["sample_size"] == 0
        assert sli1["meets_target"] is None

        sli2 = slo_metrics.board_dispatch_latency_p95_seconds(s, workspace_id=ws_id)
        assert sli2["value"] is None and sli2["sample_size"] == 0
        assert sli2["meets_target"] is None

        sli3 = slo_metrics.board_event_freshness_seconds(s, workspace_id=ws_id)
        assert sli3["value"] is None and sli3["sample_size"] == 0
    finally:
        s.close()


# ─────────────────────────── Pure (no DB) ───────────────────────────────────

def test_meets_target_judgement_is_honest():
    """``_meets`` respects the comparator and never judges a None value."""
    assert slo_metrics._meets(99.5, 99.0, ">=") is True
    assert slo_metrics._meets(98.0, 99.0, ">=") is False
    assert slo_metrics._meets(3.0, 5.0, "<=") is True
    assert slo_metrics._meets(6.0, 5.0, "<=") is False
    # No data → no judgement (honest empty measurement).
    assert slo_metrics._meets(None, 99.0, ">=") is None


def test_meets_target_rejects_unknown_comparator():
    with pytest.raises(ValueError):
        slo_metrics._meets(1.0, 1.0, "==")
