"""PRD-185 S2: per-lane telemetry canary.

S1 repaired a type-poisoned telemetry write that had left ``tool_execution_logs``
with zero organic rows for ~2 months, unseen because nothing alarmed on "organic
rows/day = 0". This canary is that guardrail. Pure tests — the decision core takes
plain dicts; ``run_telemetry_canary`` is driven with a MagicMock session. No DB /
network.
"""
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

_orchestrator_root = str(Path(__file__).resolve().parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)

from services.telemetry_canary import (
    evaluate_telemetry_canary,
    run_telemetry_canary,
)


# ---------------------------------------------------------------------------
# evaluate_telemetry_canary — the pure decision core
# ---------------------------------------------------------------------------

def test_zero_rows_alarms():
    v = evaluate_telemetry_canary({}, window_seconds=86400, min_rows=0)
    assert v["alert"] is True
    assert v["organic_rows"] == 0


def test_rows_present_clears():
    v = evaluate_telemetry_canary(
        {"PLATFORM": 5, "WORKSPACE": 3}, window_seconds=86400, min_rows=0
    )
    assert v["alert"] is False
    assert v["organic_rows"] == 8
    assert v["per_lane"] == {"PLATFORM": 5, "WORKSPACE": 3}


def test_min_rows_threshold_catches_partial_silence():
    # 2 rows with a floor of 5 → still alarming (a lane went quiet).
    v = evaluate_telemetry_canary({"PLATFORM": 2}, window_seconds=3600, min_rows=5)
    assert v["alert"] is True


# ---------------------------------------------------------------------------
# run_telemetry_canary — the DB + log wrapper (session mocked at the boundary)
# ---------------------------------------------------------------------------

def _mock_db_returning(lane_rows):
    db = MagicMock()
    db.query.return_value.filter.return_value.group_by.return_value.all.return_value = lane_rows
    return db


def test_run_canary_alarms_when_empty(caplog):
    db = _mock_db_returning([])
    with caplog.at_level(logging.WARNING):
        v = run_telemetry_canary(db, window_seconds=86400, min_rows=0)
    assert v["alert"] is True
    assert any(
        "ALARM" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
    ), "a silent platform must log a loud WARNING"


def test_run_canary_clears_when_rows_flow():
    db = _mock_db_returning([("PLATFORM", 12), ("WORKSPACE", 4)])
    v = run_telemetry_canary(db, window_seconds=86400, min_rows=0)
    assert v["alert"] is False
    assert v["organic_rows"] == 16
    assert v["per_lane"] == {"PLATFORM": 12, "WORKSPACE": 4}


def test_run_canary_coerces_null_lane_name():
    # A NULL app_name must not blow up the group-by mapping.
    db = _mock_db_returning([(None, 3)])
    v = run_telemetry_canary(db, window_seconds=86400, min_rows=0)
    assert v["per_lane"] == {"unknown": 3}


def test_run_canary_never_raises_on_db_error():
    db = MagicMock()
    db.query.side_effect = RuntimeError("db down")
    v = run_telemetry_canary(db, window_seconds=86400, min_rows=0)
    # A broken query must neither alarm falsely nor raise into the scheduler.
    assert v["alert"] is False
    assert v.get("error") is True
