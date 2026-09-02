"""system_settings rows with NULL flag columns must not 500 the Settings page.

Two halves of one fix (2026-09-02): the trial ledger's raw INSERT now names
``is_sensitive`` / ``is_required`` / ``created_by``, and the response contract
tolerates NULLs from any other raw writer with its documented defaults.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from core.models.system_settings import SystemSettingResponse, SystemSettingsByCategory  # noqa: E402


def _row(**over):
    base = dict(
        id=100, category="llm_cost_audit", key="trial_spend_2026-09-02", value="0.001",
        value_type="number", description="PRD-222 trial daily spend", is_sensitive=None,
        is_required=None, default_value=None, validation_rules=None,
        created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc), created_by=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_response_contract_tolerates_null_flags_with_documented_defaults():
    out = SystemSettingResponse.model_validate(_row(), from_attributes=True)
    assert out.is_sensitive is False and out.is_required is False and out.created_by == "system"
    # explicit values still win
    out = SystemSettingResponse.model_validate(_row(is_sensitive=True, created_by="gerard"), from_attributes=True)
    assert out.is_sensitive is True and out.created_by == "gerard"


def test_by_category_group_survives_one_null_row():
    grouped = SystemSettingsByCategory(category="llm_cost_audit", settings=[_row(), _row(id=101)], total_count=2)
    assert len(grouped.settings) == 2 and all(s.is_required is False for s in grouped.settings)


class _Result:
    def __init__(self, row=None):
        self._row = row

    def fetchone(self):
        return self._row


class _CapturingDB:
    """SELECTs find nothing (so the INSERT path runs); every statement is recorded."""

    def __init__(self):
        self.statements = []

    def execute(self, clause, params=None):
        sql = str(getattr(clause, "text", clause))
        self.statements.append((sql, dict(params or {})))
        return _Result(None)


def test_trial_ledger_insert_names_the_flag_columns():
    from services import trial_ledger

    db = _CapturingDB()
    total = trial_ledger._increment_daily_spend(db, 0.0025)
    assert total == 0.0025
    inserts = [s for s, _ in db.statements if s.lstrip().upper().startswith("INSERT INTO SYSTEM_SETTINGS")]
    assert len(inserts) == 1, db.statements
    sql = inserts[0]
    for col in ("is_sensitive", "is_required", "created_by"):
        assert col in sql, f"INSERT must set {col}: {sql}"
    assert "false, false, 'system'" in sql
