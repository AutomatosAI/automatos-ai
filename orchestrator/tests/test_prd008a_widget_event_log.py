"""
PRD-008-A Phase 4 — WidgetEventLog model + telemetry helper tests
====================================================================

Pure-Python unit tests with mocked SQLAlchemy session. Verifies the
fire-and-forget contract: telemetry writes never propagate failures
into the calling business path.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

# Ensure .env is loaded so core.database.database can resolve creds at
# import time, regardless of test run order.
import config  # noqa: E402,F401


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def test_table_name_is_widget_event_log():
    from core.models.widget_event_log import WidgetEventLog

    assert WidgetEventLog.__tablename__ == "widget_event_log"


def test_event_type_allowlist_includes_all_documented_events():
    """PRD-008-A spec lists these — keep them in lockstep."""
    from core.models.widget_event_log import WIDGET_EVENT_TYPES

    expected = {
        "proactive_fired", "proactive_dismissed",
        "callback_requested", "callback_delivered", "callback_failed",
        "cart_idle_fired", "cart_idle_dismissed",
        "settings_changed",
    }
    assert expected.issubset(WIDGET_EVENT_TYPES)


def test_event_type_allowlist_is_frozenset():
    """Frozen so callers can't mutate the allow-list at runtime."""
    from core.models.widget_event_log import WIDGET_EVENT_TYPES

    assert isinstance(WIDGET_EVENT_TYPES, frozenset)


def test_required_indexes_present():
    """Dashboard rollups query by (site_id, created_at) and (event_type, created_at).
    Without these indexes the dashboard scans the whole log."""
    from core.models.widget_event_log import WidgetEventLog

    indexed = {idx.name for idx in WidgetEventLog.__table__.indexes}
    assert "idx_widget_event_log_site_created" in indexed
    assert "idx_widget_event_log_type_created" in indexed


def test_event_data_defaults_to_empty_jsonb():
    from core.models.widget_event_log import WidgetEventLog

    server_default = WidgetEventLog.__table__.c.event_data.server_default
    assert server_default is not None
    assert "{}" in str(server_default.arg)


# ---------------------------------------------------------------------------
# Helper — happy path
# ---------------------------------------------------------------------------

def _await(coro):
    return asyncio.run(coro)


def test_log_widget_event_writes_row_for_known_event():
    from modules.widgets.telemetry import log_widget_event

    db = MagicMock()
    _await(log_widget_event(
        db,
        site_id=uuid4(),
        event_type="callback_requested",
        session_id="sess_abc",
        event_data={"phone_hash": "abc123"},
    ))

    db.add.assert_called_once()
    db.commit.assert_called_once()

    row = db.add.call_args[0][0]
    assert row.event_type == "callback_requested"
    assert row.session_id == "sess_abc"
    assert row.event_data == {"phone_hash": "abc123"}


def test_log_widget_event_handles_missing_optional_args():
    from modules.widgets.telemetry import log_widget_event

    db = MagicMock()
    _await(log_widget_event(
        db, site_id=uuid4(), event_type="proactive_fired",
    ))

    row = db.add.call_args[0][0]
    assert row.session_id is None
    assert row.event_data == {}


def test_log_widget_event_truncates_overlong_session_id():
    """session_id column is String(64). Don't blow up on a malformed input."""
    from modules.widgets.telemetry import log_widget_event

    db = MagicMock()
    huge_session = "x" * 500
    _await(log_widget_event(
        db, site_id=uuid4(), event_type="proactive_fired", session_id=huge_session,
    ))

    row = db.add.call_args[0][0]
    assert len(row.session_id) == 64


# ---------------------------------------------------------------------------
# Helper — fire-and-forget contract
# ---------------------------------------------------------------------------

def test_log_widget_event_swallows_db_failures(caplog):
    """Telemetry MUST NOT fail the calling business path."""
    from modules.widgets.telemetry import log_widget_event

    db = MagicMock()
    db.commit.side_effect = RuntimeError("simulated DB outage")

    # Must NOT raise
    _await(log_widget_event(
        db, site_id=uuid4(), event_type="callback_requested",
    ))

    db.rollback.assert_called_once()
    assert any("widget_event_log write failed" in rec.message for rec in caplog.records)


def test_log_widget_event_swallows_rollback_failures():
    """Even rollback failure must be swallowed — never propagate."""
    from modules.widgets.telemetry import log_widget_event

    db = MagicMock()
    db.commit.side_effect = RuntimeError("commit broke")
    db.rollback.side_effect = RuntimeError("rollback also broke")

    # Must NOT raise
    _await(log_widget_event(
        db, site_id=uuid4(), event_type="callback_requested",
    ))


def test_log_widget_event_warns_on_unknown_event_type(caplog):
    """Unknown types still get written (so we don't lose data) but logged."""
    from modules.widgets.telemetry import log_widget_event

    db = MagicMock()
    _await(log_widget_event(
        db, site_id=uuid4(), event_type="brand_new_unknown_event_type",
    ))

    # Row still attempted
    db.add.assert_called_once()
    # And we warned
    assert any(
        "unknown event_type" in rec.message and "brand_new_unknown_event_type" in rec.message
        for rec in caplog.records
    )


def test_log_widget_event_truncates_overlong_event_type():
    """Defensive: an attacker-supplied event_type that's too long shouldn't
    explode the DB write. event_type column is String(64)."""
    from modules.widgets.telemetry import log_widget_event

    db = MagicMock()
    _await(log_widget_event(
        db, site_id=uuid4(), event_type="x" * 500,
    ))

    row = db.add.call_args[0][0]
    assert len(row.event_type) == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
