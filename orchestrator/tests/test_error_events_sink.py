"""PRD-142 Wave 0 US-001 — error_events sink + record_error persistence.

The ``record_error`` function (``core/utils/exception_telemetry.py``) MUST
persist its records to the ``error_events`` table in addition to emitting on
the ``automatos.errors`` logger, so the dashboard can query subsystem error
rates without scraping log files.

Fire-and-forget contract (mirrors ``modules/widgets/telemetry.py``):

* The signature stays keyword-only and unchanged.
* The sink write is best-effort: a DB outage, a missing table, or a
  serialisation failure MUST NOT propagate. ``record_error`` never raises.
* If persistence fails, the original ``automatos.errors`` log line still
  goes out, so we never blind-spot a failure.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

# Ensure .env is loaded so core.database.database can resolve creds at
# import time (matches PRD-008-A test pattern).
import config  # noqa: E402,F401


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def test_error_event_table_name_and_indexes():
    """The model is named ``error_events`` and carries the two indexes the
    aggregation endpoint (US-002) needs."""
    from core.models.error_event import ErrorEvent

    assert ErrorEvent.__tablename__ == "error_events"

    indexed = {idx.name for idx in ErrorEvent.__table__.indexes}
    assert "idx_error_events_subsystem_created" in indexed
    assert "idx_error_events_workspace_created" in indexed


def test_error_event_event_data_defaults_to_empty_jsonb():
    from core.models.error_event import ErrorEvent

    server_default = ErrorEvent.__table__.c.event_data.server_default
    assert server_default is not None
    assert "{}" in str(server_default.arg)


# ---------------------------------------------------------------------------
# record_error — persistence
# ---------------------------------------------------------------------------

def test_record_error_persists_row(caplog):
    """Happy path: record_error opens a session, builds an ErrorEvent row,
    and commits it. The logger emit still happens."""
    from core.utils import exception_telemetry

    fake_session = MagicMock()
    session_factory = MagicMock(return_value=fake_session)

    ws = uuid4()
    with patch.object(exception_telemetry, "SessionLocal", session_factory, create=True):
        with caplog.at_level(logging.ERROR, logger="automatos.errors"):
            try:
                raise ValueError("boom")
            except ValueError as exc:
                exception_telemetry.record_error(
                    subsystem="memory",
                    operation="add_memory",
                    error=exc,
                    workspace_id=ws,
                    agent_id=42,
                    action_name="platform_add_memory",
                    extra={"correlation_id": "abc"},
                )

    # Logger emit must still happen (never replaced).
    assert any(r.name == "automatos.errors" for r in caplog.records)

    # Sink got a row added and committed.
    fake_session.add.assert_called_once()
    fake_session.commit.assert_called_once()
    row = fake_session.add.call_args[0][0]
    assert row.subsystem == "memory"
    assert row.operation == "add_memory"
    assert row.error_type == "ValueError"
    assert row.error_message == "boom"
    assert row.workspace_id == ws
    assert row.agent_id == 42
    assert row.action_name == "platform_add_memory"
    assert row.event_data == {"correlation_id": "abc"}


def test_record_error_sink_failure_is_swallowed(caplog):
    """If commit() raises (DB outage, missing table, anything), record_error
    must still return cleanly and the logger emit must still happen."""
    from core.utils import exception_telemetry

    fake_session = MagicMock()
    fake_session.commit.side_effect = RuntimeError("simulated DB outage")
    session_factory = MagicMock(return_value=fake_session)

    with patch.object(exception_telemetry, "SessionLocal", session_factory, create=True):
        with caplog.at_level(logging.ERROR, logger="automatos.errors"):
            try:
                raise RuntimeError("real failure")
            except RuntimeError as exc:
                # MUST NOT raise — telemetry must not mask the original failure.
                exception_telemetry.record_error(
                    subsystem="tools",
                    operation="dispatch",
                    error=exc,
                )

    # Logger emit still happened.
    assert any(r.name == "automatos.errors" for r in caplog.records)
    # Sink attempted rollback.
    fake_session.rollback.assert_called_once()


def test_record_error_truncates_message_in_db():
    """The DB column is VARCHAR(500). A pathological long exception message
    must be truncated before persistence so the INSERT doesn't fail."""
    from core.utils import exception_telemetry

    fake_session = MagicMock()
    session_factory = MagicMock(return_value=fake_session)

    long_message = "x" * 1000
    with patch.object(exception_telemetry, "SessionLocal", session_factory, create=True):
        try:
            raise RuntimeError(long_message)
        except RuntimeError as exc:
            exception_telemetry.record_error(
                subsystem="memory",
                operation="add_memory",
                error=exc,
            )

    row = fake_session.add.call_args[0][0]
    assert len(row.error_message) == 500
    assert row.error_message == "x" * 500


def test_none_workspace_persists():
    """System-level errors have workspace_id=None. They must still persist —
    the column is nullable. The aggregation endpoint excludes NULL workspace
    rows from per-workspace views by design (US-002 notes)."""
    from core.utils import exception_telemetry

    fake_session = MagicMock()
    session_factory = MagicMock(return_value=fake_session)

    with patch.object(exception_telemetry, "SessionLocal", session_factory, create=True):
        try:
            raise KeyError("missing")
        except KeyError as exc:
            exception_telemetry.record_error(
                subsystem="harness",
                operation="apply_prescription",
                error=exc,
                workspace_id=None,
            )

    fake_session.add.assert_called_once()
    fake_session.commit.assert_called_once()
    row = fake_session.add.call_args[0][0]
    assert row.workspace_id is None
    assert row.subsystem == "harness"
    assert row.error_type == "KeyError"


def test_record_error_rollback_failure_is_also_swallowed():
    """Even a rollback() that itself raises must not propagate — defensive
    parity with log_widget_event."""
    from core.utils import exception_telemetry

    fake_session = MagicMock()
    fake_session.commit.side_effect = RuntimeError("commit broke")
    fake_session.rollback.side_effect = RuntimeError("rollback also broke")
    session_factory = MagicMock(return_value=fake_session)

    with patch.object(exception_telemetry, "SessionLocal", session_factory, create=True):
        try:
            raise ValueError("boom")
        except ValueError as exc:
            # MUST NOT raise.
            exception_telemetry.record_error(
                subsystem="memory",
                operation="add_memory",
                error=exc,
            )


def test_record_error_signature_unchanged():
    """Guardrail: record_error keeps its keyword-only signature. Adding a
    positional ``db`` parameter would break every existing caller."""
    import inspect

    from core.utils.exception_telemetry import record_error

    sig = inspect.signature(record_error)
    expected = {
        "subsystem", "operation", "error", "workspace_id",
        "agent_id", "action_name", "extra",
    }
    assert set(sig.parameters.keys()) == expected
    # All keyword-only.
    for name, param in sig.parameters.items():
        assert param.kind == inspect.Parameter.KEYWORD_ONLY, (
            f"{name} must remain keyword-only"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
