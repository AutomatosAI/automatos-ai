"""Structured error telemetry for hot-path exception handlers.

``record_error`` emits a single machine-parseable error record so subsystems
(memory, tools, harness, ...) report failures uniformly instead of via opaque
``logger.warning`` calls. It is the measurement instrument for PRD-141: per
subsystem error rates (e.g. Phase 1's "Mem0 error rate < 0.1%") are counted by
filtering the ``automatos.errors`` logger on ``structured_error.subsystem``.

PRD-142 Wave 0 US-001 added a best-effort persistence path so the same
records also land in the queryable ``error_events`` table, which backs the
dashboard "error rate by subsystem" tile. The persistence path is purely
additive: the logger emit is unchanged, the signature is unchanged, and a
failed sink write (DB outage, missing table, anything) is swallowed because
telemetry must not mask the original failure.

This module is pure-additive. It does not replace existing handlers; callers
opt in by invoking ``record_error`` from within their ``except`` block.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from uuid import UUID

# Imported at module scope so tests can monkey-patch SessionLocal. The
# import is wrapped to tolerate environments where the DB layer can't be
# initialised (alembic offline mode, lightweight smoke tests, ...) — we
# fall back to None and the sink path becomes a no-op, preserving the
# never-raises contract.
try:  # pragma: no cover — import-time guard
    from core.database.database import SessionLocal
except Exception:  # noqa: BLE001 — telemetry must never crash at import
    SessionLocal = None  # type: ignore[assignment]

# Dedicated channel so log processors can route/aggregate errors without
# scraping the message text. The existing ContextFilter (see
# core/utils/logging_adapter.py) still enriches these records with ambient
# request/workspace context when installed.
_ERROR_LOGGER = logging.getLogger("automatos.errors")
_SINK_LOGGER = logging.getLogger(__name__)

# Bound the message so a pathological exception string cannot blow up log
# storage or downstream JSON parsers.
_MAX_MESSAGE_LEN = 500


def record_error(
    *,
    subsystem: str,
    operation: str,
    error: Exception,
    workspace_id: Optional[UUID] = None,
    agent_id: Optional[int] = None,
    action_name: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a structured error record on the ``automatos.errors`` logger.

    Args:
        subsystem: Coarse origin of the failure (e.g. ``"memory"``, ``"tools"``).
        operation: The specific operation that failed (e.g. ``"add_memory"``).
        error: The caught exception. Its type and message are recorded.
        workspace_id: Owning workspace, if known. ``None`` is tolerated.
        agent_id: Owning agent, if known.
        action_name: Platform action involved, if any.
        extra: Additional machine-parseable fields merged into the record.

    The record carries a ``structured_error`` ``extra`` dict and ``exc_info=True``
    so handlers within an ``except`` block also capture the traceback. This
    function never raises — telemetry must not mask the original failure.

    PRD-142 Wave 0 US-001: the record is also persisted to ``error_events``
    on a best-effort basis. A failed sink write is logged at WARNING and
    swallowed; the ``automatos.errors`` emit above always happens first so
    we never blind-spot a failure when the DB is down.
    """
    error_message = str(error)[:_MAX_MESSAGE_LEN]

    structured_error: Dict[str, Any] = {
        "subsystem": subsystem,
        "operation": operation,
        "error_type": type(error).__name__,
        "error_message": error_message,
        "workspace_id": str(workspace_id) if workspace_id is not None else None,
        "agent_id": agent_id,
        "action_name": action_name,
    }
    if extra:
        structured_error.update(extra)

    _ERROR_LOGGER.error(
        "%s.%s failed: %s",
        subsystem,
        operation,
        error_message,
        exc_info=True,
        extra={"structured_error": structured_error},
    )

    _persist_error_event(
        subsystem=subsystem,
        operation=operation,
        error=error,
        error_message=error_message,
        workspace_id=workspace_id,
        agent_id=agent_id,
        action_name=action_name,
        extra=extra,
    )


def _persist_error_event(
    *,
    subsystem: str,
    operation: str,
    error: Exception,
    error_message: str,
    workspace_id: Optional[UUID],
    agent_id: Optional[int],
    action_name: Optional[str],
    extra: Optional[Dict[str, Any]],
) -> None:
    """Best-effort write to ``error_events``.

    Mirrors ``modules/widgets/telemetry.log_widget_event``: catch-all,
    rollback, swallow. Telemetry MUST NEVER fail the calling business path.
    """
    if SessionLocal is None:  # pragma: no cover — defensive
        return

    db = None
    try:
        from core.models.error_event import ErrorEvent  # local import

        db = SessionLocal()
        row = ErrorEvent(
            subsystem=subsystem[:64],
            operation=operation[:128],
            error_type=type(error).__name__[:128],
            error_message=error_message,  # already truncated to _MAX_MESSAGE_LEN
            workspace_id=workspace_id,
            agent_id=agent_id,
            action_name=action_name[:128] if action_name else None,
            event_data=extra or {},
        )
        db.add(row)
        db.commit()
    except Exception as exc:  # noqa: BLE001 — best-effort by contract
        _SINK_LOGGER.warning(
            "error_events sink write failed (subsystem=%r operation=%r): %s",
            subsystem,
            operation,
            exc,
        )
        if db is not None:
            try:
                db.rollback()
            except Exception:  # noqa: BLE001 — give up; never propagate
                pass
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:  # noqa: BLE001 — never propagate
                pass
