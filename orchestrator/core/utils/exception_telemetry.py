"""Structured error telemetry for hot-path exception handlers.

``record_error`` emits a single machine-parseable error record so subsystems
(memory, tools, harness, ...) report failures uniformly instead of via opaque
``logger.warning`` calls. It is the measurement instrument for PRD-141: per
subsystem error rates (e.g. Phase 1's "Mem0 error rate < 0.1%") are counted by
filtering the ``automatos.errors`` logger on ``structured_error.subsystem``.

This module is pure-additive. It does not replace existing handlers; callers
opt in by invoking ``record_error`` from within their ``except`` block.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from uuid import UUID

# Dedicated channel so log processors can route/aggregate errors without
# scraping the message text. The existing ContextFilter (see
# core/utils/logging_adapter.py) still enriches these records with ambient
# request/workspace context when installed.
_ERROR_LOGGER = logging.getLogger("automatos.errors")

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
