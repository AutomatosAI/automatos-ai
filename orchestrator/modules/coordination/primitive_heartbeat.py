"""Missions primitive heartbeat emit (PRD-142 Wave 3 · W3-S11).

A tiny stateless helper that wires the mission lifecycle to the W3-S1
``emit_primitive_finding`` writer. Lives in its own module so
``services/coordinator_service.py`` stays focused and the tests can
verify the contract without dragging the full orchestration import
surface (planner, agents, board bridge, etc).

Honest signal rules — match the W3-S6 (chat) / W3-S8 (rag) /
W3-S9 (nl2sql) / W3-S10 (graph) wrappers:

- ``success=True``  → emit ``status="green"`` (a clean RunState.COMPLETED).
- ``success=False`` → emit ``status="down"`` with the caught error /
  failure reason in ``detail`` (so the W3-S2 endpoint surfaces the most
  recent failure on the missions tile).
- No ``workspace_id`` → emit nothing. The tile stays ``unknown`` for that
  workspace instead of fabricating a default (A4 — honest gap over fake
  green).
- The emit is best-effort: a failure inside ``emit_primitive_finding``
  is logged and swallowed, NEVER raised back to the coordinator (a
  busted heartbeat MUST NOT break mission completion).
"""
from __future__ import annotations

import logging
from typing import Optional

from services.heartbeat_service import emit_primitive_finding

logger = logging.getLogger(__name__)


def _emit_missions_primitive(
    workspace_id: Optional[str],
    *,
    success: bool,
    detail: str = "",
) -> None:
    """Emit one ``missions`` primitive heartbeat finding for a mission
    terminal transition.

    Args:
        workspace_id: The workspace the mission ran in. If falsy, NO
            emit happens — the tile honestly reads ``unknown`` for that
            workspace instead of borrowing another's id.
        success: True on a clean ``RunState.COMPLETED`` (green); False
            on a caught failure / ``RunState.FAILED`` / ``CANCELLED`` (down).
        detail: Short human-readable context (truncated to 500 chars
            before reaching ``emit_primitive_finding``).
    """
    if not workspace_id:
        return
    status = "green" if success else "down"
    try:
        emit_primitive_finding(
            str(workspace_id),
            "missions",
            status,
            (detail or "")[:500],
        )
    except Exception:  # noqa: BLE001 — best-effort; never break the coordinator
        logger.error(
            "[missions-heartbeat] emit failed ws=%s status=%s",
            workspace_id, status, exc_info=True,
        )
