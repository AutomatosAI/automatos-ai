"""Playbooks primitive heartbeat emit (PRD-142 Wave 3 · W3-S12).

A tiny stateless helper that wires the executor's terminal transitions to the
W3-S1 ``emit_primitive_finding`` writer. Lives in its own module so
``api/recipe_executor.py`` stays focused and the tests can verify the
contract without dragging the full executor import surface (board bridge,
playbook memory service, learning service, etc).

Honest signal rules — match the W3-S6 (chat) / W3-S8 (rag) / W3-S9 (nl2sql) /
W3-S10 (graph) / W3-S11 (missions) wrappers:

- ``success=True``  → emit ``status="green"`` (a clean ``status='completed'``).
- ``success=False`` → emit ``status="down"`` with the caught error /
  failure reason in ``detail`` (so the W3-S2 endpoint surfaces the most
  recent failure on the playbooks tile).
- No ``workspace_id`` → emit nothing. The tile stays ``unknown`` for that
  workspace instead of fabricating a default (A4 — honest gap over fake
  green).
- The emit is best-effort: a failure inside ``emit_primitive_finding``
  is logged and swallowed, NEVER raised back to the executor (a busted
  heartbeat MUST NOT break playbook completion).
"""
from __future__ import annotations

import logging
from typing import Optional

from services.heartbeat_service import emit_primitive_finding

logger = logging.getLogger(__name__)


def _emit_playbooks_primitive(
    workspace_id: Optional[str],
    *,
    success: bool,
    detail: str = "",
) -> None:
    """Emit one ``playbooks`` primitive heartbeat finding for a terminal
    playbook-execution transition.

    Args:
        workspace_id: The workspace the playbook ran in. If falsy, NO
            emit happens — the tile honestly reads ``unknown`` for that
            workspace instead of borrowing another's id.
        success: True on a clean ``status='completed'`` (green); False
            on a caught failure / ``_fail_execution`` path (down).
        detail: Short human-readable context (truncated to 500 chars
            before reaching ``emit_primitive_finding``).
    """
    if not workspace_id:
        return
    status = "green" if success else "down"
    try:
        emit_primitive_finding(
            str(workspace_id),
            "playbooks",
            status,
            (detail or "")[:500],
        )
    except Exception:  # noqa: BLE001 — best-effort; never break the executor
        logger.error(
            "[playbooks-heartbeat] emit failed ws=%s status=%s",
            workspace_id, status, exc_info=True,
        )
