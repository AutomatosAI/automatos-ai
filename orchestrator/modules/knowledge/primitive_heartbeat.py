"""Graph primitive heartbeat emit (PRD-142 Wave 3 · W3-S10).

A tiny stateless helper that wires the Knowledge Graph (moat) surface to
the W3-S1 ``emit_primitive_finding`` writer. Lives in its own module so
``graph_service.py`` stays focused on the build pipeline and the tests
can verify the contract without dragging the full ``modules.knowledge``
import surface (graphify, networkx, cachetools, ...).

Honest signal rules — match the W3-S6 (chat) / W3-S8 (rag) / W3-S9
(nl2sql) wrappers:

- ``success=True``  -> emit ``status="green"``.
- ``success=False`` -> emit ``status="down"`` with the caught error in
  ``detail`` (so the W3-S2 endpoint surfaces the most recent failure).
- No ``workspace_id`` -> emit nothing. The tile stays ``unknown`` for
  that workspace instead of fabricating a default (A4 — honest gap over
  fake green).
- The emit is best-effort: a failure inside ``emit_primitive_finding``
  is logged and swallowed, NEVER raised back to the graph caller (a
  build that *succeeded* must not be reported as failed just because the
  heartbeat row couldn't be written, and a build that *failed* must
  re-raise its original exception untouched).
"""
from __future__ import annotations

import logging
from typing import Optional

from services.heartbeat_service import emit_primitive_finding

logger = logging.getLogger(__name__)


def _emit_graph_primitive(
    workspace_id: Optional[str],
    *,
    success: bool,
    detail: str = "",
) -> None:
    """Emit one ``graph`` primitive heartbeat finding for the current build.

    Args:
        workspace_id: The workspace whose moat was just (re)built. If
            falsy, NO emit happens — the tile honestly reads ``unknown``
            for that workspace instead of borrowing another's id.
        success: True on a clean build/import (green), False on a caught
            build error or timeout (down).
        detail: Short human-readable context (truncated to 500 chars by
            ``emit_primitive_finding``).
    """
    if not workspace_id:
        return
    status = "green" if success else "down"
    try:
        emit_primitive_finding(
            str(workspace_id),
            "graph",
            status,
            (detail or "")[:500],
        )
    except Exception:  # noqa: BLE001 — best-effort; never break graph build
        logger.error(
            "[graph-heartbeat] emit failed ws=%s status=%s",
            workspace_id, status, exc_info=True,
        )
