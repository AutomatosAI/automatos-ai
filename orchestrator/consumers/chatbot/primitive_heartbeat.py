"""Chat primitive heartbeat emit (PRD-142 Wave 3 · W3-S6).

A tiny stateless helper that wires the chat surface to the W3-S1
``emit_primitive_finding`` writer. Lives in its own module so the chat
service file stays focused and the tests can verify the contract without
dragging the full StreamingChatService import surface (LLM manager, tool
router, agent factory, all the model imports).

Honest signal rules:

- ``success=True``  → emit ``status="green"``.
- ``success=False`` → emit ``status="down"`` with the caught error in
  ``detail`` (so the W3-S2 endpoint surfaces the most recent failure).
- No ``workspace_id`` → emit nothing. The tile stays ``unknown`` for that
  workspace instead of fabricating a default (A4 — honest gap over fake
  green).
- The emit is best-effort: a failure inside ``emit_primitive_finding``
  is logged and swallowed, NEVER raised back to the chat caller.

This mirrors how Memory (W3-S7) and RAG (W3-S8) wire their primitive
emits — same helper, same canonical names, no per-chat caching that a
restart could lose.
"""
from __future__ import annotations

import logging
from typing import Optional

from services.heartbeat_service import emit_primitive_finding

logger = logging.getLogger(__name__)


def _emit_chat_primitive(
    workspace_id: Optional[str],
    *,
    success: bool,
    detail: str = "",
) -> None:
    """Emit one ``chat`` primitive heartbeat finding for the current turn.

    Args:
        workspace_id: The workspace running the chat turn. If falsy, NO
            emit happens — the tile honestly reads ``unknown`` for that
            workspace instead of borrowing another's id.
        success: True on a clean turn (green), False on a caught
            provider/tool error (down).
        detail: Short human-readable context (truncated to 500 chars by
            ``emit_primitive_finding``).
    """
    if not workspace_id:
        return
    status = "green" if success else "down"
    try:
        emit_primitive_finding(
            str(workspace_id),
            "chat",
            status,
            (detail or "")[:500],
        )
    except Exception:  # noqa: BLE001 — best-effort; never break chat
        logger.error(
            "[chat-heartbeat] emit failed ws=%s status=%s",
            workspace_id, status, exc_info=True,
        )
