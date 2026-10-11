"""
PRD-207 S3: Retell call-lifecycle events → ``voice_calls``
===========================================================

The meter's write path: one Retell lifecycle event (``call_started``,
``call_ended``, ``call_analyzed``) folded idempotently into the call's
``voice_calls`` row. The HMAC check and the 2xx answer stay in the route
(``api/voice_retell.py``); this module only applies a verified event.
Moved out of the route by PRD-258 US-001 (D6: that file never grows).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Lifecycle events Retell posts to the webhook.
EVENT_CALL_STARTED = "call_started"
EVENT_CALL_ENDED = "call_ended"
EVENT_CALL_ANALYZED = "call_analyzed"
# A ``disconnect_reason`` longer than the column is cut, never refused.
_DISCONNECT_REASON_MAX = 64


def _ts(ms: Any) -> Optional[datetime]:
    """A Retell millisecond epoch as a naive UTC datetime, or None when unreadable."""
    try:
        return datetime.fromtimestamp(int(ms) / 1000.0, tz=timezone.utc).replace(tzinfo=None)
    except (TypeError, ValueError, OSError):
        return None


def _call_row(db: Session, call_id: str, event: str):
    """The call's row; a call we never minted is stored as a LOUD orphan (phone-lane compat)."""
    from core.models.voice_calls import VoiceCall

    row = db.query(VoiceCall).filter(VoiceCall.call_id == call_id).first()
    if row is not None:
        return row, "updated"
    logger.warning(
        "voice_live_orphan_event event=%s call=%s — no mint row; storing loud", event, call_id
    )
    row = VoiceCall(call_id=call_id, provider="retell", status="minted")
    db.add(row)
    return row, "orphan_created"


def _apply_started(row, call: dict) -> None:
    row.started_at = _ts(call.get("start_timestamp")) or row.started_at
    if row.status in ("minted",):
        row.status = "started"


def _apply_ended(row, call: dict) -> None:
    row.started_at = _ts(call.get("start_timestamp")) or row.started_at
    row.ended_at = _ts(call.get("end_timestamp")) or row.ended_at
    if row.started_at is not None and row.ended_at is not None:
        row.duration_seconds = max(0, int((row.ended_at - row.started_at).total_seconds()))
    reason = call.get("disconnection_reason")
    if reason:
        row.disconnect_reason = str(reason)[:_DISCONNECT_REASON_MAX]
    # ended without ever starting = the call never connected.
    row.status = "ended" if row.started_at is not None else "failed"


def apply_call_event(db: Session, event: str, call: dict) -> str:
    """Idempotently fold one Retell lifecycle event into voice_calls.

    Returns a disposition string for the log line. Timestamps arrive as
    millisecond epochs (``start_timestamp``/``end_timestamp``); replays
    re-apply the same values — idempotent by construction. An event for a
    call we never minted is stored as a LOUD orphan (phone-lane compat),
    never silently dropped.
    """
    call_id = call.get("call_id")
    if not call_id:
        return "ignored_no_call_id"

    row, disposition = _call_row(db, str(call_id), event)
    if event == EVENT_CALL_STARTED:
        _apply_started(row, call)
    elif event == EVENT_CALL_ENDED:
        _apply_ended(row, call)
    elif event == EVENT_CALL_ANALYZED:
        # Post-call analysis carries no lifecycle change we bill on — ack only.
        disposition = "acknowledged"
    else:
        disposition = f"ignored_{event or 'unknown'}"

    db.commit()
    return disposition
