"""
PRD-207 S3: Retell call-lifecycle events → ``voice_calls``
===========================================================

The meter's write path: one Retell lifecycle event (``call_started``,
``call_ended``, ``call_analyzed``) folded idempotently into the call's
``voice_calls`` row. The HMAC check and the 2xx answer stay in the route
(``api/voice_retell.py``); this module only applies a verified event.
Moved out of the route by PRD-258 US-001 (D6: that file never grows).

PRD-258 US-002: a call's own latency (Retell's p50/p90 per stage) is kept on
the row as ``vendor_latency``; when ``call_analyzed`` arrives without it, one
get-call fetches it. A failed fetch is logged and never fails the webhook.

PRD-265 D-6: ``call_ended`` also charges the call to its workspace's credits
(``live_billing.charge_call``), in the same transaction as the row, once per call.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from modules.voice import live_billing, live_settings, retell_api

logger = logging.getLogger(__name__)

# Lifecycle events Retell posts to the webhook.
EVENT_CALL_STARTED = "call_started"
EVENT_CALL_ENDED = "call_ended"
EVENT_CALL_ANALYZED = "call_analyzed"
# A ``disconnect_reason`` longer than the column is cut, never refused.
_DISCONNECT_REASON_MAX = 64
# Retell's call object carries per-stage latency under this key.
CALL_LATENCY_KEY = "latency"
# The stages and the statistics of each that voice_calls.vendor_latency keeps;
# everything else Retell sends (min, max, p95, the raw values) is dropped.
VENDOR_LATENCY_STAGES = ("e2e", "asr", "llm", "llm_websocket_network_rtt", "tts")
VENDOR_LATENCY_STATS = ("p50", "p90", "num")


def _ts(ms: Any) -> Optional[datetime]:
    """A Retell millisecond epoch as a naive UTC datetime, or None when unreadable."""
    try:
        return datetime.fromtimestamp(int(ms) / 1000.0, tz=timezone.utc).replace(tzinfo=None)
    except (TypeError, ValueError, OSError):
        return None


def _is_number(value: Any) -> bool:
    """A finite int or float; a bool is not a number here."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def _stage_subset(stage: Any) -> Optional[Dict[str, Any]]:
    """One stage's kept statistics, or None when it has no numeric one."""
    if not isinstance(stage, dict):
        return None
    kept = {stat: stage[stat] for stat in VENDOR_LATENCY_STATS if _is_number(stage.get(stat))}
    return kept or None


def vendor_latency_subset(latency: Any) -> Optional[Dict[str, Dict[str, Any]]]:
    """Pure: a Retell call's ``latency`` block → what ``vendor_latency`` stores.

    For each of ``VENDOR_LATENCY_STAGES`` only ``VENDOR_LATENCY_STATS``, and
    only finite numbers. An empty or malformed block → None, never an empty
    dict standing in for a measurement.
    """
    if not isinstance(latency, dict):
        return None
    stages = {name: _stage_subset(latency.get(name)) for name in VENDOR_LATENCY_STAGES}
    kept = {name: stats for name, stats in stages.items() if stats}
    return kept or None


async def _fetch_latency(call_id: str) -> Any:
    """The call's ``latency`` block from one get-call, or None (logged) on any failure."""
    api_key = live_settings.retell_credentials().api_key
    if not api_key:
        logger.warning("voice_live_get_call_skipped call=%s reason=no_api_key", call_id)
        return None
    try:
        fetched = await retell_api.get_call(api_key, call_id)
    except retell_api.RetellApiError as exc:
        logger.warning("voice_live_get_call_failed call=%s reason=%s", call_id, exc)
        return None
    return fetched.get(CALL_LATENCY_KEY)


async def with_vendor_latency(event: str, call: Any) -> Dict[str, Any]:
    """The webhook's call object, with its latency fetched when Retell left it out.

    Only a ``call_analyzed`` event with no ``latency`` block fetches, once.
    Returns a new dict (the payload is never mutated); a failed fetch returns
    the call as it came, so the webhook still answers 2xx.
    """
    call = call if isinstance(call, dict) else {}
    call_id = call.get("call_id")
    if event != EVENT_CALL_ANALYZED or call.get(CALL_LATENCY_KEY) or not call_id:
        return call
    latency = await _fetch_latency(str(call_id))
    return {**call, CALL_LATENCY_KEY: latency} if latency else call


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


def _apply_latency(row, call: dict) -> None:
    """Store the call's latency subset; a missing one never overwrites a stored value."""
    latency = vendor_latency_subset(call.get(CALL_LATENCY_KEY))
    if latency is not None:
        row.vendor_latency = latency


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
    charged = 0
    if event == EVENT_CALL_STARTED:
        _apply_started(row, call)
    elif event == EVENT_CALL_ENDED:
        _apply_ended(row, call)
        _apply_latency(row, call)
        charged = live_billing.charge_call(db, row, call)
    elif event == EVENT_CALL_ANALYZED:
        # Post-call analysis carries no lifecycle change we bill on; only
        # Retell's own latency is kept (PRD-258 US-002).
        _apply_latency(row, call)
        disposition = "acknowledged"
    else:
        disposition = f"ignored_{event or 'unknown'}"

    db.commit()
    if charged:
        live_billing.after_charge(db, row.workspace_id)
    return disposition
