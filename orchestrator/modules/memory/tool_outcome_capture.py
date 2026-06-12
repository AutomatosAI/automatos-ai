"""Tool-execution outcome capture (PRD-159 S2).

A post-execution hook turns notable tool outcomes — failures and *notable*
successes (new channel/record ids, auth quirks, rate limits, schema surprises)
— into typed ``tool_outcome`` memories under the workspace namespace, written
direct (``infer:false`` via ``store_two_tier``) and deduped by content-hash.

Design seams (so this is unit-testable without a DB / event-loop):
  - ``build_tool_outcome``   — pure: result → outcome record or None (noise gate)
  - ``should_dedupe``        — content-hash dedup against a bounded in-process set
  - ``write_tool_outcome``   — async: persist one record via UnifiedMemoryService
  - ``capture_tool_outcome`` — fire-and-forget entry called from the executor

Never raises into the caller: the tool call must not fail because memory did.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import OrderedDict
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Memory category for these records (subset of the PRD-159 taxonomy).
TOOL_OUTCOME_TYPE = "tool_outcome"

# Keys that, when present in a successful result's payload, mark it "notable"
# (a resource was created / an external id surfaced). Trivial successes (reads,
# searches with none of these) are gated OUT — that is the noise gate.
_NOTABLE_SUCCESS_KEYS = frozenset({
    "id", "ids", "channel", "channel_id", "ts", "url", "number", "sid",
    "message_id", "thread_ts", "record_id", "issue_key", "pr_number",
})

# Bounded in-process dedup of recently-seen outcome hashes (identical outcome
# twice → one row). Bounded so a long-running worker never grows unbounded.
_SEEN_HASHES: "OrderedDict[str, None]" = OrderedDict()
_SEEN_MAX = 1024


def _classify_error(error: str) -> str:
    """Map a raw error string to a coarse, stable class for dedup + recall."""
    e = (error or "").lower()
    if any(k in e for k in ("rate limit", "rate_limit", "429", "too many requests")):
        return "rate_limit"
    if any(k in e for k in ("unauthor", "auth", "token", "401", "403", "forbidden", "permission")):
        return "auth"
    if any(k in e for k in ("not_in_channel", "not in channel", "not_found", "not found", "404", "missing")):
        return "not_found"
    if any(k in e for k in ("timeout", "timed out", "deadline")):
        return "timeout"
    if any(k in e for k in ("schema", "invalid", "validation", "required", "bad request", "400")):
        return "schema"
    return "error"


def _flatten_top_keys(payload: Any) -> set:
    """Top-level keys of a dict payload (or of each item if it's a list)."""
    keys: set = set()
    if isinstance(payload, dict):
        keys.update(payload.keys())
        # one level down — Composio often nests under data/response_data
        for v in payload.values():
            if isinstance(v, dict):
                keys.update(v.keys())
    elif isinstance(payload, list):
        for item in payload[:5]:
            if isinstance(item, dict):
                keys.update(item.keys())
    return {str(k).lower() for k in keys}


def _is_notable_success(result: Dict[str, Any]) -> bool:
    payload = result.get("data")
    if payload is None:
        payload = result.get("result")
    return bool(_flatten_top_keys(payload) & _NOTABLE_SUCCESS_KEYS)


def _content_hash(workspace_id: str, app: str, action: str, signature: str) -> str:
    raw = f"{workspace_id}|{app}|{action}|{signature}".encode("utf-8", "ignore")
    return hashlib.sha256(raw).hexdigest()


def build_tool_outcome(
    *,
    tool_name: str,
    parameters: Dict[str, Any],
    result: Dict[str, Any],
    workspace_id: str,
) -> Optional[Dict[str, Any]]:
    """Pure noise gate + record builder.

    Returns a ``{fact, type, importance, metadata}`` record for a failure or a
    notable success, or ``None`` when the outcome is trivial (no write).
    """
    if not isinstance(result, dict) or not workspace_id:
        return None

    params = parameters if isinstance(parameters, dict) else {}
    action = str(params.get("action") or tool_name or "unknown_action")
    app = str(params.get("app_name") or result.get("app") or "platform")
    success = bool(result.get("success"))

    if success:
        if not _is_notable_success(result):
            return None  # noise gate: trivial success → no memory
        signature = "success:notable"
        fact = f"Tool {action} ({app}) succeeded with a notable result."
        importance = 0.5
        error_class = ""
    else:
        error = str(result.get("error") or "unknown error")
        error_class = _classify_error(error)
        signature = f"fail:{error_class}"
        short = error.strip().splitlines()[0][:160] if error else ""
        fact = f"Tool {action} ({app}) failed [{error_class}]: {short}"
        importance = 0.6

    outcome_hash = _content_hash(workspace_id, app, action, signature)
    return {
        "fact": fact,
        "type": TOOL_OUTCOME_TYPE,
        "importance": importance,
        "metadata": {
            "category": TOOL_OUTCOME_TYPE,
            "importance": importance,
            "app": app,
            "action": action,
            "error_class": error_class,
            "success": success,
            "outcome_hash": outcome_hash,
        },
    }


def should_dedupe(outcome_hash: str) -> bool:
    """True if this hash was seen recently (caller should skip the write)."""
    if outcome_hash in _SEEN_HASHES:
        _SEEN_HASHES.move_to_end(outcome_hash)
        return True
    _SEEN_HASHES[outcome_hash] = None
    while len(_SEEN_HASHES) > _SEEN_MAX:
        _SEEN_HASHES.popitem(last=False)
    return False


async def write_tool_outcome(
    record: Dict[str, Any],
    *,
    workspace_id: str,
    agent_id: Optional[int],
    service: Any = None,
) -> bool:
    """Persist one outcome record to L3 (global tier, infer:false). Best-effort."""
    try:
        if service is None:
            from modules.memory.unified_memory_service import get_unified_memory_service
            service = get_unified_memory_service()
        await service.store_two_tier(
            workspace_id=str(workspace_id),
            messages=[{"role": "user", "content": record["fact"]}],
            agent_id=agent_id,
            tier="global",
            metadata=record["metadata"],
        )
        return True
    except Exception:
        logger.warning("[ToolOutcome] write failed", exc_info=True)
        return False


def capture_tool_outcome(
    *,
    tool_name: str,
    parameters: Dict[str, Any],
    result: Dict[str, Any],
    workspace_id: Any,
    agent_id: Optional[int],
) -> Optional["asyncio.Task"]:
    """Fire-and-forget entry. Builds + dedupes, then schedules the async write.

    Returns the scheduled task (so tests can await it) or ``None`` when gated out
    / deduped / no event loop. Never raises.
    """
    try:
        if not workspace_id:
            return None
        record = build_tool_outcome(
            tool_name=tool_name,
            parameters=parameters,
            result=result,
            workspace_id=str(workspace_id),
        )
        if record is None:
            return None
        if should_dedupe(record["metadata"]["outcome_hash"]):
            return None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None
        return loop.create_task(
            write_tool_outcome(record, workspace_id=str(workspace_id), agent_id=agent_id)
        )
    except Exception:
        logger.debug("[ToolOutcome] capture skipped", exc_info=True)
        return None
