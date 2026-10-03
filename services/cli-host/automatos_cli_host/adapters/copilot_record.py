"""GitHub Copilot's session record (PRD-253 S1.5, D8, S2.1).

``$COPILOT_HOME/session-state/<session id>/events.jsonl`` — one JSON event per line,
``{"type": …, "data": {…}}``. From it the host reads:

* **usage** — every ``assistant.usage`` event, summed per model: tokens in the
  host's normalized names, plus the plan's own units: ``ai_credits`` (from
  ``copilotUsage.totalNanoAiu``) and ``premium_requests`` (``session.shutdown``,
  legacy billing). Never a price;
* **the final text** — the last ``assistant.message``;
* **whether the Automatos MCP server failed to load** — the last status the
  record gives it (an organisation's MCP policy can refuse it).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

SESSION_STATE_DIRNAME = "session-state"
EVENTS_FILENAME = "events.jsonl"
NANO = 1_000_000_000          # ``totalNanoAiu`` counts billionths of an AI credit
_TOKEN_FIELDS = (             # Copilot's name → the host's normalized name
    ("inputTokens", "input_tokens"),
    ("outputTokens", "output_tokens"),
    ("cacheReadTokens", "cache_read_input_tokens"),
    ("cacheWriteTokens", "cache_creation_input_tokens"),
)
# ``McpServerStatus`` (1.0.91): connected | failed | needs-auth | pending | disabled | stopped | not_configured.
_MCP_FAILED_STATES = frozenset({"failed", "needs-auth", "disabled", "not_configured"})


def events_path(home: Path, session_id: str) -> Path:
    return home / SESSION_STATE_DIRNAME / session_id / EVENTS_FILENAME


def _events(path: Path) -> Iterator[Dict[str, Any]]:
    """Every JSON object in the record, in order; a line that is not one is skipped."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return
    for line in lines:
        try:
            event = json.loads(line) if line.strip() else None
        except ValueError:
            continue
        if isinstance(event, dict):
            yield event


def _data(event: Dict[str, Any]) -> Dict[str, Any]:
    data = event.get("data")
    return data if isinstance(data, dict) else event


def _n(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and value == value else 0


def _empty() -> Dict[str, Any]:
    out: Dict[str, Any] = {name: 0 for _, name in _TOKEN_FIELDS}
    return {**out, "reasoning_output_tokens": 0, "assistant_messages": 0, "model": None, "per_model": {},
            "total_tokens": 0}


def _add_usage(totals: Dict[str, Any], data: Dict[str, Any], nano_aiu: int) -> int:
    model = str(data.get("model") or totals["model"] or "unknown")
    bucket = dict(totals["per_model"].get(model) or {name: 0 for _, name in _TOKEN_FIELDS})
    for source, name in _TOKEN_FIELDS:
        totals[name] += _n(data.get(source))
        bucket[name] += _n(data.get(source))
    totals["reasoning_output_tokens"] += _n(data.get("reasoningTokens"))
    totals["per_model"] = {**totals["per_model"], model: bucket}
    totals["model"] = model
    copilot_usage = data.get("copilotUsage") if isinstance(data.get("copilotUsage"), dict) else {}
    return nano_aiu + _n(copilot_usage.get("totalNanoAiu"))


def read_events_usage(path: Path) -> Dict[str, Any]:
    """What one session used, in the host's normalized shape (``read_usage``)."""
    totals = _empty()
    nano_aiu = 0
    premium: Optional[float] = None
    for event in _events(path):
        kind = event.get("type")
        if kind == "assistant.usage":
            nano_aiu = _add_usage(totals, _data(event), nano_aiu)
        elif kind == "assistant.message":
            totals["assistant_messages"] += 1
        elif kind == "session.shutdown":
            requests = _data(event).get("totalPremiumRequests")
            premium = requests if isinstance(requests, (int, float)) else premium
    totals["total_tokens"] = totals["input_tokens"] + totals["output_tokens"]
    if nano_aiu:
        totals["ai_credits"] = round(nano_aiu / NANO, 6)
    if premium is not None:
        totals["premium_requests"] = premium
    return totals


def last_message(path: Path) -> Optional[str]:
    """The session's final answer: the last non-empty ``assistant.message``."""
    text: Optional[str] = None
    for event in _events(path):
        if event.get("type") != "assistant.message":
            continue
        content = _data(event).get("content")
        candidate = content if isinstance(content, str) else ""
        if candidate.strip():
            text = candidate.strip()
    return text


def mcp_server_blocked(path: Path, server: str) -> Optional[str]:
    """The status ``server`` ended in when the record says it did not load — an
    organisation's registry-only MCP policy refuses a server it does not list —
    else None. ``session.mcp_servers_loaded`` lists ``servers[{name, status}]``;
    ``session.mcp_server_status_changed`` names one ``{serverName, status}``."""
    status: Optional[str] = None
    for event in _events(path):
        kind, data = event.get("type"), _data(event)
        if kind == "session.mcp_servers_loaded":
            entries = [s for s in (data.get("servers") or []) if isinstance(s, dict) and s.get("name") == server]
        elif kind == "session.mcp_server_status_changed" and data.get("serverName") == server:
            entries = [data]
        else:
            continue
        for entry in entries:
            status = str(entry.get("status") or "") or status
    return status if status in _MCP_FAILED_STATES else None


__all__ = ["EVENTS_FILENAME", "SESSION_STATE_DIRNAME", "events_path", "last_message", "mcp_server_blocked",
           "read_events_usage"]
