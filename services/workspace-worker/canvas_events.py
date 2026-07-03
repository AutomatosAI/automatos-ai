"""
Canvas Session Event Schema + Serializer
========================================
PRD-170 S3: bridge headless Claude Agent SDK session messages onto the
platform's existing Redis pub/sub event shape so the session panel can render
streaming turns beside the file tree.

The wire envelope reuses the worker's task-event shape (``_publish_event`` in
main.py): ``{event_type, workspace_id, data, timestamp, schema_version}``. The
``event_type`` values here are a CLOSED, VERSIONED vocabulary — the frontend
validator (``frontend/.../canvasEvents.ts``) mirrors it and a vitest fails on
any drift, so a renamed event on either side is caught before it silently drops
a turn in the UI.

Design constraints:
  * Pure stdlib. Duck-typed against the SDK message surface (AssistantMessage,
    UserMessage, ResultMessage, and the TextBlock / ToolUseBlock / ToolResult
    content blocks) so this stays importable and unit-testable WITHOUT the
    ``claude_agent_sdk`` package or a container.
  * NEVER emit secrets. Only structural fields (tool name, path, event kind,
    text) are surfaced; raw tool_use ``input`` dicts are summarised to a path +
    a size, not echoed verbatim, and no token/credential material is carried.
  * A message the serializer does not recognise yields ``None`` (skipped), not a
    crash and not an ``unknown`` event — the vocabulary stays closed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Bump when the envelope shape or the event vocabulary changes in a
# non-additive way. The frontend asserts equality against this literal.
CANVAS_EVENT_SCHEMA_VERSION = 1

# The CLOSED set of canvas event types. Frontend mirrors this exactly.
EVENT_SESSION_STATUS = "canvas.session.status"
EVENT_ASSISTANT_TEXT = "canvas.assistant.text"
EVENT_TOOL_CALL = "canvas.tool.call"
EVENT_TOOL_RESULT = "canvas.tool.result"
EVENT_FILE_EDIT = "canvas.file.edit"
EVENT_PERMISSION_REQUEST = "canvas.permission.request"
EVENT_TURN_COMPLETE = "canvas.turn.complete"

CANVAS_EVENT_TYPES = frozenset(
    {
        EVENT_SESSION_STATUS,
        EVENT_ASSISTANT_TEXT,
        EVENT_TOOL_CALL,
        EVENT_TOOL_RESULT,
        EVENT_FILE_EDIT,
        EVENT_PERMISSION_REQUEST,
        EVENT_TURN_COMPLETE,
    }
)

# Tool names (Claude Code surface) whose calls mutate files — surfaced as a
# dedicated file-edit event so the tree can live-refresh (S3 AC).
_FILE_EDIT_TOOLS = frozenset({"Write", "Edit", "MultiEdit", "NotebookEdit"})

# Where a file path lives inside a given tool's input (mirrors
# canvas_confinement.PATH_INPUT_KEYS ordering intent).
_PATH_KEYS = ("file_path", "path", "notebook_path", "target_file")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _envelope(workspace_id: str, event_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Build one wire event. ``event_type`` MUST be in CANVAS_EVENT_TYPES."""
    if event_type not in CANVAS_EVENT_TYPES:
        raise ValueError(f"Unknown canvas event type: {event_type!r}")
    return {
        "schema_version": CANVAS_EVENT_SCHEMA_VERSION,
        "event_type": event_type,
        "workspace_id": workspace_id,
        "data": data,
        "timestamp": _utcnow(),
    }


def _extract_path(tool_input: Any) -> Optional[str]:
    """Return the first path-bearing field in a tool input, if any."""
    if not isinstance(tool_input, dict):
        return None
    for key in _PATH_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def session_status_event(workspace_id: str, status: str, error: Optional[str] = None) -> Dict[str, Any]:
    """Explicit lifecycle event (start/running/stopped/failed) for the panel."""
    data: Dict[str, Any] = {"status": status}
    if error:
        data["error"] = error
    return _envelope(workspace_id, EVENT_SESSION_STATUS, data)


def _block_type(block: Any) -> str:
    """Duck-typed discriminator for an SDK content block."""
    explicit = getattr(block, "type", None)
    if isinstance(explicit, str):
        return explicit
    name = type(block).__name__
    if name.endswith("Block"):
        name = name[: -len("Block")]
    return name.lower()


def _serialize_block(workspace_id: str, block: Any) -> List[Dict[str, Any]]:
    """Map one assistant content block to zero or more canvas events."""
    kind = _block_type(block)

    if kind in ("text",):
        text = getattr(block, "text", None)
        if isinstance(text, str) and text:
            return [_envelope(workspace_id, EVENT_ASSISTANT_TEXT, {"text": text})]
        return []

    if kind in ("tooluse", "tool_use"):
        tool_name = getattr(block, "name", None) or ""
        tool_id = getattr(block, "id", None)
        tool_input = getattr(block, "input", None)
        path = _extract_path(tool_input)
        events: List[Dict[str, Any]] = [
            _envelope(
                workspace_id,
                EVENT_TOOL_CALL,
                {"tool_name": tool_name, "tool_id": tool_id, "path": path},
            )
        ]
        # A file-mutating tool ALSO emits a file-edit event so the tree refreshes.
        if tool_name in _FILE_EDIT_TOOLS and path:
            events.append(
                _envelope(
                    workspace_id,
                    EVENT_FILE_EDIT,
                    {"tool_name": tool_name, "tool_id": tool_id, "path": path},
                )
            )
        return events

    if kind in ("toolresult", "tool_result"):
        tool_id = getattr(block, "tool_use_id", None) or getattr(block, "id", None)
        is_error = bool(getattr(block, "is_error", False))
        return [
            _envelope(
                workspace_id,
                EVENT_TOOL_RESULT,
                {"tool_id": tool_id, "is_error": is_error},
            )
        ]

    return []


def _message_role(message: Any) -> str:
    """Duck-typed role/kind for an SDK message."""
    explicit = getattr(message, "role", None)
    if isinstance(explicit, str):
        return explicit
    name = type(message).__name__.lower()
    if "assistant" in name:
        return "assistant"
    if "result" in name:
        return "result"
    if "user" in name:
        return "user"
    if "system" in name:
        return "system"
    return name


def serialize_sdk_message(workspace_id: str, message: Any) -> List[Dict[str, Any]]:
    """Map one SDK message to an ordered list of canvas wire events.

    Unknown / uninteresting messages (init SystemMessage, user echoes) yield an
    empty list — they are skipped, never surfaced as an ``unknown`` type.
    """
    role = _message_role(message)

    if role == "assistant":
        content = getattr(message, "content", None) or []
        out: List[Dict[str, Any]] = []
        for block in content:
            out.extend(_serialize_block(workspace_id, block))
        return out

    if role == "result":
        # End-of-turn marker; carry only non-secret usage/summary fields.
        data: Dict[str, Any] = {}
        is_error = getattr(message, "is_error", None)
        if is_error is not None:
            data["is_error"] = bool(is_error)
        for attr in ("num_turns", "duration_ms"):
            value = getattr(message, attr, None)
            if isinstance(value, (int, float)):
                data[attr] = value
        return [_envelope(workspace_id, EVENT_TURN_COMPLETE, data)]

    return []


def permission_request_event(
    workspace_id: str,
    tool_name: str,
    path: Optional[str] = None,
    request_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """A tool-permission prompt surfaced as an approval card (S4 consumes it).

    ``extra`` carries the non-secret render payload the DiffCard needs — for a
    file edit, ``old_content``/``new_content``; for a bash tool, ``command``. It
    is merged into ``data`` (the base fields win on key collision). NEVER pass
    secrets here: only agent-supplied file text / the command it wants to run.
    """
    data: Dict[str, Any] = {"tool_name": tool_name, "path": path, "request_id": request_id}
    if extra:
        for key, value in extra.items():
            data.setdefault(key, value)
    return _envelope(workspace_id, EVENT_PERMISSION_REQUEST, data)
