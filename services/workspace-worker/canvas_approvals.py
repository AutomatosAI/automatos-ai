"""Canvas approval gate — nothing applies without approval (PRD-170 S4).

The confinement gate (``canvas_confinement``) is a HARD tenancy boundary: a path
escape is denied outright, no human in the loop. This module is the SECOND gate —
the human approval loop the PRD's headline guarantee rests on: a mutating tool
call (file edit or shell command) pauses, surfaces a ``permission.request`` event
(a diff for file edits) to the UI, and the SDK ``can_use_tool`` callback AWAITS a
human approve/deny decision before the tool runs.

Split into a PURE policy + an async registry so the decision logic is
exhaustively unit-testable without the SDK, a container, or an event loop:

  * ``requires_approval`` — is this (confined) tool call one that must be gated?
  * ``build_permission_payload`` — the non-secret ``permission.request`` data
    (tool, path, and for a file edit the old/new content the DiffCard renders);
  * ``PendingApprovals`` — the asyncio registry: register a request → await its
    decision (indefinitely; the SDK imposes no callback timeout), resolved out of
    band by the worker's decision endpoint. Serialized SDK callbacks mean a single
    request is typically in flight, but requests are keyed by a generated
    ``request_id`` (the SDK exposes no correlation id) so the registry is robust to
    more than one and to a stop() that must fail every waiter.

NEVER carries secrets: only structural fields + file text the agent itself
supplied/where it wrote. Pure stdlib.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("workspace-worker.canvas")

# Tools whose calls mutate the workspace and therefore require approval. Mirrors
# canvas_events._FILE_EDIT_TOOLS plus the shell surface. Read-only navigation
# (Read/Glob/Grep) is NOT gated — it changes nothing and gating it would bury the
# user in noise and defeat the "streamed work" demo.
FILE_EDIT_TOOLS = frozenset({"Write", "Edit", "MultiEdit", "NotebookEdit"})
COMMAND_TOOLS = frozenset({"Bash"})
_APPROVAL_TOOLS = FILE_EDIT_TOOLS | COMMAND_TOOLS

# Where a file path lives in a tool input (mirrors canvas_events._PATH_KEYS).
_PATH_KEYS = ("file_path", "path", "notebook_path", "target_file")


def new_request_id() -> str:
    """A short, non-secret id correlating a permission request to its decision."""
    return f"perm_{uuid.uuid4().hex[:16]}"


def requires_approval(tool_name: str, auto_accept_edits: bool) -> bool:
    """Whether a (already confinement-passed) tool call must be human-approved.

    Auto-accept (session-scoped, default OFF) short-circuits FILE EDITS only — a
    shell command ALWAYS requires an explicit decision, so a destructive command
    can never be silently auto-run even with auto-accept on (matches the frontend
    ``diffApproval`` model exactly).
    """
    if tool_name in COMMAND_TOOLS:
        return True
    if tool_name in FILE_EDIT_TOOLS:
        return not auto_accept_edits
    return False


def _extract_path(tool_input: Any) -> Optional[str]:
    if not isinstance(tool_input, dict):
        return None
    for key in _PATH_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _read_text_safe(path: Path, limit: int = 200_000) -> str:
    """Best-effort read of the CURRENT file content for the diff's 'before'.

    A new file (or an unreadable/oversized one) yields ``""`` — the DiffCard then
    renders an all-additions diff, which is the honest picture. Never raises.
    """
    try:
        if not path.is_file():
            return ""
        if path.stat().st_size > limit:
            return ""
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def build_permission_payload(
    tool_name: str,
    tool_input: Any,
    request_id: str,
    root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build the non-secret ``permission.request`` data dict for the UI card.

    For a file edit, includes ``old_content``/``new_content`` so the DiffCard can
    render a diff:
      * Write  → old = current file text (``""`` if new), new = supplied content;
      * Edit   → old = ``old_string``,  new = ``new_string`` (the changed hunk —
                 the standard Claude Code edit-card shape, not a whole-file diff);
      * MultiEdit / NotebookEdit → no inline diff (multiple hunks); the card shows
        tool + path and the user approves the edit set.
    A bash/command tool carries the command string (already inside the mount —
    confinement ran first) and no diff.
    """
    path = _extract_path(tool_input)
    data: Dict[str, Any] = {
        "request_id": request_id,
        "tool_name": tool_name,
        "path": path,
    }
    ti = tool_input if isinstance(tool_input, dict) else {}

    if tool_name == "Write" and path:
        abs_path = Path(ti.get("file_path") or path)
        old = _read_text_safe(abs_path) if abs_path.is_absolute() else ""
        if not old and root is not None:
            old = _read_text_safe((root / path))
        content = ti.get("content")
        data["old_content"] = old
        data["new_content"] = content if isinstance(content, str) else ""
    elif tool_name == "Edit" and path:
        old_s = ti.get("old_string")
        new_s = ti.get("new_string")
        data["old_content"] = old_s if isinstance(old_s, str) else ""
        data["new_content"] = new_s if isinstance(new_s, str) else ""
    elif tool_name in COMMAND_TOOLS:
        cmd = ti.get("command")
        data["command"] = cmd if isinstance(cmd, str) else ""

    return data


@dataclass
class _Pending:
    future: "asyncio.Future[bool]"
    tool_name: str


class PendingApprovals:
    """asyncio registry bridging an awaiting ``can_use_tool`` callback to an
    out-of-band human decision (the worker's decision endpoint resolves it).

    One instance per session. The callback ``register``s a request and ``await``s
    the returned future; the decision handler calls ``resolve``. ``fail_all`` is
    called on session stop so no callback is left hanging.
    """

    def __init__(self) -> None:
        self._pending: Dict[str, _Pending] = {}
        self._lock = asyncio.Lock()

    async def register(self, request_id: str, tool_name: str) -> "asyncio.Future[bool]":
        loop = asyncio.get_running_loop()
        future: "asyncio.Future[bool]" = loop.create_future()
        async with self._lock:
            self._pending[request_id] = _Pending(future=future, tool_name=tool_name)
        return future

    async def resolve(self, request_id: str, approved: bool) -> bool:
        """Resolve a pending request. Returns True if a waiter was found."""
        async with self._lock:
            pending = self._pending.pop(request_id, None)
        if pending is None:
            return False
        if not pending.future.done():
            pending.future.set_result(approved)
        return True

    async def fail_all(self) -> None:
        """Deny every outstanding request (session stopping / dying)."""
        async with self._lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for p in pending:
            if not p.future.done():
                p.future.set_result(False)

    def pending_ids(self) -> Dict[str, str]:
        """Snapshot of outstanding request_id → tool_name (diagnostics/tests)."""
        return {rid: p.tool_name for rid, p in self._pending.items()}
