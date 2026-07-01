"""
Canvas Session Confinement
==========================
PRD-170 S1: server-side tenancy gate for the headless Claude Agent SDK
session that runs per workspace in this worker container.

Every path the agent supplies through a tool call is re-bound/validated
against the workspace root BEFORE the tool runs:

  * relative paths are re-bound to absolute paths under the workspace root;
  * absolute paths are allowed only when they resolve INSIDE the root
    (the SDK session's cwd is the workspace root, so in-root absolute
    paths are legitimate — unlike the file-browser API, whose
    ``WorkspaceManager.resolve_safe_path`` rejects all absolute paths
    because its callers always send workspace-relative paths);
  * ``..`` traversal, symlink escapes and null bytes are rejected with
    the worker's canonical ``SecurityError`` (same resolve+relative_to
    idiom as ``WorkspaceManager.resolve_safe_path``);
  * bash commands are rejected when they reference the workspaces volume
    outside this workspace (cross-tenant) or contain a parent-directory
    path component.

Pure stdlib — unit-testable without the SDK, a container, or a DB.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from workspace_manager import SecurityError

# Tool-input keys that carry filesystem paths (Claude Code tool surface:
# Read/Write/Edit -> file_path, Glob/Grep -> path, NotebookEdit ->
# notebook_path, Bash -> cwd; generic fallbacks included).
PATH_INPUT_KEYS = (
    "file_path",
    "path",
    "notebook_path",
    "cwd",
    "directory",
    "target_file",
)

# Tool-input keys that carry shell command strings (Bash tool).
COMMAND_INPUT_KEYS = ("command",)

# ``..`` as a standalone path component inside a command string
# (start/whitespace/quote/separator on the left, /, \\, separator or end
# on the right). Conservative: blocks `cd ..`, `cat ../x`, `ls a/../../b`.
_PARENT_COMPONENT_RE = re.compile(r"""(^|[\s'"=:;|&(])\.\.([/\\]|[\s'");|&]|$)""")

# Characters that legitimately precede / follow a path inside a command.
_BOUNDARY_CHARS = " \t\n'\"=:;|&()"


def confine_path(root: Path, supplied: str) -> Path:
    """Resolve *supplied* and guarantee it stays inside *root*.

    Returns the re-bound absolute path on success.

    Raises:
        SecurityError: null byte, ``..`` traversal, symlink escape, or an
            absolute path outside the workspace root.
    """
    if "\x00" in supplied:
        raise SecurityError("Null byte in path")

    base = root.resolve()
    candidate = Path(supplied)
    resolved = (candidate if candidate.is_absolute() else base / candidate).resolve()

    try:
        resolved.relative_to(base)
    except ValueError:
        raise SecurityError(
            f"Path escapes the workspace mount: {supplied!r}"
        ) from None

    return resolved


def _command_escape_reason(command: str, root: Path) -> Optional[str]:
    """Return a denial reason if *command* references paths outside the
    workspace mount, else ``None``.

    Two gates:
      1. Cross-tenant: any reference to the workspaces volume root that is
         not immediately scoped to THIS workspace's directory.
      2. Traversal: any ``..`` path component.

    Absolute system paths (``/usr/bin/python3`` …) stay allowed — parity
    with the existing ``workspace_files`` POST /exec surface, where the
    container is the boundary for system files.
    """
    base = root.resolve()
    volume_root = base.parent
    vstr = str(volume_root)
    own_prefix = f"{vstr}/{base.name}"

    if vstr and vstr != "/":
        for match in re.finditer(re.escape(vstr), command):
            start = match.start()
            if start > 0 and command[start - 1] not in _BOUNDARY_CHARS:
                # Substring of a longer, unrelated path (e.g. /data/workspaces).
                continue
            rest = command[start:]
            if not rest.startswith(own_prefix):
                return (
                    "command references the workspaces volume outside this "
                    "workspace"
                )
            tail = rest[len(own_prefix):]
            if tail and tail[0] != "/" and tail[0] not in _BOUNDARY_CHARS:
                # Workspace-id prefix collision (e.g. <id>-other).
                return (
                    "command references the workspaces volume outside this "
                    "workspace"
                )

    if _PARENT_COMPONENT_RE.search(command):
        return "command contains a parent-directory ('..') path component"

    return None


@dataclass(frozen=True)
class ConfinementVerdict:
    """Outcome of confining one tool call to the workspace mount."""

    allowed: bool
    updated_input: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None


def evaluate_tool_confinement(
    tool_name: str,
    tool_input: Any,
    root: Path,
) -> ConfinementVerdict:
    """Validate every agent-supplied path in *tool_input* against *root*.

    Returns an allow verdict (with relative paths re-bound to absolute
    in-root paths via ``updated_input``) or a deny verdict with a reason.
    Never mutates *tool_input*.
    """
    if not isinstance(tool_input, dict):
        return ConfinementVerdict(
            allowed=False,
            reason=f"{tool_name}: tool input must be an object",
        )

    rebound: Dict[str, Any] = {}

    for key in PATH_INPUT_KEYS:
        value = tool_input.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        try:
            resolved = confine_path(root, value)
        except SecurityError as exc:
            return ConfinementVerdict(
                allowed=False,
                reason=f"{tool_name}.{key}: {exc}",
            )
        if str(resolved) != value:
            rebound[key] = str(resolved)

    for key in COMMAND_INPUT_KEYS:
        value = tool_input.get(key)
        if not isinstance(value, str):
            continue
        reason = _command_escape_reason(value, root)
        if reason:
            return ConfinementVerdict(
                allowed=False,
                reason=f"{tool_name}.{key}: {reason}",
            )

    if rebound:
        return ConfinementVerdict(allowed=True, updated_input={**tool_input, **rebound})
    return ConfinementVerdict(allowed=True)
