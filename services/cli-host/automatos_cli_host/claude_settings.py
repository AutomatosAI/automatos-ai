"""Per-session Claude Code settings: hooks to the host, and the trust decision.

Every session gets its own ``settings.json`` passed with ``--settings``. It
declares one hook command per lifecycle event — the host's shim, which forwards
the payload to the host's loopback socket and prints the host's answer. Nothing
in ``~/.claude`` is edited for hooks; the file is per session and disposable.

Two things ARE written into Claude Code's own state, deliberately and minimally,
because interactive mode has no flag for them:

* ``~/.claude.json`` → ``projects[<cwd>].hasTrustDialogAccepted = true`` — the
  folder-trust dialog. Registering a directory with the host IS the user's
  trust decision; we record it where Claude Code reads it (munder
  ``config.ts:790-822``), backup-first, atomic, touching nothing else.

Everything else (permission mode, allowed tools, model, system prompt) rides on
the command line — see ``session.py``.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

# Events the host listens to. PermissionRequest is the safety net: if a call
# ever reaches Claude's own prompt (nobody watches the TUI), it is denied
# loudly rather than parking the session.
HOOK_EVENTS: List[str] = [
    "SessionStart",
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "PermissionRequest",
    "Notification",
    "Stop",
    "SubagentStop",
    "PreCompact",
    "PostCompact",
    "SessionEnd",
]

# PreToolUse may HOLD while the approvals inbox answers; the CLI's default for a
# command hook is 600 s — we stay under it and deny on our own clock.
HOLD_TIMEOUT_SECONDS = 540
DEFAULT_HOOK_TIMEOUT_SECONDS = 60
_EVENT_TIMEOUTS = {"PreToolUse": HOLD_TIMEOUT_SECONDS, "PermissionRequest": HOLD_TIMEOUT_SECONDS}


def shim_command(python: Optional[str] = None) -> str:
    """The hook command: this interpreter, this package's shim module."""
    exe = python or sys.executable
    return f"{json.dumps(exe)} -m automatos_cli_host.hook_shim"


def build_settings(*, python: Optional[str] = None) -> Dict[str, Any]:
    """The per-session settings document (hooks only)."""
    cmd = shim_command(python)
    hooks: Dict[str, Any] = {}
    for event in HOOK_EVENTS:
        entry: Dict[str, Any] = {
            "hooks": [{
                "type": "command",
                "command": cmd,
                "timeout": _EVENT_TIMEOUTS.get(event, DEFAULT_HOOK_TIMEOUT_SECONDS),
            }]
        }
        if event in ("PreToolUse", "PostToolUse", "PermissionRequest"):
            entry["matcher"] = "*"
        hooks[event] = [entry]
    return {"hooks": hooks}


def write_settings(path: Path, *, python: Optional[str] = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(json.dumps(build_settings(python=python), indent=2) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)
    return path


# ── Claude Code's own state: the trust decision ──────────────────────────────

def claude_state_path(home: Optional[Path] = None) -> Path:
    return (home or Path.home()) / ".claude.json"


def read_claude_state(home: Optional[Path] = None) -> Dict[str, Any]:
    p = claude_state_path(home)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def has_completed_onboarding(home: Optional[Path] = None) -> bool:
    """Read-only: whether the user has run Claude Code interactively at least once.
    A fresh install shows onboarding dialogs no supervised session can answer."""
    return bool(read_claude_state(home).get("hasCompletedOnboarding"))


def is_directory_trusted(cwd: Path, home: Optional[Path] = None) -> bool:
    projects = read_claude_state(home).get("projects") or {}
    entry = projects.get(str(cwd)) if isinstance(projects, dict) else None
    return bool(isinstance(entry, dict) and entry.get("hasTrustDialogAccepted"))


def record_directory_trust(cwd: Path, home: Optional[Path] = None) -> bool:
    """Record the user's registration decision where Claude Code reads it.

    Returns True when the file was changed. Backup-first (``.claude.json.automatos-bak``),
    atomic replace, and ONLY the one flag under ``projects[<cwd>]`` is touched.
    """
    if is_directory_trusted(cwd, home):
        return False
    path = claude_state_path(home)
    state = read_claude_state(home)
    projects = state.get("projects")
    if not isinstance(projects, dict):
        projects = {}
    entry = projects.get(str(cwd))
    entry = dict(entry) if isinstance(entry, dict) else {}
    entry["hasTrustDialogAccepted"] = True
    projects = dict(projects)
    projects[str(cwd)] = entry
    new_state = dict(state)
    new_state["projects"] = projects

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.copy2(path, path.with_name(path.name + ".automatos-bak"))
    fd, tmp = tempfile.mkstemp(prefix=".claude.json.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(new_state, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        if path.exists():
            os.chmod(tmp, os.stat(path).st_mode & 0o777)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return True
