"""Claude Code — the native tier (design §3): its own ``--settings`` hooks and a
real ``--append-system-prompt-file``. Everything Claude-specific that used to sit
in ``session.py`` / ``claude_settings.py`` / ``policy.py`` lives here now,
moved, not rewritten:

* the per-session ``settings.json`` (hooks only) passed with ``--settings`` —
  nothing in ``~/.claude`` is edited for hooks; the file is per session and
  disposable;
* the trust decision: ``~/.claude.json`` → ``projects[<cwd>].hasTrustDialogAccepted``
  — registering a directory with the host IS the operator's trust decision; we
  record it where Claude Code reads it (munder ``config.ts:790-822``),
  backup-first, atomic, touching nothing else;
* the onboarding probe (read-only: has the operator ever run ``claude``?);
* the tool map — which of Claude's tools read, write, run a shell;
* the transcript: ``~/.claude/projects/<cwd-key>/<session_id>.jsonl``.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from ..presets import CliPreset
from ..transcript import last_assistant_text, read_usage, transcript_path
from .base import LaunchContext, Prepared, PresetAdapter, ToolClass, ToolIntent, hook_command

FILE_READ_TOOLS = frozenset({"Read", "Glob", "Grep", "LS"})
FILE_WRITE_TOOLS = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit"})
SHELL_TOOLS = frozenset({"Bash"})
WEB_TOOLS = frozenset({"WebFetch", "WebSearch"})
BENIGN_TOOLS = frozenset({"TodoWrite", "TodoRead", "AskUserQuestion"})
_PATH_KEYS = ("file_path", "notebook_path", "path")


def subject_of_input(tool_input: Any) -> Optional[str]:
    """The one thing a Claude tool call is about — a command, a path, a pattern.
    Never the whole tool input."""
    if not isinstance(tool_input, dict):
        return None
    for key in ("command", "file_path", "notebook_path", "path", "pattern", "url", "query"):
        value = tool_input.get(key)
        if value:
            return str(value)[:200]
    return None


# ── settings + trust (Claude Code's own state) ──────────────────────────────

def build_settings(preset: CliPreset, *, python: Optional[str] = None) -> Dict[str, Any]:
    """The per-session settings document (hooks only), one command hook per
    lifecycle event, all pointing at the host's shim."""
    cmd = hook_command(python)
    hooks: Dict[str, Any] = {}
    for event in sorted(preset.hook_events):
        entry: Dict[str, Any] = {"hooks": [{"type": "command", "command": cmd, "timeout": preset.hook_timeout(event)}]}
        if event in ("PreToolUse", "PostToolUse", "PermissionRequest"):
            entry["matcher"] = "*"
        hooks[event] = [entry]
    return {"hooks": hooks}


def write_settings(preset: CliPreset, path: Path, *, python: Optional[str] = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(json.dumps(build_settings(preset, python=python), indent=2) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)
    return path


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
    """Read-only: whether the operator has run Claude Code interactively at least
    once. A fresh install shows onboarding dialogs no supervised session can answer."""
    return bool(read_claude_state(home).get("hasCompletedOnboarding"))


def is_directory_trusted(cwd: Path, home: Optional[Path] = None) -> bool:
    projects = read_claude_state(home).get("projects") or {}
    entry = projects.get(str(cwd)) if isinstance(projects, dict) else None
    return bool(isinstance(entry, dict) and entry.get("hasTrustDialogAccepted"))


def record_directory_trust(cwd: Path, home: Optional[Path] = None) -> bool:
    """Record the operator's registration decision where Claude Code reads it.

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


# ── the adapter ─────────────────────────────────────────────────────────────

class ClaudeAdapter(PresetAdapter):
    def logged_in(self):
        return None if has_completed_onboarding() else super().logged_in()

    def detect(self) -> Dict[str, Any]:
        out = super().detect()
        out["onboarded"] = has_completed_onboarding() if out["path"] else False
        return out

    def prepare(self, ctx: LaunchContext) -> Prepared:
        """A hooks-only settings.json in the session dir (``--settings``), and the
        folder-trust decision recorded where Claude reads it."""
        settings_path = write_settings(self.preset, ctx.session_dir / "settings.json")
        record_directory_trust(ctx.cwd)
        return Prepared(args=["--settings", str(settings_path)])

    def tool_intent(self, tool_name: str, tool_input: Mapping[str, Any]) -> ToolIntent:
        ti = tool_input if isinstance(tool_input, Mapping) else {}
        if tool_name in FILE_WRITE_TOOLS or tool_name in FILE_READ_TOOLS:
            paths = tuple(str(ti[k]) for k in _PATH_KEYS if ti.get(k))
            cls = ToolClass.FILE_WRITE if tool_name in FILE_WRITE_TOOLS else ToolClass.FILE_READ
            return ToolIntent(tool=tool_name, cls=cls, paths=paths)
        if tool_name in SHELL_TOOLS:
            return ToolIntent(tool=tool_name, cls=ToolClass.SHELL, command=str(ti.get("command") or ""))
        if tool_name in WEB_TOOLS:
            return ToolIntent(tool=tool_name, cls=ToolClass.WEB, paths=tuple(str(ti[k]) for k in ("url", "query") if ti.get(k)))
        if tool_name in BENIGN_TOOLS:
            return ToolIntent(tool=tool_name, cls=ToolClass.BENIGN)
        return ToolIntent(tool=tool_name, cls=ToolClass.UNKNOWN)

    def read_usage(self, transcript: Path) -> Dict[str, Any]:
        return read_usage(transcript)

    def last_text(self, transcript: Path) -> Optional[str]:
        return last_assistant_text(transcript)

    def transcript_path(self, cwd: str, session_id: str, home: Optional[Path] = None) -> Optional[Path]:
        return transcript_path(cwd, session_id, home)

    def record_trust(self, cwd: Path, home: Optional[Path] = None) -> bool:
        return record_directory_trust(cwd, home)


__all__ = [
    "BENIGN_TOOLS", "ClaudeAdapter", "FILE_READ_TOOLS", "FILE_WRITE_TOOLS", "SHELL_TOOLS", "WEB_TOOLS",
    "build_settings", "claude_state_path", "has_completed_onboarding", "is_directory_trusted",
    "read_claude_state", "record_directory_trust", "subject_of_input", "write_settings",
]
