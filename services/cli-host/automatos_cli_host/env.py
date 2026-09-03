"""The session environment: the user's login-shell PATH, minus what must not leak.

Ported from munder-difflin (``shellEnv.ts``, ``ptyEnv.ts``) and tightened for
the PRD-234 §Terms invariant:

* PATH comes from the user's INTERACTIVE login shell (nvm/asdf/brew edits live
  there; a host started by launchd or from inside another program does not
  have them), captured once, fenced so rc-file chatter cannot poison it.
* Every ``CLAUDE*`` session marker is stripped — this host is often started
  from inside a Claude Code terminal, and an inherited
  ``CLAUDE_CODE_CHILD_SESSION`` silently disables transcript saving (which
  breaks ``--resume``). The operator's own configuration keys are kept.
* ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_AUTH_TOKEN`` / ``ANTHROPIC_BASE_URL`` are
  stripped: a session must bill the user's plan, never a key, and never be
  redirected through a proxy.
* ``CLAUDE_CODE_ENTRYPOINT`` is never set (no identity games).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import Dict, List, Optional

_FENCE = "__AUTOMATOS_SHELL_FENCE__"
_SAFE_COMMAND_RE = re.compile(r"^[A-Za-z0-9._+-]+$")
_CLAUDE_MARKER_RE = re.compile(r"^CLAUDE(CODE|_)")

# Configuration, not identity: the operator's own choices about how the CLI runs.
CLAUDE_CONFIG_KEEP = frozenset({
    "CLAUDE_CONFIG_DIR",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
})

# Never forwarded into a session (billing / redirection / identity).
STRIPPED_EXACT = frozenset({
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_CODE_OAUTH_TOKEN",  # the CLI reads its own login; we never carry a token
})

# Common install locations appended after the shell PATH (munder's list).
_EXTRA_BIN_DIRS = (
    "/opt/homebrew/bin",
    "/usr/local/bin",
    os.path.expanduser("~/.local/bin"),
    os.path.expanduser("~/.claude/local"),
    os.path.expanduser("~/.npm-global/bin"),
)

_cached_shell_path: Optional[str] = None


def _capture_from_login_shell(script: str, timeout: float = 3.0) -> Optional[str]:
    shell = os.environ.get("SHELL") or "/bin/zsh"
    try:
        proc = subprocess.run(
            [shell, "-ilc", f"printf %s {_FENCE}; {script}; printf %s {_FENCE}"],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    out = proc.stdout or ""
    start, end = out.find(_FENCE), out.rfind(_FENCE)
    if start < 0 or end <= start:
        return None
    return out[start + len(_FENCE):end]


def user_shell_path() -> str:
    """The interactive-shell PATH, captured once; the process PATH as fallback."""
    global _cached_shell_path
    if _cached_shell_path is not None:
        return _cached_shell_path
    captured = _capture_from_login_shell('printf %s "$PATH"')
    path = captured.strip() if captured and "\n" not in captured.strip() else ""
    if not path:
        path = os.environ.get("PATH", "")
    parts = [p for p in path.split(os.pathsep) if p]
    for extra in _EXTRA_BIN_DIRS:
        if extra not in parts and os.path.isdir(extra):
            parts.append(extra)
    _cached_shell_path = os.pathsep.join(parts)
    return _cached_shell_path


def resolve_binary(command: str, path: Optional[str] = None) -> Optional[str]:
    """Absolute path of ``command`` on the user's PATH, or ``None``.

    A command that already looks like a path is returned as is when it exists;
    a bare name must be a plain executable name (never interpolated into a shell).
    """
    if os.sep in command:
        return command if os.access(command, os.X_OK) else None
    if not _SAFE_COMMAND_RE.match(command):
        return None
    return shutil.which(command, path=path or user_shell_path())


def build_session_env(
    parent: Optional[Dict[str, str]] = None,
    *,
    extra: Optional[Dict[str, str]] = None,
    path: Optional[str] = None,
) -> Dict[str, str]:
    """Layer the session environment: inherited minus markers/credentials, then
    the host's own values (``extra``), which always win."""
    src = dict(os.environ if parent is None else parent)
    env: Dict[str, str] = {}
    for key, value in src.items():
        if key in STRIPPED_EXACT:
            continue
        if _CLAUDE_MARKER_RE.match(key) and key not in CLAUDE_CONFIG_KEEP:
            continue
        env[key] = value
    env["PATH"] = path or user_shell_path()
    env.setdefault("TERM", "xterm-256color")
    env.setdefault("COLORTERM", "truecolor")
    env.setdefault("LANG", "en_US.UTF-8")
    if extra:
        env.update(extra)
    return env


def forbidden_keys_present(env: Dict[str, str]) -> List[str]:
    """Source-guard helper: which forbidden keys a built environment still carries."""
    return sorted(k for k in env if k in STRIPPED_EXACT or (_CLAUDE_MARKER_RE.match(k) and k not in CLAUDE_CONFIG_KEEP))
