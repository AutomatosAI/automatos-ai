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

CLI adapter design §9.1: WHICH keys and markers is the preset's business
(``strip_env`` / ``strip_env_prefixes`` / ``keep_env``) — the same guarantee is
spelled differently on every binary, and a global constant would silently mean
"unchecked" for CLI #2. The operator's own shell (the Canvas terminal) gets the
union over every preset.

One deliberate exception (dev-box mode, ``dev_box.py``): a sandboxed Claude Code
session on a ``--dev-box`` host carries the GitHub token the owner put in the host's
state directory, as ``GH_TOKEN`` and nowhere else, and the host user's own git identity
(``github_env``).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from .presets import CliPreset, union_strip_env

_FENCE = "__AUTOMATOS_SHELL_FENCE__"
_SAFE_COMMAND_RE = re.compile(r"^[A-Za-z0-9._+-]+$")

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


def _forbidden(key: str, strip: FrozenSet[str], prefixes: Sequence[str], keep: FrozenSet[str]) -> bool:
    if key in strip:
        return True
    return any(key.startswith(p) for p in prefixes) and key not in keep


def build_env(
    parent: Optional[Dict[str, str]] = None,
    *,
    strip: FrozenSet[str],
    prefixes: Sequence[str],
    keep: FrozenSet[str],
    extra: Optional[Dict[str, str]] = None,
    path: Optional[str] = None,
) -> Dict[str, str]:
    """Layer an environment: inherited minus the given credentials/markers, then
    the host's own values (``extra``), which always win."""
    src = dict(os.environ if parent is None else parent)
    env: Dict[str, str] = {k: v for k, v in src.items() if not _forbidden(k, strip, prefixes, keep)}
    env["PATH"] = path or user_shell_path()
    env.setdefault("TERM", "xterm-256color")
    env.setdefault("COLORTERM", "truecolor")
    env.setdefault("LANG", "en_US.UTF-8")
    if extra:
        env.update(extra)
    return env


# ── dev-box mode: the one credential a session carries ───────────────────────
# The token reaches gh as GH_TOKEN, and git through its own config, delivered in
# GIT_CONFIG_COUNT / GIT_CONFIG_KEY_n / GIT_CONFIG_VALUE_n (never a file in the session
# or the repository): an https push to github.com asks a helper that answers with
# $GH_TOKEN, read when git runs — the value itself is in no config entry.
GITHUB_TOKEN_VAR = "GH_TOKEN"
# The host user's own git identity, read once from their global git config: the
# sandbox keeps ~/.gitconfig out of reach, and git refuses to commit without one.
IDENTITY_VARS = (("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"), ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"))
# What the host inherited that could carry another token, git configuration or identity.
GITHUB_INHERITED = frozenset({"GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN", "GITHUB_ENTERPRISE_TOKEN",
                              "GH_CONFIG_DIR", "GIT_CONFIG_COUNT", "GIT_CONFIG_PARAMETERS",
                              *(var for pair in IDENTITY_VARS for var in pair)})
GITHUB_INHERITED_PREFIXES = ("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")
GITHUB_CREDENTIAL_HELPER = ('!f() { test "$1" = get || exit 0; cat >/dev/null; '
                            'echo username=x-access-token; echo "password=$GH_TOKEN"; }; f')
GITHUB_GIT_CONFIG = (
    ("credential.https://github.com.helper", ""),       # forget every helper configured before ours
    ("credential.https://github.com.helper", GITHUB_CREDENTIAL_HELPER),
    ("push.default", "simple"),                          # a push never maps a branch onto its upstream's name
    ("url.https://github.com/.insteadOf", "git@github.com:"),     # an SSH remote pushes over https, with the token
    ("url.https://github.com/.insteadOf", "ssh://git@github.com/"),
)


def github_env(token: str, config_dir: str, identity: Optional[Tuple[str, str]] = None) -> Dict[str, str]:
    """The variables that carry the dev-box token into one session: GH_TOKEN, git's
    credential config, gh's own config folder (``config_dir``: the sandbox keeps
    ~/.config/gh unreadable, and gh stops on a config folder it cannot read), and the
    host user's git identity (name, email) when they have one."""
    entries: Dict[str, str] = {GITHUB_TOKEN_VAR: token, "GH_CONFIG_DIR": config_dir,
                               "GIT_CONFIG_COUNT": str(len(GITHUB_GIT_CONFIG))}
    for index, (key, value) in enumerate(GITHUB_GIT_CONFIG):
        entries = {**entries, f"GIT_CONFIG_KEY_{index}": key, f"GIT_CONFIG_VALUE_{index}": value}
    for names, value in zip(IDENTITY_VARS, identity or ()):
        entries = {**entries, **{name: value for name in names}}
    return entries


def build_session_env(
    preset: CliPreset,
    parent: Optional[Dict[str, str]] = None,
    *,
    extra: Optional[Dict[str, str]] = None,
    path: Optional[str] = None,
    github: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """The environment for one CLI's session: the preset says what must not leak
    and what operator configuration is kept; the preset's own ``extra_env`` sits
    under the host's values. ``github`` (dev-box mode only, ``github_env``): the one
    credential a session carries, its git config and identity; what the host
    inherited that could carry another is dropped."""
    merged = {**dict(preset.extra_env), **(extra or {}), **(github or {})}
    strip = preset.strip_env | (GITHUB_INHERITED if github else frozenset())
    prefixes = (*preset.strip_env_prefixes, *(GITHUB_INHERITED_PREFIXES if github else ()))
    return build_env(parent, strip=strip, prefixes=prefixes, keep=preset.keep_env, extra=merged, path=path)


def build_shell_env(
    parent: Optional[Dict[str, str]] = None,
    *,
    extra: Optional[Dict[str, str]] = None,
    path: Optional[str] = None,
) -> Dict[str, str]:
    """The operator's own shell in the Canvas: no CLI's credential or session
    marker is inherited, every CLI's configuration is."""
    strip, prefixes, keep = union_strip_env()
    return build_env(parent, strip=strip, prefixes=prefixes, keep=keep, extra=extra, path=path)


def hook_pythonpath(parent: Optional[Dict[str, str]] = None) -> str:
    """``PYTHONPATH`` for a CLI's hooks. They run from the session's folder, and the
    shim (``python -m automatos_cli_host.hook_shim``) must find this package there;
    whatever the host inherited stays behind it."""
    inherited = (os.environ if parent is None else parent).get("PYTHONPATH", "")
    package_root = str(Path(__file__).resolve().parents[1])
    return package_root + (os.pathsep + inherited if inherited else "")

def forbidden_keys_present(env: Dict[str, str], preset: Optional[CliPreset] = None) -> List[str]:
    """Source-guard helper: which forbidden keys a built environment still carries
    — for one CLI, or for every CLI when no preset is given."""
    if preset is not None:
        strip, prefixes, keep = preset.strip_env, preset.strip_env_prefixes, preset.keep_env
    else:
        strip, prefixes, keep = union_strip_env()
    return sorted(k for k in env if _forbidden(k, strip, prefixes, keep))
