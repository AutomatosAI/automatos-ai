"""The OS sandbox a Claude Code session's shell commands run in.

The ``PreToolUse`` gate (``policy.py``) judges what a command line NAMES. It
cannot see what an allowed command RUNS: ``python x.py`` on a file the session
just wrote, ``npm run <script>``, ``pytest`` collecting a ``conftest.py``, a git
hook under ``git commit`` — each executes code the session wrote, as the
operator's own user, with the operator's home directory and network. No static
gate closes that (``secret_reach.py`` says as much); the operating system can.

So a Claude session's per-session ``settings.json`` also switches on Claude
Code's own Bash sandbox (bubblewrap on Linux and WSL2, Seatbelt on macOS). It
confines every shell command AND every process that command starts:

* writes: the session's folders only (Claude Code's default: the working
  directory, ``--add-dir`` folders, the temp directory);
* reads: everywhere except the credential stores below, the host's own state
  and the platform's secrets;
* network: the package registries plus what the operator adds
  (``--session-allow-domain``); any other host is refused, never asked;
* no way out: ``allowUnsandboxedCommands`` is false, and ``failIfUnavailable``
  is true — a machine that cannot sandbox never runs the session unsandboxed.

The gate still decides every call first (``autoAllowBashIfSandboxed`` false):
the sandbox sits under the gate, not in place of it. Codex brings its own
sandbox (``-s workspace-write`` in its preset). ``--no-session-sandbox`` turns
this off, for a host that is already isolated (a VM, a container, a dedicated
OS user with no credentials).
"""
from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .env import user_shell_path

# What ``npm install`` / ``pip install`` need, and nothing that accepts an upload.
DEFAULT_ALLOWED_DOMAINS = (
    "registry.npmjs.org", "registry.yarnpkg.com", "pypi.org", "files.pythonhosted.org",
)

# Where the usual CLIs keep a login. Claude Code's sandbox can READ the whole
# machine by default, these included (its docs say so); a session never needs them.
CREDENTIAL_PATHS = (
    "~/.aws", "~/.ssh", "~/.gnupg", "~/.config/gh", "~/.config/gcloud", "~/.azure",
    "~/.kube", "~/.docker/config.json", "~/.git-credentials", "~/.netrc", "~/.pypirc",
    "~/.terraform.d/credentials.tfrc.json",
)

# bubblewrap isolates, socat relays the network through Claude Code's proxy.
LINUX_TOOLS = ("bwrap", "socat")
INSTALL_HINT = ("Session sandbox unavailable: install bubblewrap and socat "
                "(Debian/Ubuntu: sudo apt-get install bubblewrap socat) and restart the host — "
                "or run the host with --no-session-sandbox on a machine that is already isolated.")


WINDOWS_HINT = ("Session sandbox unavailable on Windows: Claude Code sandboxes its commands on macOS and "
                "Linux only. Run the host with --no-session-sandbox on a Windows machine you accept as "
                "isolated, or run it inside WSL2 with bubblewrap and socat.")


@dataclass(frozen=True)
class SessionSandbox:
    """The host's choice, from its flags: sandbox sessions or not, and which
    hosts their commands may reach."""
    enabled: bool = True
    allowed_domains: Tuple[str, ...] = DEFAULT_ALLOWED_DOMAINS


def missing_tools(system: Optional[str] = None, path: Optional[str] = None) -> List[str]:
    """What this machine lacks to sandbox a session. macOS needs nothing
    (Seatbelt is built in); Linux and WSL2 need bubblewrap and socat."""
    if (system or platform.system()) != "Linux":
        return []
    search = path or user_shell_path()  # the PATH the session's CLI will search
    return [tool for tool in LINUX_TOOLS if shutil.which(tool, path=search) is None]


def unavailable_reason(sandbox: Optional[SessionSandbox], system: Optional[str] = None,
                       path: Optional[str] = None) -> Optional[str]:
    """None = a session can run as configured; else the sentence that lands on
    the ticket and in the fleet view."""
    if sandbox is None or not sandbox.enabled:
        return None
    if (system or platform.system()) == "Windows":   # #818: no sandbox exists for it there; fail closed
        return WINDOWS_HINT
    missing = missing_tools(system, path)
    return f"{INSTALL_HINT} Missing: {', '.join(missing)}." if missing else None


def _claude_credentials_file() -> str:
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    return str(Path(config_dir).expanduser() / ".credentials.json") if config_dir else "~/.claude/.credentials.json"


def _secret_files(roots: Iterable[Path]) -> List[str]:
    """The platform checkout's ``.env`` family and credential key (F042), by the
    same names ``policy.py`` refuses."""
    out: List[str] = []
    for root in roots:
        base = str(Path(root).expanduser())
        out += [f"{base}/.env", f"{base}/.env.*", f"{base}/.credential_key"]
    return out


def deny_read(secret_roots: Sequence[Path] = (), off_limits: Sequence[Path] = ()) -> List[str]:
    """Every path a sandboxed command may not read."""
    return [*CREDENTIAL_PATHS, _claude_credentials_file(), *_secret_files(secret_roots),
            *(str(Path(p).expanduser()) for p in off_limits)]


def claude_settings(sandbox: SessionSandbox, *, secret_roots: Sequence[Path] = (),
                    off_limits: Sequence[Path] = ()) -> Dict[str, Any]:
    """The ``sandbox`` block of a Claude session's ``--settings`` file. A
    ``--settings`` file loads whatever ``--setting-sources`` says; its booleans
    win over the operator's user settings, its lists add to theirs."""
    return {
        "enabled": True,
        "failIfUnavailable": True,
        "allowUnsandboxedCommands": False,
        "autoAllowBashIfSandboxed": False,
        "filesystem": {"denyRead": deny_read(secret_roots, off_limits)},
        "network": {"allowedDomains": list(sandbox.allowed_domains), "strictAllowlist": True},
    }


__all__ = [
    "CREDENTIAL_PATHS", "DEFAULT_ALLOWED_DOMAINS", "INSTALL_HINT", "LINUX_TOOLS", "SessionSandbox",
    "claude_settings", "deny_read", "missing_tools", "unavailable_reason",
]
