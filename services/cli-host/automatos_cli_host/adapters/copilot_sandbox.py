"""GitHub Copilot's command sandbox under a session (PRD-253 S2.2, O3: on by default).

The host's gate judges what a command NAMES; the OS sandbox bounds what it RUNS —
the same split as Claude's (``sandbox.py``). Copilot brings its own (Microsoft
Execution Containers: Seatbelt on macOS, bubblewrap on Linux), still an
experimental feature in 1.0.91, switched on by a saved ``sandbox.enabled`` in the
agent home's ``settings.json`` (``copilot_home.seeded_settings``). The block:

* writes: the working folder and the session's folders only;
* reads: everywhere Copilot grants by default except the credential stores, the
  platform's secrets and this host's own state (``deniedPaths``);
* no escape hatch (``allowBypass`` false), no git/gh credentials injected, no keychain;
* network: outbound only to the allowed hosts (the package registries plus
  ``--session-allow-domain``) — a non-empty ``allowedHosts`` blocks every other
  host — and never the local network.

A machine that cannot sandbox never runs a session unsandboxed: the host refuses
the CLI (``copilot_sandbox_unavailable``). ``--no-session-sandbox`` turns it off.
"""
from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..env import user_shell_path
from ..sandbox import SessionSandbox, deny_read

MACOS_TOOLS = ("sandbox-exec",)
LINUX_TOOLS = ("bwrap", "slirp4netns", "iptables")
LINUX_TUN = Path("/dev/net/tun")
PREREQUISITES = {
    "Darwin": "macOS: sandbox-exec",
    "Linux": "Linux: bubblewrap 0.5 or later, slirp4netns, util-linux 2.35 or later, iptables and /dev/net/tun",
}


def missing(system: Optional[str] = None, path: Optional[str] = None, tun: Path = LINUX_TUN) -> List[str]:
    """What this machine lacks for Copilot's sandbox."""
    system = system or platform.system()
    search = path or user_shell_path()
    tools = MACOS_TOOLS if system == "Darwin" else LINUX_TOOLS if system == "Linux" else ()
    lacking = [tool for tool in tools if shutil.which(tool, path=search) is None]
    if system == "Linux" and not tun.exists():
        lacking = [*lacking, str(tun)]
    return lacking


def unavailable_reason(sandbox: Optional[SessionSandbox], system: Optional[str] = None,
                       path: Optional[str] = None) -> Optional[str]:
    """None = Copilot sessions can be sandboxed as configured; else the sentence."""
    if sandbox is None or not sandbox.enabled:
        return None
    system = system or platform.system()
    lacking = missing(system, path)
    if not lacking and system in PREREQUISITES:
        return None
    needs = PREREQUISITES.get(system, f"{system} is not a platform Copilot can sandbox on")
    return (f"GitHub Copilot cannot sandbox sessions on this machine (needs {needs}; missing: "
            f"{', '.join(lacking) or system}). Install them and restart the host — or run the host with "
            "--no-session-sandbox on a machine that is already isolated.")


def sandbox_settings(sandbox: SessionSandbox, *, writable: Sequence[Path], secret_roots: Sequence[Path] = (),
                     off_limits: Sequence[Path] = ()) -> Dict[str, Any]:
    """The ``sandbox`` block of the agent home's ``settings.json`` (1.0.91 schema).
    Every path absolute: the policy names absolute paths only."""
    return {
        "enabled": True,
        "addCurrentWorkingDirectory": True,
        "allowBypass": False,
        "auth": {"git": False, "gh": False},
        "sandboxMcpServers": True,
        "userPolicy": {
            "filesystem": {
                "readwritePaths": [str(Path(p).expanduser()) for p in writable],
                "deniedPaths": [os.path.expanduser(p) for p in deny_read(secret_roots, off_limits)],
            },
            "network": {"allowOutbound": True, "allowLocalNetwork": False,
                        "allowedHosts": list(sandbox.allowed_domains)},
            "seatbelt": {"keychainAccess": False},
        },
    }


__all__ = ["LINUX_TOOLS", "MACOS_TOOLS", "PREREQUISITES", "missing", "sandbox_settings", "unavailable_reason"]
