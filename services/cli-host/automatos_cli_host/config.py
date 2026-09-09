"""Host configuration — flags and environment, resolved once.

Everything the host needs to know about ITS machine and the backend it serves:

* ``--url`` / ``AUTOMATOS_URL``      — the local edition's API (default loopback:8000)
* ``--dir`` / ``AUTOMATOS_CLI_HOST_DIR`` — the host's state directory
  (token, allowlist, process table; files are ``0600``)
* ``--allow DIR`` (repeatable)      — directories sessions may work in; the
  Makefile passes ``./workspaces`` so the compose default works out of the box
* ``--pair CODE``                   — pair with the one-time code from Settings
* ``--name``                        — how this host appears in the fleet
* ``--once``                        — one claim/run cycle then exit (tests, cron)
* ``--max-sessions``                — 0 = no cap (owner decision Q5)

No secrets are ever taken from flags or the environment: the host token is
minted by the backend at pairing and lives only in the state directory.
"""
from __future__ import annotations

import argparse
import os
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

DEFAULT_URL = "http://127.0.0.1:8000"
DEFAULT_STATE_DIR = Path.home() / ".automatos" / "cli-host"
DEFAULT_POLL_SECONDS = 5.0
DEFAULT_HEARTBEAT_SECONDS = 30.0
DEFAULT_EVENT_FLUSH_SECONDS = 5.0
DEFAULT_SESSION_TIMEOUT_SECONDS = 4 * 3600
DEFAULT_STARTUP_TIMEOUT_SECONDS = 180.0
DEFAULT_CLAIM_BATCH = 5


@dataclass
class HostConfig:
    url: str = DEFAULT_URL
    state_dir: Path = DEFAULT_STATE_DIR
    allow_dirs: List[Path] = field(default_factory=list)
    # Where a ticket with no folder runs (<root>/sessions/<ticket>): the deliverables
    # root compose mounts as the workspace root. None = the first registered root.
    default_root: Optional[Path] = None
    pair_code: Optional[str] = None
    name: str = field(default_factory=lambda: socket.gethostname() or "cli-host")
    once: bool = False
    max_sessions: int = 0  # 0 = unlimited
    poll_seconds: float = DEFAULT_POLL_SECONDS
    heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS
    event_flush_seconds: float = DEFAULT_EVENT_FLUSH_SECONDS
    session_timeout_seconds: float = DEFAULT_SESSION_TIMEOUT_SECONDS
    ask_timeout: float = 120.0
    startup_timeout_seconds: float = DEFAULT_STARTUP_TIMEOUT_SECONDS
    claim_batch: int = DEFAULT_CLAIM_BATCH
    claude_binary: Optional[str] = None  # explicit path; default = the user's PATH
    use_worktrees: bool = True
    verbose: bool = False
    # PRD-239 S7: the Canvas terminal — the operator's own shell served on the
    # loopback. 0 = an ephemeral port (announced in the host's capabilities).
    terminal_enabled: bool = True
    terminal_port: int = 0

    # service actions (PRD-235 W3): install | uninstall | status | restart | nudge | None (= run)
    service_action: Optional[str] = None

    @property
    def token_path(self) -> Path:
        return self.state_dir / "host.json"

    @property
    def pid_path(self) -> Path:
        return self.state_dir / "host.pid"

    @property
    def allowlist_path(self) -> Path:
        return self.state_dir / "allowlist.json"

    @property
    def process_table_path(self) -> Path:
        return self.state_dir / "sessions.json"

    @property
    def socket_path(self) -> Path:
        return self.state_dir / "hooks.sock"

    @property
    def sessions_dir(self) -> Path:
        return self.state_dir / "sessions"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="automatos-cli-host",
        description="Run Automatos tickets as your own Claude Code sessions on this machine.",
    )
    p.add_argument("--url", default=os.environ.get("AUTOMATOS_URL", DEFAULT_URL),
                   help=f"local edition API (default {DEFAULT_URL})")
    p.add_argument("--dir", default=os.environ.get("AUTOMATOS_CLI_HOST_DIR", str(DEFAULT_STATE_DIR)),
                   help="host state directory (token, allowlist, process table)")
    p.add_argument("--allow", action="append", default=[], metavar="DIR",
                   help="a directory sessions may work in (repeatable); registered directories only")
    p.add_argument("--default-root", default=None, metavar="DIR",
                   help="where a ticket with no folder runs (<DIR>/sessions/<ticket>); registered too. "
                        "make cli-host passes the deliverables root. Default: the first registered directory")
    p.add_argument("--pair", default=None, metavar="CODE",
                   help="pair this host with the one-time code from Settings → Session mode")
    p.add_argument("--name", default=None, help="how this host appears in the fleet")
    p.add_argument("--once", action="store_true", help="one claim/run cycle, then exit")
    p.add_argument("--max-sessions", type=int, default=0,
                   help="concurrent sessions (0 = no cap, the default)")
    p.add_argument("--claude", default=os.environ.get("AUTOMATOS_CLAUDE_BINARY"),
                   help="path to the claude binary (default: the one on your login-shell PATH)")
    p.add_argument("--no-worktrees", action="store_true",
                   help="run sessions in the registered directory itself instead of a git worktree")
    p.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    p.add_argument("--session-timeout", type=float, default=DEFAULT_SESSION_TIMEOUT_SECONDS,
                   help="wall-clock cap per session turn, seconds")
    p.add_argument("--ask-timeout", type=float, default=120.0,
                   help="seconds a session waits for the operator to answer a permission card before denying (default 120)")
    p.add_argument("--startup-timeout", type=float, default=DEFAULT_STARTUP_TIMEOUT_SECONDS,
                   help="seconds to wait for a session to report SessionStart (login screens and dialogs never do)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--no-terminal", action="store_true",
                   help="do not serve the Canvas terminal (your own shell on 127.0.0.1 for the browser on this machine)")
    p.add_argument("--terminal-port", type=int, default=0,
                   help="fixed loopback port for the Canvas terminal (default: an ephemeral port, announced to the backend)")
    svc = p.add_mutually_exclusive_group()
    svc.add_argument("--install", dest="service_action", action="store_const", const="install",
                     help="run this host as a login service (launchd on macOS, systemd --user on Linux) with these arguments")
    svc.add_argument("--uninstall", dest="service_action", action="store_const", const="uninstall",
                     help="remove the login service")
    svc.add_argument("--service-status", dest="service_action", action="store_const", const="status",
                     help="is the login service installed and running?")
    svc.add_argument("--restart-service", dest="service_action", action="store_const", const="restart",
                     help="restart the login service now")
    svc.add_argument("--nudge", dest="service_action", action="store_const", const="nudge",
                     help="ask the running host to drain and restart (SIGHUP) — `make up` does this after a rebuild")
    return p


def parse_args(argv: Optional[List[str]] = None) -> HostConfig:
    ns = build_parser().parse_args(argv)
    cfg = HostConfig(
        url=ns.url.rstrip("/"),
        state_dir=Path(ns.dir).expanduser(),
        allow_dirs=[Path(d).expanduser() for d in ns.allow],
        default_root=Path(ns.default_root).expanduser() if ns.default_root else None,
        pair_code=ns.pair,
        once=ns.once,
        max_sessions=max(0, ns.max_sessions),
        poll_seconds=max(1.0, ns.poll_seconds),
        session_timeout_seconds=max(60.0, ns.session_timeout),
        ask_timeout=max(5.0, ns.ask_timeout),
        startup_timeout_seconds=max(10.0, ns.startup_timeout),
        claude_binary=ns.claude,
        use_worktrees=not ns.no_worktrees,
        verbose=ns.verbose,
        service_action=ns.service_action,
        terminal_enabled=not ns.no_terminal,
        terminal_port=max(0, min(65535, int(ns.terminal_port or 0))),
    )
    if ns.name:
        cfg.name = ns.name
    return cfg
