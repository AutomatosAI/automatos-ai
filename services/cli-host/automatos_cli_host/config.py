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
DEFAULT_CLAIM_BATCH = 5


@dataclass
class HostConfig:
    url: str = DEFAULT_URL
    state_dir: Path = DEFAULT_STATE_DIR
    allow_dirs: List[Path] = field(default_factory=list)
    pair_code: Optional[str] = None
    name: str = field(default_factory=lambda: socket.gethostname() or "cli-host")
    once: bool = False
    max_sessions: int = 0  # 0 = unlimited
    poll_seconds: float = DEFAULT_POLL_SECONDS
    heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS
    event_flush_seconds: float = DEFAULT_EVENT_FLUSH_SECONDS
    session_timeout_seconds: float = DEFAULT_SESSION_TIMEOUT_SECONDS
    claim_batch: int = DEFAULT_CLAIM_BATCH
    claude_binary: Optional[str] = None  # explicit path; default = the user's PATH
    use_worktrees: bool = True
    verbose: bool = False

    @property
    def token_path(self) -> Path:
        return self.state_dir / "host.json"

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
    p.add_argument("--verbose", action="store_true")
    return p


def parse_args(argv: Optional[List[str]] = None) -> HostConfig:
    ns = build_parser().parse_args(argv)
    cfg = HostConfig(
        url=ns.url.rstrip("/"),
        state_dir=Path(ns.dir).expanduser(),
        allow_dirs=[Path(d).expanduser() for d in ns.allow],
        pair_code=ns.pair,
        once=ns.once,
        max_sessions=max(0, ns.max_sessions),
        poll_seconds=max(1.0, ns.poll_seconds),
        session_timeout_seconds=max(60.0, ns.session_timeout),
        claude_binary=ns.claude,
        use_worktrees=not ns.no_worktrees,
        verbose=ns.verbose,
    )
    if ns.name:
        cfg.name = ns.name
    return cfg
