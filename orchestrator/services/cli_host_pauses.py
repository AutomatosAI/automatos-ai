"""A CLI a host paused at its plan's usage limit is waiting, not missing (F083).

When a CLI's usage window closes, the host stops serving it and says so in its
heartbeat: ``capabilities.paused`` maps the CLI to the host's own line ("paused:
claude usage limit, resumes ~09:00"). Without reading it, every agent on that CLI
said "install and log in to claude on a host": the wrong fix for a limit that
lifts by itself (5 Nov 2026).
"""
from __future__ import annotations

from typing import Any, List

from sqlalchemy.orm import Session

from core.models.cli_hosts import CliHost, CliHostStatus

PAUSED_ON_HOST = "{why} (host {host})"


def paused_lines(db: Session, workspace_id: Any, cli: str) -> List[str]:
    """One line per ONLINE host of the workspace that paused ``cli``: the host's own
    words and its name. Empty when no online host paused it."""
    hosts = (
        db.query(CliHost)
        .filter(CliHost.workspace_id == workspace_id, CliHost.status == CliHostStatus.PAIRED.value)
        .all()
    )
    lines: List[str] = []
    for host in hosts:
        why = _pause_of(host, cli) if host.is_online() else None
        if why:
            lines.append(PAUSED_ON_HOST.format(why=why, host=host.name))
    return lines


def _pause_of(host: Any, cli: str) -> str:
    caps = host.capabilities if isinstance(host.capabilities, dict) else {}
    paused = caps.get("paused")
    why = paused.get(cli) if isinstance(paused, dict) else None
    return why.strip() if isinstance(why, str) else ""
