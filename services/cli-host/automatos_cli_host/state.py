"""Host state on disk — token, allowlist, process table. Small JSON files, ``0600``.

The state directory is the only place a secret lives (the host token the backend
minted at pairing). Nothing here is ever read from ``~/.claude`` — that is
Claude Code's, not ours.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional


def _write_private_json(path: Path, data: Any) -> None:
    """Atomic, owner-only write (tmp file in the same dir → fsync → rename)."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


# ── pairing / token ──────────────────────────────────────────────────────────

def load_host_identity(path: Path) -> Optional[Dict[str, Any]]:
    """``{host_id, token, workspace_id, url}`` or ``None`` when not paired yet."""
    data = _read_json(path, None)
    if not isinstance(data, dict) or not data.get("token") or not data.get("host_id"):
        return None
    return data


def save_host_identity(path: Path, *, host_id: str, token: str, workspace_id: str, url: str) -> None:
    _write_private_json(path, {"host_id": host_id, "token": token, "workspace_id": workspace_id, "url": url})


def forget_host_identity(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


# ── allowlist ────────────────────────────────────────────────────────────────

def load_allowlist(path: Path) -> List[str]:
    data = _read_json(path, [])
    return [d for d in data if isinstance(d, str)] if isinstance(data, list) else []


def save_allowlist(path: Path, dirs: List[str]) -> None:
    _write_private_json(path, sorted(set(dirs)))


# ── process table ────────────────────────────────────────────────────────────

def load_process_table(path: Path) -> Dict[str, Dict[str, Any]]:
    """task_id (str) → ``{pid, pgid, session_id, attempt, cwd, started_at}``."""
    data = _read_json(path, {})
    return data if isinstance(data, dict) else {}


def save_process_table(path: Path, table: Dict[str, Dict[str, Any]]) -> None:
    _write_private_json(path, table)
