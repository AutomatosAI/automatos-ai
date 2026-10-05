"""F333 (night 10) — a session file under the projects folder is found from any local workspace.

In the local edition the owner's projects folder (``LOCAL_PROJECTS_DIR``) is bind-mounted
into the workspace worker ONCE, under the default workspace:
``/workspaces/<DEFAULT_WORKSPACE_ID>/projects`` (docker-compose.yml). It is the owner's
machine folder, not a folder of any one workspace. A host paired to a second workspace
writes ``~/Development/deliverables-sim/sessions/2050/x.html``; the backend maps that to
``projects/deliverables-sim/sessions/2050/x.html`` and used to look for it under
``/workspaces/<second workspace>/projects/…``, which does not exist. The Deliverables link
answered 404, the completion check sent the card to review ("the workspace has no such
file") and the next mission step's ``read_step_file`` said "File not found", with every
file on disk.

The worker workspace a READ of a ``projects/…`` path goes to is therefore the default
workspace, in the local edition only — where there is one operator and every workspace on
the stack is theirs. The hosted edition is unchanged. A path that is not a plain relative
path under ``projects/`` (``..``, ``.``, an empty part, an absolute path) is never sent to
another workspace; the worker's own confinement (``resolve_safe_path``) still applies.
"""
from __future__ import annotations

from typing import Any, Optional

from config import config

PROJECTS_FOLDER = "projects"


def clean_relative(rel: str) -> Optional[str]:
    """``rel`` without its outer slashes, or ``None`` when any part of it is empty,
    ``.`` or ``..`` — the path-safety rule session paths are mapped with."""
    rel = rel.strip("/")
    if not rel or any(part in ("", ".", "..") for part in rel.split("/")):
        return None
    return rel


def is_projects_path(path: Any) -> bool:
    """A plain relative path at or under ``projects/``: never absolute, never stepping out."""
    raw = str(path or "")
    if raw.startswith("/") or clean_relative(raw) != raw.rstrip("/"):
        return False
    return raw.split("/", 1)[0] == PROJECTS_FOLDER


def projects_mount_workspace() -> Optional[str]:
    """The workspace the projects folder is mounted under, when this stack mounts the
    owner's folder there: the local edition, with ``LOCAL_PROJECTS_DIR`` and
    ``DEFAULT_WORKSPACE_ID`` set. ``None`` otherwise (the hosted edition always)."""
    if not config.IS_LOCAL_EDITION:
        return None
    if not (getattr(config, "LOCAL_PROJECTS_DIR", "") or "").strip():
        return None
    default = (getattr(config, "DEFAULT_WORKSPACE_ID", "") or "").strip()
    return default or None


def worker_workspace_for(workspace_id: Any, path: Any) -> str:
    """The worker workspace a read of ``path`` for ``workspace_id`` is served from: the
    projects mount's workspace for a ``projects/…`` path on a local stack, else its own."""
    own = str(workspace_id)
    mount = projects_mount_workspace()
    if mount and own != mount and is_projects_path(path):
        return mount
    return own
