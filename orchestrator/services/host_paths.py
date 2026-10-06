"""A session host's paths, as the backend compares them (#818).

The CLI host reports where a session ran and the files it wrote as absolute paths
on its own machine, and the backend maps them onto the workspace by the folders the
stack was started with (``AUTOMATOS_WORKSPACE_DIR``, ``LOCAL_PROJECTS_DIR``). On
macOS and Linux those are all ``/…``. A host on native Windows reports
``C:\\Users\\…``, while the stack (Docker Desktop) is configured as ``C:/Users/…``,
and NTFS ignores case. So:

- separators are compared as forward slashes;
- a Windows drive path counts as absolute;
- under a Windows drive root, the folders match without regard to case. Unix
  roots still match exactly.
"""
from __future__ import annotations

import re
from typing import Optional

_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:(/|$)")


def as_host_path(path: str) -> str:
    """``path`` with forward slashes, the form every root is compared in."""
    return str(path).replace("\\", "/")


def is_windows_drive_path(path: str) -> bool:
    return bool(_WINDOWS_DRIVE.match(as_host_path(path)))


def is_absolute_host_path(path: str) -> bool:
    return as_host_path(path).startswith("/") or is_windows_drive_path(path)


def relative_to_root(path: str, root: str) -> Optional[str]:
    """What follows ``root`` in ``path`` (``""`` when it is the root itself), or
    ``None`` when ``path`` is not under ``root``. Both in ``as_host_path`` form."""
    head, rest = path[:len(root)], path[len(root):]
    same = head.lower() == root.lower() if is_windows_drive_path(root) else head == root
    if not same or (rest and not rest.startswith("/")):
        return None
    return rest


__all__ = ["as_host_path", "is_absolute_host_path", "is_windows_drive_path", "relative_to_root"]
