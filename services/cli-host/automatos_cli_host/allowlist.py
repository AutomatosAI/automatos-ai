"""Host-side directory allowlist — the second half of the two-sided rule.

The backend stores the registered directories; the host keeps its OWN list and
refuses any working directory outside it. A compromised or misconfigured
backend can therefore never point a session at ``~`` (PRD-234 §Design 6c).

Path rules follow the worker's ``canvas_confinement``: resolve symlinks, refuse
null bytes and ``..`` escapes, and require the resolved path to sit inside a
resolved allowed root.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional


class NotAllowed(PermissionError):
    """The directory is outside every allowed root."""


def normalise_roots(roots: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    for raw in roots:
        try:
            p = Path(raw).expanduser().resolve()
        except (OSError, RuntimeError):
            continue
        if p not in out:
            out.append(p)
    return out


def is_inside(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_allowed(cwd: Optional[str], roots: Iterable[str], *, default_root: Optional[str] = None) -> Path:
    """The resolved working directory a session may use, or raise ``NotAllowed``.

    ``cwd`` ``None``/empty means "the default root" (the compose ``./workspaces``
    directory the Makefile registers). A relative ``cwd`` resolves under the
    default root; an absolute one must sit inside an allowed root.
    """
    allowed = normalise_roots(roots)
    if not allowed:
        raise NotAllowed("no directories are registered on this host — start with --allow DIR")
    if cwd is not None and "\x00" in cwd:
        raise NotAllowed("null byte in working directory")
    if not cwd:
        if default_root is None:
            raise NotAllowed("the ticket names no working directory and the host has no default root")
        target = Path(default_root).expanduser()
    else:
        target = Path(cwd).expanduser()
        if not target.is_absolute():
            base = Path(default_root).expanduser() if default_root else allowed[0]
            target = base / target
    try:
        resolved = target.resolve()
    except (OSError, RuntimeError) as exc:
        raise NotAllowed(f"cannot resolve working directory {cwd!r}: {exc}") from None
    if any(is_inside(resolved, root) for root in allowed):
        return resolved
    raise NotAllowed(
        f"{resolved} is outside every registered directory "
        f"({', '.join(str(r) for r in allowed)}); register it with --allow"
    )
