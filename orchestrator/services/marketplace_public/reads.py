"""PRD-270 US-270-07a (K10) — the public Marketplace reads, by type.

``reader_for`` names each type's reader, and ``None`` for a type with none (the route's 404).
``public_page`` answers one page of a type as ``{type, items, limit, offset, more}``: the reader is
asked for one item more than the page, so ``more`` is known without a count. Each reader takes
``(db, limit, offset)``, runs its own query over the global catalogue rows and maps every row through
its type's allowlist (``fields.py``); nothing here reads a workspace, a person or an install.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional

from sqlalchemy.orm import Session

from services.marketplace_public.agents import read_agents
from services.marketplace_public.packages import read_packages
from services.marketplace_public.playbooks import read_playbooks

Reader = Callable[[Session, int, int], List[Dict[str, Any]]]

READERS: Mapping[str, Reader] = MappingProxyType({
    "agents": read_agents,
    "playbooks": read_playbooks,
    "packages": read_packages,
})


def reader_for(kind: str) -> Optional[Reader]:
    """The reader of the type ``kind``, or ``None`` when the public Marketplace has no such type."""
    return READERS.get(kind)


def public_page(db: Session, kind: str, limit: int, offset: int) -> Optional[Dict[str, Any]]:
    """One page of the type ``kind`` from ``offset``, at most ``limit`` items, or ``None`` for an unknown type."""
    reader = reader_for(kind)
    if reader is None:
        return None
    found = reader(db, limit + 1, offset)
    return {"type": kind, "items": found[:limit], "limit": limit, "offset": offset, "more": len(found) > limit}


__all__ = ["READERS", "Reader", "public_page", "reader_for"]
