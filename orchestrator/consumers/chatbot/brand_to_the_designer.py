"""PRD-255 US-014: the workspace's Brand designer, found or seeded for a brand ask's ticket.

The owner, 4 Oct: "Automatos becomes your brand; Auto is the one voice and delegates, agents do
the work." A brand ask (the kit, the look of the documents, the templates) is the Brand
designer's ticket: the words, the pin and the note are a row of the hand-off table
(``handoffs``, PRD-256 US-011). What stays here is the designer itself: a workspace with no
designer yet gets its one (``seed_brand_designer``, find-or-seed; an existing hosted workspace
whose Auto was seeded before PRD-255 gets it here), and one the owner removed is not brought back.
"""
from __future__ import annotations

from typing import Any, Callable, Optional
from uuid import UUID


def _seed_in_own_session(workspace_id: UUID) -> Optional[str]:
    """Seed the workspace's designer in a session of its own, committed before the turn files the
    ticket; its name, or None when the owner removed it."""
    from core.database.database import SessionLocal
    from core.seeds.seed_brand_designer import seed_brand_designer

    db = SessionLocal()
    try:
        agent = seed_brand_designer(db, workspace_id)
        db.commit()
        return getattr(agent, "name", None)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


Seeder = Callable[[UUID], Optional[str]]


def designer_name(db: Any, workspace_id: UUID, seed: Seeder = _seed_in_own_session) -> Optional[str]:
    """The workspace's Brand designer's name: found, or seeded now. None when the owner removed it."""
    from core.seeds.seed_brand_designer import find_brand_designer

    found = find_brand_designer(db, workspace_id)
    return found.name if found is not None else seed(workspace_id)


__all__ = ["designer_name"]
