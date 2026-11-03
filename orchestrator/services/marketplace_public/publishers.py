"""PRD-270 K10 — who published a Marketplace agent or Playbook, as a visitor reads it.

The original creator's ``users.name``, else ``Automatos``. Never an email: the member route falls back
to a creator's email (``api/marketplace.py`` ``list_items``), and a name that holds an ``@`` reads
``Automatos`` too, since some accounts carry their address as their name. Never ``Agent.created_by``
either, a free-text field the member route also falls back to.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

from sqlalchemy.orm import Session

from core.models.core import User
from services.marketplace_public.fields import DEFAULT_PUBLISHER, text_or_none

EMAIL_MARK = "@"


def display_name(name: Optional[str]) -> str:
    """A creator's name as a visitor reads it: trimmed, else ``Automatos``; one that holds an ``@`` too."""
    text = text_or_none(name)
    return DEFAULT_PUBLISHER if text is None or EMAIL_MARK in text else text


def publisher_names(db: Session, user_ids: Iterable[Optional[int]]) -> Dict[int, str]:
    """Each named creator's display name, read in one query of ``users.id`` and ``users.name`` only."""
    ids = sorted({user_id for user_id in user_ids if user_id is not None})
    if not ids:
        return {}
    rows = db.query(User.id, User.name).filter(User.id.in_(ids)).all()
    return {row.id: display_name(row.name) for row in rows}


def publisher_of(names: Mapping[int, str], user_id: Optional[int]) -> str:
    """The publisher of an item whose original creator is ``user_id``: their display name, else ``Automatos``."""
    return names.get(user_id, DEFAULT_PUBLISHER) if user_id is not None else DEFAULT_PUBLISHER


__all__ = ["display_name", "publisher_names", "publisher_of"]
