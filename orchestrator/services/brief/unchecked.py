"""PRD-268 §6.11.2, §6.11.3 — the brief's "Couldn't check": what the brief depends on that isn't ready.

The workspace's readiness (``services.readiness.service.current``: the last check while it
is held, else a check now, each probe bounded by its timeout), kept to the items needed by
the brief whose state is not ``ready``. Never hidden: a quiet brief must not hide a broken
connection. Each row is the item's label and its detail, in the person's language. An item
that needs setting up is a person's to act on (◆, ``needs_you``); one that didn't answer in
time draws no state (○ is not a ``DisplayState``). A fix opens its app path in Settings, with
its own label as the action.
"""
from __future__ import annotations

from typing import Any, List, Mapping
from uuid import UUID

from services.brief.model import BriefAction, BriefRow
from services.brief.rows import WHY_CHARS, cut, opens, title_of
from services.proactive.display_state import NEEDS_YOU
from services.readiness.items import FOR_BRIEF, NEEDS_SETUP, READY
from services.readiness.service import current

__all__ = ["unchecked_rows"]


def unchecked_rows(workspace_id: UUID, language: str) -> List[BriefRow]:
    """Every readiness item the brief needs that is not ready, as a brief row."""
    readiness = current(workspace_id, language)
    return [_row(item, language) for item in readiness.get("items") or []
            if FOR_BRIEF in (item.get("needed_by") or []) and item.get("state") != READY]


def _row(item: Mapping[str, Any], language: str) -> BriefRow:
    fix = item.get("fix") or {}
    return BriefRow(
        id=f"readiness:{item.get('id')}",
        title=title_of(item.get("label"), language),
        why=cut(item.get("detail"), WHY_CHARS),
        display_state=NEEDS_YOU if item.get("state") == NEEDS_SETUP else None,
        opens=opens("settings", fix.get("href")),
        action=BriefAction(verb="connect", label=fix["label"]) if fix.get("label") else None,
    )
