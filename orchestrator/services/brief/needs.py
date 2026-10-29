"""PRD-268 §6.11.2 — the brief's "Needs you": ``needs_you()``'s rows, with W1's status and next step.

Needs-you is never recounted (PRD-252 R5): the count is ``needs_you()``'s ``total``, the
number every badge shows, and the rows are its rows in its kinds' order, each with the
``display_state`` and ``next_step`` W1 gives it (``needs_you_with_status``) in the person's
language. ``may_answer`` False (a viewer, PRD-266 D16) leaves questions and approval grants
out of both, as ``needs_you()`` does.

A row opens what its needs-you row opens: a question or approval grant opens the grant
(``question``), a Mission waiting for its plan's approval and a row whose decision is its
Mission's open the Mission, any other row its ticket. A question's action is Answer; any
other row's, Open.
"""
from __future__ import annotations

from itertools import islice
from typing import Any, List, Mapping, Optional, Tuple

from sqlalchemy.orm import Session

from services.brief.model import BriefOpens, BriefRow
from services.brief.rows import ROWS_READ, WHY_CHARS, action, cut, opens, title_of
from services.needs_you import KINDS, OPENS_MISSION, needs_you
from services.proactive.display_state import GRANT, MISSION, needs_you_with_status

__all__ = ["needs_rows"]

KIND_QUESTION = "question"


def needs_rows(db: Session, workspace_id: Any, language: str, *, may_answer: bool) -> Tuple[int, List[BriefRow]]:
    """The Needs-you number and its first ``ROWS_READ`` rows, as brief rows."""
    found = needs_you_with_status(needs_you(db, workspace_id, may_answer=may_answer), language)
    listed = ((kind, row) for kind in KINDS for row in found["rows"].get(kind, []))
    return int(found["total"]), [_row(kind, row, language) for kind, row in islice(listed, ROWS_READ)]


def _native(row: Mapping[str, Any]) -> Any:
    """The id the row is: a grant's or a Mission's own, else its ticket's."""
    return row.get("id") if row.get("source") in (GRANT, MISSION) else row.get("ticket_id")


def _opens(row: Mapping[str, Any]) -> Optional[BriefOpens]:
    source = row.get("source")
    if source == GRANT:
        return opens("question", row.get("id"))
    if source == MISSION:
        return opens("mission", row.get("id"))
    if row.get("opens") == OPENS_MISSION and row.get("mission_id"):
        return opens("mission", row["mission_id"])
    return opens("ticket", row.get("ticket_id"))


def _row(kind: str, row: Mapping[str, Any], language: str) -> BriefRow:
    return BriefRow(
        id=f"{kind}:{_native(row)}",
        title=title_of(row.get("title") or row.get("number"), language),
        why=cut(row.get("next_step"), WHY_CHARS),
        at=row.get("at"),
        number=row.get("number"),
        display_state=row.get("display_state"),
        opens=_opens(row),
        action=action("answer" if kind == KIND_QUESTION else "open", language),
    )
