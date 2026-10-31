"""PRD-268 §6.4, FR-10, K11 — one brief per person, workspace and local date, whichever comes first.

The scheduled digest (``services.mobile.digest``, via ``schedule``) and the brief on open (via
``open``, every device and tab) share one claim: the (workspace, person, local date) row of
``mobile_digest_posts``, its primary key, unchanged (K6). Whoever wins it builds the brief and
posts it; every other claimant answers the day's message, found by its source in the person's
Auto thread. A build or post that fails gives the claim back (``services.mobile.digest.release``),
so the next open or tick tries again (unlike OpenDuet, the PRD's reference, where a failed day
goes without a brief). The claim is a lease on its ``posted_at`` (P268W2-RVW-1): a claim left by
a claimant that died before its post is taken over once it is stale (``digest.take_over``), and
a claimant posts only while the day is still the one it holds (``digest.post_held_brief``).

* ``claim_brief``: the digest's own claim, ``INSERT … ON CONFLICT DO NOTHING``, committed;
* ``todays_brief_message``: the day's message (the newest of them), read in the person's own
  Auto thread in the workspace only;
* ``post_brief``: ``[text(prose), brief@1]`` through the digest's post
  (``deliver_background_message``), with source {origin ``daily_digest``, local_date, via}; given
  the caller's lease, only while the day is still the caller's. A brief posted on open is never
  announced as ``auto_message``: opening the app is the person's own action (PRD-266 K11). The
  scheduled one is, as today.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any, Mapping, Optional

from sqlalchemy.orm import Session

from core.models.core import Chat, Message
from services.brief.model import VIAS
from services.mobile import digest
from services.mobile.digest import DuePerson
from services.notification_events.auto_message import DAILY_DIGEST

__all__ = ["claim_brief", "post_brief", "todays_brief_message"]

# The kind of the per-person Auto thread (services.chat_messenger.find_or_create_auto_chat).
AUTO_THREAD = "auto"
ASSISTANT = "assistant"


def _workspace(workspace_id: Any) -> uuid.UUID:
    return workspace_id if isinstance(workspace_id, uuid.UUID) else uuid.UUID(str(workspace_id))


def claim_brief(db: Session, workspace_id: Any, user_id: int, local_date: date) -> bool:
    """Claim the person's brief for ``local_date``, committed; False when it was claimed already."""
    return digest.claim(db, DuePerson(_workspace(workspace_id), int(user_id), local_date))


def todays_brief_message(db: Session, workspace_id: Any, user_id: int, local_date: date) -> Optional[Message]:
    """The day's brief: the newest of Auto's messages in the person's Auto thread in the workspace
    whose source is the daily digest for ``local_date``; None when it is not there (yet)."""
    workspace = _workspace(workspace_id)
    return (
        db.query(Message)
        .join(Chat, Chat.id == Message.chat_id)
        .filter(
            Chat.workspace_id == workspace,
            Chat.user_id == int(user_id),
            Chat.kind == AUTO_THREAD,
            Message.workspace_id == workspace,
            Message.role == ASSISTANT,
            Message.source["origin"].astext == DAILY_DIGEST,
            Message.source["local_date"].astext == local_date.isoformat(),
        )
        .order_by(Message.created_at.desc(), Message.id.desc())
        .first()
    )


def post_brief(
    db: Session, person: DuePerson, brief: Mapping[str, Any], via: str, *, held: Optional[datetime] = None,
) -> Optional[Message]:
    """Post ``brief`` (a ``Brief@1``) into the person's Auto thread for their local date, as
    ``[text(prose), brief@1]`` with source {origin ``daily_digest``, local_date, ``via``}; the saved
    message, or None when it did not land. ``via`` is ``schedule`` or ``open`` (ValueError else,
    before anything is read or written). With ``held``, the claim's ``posted_at`` the caller
    holds, it posts only while the day is still the caller's (``digest.post_held_brief``:
    ``LeaseLost`` when it was taken over, the claim given back when the post does not land)."""
    if via not in VIAS:
        raise ValueError(f"a brief is posted via one of {VIAS}, not {str(via)[:32]!r}")
    if held is None:
        return digest.post_brief_message(db, person, brief, via)
    return digest.post_held_brief(db, person, brief, via, held)
