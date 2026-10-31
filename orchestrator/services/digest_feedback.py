"""One thumbs rating per person for each Auto's Read or brief (PRD-221 S10, PRD-268 §6.4).

``digest_feedback`` keeps ONE row per (workspace, person, state_hash). A second
press by the same person updates that row's rating instead of adding a row, so
a double press, a remounted card and a theme's own thumbs (SDK ``useRateBrief``)
all land on the same rating.

* **The person.** ``user_id`` holds the rater's ``users.id`` as text, from
  ``resolve_user_pk``, the one principal resolver: the Clerk, local-operator and
  paired-device lanes all name the same person the same way. Rows written
  before this rule hold a Clerk id (or NULL) instead; they are left as they
  were and never match a person's rating.
* **No person.** A request with no resolvable person (an API key, or a local
  operator with no ``users`` row) has no rating of its own: its press is kept
  as a row of its own (``user_id`` NULL), as before, and its read is null, so
  no caller ever reads a rating it can't be told apart from.
* **A double press.** The write holds a transaction-scoped advisory lock on the
  (workspace, person, state_hash) key, so two presses at once run one after the
  other: the second sees the first's row and updates it. The table has no unique
  index on the key, so the lock is what keeps it to one row.
* **More than one row.** Every row the person already has for the state_hash
  takes the new rating, so a read of any of them agrees.
"""
from __future__ import annotations

from typing import Any, Optional

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from core.auth.principal import resolve_user_pk
from core.models.core import DigestFeedback

# The longest state_hash the table holds (digest_feedback.state_hash).
STATE_HASH_MAX_LENGTH = DigestFeedback.__table__.c.state_hash.type.length
# A rating's lock in the database, in a key space of its own ('dgfb').
RATING_LOCK_NAMESPACE = 0x64676662
_RATING_LOCK = text("SELECT pg_advisory_xact_lock(:namespace, hashtext(:key))")


def rater_of(db: Session, ctx: Any) -> Optional[str]:
    """The caller's rater key: their ``users.id`` as text, or None when no person stands behind the request."""
    user_pk = resolve_user_pk(db, ctx)
    return None if user_pk is None else str(user_pk)


def _ratings_of(workspace_id: Any, rater: str, state_hash: str) -> tuple:
    """The filter for this person's rows of this state in this workspace."""
    return (
        DigestFeedback.workspace_id == workspace_id,
        DigestFeedback.user_id == rater,
        DigestFeedback.state_hash == state_hash,
    )


def hold_rating_lock(db: Session, workspace_id: Any, rater: str, state_hash: str) -> None:
    """Hold the (workspace, person, state_hash) rating lock until this transaction ends."""
    key = f"{workspace_id}:{rater}:{state_hash}"
    db.execute(_RATING_LOCK, {"namespace": RATING_LOCK_NAMESPACE, "key": key})


def record_rating(db: Session, workspace_id: Any, rater: Optional[str], state_hash: str, rating: int) -> None:
    """Keep the person's one rating of the state: update the row they have, or add it; then commit.

    With no person (``rater`` None) there is no rating to keep: the press is a row of its own.
    """
    rows = []
    if rater is not None:
        hold_rating_lock(db, workspace_id, rater, state_hash)
        rows = list(db.scalars(select(DigestFeedback).where(*_ratings_of(workspace_id, rater, state_hash))))
    for row in rows:
        row.rating = rating
    if not rows:
        db.add(DigestFeedback(workspace_id=workspace_id, user_id=rater, state_hash=state_hash, rating=rating))
    db.commit()


def rating_of(db: Session, workspace_id: Any, rater: Optional[str], state_hash: str) -> Optional[int]:
    """The person's own rating of the state (1 or -1); None when they haven't rated it, or there is no person."""
    if rater is None:
        return None
    newest = (
        select(DigestFeedback.rating)
        .where(*_ratings_of(workspace_id, rater, state_hash))
        .order_by(DigestFeedback.created_at.desc())
        .limit(1)
    )
    return db.scalars(newest).first()
