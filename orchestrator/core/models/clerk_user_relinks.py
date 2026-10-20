"""The audit record of the Clerk production cutover's relinks (19 Oct 2026).

Moving Clerk from its development instance to production gives every returning
person a new Clerk user ID. ``users`` rows are found by ``clerk_user_id``, so each
returning row is moved from its development-instance ID to its production ID ONCE,
by the operator step ``scripts/clerk_cutover_relink.py`` (logic in
``core.auth.clerk_cutover``). Sign-in never relinks.

One row here per relinked ``users`` row, written in the same transaction as the
move:

- ``user_id`` is unique: a ``users`` row is relinked at most once, ever, and only
  to the one production identity recorded here. A rollback to the development
  instance moves it back (``reverted_at``); a retried cutover re-applies only this
  same pair.
- ``new_clerk_user_id`` is unique: one production identity holds at most one row.
- ``verified_email`` is the verified address of the production user that matched
  the row's email when it moved.

The Alembic revision ``clerk_user_relinks_20261019`` declares the same names.
"""
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, UniqueConstraint, func

from core.database.base import Base


class ClerkUserRelink(Base):
    """One ``users`` row moved from its development-instance Clerk ID to its production ID."""

    __tablename__ = "clerk_user_relinks"

    id = Column(Integer, primary_key=True)
    # CASCADE: erasing a person erases this record of them too.
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    old_clerk_user_id = Column(String(255), nullable=False)
    new_clerk_user_id = Column(String(255), nullable=False)
    verified_email = Column(String(255), nullable=False)
    relinked_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    reverted_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint("user_id", name="uq_clerk_user_relinks_user"),
        UniqueConstraint("new_clerk_user_id", name="uq_clerk_user_relinks_new_id"),
        {"extend_existing": True},
    )
