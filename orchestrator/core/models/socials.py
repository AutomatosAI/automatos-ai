"""PRD-251 S0.2 (D2): the Socials tables.

* ``social_posts``: one row per post. Its approval binds to ``content_hash``
  (D6), so the publisher refuses a post whose ``approved_hash`` no longer
  matches. ``review_log`` is the history the approval UI shows.
* ``social_post_targets``: one row per channel per post, each with its own
  idempotency key, attempts, remote id, permalink and error.

JSON columns are ``JSON().with_variant(JSONB(), "postgresql")`` and id columns
the portable ``Uuid`` (native UUID on Postgres, CHAR(32) elsewhere), so the
tables also build under SQLite in the unit tests. The ``prd251_socials`` migration
creates the same shape, and ``tests/test_prd251_models.py`` holds the two
together. There is no campaigns table: D2 creates it in Wave 2, and only if
series approval ships.
"""

from __future__ import annotations

from typing import Any, Dict
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    false,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from core.database.base import Base

# D2: the twelve post statuses. Wave 0 moves posts through draft,
# needs_approval, changes_requested, approved, scheduled and archived; the rest
# arrive with rendering (Wave 1) and publishing (Wave 3).
SOCIAL_POST_STATUSES = (
    "draft",
    "rendering",
    "needs_approval",
    "changes_requested",
    "approved",
    "scheduled",
    "publishing",
    "published",
    "partially_published",
    "failed",
    "missed",
    "archived",
)
SOCIAL_POST_FORMATS = ("video", "image", "carousel", "fact_card", "infographic")
SOCIAL_TARGET_POST_KINDS = ("text", "image", "carousel", "video", "reel", "short", "story")
SOCIAL_TARGET_STATUSES = ("pending", "uploading", "published", "failed")


def _in_list(column: str, values: tuple) -> str:
    quoted = ", ".join(f"'{v}'" for v in values)
    return f"{column} IN ({quoted})"


def _json_type():
    return JSON().with_variant(JSONB(), "postgresql")


def _iso(value) -> Any:
    return value.isoformat() if value is not None else None


class SocialPost(Base):
    __tablename__ = "social_posts"
    __table_args__ = (
        CheckConstraint(_in_list("status", SOCIAL_POST_STATUSES), name="ck_social_posts_status"),
        CheckConstraint(_in_list("format", SOCIAL_POST_FORMATS), name="ck_social_posts_format"),
        Index("ix_social_posts_workspace_status", "workspace_id", "status"),
        Index("ix_social_posts_workspace_scheduled_for", "workspace_id", "scheduled_for"),
        {"extend_existing": True},
    )

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    workspace_id = Column(
        Uuid(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    # The user or agent that made the post.
    created_by = Column(String(255), nullable=False)
    # The campaigns table arrives in Wave 2, only if series approval ships: no FK yet.
    campaign_id = Column(Uuid(as_uuid=True), nullable=True)

    title = Column(String(500), nullable=False)
    # The ask the post answers.
    brief = Column(Text, nullable=True)
    # {"base": text, "channels": {toolkit: text}}: the base text plus per-channel overrides.
    copy = Column(_json_type(), nullable=False, default=dict)
    # One of SOCIAL_POST_FORMATS; NULL until a format is chosen.
    format = Column(String(20), nullable=True)
    # A document_templates id (social templates are Wave 1, S1.2).
    template_id = Column(Uuid(as_uuid=True), nullable=True)
    # {name: {"value": ..., "claim": bool}}
    variables = Column(_json_type(), nullable=False, default=dict)
    # {claim name: {"kind", "ref", "as_of"}} (D7)
    sources = Column(_json_type(), nullable=False, default=dict)
    # {aspect: [deliverable ids]}
    media = Column(_json_type(), nullable=False, default=dict)
    # D11 (Wave 1, S1.5): the voice a render speaks the script with. NULL is
    # Kokoro, the template's own voice; {"toolkit", "voice_id", "name"} is a
    # voice toolkit the workspace has connected in Composio. A render setting,
    # not content: the rendered files' digests carry what it changed into the hash.
    # Added by the prd251_wave1 migration.
    voice = Column(_json_type(), nullable=True)
    # S1.8 (D12, Wave 1): footage and stills for the template's slots from the
    # workspace's Composio generation toolkit. {slot: {"prompt"}} is what the
    # post asks for; a render generates it, copies the file into our storage,
    # and records it there ("status": "done", its Deliverable, sha256, what it
    # cost). NULL: every slot plays the template's own motion graphics. A render
    # setting like voice: outside the content hash, which binds the rendered
    # files. Added by the prd251_wave1 migration.
    footage = Column(_json_type(), nullable=True)

    status = Column(String(32), nullable=False, default="draft", server_default="draft")
    content_hash = Column(String(64), nullable=False)
    approved_hash = Column(String(64), nullable=True)
    approved_by = Column(String(255), nullable=True)
    approved_at = Column(DateTime(timezone=True), nullable=True)
    override_unsourced = Column(Boolean, nullable=False, default=False, server_default=false())
    # [{"at", "by", "action", "comment"}]: request-changes comments and reject reasons live here.
    review_log = Column(_json_type(), nullable=False, default=list)

    scheduled_for = Column(DateTime(timezone=True), nullable=True)  # UTC
    timezone = Column(String(64), nullable=True)  # IANA name, for display

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "workspace_id": str(self.workspace_id),
            "created_by": self.created_by,
            "campaign_id": str(self.campaign_id) if self.campaign_id else None,
            "title": self.title,
            "brief": self.brief,
            "copy": self.copy or {},
            "format": self.format,
            "template_id": str(self.template_id) if self.template_id else None,
            "variables": self.variables or {},
            "sources": self.sources or {},
            "media": self.media or {},
            "voice": self.voice or None,
            "footage": self.footage or None,
            "status": self.status,
            "content_hash": self.content_hash,
            "approved_hash": self.approved_hash,
            "approved_by": self.approved_by,
            "approved_at": _iso(self.approved_at),
            "override_unsourced": bool(self.override_unsourced),
            "review_log": self.review_log or [],
            "scheduled_for": _iso(self.scheduled_for),
            "timezone": self.timezone,
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
        }


class SocialPostTarget(Base):
    __tablename__ = "social_post_targets"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_social_post_targets_idempotency_key"),
        CheckConstraint(
            _in_list("post_kind", SOCIAL_TARGET_POST_KINDS), name="ck_social_post_targets_post_kind"
        ),
        CheckConstraint(
            _in_list("status", SOCIAL_TARGET_STATUSES), name="ck_social_post_targets_status"
        ),
        Index("ix_social_post_targets_post_id", "post_id"),
        {"extend_existing": True},
    )

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    post_id = Column(
        Uuid(as_uuid=True), ForeignKey("social_posts.id", ondelete="CASCADE"), nullable=False
    )
    # The Composio app name (linkedin, twitter, instagram, ...).
    toolkit = Column(String(100), nullable=False)
    post_kind = Column(String(20), nullable=False)
    # The resolved action slugs and parameters.
    action_plan = Column(_json_type(), nullable=False, default=dict)
    idempotency_key = Column(String(128), nullable=False)
    status = Column(String(20), nullable=False, default="pending", server_default="pending")
    attempts = Column(Integer, nullable=False, default=0, server_default="0")
    remote_id = Column(String(255), nullable=True)
    permalink = Column(String(1000), nullable=True)
    error = Column(Text, nullable=True)
    published_at = Column(DateTime(timezone=True), nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "post_id": str(self.post_id),
            "toolkit": self.toolkit,
            "post_kind": self.post_kind,
            "action_plan": self.action_plan or {},
            "idempotency_key": self.idempotency_key,
            "status": self.status,
            "attempts": self.attempts,
            "remote_id": self.remote_id,
            "permalink": self.permalink,
            "error": self.error,
            "published_at": _iso(self.published_at),
        }
