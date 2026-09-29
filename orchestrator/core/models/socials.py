"""PRD-251 S0.2 (D2): the Socials tables.

* ``social_campaigns`` (Wave 2, S2.4): a named set of posts, approved post by
  post or as a series (D6). ``approved_hash_set`` holds the content hashes a
  series approval approved.
* ``social_posts``: one row per post. Its approval binds to ``content_hash``
  (D6), so the publisher refuses a post whose ``approved_hash`` no longer
  matches. ``review_log`` is the history the approval UI shows.
* ``social_post_targets``: one row per channel per post, each with its own
  idempotency key, attempts, remote id, permalink and error. The post's
  channels are approved content (Wave 2, US-204): ``SocialPost.targets`` loads
  them with the post, and the content hash covers each one's toolkit, post kind
  and options (``modules/socials/targets.py``).

JSON columns are ``JSON().with_variant(JSONB(), "postgresql")`` and id columns
the portable ``Uuid`` (native UUID on Postgres, CHAR(32) elsewhere), so the
tables also build under SQLite in the unit tests. The ``prd251_socials``
migration creates the posts and targets, and ``tests/test_prd251_models.py``
holds them and the model together. D2 creates the campaigns table in Wave 2,
because series approval ships there: the ``prd251_wave2`` migration builds it
and links ``social_posts.campaign_id`` to it, and
``tests/test_prd251w2_campaigns.py`` holds that migration and the model together.
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
from sqlalchemy.orm import relationship
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
# D6 (Wave 2, S2.4): per_post approves each post on its own; series approves
# every post of the campaign whose content hash is in the approved set when the
# approval is given. A post added or edited later still needs its own approval.
SOCIAL_CAMPAIGN_APPROVAL_MODES = ("per_post", "series")


def _in_list(column: str, values: tuple) -> str:
    quoted = ", ".join(f"'{v}'" for v in values)
    return f"{column} IN ({quoted})"


def _json_type():
    return JSON().with_variant(JSONB(), "postgresql")


def _iso(value) -> Any:
    return value.isoformat() if value is not None else None


class SocialCampaign(Base):
    """A named set of posts (D2, Wave 2).

    Series approval (S2.4) approves every post whose content hash is in the
    approved set when the approval is given, each through the same hash-bound
    approve as a single post (D6); ``approved_hash_set`` records those hashes.
    """

    __tablename__ = "social_campaigns"
    __table_args__ = (
        CheckConstraint(
            _in_list("approval_mode", SOCIAL_CAMPAIGN_APPROVAL_MODES),
            name="ck_social_campaigns_approval_mode",
        ),
        Index("ix_social_campaigns_workspace_created", "workspace_id", "created_at"),
        {"extend_existing": True},
    )

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    workspace_id = Column(
        Uuid(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    name = Column(String(200), nullable=False)
    # One of SOCIAL_CAMPAIGN_APPROVAL_MODES.
    approval_mode = Column(
        String(16), nullable=False, default="per_post", server_default="per_post"
    )
    # [content hash]: the posts' hashes (D6) a series approval approved.
    approved_hash_set = Column(_json_type(), nullable=False, default=list)
    approved_by = Column(String(255), nullable=True)
    approved_at = Column(DateTime(timezone=True), nullable=True)
    # The user or agent that made the campaign.
    created_by = Column(String(255), nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "workspace_id": str(self.workspace_id),
            "name": self.name,
            "approval_mode": self.approval_mode,
            "approved_hash_set": list(self.approved_hash_set or []),
            "approved_by": self.approved_by,
            "approved_at": _iso(self.approved_at),
            "created_by": self.created_by,
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
        }


class SocialPost(Base):
    __tablename__ = "social_posts"
    __table_args__ = (
        CheckConstraint(_in_list("status", SOCIAL_POST_STATUSES), name="ck_social_posts_status"),
        CheckConstraint(_in_list("format", SOCIAL_POST_FORMATS), name="ck_social_posts_format"),
        Index("ix_social_posts_workspace_status", "workspace_id", "status"),
        Index("ix_social_posts_workspace_scheduled_for", "workspace_id", "scheduled_for"),
        Index("ix_social_posts_campaign_id", "campaign_id"),
        {"extend_existing": True},
    )

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    workspace_id = Column(
        Uuid(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    # The user or agent that made the post.
    created_by = Column(String(255), nullable=False)
    # The campaign the post belongs to (Wave 2). Deleting a campaign keeps its
    # posts, unlinked. The key carries the name Postgres gives an unnamed one, so
    # a database create_all built and one the prd251_wave2 migration linked agree.
    campaign_id = Column(
        Uuid(as_uuid=True),
        ForeignKey(
            "social_campaigns.id", ondelete="SET NULL", name="social_posts_campaign_id_fkey"
        ),
        nullable=True,
    )

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
    # S2.2b (US-208, Wave 2): the composer's last preview render, half resolution:
    # {"status": "rendering" | "done" | "failed", "content_hash" (the version it
    # rendered), "files": [{"name", "url", "content_type", "aspect", "duration",
    # "width", "height"}], "error", "at"}. Not content: outside the content hash
    # and media, and never a move of the status. Added by the prd251_wave2 migration.
    preview = Column(_json_type(), nullable=True)

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

    # The channels the post publishes to (US-204): approved content, like the
    # copy, so the content hash reads them. Loaded with the post (selectin: one
    # query for a whole list of posts). A target dropped from the list is deleted.
    targets = relationship(
        "SocialPostTarget", cascade="all, delete-orphan", passive_deletes=True, lazy="selectin"
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
            "preview": self.preview or None,
            "status": self.status,
            "content_hash": self.content_hash,
            "approved_hash": self.approved_hash,
            "approved_by": self.approved_by,
            "approved_at": _iso(self.approved_at),
            "override_unsourced": bool(self.override_unsourced),
            "review_log": self.review_log or [],
            "scheduled_for": _iso(self.scheduled_for),
            "timezone": self.timezone,
            "targets": [
                target.to_dict()
                for target in sorted(self.targets or [], key=lambda t: (t.toolkit, t.post_kind))
            ],
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
    # The resolved action slugs and parameters: {"options": {name: value}, the
    # target's own choices (``$option.<name>`` in its steps), and "steps", the
    # kind's action sequence as the channel registry resolved it (US-204).
    action_plan = Column(_json_type(), nullable=False, default=dict)
    # sp:{post_id}:{toolkit}:{post_kind}, one per channel and kind of a post.
    idempotency_key = Column(String(128), nullable=False)
    status = Column(String(20), nullable=False, default="pending", server_default="pending")
    attempts = Column(Integer, nullable=False, default=0, server_default="0")
    remote_id = Column(String(255), nullable=True)
    permalink = Column(String(1000), nullable=True)
    error = Column(Text, nullable=True)
    published_at = Column(DateTime(timezone=True), nullable=True)

    @property
    def options(self) -> Dict[str, Any]:
        """The target's options, from its ``action_plan`` (a new dict)."""
        plan = self.action_plan if isinstance(self.action_plan, dict) else {}
        options = plan.get("options")
        return dict(options) if isinstance(options, dict) else {}

    def to_dict(self) -> Dict[str, Any]:
        """As the post answers it (US-204): the options and the receipt. The
        action sequence and the idempotency key stay server-side."""
        return {
            "id": str(self.id) if self.id else None,
            "toolkit": self.toolkit,
            "post_kind": self.post_kind,
            "options": self.options,
            "status": self.status,
            "attempts": self.attempts,
            "remote_id": self.remote_id,
            "permalink": self.permalink,
            "error": self.error,
            "published_at": _iso(self.published_at),
        }
