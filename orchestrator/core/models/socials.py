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

from typing import Any, Dict, List
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Column,
    Date,
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
SOCIAL_POST_FORMATS = ("video", "image", "carousel", "fact_card", "infographic", "text")
# PRD-251B (US-B101, B5): a video length the editor chose. NULL until chosen; a
# chosen one is a positive number of seconds the template declares.
POST_LENGTH_SECONDS_CHECK = "length_seconds IS NULL OR length_seconds > 0"
SOCIAL_TARGET_POST_KINDS = ("text", "image", "carousel", "video", "reel", "short", "story")
SOCIAL_TARGET_STATUSES = ("pending", "uploading", "published", "failed")
# D6 (Wave 2, S2.4): per_post approves each post on its own; series approves
# every post of the campaign whose content hash is in the approved set when the
# approval is given. A post added or edited later still needs its own approval.
SOCIAL_CAMPAIGN_APPROVAL_MODES = ("per_post", "series")
# PRD-251B (B6, B7, B11; US-B201): a plan is a campaign of kind "plan": dates, a
# cadence, sources to research, how and when posts are made, and what a passed
# slot does. Its posts are made on their day (B7), never ahead.
SOCIAL_CAMPAIGN_KINDS = ("campaign", "plan")
SOCIAL_PLAN_STATUSES = ("active", "paused", "ended")
SOCIAL_LATE_POLICIES = ("skip", "next_slot")
# B8: a content-bank topic comes from the research run or from a person.
SOCIAL_TOPIC_ORIGINS = ("research", "person")


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
        CheckConstraint(_in_list("kind", SOCIAL_CAMPAIGN_KINDS), name="ck_social_campaigns_kind"),
        CheckConstraint(_in_list("status", SOCIAL_PLAN_STATUSES), name="ck_social_campaigns_status"),
        CheckConstraint(_in_list("late_policy", SOCIAL_LATE_POLICIES), name="ck_social_campaigns_late_policy"),
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

    # PRD-251B (B6, US-B201): the plan fields, added by the prd251b_wave2 migration.
    # Server defaults only, so a database a step behind still takes the inserts.
    kind = Column(String(16), nullable=False, server_default="campaign")
    status = Column(String(16), nullable=False, server_default="active")
    goal = Column(Text, nullable=True)
    audience = Column(Text, nullable=True)
    starts_on = Column(Date, nullable=True)
    ends_on = Column(Date, nullable=True)
    timezone = Column(String(64), nullable=True)  # IANA name: the plan's days and times are local to it
    # [{"id", "channels": [toolkit], "format", "length_seconds", "template_id", "days": ["mon", ...], "time": "HH:MM"}]
    cadence = Column(_json_type(), nullable=True)
    # {"knowledge", "deliverables", "website", "github": bool, "notes": text, "never_say": [phrase]}
    sources = Column(_json_type(), nullable=True)
    # {"time": "HH:MM", "video_days_early": int, "visual_mix": {...}, "max_per_day": int}
    make = Column(_json_type(), nullable=True)
    late_policy = Column(String(16), nullable=False, server_default="skip")
    # {"day": "mon", "time": "HH:MM", "last_run_at", "last_run_id"}: the weekly research run
    research = Column(_json_type(), nullable=True)
    # {slot_key: {"skip": true} | {"to": ISO datetime}}: planned slots moved or skipped
    slot_overrides = Column(_json_type(), nullable=True)

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
            "kind": self.kind or "campaign",
            "status": self.status or "active",
            "goal": self.goal,
            "audience": self.audience,
            "starts_on": _iso(self.starts_on),
            "ends_on": _iso(self.ends_on),
            "timezone": self.timezone,
            "cadence": list(self.cadence or []),
            "sources": dict(self.sources or {}),
            "make": dict(self.make or {}),
            "late_policy": self.late_policy or "skip",
            "research": dict(self.research or {}),
            "slot_overrides": dict(self.slot_overrides or {}),
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
        CheckConstraint(POST_LENGTH_SECONDS_CHECK, name="ck_social_posts_length_seconds"),
        Index("ix_social_posts_workspace_planned_for", "workspace_id", "planned_for"),
        # PRD-251B (B7): one post per planned slot of a plan; a NULL key never collides.
        Index("uq_social_posts_campaign_slot_key", "campaign_id", "slot_key", unique=True),
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
    # PRD-251B (US-B101, B11): the slot the post is planned for, UTC, set from the
    # editor's When or by a plan, before approval. Not content: outside the hash,
    # so moving a slot never voids an approval; approving a post with a future slot
    # schedules it there (US-B105). Added by the prd251b_wave1 migration.
    planned_for = Column(DateTime(timezone=True), nullable=True)
    # PRD-251B (US-B101, B5): the video length the editor chose, in seconds, one the
    # template declares (US-B104). Content: the hash covers it once set. Added by
    # the prd251b_wave1 migration.
    length_seconds = Column(Integer, nullable=True)
    # PRD-251B (B7, US-B205): the plan's slot the post was made for
    # ("<row id>|<YYYY-MM-DD>|<HH:MM>", unique with campaign_id). Not content.
    slot_key = Column(String(160), nullable=True)
    # The editor's music (US-B109): NULL plays the template's own track; {"track": id}
    # a track of media-render's library; {"track": null} no music. A render setting
    # like voice: outside the content hash. Both added by the prd251b_wave2 migration.
    music = Column(_json_type(), nullable=True)

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
            "planned_for": _iso(self.planned_for),
            "length_seconds": self.length_seconds,
            "slot_key": self.slot_key,
            "music": self.music or None,
            "targets": [
                target.to_dict()
                for target in sorted(self.targets or [], key=lambda t: (t.toolkit, t.post_kind))
            ],
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
        }


class SocialTopic(Base):
    """A plan's content-bank topic (PRD-251B B8, US-B201/B203).

    Research or a person adds it; every fact carries its source. The make tick
    (US-B205) takes the next unused topic that suits a slot's format, and records
    the post that used it. Deleting the plan deletes its bank.
    """

    __tablename__ = "social_topics"
    __table_args__ = (
        CheckConstraint(_in_list("origin", SOCIAL_TOPIC_ORIGINS), name="ck_social_topics_origin"),
        Index("ix_social_topics_campaign_used", "campaign_id", "used_at"),
        Index("ix_social_topics_workspace_id", "workspace_id"),
        {"extend_existing": True},
    )

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    workspace_id = Column(Uuid(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    campaign_id = Column(Uuid(as_uuid=True), ForeignKey("social_campaigns.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(200), nullable=False)
    angle = Column(Text, nullable=True)
    # [{"text", "source": {"kind": knowledge|deliverable|web|github|note, "ref", "label"}}]
    facts = Column(_json_type(), nullable=False, default=list)
    # The post formats it suits (SOCIAL_POST_FORMATS); empty suits any.
    formats = Column(_json_type(), nullable=False, default=list)
    # A pinned topic is used on this day first.
    pinned_on = Column(Date, nullable=True)
    used_post_id = Column(Uuid(as_uuid=True), ForeignKey("social_posts.id", ondelete="SET NULL"), nullable=True)
    used_at = Column(DateTime(timezone=True), nullable=True)
    origin = Column(String(16), nullable=False, server_default="person")
    created_by = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "workspace_id": str(self.workspace_id),
            "plan_id": str(self.campaign_id),
            "title": self.title,
            "angle": self.angle,
            "facts": list(self.facts or []),
            "formats": list(self.formats or []),
            "pinned_on": _iso(self.pinned_on),
            "used_post_id": str(self.used_post_id) if self.used_post_id else None,
            "used_at": _iso(self.used_at),
            "origin": self.origin or "person",
            "created_by": self.created_by,
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
            "notes": self.notes,
        }

    @property
    def notes(self) -> List[str]:
        """What the receipt says beside the id and link (Wave 3, US-301): an optional
        step the publish skipped, and why. Kept in ``action_plan``."""
        plan = self.action_plan if isinstance(self.action_plan, dict) else {}
        notes = plan.get("notes")
        return [str(note) for note in notes] if isinstance(notes, list) else []
