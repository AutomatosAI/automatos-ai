"""PRD-251 Socials — the ONE migration for Wave 1 (the video engine).

* S1.2a (US-105, D4): ``document_templates.format`` gains ``social_image`` and
  ``social_video``, the compositions the media-render service renders. The
  CHECK ``check_document_template_format`` is dropped (IF EXISTS) and added
  again with the wider list (``core/models/core.py`` DOCUMENT_TEMPLATE_FORMATS
  declares the same list; ``tests/test_prd251w1_social_templates.py`` holds the
  two together).
* D16 (US-110): seeds the Socials media allowlist (category ``socials``, key
  ``media_actions``): per Composio toolkit, the actions Socials may call and
  what for, as a JSON object ``{toolkit: {capability: [action slug, ...]}}``
  (``modules/socials/capabilities.py`` reads it). The allowlist is data a
  super-admin edits; this seed is its first value and its default. An unknown
  toolkit offers nothing until it is added there.
* S1.5 (US-111, D11): ``social_posts.voice``, the voice a render speaks the
  post's script with. NULL is Kokoro, the template's own voice; otherwise
  ``{"toolkit", "voice_id", "name"}`` names a voice toolkit the workspace has
  connected in Composio (``modules/socials/recipes/voice.py``). Nullable JSON
  (JSONB on Postgres), added only when missing.
* S1.8 (US-114, D12): ``social_posts.footage``, the footage and stills a post
  asks its template's slots to be filled with, and what its renders generated
  for them (``modules/socials/recipes/footage.py``). NULL: every slot plays the
  template's own motion graphics. Nullable JSON (JSONB on Postgres), added only
  when missing.
* S3.5 (US-118, D14b): seeds the Socials post gate's list (category
  ``socials``, key ``post_actions``): the Composio actions that publish a post on
  Instagram, X and LinkedIn, as a JSON list of action slugs
  (``core/composio/post_gate.py`` reads it). In a workspace with Socials on, an
  agent's direct call to a listed action is refused before any Composio call;
  it drafts the post, a person approves it in the Socials tab and the platform
  publishes it. Data a super-admin edits, like the Wave 0 deny list.

Create_all-first safe (the 89d89c250 lesson: on the 2026-09-23 refresh a backend
that had already loaded the new models ran ``create_all`` before the migration,
and ``prd251_socials`` crash-looped on DuplicateTable). Here ``create_all`` has
already built the wide CHECK itself, and the upgrade drops it and adds the same
rule again; ``social_posts.voice`` and ``social_posts.footage`` are added only
when the table does not carry them yet. The seed is insert-if-absent: ``system_settings`` has no (category,
key) unique constraint, so the upgrade checks first, and a re-run never
overwrites a super-admin's edit. Running the upgrade twice changes nothing.
Later Wave 1 stories extend THIS revision, so the wave stays one migration, and
every step they add must tolerate what ``create_all`` already built.

The downgrade brings the narrow CHECK back ``NOT VALID``: the social templates a
workspace already holds are kept, and new rows and updates follow the old rule.
It drops ``social_posts.voice`` (each post renders with Kokoro again) and
``social_posts.footage`` (each slot plays the template's own motion graphics
again; the generated files stay in storage and in Deliverables), and it
deletes the settings rows this revision created (``created_by`` = this
revision), edited since or not, and never a row a person created.

Chains single-parent on f049_prd251_merge_heads (the single head of main after
Wave 0 landed through the customer-night PR).

Revision ID: prd251_wave1
Revises: f049_prd251_merge_heads
Create Date: 2026-09-25
"""
from __future__ import annotations

import json
from typing import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "prd251_wave1"
down_revision = "f049_prd251_merge_heads"
branch_labels = None
depends_on = None

TEMPLATES_TABLE = "document_templates"
FORMAT_CHECK_NAME = "check_document_template_format"
FORMATS_BEFORE = ("pdf", "docx", "xlsx")
FORMATS_AFTER = FORMATS_BEFORE + ("social_image", "social_video")

POSTS_TABLE = "social_posts"
POST_VOICE_COLUMN = "voice"
POST_FOOTAGE_COLUMN = "footage"

SEED_CREATED_BY = "prd251_wave1"

# D16: the Composio actions Socials may call, per toolkit and capability. Slugs as
# docs.composio.dev lists them (fal.ai, Kie.ai, Higgsfield MCP and Fish Audio checked
# 2026-09-23; ElevenLabs' "Text to speech" and "Get voices list" 2026-09-25). A slug
# the workspace's cached action schemas do not hold is simply not offered.
SOCIALS_MEDIA_ACTIONS_SEED = {
    "fal_ai": {
        "generate_video": ["FAL_AI_SUBMIT_ASYNC_JOB"],
        "generate_image": ["FAL_AI_SUBMIT_ASYNC_JOB"],
        "status": ["FAL_AI_QUEUE_GET_STATUS", "FAL_AI_GET_QUEUE_REQUEST_RESULT"],
        "upload": ["FAL_AI_UPLOAD_FILE"],
        "estimate": ["FAL_AI_ESTIMATE_PRICING"],
    },
    "kieai": {
        "generate_video": ["KIEAI_GENERATE_VEO_VIDEO"],
        "generate_image": ["KIEAI_GENERATE_FLUX_KONTEXT_IMAGE"],
        "status": ["KIEAI_GET_VEO_VIDEO_DETAILS", "KIEAI_GET_FLUX_KONTEXT_IMAGE_DETAILS"],
        "balance": ["KIEAI_GET_ACCOUNT_CREDITS"],
    },
    "higgsfield_mcp": {
        "generate_video": ["HIGGSFIELD_MCP_GENERATE_VIDEO"],
        "generate_image": ["HIGGSFIELD_MCP_GENERATE_IMAGE"],
        "status": ["HIGGSFIELD_MCP_JOBS_WAIT", "HIGGSFIELD_MCP_JOB_STATUS"],
        "upload": ["HIGGSFIELD_MCP_MEDIA_UPLOAD"],
        "balance": ["HIGGSFIELD_MCP_BALANCE"],
    },
    "fish_audio": {
        "tts": ["FISH_AUDIO_SYNTHESIZE_SPEECH"],
        "voices": ["FISH_AUDIO_LIST_VOICE_MODELS"],
        "balance": ["FISH_AUDIO_GET_ACCOUNT_BALANCE"],
    },
    "elevenlabs": {
        "tts": ["ELEVENLABS_TEXT_TO_SPEECH"],
        "voices": ["ELEVENLABS_GET_VOICES_LIST"],
    },
}

# D14b: the Composio actions that PUBLISH a post on a social channel. The publish
# actions the old publisher skills call (automatos-skills social/instagram-curator,
# twitter-engager, linkedin-content-creator); the other actions docs.composio.dev
# lists as publishing a post on those three channels (checked 2026-09-26); and the
# LinkedIn posting slugs the executor names. Media uploads, containers, comments,
# replies, reposts, DMs and reads are not listed. TikTok and YouTube publish actions
# join when composio_actions_cache confirms their slugs (the story's notes: never
# guess). A slug no workspace's cached action schemas hold matches nothing.
SOCIALS_POST_ACTIONS_SEED = [
    "INSTAGRAM_POST_IG_USER_MEDIA_PUBLISH",
    "INSTAGRAM_CREATE_POST",
    "TWITTER_CREATION_OF_A_POST",
    "LINKEDIN_CREATE_LINKED_IN_POST",
    "LINKEDIN_CREATE_POST",
    "LINKEDIN_CREATE_ARTICLE_OR_URL_SHARE",
    "LINKEDIN_CREATE_VIDEO_POST",
    "LINKEDIN_CREATE_IMAGE_POST",
    "LINKEDIN_CREATE_SHARE",
]


def format_check(formats: Sequence[str]) -> str:
    """The CHECK's SQL, written the way the model writes it."""
    return "format IN (" + ", ".join(f"'{fmt}'" for fmt in formats) + ")"


def _replace_format_check(formats: Sequence[str], *, validate: bool) -> None:
    op.execute(f"ALTER TABLE {TEMPLATES_TABLE} DROP CONSTRAINT IF EXISTS {FORMAT_CHECK_NAME}")
    op.execute(
        f"ALTER TABLE {TEMPLATES_TABLE} ADD CONSTRAINT {FORMAT_CHECK_NAME} "
        f"CHECK ({format_check(formats)}){'' if validate else ' NOT VALID'}"
    )


def settings_seed() -> tuple:
    """The system-settings rows this revision seeds."""
    media_actions = json.dumps(SOCIALS_MEDIA_ACTIONS_SEED)
    post_actions = json.dumps(SOCIALS_POST_ACTIONS_SEED)
    return (
        {
            "category": "socials",
            "key": "media_actions",
            "value": media_actions,
            "value_type": "json",
            "description": (
                "Socials media allowlist (PRD-251 D16): per Composio toolkit, the actions "
                "Socials may call and what for — a JSON object {toolkit: {capability: "
                "[action slug, ...]}}, capabilities generate_video, generate_image, tts, "
                "voices, estimate, status, upload, balance. A connected toolkit offers only "
                "the actions listed here and held in its cached action schemas; a toolkit "
                "not listed offers nothing, and the Composio deny list always wins. Takes "
                "effect on the next read — no restart or redeploy."
            ),
            "is_sensitive": False,
            "is_required": True,
            "default_value": media_actions,
        },
        {
            "category": "socials",
            "key": "post_actions",
            "value": post_actions,
            "value_type": "json",
            "description": (
                "Socials post gate (PRD-251 S3.5, D14b): the Composio actions that publish "
                "a post on a social channel, as a JSON list of action slugs. In a workspace "
                "with Socials on, an agent's direct call to a listed action is refused before "
                "any Composio call: it drafts the post with platform_create_social_post, a "
                "person approves it in the Socials tab and the platform publishes it. Media "
                "uploads, containers, comments, replies, reposts, DMs and reads are not "
                "listed; an upload action that publishes by itself belongs here. A value "
                "that is not a JSON list refuses every Composio call in Socials-on "
                "workspaces until it is fixed, and the Composio deny list always wins. "
                "Takes effect within the cache TTL (30 s by default) — no restart or "
                "redeploy."
            ),
            "is_sensitive": False,
            "is_required": True,
            "default_value": post_actions,
        },
    )


def seed_settings(conn, rows) -> None:
    """Insert each row unless its (category, key) is already there."""
    for row in rows:
        exists = conn.execute(
            sa.text("SELECT 1 FROM system_settings WHERE category = :category AND key = :key"),
            {"category": row["category"], "key": row["key"]},
        ).first()
        if exists:
            continue
        conn.execute(
            sa.text(
                "INSERT INTO system_settings "
                "(category, key, value, value_type, description, is_sensitive, "
                " is_required, default_value, created_by) "
                "VALUES (:category, :key, :value, :value_type, :description, "
                "        :is_sensitive, :is_required, :default_value, :created_by)"
            ),
            {**row, "created_by": SEED_CREATED_BY},
        )


def unseed_settings(conn, rows) -> None:
    """Delete the rows this revision seeded (never one a person created)."""
    for row in rows:
        conn.execute(
            sa.text(
                "DELETE FROM system_settings "
                "WHERE category = :category AND key = :key AND created_by = :created_by"
            ),
            {"category": row["category"], "key": row["key"], "created_by": SEED_CREATED_BY},
        )


def _post_columns() -> Sequence[str]:
    """The columns ``social_posts`` carries; none when the table is not there."""
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table(POSTS_TABLE):
        return ()
    return [column["name"] for column in inspector.get_columns(POSTS_TABLE)]


def add_post_voice_column() -> None:
    """US-111 (D11): ``social_posts.voice``, unless ``create_all`` already built it.

    ``prd251_socials`` builds the table before this revision runs; a schema
    without it (a partial test schema) has no table to add the column to."""
    columns = _post_columns()
    if not columns or POST_VOICE_COLUMN in columns:
        return
    op.add_column(
        "social_posts",
        sa.Column(POST_VOICE_COLUMN, sa.JSON().with_variant(JSONB(), "postgresql"), nullable=True),
    )


def drop_post_voice_column() -> None:
    if POST_VOICE_COLUMN in _post_columns():
        op.drop_column("social_posts", POST_VOICE_COLUMN)


def add_post_footage_column() -> None:
    """US-114 (D12): ``social_posts.footage``, unless ``create_all`` already built it."""
    columns = _post_columns()
    if not columns or POST_FOOTAGE_COLUMN in columns:
        return
    op.add_column(
        "social_posts",
        sa.Column(POST_FOOTAGE_COLUMN, sa.JSON().with_variant(JSONB(), "postgresql"), nullable=True),
    )


def drop_post_footage_column() -> None:
    if POST_FOOTAGE_COLUMN in _post_columns():
        op.drop_column("social_posts", POST_FOOTAGE_COLUMN)


def upgrade() -> None:
    _replace_format_check(FORMATS_AFTER, validate=True)
    add_post_voice_column()
    add_post_footage_column()
    seed_settings(op.get_bind(), settings_seed())


def downgrade() -> None:
    unseed_settings(op.get_bind(), settings_seed())
    drop_post_footage_column()
    drop_post_voice_column()
    _replace_format_check(FORMATS_BEFORE, validate=False)
