"""PRD-251 Socials — the ONE migration for Wave 1 (the video engine).

* S1.2a (US-105, D4): ``document_templates.format`` gains ``social_image`` and
  ``social_video``, the compositions the media-render service renders. The
  CHECK ``check_document_template_format`` is dropped (IF EXISTS) and added
  again with the wider list (``core/models/core.py`` DOCUMENT_TEMPLATE_FORMATS
  declares the same list; ``tests/test_prd251w1_social_templates.py`` holds the
  two together).

Create_all-first safe (the 89d89c250 lesson: on the 2026-09-23 refresh a backend
that had already loaded the new models ran ``create_all`` before the migration,
and ``prd251_socials`` crash-looped on DuplicateTable). Here ``create_all`` has
already built the wide CHECK itself, and the upgrade drops it and adds the same
rule again. Running the upgrade twice changes nothing. Later Wave 1 stories
extend THIS revision, so the wave stays one migration, and every step they add
must tolerate what ``create_all`` already built.

The downgrade brings the narrow CHECK back ``NOT VALID``: the social templates a
workspace already holds are kept, and new rows and updates follow the old rule.
Nothing is deleted.

Chains single-parent on f049_prd251_merge_heads (the single head of main after
Wave 0 landed through the customer-night PR).

Revision ID: prd251_wave1
Revises: f049_prd251_merge_heads
Create Date: 2026-09-25
"""
from __future__ import annotations

from typing import Sequence

from alembic import op

revision = "prd251_wave1"
down_revision = "f049_prd251_merge_heads"
branch_labels = None
depends_on = None

TEMPLATES_TABLE = "document_templates"
FORMAT_CHECK_NAME = "check_document_template_format"
FORMATS_BEFORE = ("pdf", "docx", "xlsx")
FORMATS_AFTER = FORMATS_BEFORE + ("social_image", "social_video")


def format_check(formats: Sequence[str]) -> str:
    """The CHECK's SQL, written the way the model writes it."""
    return "format IN (" + ", ".join(f"'{fmt}'" for fmt in formats) + ")"


def _replace_format_check(formats: Sequence[str], *, validate: bool) -> None:
    op.execute(f"ALTER TABLE {TEMPLATES_TABLE} DROP CONSTRAINT IF EXISTS {FORMAT_CHECK_NAME}")
    op.execute(
        f"ALTER TABLE {TEMPLATES_TABLE} ADD CONSTRAINT {FORMAT_CHECK_NAME} "
        f"CHECK ({format_check(formats)}){'' if validate else ' NOT VALID'}"
    )


def upgrade() -> None:
    _replace_format_check(FORMATS_AFTER, validate=True)


def downgrade() -> None:
    _replace_format_check(FORMATS_BEFORE, validate=False)
