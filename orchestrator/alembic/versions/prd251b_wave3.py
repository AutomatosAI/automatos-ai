"""PRD-251B Socials Studio — the ONE migration for Wave 3 (the brand kit's style references).

US-B305: a brand kit's liked style references go to an AI tool only when its generate
action takes a reference image. That is a flag in the Socials media allowlist (system
setting ``socials.media_actions``, PRD-251 D16): the capability ``reference_image``. This
revision lists fal.ai's queue submit under it, in the row's value and its default, because
fal's still model (FLUX1.1 [pro] ultra) takes a reference as its image prompt. Kie.ai's
Flux Kontext takes one too, but as the image to EDIT, so it is left for a super-admin to
list. A row a person reshaped keeps its shape: a toolkit missing from it, or a value that
is not a JSON object, is left alone, and a toolkit already flagging the slug is unchanged,
so running it twice changes nothing. No row (a database not yet seeded) is nothing to do.
The style references themselves live in ``workspace.settings['brand_style']``: no schema.

Chains single-parent on prd251b_wave2.

Revision ID: prd251b_wave3
Revises: prd251b_wave2
Create Date: 2026-10-03
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

import sqlalchemy as sa
from alembic import op

revision = "prd251b_wave3"
down_revision = "prd251b_wave2"
branch_labels = None
depends_on = None

CAPABILITY = "reference_image"
# toolkit → the generate actions that take a reference image.
REFERENCE_SEED = {"fal_ai": ["FAL_AI_SUBMIT_ASYNC_JOB"]}
COLUMNS = ("value", "default_value")
OLD_WORDS = "upload, balance."
NEW_WORDS = "upload, balance, reference_image (a generate action that takes a reference image)."


def _parsed(raw: Any) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(raw) if isinstance(raw, str) and raw.strip() else None
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def flagged(raw: Any) -> Any:
    """``raw`` with each seeded slug listed under the capability, where its toolkit is listed."""
    value = _parsed(raw)
    if value is None:
        return raw
    changed = dict(value)
    for toolkit, slugs in REFERENCE_SEED.items():
        listed = changed.get(toolkit)
        if not isinstance(listed, dict):
            continue
        present = listed.get(CAPABILITY) if isinstance(listed.get(CAPABILITY), list) else []
        changed[toolkit] = {**listed, CAPABILITY: [*present, *(slug for slug in slugs if slug not in present)]}
    return json.dumps(changed)


def unflagged(raw: Any) -> Any:
    """``raw`` without the seeded slugs under the capability (the capability gone when empty)."""
    value = _parsed(raw)
    if value is None:
        return raw
    changed = dict(value)
    for toolkit, slugs in REFERENCE_SEED.items():
        listed = changed.get(toolkit)
        if not isinstance(listed, dict) or not isinstance(listed.get(CAPABILITY), list):
            continue
        kept = [slug for slug in listed[CAPABILITY] if slug not in slugs]
        rest = {key: item for key, item in listed.items() if key != CAPABILITY}
        changed[toolkit] = {**rest, CAPABILITY: kept} if kept else rest
    return json.dumps(changed)


def _rewrite(conn: Any, change: Any, words: tuple) -> None:
    row = conn.execute(
        sa.text("SELECT value, default_value, description FROM system_settings WHERE category = 'socials' AND key = 'media_actions'")
    ).first()
    if row is None:
        return
    description = row.description.replace(*words) if isinstance(row.description, str) else row.description
    conn.execute(
        sa.text(
            "UPDATE system_settings SET value = :value, default_value = :default_value, description = :description "
            "WHERE category = 'socials' AND key = 'media_actions'"
        ),
        {"value": change(row.value), "default_value": change(row.default_value), "description": description},
    )


def upgrade() -> None:
    _rewrite(op.get_bind(), flagged, (OLD_WORDS, NEW_WORDS))


def downgrade() -> None:
    _rewrite(op.get_bind(), unflagged, (NEW_WORDS, OLD_WORDS))
