"""Auto's soul has one home: its agent row (7 Nov 2026). THIS CHANGES EXISTING DATA.

Until this release Auto's soul lived in two places. ``workspaces.settings.orchestrator.custom_soul``
was what the chat spoke with (``consumers/chatbot/personality.py``, in ``custom`` mode), and the
Auto agent row's ``custom_persona_prompt`` was what Settings → Orchestrator showed and edited,
appended to the chat's prompt a second time. A Settings save wrote both; anything else that
wrote the row (Auto's own ``platform_update_agent``, the persona route) wrote only the row, so
the two drifted. On Automatos Cloud one workspace spoke with an 8,553-character soul while
Settings showed a different 2,187-character one.

From this release the row is the only copy: the chat reads it and nothing writes the setting.
This migration moves what users have been hearing onto the row, once:

1. For each workspace whose ``settings.orchestrator`` holds ``personality_mode = 'custom'`` and a
   non-empty ``custom_soul`` that its Auto row (slug ``auto-<workspace_id>``, a system agent)
   does not already speak with (the same text, stripped, with ``use_custom_persona`` true), the
   row gets that soul, ``use_custom_persona = true`` and ``configuration.personality_mode =
   'custom'``, so Settings shows the soul the chat used. Each copy is logged with the row's id
   and both lengths (never the text).
2. ``custom_soul`` is then removed from the orchestrator settings of every workspace that has an
   Auto row, whatever its mode: nothing reads it any more.

A workspace with no Auto row is left alone, setting included: there is no row to copy to, and
removing the setting would delete the only copy of its soul. It is logged.

Idempotent: once the key is gone a second run finds no workspace and writes nothing. No DDL: a
database ``create_all`` built has both tables, and a database without them is left as it is.
The rows are locked (``FOR UPDATE``) until the migration's transaction ends.

The downgrade does nothing: the soul stays on the row, which is where it belongs, and restoring
the second copy would bring the drift back.

Revision ID: auto_soul_one_source_20261107
Revises: prd270_w2_cloud_bundles
Create Date: 2026-11-07
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Mapping, Optional

import sqlalchemy as sa
from alembic import op

revision = "auto_soul_one_source_20261107"
down_revision = "prd270_w2_cloud_bundles"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.runtime.migration")

ORCHESTRATOR_KEY = "orchestrator"
SOUL_KEY = "custom_soul"
MODE_KEY = "personality_mode"
CUSTOM_MODE = "custom"
AUTO_SLUG = "auto-{workspace_id}"
TAG = "[auto_soul_one_source]"

_WORKSPACES_WITH_A_SOUL = sa.text(
    "SELECT id, settings FROM workspaces "
    "WHERE (settings::jsonb -> 'orchestrator' -> 'custom_soul') IS NOT NULL ORDER BY id FOR UPDATE"
)
_AUTO_ROW = sa.text(
    "SELECT id, custom_persona_prompt, use_custom_persona, configuration FROM agents "
    "WHERE slug = :slug AND is_system_agent IS TRUE AND workspace_id = CAST(:ws AS uuid) FOR UPDATE"
)


def _decoded(raw: Any) -> Any:
    """A JSON column's value as Python: already decoded, or a string to load (None if it is not JSON)."""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except ValueError:
            return None
    return raw


def soul_to_copy(orchestrator: Any) -> Optional[str]:
    """The settings soul the chat spoke with: in ``custom`` mode, a non-empty ``custom_soul``. Pure."""
    if not isinstance(orchestrator, Mapping) or orchestrator.get(MODE_KEY) != CUSTOM_MODE:
        return None
    soul = orchestrator.get(SOUL_KEY)
    return soul if isinstance(soul, str) and soul.strip() else None


def row_speaks_with(soul: str, prompt: Optional[str], use_custom_persona: Optional[bool]) -> bool:
    """True when the row already carries ``soul`` (stripped) and uses it. Pure."""
    return use_custom_persona is True and (prompt or "").strip() == soul.strip()


def without_soul(settings: Any) -> Optional[Dict[str, Any]]:
    """NEW settings without ``orchestrator.custom_soul``; None when there is none to remove. Pure."""
    if not isinstance(settings, Mapping):
        return None
    orchestrator = settings.get(ORCHESTRATOR_KEY)
    if not isinstance(orchestrator, Mapping) or SOUL_KEY not in orchestrator:
        return None
    kept = {key: value for key, value in orchestrator.items() if key != SOUL_KEY}
    return {**settings, ORCHESTRATOR_KEY: kept}


def with_custom_mode(configuration: Any) -> Dict[str, Any]:
    """NEW agent configuration whose personality_mode is ``custom``, every other key kept. Pure."""
    kept = dict(configuration) if isinstance(configuration, Mapping) else {}
    return {**kept, MODE_KEY: CUSTOM_MODE}


def _has_tables(bind) -> bool:
    inspector = sa.inspect(bind)
    return inspector.has_table("workspaces") and inspector.has_table("agents")


def _json_type(bind, table: str, column: str) -> str:
    """The column's SQL type: jsonb, or json (the models declare one, older DDL the other)."""
    row = bind.execute(sa.text(
        "SELECT data_type FROM information_schema.columns "
        "WHERE table_schema = current_schema() AND table_name = :table AND column_name = :column"
    ), {"table": table, "column": column}).first()
    return "jsonb" if row and row[0] == "jsonb" else "json"


def _copy_soul(bind, workspace_id: str, row: Any, soul: str) -> None:
    """Step 1 for one workspace: the settings soul onto its Auto row, in custom mode."""
    config_type = _json_type(bind, "agents", "configuration")
    bind.execute(sa.text(
        "UPDATE agents SET custom_persona_prompt = :soul, use_custom_persona = TRUE, "
        f"configuration = CAST(:configuration AS {config_type}) WHERE id = :id"
    ), {"soul": soul, "configuration": json.dumps(with_custom_mode(_decoded(row.configuration))), "id": row.id})
    logger.info("%s workspace %s: Auto agent %s soul %d -> %d chars, from settings", TAG, workspace_id,
                row.id, len((row.custom_persona_prompt or "").strip()), len(soul.strip()))


def _move_one(bind, workspace_id: str, settings: Any, settings_type: str) -> str:
    """One workspace: copy its soul when the row differs, then drop the setting. Returns what it did."""
    row = bind.execute(_AUTO_ROW, {"slug": AUTO_SLUG.format(workspace_id=workspace_id), "ws": workspace_id}).first()
    if row is None:
        logger.warning("%s workspace %s has no Auto row: its custom_soul setting is left as it is",
                       TAG, workspace_id)
        return "no_auto_row"
    soul = soul_to_copy(settings.get(ORCHESTRATOR_KEY) if isinstance(settings, Mapping) else None)
    copied = soul is not None and not row_speaks_with(soul, row.custom_persona_prompt, row.use_custom_persona)
    if copied:
        _copy_soul(bind, workspace_id, row, soul)
    cleaned = without_soul(settings)
    if cleaned is not None:
        bind.execute(sa.text(
            f"UPDATE workspaces SET settings = CAST(:settings AS {settings_type}) WHERE id = CAST(:id AS uuid)"
        ), {"settings": json.dumps(cleaned), "id": workspace_id})
    return "copied" if copied else "removed"


def upgrade() -> None:
    bind = op.get_bind()
    if not _has_tables(bind):
        logger.info("%s no workspaces or agents table: nothing to move", TAG)
        return
    settings_type = _json_type(bind, "workspaces", "settings")
    counts: Dict[str, int] = {"copied": 0, "removed": 0, "no_auto_row": 0}
    for workspace_id, raw in bind.execute(_WORKSPACES_WITH_A_SOUL).fetchall():
        counts[_move_one(bind, str(workspace_id), _decoded(raw), settings_type)] += 1
    logger.info("%s souls copied onto the Auto row: %d; setting removed only: %d; no Auto row (left): %d",
                TAG, counts["copied"], counts["removed"], counts["no_auto_row"])


def downgrade() -> None:
    """No-op: the soul stays on the row; a second copy would bring the drift back (docstring)."""
