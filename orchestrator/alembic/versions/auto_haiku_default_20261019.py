"""Auto's platform default becomes Haiku 5.5 on deploy, with no manual step (19 Oct 2026).

Gerard, 19 Oct 2026: Auto agents default to ``anthropic`` / ``claude-haiku-5-5``, and
the change happens when the release deploys, not by an operator running a script.
This data migration does it once:

1. The platform default (system_settings ``orchestrator_llm`` ``provider`` and
   ``model``, read by ``core/llm/new_agent_model.default_route``) is set to that pair.
   On the hosted edition (``config.AUTH_EDITION`` saas, which an unset or unknown
   ``AUTH_EDITION`` reads as; alembic/env.py loads the same .env) it is always set: production held ``openrouter`` /
   ``openai/gpt-5.5``, written there by one tenant's Settings > Orchestrator save.
   On the local edition it is set only when the stored pair is empty or still the
   old shipped one (``openrouter`` / ``google/gemini-2.5-flash``), so a self-hoster's
   own choice survives the upgrade. A missing row is inserted; the legacy
   ``llm_provider`` / ``llm_model`` rows, which ``default_route`` reads when the new
   keys are empty, follow when they exist. On every edition the rows'
   ``default_value``, which Reset to defaults restores, becomes the target too, as
   ``seed_system_settings`` writes it on boot, so the migration does not depend on
   the seed step running after it.
2. Every workspace's Auto row still on the platform default moves to the value now
   stored (read back, so a self-hoster's kept choice is what their seed rows move
   to), marked ``source=platform_default`` with its ``platform_route``. The rules are
   ``core/llm/auto_model_sync.is_inheriting``'s, copied here so the migration does not
   change with that module (it imports only ``config``, as f125 and prd251 do): a row with no marker holding exactly a shipped seed pair, or
   a marked row still on the route it was stamped with. Every other model_config key
   is kept. A row a workspace chose, and every non-Auto agent, is not touched.

Idempotent: on a database already at the target, nothing is written. The Auto rows
are locked (``FOR UPDATE``) until the migration's transaction ends, so a workspace's
save on an old container meanwhile waits and is not overwritten. Each change is
logged with its value before.

The downgrade does nothing. The values before are in the deploy log line by line;
they were a stale seed pair and one tenant's save leaking into every workspace, so
restoring them would put the bug back. A snapshot table would be a schema object no
model declares, for data nobody should restore.

Revision ID: auto_haiku_default_20261019
Revises: prd263_harness_runs
Create Date: 2026-10-19
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any, Dict, Mapping, Optional, Tuple

import sqlalchemy as sa
from alembic import op

revision = "auto_haiku_default_20261019"
down_revision = "prd263_harness_runs"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.runtime.migration")

Route = Tuple[Optional[str], Optional[str]]  # (provider, model_id)

# The decision of 19 Oct 2026, as a dated record: later changes are an admin's, in
# System Settings, not a new literal here.
TARGET: Route = ("anthropic", "claude-haiku-5-5")
# The pair seed_system_settings gave orchestrator_llm before 19 Oct 2026.
OLD_SHIPPED_DEFAULT: Route = ("openrouter", "google/gemini-2.5-flash")
# auto_model_sync.KNOWN_SEED_ROUTES: the pairs shipped seeds wrote into Auto rows.
KNOWN_SEED_ROUTES = frozenset({("openrouter", "google/gemini-2.5-flash"), ("openrouter", "openai/gpt-4o")})

CATEGORY = "orchestrator_llm"
# setting key -> the legacy key default_route falls back to.
ROUTE_KEYS = (("provider", "llm_provider"), ("model", "llm_model"))
SOURCE_KEY = "source"
PLATFORM_DEFAULT_SOURCE = "platform_default"
STAMPED_ROUTE_KEY = "platform_route"
CREATED_BY = "auto_haiku_default_20261019"
DESCRIPTIONS = {
    "provider": "Which LLM provider serves this tier.",
    "model": "Which model handles requests for this tier.",
}


def platform_default_to_write(auth_edition: str, stored: Route) -> Optional[Route]:
    """The pair to store as the platform default, or None to leave it. Pure."""
    if tuple(stored) == TARGET:
        return None
    if auth_edition == "local" and all(stored) and tuple(stored) != OLD_SHIPPED_DEFAULT:
        return None  # a self-hoster's own choice
    return TARGET


def _route_of(model_config: Mapping[str, Any]) -> Route:
    return model_config.get("provider"), model_config.get("model_id")


def is_inheriting(model_config: Optional[Mapping[str, Any]]) -> bool:
    """auto_model_sync.is_inheriting: is this Auto row still on the platform default?"""
    if not model_config:
        return False
    route = _route_of(model_config)
    if SOURCE_KEY in model_config:
        stamped = model_config.get(STAMPED_ROUTE_KEY)
        return (model_config[SOURCE_KEY] == PLATFORM_DEFAULT_SOURCE
                and isinstance(stamped, Mapping) and _route_of(stamped) == route)
    return route in KNOWN_SEED_ROUTES


def restamped(model_config: Any, target: Route) -> Optional[Dict[str, Any]]:
    """The NEW model_config for an inheriting Auto row on ``target``, every other key
    kept; None when the row was chosen or is already there. Pure."""
    if not isinstance(model_config, Mapping) or not is_inheriting(model_config):
        return None
    provider, model_id = target
    moved = {
        **model_config, "provider": provider, "model_id": model_id,
        SOURCE_KEY: PLATFORM_DEFAULT_SOURCE, STAMPED_ROUTE_KEY: {"provider": provider, "model_id": model_id},
    }
    return None if moved == dict(model_config) else moved


def _has_tables(bind) -> bool:
    inspector = sa.inspect(bind)
    return inspector.has_table("system_settings") and inspector.has_table("agents")


def _setting(bind, key: str) -> Optional[str]:
    row = bind.execute(
        sa.text("SELECT value FROM system_settings WHERE category = :category AND key = :key ORDER BY id LIMIT 1"),
        {"category": CATEGORY, "key": key},
    ).first()
    return row[0] if row else None


def stored_route(bind) -> Route:
    """The platform default as default_route reads it: each key, else its legacy key."""
    provider, model = ((_setting(bind, key) or _setting(bind, legacy)) for key, legacy in ROUTE_KEYS)
    return provider, model


def _set_shipped_default(bind) -> None:
    """Each route row's default_value (what Reset to defaults restores) is TARGET:
    read first, so a database already there sees no write."""
    select = sa.text("SELECT id, default_value FROM system_settings WHERE category = :category AND key = :key")
    update = sa.text("UPDATE system_settings SET default_value = :value WHERE id = :id")
    for (key, _legacy), value in zip(ROUTE_KEYS, TARGET):
        for row_id, default_value in bind.execute(select, {"category": CATEGORY, "key": key}).fetchall():
            if default_value != value:
                bind.execute(update, {"value": value, "id": row_id})
                logger.info("[auto_haiku_default] %s default_value %s -> %s", key, default_value, value)


def _write_setting(bind, key: str, legacy: str, value: str) -> None:
    params = {"category": CATEGORY, "key": key, "value": value}
    updated = bind.execute(sa.text(
        "UPDATE system_settings SET value = :value, updated_at = now() WHERE category = :category AND key = :key"
    ), params).rowcount
    if not updated:
        bind.execute(sa.text(
            "INSERT INTO system_settings (category, key, value, value_type, description, is_sensitive, "
            " is_required, default_value, created_by) "
            "VALUES (:category, :key, :value, 'string', :description, false, true, :value, :created_by)"
        ), {**params, "description": DESCRIPTIONS[key], "created_by": CREATED_BY})
    bind.execute(sa.text(
        "UPDATE system_settings SET value = :value, updated_at = now() WHERE category = :category AND key = :key"
    ), {**params, "key": legacy})


def _set_platform_default(bind) -> Route:
    """Step 1; the platform default as stored afterwards."""
    from config import config

    before = stored_route(bind)
    auth_edition = config.AUTH_EDITION
    write = platform_default_to_write(auth_edition, before)
    if write:
        for (key, legacy), value in zip(ROUTE_KEYS, write):
            _write_setting(bind, key, legacy, value)
        logger.info("[auto_haiku_default] %s platform default %s/%s -> %s/%s", auth_edition, *before, *write)
    else:
        logger.info("[auto_haiku_default] %s platform default %s/%s kept", auth_edition, *before)
    _set_shipped_default(bind)
    return stored_route(bind)


def _model_config_type(bind) -> str:
    """The SQL type of agents.model_config (json from the models, jsonb in older DDL)."""
    row = bind.execute(sa.text(
        "SELECT data_type FROM information_schema.columns "
        "WHERE table_schema = current_schema() AND table_name = 'agents' AND column_name = 'model_config'"
    )).first()
    return "jsonb" if row and row[0] == "jsonb" else "json"


def _decoded(model_config: Any) -> Any:
    if isinstance(model_config, str):
        try:
            return json.loads(model_config)
        except ValueError:
            return None
    return model_config


def _restamp_auto_rows(bind, target: Route) -> None:
    """Step 2: every inheriting Auto row onto ``target``; chosen rows stay."""
    rows = bind.execute(sa.text(
        "SELECT id, model_config FROM agents WHERE is_system_agent IS TRUE AND slug LIKE 'auto-%' "
        "AND workspace_id IS NOT NULL ORDER BY id FOR UPDATE"
    )).fetchall()
    column_type = _model_config_type(bind)
    update = sa.text(f"UPDATE agents SET model_config = CAST(:model_config AS {column_type}) WHERE id = :id")
    moved: Counter = Counter()
    for agent_id, raw in rows:
        model_config = _decoded(raw)
        new_config = restamped(model_config, target)
        if new_config is None:
            continue
        bind.execute(update, {"model_config": json.dumps(new_config), "id": agent_id})
        before = "/".join(str(part) for part in _route_of(model_config))
        moved[before] += 1
        logger.info("[auto_haiku_default] Auto agent %s: %s -> %s/%s", agent_id, before, *target)
    logger.info("[auto_haiku_default] Auto rows: %d of %d moved to %s/%s (%s); the rest chosen or current",
                sum(moved.values()), len(rows), *target, dict(moved))


def upgrade() -> None:
    bind = op.get_bind()
    if not _has_tables(bind):
        logger.info("[auto_haiku_default] no system_settings or agents table: nothing to move")
        return
    target = _set_platform_default(bind)
    if not all(target):
        logger.warning("[auto_haiku_default] platform default unreadable after the write: Auto rows left")
        return
    _restamp_auto_rows(bind, target)


def downgrade() -> None:
    """No-op: the values before are in the upgrade's log, and were the bug (docstring)."""
