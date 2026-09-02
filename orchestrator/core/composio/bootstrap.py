"""PRD-233 S2 — Composio catalogue bootstrap + seeded-agent rebind.

Nothing used to sync the Composio catalogue on boot: the only triggers were
``POST /api/tools/sync`` (Tools page **Sync**) and two manual scripts. On a
keyless first boot the two marketplace seeders (``scripts/seed_starter_agents.py``,
``scripts/seed_marketplace_agents_v2.py``) look each agent's apps up in
``composio_apps_cache``, find nothing ("Tool 'SLACK' not found"), and write
``metadata.tools = []`` / ``metadata.tool_icons = []`` — keeping only the
intended names in ``metadata.tool_names``. Their by-name idempotency never
re-binds later, so adding a key afterwards left every seeded agent unbound.

This module closes both gaps from ONE startup stage (``main.py`` →
``BootstrapStage.TOOL_SYNC``):

* key + SDK, **empty** catalogue     → full sync in a background thread, then rebind
* key + SDK, **populated** catalogue → rebind only (a no-op once every seed is bound)
* no key / no SDK                    → log the reason, touch nothing

Never blocks or fails boot: the sync runs off the event loop, and every
failure is logged + recorded via ``record_error(subsystem="startup")`` like the
other boot seeds (``core/boot/startup_tasks.py``).
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import func, text

from core.composio.client import composio_available, composio_unavailable_reason
from core.database.database import SessionLocal
from core.models.composio_cache import ComposioAppCache, ComposioSyncJob
from core.utils.exception_telemetry import record_error

logger = logging.getLogger(__name__)

# Both seeders stamp their marketplace rows with this creator; user-published
# items carry the publisher's name, so this is the seeded-set discriminator.
SEEDED_CREATOR_NAME = "Automatos Team"
SEEDED_ITEM_TYPE = "agent"
# A 'running' composio_sync_jobs row younger than this means a sync is already
# in flight (another worker, or the Tools page) — don't start a second one.
# Kept short on purpose: a full sync takes a few minutes, and a dev hot-reload
# that kills the background thread leaves its job row 'running' forever, so a
# long window would silently postpone the next boot's sync.
RUNNING_SYNC_FRESH_MINUTES = 10
BACKGROUND_THREAD_NAME = "composio-catalog-bootstrap"

# {APP_NAME: (cache_id, logo_url)} — the seeder lookup, materialised once.
Catalog = Dict[str, Tuple[int, Optional[str]]]


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without a DB)
# ---------------------------------------------------------------------------


def resolve_seeded_bindings(tool_names: Sequence[Any], catalog: Catalog) -> Tuple[List[int], List[str]]:
    """Re-run the seeders' per-name lookup against ``catalog``.

    Mirrors ``scripts/seed_starter_agents.py`` exactly: an app that is not in
    the cache is skipped (no id, no icon); a hit contributes its id and its
    logo URL (``""`` when the cache has none).
    """
    ids: List[int] = []
    icons: List[str] = []
    for name in tool_names:
        hit = catalog.get(str(name).upper())
        if hit is None:
            continue
        ids.append(hit[0])
        icons.append(hit[1] or "")
    return ids, icons


def rebound_metadata(metadata: Dict[str, Any], catalog: Catalog) -> Optional[Dict[str, Any]]:
    """New metadata dict with re-resolved bindings, or ``None`` when unchanged.

    Never mutates ``metadata``; the caller writes only when a dict comes back,
    which is what makes the rebind idempotent.
    """
    names = metadata.get("tool_names") or []
    ids, icons = resolve_seeded_bindings(names, catalog)
    if ids == (metadata.get("tools") or []) and icons == (metadata.get("tool_icons") or []):
        return None
    return {**metadata, "tools": ids, "tool_icons": icons}


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def catalog_app_count(db) -> int:
    """Number of apps in ``composio_apps_cache`` (0 ⇒ never synced)."""
    return int(db.query(func.count(ComposioAppCache.id)).scalar() or 0)


def _sync_in_flight(db) -> bool:
    cutoff = datetime.utcnow() - timedelta(minutes=RUNNING_SYNC_FRESH_MINUTES)
    row = (
        db.query(ComposioSyncJob.id)
        .filter(ComposioSyncJob.status == "running", ComposioSyncJob.started_at >= cutoff)
        .first()
    )
    return row is not None


def _load_catalog(db, names: Sequence[str]) -> Catalog:
    if not names:
        return {}
    rows = (
        db.query(ComposioAppCache.id, ComposioAppCache.app_name, ComposioAppCache.logo_url)
        .filter(func.upper(ComposioAppCache.app_name).in_(list(names)))
        .all()
    )
    return {str(app_name).upper(): (int(app_id), logo_url) for app_id, app_name, logo_url in rows}


def _seeded_rows_needing_bindings(db):
    """Seeded rows whose bound ids lag their intended names.

    On SaaS (every seed bound at seed time) this returns nothing, so the boot
    rebind is a single indexed-free SELECT and no write.
    """
    return db.execute(
        text(
            """
            SELECT id, metadata FROM marketplace_items
            WHERE type = :item_type AND creator_name = :creator
              AND jsonb_array_length(COALESCE(metadata->'tools', CAST('[]' AS jsonb)))
                  < jsonb_array_length(COALESCE(metadata->'tool_names', CAST('[]' AS jsonb)))
            ORDER BY id
            """
        ),
        {"item_type": SEEDED_ITEM_TYPE, "creator": SEEDED_CREATOR_NAME},
    ).fetchall()


def rebind_seeded_agents(db) -> int:
    """Re-resolve the seeded marketplace agents' app bindings from the cache.

    Touches ONLY ``marketplace_items`` rows of type ``agent`` created by the
    seeders (``creator_name = 'Automatos Team'``) — never the ``agents`` table,
    never user-published items. Idempotent: a row is written only when its
    recomputed bindings differ. Returns the number of rows updated.
    """
    rows = _seeded_rows_needing_bindings(db)
    if not rows:
        return 0
    wanted = sorted(
        {
            str(name).upper()
            for _item_id, metadata in rows
            for name in ((metadata or {}).get("tool_names") or [])
        }
    )
    catalog = _load_catalog(db, wanted)
    updated = 0
    for item_id, metadata in rows:
        new_metadata = rebound_metadata(metadata or {}, catalog)
        if new_metadata is None:
            continue
        db.execute(
            text(
                "UPDATE marketplace_items SET metadata = CAST(:metadata AS jsonb), updated_at = NOW() "
                "WHERE id = :item_id"
            ),
            {"metadata": json.dumps(new_metadata), "item_id": item_id},
        )
        updated += 1
    if updated:
        db.commit()
    return updated


# ---------------------------------------------------------------------------
# Background sync + the boot entry point
# ---------------------------------------------------------------------------


def run_catalog_sync_and_rebind() -> Dict[str, Any]:
    """Blocking full catalogue sync, then the seeded-agent rebind.

    Runs on the background thread with its OWN session — the boot session is
    long gone by the time the SDK paging finishes. Self-guarding: never raises.
    """
    from services.metadata_sync_service import MetadataSyncService

    logger.info("[PRD-233 S2] Composio catalogue sync starting (composio_apps_cache was empty)")
    db = SessionLocal()
    try:
        result = MetadataSyncService(db).run_full_sync()
        rebound = rebind_seeded_agents(db)
        logger.info(
            "[PRD-233 S2] Composio catalogue sync finished: apps=%s actions=%s errors=%s; "
            "seeded agents re-bound=%d",
            result.get("apps_synced"),
            result.get("actions_synced"),
            result.get("errors_count", result.get("errors")),
            rebound,
        )
        return {"status": "completed", "sync": result, "rebound": rebound}
    except Exception as exc:  # noqa: BLE001 — background task must not raise
        db.rollback()
        logger.warning("[PRD-233 S2] Composio catalogue sync failed: %s", exc, exc_info=True)
        record_error(subsystem="startup", operation="composio_catalog_sync", error=exc)
        return {"status": "failed", "error": str(exc)}
    finally:
        db.close()


def _start_daemon_thread(target: Callable[[], Any]) -> None:
    threading.Thread(target=target, name=BACKGROUND_THREAD_NAME, daemon=True).start()


def ensure_catalog_on_boot(
    *, start_background: Callable[[Callable[[], Any]], None] = _start_daemon_thread
) -> Dict[str, Any]:
    """The boot stage body. Cheap, self-guarding, never blocks boot.

    ``start_background`` is injectable so tests can capture the scheduled
    callable instead of spawning a thread.
    """
    if not composio_available():
        reason = composio_unavailable_reason()
        logger.info(
            "[PRD-233 S2] Composio integrations disabled — %s. Composio tools are not "
            "offered to agents; the Tools page shows why.",
            reason,
        )
        return {"status": "unavailable", "reason": reason}

    try:
        db = SessionLocal()
        try:
            apps_cached = catalog_app_count(db)
            if apps_cached > 0:
                rebound = rebind_seeded_agents(db)
                if rebound:
                    logger.info(
                        "[PRD-233 S2] Re-bound %d seeded marketplace agent(s) to the Composio catalogue",
                        rebound,
                    )
                return {"status": "ready", "apps_cached": apps_cached, "rebound": rebound}
            if _sync_in_flight(db):
                logger.info("[PRD-233 S2] Composio catalogue sync already in flight — not starting another")
                return {"status": "in_flight", "apps_cached": 0}
        finally:
            db.close()
    except Exception as exc:  # noqa: BLE001 — a boot probe must never abort boot
        logger.warning("[PRD-233 S2] Composio catalogue probe failed (non-fatal): %s", exc, exc_info=True)
        record_error(subsystem="startup", operation="composio_catalog_probe", error=exc)
        return {"status": "error", "error": str(exc)}

    start_background(run_catalog_sync_and_rebind)
    logger.info(
        "[PRD-233 S2] composio_apps_cache is empty and a Composio key is configured — "
        "full catalogue sync scheduled in the background (thread %s)",
        BACKGROUND_THREAD_NAME,
    )
    return {"status": "scheduled", "apps_cached": 0}
