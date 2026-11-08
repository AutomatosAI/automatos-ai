"""PRD-181 S3 + S4 — GDPR data export + erasure-with-derived-data-cascade.

Risk #4 (UK right-to-erasure, pilot-binding): a delete that leaves the subject in
field memory / vectors / durable memory is **not** a GDPR delete. This service is the real
cascade — it reaches every store, primary and derived:

  - **SQL** — reuses ``services.workspace_purge`` (self-maintaining over every
    ``workspace_id`` table, incl. learned-edge tables + S3 document objects).
  - **Qdrant field memory** — ``VectorFieldSharedContext.erase_workspace`` /
    ``export_workspace`` (workspace_id payload filter, PRD-166 S1).
  - **durable memories** — ``UnifiedMemoryService.erase_workspace_memories``
    (one workspace-filter delete on the in-process Qdrant store, PRD-187 S1).
  - **RAG document vectors** — S3-Vectors objects live under the
    ``workspaces/{id}/`` prefix and are already wiped by the SQL purge's S3 step;
    pgvector rows carry ``workspace_id`` and are caught by the SQL purge.

Every export and every erasure is audited (governance action).

**Subject granularity:** subject-level erasure is implemented where a
data-subject tag exists. PRD-196 S6 added the ``subject_id`` tag to field memory
and durable memory, so a subject erase now does a real filter-delete in both
(reported as deleted counts). **SQL** remains workspace-scoped (a documented
``gaps`` entry — per-table subject resolution belongs to the Shopify redact
handler), and rows written *before* the tag existed are reported as an
``untagged_history`` caveat (additive, no backfill — never claimed erased). The
single ``erase_data_subject`` / ``erase_workspace`` entrypoints are what the
Shopify ``customers/redact`` webhook calls.

The store-specific readers/erasers are module-level functions so they can be
swapped in tests and so the sync↔async bridge (SQL/audit are sync; Qdrant legs
are async) lives in exactly one place (:func:`_run_async`).
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional
from uuid import UUID

from services.workspace_move.rows import redacted as without_redacted_keys
from services.workspace_move.rules import looks_secret
from services.workspace_move.table_map import SECRET_TABLES, TABLE_RULES

logger = logging.getLogger(__name__)

# PRD-273 US-273-04: v2 leaves out the secret tables and columns and carries the workspace's
# allow-listed settings (the brand kit, the Orchestrator block, voice and push).
EXPORT_FORMAT = "automatos.gdpr.export/v2"

# PRD-266 K16, PRD-273 §5.6: columns an export never carries, by table — secrets the server keeps
# for itself, named once in PRD-273's table map (``services/workspace_move/table_map.py``): a
# phone's hashes, push key and Expo token (anyone holding it can push to that phone), a paired
# machine's hashes, an invitation's token, a site's secrets, a credential link. A column whose
# name looks like a secret (``rules.SECRET_COLUMN_NAME``) is left out too, named or not.
EXPORT_REDACTED_COLUMNS: Mapping[str, tuple[str, ...]] = MappingProxyType({
    table: rule.secret_columns for table, rule in TABLE_RULES.items() if rule.secret_columns
})


# ---------------------------------------------------------------------------
# Async bridge — SQL purge + audit are sync; the Qdrant legs are async. Callers
# (a FastAPI endpoint / a webhook) may or may not have a running loop, so run the
# async legs on a fresh loop when needed.
# ---------------------------------------------------------------------------

def _run_async(coro) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # A loop is already running (shouldn't happen from the sync API path, but be
    # safe): run the coroutine on a dedicated loop in this thread.
    new_loop = asyncio.new_event_loop()
    try:
        return new_loop.run_until_complete(coro)
    finally:
        new_loop.close()


# ---------------------------------------------------------------------------
# Store adapters (module-level so tests can monkeypatch each independently).
# ---------------------------------------------------------------------------

def _exported_row(table: str, row: Mapping[str, Any]) -> dict:
    """One row as the export carries it: without the columns ``EXPORT_REDACTED_COLUMNS`` names for
    its table, nor any whose name looks like a secret, nor a secret-looking key inside a JSON value,
    nor the JSON keys its rule removes (a Playbook trigger's webhook id and secret)."""
    redacted = EXPORT_REDACTED_COLUMNS.get(table, ())
    kept = {column: value for column, value in row.items() if column not in redacted and not looks_secret(column)}
    return without_redacted_keys(kept, TABLE_RULES.get(table))


def _export_sql_tables(db: Any, workspace_id: UUID | str) -> Dict[str, List[dict]]:
    """Read every ``workspace_id``-scoped SQL table into plain rows (portable), less the tables that
    hold nothing but secrets and connections (PRD-273 §5.6: credentials, API keys, channel and
    Composio connections)."""
    from sqlalchemy import text
    from services.workspace_purge import _discover_scoped_tables

    out: Dict[str, List[dict]] = {}
    try:
        tables = _discover_scoped_tables(db)
    except Exception:
        logger.warning("[gdpr] scoped-table discovery failed", exc_info=True)
        return out
    for tbl in (table for table in tables if table not in SECRET_TABLES):
        try:
            rows = db.execute(
                text(f'SELECT * FROM "{tbl}" WHERE workspace_id::text = :ws'),
                {"ws": str(workspace_id)},
            ).mappings().all()
            out[tbl] = [_exported_row(tbl, r) for r in rows]
        except Exception:
            logger.warning("[gdpr] export read failed for table %s", tbl, exc_info=True)
    return out


def _export_workspace_settings(db: Any, workspace_id: UUID | str) -> Optional[dict]:
    """PRD-273 US-273-04: the workspace's name and allow-listed settings, as a move's
    ``workspace.json`` carries them; None when the workspace is gone."""
    from services.workspace_move.auto_part import WorkspaceGone, workspace_document

    try:
        return workspace_document(db, workspace_id)
    except WorkspaceGone:
        return None


def _export_field_memory(workspace_id: UUID | str, subject_id: Optional[str] = None) -> List[dict]:
    from config import config as _config

    if not getattr(_config, "QDRANT_URL", ""):
        return []  # PRD-238 S10: field memory is off in this edition — nothing to export
    from modules.context.adapters.vector_field import VectorFieldSharedContext

    svc = VectorFieldSharedContext()
    return _run_async(svc.export_workspace(str(workspace_id)))


def _export_durable_memory(workspace_id: UUID | str, subject_id: Optional[str] = None) -> List[dict]:
    from modules.memory.unified_memory_service import UnifiedMemoryService

    svc = UnifiedMemoryService()
    return _run_async(svc.export_workspace_memories(str(workspace_id)))


def _close_workspace(workspace_id: UUID | str) -> bool:
    """Mark the workspace deleted, as the admin delete does, before anything is erased (#1442).

    The purge refuses a workspace that was not soft-deleted first, and closing it
    first also stops new writes landing while the stores are erased. Idempotent: a
    workspace already closed keeps its ``deleted_at``. False when there is no
    such workspace.
    """
    from sqlalchemy import text

    from core.database.database import SessionLocal

    db = SessionLocal()
    try:
        row = db.execute(
            text("UPDATE workspaces SET deleted_at = COALESCE(deleted_at, NOW()), is_active = false "
                 "WHERE id = :id RETURNING id"),
            {"id": str(workspace_id)},
        ).fetchone()
        db.commit()
        return row is not None
    finally:
        db.close()


def _purge_sql(workspace_id: UUID | str) -> Dict[str, Any]:
    """Reuse the self-maintaining workspace purge for the SQL + S3 legs."""
    from services.workspace_purge import purge_workspace_sync

    result = purge_workspace_sync(workspace_id if isinstance(workspace_id, UUID) else UUID(str(workspace_id)))
    return result.to_dict()


def _erase_subject_sql(db: Any, workspace_id: UUID | str, subject_id: str) -> Dict[str, Any]:
    """Subject-level SQL erasure (see the GDPR-GAP note below)."""
    # GDPR-GAP: the platform's SQL schema is workspace-scoped, not
    # data-subject-scoped — there is no generic `subject_id` column across the
    # scoped tables to filter a single human's rows. Subject rows would have to
    # be resolved per-table (e.g. a Shopify customer id on commerce tables),
    # which is domain-specific and belongs to the Shopify pod's redact handler.
    # This returns 0 and the gap is surfaced; workspace-level SQL erasure is the
    # supported path today.
    logger.warning(
        "[gdpr] subject-level SQL erasure requested for subject=%s ws=%s but no "
        "generic data-subject column exists (GDPR-GAP) — 0 rows",
        subject_id, workspace_id,
    )
    return {"deleted": 0}


def _erase_field_memory(workspace_id: UUID | str, subject_id: Optional[str] = None) -> int:
    from config import config as _config

    if not getattr(_config, "QDRANT_URL", ""):
        return 0  # field memory is off in this edition: nothing to erase, not a failure (#1442)
    from modules.context.adapters.vector_field import VectorFieldSharedContext

    svc = VectorFieldSharedContext()
    if subject_id is not None:
        return _run_async(svc.erase_subject(str(workspace_id), subject_id))
    return _run_async(svc.erase_workspace(str(workspace_id)))


def _erase_durable_memory(workspace_id: UUID | str, subject_id: Optional[str] = None) -> int:
    from modules.memory.unified_memory_service import UnifiedMemoryService

    svc = UnifiedMemoryService()
    if subject_id is not None:
        return _run_async(svc.erase_subject_memories(str(workspace_id), subject_id))
    return _run_async(svc.erase_workspace_memories(str(workspace_id)))


def _audit_gdpr(db: Any, workspace_id: UUID | str, action: str, **details: Any) -> None:
    """Write a GDPR governance AuditLog row (S1 nullable-user path)."""
    try:
        from core.workspaces.audit import AuditService

        actor = details.get("requested_by")
        AuditService(db).log(
            workspace_id=str(workspace_id) if workspace_id is not None else None,
            user_id=None,
            actor_type="system",
            action=action,
            resource_type="workspace",
            resource_id=str(workspace_id),
            resource_name=details.get("subject_id"),
            details={**details, "actor": actor},
        )
    except Exception:
        logger.warning("[gdpr] audit failed for %s", action, exc_info=True)


# ---------------------------------------------------------------------------
# Gap ledger — the stores that still lack a data-subject tag, surfaced (never
# hidden) on a subject-level operation. PRD-196 S6 closed field_memory and
# durable_memory (they now filter-delete by subject_id), so only SQL remains a
# structural gap. Pre-tag rows in the tagged stores are reported separately as
# an untagged-history caveat (below) — never claimed erased.
# ---------------------------------------------------------------------------

_SUBJECT_GAPS: List[Dict[str, str]] = [
    {
        "store": "sql",
        "reason": "SQL schema is workspace-scoped, not subject-scoped; per-table "
                  "subject resolution (e.g. Shopify customer id on commerce tables) "
                  "is domain-specific and owned by the Shopify redact handler.",
    },
]

# PRD-196 S6: the subject tag is ADDITIVE (no backfill — there is no identity to
# backfill from). Memories written before the tag existed carry no subject_id and
# cannot be attributed to a subject, so they stay under workspace-level erasure.
# Reported honestly on every subject erase — never silently claimed erased.
_UNTAGGED_HISTORY_CAVEAT: Dict[str, Any] = {
    "stores": ["field_memory", "durable_memory"],
    "reason": "Subject tagging is additive from PRD-196 S6 onward. Memories "
              "written before the tag existed carry no data-subject tag and are "
              "NOT attributable to a subject for deletion; they remain reachable "
              "only by workspace-level erasure. This is reported, never hidden.",
}


# ===========================================================================
# S3 — export
# ===========================================================================

def export_workspace(db: Any, workspace_id: UUID | str, *, requested_by: str = "system") -> Dict[str, Any]:
    """Export a portable JSON bundle of a workspace's data across all stores."""
    bundle = {
        "format": EXPORT_FORMAT,
        "workspace_id": str(workspace_id),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "requested_by": requested_by,
        "workspace": _export_workspace_settings(db, workspace_id),
        "sql": _export_sql_tables(db, workspace_id),
        "derived": {
            "field_memory": _export_field_memory(workspace_id),
            "durable_memory": _export_durable_memory(workspace_id),
        },
    }
    _audit_gdpr(db, workspace_id, "gdpr:export", requested_by=requested_by)
    return bundle


# ===========================================================================
# S4 — erasure cascade
# ===========================================================================

def erase_workspace(db: Any, workspace_id: UUID | str, *, requested_by: str) -> Dict[str, Any]:
    """Erase a whole workspace across SQL + Qdrant field + durable memory (the real cascade).

    This is the ``erase_workspace`` entrypoint the future Shopify webhook and the
    admin purge path call. SQL (incl. S3 objects + learned-edge tables) via the
    self-maintaining purge; derived stores via their workspace erasers.

    #1442: the workspace is closed (soft-deleted) first, since the purge refuses a
    live one. ``complete`` is true only when every step ran: ``failed`` names any
    step that raised, an SQL purge that did not delete the workspace row, and S3
    objects it could not delete.
    """
    closed = _safe(lambda: _close_workspace(workspace_id), "close")
    field_deleted = _safe(lambda: _erase_field_memory(workspace_id), "field_memory")
    durable_deleted = _safe(lambda: _erase_durable_memory(workspace_id), "durable_memory")
    sql_result = _safe(lambda: _purge_sql(workspace_id), "sql") or {}
    failed = _failed_steps(
        {"close": closed, "field_memory": field_deleted, "durable_memory": durable_deleted}, sql_result,
    )
    if failed:
        logger.error("[gdpr] workspace erasure incomplete ws=%s failed=%s", workspace_id, failed)

    result = {
        "workspace_id": str(workspace_id),
        "erased_at": datetime.now(timezone.utc).isoformat(),
        "requested_by": requested_by,
        "sql": sql_result,
        "derived": {
            "field_memory_deleted": field_deleted or 0,
            "durable_memory_deleted": durable_deleted or 0,
        },
        "failed": failed,
        "complete": not failed,
    }
    _audit_gdpr(
        db, workspace_id, "gdpr:erasure",
        requested_by=requested_by,
        scope="workspace",
        field_memory_deleted=field_deleted or 0,
        durable_memory_deleted=durable_deleted or 0,
        sql_rows=sql_result.get("rows_deleted"),
        failed=failed,
        complete=not failed,
    )
    return result


def _failed_steps(steps: Mapping[str, Any], sql_result: Mapping[str, Any]) -> List[str]:
    """The erasure steps that did not do their job (#1442): a step ``_safe`` caught raising
    (None), an SQL purge that left the workspace row, and S3 objects left behind."""
    failed = [name for name, outcome in steps.items() if outcome is None]
    if not sql_result.get("workspace_row_deleted"):
        failed.append("sql")
    if sql_result.get("s3_errors"):
        failed.append("s3")
    return failed


def erase_data_subject(
    db: Any, *, workspace_id: UUID | str, subject_id: str, requested_by: str
) -> Dict[str, Any]:
    """Erase a single data subject's data where a subject tag exists (S4).

    The single subject-level entrypoint the future Shopify ``customers/redact``
    webhook calls. Cascades to each store's subject eraser; where a store lacks a
    data-subject tag the erasure returns 0 and the gap is reported in ``gaps`` —
    the caller is never told a subject was erased from a store where it was not.
    """
    field_deleted = _safe(lambda: _erase_field_memory(workspace_id, subject_id), "field_memory")
    durable_deleted = _safe(lambda: _erase_durable_memory(workspace_id, subject_id), "durable_memory")
    sql_result = _safe(lambda: _erase_subject_sql(db, workspace_id, subject_id), "sql") or {"deleted": 0}

    # Dynamic + honest (PRD-196 S6): field/durable report real deleted counts;
    # SQL stays a documented gap; pre-tag rows are an untagged-history caveat.
    result = {
        "workspace_id": str(workspace_id),
        "subject_id": subject_id,
        "erased_at": datetime.now(timezone.utc).isoformat(),
        "requested_by": requested_by,
        "sql": sql_result,
        "derived": {
            "field_memory_deleted": field_deleted or 0,
            "durable_memory_deleted": durable_deleted or 0,
        },
        "gaps": list(_SUBJECT_GAPS),
        "untagged_history": dict(_UNTAGGED_HISTORY_CAVEAT),
    }
    _audit_gdpr(
        db, workspace_id, "gdpr:erasure",
        requested_by=requested_by,
        scope="subject",
        subject_id=subject_id,
        field_memory_deleted=field_deleted or 0,
        durable_memory_deleted=durable_deleted or 0,
        gaps=[g["store"] for g in _SUBJECT_GAPS],
        untagged_history=True,
    )
    return result


def _safe(fn, label: str) -> Any:
    """Run one erasure/export leg; log + swallow so a failure in one store does
    not abort the cascade (partial deletion is reported, never silent)."""
    try:
        return fn()
    except Exception:
        logger.warning("[gdpr] leg '%s' failed", label, exc_info=True)
        return None
