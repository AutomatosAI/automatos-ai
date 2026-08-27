"""
Workspace Hard-Delete (Purge) Service
=====================================

Triggered from the admin "Delete" action on /admin/workspaces.

Sequence:
1. Validate the workspace row is soft-deleted (`deleted_at IS NOT NULL`).
2. Wipe all S3 objects under `s3://{S3_DOCUMENTS_BUCKET}/workspaces/{id}/`.
3. Delete the owning Clerk user (workspaces are per-user; Automatos is the only
   Clerk org and is never touched here).
4. DELETE every row referencing `workspace_id` (explicit list — covers both
   CASCADE and non-CASCADE foreign keys + orphan tables that store workspace_id
   without a FK constraint).
5. DELETE the workspace row itself.

Designed to run from FastAPI `BackgroundTasks` for admin-triggered, low-volume
deletion. Safe to retry: idempotent at every step (404/empty results = ok).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from uuid import UUID

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError
from sqlalchemy import text
from sqlalchemy.orm import Session

from config import config
from core.auth.clerk import get_clerk_auth
from core.database.database import SessionLocal

logger = logging.getLogger(__name__)


# Tables ignored even if they have a `workspace_id` column. Reserved for
# audit / billing tables that should outlive a workspace deletion.
#
# Currently empty — every other workspace-scoped table is purged. Add a
# table name here only if you have a concrete reason (e.g. retain audit
# logs for compliance after the workspace is gone).
_PURGE_SKIP_TABLES: frozenset[str] = frozenset()


def _discover_scoped_tables(db: Session, *, skip: frozenset[str] = frozenset()) -> list[str]:
    """Return every base table (not view) in `public` with a `workspace_id` column.

    Self-maintaining: when a new workspace-scoped table is added to a
    migration, it is purged automatically without code changes here.

    ``skip`` is an EXTRA per-call exclusion (unioned with the module-level
    ``_PURGE_SKIP_TABLES``): the PRD-222 onboarding-reset path passes the
    identity/access/credential survivor tables here so a dev reset spares them
    while still deriving the list dynamically (never a hand-maintained copy).
    """
    rows = db.execute(text(
        """
        SELECT c.table_name
          FROM information_schema.columns c
          JOIN information_schema.tables  t
            ON t.table_name   = c.table_name
           AND t.table_schema = c.table_schema
         WHERE c.column_name  = 'workspace_id'
           AND c.table_schema = 'public'
           AND t.table_type   = 'BASE TABLE'
         ORDER BY c.table_name
        """
    )).fetchall()
    return [r[0] for r in rows if r[0] not in _PURGE_SKIP_TABLES and r[0] not in skip]


@dataclass
class PurgeResult:
    workspace_id: str
    s3_objects_deleted: int = 0
    s3_errors: int = 0
    clerk_user_deleted: bool = False
    clerk_user_id: Optional[str] = None
    rows_deleted: Dict[str, int] = field(default_factory=dict)
    workspace_row_deleted: bool = False
    skipped_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "workspace_id": self.workspace_id,
            "s3_objects_deleted": self.s3_objects_deleted,
            "s3_errors": self.s3_errors,
            "clerk_user_deleted": self.clerk_user_deleted,
            "clerk_user_id": self.clerk_user_id,
            "rows_deleted": self.rows_deleted,
            "workspace_row_deleted": self.workspace_row_deleted,
            "skipped_reason": self.skipped_reason,
        }


def _build_s3_client():
    """Return a configured boto3 S3 client, or None when AWS creds are absent.

    Mirrors the construction pattern in modules/attachments/store.py so behavior
    stays consistent across the codebase (region, retries, sigv4).
    """
    if not (
        getattr(config, "AWS_ACCESS_KEY_ID", None)
        and getattr(config, "AWS_SECRET_ACCESS_KEY", None)
    ):
        return None
    boto_cfg = BotoConfig(
        region_name=config.AWS_REGION or "eu-west-1",
        signature_version="v4",
        retries={"max_attempts": 3, "mode": "adaptive"},
    )
    return boto3.client(
        "s3",
        aws_access_key_id=config.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
        config=boto_cfg,
    )


def _purge_s3_prefix(client, bucket: str, prefix: str) -> tuple[int, int]:
    """Delete all objects under prefix. Returns (deleted_count, error_count)."""
    deleted = 0
    errors = 0
    paginator = client.get_paginator("list_objects_v2")
    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            keys = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
            if not keys:
                continue
            # delete_objects supports up to 1000 keys per call; paginator
            # delivers ≤1000 by default so a single call per page is safe.
            resp = client.delete_objects(Bucket=bucket, Delete={"Objects": keys, "Quiet": True})
            deleted += len(keys) - len(resp.get("Errors", []))
            errors += len(resp.get("Errors", []))
            for err in resp.get("Errors", []):
                logger.error(
                    "S3 delete error for %s: code=%s message=%s",
                    err.get("Key"), err.get("Code"), err.get("Message"),
                )
    except (BotoCoreError, ClientError) as exc:
        logger.exception("S3 purge failed for prefix=%s: %s", prefix, exc)
        errors += 1
    return deleted, errors


def _resolve_owner_clerk_id(db: Session, owner_id: Optional[int]) -> Optional[str]:
    if not owner_id:
        return None
    row = db.execute(
        text("SELECT clerk_user_id FROM users WHERE id = :id"),
        {"id": owner_id},
    ).fetchone()
    return row[0] if row and row[0] else None


async def _delete_clerk_user(clerk_user_id: str) -> bool:
    """Delete the Clerk user. Failures logged, not raised (best-effort cleanup)."""
    try:
        clerk = get_clerk_auth()
        return await clerk.delete_user(clerk_user_id)
    except Exception as exc:  # noqa: BLE001 — never block DB cleanup on Clerk
        logger.exception("Clerk user delete raised: %s", exc)
        return False


def _pre_cascade_external_refs(
    db: Session,
    workspace_id: UUID,
    *,
    ref_predicates: Optional[Dict[str, str]] = None,
    skip_ref_tables: frozenset[str] = frozenset(),
) -> Dict[str, int]:
    """Delete rows from non-workspace-scoped tables that reference
    workspace-scoped tables via NO ACTION / RESTRICT FKs.

    These are the rows that *permanently* block deletion of a workspace-scoped
    parent: they don't carry `workspace_id` themselves (so the main loop's
    discovery skips them), and their FK delete rule won't auto-clear them.
    Examples in this DB: `agent_skills.agent_id`, `tasks.assigned_to`,
    `workflow_agents.agent_id`, `learning_outcomes.agent_id`, …

    Discovered dynamically from `information_schema`, so new dependents
    added by future migrations are handled automatically — no hardcoded
    table list. CASCADE / SET NULL FKs are intentionally excluded because
    Postgres handles them on its own.

    Two optional knobs, used by the PRD-222 onboarding-reset path (defaults keep
    the full-purge behavior byte-identical):
      * ``ref_predicates`` — extra SQL appended to a ref table's
        ``SELECT id … WHERE workspace_id = :wid`` subquery, so only the parents
        actually being deleted have their dependents cleared (e.g. spare the
        system/onboarding agents' skills by cascading dependents of the
        *non-survivor* agents only).
      * ``skip_ref_tables`` — ref tables to leave entirely alone (the survivor
        tables the reset keeps, so their dependents are never touched).
    Predicate strings are hardcoded constants at the call site (never user
    input), consistent with the identifier interpolation already used below.
    """
    ref_predicates = ref_predicates or {}
    refs = db.execute(text(
        """
        WITH ws_tables AS (
          SELECT DISTINCT table_name FROM information_schema.columns
           WHERE column_name = 'workspace_id'
             AND table_schema = 'public'
        )
        SELECT tc.table_name  AS dep_table,
               kcu.column_name AS dep_column,
               ccu.table_name  AS ref_table
          FROM information_schema.table_constraints tc
          JOIN information_schema.key_column_usage kcu
            ON tc.constraint_name = kcu.constraint_name
           AND tc.table_schema   = kcu.table_schema
          JOIN information_schema.constraint_column_usage ccu
            ON tc.constraint_name = ccu.constraint_name
           AND tc.table_schema   = ccu.table_schema
          JOIN information_schema.referential_constraints rc
            ON tc.constraint_name = rc.constraint_name
           AND tc.table_schema   = rc.constraint_schema
         WHERE tc.constraint_type = 'FOREIGN KEY'
           AND tc.table_schema    = 'public'
           AND ccu.table_name IN (SELECT table_name FROM ws_tables)
           AND tc.table_name  NOT IN (SELECT table_name FROM ws_tables)
           AND rc.delete_rule IN ('NO ACTION', 'RESTRICT')
        """
    )).fetchall()

    counts: Dict[str, int] = {}
    for r in refs:
        if r.ref_table in skip_ref_tables:
            continue
        sp = db.begin_nested()  # SAVEPOINT — isolate per-FK failure
        try:
            ref_pred = ref_predicates.get(r.ref_table, "")
            sql = text(
                f'DELETE FROM "{r.dep_table}" '
                f'WHERE "{r.dep_column}" IN '
                f'(SELECT id FROM "{r.ref_table}" WHERE workspace_id::text = :wid {ref_pred})'
            )
            result = db.execute(sql, {"wid": str(workspace_id)})
            sp.commit()
            n = result.rowcount or 0
            if n:
                counts[r.dep_table] = counts.get(r.dep_table, 0) + n
        except Exception as exc:  # noqa: BLE001
            sp.rollback()
            logger.warning(
                "Pre-cascade %s.%s -> %s failed for ws=%s: %s",
                r.dep_table, r.dep_column, r.ref_table, workspace_id, exc,
            )
    return counts


def _delete_rows(
    db: Session,
    workspace_id: UUID,
    *,
    skip_tables: frozenset[str] = frozenset(),
    only_tables: Optional[frozenset[str]] = None,
    row_predicates: Optional[Dict[str, str]] = None,
    ref_predicates: Optional[Dict[str, str]] = None,
    pre_cascade: bool = True,
) -> Dict[str, int]:
    """Delete from every workspace-scoped table; return per-table row counts.

    Tables are discovered dynamically from `information_schema` so new
    workspace-scoped tables are purged automatically. Each table runs in
    its own SAVEPOINT so a single failure doesn't abort the whole purge.

    Two-stage strategy:
      1. Pre-cascade: clear non-scoped dependents (NO ACTION FKs from tables
         without `workspace_id`) — these would otherwise permanently block
         their workspace-scoped parents.
      2. Multi-pass scoped delete: alphabetical order naturally violates FK
         dependencies between scoped tables (e.g. `agents` before `chats`
         even though `chats.current_agent_id` references agents). Retry
         failed tables across passes; each pass removes more dependents,
         eventually unblocking the rest. Bounded at 4 passes.

    All keyword knobs default to the full-purge behavior (byte-identical to the
    original) and exist so the PRD-222 onboarding-reset path can REUSE this exact
    discovery + FK-safe ordering instead of maintaining a second table list:
      * ``skip_tables`` — tables excluded from discovery AND from pre-cascade
        (the reset's identity/access/credential survivors).
      * ``only_tables`` — restrict the scoped delete to this set ∩ discovery
        (the credential-only wipe).
      * ``row_predicates`` — extra SQL appended to a table's DELETE ``WHERE``
        (spare the system/onboarding agents row-level).
      * ``ref_predicates`` — same, forwarded to the pre-cascade subqueries.
      * ``pre_cascade`` — skip stage 1 when the target set has no blocking
        non-scoped dependents (the credential-only wipe).
    """
    row_predicates = row_predicates or {}
    counts: Dict[str, int] = (
        _pre_cascade_external_refs(
            db, workspace_id,
            ref_predicates=ref_predicates,
            skip_ref_tables=skip_tables,
        )
        if pre_cascade else {}
    )
    discovered: set[str] = set(_discover_scoped_tables(db, skip=skip_tables))
    if only_tables is not None:
        discovered &= set(only_tables)
    pending: set[str] = discovered
    last_errors: Dict[str, Exception] = {}

    for _ in range(4):
        if not pending:
            break
        succeeded_this_pass: set[str] = set()
        for tbl in sorted(pending):
            sp = db.begin_nested()  # SAVEPOINT — isolates per-table failures
            try:
                pred = row_predicates.get(tbl, "")
                result = db.execute(
                    text(f'DELETE FROM "{tbl}" WHERE workspace_id::text = :wid {pred}'),
                    {"wid": str(workspace_id)},
                )
                sp.commit()
                count = result.rowcount or 0
                if count:
                    counts[tbl] = count
                succeeded_this_pass.add(tbl)
            except Exception as exc:  # noqa: BLE001
                sp.rollback()
                last_errors[tbl] = exc
        if not succeeded_this_pass:
            # No progress this pass — remaining tables have unresolvable
            # blockers (likely a non-workspace-scoped referencer). Bail.
            break
        pending -= succeeded_this_pass

    for tbl in pending:
        logger.warning(
            "Skip purge of %s for ws=%s: %s",
            tbl, workspace_id, last_errors.get(tbl),
        )
        counts[tbl] = -1  # sentinel: errored — kept in result for visibility
    return counts


def purge_workspace_sync(workspace_id: UUID) -> PurgeResult:
    """Synchronous purge entry point — used by `BackgroundTasks` callers.

    Internally runs the async Clerk call via asyncio.run since BackgroundTasks
    invokes plain callables on a thread.
    """
    result = PurgeResult(workspace_id=str(workspace_id))

    db: Session = SessionLocal()
    try:
        ws = db.execute(
            text("SELECT id, owner_id, deleted_at FROM workspaces WHERE id = :id"),
            {"id": str(workspace_id)},
        ).fetchone()

        if not ws:
            result.skipped_reason = "workspace_not_found"
            logger.info("Purge skipped — workspace %s not found", workspace_id)
            return result

        if ws.deleted_at is None:
            # Defensive: callers MUST soft-delete first. Refuse to purge a live
            # workspace even if a buggy admin path tries to call us directly.
            result.skipped_reason = "not_soft_deleted"
            logger.error("Purge refused — workspace %s is not soft-deleted", workspace_id)
            return result

        owner_id = ws.owner_id
        clerk_uid = _resolve_owner_clerk_id(db, owner_id)

        # 1) S3
        s3 = _build_s3_client()
        # Match the fallback used elsewhere (see modules/documents/generation_service.py)
        bucket = getattr(config, "S3_DOCUMENTS_BUCKET", None) or "automatos-ai"
        if s3 and bucket:
            prefix = f"workspaces/{workspace_id}/"
            result.s3_objects_deleted, result.s3_errors = _purge_s3_prefix(s3, bucket, prefix)
            logger.info(
                "S3 purge for ws=%s: deleted=%d errors=%d (s3://%s/%s)",
                workspace_id, result.s3_objects_deleted, result.s3_errors, bucket, prefix,
            )
        else:
            logger.info("S3 purge skipped — no AWS creds or bucket configured")

        # 2) Clerk (best-effort, never blocks DB cleanup)
        if clerk_uid:
            result.clerk_user_id = clerk_uid
            result.clerk_user_deleted = asyncio.run(_delete_clerk_user(clerk_uid))

        # 3) DB cascade — explicit per-table deletes
        result.rows_deleted = _delete_rows(db, workspace_id)

        # 4) The workspace row itself — must come BEFORE the users row,
        # because workspaces.owner_id has a FK to users.id with NO ACTION.
        # Deleting users first violates that constraint.
        wd = db.execute(text("DELETE FROM workspaces WHERE id = :id"), {"id": str(workspace_id)})
        result.workspace_row_deleted = bool(wd.rowcount)

        # 5) Owner user row (only if no other workspaces remain owned by them).
        # Re-check after the workspace row is gone so the lookup is accurate.
        if owner_id:
            other = db.execute(
                text("SELECT 1 FROM workspaces WHERE owner_id = :oid LIMIT 1"),
                {"oid": owner_id},
            ).fetchone()
            if not other:
                db.execute(text("DELETE FROM users WHERE id = :id"), {"id": owner_id})
                result.rows_deleted["users"] = 1

        db.commit()
        logger.warning(
            "Workspace purged: id=%s clerk_user=%s s3_deleted=%d rows=%s",
            workspace_id, clerk_uid, result.s3_objects_deleted, result.rows_deleted,
        )
        return result

    except Exception as exc:  # noqa: BLE001
        db.rollback()
        logger.exception("Workspace purge failed for %s: %s", workspace_id, exc)
        result.skipped_reason = f"exception: {exc}"
        return result
    finally:
        db.close()


# =========================================================================== #
# PRD-222 W1·S10 (D9) — scoped reuse for the dev onboarding-reset path.
#
# These REUSE the discovery + FK-safe ordering + S3 machinery above, restricted
# to ONE surviving workspace. They deliberately do NOT run the soft-delete
# precondition, the workspace-row DELETE, the users DELETE, or the Clerk-user
# delete that ``purge_workspace_sync`` performs — a reset keeps the workspace,
# its owner, its membership, and (unless asked) its credentials. No second table
# list is maintained anywhere: the survivors below are the only hardcoded names,
# and they are EXCLUSIONS from the dynamic discovery, not a copy of it.
# =========================================================================== #

# Identity/access rows that keep the workspace a valid, owned, reachable
# workspace after a reset. ``workspace_members`` gates auth (a member row with
# ``is_active`` is what lets the operator back in) — wiping it would lock them
# out of the very workspace they are trying to re-onboard.
_RESET_ACCESS_SURVIVOR_TABLES: frozenset[str] = frozenset({"workspace_members"})

# The workspace's credential stores. Onboarding does NOT "build" these, so the
# built-artifact wipe always spares them; ``wipe_credentials`` deletes them on
# its own, separate path. ``credential_audit_logs`` is NOT listed: it has no
# ``workspace_id`` and CASCADEs from ``credentials`` automatically.
_CREDENTIAL_TABLES: frozenset[str] = frozenset({"user_api_keys", "credentials"})

# Row-level survivor filter for the ``agents`` table: onboarding builds the
# workspace's ordinary agents, but the platform's system agents and the hidden
# onboarding-role agents (VOYAGER/BLUEPRINT/SCRIBE/FORGE) are seeded, not built —
# they must survive. Applied to BOTH the DELETE and the pre-cascade subquery so
# a spared agent keeps its dependent rows (skills, tool routes, …) too.
#
# MUST be NULL-safe. A built agent has ``required_role IS NULL``; the naive form
# ``AND NOT (is_system_agent IS TRUE OR required_role = 'onboarding')`` evaluates
# to ``NOT (FALSE OR NULL)`` = ``NOT NULL`` = NULL for those rows, and a NULL
# WHERE-clause matches NOTHING — so ordinary built agents would silently survive
# ``wipe_built`` (the common case). Written as two positive, NULL-safe predicates
# instead: ``IS NOT TRUE`` keeps NULL/false system flags out of the survivor set,
# and ``IS DISTINCT FROM 'onboarding'`` treats a NULL role as "not onboarding".
_AGENT_SURVIVOR_SQL: str = "AND is_system_agent IS NOT TRUE AND required_role IS DISTINCT FROM 'onboarding'"


def _wipe_workspace_s3(workspace_id: UUID) -> Dict[str, Any]:
    """Delete the workspace's S3 document prefix, REUSING the purge S3 helpers.

    No AWS creds / bucket → a clean skip (the local + test path), never an error.
    """
    client = _build_s3_client()
    bucket = getattr(config, "S3_DOCUMENTS_BUCKET", None) or "automatos-ai"
    if not client or not bucket:
        return {"s3_objects_deleted": 0, "s3_errors": 0, "skipped": "no_client_or_bucket"}
    prefix = f"workspaces/{workspace_id}/"
    deleted, errors = _purge_s3_prefix(client, bucket, prefix)
    logger.info(
        "Onboarding-reset S3 wipe ws=%s: deleted=%d errors=%d (s3://%s/%s)",
        workspace_id, deleted, errors, bucket, prefix,
    )
    return {"s3_objects_deleted": deleted, "s3_errors": errors}


def purge_built_artifacts(db: Session, workspace_id: UUID, *, wipe_s3: bool = True) -> Dict[str, Any]:
    """Delete what onboarding BUILT in one workspace, sparing survivors.

    Reuses :func:`_delete_rows` (dynamic discovery + FK-safe multi-pass) with:
      * the identity/access + credential tables excluded (survivors), and
      * the ``agents`` table filtered to non-system, non-onboarding rows.
    Then wipes the workspace's S3 document prefix via :func:`_purge_s3_prefix`.
    Never touches the workspace row, users, membership, or Clerk. Does NOT commit
    — the caller (``reset_onboarding``) owns the transaction.
    """
    rows = _delete_rows(
        db,
        workspace_id,
        skip_tables=_RESET_ACCESS_SURVIVOR_TABLES | _CREDENTIAL_TABLES,
        row_predicates={"agents": _AGENT_SURVIVOR_SQL},
        ref_predicates={"agents": _AGENT_SURVIVOR_SQL},
    )
    s3 = _wipe_workspace_s3(workspace_id) if wipe_s3 else {"skipped": "wipe_s3=False"}
    return {"rows_deleted": rows, "s3": s3}


def purge_workspace_credentials(db: Session, workspace_id: UUID) -> Dict[str, int]:
    """Delete ONLY this workspace's credential rows (BYOK keys + generic
    credentials), REUSING the scoped-delete machinery restricted to the credential
    tables. ``credential_audit_logs`` CASCADEs from ``credentials`` on its own, so
    no pre-cascade is needed. Scoped to ``workspace_id`` — another workspace's
    rows (including the platform-key workspace) are never in range. Does NOT
    commit — the caller owns the transaction.
    """
    return _delete_rows(
        db,
        workspace_id,
        only_tables=_CREDENTIAL_TABLES,
        pre_cascade=False,
    )
