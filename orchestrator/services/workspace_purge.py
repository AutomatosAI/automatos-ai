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
from typing import Dict, Optional
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


# Tables that hold a `workspace_id` column (FK or orphan). Order does not
# matter — we issue DELETEs per table, then the workspaces row last. The DB
# will silently accept no-op deletes (rows already gone via CASCADE).
#
# Source of truth: `git grep -n "workspace_id = Column" orchestrator/core/models/`.
# Update this list when a new workspace-scoped table is added.
_WORKSPACE_SCOPED_TABLES = [
    # Core domain
    "agents",
    "documents",
    "chats",
    "messages",
    "recipes",
    "recipe_templates",
    "recipe_runs",
    "blog_posts",
    "credentials",
    "personas",
    "voice_profiles",
    "business_profiles",
    "blueprints",
    "channels",
    # Marketplace / plugins
    "marketplace_plugins",
    "workspace_enabled_plugins",
    "workspace_enabled_skills",
    # Composio / tools
    "composio_cache",
    "composio_workspace_state",
    "tool_assignments",
    "workspace_tool_configs",
    # Knowledge / NL2SQL
    "database_knowledge",
    "nl2sql_examples",
    # Routing / orchestration
    "routing_rules",
    "routing_decisions",
    "routing_overrides",
    "orchestration_runs",
    "orchestration_archive",
    # Cloud sync / widget / SDK
    "cloud_sync_state",
    "cloud_sync_objects",
    "widget_installations",
    "sdk_api_keys",
    # Membership / access
    "workspace_members",
]


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


def _delete_rows(db: Session, workspace_id: UUID) -> Dict[str, int]:
    """Delete from each workspace-scoped table; return per-table row counts.

    Tables that don't exist in this deployment are silently skipped.
    """
    counts: Dict[str, int] = {}
    for tbl in _WORKSPACE_SCOPED_TABLES:
        try:
            result = db.execute(
                text(f"DELETE FROM {tbl} WHERE workspace_id = :wid"),
                {"wid": str(workspace_id)},
            )
            counts[tbl] = result.rowcount or 0
        except Exception as exc:  # noqa: BLE001
            # Missing table = skip. Anything else, log and continue (admin can
            # rerun; partial cleanup is better than no cleanup).
            db.rollback()
            logger.warning("Skip purge of %s for ws=%s: %s", tbl, workspace_id, exc)
            counts[tbl] = -1  # sentinel: errored
            continue
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
        bucket = getattr(config, "S3_DOCUMENTS_BUCKET", None)
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

        # 4) Owner user row (only if no other workspaces remain owned by them)
        if owner_id:
            other = db.execute(
                text(
                    "SELECT 1 FROM workspaces "
                    "WHERE owner_id = :oid AND id != :wid LIMIT 1"
                ),
                {"oid": owner_id, "wid": str(workspace_id)},
            ).fetchone()
            if not other:
                db.execute(text("DELETE FROM users WHERE id = :id"), {"id": owner_id})
                result.rows_deleted["users"] = 1

        # 5) Final: the workspace row itself
        wd = db.execute(text("DELETE FROM workspaces WHERE id = :id"), {"id": str(workspace_id)})
        result.workspace_row_deleted = bool(wd.rowcount)

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
