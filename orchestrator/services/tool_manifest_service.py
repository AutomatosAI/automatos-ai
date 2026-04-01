"""
Tool Manifest Service — PRD-123 Pattern #3
===========================================

Versioned snapshots of tool definitions for change tracking and debugging.
Writes manifests to S3 after each Composio sync.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

import boto3

from config import config

logger = logging.getLogger(__name__)

_BUCKET = config.RECIPE_LOG_S3_BUCKET
_MANIFEST_PREFIX = "manifests"


def _s3_client():
    return boto3.client("s3")


def _schema_hash(schema: dict) -> str:
    """SHA-256 hash of a tool's JSON schema for change detection."""
    raw = json.dumps(schema, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


async def snapshot_tool_manifest(workspace_id: UUID, db_session=None) -> dict:
    """
    Capture a versioned snapshot of all tools for a workspace.

    Args:
        workspace_id: The workspace to snapshot.
        db_session: SQLAlchemy session for querying tools.

    Returns:
        The manifest dict that was written to S3.
    """
    from modules.tools.registry.tool_registry import get_tool_registry

    registry = get_tool_registry(db_session=db_session)
    all_tools = registry.get_all_tools(active_only=True)

    version = datetime.now(timezone.utc).isoformat()

    tools_data = []
    for tool in all_tools:
        schema = tool.to_openai_format() if hasattr(tool, "to_openai_format") else {}
        tools_data.append({
            "name": tool.name,
            "tier": getattr(tool, "tier", "marketplace"),
            "schema_hash": _schema_hash(schema),
            "provider": getattr(tool, "provider", "unknown"),
            "category": str(getattr(tool, "category", "")),
        })

    manifest = {
        "version": version,
        "workspace_id": str(workspace_id),
        "tool_count": len(tools_data),
        "tools": tools_data,
    }

    key = f"{_MANIFEST_PREFIX}/{workspace_id}/{version}.json"

    try:
        _s3_client().put_object(
            Bucket=_BUCKET,
            Key=key,
            Body=json.dumps(manifest),
            ContentType="application/json",
        )
        logger.info(
            "Tool manifest snapshot: workspace=%s tools=%d key=%s",
            workspace_id, len(tools_data), key,
        )
    except Exception as exc:
        logger.error("Failed to write tool manifest: %s", exc)
        raise

    return manifest


async def diff_manifests(
    workspace_id: UUID,
    from_version: str,
    to_version: str,
) -> dict[str, Any]:
    """
    Diff two manifest versions to find added/removed/changed tools.

    Args:
        workspace_id: The workspace.
        from_version: ISO timestamp of the earlier manifest.
        to_version: ISO timestamp of the later manifest.

    Returns:
        Dict with added, removed, changed tool lists.
    """
    try:
        from_key = f"{_MANIFEST_PREFIX}/{workspace_id}/{from_version}.json"
        to_key = f"{_MANIFEST_PREFIX}/{workspace_id}/{to_version}.json"

        s3 = _s3_client()
        from_data = json.loads(s3.get_object(Bucket=_BUCKET, Key=from_key)["Body"].read())
        to_data = json.loads(s3.get_object(Bucket=_BUCKET, Key=to_key)["Body"].read())

        from_tools = {t["name"]: t for t in from_data.get("tools", [])}
        to_tools = {t["name"]: t for t in to_data.get("tools", [])}

        from_names = set(from_tools.keys())
        to_names = set(to_tools.keys())

        added = sorted(to_names - from_names)
        removed = sorted(from_names - to_names)
        changed = sorted(
            name for name in from_names & to_names
            if from_tools[name].get("schema_hash") != to_tools[name].get("schema_hash")
        )

        return {
            "workspace_id": str(workspace_id),
            "from_version": from_version,
            "to_version": to_version,
            "added": added,
            "removed": removed,
            "changed": changed,
            "summary": f"{len(added)} added, {len(removed)} removed, {len(changed)} changed",
        }

    except Exception as exc:
        logger.error("Manifest diff failed: %s", exc)
        return {"error": str(exc)}
