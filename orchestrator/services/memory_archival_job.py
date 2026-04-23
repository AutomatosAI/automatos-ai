"""
Memory Archival Job — PRD-131d Phase 4
======================================
Monthly archival that promotes aged memories into the workspace business
knowledge graph, then purges the source rows. Two inputs per workspace:

  • L2 (Postgres memory_short_term):
      rows that are already archived (archived_at IS NOT NULL) OR whose
      decay_score has fallen below MEMORY_ARCHIVAL_L2_DECAY_THRESHOLD.
      These rows are going to be dropped from working memory anyway — this
      job is their final chance to contribute to the long-term graph.

  • L3 (Mem0 long-term):
      memories older than MEMORY_ARCHIVAL_L3_RETENTION_DAYS. Kept as raw
      facts for a long time, then folded into the graph where clustering
      can surface structural relationships.

For each workspace with candidates, the job:
  1. Builds a NetworkX node_link_data payload (one node per memory, no
     inferred edges — graphify's clustering will group them).
  2. Calls GraphifyService.import_graph(workspace_id, data, merge=True).
  3. On success, hard-deletes L2 rows and removes L3 entries via Mem0.
  4. On failure, leaves the data in place for the next run.

The job is intentionally monthly and per-workspace-scoped so a single
workspace failure never blocks the others.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sync DB helpers (run in executor)
# ---------------------------------------------------------------------------


def _fetch_l2_archival_candidates_sync(
    workspace_id: str,
    decay_threshold: float,
    batch_size: int,
) -> List[Dict[str, Any]]:
    """Return L2 rows eligible for archival (archived or decayed below threshold)."""
    from core.database.database import get_db_session
    from modules.memory.models import MemoryShortTerm
    from sqlalchemy import or_

    with get_db_session() as db:
        rows = (
            db.query(MemoryShortTerm)
            .filter(
                MemoryShortTerm.workspace_id == workspace_id,
                or_(
                    MemoryShortTerm.archived_at.isnot(None),
                    MemoryShortTerm.decay_score < decay_threshold,
                ),
            )
            .order_by(MemoryShortTerm.created_at.asc())
            .limit(batch_size)
            .all()
        )
        return [
            {
                "id": str(row.id),
                "content": row.content or "",
                "content_type": row.content_type,
                "importance": float(row.importance or 0),
                "decay_score": float(row.decay_score or 0),
                "agent_id": row.agent_id,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "metadata": row.metadata_ or {},
            }
            for row in rows
        ]


def _delete_l2_rows_sync(workspace_id: str, ids: List[str]) -> int:
    """Hard-delete L2 rows by id (scoped to workspace). Returns deleted count."""
    from core.database.database import get_db_session
    from modules.memory.models import MemoryShortTerm

    if not ids:
        return 0

    with get_db_session() as db:
        deleted = (
            db.query(MemoryShortTerm)
            .filter(
                MemoryShortTerm.workspace_id == workspace_id,
                MemoryShortTerm.id.in_(ids),
            )
            .delete(synchronize_session=False)
        )
        db.commit()
        return int(deleted or 0)


def _active_workspace_ids_sync() -> List[str]:
    """Distinct workspace_ids with any L2 rows (archived or not)."""
    from core.database.database import get_db_session
    from modules.memory.models import MemoryShortTerm
    from sqlalchemy import distinct

    with get_db_session() as db:
        rows = db.query(distinct(MemoryShortTerm.workspace_id)).all()
        return [str(r[0]) for r in rows]


# ---------------------------------------------------------------------------
# L3 helpers
# ---------------------------------------------------------------------------


def _parse_created_at(value: Any) -> datetime | None:
    """Mem0 returns created_at as ISO string OR unix int — normalize to datetime."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


async def _fetch_l3_archival_candidates(
    service,
    workspace_id: str,
    retention_days: int,
    limit: int,
) -> List[Dict[str, Any]]:
    """L3 memories older than the retention cutoff."""
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=retention_days)

    try:
        all_memories = await service.get_all_memories(workspace_id, limit=limit)
    except Exception:
        logger.warning(
            "[archival] L3 fetch failed ws=%s", workspace_id, exc_info=True,
        )
        return []

    aged: List[Dict[str, Any]] = []
    for m in all_memories or []:
        created = _parse_created_at(m.get("created_at"))
        if created and created < cutoff:
            aged.append(
                {
                    "id": str(m.get("id") or ""),
                    "content": m.get("memory") or m.get("content") or "",
                    "created_at": m.get("created_at"),
                    "metadata": m.get("metadata") or {},
                }
            )
    return aged


# ---------------------------------------------------------------------------
# Graph payload builder
# ---------------------------------------------------------------------------


def _build_node_link_data(
    workspace_id: str,
    l2_items: List[Dict[str, Any]],
    l3_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Turn archival candidates into a NetworkX node_link_data payload."""
    nodes: List[Dict[str, Any]] = []

    for row in l2_items:
        content = row.get("content") or ""
        label = content[:80].rstrip() or row.get("content_type") or "memory"
        nodes.append(
            {
                "id": f"l2:{row['id']}",
                "label": label,
                "type": row.get("content_type") or "memory",
                "source_tier": "l2",
                "content": content,
                "importance": row.get("importance"),
                "decay_score": row.get("decay_score"),
                "created_at": row.get("created_at"),
                "agent_id": row.get("agent_id"),
                "workspace_id": workspace_id,
            }
        )

    for row in l3_items:
        content = row.get("content") or ""
        label = content[:80].rstrip() or "fact"
        nodes.append(
            {
                "id": f"l3:{row['id']}",
                "label": label,
                "type": "fact",
                "source_tier": "l3",
                "content": content,
                "created_at": row.get("created_at"),
                "workspace_id": workspace_id,
            }
        )

    return {
        "directed": False,
        "multigraph": False,
        "graph": {"source": "memory_archival", "workspace_id": workspace_id},
        "nodes": nodes,
        "links": [],
    }


# ---------------------------------------------------------------------------
# Main job
# ---------------------------------------------------------------------------


class MemoryArchivalJob:
    """Run a single pass of memory→graph archival across every workspace."""

    async def run_once(self) -> Dict[str, Any]:
        from config import config as app_config
        from modules.knowledge.graph_service import get_graph_service
        from modules.memory.unified_memory_service import (
            get_unified_memory_service,
        )

        decay_threshold = float(
            getattr(app_config, "MEMORY_ARCHIVAL_L2_DECAY_THRESHOLD", 0.2)
        )
        retention_days = int(
            getattr(app_config, "MEMORY_ARCHIVAL_L3_RETENTION_DAYS", 180)
        )
        batch_size = int(
            getattr(app_config, "MEMORY_ARCHIVAL_BATCH_SIZE", 500)
        )

        loop = asyncio.get_event_loop()
        workspace_ids = await loop.run_in_executor(
            None, _active_workspace_ids_sync,
        )

        memory_service = get_unified_memory_service()
        graph_service = get_graph_service()

        summary = {
            "workspaces_processed": 0,
            "workspaces_with_candidates": 0,
            "l2_archived": 0,
            "l3_archived": 0,
            "nodes_imported": 0,
            "errors": 0,
        }

        for ws_id in workspace_ids:
            summary["workspaces_processed"] += 1
            try:
                l2_items, l3_items = await self._collect_candidates(
                    memory_service, ws_id, decay_threshold, retention_days, batch_size,
                )
                if not l2_items and not l3_items:
                    continue

                summary["workspaces_with_candidates"] += 1

                graph_data = _build_node_link_data(ws_id, l2_items, l3_items)
                await graph_service.import_graph(
                    workspace_id=ws_id,
                    graph_data=graph_data,
                    merge=True,
                )
                summary["nodes_imported"] += len(graph_data["nodes"])

                # Purge sources ONLY after import succeeds.
                l2_deleted = await loop.run_in_executor(
                    None,
                    _delete_l2_rows_sync,
                    ws_id,
                    [row["id"] for row in l2_items],
                )
                summary["l2_archived"] += l2_deleted

                l3_deleted = 0
                for row in l3_items:
                    try:
                        if await memory_service.delete_memory(row["id"]):
                            l3_deleted += 1
                    except Exception:
                        logger.warning(
                            "[archival] L3 delete failed ws=%s id=%s",
                            ws_id, row.get("id"), exc_info=True,
                        )
                summary["l3_archived"] += l3_deleted

                logger.info(
                    "[archival] ws=%s imported=%d l2_deleted=%d l3_deleted=%d",
                    ws_id, len(graph_data["nodes"]), l2_deleted, l3_deleted,
                )
            except Exception:
                summary["errors"] += 1
                logger.error(
                    "[archival] workspace %s failed", ws_id, exc_info=True,
                )

        logger.info(
            "[archival] complete ws=%d with_candidates=%d imported=%d "
            "l2_archived=%d l3_archived=%d errors=%d",
            summary["workspaces_processed"],
            summary["workspaces_with_candidates"],
            summary["nodes_imported"],
            summary["l2_archived"],
            summary["l3_archived"],
            summary["errors"],
        )
        return summary

    async def _collect_candidates(
        self,
        memory_service,
        workspace_id: str,
        decay_threshold: float,
        retention_days: int,
        batch_size: int,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        loop = asyncio.get_event_loop()
        l2_items = await loop.run_in_executor(
            None,
            _fetch_l2_archival_candidates_sync,
            workspace_id,
            decay_threshold,
            batch_size,
        )
        l3_items = await _fetch_l3_archival_candidates(
            memory_service, workspace_id, retention_days, batch_size,
        )
        return l2_items, l3_items
