"""
DbWorkspaceClient — Postgres-backed graph artefact storage
===========================================================

Drop-in replacement for ``WorkspaceClient`` used exclusively by
``modules.knowledge.graph_service``. Exposes the same async interface
(``read_file``, ``write_file``, ``list_dir``, ``delete_file``) but persists
to the ``workspace_graphs`` table instead of round-tripping to the
workspace-worker HTTP API.

Why this exists
---------------
The workspace worker is only provisioned on demand for workspaces that
have active agent shell sessions. Wizard-created workspaces do not have a
worker container, so every graph write was failing with
``404 Workspace not found`` — the graph built fine in memory then
evaporated. Persisting graphs in Postgres makes the write path
deterministic for every workspace and keyed to the tenant row, which
is where the profile + business data already live.

Interface contract
------------------
All methods return ``{"success": bool, ...}`` mirroring the WorkspaceClient
contract so ``graph_service._write_json`` / ``_read_json`` continue to
work unmodified.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from sqlalchemy import text as sa_text

from core.database.database import get_db_session

logger = logging.getLogger(__name__)


class DbWorkspaceClient:
    """Postgres-backed storage for workspace graph artefacts.

    Each instance is scoped to a single workspace. All operations run on
    the ``workspace_graphs`` table with composite key ``(workspace_id, path)``.
    """

    def __init__(self, workspace_id: str) -> None:
        self.workspace_id = workspace_id

    # ── File operations ────────────────────────────────────────────

    async def read_file(self, path: str) -> Dict[str, Any]:
        """Load a single artefact by path."""
        return await asyncio.to_thread(self._read_file_sync, path)

    def _read_file_sync(self, path: str) -> Dict[str, Any]:
        try:
            with get_db_session() as db:
                row = db.execute(
                    sa_text(
                        "SELECT content FROM workspace_graphs "
                        "WHERE workspace_id = :ws AND path = :path"
                    ),
                    {"ws": self.workspace_id, "path": path},
                ).fetchone()
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "DbWorkspaceClient.read_file ws=%s path=%s failed: %s",
                self.workspace_id, path, exc, exc_info=True,
            )
            return {"success": False, "error": str(exc)}

        if row is None:
            return {"success": False, "error": "not_found", "status_code": 404}
        return {"success": True, "content": row[0]}

    async def write_file(self, path: str, content: str) -> Dict[str, Any]:
        """Upsert an artefact into ``workspace_graphs``."""
        return await asyncio.to_thread(self._write_file_sync, path, content)

    def _write_file_sync(self, path: str, content: str) -> Dict[str, Any]:
        try:
            with get_db_session() as db:
                db.execute(
                    sa_text(
                        "INSERT INTO workspace_graphs (workspace_id, path, content, updated_at) "
                        "VALUES (:ws, :path, :content, now()) "
                        "ON CONFLICT (workspace_id, path) DO UPDATE "
                        "SET content = EXCLUDED.content, updated_at = now()"
                    ),
                    {"ws": self.workspace_id, "path": path, "content": content},
                )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "DbWorkspaceClient.write_file ws=%s path=%s failed: %s",
                self.workspace_id, path, exc, exc_info=True,
            )
            return {"success": False, "error": str(exc)}

        logger.debug(
            "DbWorkspaceClient.write_file ws=%s path=%s bytes=%d",
            self.workspace_id, path, len(content),
        )
        return {"success": True}

    async def list_dir(self, path: str = ".") -> Dict[str, Any]:
        """List artefacts whose path starts with ``path/``.

        Returns entries shaped like ``{"name": basename}`` to match
        ``WorkspaceClient.list_dir``'s response. Only direct children are
        returned — nested paths are stripped to their last segment.
        """
        return await asyncio.to_thread(self._list_dir_sync, path)

    def _list_dir_sync(self, path: str) -> Dict[str, Any]:
        prefix = path.rstrip("/") + "/" if path and path != "." else ""
        try:
            with get_db_session() as db:
                rows = db.execute(
                    sa_text(
                        "SELECT path FROM workspace_graphs "
                        "WHERE workspace_id = :ws AND path LIKE :prefix"
                    ),
                    {"ws": self.workspace_id, "prefix": f"{prefix}%"},
                ).fetchall()
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "DbWorkspaceClient.list_dir ws=%s path=%s failed: %s",
                self.workspace_id, path, exc, exc_info=True,
            )
            return {"success": False, "error": str(exc), "entries": []}

        entries: list[Dict[str, str]] = []
        seen: set[str] = set()
        for (row_path,) in rows:
            remainder = row_path[len(prefix):] if prefix else row_path
            # Only keep direct children (strip any further path segments)
            name = remainder.split("/", 1)[0]
            if name and name not in seen:
                seen.add(name)
                entries.append({"name": name})
        return {"success": True, "entries": entries}

    async def delete_file(self, path: str) -> Dict[str, Any]:
        """Remove a single artefact by path."""
        return await asyncio.to_thread(self._delete_file_sync, path)

    def _delete_file_sync(self, path: str) -> Dict[str, Any]:
        try:
            with get_db_session() as db:
                db.execute(
                    sa_text(
                        "DELETE FROM workspace_graphs "
                        "WHERE workspace_id = :ws AND path = :path"
                    ),
                    {"ws": self.workspace_id, "path": path},
                )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "DbWorkspaceClient.delete_file ws=%s path=%s failed: %s",
                self.workspace_id, path, exc, exc_info=True,
            )
            return {"success": False, "error": str(exc)}
        return {"success": True}
