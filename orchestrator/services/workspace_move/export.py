"""PRD-273 US-273-02 — write one workspace to one ``.automatos-workspace`` file.

``write_workspace_file`` reads the database's schema, then writes, for the chosen categories
(§5.2; Auto always): ``workspace.json`` and ``auto.json``; ``sql/<table>.jsonl`` for every
moved table (rows in batches, secret and derived columns left out); ``refs/`` for the
install-level rows they name; the agent and Playbook bundles; durable and field memory; the
document and Deliverable files; and ``manifest.json`` last. Memory stays flat: rows stream
from a server-side cursor and files go through a temporary file, so nothing grows with the
workspace but the zip on disk.

The caller owns the session (one consistent snapshot is the job runner's choice, ``jobs.py``)
and the output file. A failure raises; the zip is closed and the caller deletes it.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, BinaryIO, Dict, List, Mapping, Optional, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

from config import config
from services.workspace_move.auto_part import auto_document, workspace_document
from services.workspace_move.bundles import AGENT_TYPE, external_id, write_bundles
from services.workspace_move.crypto import new_file_key
from services.workspace_move.files import write_files
from services.workspace_move.manifest import (
    AUTO_PART, WORKSPACE_PART, Contents, Source, build_manifest, sql_part, summary,
)
from services.workspace_move.memory import write_memory
from services.workspace_move.refs import write_refs
from services.workspace_move.rows import TablePlan, plan_table, table_rows
from services.workspace_move.rules import AGENTS, CATEGORIES, DELIVERABLES, KNOWLEDGE, MEMORY, resolve_categories
from services.workspace_move.schema import SchemaView, alembic_head, read_schema, unclassified_tables
from services.workspace_move.table_map import TABLE_RULES
from services.workspace_move.writer import MoveFileWriter

MEGABYTE = 1024 * 1024
UNCLASSIFIED = "no rule classifies it"


@dataclass(frozen=True)
class ExportResult:
    """What one export wrote: its manifest and the job's short summary of it."""

    manifest: Dict[str, Any]
    summary: Dict[str, Any]


def _moved_rules(categories: Tuple[str, ...]):
    """The moved rules of ``categories``, in §5.4's category order, then by table."""
    order = {category: index for index, category in enumerate(CATEGORIES)}
    chosen = [rule for rule in TABLE_RULES.values() if rule.moved and rule.category in categories]
    return sorted(chosen, key=lambda rule: (order[rule.category], rule.table))


def _write_tables(db: Session, writer: MoveFileWriter, schema: SchemaView, workspace_id: UUID | str,
                  categories: Tuple[str, ...]) -> Tuple[Dict[str, TablePlan], List[Dict[str, str]]]:
    """One ``sql/`` part per moved table this database has; (the plans written, the skips)."""
    batch = max(1, int(config.WORKSPACE_EXPORT_BATCH_ROWS))
    plans: Dict[str, TablePlan] = {}
    skipped = [{"table": table, "reason": UNCLASSIFIED} for table in unclassified_tables(schema)]
    for rule in _moved_rules(categories):
        if not schema.has(rule.table):
            continue
        plan, why = plan_table(schema, rule)
        if plan is None:
            skipped.append({"table": rule.table, "reason": why})
            continue
        writer.write_rows(sql_part(rule.table), table_rows(db, schema, plan, workspace_id, batch))
        plans[rule.table] = plan
    return plans, skipped


def _write_auto(db: Session, writer: MoveFileWriter, workspace_id: UUID | str) -> Dict[str, Any]:
    """``workspace.json`` and ``auto.json``; the workspace document."""
    workspace = workspace_document(db, workspace_id)
    writer.write_json(WORKSPACE_PART, workspace)
    auto = auto_document(db, workspace_id)
    if auto["agent"] is not None:
        ext_id = external_id(AGENT_TYPE, workspace_id, auto["agent"]["source_agent_id"])
        auto = {"agent": {**auto["agent"], "externalId": ext_id}}
    writer.write_json(AUTO_PART, auto)
    return workspace


def _write_parts(db: Session, writer: MoveFileWriter, workspace_id: UUID | str, categories: Tuple[str, ...],
                 encryption: Optional[Mapping[str, Any]]) -> Tuple[Dict[str, Any], Contents]:
    """Every part but the manifest; (the workspace document, what was written)."""
    schema = read_schema(db)
    workspace = _write_auto(db, writer, workspace_id)
    plans, skipped_tables = _write_tables(db, writer, schema, workspace_id, categories)
    write_refs(db, writer, schema, plans, workspace_id)
    unbundled: List[Dict[str, Any]] = []
    if AGENTS in categories:
        _, unbundled = write_bundles(db, writer, workspace_id, workspace["name"])
    memory = write_memory(writer, str(workspace_id)) if MEMORY in categories else None
    files, skipped_files = 0, []
    if KNOWLEDGE in categories or DELIVERABLES in categories:
        files, skipped_files = write_files(db, writer, workspace_id, categories)
    contents = Contents(
        categories=categories, plans=plans, parts=writer.parts, skipped_tables=tuple(skipped_tables),
        skipped_files=tuple(skipped_files), unbundled=tuple(unbundled), memory=memory, files=files,
        encryption=encryption,
    )
    return workspace, contents


def write_workspace_file(db: Session, workspace_id: UUID | str, out: BinaryIO, *,
                         categories: Optional[Tuple[str, ...]] = None, passphrase: Optional[str] = None,
                         exported_by: Optional[str] = None, now: Optional[datetime] = None) -> ExportResult:
    """Write the workspace to ``out`` (a seekable binary file). Raises ValueError on an unknown
    category, auto_part.WorkspaceGone when the workspace does not exist, writer.ExportTooLarge over the
    size cap; any failure leaves ``out`` closed for the caller to delete."""
    chosen = resolve_categories(categories)
    workspace_id = UUID(str(workspace_id))
    stamp = now or datetime.now(timezone.utc)
    key = new_file_key(passphrase) if passphrase else None
    writer = MoveFileWriter(out, key=key, stamp=stamp, max_bytes=int(config.WORKSPACE_EXPORT_MAX_TOTAL_MB) * MEGABYTE)
    try:
        encryption = key.manifest_block() if key else None
        workspace, contents = _write_parts(db, writer, workspace_id, chosen, encryption)
        source = Source(
            edition=config.AUTH_EDITION, alembic_head=alembic_head(db), workspace_id=str(workspace_id),
            workspace_name=workspace["name"], exported_at=stamp, exported_by=exported_by,
        )
        manifest = build_manifest(source, contents)
        writer.finish(manifest)
    except BaseException:
        writer.abort()
        raise
    return ExportResult(manifest=manifest, summary=summary(contents))


__all__ = ["ExportResult", "write_workspace_file"]
