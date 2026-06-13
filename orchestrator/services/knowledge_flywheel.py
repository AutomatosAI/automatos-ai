"""Knowledge Flywheel (PRD-164 S3, Q58/D6)
==========================================

ONE choke point that turns agent outputs into retrievable knowledge:
mission syntheses, generated documents, and submitted reports all route
through :func:`ingest_agent_output`, which

1. honors the per-workspace opt-out (flywheel is ON by default — Q58);
2. routes the content through the EXISTING ingestion manager
   (``modules/rag/ingestion/manager.py`` → chunked, embedded, searchable),
   tagged ``source_type='agent_output'``;
3. schedules the Knowledge-Graph incremental build with the SPECIFIC
   agent-output source type, so the KG learns the three source types it
   used to drop (see ``graph_service.partition_pending_sources``).

Opt-out contract: a workspace with ``settings['knowledge_flywheel_enabled']
= false`` ingests NOTHING — no document row, no chunks, no KG pending.
There is deliberately no second gate anywhere else; every caller goes
through this module so the opt-out is provable at one seam.

No parallel ingestion path: this module never chunks/embeds itself — it
hands the content to ``DocumentManager.upload_document`` and cleans up.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Canonical Q58 tag on the documents row (real column — see alembic revision
# prd164_doc_source_type) so agent outputs are a filterable, team-like scope.
AGENT_OUTPUT_SOURCE_TYPE = "agent_output"

# The three agent-output sources the flywheel routes (PRD-164 S3).
SOURCE_MISSION_SYNTHESIS = "mission_synthesis"
SOURCE_GENERATED_DOCUMENT = "generated_document"
SOURCE_REPORT = "report"
AGENT_OUTPUT_SOURCES = (
    SOURCE_MISSION_SYNTHESIS,
    SOURCE_GENERATED_DOCUMENT,
    SOURCE_REPORT,
)

# workspace.settings key (same JSONB home as the PRD-167 brand kit — no new
# table). Absent/None/anything-but-False == enabled: Q58 ON by default.
FLYWHEEL_SETTINGS_KEY = "knowledge_flywheel_enabled"

# Cap on report text carried inside a KG pending — mirrors the extraction cap
# in graph_service (_MAX_DOC_CHARS) so the debounce buffer stays bounded.
KG_PENDING_TEXT_CAP = 8000


def flywheel_enabled(db: Session, workspace_id: UUID | str) -> bool:
    """Q58: ON by default; only an explicit ``false`` opts the workspace out.

    Fail-open by design: a missing workspace row or settings read error keeps
    the default (enabled) — the flywheel is platform behaviour, the opt-out is
    the exception.
    """
    try:
        from core.models.workspaces import Workspace

        ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
        settings = getattr(ws, "settings", None) or {}
        return settings.get(FLYWHEEL_SETTINGS_KEY) is not False
    except Exception:
        logger.warning(
            "[Flywheel] Could not read workspace settings for %s — defaulting to enabled",
            workspace_id,
            exc_info=True,
        )
        return True


def _build_kg_pending(
    *,
    source: str,
    source_id: Optional[str],
    document_id: Optional[int],
    title: str,
    content: str,
    agent_name: Optional[str],
) -> Dict[str, Any]:
    """Build the typed KG pending for this agent output.

    Reports carry their text so the incremental build can run the
    agent-attributed ``extract_from_report`` without re-fetching workspace
    files; synthesis/generated-document pendings reference the ingested
    document id (the document extractor picks them up — no double LLM pass).
    """
    if source == SOURCE_REPORT:
        return {
            "type": SOURCE_REPORT,
            "id": source_id,
            "path": title,
            "text": (content or "")[:KG_PENDING_TEXT_CAP],
            "agent_name": agent_name or "unknown",
        }
    return {
        "type": source,
        "id": source_id,
        "document_id": document_id,
        "path": title,
    }


def _schedule_kg_pending(workspace_id: UUID | str, pending: Dict[str, Any]) -> None:
    """Best-effort KG schedule — never fails the ingest."""
    try:
        from modules.knowledge.graph_service import get_graph_service

        get_graph_service().schedule_incremental_update(str(workspace_id), [pending])
    except Exception:
        logger.debug("[Flywheel] KG schedule skipped — service not available")


async def ingest_agent_output(
    db: Session,
    workspace_id: UUID | str,
    *,
    content: str,
    filename: str,
    source: str,
    source_id: Optional[str] = None,
    title: Optional[str] = None,
    description: str = "",
    agent_name: Optional[str] = None,
    created_by: str = "flywheel",
    extra_tags: Optional[List[str]] = None,
) -> Optional[int]:
    """Route one agent output through the existing ingestion manager.

    Returns the ingested document id, or ``None`` when the workspace has
    opted out (Q58) or ingestion failed (fail-soft: producing the output
    must never be broken by the knowledge loop).

    Args:
        content: The output text (markdown preferred — it chunks well).
        filename: Stored document filename (extension drives extraction).
        source: One of :data:`AGENT_OUTPUT_SOURCES`.
        source_id: Native id of the output (mission id, report id, …).
        title: Human title; used for the KG pending path + description.
        agent_name: Producing agent (report KG attribution).
        extra_tags: Additional document tags (e.g. ``mission:<id>``).
    """
    if source not in AGENT_OUTPUT_SOURCES:
        raise ValueError(
            f"source must be one of {AGENT_OUTPUT_SOURCES}, got {source!r}"
        )
    if not content or not content.strip():
        logger.debug("[Flywheel] Empty %s content for %s — nothing to ingest", source, workspace_id)
        return None

    # Q58 opt-out: the ONE gate. Nothing below runs for an opted-out workspace.
    if not flywheel_enabled(db, workspace_id):
        logger.info(
            "[Flywheel] Workspace %s opted out — skipping %s ingest", workspace_id, source
        )
        return None

    title = title or filename
    tags = [AGENT_OUTPUT_SOURCE_TYPE, source]
    if extra_tags:
        tags.extend(t for t in extra_tags if t)

    suffix = os.path.splitext(filename)[1] or ".md"
    tmp_path: Optional[str] = None
    try:
        # Lazy import: get_document_manager carries the canonical db_config
        # (same accessor the coordinator already uses).
        from api.documents import get_document_manager

        with tempfile.NamedTemporaryFile(
            "w", suffix=suffix, delete=False, encoding="utf-8"
        ) as fh:
            fh.write(content)
            tmp_path = fh.name

        manager = get_document_manager(str(workspace_id))
        document_id = await manager.upload_document(
            file_path=tmp_path,
            filename=filename,
            tags=tags,
            description=description or f"Agent output ({source}): {title}"[:500],
            created_by=created_by,
            source_type=AGENT_OUTPUT_SOURCE_TYPE,
        )

        _schedule_kg_pending(
            workspace_id,
            _build_kg_pending(
                source=source,
                source_id=str(source_id) if source_id is not None else None,
                document_id=document_id,
                title=title,
                content=content,
                agent_name=agent_name,
            ),
        )

        logger.info(
            "[Flywheel] Ingested %s '%s' as document %s (workspace %s)",
            source,
            title,
            document_id,
            workspace_id,
        )
        return document_id
    except Exception:
        logger.error(
            "[Flywheel] %s ingest failed for workspace %s (output flow unaffected)",
            source,
            workspace_id,
            exc_info=True,
        )
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                logger.debug("[Flywheel] temp cleanup failed for %s", tmp_path)
