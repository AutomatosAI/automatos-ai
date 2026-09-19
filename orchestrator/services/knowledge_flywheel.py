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
from typing import Any, Dict, List, Optional, Tuple
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

# ── What the knowledge graph is FOR (Gerard, 2026-09-18) ────────────────────
# "This is not what the knowledge graph is for — not all heartbeats and reports
# from all agents. Knowledge graph and RAG are for how my business runs,
# customer data, Shopify products and orders."
#
# Night 1 entity-extracted all 240 agent reports, standups and heartbeat logs
# included: 497 graph_extraction calls and most of a $9.98 gemini line, to learn
# that an agent posted a standup. Reports still become RETRIEVABLE (the RAG
# ingest above is untouched) — only graph EXTRACTION is scoped, and only by what
# the report is.
KG_REPORT_TYPES_IN = ("research", "delivery", "analysis", "findings", "recommendation")
KG_REPORT_TYPES_OUT = ("standup", "heartbeat", "progress", "status", "log")

# workspace.settings key: a list of report types to extract, overriding the
# default allowlist. An empty list turns report extraction off for a workspace.
KG_REPORT_TYPES_SETTINGS_KEY = "knowledge_graph_report_types"
# workspace.settings key: how many graph extractions a workspace may spend a day.
KG_DAILY_CAP_SETTINGS_KEY = "knowledge_graph_daily_extraction_cap"
KG_DAILY_EXTRACTION_CAP = 200


# Titles that are the harness talking to itself, not the business. Night 1
# (2026-09-18): the platform's own copy of every CLI ticket ("2026-09-18_180245_
# 7a942e_task-…"), heartbeat and playbook logs, and "pass N / nothing changed"
# no-op passes were all entity-extracted. F024: telemetry never.
_TELEMETRY_TITLE_MARKERS: Tuple[str, ...] = (
    "_task-",            # the platform's dated copy of a CLI ticket
    "heartbeat",
    "playbook run",
    "nothing changed",
    "no action taken",
    "no changes",
    "execution metrics",
    "standup",
)


def title_is_telemetry(title: Optional[str]) -> bool:
    """True when this output is the system describing its own operation.

    Matched on the title because that is what every one of these shares and it
    costs nothing to read — the alternative is paying an LLM to discover that a
    heartbeat log contains no business entities.
    """
    haystack = (title or "").strip().lower()
    if not haystack:
        return False
    return any(marker in haystack for marker in _TELEMETRY_TITLE_MARKERS)


def report_type_in_graph_scope(db: Session, workspace_id: UUID | str, report_type: Optional[str]) -> bool:
    """Whether a report of this type earns an entity-extraction pass.

    The workspace can override the allowlist wholesale; an explicit empty list
    means "no reports in the graph at all". An unrecognised type is OUT: the
    graph is opt-in for report kinds nobody has vouched for.
    """
    kind = (report_type or "").strip().lower()
    override = _workspace_setting(db, workspace_id, KG_REPORT_TYPES_SETTINGS_KEY)
    if isinstance(override, list):
        return kind in {str(t).strip().lower() for t in override}
    if kind in KG_REPORT_TYPES_OUT:
        return False
    return kind in KG_REPORT_TYPES_IN


def graph_extraction_budget_left(db: Session, workspace_id: UUID | str) -> bool:
    """False once this workspace has spent its day's graph-extraction budget.

    Counted off ``llm_usage`` rows for the ``graph_extraction`` service, which is
    where the cost actually shows up. Unreadable ledger → allow (the cap is a
    guard rail, not a gate).
    """
    cap = _workspace_setting(db, workspace_id, KG_DAILY_CAP_SETTINGS_KEY)
    try:
        cap = int(cap) if cap is not None else KG_DAILY_EXTRACTION_CAP
    except (TypeError, ValueError):
        cap = KG_DAILY_EXTRACTION_CAP
    if cap <= 0:
        return False
    try:
        from sqlalchemy import text as sa_text

        spent = db.execute(
            sa_text(
                "SELECT COUNT(*) FROM llm_usage "
                "WHERE workspace_id = CAST(:ws AS uuid) "
                "  AND request_type = 'graph_extraction' "
                "  AND created_at >= NOW() - INTERVAL '1 day'"
            ),
            {"ws": str(workspace_id)},
        ).scalar()
    except Exception:  # noqa: BLE001
        logger.debug("[Flywheel] graph-extraction budget unreadable — allowing", exc_info=True)
        return True
    return int(spent or 0) < cap


def graph_store_available() -> bool:
    """False when the store the graph feeds is down.

    Extracting entities into a store that cannot take them spends money for
    nothing — night 1 kept extracting while memory was off.
    """
    from config import config as app_config

    return bool(getattr(app_config, "QDRANT_URL", ""))


def _workspace_setting(db: Session, workspace_id: UUID | str, key: str) -> Any:
    try:
        from core.models.workspaces import Workspace

        ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
        return (getattr(ws, "settings", None) or {}).get(key)
    except Exception:  # noqa: BLE001
        logger.debug("[Flywheel] workspace setting %s unreadable", key, exc_info=True)
        return None


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


def _kg_extraction_allowed(
    db: Session, workspace_id: UUID | str, source: str, report_type: Optional[str],
    title: Optional[str] = None,
) -> bool:
    """Whether this output earns a graph-extraction pass (Gerard, 2026-09-18).

    Mission syntheses and generated documents are business output and stay in.
    Reports are filtered by what kind of report they are. Both are subject to
    the workspace's daily budget and to the store being up.
    """
    if not graph_store_available():
        logger.info("[Flywheel] graph store is down — no extraction for %s", source)
        return False
    if title_is_telemetry(title):
        logger.debug("[Flywheel] '%s' is telemetry, not business knowledge — RAG only", title)
        return False
    if source == SOURCE_REPORT and not report_type_in_graph_scope(db, workspace_id, report_type):
        logger.debug(
            "[Flywheel] report_type=%s is outside the graph's scope — RAG only", report_type,
        )
        return False
    if not graph_extraction_budget_left(db, workspace_id):
        logger.warning(
            "[Flywheel] workspace %s has spent its daily graph-extraction budget", workspace_id,
        )
        return False
    return True


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
    report_type: Optional[str] = None,
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

        # The report is now retrievable either way. Whether it also gets an
        # entity-extraction pass is a separate, narrower question.
        if _kg_extraction_allowed(db, workspace_id, source, report_type, title):
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
