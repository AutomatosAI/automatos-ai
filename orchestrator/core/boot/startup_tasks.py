"""Observable boot-time background seeds (PRD-142 Wave 1 · WS-C · W1-S7).

These two coroutines were previously nested closures launched fire-and-forget
from ``main.py`` whose failures were only ``logger.warning``-ed. Extracted here
they are importable + unit-testable, and on failure they now also fire
``record_error(subsystem="startup")`` so a failed boot seed surfaces on the
ERRORS-by-subsystem dashboard tile instead of dying silently.

Both functions are *self-guarding*: they never raise, so launching them with a
bare ``create_task`` cannot leave an unretrieved task exception.
"""
from __future__ import annotations

import logging

from core.utils.exception_telemetry import record_error

logger = logging.getLogger(__name__)


async def embed_all_agents_on_startup() -> None:
    """Seed semantic embeddings for agents across all workspaces (PRD-64).

    Non-fatal: any failure is logged + recorded and boot continues.
    """
    try:
        from core.database.database import SessionLocal
        from core.llm.embedding_manager import get_embedding_manager
        from core.models.core import Agent
        from core.models.workspaces import Workspace
        from core.routing.semantic_indexer import embed_workspace_agents

        db = SessionLocal()
        try:
            emgr = get_embedding_manager()
            emgr._ensure_provider()
            logger.info("PRD-64: Embedding provider: %s", emgr.get_provider_info())

            ws_ids = [w.id for w in db.query(Workspace.id).all()]
            total = 0
            for ws_id in ws_ids:
                try:
                    total += await embed_workspace_agents(ws_id, db)
                except Exception:
                    logger.warning("PRD-64: Failed to embed workspace %s", ws_id, exc_info=True)

            all_agents = db.query(Agent).filter(Agent.status == "active").count()
            with_embeddings = (
                db.query(Agent)
                .filter(Agent.status == "active", Agent.semantic_embedding.isnot(None))
                .count()
            )
            logger.info(
                "PRD-64: Semantic embeddings seeded — %d new, %d/%d agents have embeddings",
                total,
                with_embeddings,
                all_agents,
            )
        finally:
            db.close()
    except Exception as exc:
        logger.warning("PRD-64: Startup embedding seed failed (non-fatal): %s", exc, exc_info=True)
        record_error(subsystem="startup", operation="embed_all_agents", error=exc)


async def ensure_field_memory_collection() -> None:
    """Ensure the shared ``field_memory`` collection + indexes exist (PRD-108).

    Must run before the coordinator boots. Non-fatal: any failure is logged +
    recorded and boot continues.
    """
    try:
        from modules.context.adapters.vector_field import VectorFieldSharedContext
        from modules.context.factory import get_shared_context

        ctx = get_shared_context()
        inner = getattr(ctx, "_inner", ctx)
        if isinstance(inner, VectorFieldSharedContext):
            await inner.ensure_shared_collection()
            logger.info("PRD-108: shared field_memory collection ready")
    except Exception as exc:
        logger.warning("PRD-108: shared field_memory bootstrap failed (non-fatal)", exc_info=True)
        record_error(subsystem="startup", operation="ensure_field_memory_collection", error=exc)
