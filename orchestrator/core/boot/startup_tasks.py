"""Observable boot-time background seeds (PRD-142 Wave 1 · WS-C · W1-S7).

The agent and field-memory seeds were previously nested closures launched
fire-and-forget from ``main.py`` whose failures were only ``logger.warning``-ed.
Extracted here they are importable + unit-testable, and on failure they now also
fire ``record_error(subsystem="startup")`` so a failed boot seed surfaces on the
ERRORS-by-subsystem dashboard tile instead of dying silently. The action-index
warm-up (#927) and the output-budget load (#836) follow the same contract.

Every function here is *self-guarding*: it never raises, so launching it with a
bare ``create_task`` cannot leave an unretrieved task exception.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Set

from core.utils.exception_telemetry import record_error

logger = logging.getLogger(__name__)

# Long-lived refresh tasks, held so the loop does not drop them.
_refreshers: Set["asyncio.Task[None]"] = set()


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


async def warm_action_index_on_startup() -> None:
    """Embed the platform action catalogue before the first chat turn (#927).

    A turn no longer waits for a cold index (it ranks nothing and falls back), so
    this is what keeps the first turns after a restart, or after an upgrade that
    rewords actions, on semantic ranking. Texts already in Redis are cache hits;
    only new or changed ones go upstream. With no embedding provider configured it
    skips with one line. Non-fatal: any other failure is logged + recorded.
    """
    try:
        from core.llm.clients.base import EmbeddingUnavailableError
        from modules.tools.discovery.action_semantic_index import get_action_semantic_index

        try:
            await get_action_semantic_index().warm()
        except EmbeddingUnavailableError as exc:
            logger.info("#927: action index not warmed — no embedding provider (%s)", exc)
            return
        logger.info("#927: action index warmed")
    except Exception as exc:
        logger.warning("#927: warming the action index failed (non-fatal): %s", exc, exc_info=True)
        record_error(subsystem="startup", operation="warm_action_index", error=exc)


async def warm_output_budgets_on_startup() -> None:
    """Load the output budgets' snapshot before the worker serves, then keep it
    fresh in the background (#836, core/llm/budget_snapshot.py).

    A manager built on the event loop reads budgets from memory only. Without
    this load the first managers after a restart used the table's budgets, and
    an llm_output_budget row looked ignored. Every worker runs it. Non-fatal: a
    failed database read is logged by refresh() and the table's budgets answer;
    any other failure is logged + recorded.
    """
    try:
        from core.llm import budget_snapshot

        snapshot = await budget_snapshot.warm()
    except Exception as exc:
        logger.warning("#836: loading the output budgets failed (non-fatal): %s", exc, exc_info=True)
        record_error(subsystem="startup", operation="warm_output_budgets", error=exc)
        return
    logger.info("#836: output budgets loaded (%d override(s), %d thinking model(s))",
                len(snapshot.budgets), len(snapshot.models))
    refresher = asyncio.create_task(budget_snapshot.keep_fresh())
    _refreshers.add(refresher)
    refresher.add_done_callback(_refreshers.discard)


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
