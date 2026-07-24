"""
Agent-Match Semantic Signals — PRD-164 S2 (Q21)
================================================

Computes the two Q21 signals the (one and only) AgentMatcher blends into its
score. This module is NOT a matcher — it produces per-agent signal maps the
matcher consumes:

  * capability-card similarity — cosine between the task text and each
    agent's capability card (PRD-64 ``agents.semantic_embedding``: JSONB
    float columns + python cosine, the platform's model-level embedding
    convention; cards are populated at boot and on agent save);
  * live field signal — PRD-166 workspace field recall
    (``query_workspace``), crediting agents whose recently-contributed
    patterns resonate with the task. The task embedding is REUSED via
    ``query_vector`` so dispatch stays at one embedding call (Q21).

Everything is best-effort and fail-open: any backend failure yields absent
signals and the matcher degrades to lexical-only scoring — never to a failed
dispatch. The dispatch path bridges sync→async via a helper-thread event loop
(tool_router's ``_run_coroutine_blocking`` idiom), so backend clients are
created FRESH per call and never outlive their loop.
"""

import asyncio
import concurrent.futures
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence
from uuid import UUID

from config import Config
from core.models.core import Agent
from core.models.orchestration import OrchestrationTask

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SemanticSignals:
    """PRD-164 S2: per-dispatch semantic signals, computed once per task.

    ``similarity_by_agent`` — cosine similarity between the task text and each
    agent's capability card (PRD-64 ``agents.semantic_embedding``). Agents
    missing from a NON-empty map have no card and score neutral (0.5).
    An empty map means the component is absent and renormalized away.

    ``field_by_agent`` — normalized live field signal (PRD-166): each agent's
    share of resonant workspace-field patterns for this task, in [0, 1].
    """

    similarity_by_agent: Dict[int, float]
    field_by_agent: Dict[int, float]


async def compute_signals_for_tasks(
    tasks: Sequence[OrchestrationTask],
    agents: Sequence[Agent],
    workspace_id: Optional[UUID],
) -> Dict[Any, SemanticSignals]:
    """Compute Q21 signals for each task. Returns ``{task.id:
    SemanticSignals}`` — tasks with no usable signal are simply absent.
    Never raises.
    """
    results: Dict[Any, SemanticSignals] = {}

    cards: Dict[int, Any] = {
        a.id: a.semantic_embedding
        for a in agents
        if getattr(a, "semantic_embedding", None)
    }

    backend = getattr(Config, "SHARED_CONTEXT_BACKEND", "vector_field")
    field_ctx = None
    if workspace_id is not None and backend == "vector_field":
        try:
            # Lazy + fresh per call: the dispatch path runs this in a
            # short-lived helper-thread event loop, so the async Qdrant
            # client must never outlive its loop (no factory singleton).
            from modules.context.adapters.vector_field import (
                VectorFieldSharedContext,
            )
            field_ctx = VectorFieldSharedContext()
        except Exception:
            logger.debug(
                "[MatchSignals] field backend unavailable (fail-open)",
                exc_info=True,
            )

    if not cards and field_ctx is None:
        return results

    cosine = None
    if cards:
        try:
            from core.routing.semantic_indexer import cosine_similarity as cosine
        except Exception:
            logger.debug(
                "[MatchSignals] cosine unavailable (fail-open)", exc_info=True,
            )

    embedder = None
    try:
        from core.llm.embedding_manager import EmbeddingManager
        embedder = EmbeddingManager()
    except Exception:
        logger.debug(
            "[MatchSignals] embedding manager unavailable (fail-open)",
            exc_info=True,
        )

    top_k = int(getattr(Config, "FIELD_QUERY_TOP_K", 10))

    for task in tasks:
        text = _task_semantic_text(task)
        if not text:
            continue

        vec = None
        if embedder is not None:
            try:
                vec = await embedder.generate_embedding(text)
            except Exception:
                logger.debug(
                    "[MatchSignals] task embedding failed (fail-open)",
                    exc_info=True,
                )

        similarity: Dict[int, float] = {}
        if vec and cards and cosine is not None:
            for agent_id, card in cards.items():
                try:
                    similarity[agent_id] = round(
                        max(0.0, float(cosine(vec, card))), 4
                    )
                except Exception:
                    continue

        field_signal: Dict[int, float] = {}
        if field_ctx is not None:
            try:
                patterns = await field_ctx.query_workspace(
                    str(workspace_id), text, agent_id=0, top_k=top_k,
                    query_vector=vec,
                )
                field_signal = _field_signal_from_patterns(patterns)
            except Exception:
                logger.debug(
                    "[MatchSignals] field signal failed (fail-open)",
                    exc_info=True,
                )
                # A dead backend stays dead for this batch — don't retry
                # per task.
                field_ctx = None

        if similarity or field_signal:
            results[task.id] = SemanticSignals(
                similarity_by_agent=similarity,
                field_by_agent=field_signal,
            )

    if results:
        logger.info(
            "[MatchSignals] semantic signals computed for %d/%d task(s) "
            "(cards=%d, field=%s)",
            len(results), len(tasks), len(cards), field_ctx is not None,
        )
    return results


def compute_semantic_signals_sync(
    *,
    task: OrchestrationTask,
    agents: Sequence[Agent],
    workspace_id: Optional[UUID],
) -> Optional[SemanticSignals]:
    """Sync bridge for the dispatch path (Q21: one embedding call per
    dispatch is acceptable). Mirrors tool_router's
    ``_run_coroutine_blocking`` idiom: inside a running loop the coroutine
    ships to a helper thread that owns its own loop. Bounded by
    ``Config.AGENT_MATCH_SIGNAL_TIMEOUT_SECONDS``; ANY failure returns None
    and dispatch proceeds lexical-only.
    """
    try:
        timeout = float(
            getattr(Config, "AGENT_MATCH_SIGNAL_TIMEOUT_SECONDS", 10.0)
        )
        coro = asyncio.wait_for(
            compute_signals_for_tasks([task], agents, workspace_id),
            timeout=timeout,
        )
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            by_task = asyncio.run(coro)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                by_task = pool.submit(asyncio.run, coro).result()
        return by_task.get(task.id)
    except Exception:
        logger.debug(
            "[MatchSignals] semantic signal computation failed (fail-open)",
            exc_info=True,
        )
        return None


def _task_semantic_text(task: OrchestrationTask) -> str:
    """The text embedded once per dispatch (Q21) — role + title + description."""
    parts = [
        str(getattr(task, "agent_role", None) or ""),
        str(getattr(task, "title", None) or ""),
        str(getattr(task, "description", None) or ""),
    ]
    return " ".join(p for p in parts if p).strip()


def _field_signal_from_patterns(
    patterns: Optional[Sequence[Dict[str, Any]]],
) -> Dict[int, float]:
    """Aggregate PRD-166 field-query results into a per-agent signal in [0, 1].

    Each agent's resonance scores are summed and normalized by the strongest
    contributor. Seeder injections (agent_id 0) and non-agent patterns never
    credit an agent.
    """
    totals: Dict[int, float] = {}
    for p in patterns or []:
        if not isinstance(p, dict):
            continue
        agent_id = p.get("agent_id")
        score = p.get("score", 0.0)
        if not isinstance(agent_id, int) or agent_id <= 0:
            continue
        if not isinstance(score, (int, float)) or score <= 0:
            continue
        totals[agent_id] = totals.get(agent_id, 0.0) + float(score)

    if not totals:
        return {}
    peak = max(totals.values())
    if peak <= 0:
        return {}
    return {agent_id: round(value / peak, 4) for agent_id, value in totals.items()}
