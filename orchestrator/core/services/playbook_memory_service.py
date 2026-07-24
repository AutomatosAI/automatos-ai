"""
Playbook Memory Service - Stage 8 (Memory)
===========================================

Stores execution experiences in durable memory with workspace+playbook+agent
scoping. Retrieves relevant memories for pre-execution context enhancement.

All durable-store calls delegate to UnifiedMemoryService (shared singleton).
User IDs are built via MemoryNamespace — never raw string concatenation.

PRD-187 S2 — the write side no longer floods: outcomes are planned by a PURE
layer (``plan_execution_records``) that emits ONE playbook-level record per
run plus per-step records only for failures whose error class DIFFERS from
the run-level one, and every record passes signature-based repeat-suppression
(``record_recurrence``): the same failure recurring daily becomes ONE memory
whose recurrence count climbs — itself a signal — instead of 2+N new rows
per run. Mirrors ``tool_outcome_capture`` (same error classifier, same
bounded-registry shape); the old writer double-wrote every run, unconditionally.
"""

import asyncio
import hashlib
import logging
from collections import OrderedDict
from datetime import datetime
from typing import Dict, List, Any, Optional

from sqlalchemy.orm import Session

from core.database.database import get_db
from core.models.core import RecipeExecution, WorkflowTemplate
from modules.memory.tool_outcome_capture import _classify_error
from modules.memory.unified_memory_service import (
    get_unified_memory_service,
    MemoryNamespace,
)

logger = logging.getLogger(__name__)

PLAYBOOK_OUTCOME_TYPE = "playbook_summary"

# Bounded in-process recurrence registry: outcome-signature hash → times seen.
# Same shape as tool_outcome_capture's _SEEN_HASHES, but counting — the count
# is the "this exact failure keeps happening" signal Auto can act on.
_RECURRENCE: "OrderedDict[str, int]" = OrderedDict()
_RECURRENCE_MAX = 512


def _signature_hash(workspace_id: str, playbook_id: str, signature: str) -> str:
    raw = f"{workspace_id}|{playbook_id}|{signature}".encode("utf-8", "ignore")
    return hashlib.sha256(raw).hexdigest()


def record_recurrence(outcome_hash: str) -> int:
    """Bump and return how many times this outcome signature has been seen.

    1 means first occurrence (write it); >1 means a repeat (suppress the
    write, surface the count). Bounded so a long-running worker never grows
    without limit.
    """
    count = _RECURRENCE.get(outcome_hash, 0) + 1
    _RECURRENCE[outcome_hash] = count
    _RECURRENCE.move_to_end(outcome_hash)
    while len(_RECURRENCE) > _RECURRENCE_MAX:
        _RECURRENCE.popitem(last=False)
    return count


def reset_recurrence_registry() -> None:
    """Testing seam — clear the in-process registry."""
    _RECURRENCE.clear()


def plan_execution_records(
    *,
    workspace_id: str,
    playbook_id: str,
    execution_id: str,
    status: str,
    error_message: Optional[str],
    learnings: Optional[Dict[str, Any]],
    quality_data: Optional[Dict[str, Any]],
    step_results: List[Any],
    step_agent_ids: List[Any],
) -> List[Dict[str, Any]]:
    """PURE: one execution → the outcome records worth remembering.

    Exactly one playbook-level record (failure, or success), plus a per-step
    record ONLY for a failed step whose error class differs from the run-level
    record's class — the same failure is never memorised twice in one run.
    Each record carries a signature-based ``outcome_hash`` for suppression.
    """
    records: List[Dict[str, Any]] = []
    failed = status == "failed" or bool(error_message)
    run_error_class = _classify_error(error_message or "") if failed else ""
    signature = f"fail:{run_error_class}" if failed else "success"

    fact = _build_playbook_fact(
        execution_id=execution_id, status=status, error_message=error_message,
        learnings=learnings, quality_data=quality_data, step_results=step_results,
    )
    records.append({
        "scope": "playbook",
        "fact": fact,
        "importance": 0.6 if failed else 0.5,
        "outcome_hash": _signature_hash(workspace_id, playbook_id, signature),
        "metadata": {
            "type": "playbook_execution",
            "execution_id": execution_id,
            "playbook_id": playbook_id,
            "workspace_id": workspace_id,
            "status": status,
            "error_class": run_error_class,
        },
    })

    for idx, step_result in enumerate(step_results):
        if not isinstance(step_result, dict):
            continue
        step_error = step_result.get("error")
        if step_result.get("status") != "failed" and not step_error:
            continue  # successful steps fold into the run-level fact
        step_class = _classify_error(str(step_error or ""))
        if failed and step_class == run_error_class:
            continue  # same failure as the run record — one memory, not two
        agent_id = step_agent_ids[idx] if idx < len(step_agent_ids) else None
        if not agent_id:
            continue
        records.append({
            "scope": "step",
            "agent_id": agent_id,
            "step_index": idx,
            "fact": _build_step_fact(idx, step_result, execution_id),
            "importance": 0.6,
            "outcome_hash": _signature_hash(
                workspace_id, playbook_id, f"step-fail:{agent_id}:{step_class}"
            ),
            "metadata": {
                "type": "agent_step_execution",
                "execution_id": execution_id,
                "playbook_id": playbook_id,
                "workspace_id": workspace_id,
                "agent_id": agent_id,
                "step_index": idx,
                "error_class": step_class,
            },
        })

    return records


def _build_playbook_fact(
    *,
    execution_id: str,
    status: str,
    error_message: Optional[str],
    learnings: Optional[Dict[str, Any]],
    quality_data: Optional[Dict[str, Any]],
    step_results: List[Any],
) -> str:
    parts: List[str] = [f"The playbook execution {execution_id} {status}"]

    if error_message:
        parts.append(f"It failed with error: {error_message}")

    if quality_data:
        score = quality_data.get("quality_score")
        grade = quality_data.get("grade")
        if score is not None:
            parts.append(f"The quality score was {score} (grade {grade})")
        bottlenecks = quality_data.get("bottlenecks", [])
        bn_descs = [b.get("description", "") for b in bottlenecks[:3] if b.get("description")]
        if bn_descs:
            parts.append(f"Bottlenecks identified: {'; '.join(bn_descs)}")

    if learnings:
        for p in learnings.get("patterns", [])[:3]:
            if p.get("description"):
                parts.append(p["description"])
        for s in learnings.get("suggestions", [])[:2]:
            if s.get("description"):
                parts.append(f"Suggestion: {s['description']}")
        perf = learnings.get("performance_metrics", {})
        if perf:
            duration = perf.get("total_duration_ms", 0)
            success_rate = perf.get("success_rate", 0)
            parts.append(
                f"The execution took {duration / 1000:.1f} seconds "
                f"with a {success_rate:.0%} step success rate"
            )

    if step_results:
        completed = sum(
            1 for sr in step_results
            if isinstance(sr, dict) and sr.get("status") == "completed"
        )
        failed_n = sum(
            1 for sr in step_results
            if isinstance(sr, dict) and sr.get("status") == "failed"
        )
        parts.append(
            f"Out of {len(step_results)} steps, {completed} completed "
            f"and {failed_n} failed"
        )

    return ". ".join(parts)


def _build_step_fact(step_index: int, step_result: Dict[str, Any], execution_id: str) -> str:
    parts: List[str] = [
        f"In execution {execution_id}, step {step_index + 1} "
        f"{step_result.get('status', 'unknown')}"
    ]
    duration = step_result.get("duration_ms")
    if duration:
        parts.append(f"It took {duration / 1000:.1f} seconds")
    error = step_result.get("error")
    if error:
        parts.append(f"It failed with error: {error}")
    retries = step_result.get("retries", 0)
    if retries > 0:
        parts.append(f"It required {retries} retries before completing")
    output = step_result.get("output")
    if output:
        output_str = str(output)
        if len(output_str) > 300:
            output_str = output_str[:300] + "..."
        parts.append(f"The step produced: {output_str}")
    return ". ".join(parts)


class PlaybookMemoryService:
    """
    Stores execution experiences in durable memory with workspace+playbook+agent
    scoping. Retrieves relevant memories for pre-execution context enhancement.

    Delegates all durable-store operations to UnifiedMemoryService singleton.
    """

    def __init__(self, db: Session):
        if db is None:
            raise ValueError("PlaybookMemoryService requires an injected DB session")
        self.db = db
        self._unified = get_unified_memory_service()

    async def store_execution_memory(
        self,
        execution_id: str,
        learnings: Optional[Dict[str, Any]] = None,
        quality_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Store an execution's outcome records (PRD-187 S2: planned, de-duped).

        The pure planner emits one playbook-level record plus per-step records
        only for distinct failure classes; each record is repeat-suppressed by
        its outcome signature, so a recurring identical failure becomes ONE
        memory whose recurrence count climbs instead of new rows per run.

        Args:
            execution_id: The execution_id string (e.g. "exec-abc123def456")
            learnings: Optional learnings dict from PlaybookLearningService.analyze_execution()
            quality_data: Optional quality dict from PlaybookQualityService.assess_quality()

        Returns:
            Dict with stored_memories count, suppressed repeats (with counts),
            scopes used, and any errors.
        """
        execution = self.db.query(RecipeExecution).filter(
            RecipeExecution.execution_id == execution_id
        ).first()

        if not execution:
            raise ValueError(f"Execution not found: {execution_id}")

        playbook = self.db.query(WorkflowTemplate).filter(
            WorkflowTemplate.id == execution.recipe_id
        ).first()

        if not playbook:
            raise ValueError(f"Playbook not found for execution: {execution_id}")

        workspace_id = str(execution.workspace_id)
        playbook_id = playbook.template_id or str(playbook.id)
        step_results = execution.step_results or []
        playbook_steps = playbook.steps or []

        step_agent_ids: List[Any] = []
        for idx, step_result in enumerate(step_results):
            agent_id = step_result.get("agent_id") if isinstance(step_result, dict) else None
            if not agent_id and idx < len(playbook_steps) and isinstance(playbook_steps[idx], dict):
                agent_id = playbook_steps[idx].get("agent_id")
            step_agent_ids.append(agent_id)

        records = plan_execution_records(
            workspace_id=workspace_id,
            playbook_id=playbook_id,
            execution_id=execution_id,
            status=execution.status,
            error_message=execution.error_message,
            learnings=learnings,
            quality_data=quality_data,
            step_results=step_results,
            step_agent_ids=step_agent_ids,
        )

        return await self.write_records(
            records, workspace_id=workspace_id, playbook_id=playbook_id,
            execution_id=execution_id,
        )

    async def write_records(
        self,
        records: List[Dict[str, Any]],
        *,
        workspace_id: str,
        playbook_id: str,
        execution_id: str,
    ) -> Dict[str, Any]:
        """Suppress repeats, then persist first-occurrence records (L3 + L2).

        Separated from the planner so tests drive it with plain dicts and a
        fake unified service — no DB.
        """
        ns = MemoryNamespace(workspace_id=workspace_id)
        stored_memories: List[Dict[str, Any]] = []
        suppressed: List[Dict[str, Any]] = []
        errors: List[str] = []

        for record in records:
            recurrence = record_recurrence(record["outcome_hash"])
            if recurrence > 1:
                suppressed.append({
                    "scope": record["scope"],
                    "recurrence": recurrence,
                    "error_class": record["metadata"].get("error_class", ""),
                })
                logger.warning(
                    "[PlaybookMemory] Repeat outcome suppressed (seen %dx): "
                    "playbook=%s scope=%s class=%s",
                    recurrence, playbook_id, record["scope"],
                    record["metadata"].get("error_class", "success"),
                )
                continue

            if record["scope"] == "playbook":
                user_id = ns.recipe(playbook_id)
            else:
                user_id = ns.recipe_agent(playbook_id, record["agent_id"])

            metadata = {**record["metadata"], "recurrence": recurrence}
            result = await self._store_memory(
                user_id, record["fact"], metadata, workspace_id=workspace_id,
            )
            if result.get("error"):
                errors.append(f"{record['scope']} scope: {result['error']}")
                continue
            stored_memories.append({"scope": user_id, "type": record["metadata"]["type"]})

            # L2 row — one per first-occurrence record (fire-and-forget)
            try:
                agent_raw = record.get("agent_id")
                agent_id_int = int(agent_raw) if str(agent_raw).isdigit() else None
                asyncio.create_task(
                    self._unified.store_short_term(
                        workspace_id=workspace_id,
                        content=record["fact"][:1500],
                        content_type=PLAYBOOK_OUTCOME_TYPE,
                        agent_id=agent_id_int,
                        importance=record["importance"],
                        metadata=metadata,
                    )
                )
            except Exception:
                logger.debug(
                    "[PlaybookMemory] L2 store_short_term failed ws=%s scope=%s",
                    workspace_id, record["scope"], exc_info=True,
                )

        result = {
            "execution_id": execution_id,
            "stored_at": datetime.utcnow().isoformat(),
            "stored_memories": len(stored_memories),
            "suppressed": suppressed,
            "scopes": [m["scope"] for m in stored_memories],
            "errors": errors,
        }

        if errors:
            logger.warning(
                "Stored %d memories for execution %s (%d errors): %s",
                len(stored_memories), execution_id, len(errors), errors,
            )
        else:
            logger.info(
                "Stored %d memories for execution %s (%d repeats suppressed)",
                len(stored_memories), execution_id, len(suppressed),
            )

        return result

    async def retrieve_relevant_memories(
        self,
        playbook_id: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve relevant memories for pre-execution context enhancement.

        Searches durable memory for past execution experiences scoped to the playbook
        and its agents to provide context before a new execution.

        Args:
            playbook_id: The integer playbook ID (WorkflowTemplate.id)
            context: Optional context dict with keys like 'input_data', 'agent_ids'

        Returns:
            Dict with playbook_memories, agent_memories, and summary.
        """
        playbook = self.db.query(WorkflowTemplate).filter(
            WorkflowTemplate.id == playbook_id
        ).first()

        if not playbook:
            raise ValueError(f"Playbook not found: {playbook_id}")

        # Resolve workspace_id: prefer context, then playbook
        workspace_id = str((context or {}).get("workspace_id") or playbook.workspace_id or "")
        if not workspace_id:
            logger.warning(
                "No workspace_id for playbook %d (marketplace playbook?), memory retrieval may be incomplete",
                playbook_id,
            )
        template_id = playbook.template_id or str(playbook.id)
        context = context or {}
        ns = MemoryNamespace(workspace_id=workspace_id)

        # 1. Retrieve playbook-level memories
        playbook_user_id = ns.recipe(template_id)
        query = self._build_retrieval_query(context)
        playbook_memories = await self._unified.search_long_term_scoped(
            user_id=playbook_user_id, query=query, limit=10,
            workspace_id=workspace_id,
        )

        # 2. Retrieve per-agent memories for agents in the playbook steps
        agent_memories: Dict[str, List[Dict]] = {}
        playbook_steps = playbook.steps or []

        agent_ids = set()
        for step in playbook_steps:
            if isinstance(step, dict) and step.get("agent_id"):
                agent_ids.add(step["agent_id"])

        # Also include agent_ids from context if provided
        if context.get("agent_ids"):
            agent_ids.update(context["agent_ids"])

        # Search all agents concurrently
        async def _search_agent(aid):
            agent_user_id = ns.recipe_agent(template_id, aid)
            return aid, await self._unified.search_long_term_scoped(
                user_id=agent_user_id, query=query, limit=5,
                workspace_id=workspace_id,
            )

        if agent_ids:
            agent_results = await asyncio.gather(
                *[_search_agent(aid) for aid in agent_ids]
            )
            for aid, memories in agent_results:
                if memories:
                    agent_memories[aid] = memories

        # Build summary
        total_memories = len(playbook_memories) + sum(
            len(mems) for mems in agent_memories.values()
        )

        result = {
            "playbook_id": playbook_id,
            "retrieved_at": datetime.utcnow().isoformat(),
            "playbook_memories": playbook_memories,
            "agent_memories": agent_memories,
            "total_memories": total_memories,
            "summary": self._build_memory_summary(playbook_memories, agent_memories),
        }

        logger.info(
            "Retrieved %d memories for playbook %d (%d playbook, %d agents)",
            total_memories, playbook_id, len(playbook_memories), len(agent_memories),
        )

        return result

    async def _store_memory(
        self,
        user_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        workspace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Store a memory in the durable store via UnifiedMemoryService.

        Keeps the conversational User->Assistant format the store's readers
        already parse; the text is stored verbatim.
        """
        messages = [
            {"role": "user", "content": "Remember the following facts about this playbook execution."},
            {"role": "assistant", "content": text},
        ]
        return await self._unified.store_long_term_messages(
            user_id=user_id, messages=messages, metadata=metadata,
            workspace_id=workspace_id,
        )

    def _build_retrieval_query(self, context: Optional[Dict[str, Any]]) -> str:
        """Build a search query for durable memory based on the retrieval context."""
        if not context:
            return "playbook execution patterns and results"

        parts: List[str] = []

        input_data = context.get("input_data")
        if input_data and isinstance(input_data, dict):
            keys = list(input_data.keys())[:5]
            parts.append(f"execution with inputs: {', '.join(keys)}")

        if context.get("focus"):
            parts.append(str(context["focus"]))

        if not parts:
            return "playbook execution patterns and results"

        return "; ".join(parts)

    def _build_memory_summary(
        self,
        playbook_memories: List[Dict],
        agent_memories: Dict[str, List[Dict]],
    ) -> str:
        """Build a human-readable summary of retrieved memories."""
        parts: List[str] = []

        if playbook_memories:
            parts.append(f"{len(playbook_memories)} playbook-level memories found")
            # Extract key themes from recent memories
            recent = playbook_memories[:3]
            for mem in recent:
                content = mem.get("memory", "")
                if content:
                    parts.append(f"- {content[:100]}")

        if agent_memories:
            for agent_id, memories in agent_memories.items():
                parts.append(f"Agent {agent_id}: {len(memories)} memories")

        if not parts:
            return "No relevant memories found"

        return "; ".join(parts)
