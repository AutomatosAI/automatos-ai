"""
Playbook Memory Service - Stage 8 (Memory)
===========================================

Stores execution experiences in Mem0 with workspace+playbook+agent scoping.
Retrieves relevant memories for pre-execution context enhancement.

All Mem0 calls delegate to UnifiedMemoryService (shared singleton).
User IDs are built via MemoryNamespace — never raw string concatenation.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from sqlalchemy.orm import Session

from core.database.database import get_db
from core.models.core import RecipeExecution, WorkflowTemplate
from modules.memory.unified_memory_service import (
    get_unified_memory_service,
    MemoryNamespace,
)

logger = logging.getLogger(__name__)


class PlaybookMemoryService:
    """
    Stores execution experiences in Mem0 with workspace+playbook+agent scoping.
    Retrieves relevant memories for pre-execution context enhancement.

    Delegates all Mem0 operations to UnifiedMemoryService singleton.
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
        Store execution experience in Mem0 with workspace+playbook+agent scoping.

        Stores successful patterns, failed patterns, quality trends, and
        performance data as memories scoped to the workspace, playbook, and
        individual agents involved.

        Args:
            execution_id: The execution_id string (e.g. "exec-abc123def456")
            learnings: Optional learnings dict from PlaybookLearningService.analyze_execution()
            quality_data: Optional quality dict from PlaybookQualityService.assess_quality()

        Returns:
            Dict with stored_memories count, scopes used, and any errors.
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
        ns = MemoryNamespace(workspace_id=workspace_id)
        stored_memories: List[Dict[str, Any]] = []
        errors: List[str] = []

        # 1. Store playbook-level memory (workspace + playbook scope)
        playbook_user_id = ns.recipe(playbook_id)
        playbook_memory = self._build_playbook_memory(execution, learnings, quality_data)

        if playbook_memory:
            result = await self._store_memory(playbook_user_id, playbook_memory, {
                "type": "playbook_execution",
                "execution_id": execution_id,
                "playbook_id": playbook_id,
                "workspace_id": workspace_id,
                "status": execution.status,
            })
            if result.get("error"):
                errors.append(f"Playbook scope: {result['error']}")
            else:
                stored_memories.append({
                    "scope": playbook_user_id,
                    "type": "playbook_execution",
                })

            # L2: Store playbook summary in short-term memory (fire-and-forget)
            try:
                asyncio.create_task(
                    self._unified.store_short_term(
                        workspace_id=workspace_id,
                        content=playbook_memory[:1500],
                        content_type="playbook_summary",
                        importance=0.6,
                        metadata={
                            "execution_id": execution_id,
                            "playbook_id": playbook_id,
                            "status": execution.status,
                        },
                    )
                )
            except Exception:
                logger.debug(
                    "[PlaybookMemory] L2 store_short_term failed for playbook ws=%s",
                    workspace_id,
                    exc_info=True,
                )

        # 2. Store per-agent memories (workspace + playbook + agent scope)
        step_results = execution.step_results or []
        playbook_steps = playbook.steps or []

        for idx, step_result in enumerate(step_results):
            if not isinstance(step_result, dict):
                continue

            agent_id = step_result.get("agent_id")
            if not agent_id:
                # Try to get agent_id from playbook step definition
                if idx < len(playbook_steps) and isinstance(playbook_steps[idx], dict):
                    agent_id = playbook_steps[idx].get("agent_id")

            if not agent_id:
                continue

            agent_user_id = ns.recipe_agent(playbook_id, agent_id)
            agent_memory = self._build_agent_step_memory(idx, step_result, execution)

            if agent_memory:
                result = await self._store_memory(agent_user_id, agent_memory, {
                    "type": "agent_step_execution",
                    "execution_id": execution_id,
                    "playbook_id": playbook_id,
                    "agent_id": agent_id,
                    "step_index": idx,
                    "workspace_id": workspace_id,
                })
                if result.get("error"):
                    errors.append(f"Agent {agent_id} step {idx}: {result['error']}")
                else:
                    stored_memories.append({
                        "scope": agent_user_id,
                        "type": "agent_step_execution",
                        "agent_id": agent_id,
                        "step_index": idx,
                    })

                # L2: Store agent step in short-term memory (fire-and-forget)
                try:
                    agent_id_int = int(agent_id) if str(agent_id).isdigit() else None
                    asyncio.create_task(
                        self._unified.store_short_term(
                            workspace_id=workspace_id,
                            content=agent_memory[:1500],
                            content_type="playbook_summary",
                            agent_id=agent_id_int,
                            importance=0.5,
                            metadata={
                                "execution_id": execution_id,
                                "playbook_id": playbook_id,
                                "agent_id": agent_id,
                                "step_index": idx,
                            },
                        )
                    )
                except Exception:
                    logger.debug(
                        "[PlaybookMemory] L2 store_short_term failed for agent step ws=%s agent=%s",
                        workspace_id,
                        agent_id,
                        exc_info=True,
                    )

        result = {
            "execution_id": execution_id,
            "stored_at": datetime.utcnow().isoformat(),
            "stored_memories": len(stored_memories),
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
                "Stored %d memories for execution %s",
                len(stored_memories), execution_id,
            )

        return result

    async def retrieve_relevant_memories(
        self,
        playbook_id: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve relevant memories for pre-execution context enhancement.

        Searches Mem0 for past execution experiences scoped to the playbook
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
    ) -> Dict[str, Any]:
        """Store a memory in Mem0 via UnifiedMemoryService.

        Uses a conversational User->Assistant format so the Mem0 LLM
        can extract factual memories from the execution data.
        """
        messages = [
            {"role": "user", "content": "Remember the following facts about this playbook execution."},
            {"role": "assistant", "content": text},
        ]
        return await self._unified.store_long_term_messages(
            user_id=user_id, messages=messages, metadata=metadata,
        )

    def _build_playbook_memory(
        self,
        execution: RecipeExecution,
        learnings: Optional[Dict[str, Any]],
        quality_data: Optional[Dict[str, Any]],
    ) -> str:
        """Build a conversational summary of the execution for Mem0 fact extraction."""
        parts: List[str] = []

        # Execution outcome as a fact
        status = execution.status
        parts.append(
            f"The playbook execution {execution.execution_id} {status}"
        )

        if execution.error_message:
            parts.append(f"It failed with error: {execution.error_message}")

        # Quality data as facts
        if quality_data:
            score = quality_data.get("quality_score")
            grade = quality_data.get("grade")
            if score is not None:
                parts.append(f"The quality score was {score} (grade {grade})")

            bottlenecks = quality_data.get("bottlenecks", [])
            if bottlenecks:
                bn_descs = [b.get("description", "") for b in bottlenecks[:3] if b.get("description")]
                if bn_descs:
                    parts.append(f"Bottlenecks identified: {'; '.join(bn_descs)}")

        # Learnings as facts
        if learnings:
            patterns = learnings.get("patterns", [])
            if patterns:
                for p in patterns[:3]:
                    desc = p.get("description", "")
                    if desc:
                        parts.append(desc)

            suggestions = learnings.get("suggestions", [])
            if suggestions:
                for s in suggestions[:2]:
                    desc = s.get("description", "")
                    if desc:
                        parts.append(f"Suggestion: {desc}")

            perf = learnings.get("performance_metrics", {})
            if perf:
                duration = perf.get("total_duration_ms", 0)
                success_rate = perf.get("success_rate", 0)
                parts.append(
                    f"The execution took {duration / 1000:.1f} seconds "
                    f"with a {success_rate:.0%} step success rate"
                )

        # Step summary as fact
        step_results = execution.step_results or []
        if step_results:
            completed = sum(
                1 for sr in step_results
                if isinstance(sr, dict) and sr.get("status") == "completed"
            )
            failed = sum(
                1 for sr in step_results
                if isinstance(sr, dict) and sr.get("status") == "failed"
            )
            parts.append(
                f"Out of {len(step_results)} steps, {completed} completed "
                f"and {failed} failed"
            )

        return ". ".join(parts)

    def _build_agent_step_memory(
        self,
        step_index: int,
        step_result: Dict[str, Any],
        execution: RecipeExecution,
    ) -> str:
        """Build a conversational summary of a single step for Mem0 fact extraction."""
        parts: List[str] = []

        status = step_result.get("status", "unknown")
        parts.append(
            f"In execution {execution.execution_id}, step {step_index + 1} {status}"
        )

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

    def _build_retrieval_query(self, context: Optional[Dict[str, Any]]) -> str:
        """Build a search query for Mem0 based on the retrieval context."""
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
