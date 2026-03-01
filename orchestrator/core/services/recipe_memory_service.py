"""
Recipe Memory Service - Stage 8 (Memory)
=========================================

Stores execution experiences in Mem0 with workspace+recipe+agent scoping.
Retrieves relevant memories for pre-execution context enhancement.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from sqlalchemy.orm import Session

from core.database.database import get_db
from core.models.core import RecipeExecution, WorkflowTemplate
from modules.memory.integrations.mem0_client import Mem0Client

logger = logging.getLogger(__name__)


class RecipeMemoryService:
    """
    Stores execution experiences in Mem0 with workspace+recipe+agent scoping.
    Retrieves relevant memories for pre-execution context enhancement.
    """

    def __init__(self, db: Session, mem0_client: Optional[Mem0Client] = None):
        if db is None:
            raise ValueError("RecipeMemoryService requires an injected DB session")
        self.db = db
        self.mem0 = mem0_client or Mem0Client()

    def store_execution_memory(
        self,
        execution_id: str,
        learnings: Optional[Dict[str, Any]] = None,
        quality_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Store execution experience in Mem0 with workspace+recipe+agent scoping.

        Stores successful patterns, failed patterns, quality trends, and
        performance data as memories scoped to the workspace, recipe, and
        individual agents involved.

        Args:
            execution_id: The execution_id string (e.g. "exec-abc123def456")
            learnings: Optional learnings dict from RecipeLearningService.analyze_execution()
            quality_data: Optional quality dict from RecipeQualityService.assess_quality()

        Returns:
            Dict with stored_memories count, scopes used, and any errors.
        """
        execution = self.db.query(RecipeExecution).filter(
            RecipeExecution.execution_id == execution_id
        ).first()

        if not execution:
            raise ValueError(f"Execution not found: {execution_id}")

        recipe = self.db.query(WorkflowTemplate).filter(
            WorkflowTemplate.id == execution.recipe_id
        ).first()

        if not recipe:
            raise ValueError(f"Recipe not found for execution: {execution_id}")

        workspace_id = execution.workspace_id
        recipe_id = recipe.template_id or str(recipe.id)
        stored_memories: List[Dict[str, Any]] = []
        errors: List[str] = []

        # 1. Store recipe-level memory (workspace + recipe scope)
        recipe_scope = f"ws_{workspace_id}_recipe_{recipe_id}"
        recipe_memory = self._build_recipe_memory(execution, learnings, quality_data)

        if recipe_memory:
            result = self._store_memory(recipe_scope, recipe_memory, {
                "type": "recipe_execution",
                "execution_id": execution_id,
                "recipe_id": recipe_id,
                "workspace_id": str(workspace_id),
                "status": execution.status,
            })
            if result.get("error"):
                errors.append(f"Recipe scope: {result['error']}")
            else:
                stored_memories.append({
                    "scope": recipe_scope,
                    "type": "recipe_execution",
                })

        # 2. Store per-agent memories (workspace + recipe + agent scope)
        step_results = execution.step_results or []
        recipe_steps = recipe.steps or []

        for idx, step_result in enumerate(step_results):
            if not isinstance(step_result, dict):
                continue

            agent_id = step_result.get("agent_id")
            if not agent_id:
                # Try to get agent_id from recipe step definition
                if idx < len(recipe_steps) and isinstance(recipe_steps[idx], dict):
                    agent_id = recipe_steps[idx].get("agent_id")

            if not agent_id:
                continue

            agent_scope = f"ws_{workspace_id}_recipe_{recipe_id}_agent_{agent_id}"
            agent_memory = self._build_agent_step_memory(idx, step_result, execution)

            if agent_memory:
                result = self._store_memory(agent_scope, agent_memory, {
                    "type": "agent_step_execution",
                    "execution_id": execution_id,
                    "recipe_id": recipe_id,
                    "agent_id": agent_id,
                    "step_index": idx,
                    "workspace_id": str(workspace_id),
                })
                if result.get("error"):
                    errors.append(f"Agent {agent_id} step {idx}: {result['error']}")
                else:
                    stored_memories.append({
                        "scope": agent_scope,
                        "type": "agent_step_execution",
                        "agent_id": agent_id,
                        "step_index": idx,
                    })

        result = {
            "execution_id": execution_id,
            "stored_at": datetime.utcnow().isoformat(),
            "stored_memories": len(stored_memories),
            "scopes": [m["scope"] for m in stored_memories],
            "errors": errors,
        }

        logger.info(
            f"Stored {len(stored_memories)} memories for execution {execution_id} "
            f"({len(errors)} errors)"
        )

        return result

    def retrieve_relevant_memories(
        self,
        recipe_id: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve relevant memories for pre-execution context enhancement.

        Searches Mem0 for past execution experiences scoped to the recipe
        and its agents to provide context before a new execution.

        Args:
            recipe_id: The integer recipe ID (WorkflowTemplate.id)
            context: Optional context dict with keys like 'input_data', 'agent_ids'

        Returns:
            Dict with recipe_memories, agent_memories, and summary.
        """
        recipe = self.db.query(WorkflowTemplate).filter(
            WorkflowTemplate.id == recipe_id
        ).first()

        if not recipe:
            raise ValueError(f"Recipe not found: {recipe_id}")

        # Resolve workspace_id: prefer context, then execution, then recipe
        workspace_id = (context or {}).get("workspace_id") or recipe.workspace_id
        if not workspace_id:
            logger.warning(f"No workspace_id for recipe {recipe_id} (marketplace recipe?), memory retrieval may be incomplete")
        template_id = recipe.template_id or str(recipe.id)
        context = context or {}

        # 1. Retrieve recipe-level memories
        recipe_scope = f"ws_{workspace_id}_recipe_{template_id}"
        query = self._build_retrieval_query(context)
        recipe_memories = self.mem0.search(query, recipe_scope, limit=10)

        # 2. Retrieve per-agent memories for agents in the recipe steps
        agent_memories: Dict[str, List[Dict]] = {}
        recipe_steps = recipe.steps or []

        agent_ids = set()
        for step in recipe_steps:
            if isinstance(step, dict) and step.get("agent_id"):
                agent_ids.add(step["agent_id"])

        # Also include agent_ids from context if provided
        if context.get("agent_ids"):
            agent_ids.update(context["agent_ids"])

        for agent_id in agent_ids:
            agent_scope = f"ws_{workspace_id}_recipe_{template_id}_agent_{agent_id}"
            memories = self.mem0.search(query, agent_scope, limit=5)
            if memories:
                agent_memories[agent_id] = memories

        # Build summary
        total_memories = len(recipe_memories) + sum(
            len(mems) for mems in agent_memories.values()
        )

        result = {
            "recipe_id": recipe_id,
            "retrieved_at": datetime.utcnow().isoformat(),
            "recipe_memories": recipe_memories,
            "agent_memories": agent_memories,
            "total_memories": total_memories,
            "summary": self._build_memory_summary(recipe_memories, agent_memories),
        }

        logger.info(
            f"Retrieved {total_memories} memories for recipe {recipe_id} "
            f"({len(recipe_memories)} recipe, {len(agent_memories)} agents)"
        )

        return result

    def _store_memory(
        self,
        user_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Store a memory in Mem0 via the client's add method."""
        messages = [{"role": "assistant", "content": text}]
        return self.mem0.add(messages, user_id=user_id, metadata=metadata)

    def _build_recipe_memory(
        self,
        execution: RecipeExecution,
        learnings: Optional[Dict[str, Any]],
        quality_data: Optional[Dict[str, Any]],
    ) -> str:
        """Build a text summary of the execution for recipe-level memory storage."""
        parts: List[str] = []

        # Execution outcome
        parts.append(f"Recipe execution {execution.execution_id}: status={execution.status}")

        if execution.error_message:
            parts.append(f"Error: {execution.error_message}")

        # Quality data
        if quality_data:
            score = quality_data.get("quality_score", "N/A")
            grade = quality_data.get("grade", "N/A")
            parts.append(f"Quality: score={score}, grade={grade}")

            breakdown = quality_data.get("breakdown", {})
            if breakdown:
                dims = ", ".join(f"{k}={v}" for k, v in breakdown.items())
                parts.append(f"Dimensions: {dims}")

            bottlenecks = quality_data.get("bottlenecks", [])
            if bottlenecks:
                bn_descs = [b.get("description", "") for b in bottlenecks[:3]]
                parts.append(f"Bottlenecks: {'; '.join(bn_descs)}")

        # Learnings
        if learnings:
            patterns = learnings.get("patterns", [])
            if patterns:
                pattern_descs = [p.get("description", "") for p in patterns[:5]]
                parts.append(f"Patterns: {'; '.join(pattern_descs)}")

            suggestions = learnings.get("suggestions", [])
            if suggestions:
                sugg_descs = [s.get("description", "") for s in suggestions[:3]]
                parts.append(f"Suggestions: {'; '.join(sugg_descs)}")

            perf = learnings.get("performance_metrics", {})
            if perf:
                duration = perf.get("total_duration_ms", 0)
                success_rate = perf.get("success_rate", 0)
                parts.append(f"Performance: duration={duration}ms, success_rate={success_rate:.0%}")

        # Step summary
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
            parts.append(f"Steps: {len(step_results)} total, {completed} completed, {failed} failed")

        return ". ".join(parts)

    def _build_agent_step_memory(
        self,
        step_index: int,
        step_result: Dict[str, Any],
        execution: RecipeExecution,
    ) -> str:
        """Build a text summary of a single step for agent-scoped memory storage."""
        parts: List[str] = []

        status = step_result.get("status", "unknown")
        parts.append(f"Step {step_index + 1} in execution {execution.execution_id}: status={status}")

        duration = step_result.get("duration_ms")
        if duration:
            parts.append(f"Duration: {duration}ms")

        error = step_result.get("error")
        if error:
            parts.append(f"Error: {error}")

        retries = step_result.get("retries", 0)
        if retries > 0:
            parts.append(f"Required {retries} retries")

        output = step_result.get("output")
        if output:
            # Store a truncated summary of the output
            output_str = str(output)
            if len(output_str) > 200:
                output_str = output_str[:200] + "..."
            parts.append(f"Output: {output_str}")

        return ". ".join(parts)

    def _build_retrieval_query(self, context: Optional[Dict[str, Any]]) -> str:
        """Build a search query for Mem0 based on the retrieval context."""
        if not context:
            return "recipe execution patterns and results"

        parts: List[str] = []

        input_data = context.get("input_data")
        if input_data and isinstance(input_data, dict):
            keys = list(input_data.keys())[:5]
            parts.append(f"execution with inputs: {', '.join(keys)}")

        if context.get("focus"):
            parts.append(str(context["focus"]))

        if not parts:
            return "recipe execution patterns and results"

        return "; ".join(parts)

    def _build_memory_summary(
        self,
        recipe_memories: List[Dict],
        agent_memories: Dict[str, List[Dict]],
    ) -> str:
        """Build a human-readable summary of retrieved memories."""
        parts: List[str] = []

        if recipe_memories:
            parts.append(f"{len(recipe_memories)} recipe-level memories found")
            # Extract key themes from recent memories
            recent = recipe_memories[:3]
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
