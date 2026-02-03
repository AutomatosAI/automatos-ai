"""
Recipe Direct Executor
======================

Simple step-by-step executor for Starter Plan recipes.
Bypasses the 9-stage pipeline — executes recipe steps sequentially,
each step calling its assigned agent with filtered Composio actions.

This is the "fast food kitchen" — no AI reasoning overhead,
just execute steps in order with the right tools.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core.database.database import get_db, SessionLocal
from core.models import Agent
from core.models.core import RecipeExecution, WorkflowTemplate as WorkflowRecipe
from core.models.composio_cache import AgentAppAssignment, ComposioActionCache

logger = logging.getLogger(__name__)


async def execute_recipe_direct(
    recipe_execution_id: str,
    recipe_id: int,
    workspace_id: UUID,
    input_data: dict,
    db_url: Optional[str] = None
):
    """
    Execute a recipe directly — step by step, no 9-stage pipeline.

    For each step:
    1. Load assigned agent
    2. Build prompt with previous step results as context
    3. Filter Composio actions for step's intent
    4. Call agent via AgentFactory.execute_with_prompt()
    5. Store step result in RecipeExecution.step_results JSONB
    6. Publish progress event
    7. Handle errors per step.error_handling config

    Args:
        recipe_execution_id: The execution_id string (e.g. "exec-abc123")
        recipe_id: Integer PK of the recipe (WorkflowTemplate.id)
        workspace_id: UUID of the workspace
        input_data: User-provided input data
        db_url: Optional database URL (uses default SessionLocal if not provided)
    """
    # Create a fresh DB session for this async task
    db = SessionLocal()

    try:
        logger.info(f"[recipe_direct] Starting execution {recipe_execution_id} for recipe {recipe_id}")

        # Load recipe and execution
        recipe = db.query(WorkflowRecipe).filter(WorkflowRecipe.id == recipe_id).first()
        if not recipe:
            await _fail_execution(db, recipe_execution_id, "Recipe not found")
            return

        execution = db.query(RecipeExecution).filter(
            RecipeExecution.execution_id == recipe_execution_id
        ).first()
        if not execution:
            logger.error(f"[recipe_direct] Execution record not found: {recipe_execution_id}")
            return

        # Mark as running
        execution.status = 'running'
        execution.current_step = 0
        execution.step_results = []
        db.commit()

        # Load and sort steps
        steps = sorted(recipe.steps or [], key=lambda s: s.get('order', 0))
        if not steps:
            await _fail_execution(db, recipe_execution_id, "Recipe has no steps")
            return

        total_steps = len(steps)
        logger.info(f"[recipe_direct] Recipe '{recipe.name}' has {total_steps} steps")

        # Validate agents exist
        agent_ids = list({s.get('agent_id') for s in steps if s.get('agent_id')})
        agents = db.query(Agent).filter(
            Agent.id.in_(agent_ids),
            Agent.workspace_id == workspace_id
        ).all()
        agent_map = {a.id: a for a in agents}

        missing_agents = [aid for aid in agent_ids if aid not in agent_map]
        if missing_agents:
            await _fail_execution(db, recipe_execution_id, f"Agents not found: {missing_agents}")
            return

        # Initialize AgentFactory (lazy import to avoid circular deps)
        from modules.agents.factory.agent_factory import AgentFactory
        factory = AgentFactory(db_session=db)

        # Execute each step sequentially
        step_results: List[Dict[str, Any]] = []
        previous_output: Optional[str] = None
        execution_start = time.time()

        for idx, step in enumerate(steps):
            step_id = step.get('step_id', f'step-{idx + 1}')
            step_order = step.get('order', idx + 1)
            agent_id = step.get('agent_id')
            prompt_template = step.get('prompt_template', '')
            error_handling = step.get('error_handling', 'stop')
            max_retries = step.get('max_retries', 1)
            pass_to = step.get('pass_to')

            agent = agent_map.get(agent_id)
            agent_name = agent.name if agent else f"Agent {agent_id}"
            model_cfg = (agent.model_config or {}) if agent else {}

            logger.info(f"[recipe_direct] Step {step_order}/{total_steps}: {agent_name} — {prompt_template[:80]}")

            # Update execution progress
            execution.current_step = idx + 1
            db.commit()

            step_start = time.time()
            step_result: Dict[str, Any] = {
                "step_id": step_id,
                "order": step_order,
                "agent_id": agent_id,
                "agent_name": agent_name,
                "status": "running",
                "output": None,
                "tool_calls": [],
                "duration_ms": 0,
                "tokens_used": 0,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": None,
                "error": None,
                "retries": 0,
            }

            # Build the prompt: inject previous step results as context
            resolved_prompt = _resolve_prompt(prompt_template, input_data, previous_output, step_results)

            # Filter Composio actions for this step's intent
            filtered_action_names = await _filter_actions_for_step(
                db, agent_id, resolved_prompt, workspace_id
            )

            # Build a system prompt hint for filtered actions
            action_hint = ""
            if filtered_action_names:
                action_hint = (
                    f"\n\nFor this task, prefer these Composio actions: "
                    f"{', '.join(filtered_action_names[:10])}. "
                    f"Only use actions from this list unless absolutely necessary."
                )

            # Execute with retries
            attempt = 0
            success = False
            last_error = None

            while attempt <= max_retries and not success:
                if attempt > 0:
                    logger.info(f"[recipe_direct] Retrying step {step_order}, attempt {attempt + 1}")
                    step_result["retries"] = attempt

                try:
                    result = await factory.execute_with_prompt(
                        agent=agent_id,
                        prompt=resolved_prompt + action_hint,
                        system_prompt=None,  # Let agent use its default system prompt with skills
                        context={"recipe_execution_id": recipe_execution_id, "step": step_order},
                        use_memory=False,  # Fresh execution, no memory carry
                        max_retries=1,  # Single attempt per retry cycle (we manage retries)
                        enable_actions=True,
                        required_tools=["research"],  # Composio tools injected by factory
                    )

                    if result.get("status") == "success":
                        step_result["status"] = "success"
                        step_result["output"] = result.get("result", "")
                        step_result["tokens_used"] = result.get("execution", {}).get("tokens_used", 0)

                        # Extract tool calls from execution metadata
                        tool_calls_raw = result.get("execution", {}).get("tool_calls", [])
                        step_result["tool_calls"] = _normalize_tool_calls(tool_calls_raw)

                        success = True
                        previous_output = step_result["output"]
                        logger.info(f"[recipe_direct] Step {step_order} succeeded ({step_result['tokens_used']} tokens)")
                    else:
                        last_error = result.get("error", "Agent returned non-success status")
                        logger.warning(f"[recipe_direct] Step {step_order} failed: {last_error}")

                except Exception as e:
                    last_error = str(e)
                    logger.error(f"[recipe_direct] Step {step_order} exception: {e}", exc_info=True)

                attempt += 1

            # Finalize step result
            step_result["duration_ms"] = int((time.time() - step_start) * 1000)
            step_result["completed_at"] = datetime.now(timezone.utc).isoformat()

            if not success:
                step_result["status"] = "failed"
                step_result["error"] = last_error

                # Handle error based on step config
                if error_handling == 'stop':
                    step_results.append(step_result)
                    _persist_step_results(db, execution, step_results)
                    await _fail_execution(
                        db, recipe_execution_id,
                        f"Step {step_order} failed: {last_error}",
                        step_results=step_results
                    )
                    return
                elif error_handling == 'skip':
                    logger.info(f"[recipe_direct] Skipping failed step {step_order} (error_handling=skip)")
                    step_results.append(step_result)
                    _persist_step_results(db, execution, step_results)
                    continue
                else:
                    # 'retry' already handled in the loop above; if still failed, stop
                    step_results.append(step_result)
                    _persist_step_results(db, execution, step_results)
                    await _fail_execution(
                        db, recipe_execution_id,
                        f"Step {step_order} failed after {attempt} attempts: {last_error}",
                        step_results=step_results
                    )
                    return

            step_results.append(step_result)
            _persist_step_results(db, execution, step_results)

        # All steps completed successfully
        total_duration = int((time.time() - execution_start) * 1000)
        total_tokens = sum(s.get("tokens_used", 0) for s in step_results)

        execution.status = 'completed'
        execution.completed_at = datetime.now(timezone.utc)
        execution.step_results = step_results
        execution.output_data = {
            "final_output": previous_output,
            "total_duration_ms": total_duration,
            "total_tokens": total_tokens,
            "steps_completed": len(step_results),
        }
        db.commit()

        logger.info(
            f"[recipe_direct] Execution {recipe_execution_id} COMPLETED — "
            f"{len(step_results)} steps, {total_duration}ms, {total_tokens} tokens"
        )

        # Optional: fire learning hook if auto_learn enabled
        exec_config = recipe.execution_config or {}
        if exec_config.get('auto_learn', False):
            try:
                from core.services.recipe_learning_service import RecipeLearningService
                learning_svc = RecipeLearningService(db=db)
                learning_svc.analyze_execution(recipe_execution_id)
                logger.info(f"[recipe_direct] Auto-learning completed for {recipe_execution_id}")
            except Exception as e:
                logger.warning(f"[recipe_direct] Auto-learning failed (non-blocking): {e}")

    except Exception as e:
        logger.error(f"[recipe_direct] Fatal error in execution {recipe_execution_id}: {e}", exc_info=True)
        try:
            await _fail_execution(db, recipe_execution_id, str(e))
        except Exception:
            pass
    finally:
        db.close()


def _resolve_prompt(
    template: str,
    input_data: dict,
    previous_output: Optional[str],
    step_results: List[dict]
) -> str:
    """
    Resolve a step's prompt template with variable substitution.

    Supports:
    - {input.field_name} — from user input_data
    - {previous_output} — output from the previous step
    - {step_N_output} — output from step N (by order)
    """
    resolved = template

    # Substitute {input.xxx} placeholders
    if input_data:
        for key, value in input_data.items():
            resolved = resolved.replace(f"{{input.{key}}}", str(value))

    # Substitute {previous_output}
    if previous_output:
        resolved = resolved.replace("{previous_output}", previous_output)
        # Also inject as context even if not explicitly referenced
        if "{previous_output}" not in template and previous_output:
            resolved = f"Context from previous step:\n{previous_output}\n\n{resolved}"

    # Substitute {step_N_output} for specific step references
    for sr in step_results:
        order = sr.get("order", 0)
        output = sr.get("output", "")
        if output:
            resolved = resolved.replace(f"{{step_{order}_output}}", output)

    return resolved


async def _filter_actions_for_step(
    db: Session,
    agent_id: int,
    prompt: str,
    workspace_id: UUID
) -> List[str]:
    """
    Filter Composio actions for a step's intent.

    Strategy:
    1. Try ActionCapabilityFilter (uses classified metadata)
    2. Fallback: keyword-based filtering from ComposioActionCache
    3. Fallback: return empty list (agent gets all its default actions)
    """
    try:
        # Get agent's assigned apps
        assignments = db.query(AgentAppAssignment).filter(
            AgentAppAssignment.agent_id == agent_id
        ).all()
        if not assignments:
            return []

        app_names = [a.app_name.upper() for a in assignments if a.app_name]
        if not app_names:
            return []

        # Try ActionCapabilityFilter first
        try:
            from modules.tools.services.action_capability_filter import ActionCapabilityFilter
            from modules.tools.capabilities.models import ComposioActionMetadata

            # Check if metadata table has data
            has_metadata = db.query(ComposioActionMetadata.action_id).limit(1).first()
            if has_metadata:
                acf = ActionCapabilityFilter(db)
                result = await acf.get_actions_for_intent(
                    intent=prompt,
                    enabled_apps=app_names,
                    include_destructive=False,
                    max_actions=5
                )
                if result.actions:
                    action_ids = [a.action_id for a in result.actions]
                    logger.info(
                        f"[recipe_direct] ActionCapabilityFilter returned {len(action_ids)} actions "
                        f"for apps {app_names}: {action_ids}"
                    )
                    return action_ids
        except ImportError:
            logger.debug("[recipe_direct] ActionCapabilityFilter not available, using fallback")
        except Exception as e:
            logger.warning(f"[recipe_direct] ActionCapabilityFilter failed: {e}")

        # Fallback: keyword-based filtering from ComposioActionCache
        return _keyword_filter_actions(db, app_names, prompt)

    except Exception as e:
        logger.warning(f"[recipe_direct] Action filtering failed entirely: {e}")
        return []


def _keyword_filter_actions(db: Session, app_names: List[str], prompt: str) -> List[str]:
    """
    Fallback action filtering using keyword matching against ComposioActionCache.

    Tokenizes the prompt and matches against action names in the cache.
    """
    import re
    prompt_lower = prompt.lower()
    tokens = [t for t in re.split(r'[^a-z0-9]+', prompt_lower) if len(t) > 2]

    if not tokens:
        return []

    # Query actions for the assigned apps
    actions = db.query(ComposioActionCache).filter(
        ComposioActionCache.app_name.in_(app_names)
    ).all()

    if not actions:
        return []

    scored: List[tuple] = []
    for action in actions:
        action_name = (action.action_name or "").lower()
        description = (action.description or "").lower()
        # Score based on token matches in action name and description
        score = 0
        for token in tokens:
            if token in action_name:
                score += 3
            if token in description:
                score += 1
        if score > 0:
            scored.append((action.action_name, score))

    # Sort by score descending, take top 5
    scored.sort(key=lambda x: x[1], reverse=True)
    filtered = [name for name, _ in scored[:5]]

    if filtered:
        logger.info(f"[recipe_direct] Keyword filter returned {len(filtered)} actions: {filtered}")

    return filtered


def _normalize_tool_calls(raw_calls: Any) -> List[Dict[str, Any]]:
    """Normalize tool call data from agent execution result."""
    if not raw_calls:
        return []
    if isinstance(raw_calls, list):
        normalized = []
        for call in raw_calls:
            if isinstance(call, dict):
                normalized.append({
                    "action": call.get("action") or call.get("function", {}).get("name", "unknown"),
                    "params": call.get("params") or call.get("function", {}).get("arguments", {}),
                    "result": call.get("result") or call.get("content", {}),
                    "duration_ms": call.get("duration_ms", 0),
                })
        return normalized
    return []


def _persist_step_results(db: Session, execution: RecipeExecution, step_results: List[dict]):
    """Persist step results to the execution record."""
    try:
        execution.step_results = list(step_results)  # Force new list for JSONB mutation detection
        db.commit()
    except Exception as e:
        logger.error(f"[recipe_direct] Failed to persist step results: {e}")
        db.rollback()


async def _fail_execution(
    db: Session,
    execution_id: str,
    error_message: str,
    step_results: Optional[List[dict]] = None
):
    """Mark an execution as failed with an error message."""
    try:
        execution = db.query(RecipeExecution).filter(
            RecipeExecution.execution_id == execution_id
        ).first()
        if execution:
            execution.status = 'failed'
            execution.error_message = error_message
            execution.completed_at = datetime.now(timezone.utc)
            if step_results is not None:
                execution.step_results = step_results
            db.commit()
            logger.info(f"[recipe_direct] Execution {execution_id} marked FAILED: {error_message}")
    except Exception as e:
        logger.error(f"[recipe_direct] Failed to mark execution as failed: {e}")
        db.rollback()
