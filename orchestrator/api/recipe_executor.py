"""
Recipe Direct Executor
======================

Simple step-by-step executor for Starter Plan recipes.
Bypasses the 9-stage pipeline — executes recipe steps sequentially
using the SAME components as the chatbot (PRD-50 alignment):

- get_chat_tools() for tools (no enum constraint)
- ComposioHintService.build_hints() for hints
- create_llm_manager().generate_response() for LLM
- tool_router.execute_and_format() for tool execution

Key design:
- Structured step_outputs dict (keyed by output_key)
- _resolve_prompt injects previous step data as structured context
- Clean task prompt goes to hint service (not polluted by step data)
- Step data goes in system message (same as chatbot pattern)
"""

import json
import logging
import time
import uuid as uuid_mod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core.database.database import get_db, SessionLocal
from core.models import Agent
from core.models.core import RecipeExecution, WorkflowTemplate as WorkflowRecipe

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step executor — uses chatbot's exact component path
# ---------------------------------------------------------------------------

async def _execute_step(
    db: Session,
    agent: Agent,
    clean_prompt: str,
    step_outputs: Dict[str, Dict[str, Any]],
    workspace_id: UUID,
    input_data: Optional[dict] = None,
) -> dict:
    """
    Execute a single recipe step using the chatbot's exact component path.

    This replaces AgentFactory.execute_with_prompt() for recipes, ensuring
    the same tools, hints, LLM, and tool execution as the chatbot.

    Args:
        db: Database session
        agent: The Agent ORM object for this step
        clean_prompt: Task instruction only (e.g. "post results to Slack channel X")
        step_outputs: Previous step data (for system context)
        workspace_id: Workspace UUID for tool permissions
        input_data: Original trigger/input data (for system context on all steps)

    Returns:
        Dict with status, result, and execution metadata.
    """
    # Lazy imports to avoid circular deps
    from modules.tools.tool_router import get_agent_tools as get_chat_tools, get_tool_router
    from modules.tools.services.composio_hint_service import ComposioHintService
    from modules.agents.factory.agent_factory import AgentFactory

    # 0. Activate agent via factory — gives us the agent's LLM manager
    #    Same as chatbot stream_response_with_agent (service.py:1197-1221)
    factory = AgentFactory(db_session=db)
    agent_runtime = await factory.activate_agent(agent.id)
    if not agent_runtime:
        return {
            "status": "error",
            "error": f"Agent {agent.id} could not be activated",
            "execution": {"tokens_used": 0, "tool_calls": []},
        }

    # 1. System prompt — build from agent's identity and skills
    system_prompt = await _build_system_prompt(agent, db)
    messages = [{"role": "system", "content": system_prompt}]

    # 2. Hints — same service, same call signature as chatbot (service.py:1314-1325)
    try:
        hint_service = ComposioHintService(db)
        hint_result = hint_service.build_hints(
            agent_id=agent.id,
            prompt=clean_prompt,
            workspace_id=workspace_id,
        )
        if hint_result.hint_lines:
            messages.append({"role": "system", "content": "\n".join(hint_result.hint_lines)})
            logger.info(
                f"[recipe_step] Hints: strategy={hint_result.strategy_used} "
                f"actions={len(hint_result.matched_actions)}"
            )
    except Exception as exc:
        logger.warning(f"[recipe_step] Hint injection failed: {exc}", exc_info=True)

    # 3a. Original trigger/input data as persistent context
    if input_data:
        input_content = input_data.get("content", "")
        input_meta = {k: v for k, v in input_data.items() if k not in ("content", "metadata")}
        if input_content or input_meta:
            ctx_parts = ["=" * 40, "ORIGINAL REQUEST / TRIGGER DATA", "=" * 40]
            if input_content:
                ctx_parts.append(input_content)
            if input_meta:
                ctx_parts.append("\nMetadata:")
                for mk, mv in input_meta.items():
                    ctx_parts.append(f"  {mk}: {mv}")
            messages.append({"role": "system", "content": "\n".join(ctx_parts)})

    # 3b. Step data as system context (if prior steps exist)
    if step_outputs:
        messages.append({"role": "system", "content": _format_step_data(step_outputs)})

    # 4. User message — clean task prompt (same as chatbot user message)
    messages.append({"role": "user", "content": clean_prompt})

    # 5. Tools — same function as chatbot (service.py:1258)
    tools = get_chat_tools(agent_id=agent.id, workspace_id=workspace_id)

    # 6. LLM — agent's own LLM manager, same as chatbot (service.py:1382)
    llm = agent_runtime.llm_manager

    # 7. Generate + tool loop — same pattern as chatbot (service.py:1382-1428)
    tool_router = get_tool_router()
    all_tool_calls = []
    max_iterations = 3
    response = None

    for iteration in range(max_iterations):
        response = await llm.generate_response(messages=messages, tools=tools)

        if not response or not response.tool_calls:
            break  # LLM done, has final text

        # Process tool calls — same as chatbot (service.py:797)
        messages.append({
            "role": "assistant",
            "content": response.content or "",
            "tool_calls": response.tool_calls,
        })

        for tc in response.tool_calls:
            tool_name = tc.get("function", {}).get("name", "unknown")
            tool_args_raw = tc.get("function", {}).get("arguments", "{}")
            tool_id = tc.get("id", str(uuid_mod.uuid4()))

            try:
                tool_args = json.loads(tool_args_raw) if isinstance(tool_args_raw, str) else (tool_args_raw or {})
            except json.JSONDecodeError:
                tool_args = {}

            result = await tool_router.execute_and_format(
                tool_name,
                tool_args,
                agent_id=agent.id,
                workspace_id=workspace_id,
                original_intent=clean_prompt,
            )

            all_tool_calls.append({
                "action": tool_args.get("action", tool_name),
                "params": tool_args.get("params", tool_args),
                "result": result.get("llm_context", ""),
            })

            messages.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "content": result.get("llm_context", "No result"),
            })

        logger.info(f"[recipe_step] Tool iteration {iteration + 1}: {len(response.tool_calls)} calls")

    # 8. Return in format recipe_executor expects
    content = (response.content or "") if response else ""
    tokens = 0
    if response and hasattr(response, 'usage') and response.usage:
        tokens = response.usage.get("total_tokens", 0) if isinstance(response.usage, dict) else 0

    return {
        "status": "success",
        "result": content,
        "execution": {
            "tokens_used": tokens,
            "tool_calls": all_tool_calls,
        },
    }


async def _build_system_prompt(agent: Agent, db: Session) -> str:
    """
    Build a system prompt for the agent from its DB record.

    Simplified version of agent_factory._build_agent_system_prompt —
    loads identity + skills without the full factory machinery.

    If the agent has assigned plugins, loads plugin context (Tier 1 + Tier 2)
    and skips skill loading entirely.
    """
    sections = [
        f"# Agent: {agent.name}",
        f"Agent ID: {agent.id}",
        f"Agent Type: {getattr(agent, 'agent_type', 'unknown')}",
    ]
    if agent.description:
        sections.append(agent.description)

    # PRD-42: Inject persona
    try:
        if getattr(agent, 'use_custom_persona', False) and agent.custom_persona_prompt:
            sections.append(f"\n## Persona & Communication Style\n{agent.custom_persona_prompt}")
            logger.info(f"[recipe] Loaded custom persona for agent {agent.id}")
        elif getattr(agent, 'persona_id', None) and getattr(agent, 'persona', None):
            persona_prompt = agent.persona.system_prompt or ""
            if persona_prompt:
                sections.append(f"\n## Persona & Communication Style\n{persona_prompt}")
                logger.info(f"[recipe] Loaded persona '{agent.persona.name}' for agent {agent.id}")
    except Exception as e:
        logger.warning(f"[recipe] Failed to load persona for agent {agent.id}: {e}")

    # PRD-42: Load plugins — if present, skip skills entirely
    has_plugins = False
    try:
        from core.services.plugin_context_service import PluginContextService

        plugin_svc = PluginContextService(db)
        plugin_rows = plugin_svc.get_assigned_plugins(agent.id)
        if plugin_rows:
            has_plugins = True
            tier1 = plugin_svc.build_tier1_summary(plugin_rows)
            tier2 = await plugin_svc.build_tier2_content(
                plugin_rows,
                task_context=agent.description,
            )
            sections.append(tier1)
            if tier2:
                sections.append(tier2)
            logger.info(
                "[recipe] Loaded plugin context for agent %s (%d plugins)",
                agent.id, len(plugin_rows),
            )
    except Exception as e:
        logger.warning(f"[recipe] Failed to load plugins for agent {agent.id}: {e}")

    # Load skills if assigned (skipped when plugins are present)
    if not has_plugins and getattr(agent, 'skills', None):
        sections.append("\n## Your Specialized Skills\n")
        try:
            from modules.agents.services.skill_loader import get_skill_loader
            loader = get_skill_loader(db)
            for skill in agent.skills:
                sections.append(f"### {skill.name}")
                core_content = None
                try:
                    core_content = loader.load_skill_core(skill.name, db=db)
                except Exception:
                    pass
                if core_content and isinstance(core_content, str) and core_content.strip():
                    sections.append(core_content)
                else:
                    fallback = skill.prompt_template or skill.description or ""
                    if fallback:
                        sections.append(str(fallback))
        except Exception as e:
            logger.warning(f"[recipe_step] Skill loading failed: {e}")

    return "\n\n".join(sections)


def _format_step_data(step_outputs: Dict[str, Dict[str, Any]]) -> str:
    """Format previous step outputs as a system context message."""
    parts = [
        "=" * 60,
        "DATA FROM PREVIOUS STEPS",
        "The user's task may reference 'results', 'output', 'data', or 'findings'.",
        "That refers to the content below. Use it directly — do not invent data.",
        "=" * 60,
    ]

    sorted_entries = sorted(
        step_outputs.items(),
        key=lambda kv: kv[1].get("step_order", 0),
    )
    for out_key, entry in sorted_entries:
        sr_order = entry.get("step_order", "?")
        sr_agent = entry.get("agent_name", "Agent")
        sr_output = entry.get("text", "")
        sr_tool_calls = entry.get("tool_calls", [])

        parts.append(f"\n--- Step {sr_order} ({out_key}): {sr_agent} ---")

        if sr_tool_calls:
            for tc in sr_tool_calls:
                action = tc.get("action", "unknown")
                tc_result = tc.get("result", "")
                if tc_result:
                    result_str = (
                        json.dumps(tc_result, indent=2)
                        if isinstance(tc_result, (dict, list))
                        else str(tc_result)
                    )
                    if len(result_str) > 1500:
                        result_str = result_str[:1500] + "\n... (truncated)"
                    parts.append(f"[Tool: {action}]\n{result_str}")

        if sr_output:
            output_preview = sr_output[:3000]
            if len(sr_output) > 3000:
                output_preview += "\n... (truncated)"
            parts.append(f"[Agent Output]\n{output_preview}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main executor
# ---------------------------------------------------------------------------

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
    2. Build clean prompt (input substitutions only)
    3. Call _execute_step() using chatbot's exact components
    4. Store step result in RecipeExecution.step_results JSONB
    5. Handle errors per step.error_handling config

    Args:
        recipe_execution_id: The execution_id string (e.g. "exec-abc123")
        recipe_id: Integer PK of the recipe (WorkflowTemplate.id)
        workspace_id: UUID of the workspace
        input_data: User-provided input data
        db_url: Optional database URL (uses default SessionLocal if not provided)
    """
    # Create a fresh DB session for this async task
    _engine = None  # Track custom engine for cleanup
    if db_url:
        _engine = create_engine(db_url)
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
        db = _SessionLocal()
    else:
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

        # Execute each step sequentially
        step_results: List[Dict[str, Any]] = []  # Flat list for DB persistence
        step_outputs: Dict[str, Dict[str, Any]] = {}  # Keyed dict for inter-step data passing
        execution_start = time.time()

        for idx, step in enumerate(steps):
            step_id = step.get('step_id', f'step-{idx + 1}')
            step_order = step.get('order', idx + 1)
            agent_id = step.get('agent_id')
            prompt_template = step.get('prompt_template', '')
            error_handling = step.get('error_handling', 'stop')
            max_retries = step.get('max_retries', 1)
            output_key = step.get('output_key', f'step_{step_order}')
            agent = agent_map.get(agent_id)
            agent_name = agent.name if agent else f"Agent {agent_id}"

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
                "output_key": output_key,
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

            # Build clean step prompt: input substitutions + trigger context.
            # Step data goes into system message via _execute_step.
            clean_step_prompt = prompt_template
            if input_data:
                for key, value in input_data.items():
                    clean_step_prompt = clean_step_prompt.replace(f"{{input.{key}}}", str(value))

            # Inject trigger/input context so agents know what they're working on.
            # Only add if input_data has real content and prompt doesn't already
            # contain the substituted values (i.e. no {input.xxx} placeholders used).
            trigger_content = input_data.get("content", "") if input_data else ""
            trigger_metadata = {k: v for k, v in (input_data or {}).items() if k not in ("content", "metadata")}
            if trigger_content and clean_step_prompt == prompt_template:
                # No placeholders were used — prepend context to prompt
                context_block = f"## Trigger Context\n{trigger_content}"
                if trigger_metadata:
                    meta_lines = "\n".join(f"- {k}: {v}" for k, v in trigger_metadata.items())
                    context_block += f"\n\n## Metadata\n{meta_lines}"
                clean_step_prompt = f"{context_block}\n\n## Your Task\n{clean_step_prompt}"

            # Execute with retries
            attempt = 0
            success = False
            last_error = None

            while attempt <= max_retries and not success:
                if attempt > 0:
                    logger.info(f"[recipe_direct] Retrying step {step_order}, attempt {attempt + 1}")
                    step_result["retries"] = attempt

                try:
                    result = await _execute_step(
                        db=db,
                        agent=agent,
                        clean_prompt=clean_step_prompt,
                        step_outputs=step_outputs,
                        workspace_id=workspace_id,
                        input_data=input_data,
                    )

                    if result.get("status") == "success":
                        step_result["status"] = "completed"
                        raw_output = result.get("result", "")
                        # Coerce non-string outputs to JSON strings for _resolve_prompt compatibility
                        if isinstance(raw_output, (dict, list)):
                            step_result["output"] = json.dumps(raw_output)
                        elif raw_output is not None:
                            step_result["output"] = str(raw_output)
                        else:
                            step_result["output"] = ""
                        step_result["tokens_used"] = result.get("execution", {}).get("tokens_used", 0)

                        # Extract tool calls from execution metadata
                        tool_calls_raw = result.get("execution", {}).get("tool_calls", [])
                        step_result["tool_calls"] = _normalize_tool_calls(tool_calls_raw)

                        success = True

                        # Store in keyed step_outputs dict for next steps
                        step_outputs[output_key] = {
                            "text": step_result["output"],
                            "tool_calls": step_result["tool_calls"],
                            "agent_name": agent_name,
                            "step_order": step_order,
                        }

                        logger.info(f"[recipe_direct] Step {step_order} completed → output_key={output_key} ({step_result['tokens_used']} tokens)")
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

        # Determine final output: last completed step's output
        final_output = None
        for sr in reversed(step_results):
            if sr.get("status") == "completed" and sr.get("output"):
                final_output = sr["output"]
                break

        execution.status = 'completed'
        execution.completed_at = datetime.now(timezone.utc)
        execution.step_results = step_results
        execution.output_data = {
            "final_output": final_output,
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
        except Exception as err:
            logger.exception(
                f"[recipe_direct] _fail_execution itself failed for {recipe_execution_id}: {err}"
            )
    finally:
        db.close()
        if _engine is not None:
            _engine.dispose()


# ---------------------------------------------------------------------------
# Prompt resolution (kept for backward compat — used by _resolve_prompt callers)
# ---------------------------------------------------------------------------

def _resolve_prompt(
    template: str,
    input_data: dict,
    step_outputs: Dict[str, Dict[str, Any]],
) -> str:
    """
    Resolve a step's prompt template with variable substitution.

    IMPORTANT: The task instruction stays at the TOP of the prompt.
    Context from previous steps is appended BELOW with a separator.

    Supports:
    - {input.field_name} — from user input_data
    - {previous_output} — text from the most recently completed step (backward compat)
    - {step_N_output} — text from step N by order (backward compat)
    - {output_key} — text from a step by its output_key (NEW)
    """
    resolved = template

    # Substitute {input.xxx} placeholders
    if input_data:
        for key, value in input_data.items():
            resolved = resolved.replace(f"{{input.{key}}}", str(value))

    # Determine "previous_output" for backward compat: last step's text
    previous_output: Optional[str] = None
    if step_outputs:
        # Find the step with the highest step_order
        last_entry = max(step_outputs.values(), key=lambda v: v.get("step_order", 0))
        previous_output = last_entry.get("text", "")

    # Substitute {previous_output} inline if template references it
    if previous_output:
        resolved = resolved.replace("{previous_output}", previous_output)

    # Substitute {step_N_output} for specific step references (backward compat)
    for key, entry in step_outputs.items():
        order = entry.get("step_order", 0)
        text = entry.get("text", "")
        if text:
            resolved = resolved.replace(f"{{step_{order}_output}}", text)

    # Substitute {output_key} references (new keyed references)
    for key, entry in step_outputs.items():
        text = entry.get("text", "")
        if text:
            resolved = resolved.replace(f"{{{key}}}", text)

    # Detect whether the template used ANY explicit references
    has_explicit_ref = (
        "{previous_output}" in template
        or any(f"{{step_{e.get('step_order', 0)}_output}}" in template for e in step_outputs.values())
        or any(f"{{{k}}}" in template for k in step_outputs)
    )

    # If previous steps completed and the template didn't explicitly reference
    # any step data, build structured context from ALL completed steps.
    if step_outputs and not has_explicit_ref:
        context_parts = []
        context_parts.append("=" * 60)
        context_parts.append("DATA FROM PREVIOUS STEPS")
        context_parts.append("When the task above mentions 'results', 'output', 'data',")
        context_parts.append("or 'findings', it refers to the content below.")
        context_parts.append("USE THIS CONTENT to complete the task — do not invent data.")
        context_parts.append("=" * 60)

        # Sort by step_order for consistent display
        sorted_entries = sorted(step_outputs.items(), key=lambda kv: kv[1].get("step_order", 0))
        for out_key, entry in sorted_entries:
            sr_order = entry.get("step_order", "?")
            sr_agent = entry.get("agent_name", "Agent")
            sr_output = entry.get("text", "")
            sr_tool_calls = entry.get("tool_calls", [])

            context_parts.append(f"\n--- Step {sr_order} ({out_key}): {sr_agent} ---")

            # Include tool call results — these contain the raw data
            if sr_tool_calls:
                for tc in sr_tool_calls:
                    action = tc.get("action", "unknown")
                    tc_result = tc.get("result", "")
                    if tc_result:
                        result_str = json.dumps(tc_result, indent=2) if isinstance(tc_result, (dict, list)) else str(tc_result)
                        if len(result_str) > 1500:
                            result_str = result_str[:1500] + "\n... (truncated)"
                        context_parts.append(f"[Tool: {action}]\n{result_str}")

            # Include the agent's summary output
            if sr_output:
                output_preview = sr_output[:3000]
                if len(sr_output) > 3000:
                    output_preview += "\n... (truncated)"
                context_parts.append(f"[Agent Output]\n{output_preview}")

        resolved = f"{resolved}\n\n" + "\n".join(context_parts)

    return resolved


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
