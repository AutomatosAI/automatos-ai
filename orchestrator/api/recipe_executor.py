"""
Recipe Direct Executor
======================

Simple step-by-step executor for Starter Plan recipes.
Bypasses the 9-stage pipeline — executes recipe steps sequentially
using the SAME components as the chatbot (PRD-50 alignment):

- ContextService(RECIPE) for system prompt + base tools (PRD-80)
- ComposioHintService.build_hints() for hints
- create_llm_manager().generate_response() for LLM
- tool_router.execute_and_format() for tool execution

v2 changes:
- RecipeScratchpad replaces verbose text dumps (80-90% token savings)
- scratchpad_write tool injected for explicit agent exports
- S3 cold storage for full step logs, DB stores compact summaries only
- Mem0 recipe memory wired for pre/post execution
"""

import asyncio
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
# Per-workspace execution semaphore — allows bounded concurrent recipe
# execution within a workspace.  Keys are workspace_id strings; values are
# asyncio.Semaphores created on first access.  The dict itself is
# process-global but safe because asyncio is single-threaded.
# ---------------------------------------------------------------------------
_workspace_semaphores: Dict[str, asyncio.Semaphore] = {}


def _get_workspace_semaphore(workspace_id: str, max_concurrent: int = 3) -> asyncio.Semaphore:
    """Return (or create) an asyncio.Semaphore for the given workspace.

    If the semaphore already exists but the limit changed, we keep the
    existing one to avoid resetting mid-flight — the concurrency guard
    pre-check handles the real enforcement.
    """
    if workspace_id not in _workspace_semaphores:
        _workspace_semaphores[workspace_id] = asyncio.Semaphore(max_concurrent)
    return _workspace_semaphores[workspace_id]


# ---------------------------------------------------------------------------
# Step executor — uses chatbot's exact component path
# ---------------------------------------------------------------------------

async def _execute_step(
    db: Session,
    agent: Agent,
    clean_prompt: str,
    workspace_id: UUID,
    scratchpad=None,
    step_order: int = 1,
    input_data: Optional[dict] = None,
    recipe_memories: Optional[dict] = None,
    prompt_for_hints: Optional[str] = None,
    max_iterations: int = 25,
    recipe_name: str = "",
    total_steps: int = 1,
) -> dict:
    """
    Execute a single recipe step using the chatbot's exact component path.

    Args:
        db: Database session
        agent: The Agent ORM object for this step
        clean_prompt: Task instruction (may include trigger context)
        workspace_id: Workspace UUID for tool permissions
        scratchpad: RecipeScratchpad instance (replaces step_outputs)
        step_order: Current step order number
        input_data: Original trigger/input data (for system context on all steps)
        recipe_memories: Mem0 memories to inject (first step only)
        prompt_for_hints: Clean task-only prompt for hint generation (avoids
            trigger metadata polluting action matching). Falls back to clean_prompt.
        max_iterations: Max LLM tool-call turns for this step. Higher values
            let the agent do more work (e.g. bug fixer needs ~15-20 turns).
            Configurable per step via step.max_iterations, per agent via
            agent.configuration.max_iterations, or falls back to 25.

    Returns:
        Dict with status, result, and execution metadata.
    """
    # Lazy imports to avoid circular deps
    from modules.tools.tool_router import get_tool_router
    from modules.tools.services.composio_hint_service import ComposioHintService
    from modules.tools.services.composio_tool_service import ComposioToolService
    from modules.agents.factory.agent_factory import AgentFactory
    from modules.context import ContextService, ContextMode
    from modules.tools.builtin.scratchpad_tool import (
        SCRATCHPAD_WRITE_TOOL_DEF,
        SCRATCHPAD_READ_TOOL_DEF,
        SCRATCHPAD_TOOL_NAME,
        SCRATCHPAD_READ_NAME,
        handle_scratchpad_write,
        handle_scratchpad_read,
    )

    # 0. Activate agent via factory — gives us the agent's LLM manager
    factory = AgentFactory(db_session=db)
    agent_runtime = await factory.activate_agent(agent.id)
    if not agent_runtime:
        return {
            "status": "error",
            "error": f"Agent {agent.id} could not be activated",
            "execution": {"tokens_used": 0, "tool_calls": [], "messages": []},
        }

    # 1. System prompt + tools via ContextService (PRD-80)
    #    Build recipe_step dict for RecipeContextSection
    previous_output = ""
    if scratchpad and step_order > 1:
        prev_ctx = scratchpad.format_context_for_step(step_order)
        if prev_ctx:
            previous_output = prev_ctx

    recipe_step_dict = {
        "name": recipe_name,
        "step_number": step_order,
        "total_steps": total_steps,
        "instructions": clean_prompt,
        "previous_output": previous_output,
    }

    context = await ContextService(db).build_context(
        mode=ContextMode.RECIPE,
        agent=agent,
        workspace_id=str(workspace_id),
        recipe_step=recipe_step_dict,
    )

    messages = [{"role": "system", "content": context.system_prompt}]
    base_tools = context.tools

    # 1b. Recipe step scope — prevent agent from wandering into other steps' tasks
    scope_instruction = (
        f"## Recipe Step Scope\n"
        f"You are executing step {step_order} of a multi-step recipe. "
        f"Focus ONLY on the task described in the user message below. "
        f"Do NOT perform tasks that belong to other steps (e.g., sending notifications, "
        f"creating PRs, or any action not explicitly required by YOUR task). "
        f"Use ONLY the external app actions that are directly relevant to your specific task."
    )
    messages.append({"role": "system", "content": scope_instruction})

    # 2. Composio tools — SDK semantic search for per-action function-calling tools.
    #    Falls back to hint-based composio_execute if SDK search returns empty.
    tool_service = ComposioToolService(db)
    composio_result = None
    try:
        composio_result = tool_service.get_tools_for_step(
            agent_id=agent.id,
            workspace_id=workspace_id,
            task_prompt=prompt_for_hints or clean_prompt,
        )
    except Exception as exc:
        logger.warning(f"[recipe_step] ComposioToolService failed: {exc}", exc_info=True)

    # If SDK search returned tools, inject a simpler scope message.
    # Otherwise fall back to hint-based composio_execute mega-tool.
    if composio_result and composio_result.tools:
        messages.append({"role": "system", "content": _composio_scope_message(composio_result.app_names)})
        logger.info(
            f"[recipe_step] SDK search: strategy={composio_result.strategy} "
            f"actions={len(composio_result.action_set)} search_ms={composio_result.search_ms}"
        )
    else:
        # Fallback: existing hint service with composio_execute mega-tool
        if composio_result:
            composio_result.strategy = "hint_fallback"
        try:
            hint_service = ComposioHintService(db)
            hint_result = hint_service.build_hints(
                agent_id=agent.id,
                prompt=prompt_for_hints or clean_prompt,
                workspace_id=workspace_id,
                recipe_mode=True,
            )
            if hint_result.hint_lines:
                messages.append({"role": "system", "content": "\n".join(hint_result.hint_lines)})
                logger.info(
                    f"[recipe_step] Hints fallback: strategy={hint_result.strategy_used} "
                    f"actions={len(hint_result.matched_actions)}"
                )
        except Exception as exc:
            logger.warning(f"[recipe_step] Hint injection failed: {exc}", exc_info=True)

    # 3a. Mem0 memories (first step only)
    if recipe_memories and step_order == 1:
        summary = recipe_memories.get("summary", "")
        if summary and summary != "No relevant memories found":
            messages.append({
                "role": "system",
                "content": (
                    "## Learnings from Previous Runs\n"
                    f"{summary}"
                ),
            })
            logger.info("[recipe_step] Injected %d Mem0 memories into step 1", recipe_memories.get("total_memories", 0))

    # 3b. Original trigger/input data as persistent context
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

    # 3c. Scratchpad context — now handled via recipe_step.previous_output
    #    in ContextService, but we still inject raw scratchpad for steps > 1
    #    in case the RecipeContextSection truncated it.

    # 4. User message — clean task prompt
    messages.append({"role": "user", "content": clean_prompt})

    # 5. Tools — base tools from ContextService (PRD-80) + Composio overlay
    #    When Composio actions are resolved (e.g. JIRA_CREATE_ISSUE), strip only
    #    the generic fallback executor. Keep platform tools, workspace tools, and
    #    knowledge tools — the LLM needs them for context gathering (e.g. fetching
    #    logs to attach to JIRA tickets, reading test reports, etc.).
    _STRIP_WHEN_COMPOSIO = {
        "composio_execute",  # Only strip the generic fallback — per-action tools replace it
    }
    if composio_result and composio_result.tools:
        # SDK search succeeded: keep only non-exploration builtins + per-action tools
        builtin_tools = [
            t for t in base_tools
            if t.get("function", {}).get("name") not in _STRIP_WHEN_COMPOSIO
        ]
        tools = builtin_tools + composio_result.tools
        logger.info(
            "[recipe_step] Composio step: %d builtins + %d actions (stripped %d exploration tools)",
            len(builtin_tools), len(composio_result.tools),
            len(_STRIP_WHEN_COMPOSIO),
        )
    else:
        # Fallback: composio_execute + hints (existing behavior)
        tools = list(base_tools)
    if scratchpad:
        scratchpad_tools = [SCRATCHPAD_WRITE_TOOL_DEF]
        if step_order > 1:
            scratchpad_tools.append(SCRATCHPAD_READ_TOOL_DEF)
        tools = list(tools) + scratchpad_tools

    # 6. LLM — agent's own LLM manager (with tracking context for recipe)
    llm = agent_runtime.llm_manager
    if hasattr(llm, '_tracking_ctx'):
        llm._tracking_ctx["request_type"] = "recipe"

    # 7. Generate + tool loop
    tool_router = get_tool_router()
    all_tool_calls = []
    _composio_call_cache: Dict[str, str] = {}  # dedup: "ACTION|args_hash" → cached result
    response = None

    for iteration in range(max_iterations):
        response = await llm.generate_response(messages=messages, tools=tools)

        if not response or not response.tool_calls:
            break  # LLM done, has final text

        # Process tool calls
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

            # Handle scratchpad tools inline (no tool_router needed)
            if tool_name == SCRATCHPAD_TOOL_NAME and scratchpad:
                result_text = handle_scratchpad_write(
                    key=tool_args.get("key", "unknown"),
                    value=tool_args.get("value", ""),
                    scratchpad=scratchpad,
                    step_order=step_order,
                )
                all_tool_calls.append({
                    "action": SCRATCHPAD_TOOL_NAME,
                    "params": tool_args,
                    "result": result_text,
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": result_text,
                })
                continue

            if tool_name == SCRATCHPAD_READ_NAME and scratchpad:
                result_text = handle_scratchpad_read(
                    key=tool_args.get("key", ""),
                    scratchpad=scratchpad,
                )
                all_tool_calls.append({
                    "action": SCRATCHPAD_READ_NAME,
                    "params": tool_args,
                    "result": result_text,
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": result_text,
                })
                continue

            # Direct Composio action execution (SDK search path).
            # Also catches LLM-inferred actions beyond search results —
            # if the tool name matches a connected app prefix (e.g. JIRA_*),
            # route to Composio direct execution instead of tool_router.
            _is_composio_action = (
                composio_result and composio_result.entity_id and (
                    tool_name in composio_result.action_set
                    or any(tool_name.startswith(f"{app}_") for app in composio_result.app_names)
                )
            )
            if _is_composio_action:
                # Dedup: if the LLM calls the same action with the same args
                # again, return the cached result instead of hitting the API.
                _dedup_key = f"{tool_name}|{json.dumps(tool_args, sort_keys=True, default=str)}"
                if _dedup_key in _composio_call_cache:
                    result_text = _composio_call_cache[_dedup_key]
                    exec_ms = 0
                    logger.info(f"[recipe_step] Composio dedup hit: {tool_name} (skipped repeat call)")
                else:
                    try:
                        t0 = time.time()
                        exec_result = tool_service.execute_action(
                            action_name=tool_name,
                            params=tool_args,
                            entity_id=composio_result.entity_id,
                        )
                        exec_ms = int((time.time() - t0) * 1000)

                        success = exec_result.get("success", False)
                        data = exec_result.get("data")
                        error = exec_result.get("error")

                        if success:
                            result_text = json.dumps(data, default=str) if isinstance(data, (dict, list)) else str(data or "")
                            logger.info(f"[recipe_step] Composio direct OK: {tool_name} in {exec_ms}ms")
                        else:
                            result_text = f"Error executing {tool_name}: {error or 'unknown error'}"
                            logger.warning(f"[recipe_step] Composio direct failed: {tool_name} — {error}")
                    except Exception as exc:
                        exec_ms = int((time.time() - t0) * 1000)
                        result_text = f"Error executing {tool_name}: {exc}"
                        logger.error(f"[recipe_step] Composio direct exception: {tool_name}: {exc}", exc_info=True)
                    _composio_call_cache[_dedup_key] = result_text

                all_tool_calls.append({
                    "action": tool_name,
                    "params": tool_args,
                    "result": result_text[:8000],
                    "duration_ms": exec_ms,
                    "composio_direct": True,
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": result_text[:20000],
                })
                continue

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

    # 8. Return with full message history for S3 logging
    content = (response.content or "") if response else ""
    tokens = 0
    if response and hasattr(response, 'usage') and response.usage:
        tokens = response.usage.get("total_tokens", 0) if isinstance(response.usage, dict) else 0

    # Composio metrics for tracking
    composio_metrics = {}
    if composio_result:
        composio_metrics = {
            "composio_strategy": composio_result.strategy,
            "composio_search_ms": composio_result.search_ms,
            "composio_tools_offered": len(composio_result.tools),
            "composio_tools_called": [
                tc["action"] for tc in all_tool_calls if tc.get("composio_direct")
            ],
        }

    return {
        "status": "success",
        "result": content,
        "execution": {
            "tokens_used": tokens,
            "tool_calls": all_tool_calls,
            "messages": messages,
            **composio_metrics,
        },
    }


def _composio_scope_message(app_names: List[str]) -> str:
    """
    Build a concise system message for SDK-search mode.

    When per-action tools are available (e.g. JIRA_GET_ISSUE as its own function),
    the LLM doesn't need action lists or param hints — just a note about which
    apps are connected and that it should use the provided tools directly.
    """
    apps = ", ".join(sorted(set(app_names))) if app_names else "your connected apps"
    return (
        f"You have these external apps connected: {apps}.\n"
        "The relevant actions are available as individual tools with their "
        "parameter schemas. Call them directly by name — do NOT use composio_execute."
    )




# ---------------------------------------------------------------------------
# S3 log storage
# ---------------------------------------------------------------------------

def _upload_step_log_to_s3(
    workspace_id: UUID,
    execution_id: str,
    step_order: int,
    log_data: Dict[str, Any],
) -> Optional[str]:
    """
    Upload full verbose step log to S3.

    Returns the S3 URL on success, None on failure.
    Path: workspaces/{workspace_id}/logs/executions/{execution_id}/step_{order}.json
    """
    try:
        import boto3
        from config import config

        bucket = config.RECIPE_LOG_S3_BUCKET
        if not bucket:
            return None

        s3_key = f"workspaces/{workspace_id}/logs/executions/{execution_id}/step_{step_order}.json"

        s3 = boto3.client(
            "s3",
            region_name=config.AWS_REGION or "us-east-1",
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
        )

        s3.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=json.dumps(log_data, default=str),
            ContentType="application/json",
        )

        log_url = f"s3://{bucket}/{s3_key}"
        logger.info("[recipe_direct] Uploaded step %d log to %s", step_order, log_url)
        return log_url

    except Exception as exc:
        logger.warning("[recipe_direct] S3 log upload failed for step %d: %s", step_order, exc)
        return None


def _build_compact_step_result(
    step_result: Dict[str, Any],
    log_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a compact step result for DB storage.

    Strips full output/tool_calls and replaces with summaries + S3 log_url.
    """
    tool_calls = step_result.get("tool_calls", [])
    tool_summaries = []
    for tc in tool_calls:
        action = tc.get("action", "unknown")
        # Infer success/failure from result content
        result_str = str(tc.get("result", ""))
        status = "error" if "error" in result_str.lower()[:100] else "success"
        tool_summaries.append(f"{action} ({status})")

    output = step_result.get("output", "")
    output_preview = output[:200] + "..." if output and len(output) > 200 else (output or "")

    compact = {
        "step_id": step_result.get("step_id"),
        "order": step_result.get("order"),
        "agent_id": step_result.get("agent_id"),
        "agent_name": step_result.get("agent_name"),
        "prompt_template": step_result.get("prompt_template", ""),
        "status": step_result.get("status"),
        "duration_ms": step_result.get("duration_ms"),
        "tokens_used": step_result.get("tokens_used"),
        "tool_calls_summary": tool_summaries,
        "output_preview": output_preview,
        "error": step_result.get("error"),
        "retries": step_result.get("retries", 0),
        "started_at": step_result.get("started_at"),
        "completed_at": step_result.get("completed_at"),
    }

    if log_url:
        compact["log_url"] = log_url

    return compact


# ---------------------------------------------------------------------------
# Safe fire-and-forget wrapper
# ---------------------------------------------------------------------------

def launch_recipe_task(
    recipe_execution_id: str,
    recipe_id: int,
    workspace_id: UUID,
    input_data: dict,
):
    """Launch execute_recipe_direct as an async task with crash protection.

    If the task raises an unhandled exception, the execution record is
    marked as 'failed' instead of silently staying in 'pending' forever.
    """

    async def _safe_execute():
        try:
            await execute_recipe_direct(
                recipe_execution_id=recipe_execution_id,
                recipe_id=recipe_id,
                workspace_id=workspace_id,
                input_data=input_data,
            )
        except Exception as e:
            logger.error(
                "[recipe_direct] Async task crashed for execution %s: %s",
                recipe_execution_id, e, exc_info=True,
            )
            # Last-resort: mark execution as failed so it doesn't hang forever
            try:
                db = SessionLocal()
                try:
                    await _fail_execution(db, recipe_execution_id, f"Task crashed: {e}")
                finally:
                    db.close()
            except Exception as inner:
                logger.error(
                    "[recipe_direct] Could not mark execution %s as failed: %s",
                    recipe_execution_id, inner,
                )

    asyncio.create_task(_safe_execute())


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
    3. Call _execute_step() with scratchpad context
    4. Auto-extract tool results into scratchpad
    5. Upload full log to S3, store compact summary in DB
    6. Handle errors per step.error_handling config
    """
    # Bounded concurrency per workspace — allows N recipes to run in
    # parallel (controlled by DEFAULT_MAX_CONCURRENT_RUNNING / plan_limits).
    from config import config as app_config
    max_concurrent = app_config.DEFAULT_MAX_CONCURRENT_RUNNING
    semaphore = _get_workspace_semaphore(str(workspace_id), max_concurrent)

    waiting = semaphore.locked()
    if waiting:
        logger.info(
            "[recipe_direct] QUEUED — waiting for workspace semaphore %s (execution=%s, recipe=%s)",
            workspace_id, recipe_execution_id, recipe_id,
        )
    async with semaphore:
        logger.info(
            "[recipe_direct] Semaphore ACQUIRED for %s (execution=%s, recipe=%s, was_waiting=%s)",
            workspace_id, recipe_execution_id, recipe_id, waiting,
        )
        try:
            await _execute_recipe_inner(
                recipe_execution_id, recipe_id, workspace_id, input_data, db_url,
            )
        finally:
            logger.info(
                "[recipe_direct] Semaphore RELEASED for %s (execution=%s)",
                workspace_id, recipe_execution_id,
            )


async def _execute_recipe_inner(
    recipe_execution_id: str,
    recipe_id: int,
    workspace_id: UUID,
    input_data: dict,
    db_url: Optional[str] = None,
):
    """Inner recipe execution — runs under the per-workspace lock."""
    # Create a fresh DB session for this async task
    _engine = None
    if db_url:
        _engine = create_engine(db_url)
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
        db = _SessionLocal()
    else:
        db = SessionLocal()

    scratchpad = None
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

        # Create board task for this execution
        try:
            from services.board_task_bridge import create_recipe_board_task
            create_recipe_board_task(db, recipe, execution)
        except Exception:
            db.rollback()
            logger.warning("Board task creation failed (non-blocking)", exc_info=True)

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

        # --- Initialize scratchpad ---
        from core.services.recipe_scratchpad import RecipeScratchpad
        scratchpad = RecipeScratchpad(recipe_execution_id)
        scratchpad.write_inputs(input_data)
        scratchpad.write_meta(recipe_id, total_steps)

        # --- Pre-execution: load Mem0 memories ---
        recipe_memories = None
        try:
            from core.services.recipe_memory_service import RecipeMemoryService
            memory_svc = RecipeMemoryService(db=db)
            recipe_memories = await memory_svc.retrieve_relevant_memories(
                recipe_id=recipe.id,
                context={"workspace_id": str(workspace_id), "input_data": input_data}
            )
            if recipe_memories and recipe_memories.get("total_memories", 0) > 0:
                logger.info(
                    "[recipe_direct] Loaded %d Mem0 memories for recipe %d",
                    recipe_memories["total_memories"], recipe.id,
                )
        except Exception as exc:
            logger.info("[recipe_direct] Mem0 memory retrieval skipped: %s", exc)

        # Read timeout config from execution_config
        # Values may be stored in ms (>=10000) or seconds (<10000) depending on
        # when the recipe was created. Normalise to seconds.
        # Threshold: 10000+ is clearly ms (e.g. 120000ms=120s).
        # Values like 1200 are valid seconds (20min), NOT milliseconds.
        exec_config = recipe.execution_config or {}

        raw_step = exec_config.get('timeout_per_step') or exec_config.get('per_step_timeout') or 300
        raw_total = exec_config.get('total_timeout') or 600

        step_timeout_sec = raw_step / 1000 if raw_step >= 10000 else raw_step   # ms → s
        total_timeout_sec = raw_total / 1000 if raw_total >= 10000 else raw_total

        # Clamp: at least 60s per step, at least 120s total
        step_timeout_sec = max(step_timeout_sec, 60)
        total_timeout_sec = max(total_timeout_sec, 120)
        logger.info(
            f"[recipe_direct] Timeouts: step={step_timeout_sec:.0f}s, "
            f"total={total_timeout_sec:.0f}s (raw: step={raw_step}, total={raw_total})"
        )

        # Execute each step sequentially
        step_results: List[Dict[str, Any]] = []
        execution_start = time.time()

        for idx, step in enumerate(steps):
            # Total timeout check
            elapsed = time.time() - execution_start
            if elapsed > total_timeout_sec:
                msg = f"Total execution timeout ({total_timeout_sec}s) exceeded after {int(elapsed)}s at step {idx + 1}"
                logger.warning(f"[recipe_direct] {msg}")
                _persist_step_results(db, execution, step_results)
                await _fail_execution(db, recipe_execution_id, msg, step_results=step_results)
                return

            step_id = step.get('step_id', f'step-{idx + 1}')
            step_order = step.get('order', idx + 1)
            agent_id = step.get('agent_id')
            prompt_template = step.get('prompt_template', '')
            error_handling = step.get('error_handling', 'stop')
            max_retries = step.get('max_retries', 1)
            output_key = step.get('output_key', f'step_{step_order}')
            agent = agent_map.get(agent_id)
            agent_name = agent.name if agent else f"Agent {agent_id}"

            agent_cfg = getattr(agent, 'configuration', None) or {}
            step_max_iter = step.get("max_iterations", agent_cfg.get("max_iterations", 25))
            logger.info(f"[recipe_direct] Step {step_order}/{total_steps}: {agent_name} (max_turns={step_max_iter}) — {prompt_template[:200]}")

            # Update execution progress
            execution.current_step = idx + 1
            db.commit()

            step_start = time.time()
            step_result: Dict[str, Any] = {
                "step_id": step_id,
                "order": step_order,
                "agent_id": agent_id,
                "agent_name": agent_name,
                "prompt_template": prompt_template,
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

            # Build clean step prompt: input substitutions + trigger context
            clean_step_prompt = prompt_template
            if input_data:
                for key, value in input_data.items():
                    clean_step_prompt = clean_step_prompt.replace(f"{{input.{key}}}", str(value))

            # Inject trigger/input context
            trigger_content = input_data.get("content", "") if input_data else ""
            trigger_metadata = {k: v for k, v in (input_data or {}).items() if k not in ("content", "metadata")}
            if trigger_content or trigger_metadata:
                context_block_parts = []
                key_fields = ["issue_key", "project", "summary", "issue_type", "priority"]
                id_lines = [f"- {k}: {trigger_metadata[k]}" for k in key_fields if trigger_metadata.get(k)]
                if id_lines:
                    context_block_parts.append("## Ticket Reference\n" + "\n".join(id_lines))
                if clean_step_prompt == prompt_template and trigger_content:
                    context_block_parts.append(f"## Trigger Context\n{trigger_content}")
                    extra_meta = {k: v for k, v in trigger_metadata.items() if k not in key_fields}
                    if extra_meta:
                        meta_lines = "\n".join(f"- {k}: {v}" for k, v in extra_meta.items())
                        context_block_parts.append(f"## Additional Metadata\n{meta_lines}")
                if context_block_parts:
                    clean_step_prompt = "\n\n".join(context_block_parts) + f"\n\n## Your Task\n{clean_step_prompt}"

            # Check for generate_document step type (PRD-63)
            step_type = step.get("type", "agent")
            if step_type == "generate_document":
                try:
                    from modules.documents.generation_service import DocumentGenerationService
                    gen_config = step.get("config", step)
                    gen_title = gen_config.get("title", "Document")
                    gen_format = gen_config.get("format", "pdf")
                    gen_data = gen_config.get("data", {})
                    gen_template_name = gen_config.get("template_name")

                    # Resolve {{step_N.field}} variables in data from scratchpad
                    if scratchpad and isinstance(gen_data, dict):
                        gen_data = _resolve_doc_step_variables(gen_data, scratchpad)

                    gen_service = DocumentGenerationService(db, workspace_id)
                    gen_result = await gen_service.generate(
                        title=gen_title,
                        format=gen_format,
                        data=gen_data,
                        workspace_id=workspace_id,
                        template_name=gen_template_name,
                    )

                    step_result["status"] = "completed"
                    step_result["output"] = json.dumps({
                        "document_url": gen_result.download_url,
                        "filename": gen_result.filename,
                        "format": gen_result.format,
                        "size_kb": gen_result.size // 1024,
                    })
                    step_result["duration_ms"] = int((time.time() - step_start) * 1000)
                    step_result["completed_at"] = datetime.now(timezone.utc).isoformat()

                    if scratchpad:
                        scratchpad.write_step_results(
                            step_order=step_order,
                            tool_calls=[],
                            agent_output=step_result["output"],
                            agent_exports={},
                        )

                    logger.info(f"[recipe_direct] Step {step_order} (generate_document) completed: {gen_result.filename}")
                    compact = _build_compact_step_result(step_result)
                    step_results.append(compact)
                    _persist_step_results(db, execution, step_results)
                    continue

                except Exception as e:
                    step_result["status"] = "failed"
                    step_result["error"] = str(e)
                    step_result["duration_ms"] = int((time.time() - step_start) * 1000)
                    step_result["completed_at"] = datetime.now(timezone.utc).isoformat()
                    logger.error(f"[recipe_direct] generate_document step failed: {e}", exc_info=True)

                    if error_handling == "stop":
                        step_results.append(_build_compact_step_result(step_result))
                        _persist_step_results(db, execution, step_results)
                        await _fail_execution(db, recipe_execution_id, f"Document generation step failed: {e}", step_results=step_results)
                        return
                    elif error_handling == "skip":
                        step_results.append(_build_compact_step_result(step_result))
                        _persist_step_results(db, execution, step_results)
                        continue

            # Ensure git push credentials are set — uses the GITHUB PAT from
            # Railway env vars (long-lived, has push scope).  Runs once per
            # step that has a git-related pre_exec.
            if workspace_id and step.get("pre_exec") and "git" in step.get("pre_exec", ""):
                try:
                    from config import config as _cfg
                    github_pat = _cfg.GITHUB_PAT
                    github_owner = _cfg.GITHUB_REPO_OWNER or "AutomatosAI"
                    github_repo = _cfg.GITHUB_REPO_NAME or "automatos-ai"
                    if github_pat:
                        from core.workspace_client import WorkspaceClient
                        import re as _re

                        ws_client_cred = WorkspaceClient(str(workspace_id))
                        # Discover repo dir from pre_exec command
                        _m = _re.search(r"cd\s+(repos/[^\s&]+)", step.get("pre_exec", ""))
                        repo_dir = _m.group(1) if _m else f"repos/{github_repo}"

                        refresh_cmd = (
                            f"cd {repo_dir} && "
                            f"git remote set-url origin "
                            f"https://x-access-token:{github_pat}@github.com/{github_owner}/{github_repo}.git"
                        )
                        await ws_client_cred.exec_command(command=refresh_cmd, timeout=10)
                        logger.info("[recipe_direct] Step %d: set GitHub push credentials for %s/%s", step_order, github_owner, github_repo)

                        # Reset workspace to clean main — previous recipe steps
                        # (e.g. bug fixers) may leave the workspace on a dirty
                        # branch, causing "git checkout main" in the pre_exec to
                        # fail silently (masked by || true).
                        reset_cmd = (
                            f"cd {repo_dir} && "
                            f"git checkout -f main 2>&1; "
                            f"git clean -fd 2>&1"
                        )
                        reset_result = await ws_client_cred.exec_command(
                            command=reset_cmd, timeout=30,
                        )
                        logger.info(
                            "[recipe_direct] Step %d: workspace reset to main (exit=%s)",
                            step_order, reset_result.get("exit_code"),
                        )
                except Exception as cred_err:
                    # Non-blocking — push may still work with cached credentials
                    logger.info("[recipe_direct] Git setup skipped: %s", cred_err)

            # Pre-exec: run deterministic workspace command before the LLM loop.
            # The output is appended to the prompt so the LLM only does analysis.
            pre_exec_cmd = step.get("pre_exec")
            pre_exec_cwd = step.get("pre_exec_cwd")
            pre_exec_timeout = step.get("pre_exec_timeout", 600)
            if pre_exec_cmd and workspace_id:
                logger.info(f"[recipe_direct] Step {step_order} pre_exec: {pre_exec_cmd[:200]}")
                try:
                    from core.workspace_client import WorkspaceClient
                    ws_client = WorkspaceClient(str(workspace_id))
                    pre_result = await ws_client.exec_command(
                        command=pre_exec_cmd,
                        cwd=pre_exec_cwd,
                        timeout=pre_exec_timeout,
                    )
                    pre_exit = pre_result.get("exit_code", -1)
                    pre_stdout = pre_result.get("stdout", "")
                    pre_stderr = pre_result.get("stderr", "") or pre_result.get("error", "")
                    pre_duration = pre_result.get("duration_ms", 0)

                    # Log the pre_exec as a tool call for visibility
                    step_result["tool_calls"].append({
                        "action": "pre_exec",
                        "params": {"command": pre_exec_cmd, "cwd": pre_exec_cwd},
                        "result": f"exit_code={pre_exit} duration={pre_duration}ms",
                    })

                    # Append output to the prompt
                    pre_exec_block = (
                        f"\n\n## Pre-Exec Output (automated)\n"
                        f"Command: `{pre_exec_cmd}`\n"
                        f"Exit code: {pre_exit}\n"
                        f"Duration: {pre_duration}ms\n"
                    )
                    if pre_stdout:
                        # Keep head + tail to preserve both setup context and
                        # test summary (pytest prints results at the end).
                        max_chars = 10000
                        if len(pre_stdout) <= max_chars:
                            stdout_text = pre_stdout
                        else:
                            head = pre_stdout[:3000]
                            tail = pre_stdout[-6000:]
                            stdout_text = head + "\n\n... (truncated middle) ...\n\n" + tail
                        pre_exec_block += f"\n### stdout\n```\n{stdout_text}\n```\n"
                    if pre_stderr and pre_exit != 0:
                        pre_exec_block += f"\n### stderr\n```\n{pre_stderr[:2000]}\n```\n"

                    clean_step_prompt += pre_exec_block

                    # If pre_exec failed and error_handling is stop, abort
                    if pre_exit != 0 and error_handling == 'stop':
                        logger.warning(f"[recipe_direct] Step {step_order} pre_exec failed (exit={pre_exit})")
                        step_result["status"] = "failed"
                        step_result["error"] = f"pre_exec failed with exit code {pre_exit}: {pre_stderr[:500]}"
                        step_result["duration_ms"] = int((time.time() - step_start) * 1000)
                        step_result["completed_at"] = datetime.now(timezone.utc).isoformat()
                        step_results.append(_build_compact_step_result(step_result))
                        _persist_step_results(db, execution, step_results)
                        await _fail_execution(
                            db, recipe_execution_id,
                            f"Step {step_order} pre_exec failed: {pre_stderr[:200]}",
                            step_results=step_results,
                        )
                        return

                    logger.info(
                        f"[recipe_direct] Step {step_order} pre_exec completed: "
                        f"exit={pre_exit} duration={pre_duration}ms stdout={len(pre_stdout)} chars"
                    )
                except Exception as pre_exc:
                    logger.error(f"[recipe_direct] Step {step_order} pre_exec error: {pre_exc}", exc_info=True)
                    if error_handling == 'stop':
                        step_result["status"] = "failed"
                        step_result["error"] = f"pre_exec error: {pre_exc}"
                        step_result["duration_ms"] = int((time.time() - step_start) * 1000)
                        step_result["completed_at"] = datetime.now(timezone.utc).isoformat()
                        step_results.append(_build_compact_step_result(step_result))
                        _persist_step_results(db, execution, step_results)
                        await _fail_execution(db, recipe_execution_id, f"Step {step_order} pre_exec error: {pre_exc}", step_results=step_results)
                        return

            # Execute with retries
            attempt = 0
            success = False
            last_error = None
            exec_messages = []

            while attempt <= max_retries and not success:
                if attempt > 0:
                    logger.info(f"[recipe_direct] Retrying step {step_order}, attempt {attempt + 1}")
                    step_result["retries"] = attempt

                try:
                    result = await asyncio.wait_for(
                        _execute_step(
                            db=db,
                            agent=agent,
                            clean_prompt=clean_step_prompt,
                            workspace_id=workspace_id,
                            scratchpad=scratchpad,
                            step_order=step_order,
                            input_data=input_data,
                            recipe_memories=recipe_memories if idx == 0 else None,
                            prompt_for_hints=prompt_template,
                            max_iterations=step_max_iter,
                            recipe_name=recipe.name,
                            total_steps=total_steps,
                        ),
                        timeout=step_timeout_sec,
                    )

                    if result.get("status") == "success":
                        step_result["status"] = "completed"
                        raw_output = result.get("result", "")
                        if isinstance(raw_output, (dict, list)):
                            step_result["output"] = json.dumps(raw_output)
                        elif raw_output is not None:
                            step_result["output"] = str(raw_output)
                        else:
                            step_result["output"] = ""
                        step_result["tokens_used"] = result.get("execution", {}).get("tokens_used", 0)

                        tool_calls_raw = result.get("execution", {}).get("tool_calls", [])
                        step_result["tool_calls"] = _normalize_tool_calls(tool_calls_raw)
                        exec_messages = result.get("execution", {}).get("messages", [])

                        success = True

                        # Write to scratchpad (auto-extract)
                        agent_exports = scratchpad.get_exports() if scratchpad else {}
                        scratchpad.write_step_results(
                            step_order=step_order,
                            tool_calls=step_result["tool_calls"],
                            agent_output=step_result["output"],
                            agent_exports=agent_exports,
                        )

                        logger.info(f"[recipe_direct] Step {step_order} completed → output_key={output_key} ({step_result['tokens_used']} tokens)")
                    else:
                        last_error = result.get("error", "Agent returned non-success status")
                        logger.warning(f"[recipe_direct] Step {step_order} failed: {last_error}")

                except asyncio.TimeoutError:
                    last_error = f"Step timed out after {step_timeout_sec}s"
                    logger.warning(f"[recipe_direct] Step {step_order} timed out ({step_timeout_sec}s)")
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
                    step_results.append(_build_compact_step_result(step_result))
                    _persist_step_results(db, execution, step_results)
                    await _fail_execution(
                        db, recipe_execution_id,
                        f"Step {step_order} failed: {last_error}",
                        step_results=step_results
                    )
                    return
                elif error_handling == 'skip':
                    logger.info(f"[recipe_direct] Skipping failed step {step_order} (error_handling=skip)")
                    step_results.append(_build_compact_step_result(step_result))
                    _persist_step_results(db, execution, step_results)
                    continue
                else:
                    step_results.append(_build_compact_step_result(step_result))
                    _persist_step_results(db, execution, step_results)
                    await _fail_execution(
                        db, recipe_execution_id,
                        f"Step {step_order} failed after {attempt} attempts: {last_error}",
                        step_results=step_results
                    )
                    return

            # --- S3: upload full verbose log ---
            full_log = {
                "step_id": step_id,
                "order": step_order,
                "agent_id": agent_id,
                "agent_name": agent_name,
                "agent_output": step_result.get("output", ""),
                "tool_calls": step_result.get("tool_calls", []),
                "messages": exec_messages,
                "scratchpad_snapshot": scratchpad._hgetall() if scratchpad else {},
                "tokens_used": step_result.get("tokens_used", 0),
                "duration_ms": step_result.get("duration_ms", 0),
            }
            log_url = _upload_step_log_to_s3(workspace_id, recipe_execution_id, step_order, full_log)

            # --- DB: store compact summary ---
            compact = _build_compact_step_result(step_result, log_url=log_url)
            step_results.append(compact)
            _persist_step_results(db, execution, step_results)

            # Update board task progress
            try:
                from services.board_task_bridge import update_recipe_board_task_progress
                update_recipe_board_task_progress(db, recipe_execution_id, len(step_results), total_steps)
            except Exception:
                db.rollback()

        # All steps completed successfully
        total_duration = int((time.time() - execution_start) * 1000)
        total_tokens = sum(s.get("tokens_used", 0) for s in step_results)

        # Determine final output: last completed step's full output
        # (step_results are compact, but we still have the last step_result dict in scope)
        final_output = None
        if step_result.get("status") == "completed":
            final_output = step_result.get("output", "")
        if not final_output:
            for sr in reversed(step_results):
                if sr.get("status") == "completed":
                    final_output = sr.get("output_preview", "")
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

        # Complete the board task
        try:
            from services.board_task_bridge import complete_recipe_board_task
            complete_recipe_board_task(db, recipe_execution_id, success=True, result=str(final_output)[:4000])
        except Exception:
            db.rollback()
            logger.warning("Board task completion failed (non-blocking)", exc_info=True)

        logger.info(
            f"[recipe_direct] Execution {recipe_execution_id} COMPLETED — "
            f"{len(step_results)} steps, {total_duration}ms, {total_tokens} tokens"
        )

        # --- Post-execution: update agent performance_metrics ---
        _update_agent_performance_metrics(db, step_results, success=True)

        # --- Post-execution: learning + memory storage ---
        post_exec_config = recipe.execution_config or {}
        learning_result = None
        if post_exec_config.get('auto_learning') or post_exec_config.get('auto_learn', False):
            try:
                from core.services.recipe_learning_service import RecipeLearningService
                learning_svc = RecipeLearningService(db=db)
                learning_result = learning_svc.analyze_execution(recipe_execution_id)
                logger.info(f"[recipe_direct] Auto-learning completed for {recipe_execution_id}")
            except Exception as e:
                logger.warning(f"[recipe_direct] Auto-learning failed (non-blocking): {e}")

        # Store execution memories in Mem0
        try:
            from core.services.recipe_memory_service import RecipeMemoryService
            memory_svc = RecipeMemoryService(db=db)
            await memory_svc.store_execution_memory(
                recipe_execution_id,
                learnings=learning_result,
            )
            logger.info(f"[recipe_direct] Stored Mem0 memories for {recipe_execution_id}")
        except Exception as e:
            logger.info(f"[recipe_direct] Mem0 memory storage skipped: {e}")

    except Exception as e:
        logger.error(f"[recipe_direct] Fatal error in execution {recipe_execution_id}: {e}", exc_info=True)
        try:
            await _fail_execution(db, recipe_execution_id, str(e))
        except Exception as err:
            logger.exception(
                f"[recipe_direct] _fail_execution itself failed for {recipe_execution_id}: {err}"
            )
    finally:
        # Cleanup scratchpad TTL
        if scratchpad:
            try:
                scratchpad.cleanup()
            except Exception:
                pass
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
        last_entry = max(step_outputs.values(), key=lambda v: v.get("step_order", 0))
        previous_output = last_entry.get("text", "")

    if previous_output:
        resolved = resolved.replace("{previous_output}", previous_output)

    for key, entry in step_outputs.items():
        order = entry.get("step_order", 0)
        text = entry.get("text", "")
        if text:
            resolved = resolved.replace(f"{{step_{order}_output}}", text)

    for key, entry in step_outputs.items():
        text = entry.get("text", "")
        if text:
            resolved = resolved.replace(f"{{{key}}}", text)

    has_explicit_ref = (
        "{previous_output}" in template
        or any(f"{{step_{e.get('step_order', 0)}_output}}" in template for e in step_outputs.values())
        or any(f"{{{k}}}" in template for k in step_outputs)
    )

    if step_outputs and not has_explicit_ref:
        context_parts = []
        context_parts.append("=" * 60)
        context_parts.append("DATA FROM PREVIOUS STEPS")
        context_parts.append("When the task above mentions 'results', 'output', 'data',")
        context_parts.append("or 'findings', it refers to the content below.")
        context_parts.append("USE THIS CONTENT to complete the task — do not invent data.")
        context_parts.append("=" * 60)

        sorted_entries = sorted(step_outputs.items(), key=lambda kv: kv[1].get("step_order", 0))
        for out_key, entry in sorted_entries:
            sr_order = entry.get("step_order", "?")
            sr_agent = entry.get("agent_name", "Agent")
            sr_output = entry.get("text", "")
            sr_tool_calls = entry.get("tool_calls", [])

            context_parts.append(f"\n--- Step {sr_order} ({out_key}): {sr_agent} ---")

            if sr_tool_calls:
                for tc in sr_tool_calls:
                    action = tc.get("action", "unknown")
                    tc_result = tc.get("result", "")
                    if tc_result:
                        result_str = json.dumps(tc_result, indent=2) if isinstance(tc_result, (dict, list)) else str(tc_result)
                        if len(result_str) > 20000:
                            result_str = result_str[:20000] + "\n... (truncated)"
                        context_parts.append(f"[Tool: {action}]\n{result_str}")

            if sr_output:
                output_preview = sr_output[:12000]
                if len(sr_output) > 12000:
                    output_preview += "\n... (truncated)"
                context_parts.append(f"[Agent Output]\n{output_preview}")

        resolved = f"{resolved}\n\n" + "\n".join(context_parts)

    return resolved


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_doc_step_variables(data: Any, scratchpad) -> Any:
    """
    Resolve {{ step_N.field }} placeholders in document step data from scratchpad.
    Works recursively on dicts, lists, and strings.
    """
    import re

    if isinstance(data, str):
        # Replace {{ step_N.field }} or {{ step_N.output }}
        def _replace(match):
            step_num = int(match.group(1))
            field = match.group(2)
            ctx = scratchpad.format_context_for_step(step_num + 1)  # get results UP TO step_num
            return ctx if ctx else match.group(0)

        return re.sub(r"\{\{\s*step_(\d+)\.(\w+)\s*\}\}", _replace, data)
    elif isinstance(data, dict):
        return {k: _resolve_doc_step_variables(v, scratchpad) for k, v in data.items()}
    elif isinstance(data, list):
        return [_resolve_doc_step_variables(item, scratchpad) for item in data]
    return data


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

            # Fail the board task
            try:
                from services.board_task_bridge import complete_recipe_board_task as _complete_board
                _complete_board(db, execution_id, success=False, error_message=error_message)
            except Exception:
                db.rollback()

            logger.info(f"[recipe_direct] Execution {execution_id} marked FAILED: {error_message}")
            # Update agent performance_metrics for failure
            _update_agent_performance_metrics(db, step_results or [], success=False)
    except Exception as e:
        logger.error(f"[recipe_direct] Failed to mark execution as failed: {e}")
        db.rollback()


def _update_agent_performance_metrics(
    db: Session,
    step_results: List[dict],
    success: bool,
):
    """Update performance_metrics on every agent that participated in this execution.

    Increments tasks_completed, tracks success/failure counts, and recalculates
    success_rate.  Follows the same JSONB pattern as UsageTracker.track().
    """
    from sqlalchemy.orm.attributes import flag_modified

    # Collect unique agent IDs from step results
    agent_ids = list({
        s.get("agent_id") for s in step_results
        if s.get("agent_id")
    })
    if not agent_ids:
        return

    try:
        agents = db.query(Agent).filter(Agent.id.in_(agent_ids)).all()
        for agent in agents:
            metrics = dict(agent.performance_metrics or {})
            total = metrics.get("total_tasks_executed", 0) + 1
            successes = metrics.get("success_count", 0) + (1 if success else 0)
            failures = metrics.get("failure_count", 0) + (0 if success else 1)

            metrics["total_tasks_executed"] = total
            metrics["tasks_completed"] = total
            metrics["success_count"] = successes
            metrics["failure_count"] = failures
            metrics["success_rate"] = round(successes / total, 4) if total > 0 else 0
            metrics["last_task_at"] = datetime.now(timezone.utc).isoformat()
            metrics["last_task_success"] = success

            agent.performance_metrics = metrics
            flag_modified(agent, "performance_metrics")

        db.commit()
        logger.info(
            f"[recipe_direct] Updated performance_metrics for {len(agents)} agents "
            f"(success={success})"
        )
    except Exception as e:
        logger.warning(f"[recipe_direct] Agent metrics update failed (non-blocking): {e}")
        try:
            db.rollback()
        except Exception:
            pass
