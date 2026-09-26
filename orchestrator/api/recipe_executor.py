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
from contextlib import ExitStack
from datetime import datetime, timezone
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core.database.database import get_db, SessionLocal
from core.models import Agent
from core.models.core import PLAYBOOK_DOCUMENT_STEP, RecipeExecution, WorkflowTemplate as WorkflowRecipe
from core.security.surface import origin_surface, widget_scopes, widget_turn
from core.security.widget_scopes import widget_tool_surface
from core.services.playbook_scratchpad import answer_for_next_step
from core.services.playbook_step_refs import resolve_step_references, step_values

logger = logging.getLogger(__name__)

# F113 (run 4): a run's failure is what the owner reads on the ticket. A
# programming error ("'str' object has no attribute 'items'") is logged with
# its traceback and reported in words; an operational error (a provider's 402,
# a timeout) keeps its own message — the owner can act on that.
_INTERNAL_ERRORS = (AttributeError, TypeError, KeyError, IndexError, NameError,
                    AssertionError, ZeroDivisionError, RecursionError)
RUN_STOPPED_TEXT = (
    "The run stopped before it finished — the backend stopped while it was in progress. "
    "Session steps it had started carry on; their results are on the board."
)
INTERNAL_ERROR_TEXT = (
    "The run stopped on an internal error, so nothing after it ran. "
    "The details are in the server log."
)
# F155: a Claude Code session's tools are not the widget key's, so a run a
# widget turn started never files a session ticket.
WIDGET_SESSION_REFUSAL = "A playbook started from the website chat does not run on a Claude Code session."


def owner_error_text(exc: BaseException) -> str:
    """What the owner reads for an exception that stopped a run."""
    if isinstance(exc, _INTERNAL_ERRORS):
        return INTERNAL_ERROR_TEXT
    return str(exc) or type(exc).__name__


# ---------------------------------------------------------------------------
# PRD-128: Unified notification dispatch helper
# ---------------------------------------------------------------------------

async def _dispatch_playbook_event(
    db: Session,
    workspace_id: UUID,
    recipe_execution_id: str,
    event_type: str,
    title: str,
    message: Optional[str],
    agent_id: Optional[int] = None,
    agent_name: Optional[str] = None,
    status: str = "ok",
) -> None:
    """Fire a playbook-related event through NotificationDispatcher.

    Uses the executor's DB session so the notification row joins whatever
    commit follows (per-step persist or final completion). Failures are
    logged but never block playbook execution.
    """
    try:
        from core.services.notification_dispatcher import NotificationDispatcher

        dispatcher = NotificationDispatcher(db, str(workspace_id))
        await dispatcher.dispatch(
            event_type=event_type,
            title=title,
            message=message,
            link_type="playbook",
            link_id=recipe_execution_id,
            agent_id=agent_id,
            agent_name=agent_name,
            status=status,
        )
    except Exception:
        logger.error(
            "[recipe_direct] %s dispatch failed for %s",
            event_type,
            recipe_execution_id,
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# PRD-204 S3: playbook terminal -> watch registry (fail-soft)
# ---------------------------------------------------------------------------

def _ingest_playbook_terminal_watch(
    db: Session,
    execution,
    terminal_state: str,
    summary: Optional[str] = None,
) -> None:
    """Report a playbook execution's terminal state to its live watch.

    ONE seam for both the success block and ``_fail_execution`` so each
    terminal path flows through the same tested call. Fail-soft end to end:
    ``watch_ingest_terminal`` never raises into the executor.
    """
    from services.watch_hooks import watch_ingest_terminal

    output_data = getattr(execution, "output_data", None) or {}
    cost_snapshot = {
        "total_tokens": output_data.get("total_tokens", 0),
        "total_duration_ms": output_data.get("total_duration_ms", 0),
    }
    output_pointer = None
    if output_data.get("final_output"):
        output_pointer = (
            f"recipe_execution:{execution.execution_id}:output_data.final_output"
        )
    watch_ingest_terminal(
        db,
        workspace_id=execution.workspace_id,
        target_type="playbook_execution",
        target_id=execution.execution_id,
        terminal_state=terminal_state,
        summary=summary,
        cost_snapshot=cost_snapshot,
        output_pointer=output_pointer,
    )


# ---------------------------------------------------------------------------
# PRD-204 S7: per-execution step overrides (rerun tweak)
# ---------------------------------------------------------------------------

def _apply_step_overrides(
    steps: List[Dict[str, Any]],
    overrides: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge ``execution_metadata.step_overrides`` into the step list.

    Shape: ``{step_id: {"prompt_template": "..."}}``. Returns NEW dicts --
    ``workflow_recipes.steps`` (the shared definition) is never mutated; a
    tweak affects exactly this execution. Only ``prompt_template`` is
    honoured; anything else in an override entry is ignored.
    """
    if not overrides or not isinstance(overrides, dict):
        return [dict(step) for step in steps]

    merged: List[Dict[str, Any]] = []
    for step in steps:
        step_id = str(step.get("step_id") or "")
        override = overrides.get(step_id)
        prompt = override.get("prompt_template") if isinstance(override, dict) else None
        if isinstance(prompt, str) and prompt.strip():
            merged.append({**step, "prompt_template": prompt})
            logger.info(
                "[recipe_direct] step %s prompt overridden for this execution "
                "(PRD-204 S7 tweak)",
                step_id,
            )
        else:
            merged.append(dict(step))
    return merged


# ---------------------------------------------------------------------------
# Auto-report on playbook completion (mirrors heartbeat & task auto-reports)
# ---------------------------------------------------------------------------
async def _auto_create_playbook_report(
    *,
    db: Session,
    workspace_id: str,
    recipe,
    recipe_execution_id: str,
    execution,
    step_results: List[dict],
    total_duration_ms: int,
    total_tokens: int,
    final_output: Any,
    success: bool,
) -> None:
    """Persist an agent_reports row summarising a playbook execution.

    File path: reports/playbook-{slug}/{date}_{exec_id}.md
    Always non-blocking — never raises.
    """
    try:
        from services.report_service import ReportService, compute_execution_metrics

        recipe_name = getattr(recipe, "name", None) or f"playbook-{recipe.id}"
        report_agent_name = f"playbook-{recipe_name}"

        # Roll up cost/model/duration across every LLM call in this execution
        exec_metrics = compute_execution_metrics(
            db,
            workspace_id,
            execution_id=recipe_execution_id,
            started_at=getattr(execution, "started_at", None),
            completed_at=getattr(execution, "completed_at", None),
            extra={
                "recipe_id": getattr(recipe, "id", None),
                "recipe_name": recipe_name,
                "recipe_execution_id": recipe_execution_id,
                "steps_count": len(step_results),
                "trigger": "playbook",
            },
        )
        # Honour totals computed by the executor when llm_usage rollup is sparse
        if not exec_metrics.get("tokens_used"):
            exec_metrics["tokens_used"] = total_tokens
        if exec_metrics.get("duration_ms") is None:
            exec_metrics["duration_ms"] = total_duration_ms

        report_status = "ok" if success else ("warning" if step_results else "critical")
        any_failed = any(
            s.get("status") in ("failed", "error") for s in step_results
        )
        if any_failed and report_status == "ok":
            report_status = "warning"

        # Markdown body
        lines = [
            f"# {recipe_name} — Playbook Report",
            f"**Execution:** {recipe_execution_id}",
            f"**Status:** {'completed' if success else 'failed'}",
            "",
            "## Execution Metrics",
            f"- Primary model: {exec_metrics.get('model') or 'unknown'}",
            f"- LLM calls: {exec_metrics.get('llm_calls', 0)}",
            f"- Tokens (in/out/total): "
            f"{exec_metrics.get('input_tokens', 0)} / "
            f"{exec_metrics.get('output_tokens', 0)} / "
            f"{exec_metrics.get('tokens_used', 0)}",
            f"- Cost: ${exec_metrics.get('cost_usd', 0):.4f}",
            f"- Duration: {exec_metrics.get('duration_ms', 0)} ms",
            f"- Steps: {len(step_results)}",
            "",
            "## Steps",
        ]
        for step in step_results:
            order = step.get("order") or step.get("step_order") or "?"
            name = step.get("name") or step.get("agent_name") or "(unnamed)"
            status = step.get("status", "?")
            duration = step.get("duration_ms")
            tokens = step.get("tokens_used", 0)
            duration_str = f"{duration} ms" if duration is not None else "n/a"
            lines.append(
                f"- **#{order} {name}** — {status} · {tokens} tokens · {duration_str}"
            )

        models_used = exec_metrics.get("models_used") or []
        if models_used:
            lines.append("")
            lines.append("## Models Used")
            for m in models_used:
                lines.append(f"- {m}")

        if final_output:
            preview = str(final_output)[:500]
            if len(str(final_output)) > 500:
                preview += "…"
            lines.append("")
            lines.append("## Final Output (preview)")
            lines.append("```")
            lines.append(preview)
            lines.append("```")

        content = "\n".join(lines)

        first_step_summary = next(
            (s.get("output_preview") for s in step_results if s.get("output_preview")),
            None,
        )
        summary = (
            f"{len(step_results)} steps · "
            f"${exec_metrics.get('cost_usd', 0):.4f} · "
            f"{exec_metrics.get('duration_ms', 0)} ms"
        )
        if first_step_summary:
            summary = f"{summary} · {str(first_step_summary)[:100]}"

        svc = ReportService(db, workspace_id)
        report_result = await svc.create_report(
            agent_id=None,
            agent_name=report_agent_name,
            title=f"Playbook: {recipe_name}",
            content=content,
            report_type="summary",
            status=report_status,
            summary=summary,
            metrics=exec_metrics,
        )
        if not report_result.get("success"):
            logger.warning(
                "[recipe_direct] Playbook auto-report DB insert failed for %s: %s",
                recipe_execution_id, report_result.get("error"),
            )
    except Exception:
        logger.error(
            "[recipe_direct] _auto_create_playbook_report raised for %s",
            recipe_execution_id,
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# Per-workspace execution semaphore — allows bounded concurrent recipe
# execution within a workspace.  Keys are workspace_id strings; values are
# asyncio.Semaphores created on first access.  The dict itself is
# process-global but safe because asyncio is single-threaded.
# ---------------------------------------------------------------------------
_workspace_semaphores: Dict[str, asyncio.Semaphore] = {}

# Process-local registry of currently running execution tasks. Lets the cancel
# endpoint signal the in-flight LLM call to abort immediately (httpx propagates
# CancelledError, which closes the TCP connection mid-request — no more cost
# burn). Cross-replica cancels are handled by the DB-poll fallback inside
# _execute_step.
_running_executions: Dict[str, asyncio.Task] = {}


def request_execution_cancel(execution_id: str) -> bool:
    """Signal a cancel to a locally-running execution. Returns True if the
    task was found and cancelled on this replica, False otherwise."""
    task = _running_executions.get(execution_id)
    if task is None or task.done():
        return False
    task.cancel()
    return True


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

def _is_session_step(db, agent) -> bool:
    """Whether this step's agent runs as a Claude Code session (runtime: cli)."""
    from services.cli_ticket_lane import is_cli_agent

    try:
        return is_cli_agent(db, getattr(agent, "id", None))
    except Exception:  # noqa: BLE001
        return False


def _step_deadline(step_timeout_sec: float, session_step: bool, cfg) -> float:
    """PRD-239 S3b: an API step keeps the recipe's step timeout; a session step
    gets at least CLI_LANE_STEP_TIMEOUT_SECONDS — a real Claude Code session
    takes minutes, and 300 s produced "timed out → retry → stalled"."""
    if not session_step:
        return float(step_timeout_sec)
    return max(float(step_timeout_sec), float(getattr(cfg, "CLI_LANE_STEP_TIMEOUT_SECONDS", 1800)))


def _progress_stamped(metadata) -> Dict[str, Any]:
    """``metadata`` with the run's last progress time (naive UTC, like started_at):
    a new dict. The reconciler's stall rule reads it."""
    from datetime import datetime as _dt

    return {**dict(metadata or {}), "last_progress_at": _dt.utcnow().replace(microsecond=0).isoformat()}


def _stamp_progress(db, execution) -> None:
    """Record that the run is alive (execution_metadata.last_progress_at). Rebuild
    the JSONB, never mutate; a failure to stamp must never end the step."""
    try:
        execution.execution_metadata = _progress_stamped(getattr(execution, "execution_metadata", None))
        db.commit()
    except Exception:  # noqa: BLE001
        logger.debug("[recipe_direct] progress stamp failed", exc_info=True)


async def _stamping_progress(work: Awaitable[Any], stamp: Callable[[], None], every_seconds: float) -> Any:
    """Await ``work``, calling ``stamp`` every ``every_seconds`` while it runs.

    A fixed generate_document step waits on media-render for minutes (PRD-251
    US-120): without a fresh stamp the reconciler fails the run as stalled after
    TASK_STALL_TIMEOUT_SECONDS and starts it again, render and all.
    """
    async def beat() -> None:
        while True:
            await asyncio.sleep(every_seconds)
            stamp()

    beating = asyncio.ensure_future(beat())
    try:
        return await work
    finally:
        beating.cancel()


def _cli_step_title(recipe_name: str, step_order: int, clean_prompt: str) -> str:
    """The ticket title for a playbook step a session agent works (PRD-239 S3)."""
    first = next((line.strip() for line in (clean_prompt or "").splitlines() if line.strip()), "")
    if len(first) > 60:
        first = first[:60].rstrip() + "…"
    head = f"{recipe_name or 'Playbook'} · step {step_order}"
    return f"{head}: {first}" if first else head


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
    max_iterations: Optional[int] = None,
    recipe_name: str = "",
    total_steps: int = 1,
    recipe_execution_id: Optional[str] = None,
    progress: Optional[Callable[[], None]] = None,
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
            agent.configuration.max_iterations. If None, falls back to
            system_settings recipe.default_max_iterations.

    Returns:
        Dict with status, result, and execution metadata.
    """
    if max_iterations is None:
        from config import config as _app_config
        max_iterations = _app_config.RECIPE_DEFAULT_MAX_ITERATIONS

    # PRD-239 S3: a session agent's step is a ticket its Claude Code session
    # works; the step waits for it to end (the caller's step timeout bounds the
    # wait — the session carries on and its result lands on the board). Before
    # the tool/LLM imports: a session step touches none of them.
    # The earlier steps' answers, for either kind of agent (F130: a session step
    # got none of them and the next api step got 500 characters).
    previous_output = ""
    if scratchpad and step_order > 1:
        previous_output = scratchpad.format_context_for_step(step_order) or ""

    from services.cli_ticket_lane import RECIPE_SOURCE_TYPE, is_cli_agent, run_cli_ticket_and_wait
    if is_cli_agent(db, agent.id):
        if widget_turn():
            return {"status": "error", "error": WIDGET_SESSION_REFUSAL,
                    "execution": {"tokens_used": 0, "tool_calls": [], "messages": []}}
        import uuid as _uuid

        return await run_cli_ticket_and_wait(
            db,
            workspace_id=workspace_id,
            agent_id=agent.id,
            title=_cli_step_title(recipe_name, step_order, clean_prompt),
            # A session reads only its ticket: the earlier answers go in it.
            prompt=f"{clean_prompt}\n\n{previous_output}" if previous_output else clean_prompt,
            source_type=RECIPE_SOURCE_TYPE,
            source_id=f"{RECIPE_SOURCE_TYPE}:{recipe_execution_id or _uuid.uuid4().hex}:{step_order}",
            tags=["playbook"],
            # S3b: the run keeps marking progress while the session works, so the
            # stall watchdog (started_at + 300 s) does not fail a live playbook.
            on_poll=(lambda _ticket: progress()) if progress is not None else None,
        )

    # Lazy imports to avoid circular deps
    from modules.tools.tool_router import get_tool_router
    from modules.tools.services.composio_hint_service import ComposioHintService
    from modules.tools.services.composio_tool_service import ComposioToolService
    from core.composio.tool_executor import resolve_file_uploads
    from core.composio.client import get_composio_client
    from core.composio.deny_list import composio_action_denial_async
    from core.composio.off_loop import ComposioLookupTimeout, composio_lookup
    from core.composio.post_gate import post_action_refusal
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
    from services.playbook_owner_ask import ASK_ACTION, NOT_RUN_AFTER_ASK, ask_call_params, take_ask

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
        query=prompt_for_hints or clean_prompt,
        input_data=input_data,
        recipe_memories=recipe_memories if step_order == 1 else None,
    )

    messages = [{"role": "system", "content": context.system_prompt}]
    base_tools = context.tools

    # 2. Composio tools — SDK semantic search for per-action function-calling tools.
    #    Falls back to hint-based composio_execute if SDK search returns empty.
    #    F155: none on a widget turn — the owner's connected apps are not the
    #    widget key's (the widget chat is never offered them either).
    #    F105: both lookups run off the loop, with a session of their own
    #    (core.composio.off_loop), so they read nothing through `agent`.
    composio_result = None
    composio_timed_out = False
    lookup_agent_id = agent.id
    lookup_prompt = prompt_for_hints or clean_prompt
    if not widget_turn():
        try:
            composio_result = await composio_lookup(
                lambda session: ComposioToolService(session).get_tools_for_step(
                    agent_id=lookup_agent_id,
                    workspace_id=workspace_id,
                    task_prompt=lookup_prompt,
                ),
                step=f"playbook step (agent {lookup_agent_id}): tool search",
            )
        except ComposioLookupTimeout:
            composio_timed_out = True  # warned where it gave up; the hints would wait on the same SDK
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
    elif not widget_turn() and not composio_timed_out:
        # Fallback: existing hint service with composio_execute mega-tool
        if composio_result:
            composio_result.strategy = "hint_fallback"
        try:
            hint_result = await composio_lookup(
                lambda session: ComposioHintService(session).build_hints(
                    agent_id=lookup_agent_id,
                    prompt=lookup_prompt,
                    workspace_id=workspace_id,
                    recipe_mode=True,
                ),
                step=f"playbook step (agent {lookup_agent_id}): action hints",
            )
            if hint_result.hint_lines:
                messages.append({"role": "system", "content": "\n".join(hint_result.hint_lines)})
                logger.info(
                    f"[recipe_step] Hints fallback: strategy={hint_result.strategy_used} "
                    f"actions={len(hint_result.matched_actions)}"
                )
        except ComposioLookupTimeout:
            pass  # warned where it gave up; the step goes on without Composio tools
        except Exception as exc:
            logger.warning(f"[recipe_step] Hint injection failed: {exc}", exc_info=True)

    # 3. Scratchpad context — now handled via recipe_step.previous_output
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
    if widget_turn():
        # F155: what the widget key's scopes allow (the executor refuses
        # anything else on the resolved action).
        tools = widget_tool_surface(tools, widget_scopes())
    if scratchpad:
        scratchpad_tools = [SCRATCHPAD_WRITE_TOOL_DEF]
        if step_order > 1:
            scratchpad_tools.append(SCRATCHPAD_READ_TOOL_DEF)
        tools = list(tools) + scratchpad_tools

    # 6. LLM — agent's own LLM manager (with tracking context for recipe)
    llm = agent_runtime.llm_manager
    if hasattr(llm, '_tracking_ctx'):
        llm._tracking_ctx["request_type"] = "recipe"
        if recipe_execution_id:
            llm._tracking_ctx["execution_id"] = recipe_execution_id

    # 7. Generate + tool loop
    tool_router = get_tool_router()
    all_tool_calls = []
    _composio_call_cache: Dict[str, str] = {}  # dedup: "ACTION|args_hash" → cached result
    response = None
    owner_ask: Optional[Dict[str, Any]] = None  # F140: the step's question to the owner

    for iteration in range(max_iterations):
        if recipe_execution_id:
            db.expire_all()
            cancel_status = db.query(RecipeExecution.status).filter(
                RecipeExecution.execution_id == recipe_execution_id
            ).scalar()
            if cancel_status == "cancelled":
                logger.info(
                    "[recipe_direct] Step %d cancelled mid-flight at iteration %d (execution=%s)",
                    step_order, iteration, recipe_execution_id,
                )
                return {
                    "status": "cancelled",
                    "result": "Cancelled by user",
                    "error": "Cancelled by user",
                    "execution": {"tool_calls": all_tool_calls, "messages": messages, "tokens_used": 0},
                }

        response = await llm.generate_response(messages=messages, tools=tools)

        # Detect empty-choices responses (OpenRouter intermittency): the call
        # returns successfully but with no content AND no tool_calls. Without
        # retry, the tool loop below would treat empty as "LLM done" and the
        # step would finish without ever emitting its final tool call (e.g.
        # platform_submit_report). Retry with backoff, then swap to a fallback
        # model if the agent's primary model is in a sustained empty-choices
        # window (e.g. Gemini 2.5 Pro degradation).
        if response and not response.tool_calls and not (response.content or "").strip():
            from config import config as _app_config
            empty_retry_budget = _app_config.RECIPE_EMPTY_COMPLETION_RETRY_BUDGET
            for retry_idx in range(empty_retry_budget):
                backoff_s = min(0.5 * (2 ** retry_idx), 4.0)
                logger.warning(
                    "[recipe_direct] Empty completion at step %d iter %d (retry %d/%d, backoff=%.1fs) "
                    "— content+tool_calls both empty, retrying",
                    step_order, iteration, retry_idx + 1, empty_retry_budget, backoff_s,
                )
                await asyncio.sleep(backoff_s)
                response = await llm.generate_response(messages=messages, tools=tools)
                if response and (response.tool_calls or (response.content or "").strip()):
                    break  # got a real response

            # If still empty after same-model retries, swap to fallback model
            # for one final attempt. This survives sustained per-model provider
            # degradation that affects the primary model only.
            if response and not response.tool_calls and not (response.content or "").strip():
                fallback_model = _app_config.RECIPE_EMPTY_COMPLETION_FALLBACK_MODEL
                if fallback_model:
                    logger.warning(
                        "[recipe_direct] Empty completion persists at step %d iter %d after %d retries "
                        "— swapping to fallback model %s",
                        step_order, iteration, empty_retry_budget, fallback_model,
                    )
                    try:
                        from core.llm.manager import create_llm_manager
                        fallback_llm = create_llm_manager(
                            service_name="orchestrator",
                            model=fallback_model,
                            request_type="recipe_fallback",
                        )
                        if recipe_execution_id and hasattr(fallback_llm, "_tracking_ctx"):
                            fallback_llm._tracking_ctx["execution_id"] = recipe_execution_id
                        response = await fallback_llm.generate_response(messages=messages, tools=tools)
                    except Exception as fallback_exc:
                        logger.error(
                            "[recipe_direct] Fallback model %s also failed at step %d iter %d: %s",
                            fallback_model, step_order, iteration, fallback_exc,
                        )

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
            except json.JSONDecodeError as jde:
                logger.warning(
                    "[recipe_direct] JSON decode failed for %s args (len=%d, first 200=%r): %s",
                    tool_name,
                    len(tool_args_raw) if isinstance(tool_args_raw, str) else 0,
                    tool_args_raw[:200] if isinstance(tool_args_raw, str) else tool_args_raw,
                    jde,
                )
                tool_args = {}

            # F140: a step asks the owner through its run, never through a subject
            # it would have to name (night 4: B15/B23 were refused, and five of six
            # calls came with params={}). The run stops after this turn.
            ask_params = ask_call_params(tool_name, tool_args) if recipe_execution_id else None
            if ask_params is not None:
                ask, result_text = take_ask(ask_params)
                owner_ask = owner_ask or ask
                all_tool_calls.append({"action": ASK_ACTION, "params": ask_params, "result": result_text,
                                       "success": ask is not None})
                messages.append({"role": "tool", "tool_call_id": tool_id, "content": result_text})
                continue
            if owner_ask is not None:  # nothing acts after the step asked
                all_tool_calls.append({"action": tool_args.get("action", tool_name),
                                       "params": tool_args.get("params", tool_args),
                                       "result": NOT_RUN_AFTER_ASK, "success": False})
                messages.append({"role": "tool", "tool_call_id": tool_id, "content": NOT_RUN_AFTER_ASK})
                continue

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
                    "success": True,
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
                    "success": True,
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": result_text,
                })
                continue

            # Per-action Composio execution (SDK search path). Also catches
            # LLM-inferred actions beyond search results — if the tool name
            # matches a connected app prefix (e.g. JIRA_*). PRD-192 S5: these
            # steps now ride the SPINE (execute_and_format → UnifiedToolExecutor
            # as the composio_execute meta-tool) instead of calling
            # ComposioToolService.execute_action raw — the policy gate,
            # telemetry, and typed outcome capture govern the daily playbook
            # lane. Dedup, the LinkedIn workaround, and file-upload resolution
            # are unchanged; the step-level result envelope keeps its shape for
            # the transcript writer below.
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
                # PRD-251 S0.6 (D16): a denied action never runs — not from the
                # dedup cache, the LinkedIn workaround, file uploads or the spine.
                _denial = await composio_action_denial_async(tool_name)
                call_ok = False  # F137: whether the call worked, never guessed from its text
                # PRD-251 S3.5 (D14b): with Socials on, the step's agent drafts a
                # post and a person approves it; it never publishes one directly —
                # not from the dedup cache, the LinkedIn workaround or the spine.
                # After the deny list, which always wins.
                _post_refusal = None if _denial else await post_action_refusal(tool_name, workspace_id)
                if _denial:
                    result_text = f"Error executing {tool_name}: {_denial}"
                    exec_ms = 0
                    logger.warning(f"[recipe_step] Composio deny list refused {tool_name}")
                elif _post_refusal:
                    result_text = f"Error executing {tool_name}: {_post_refusal}"
                    exec_ms = 0
                    logger.warning(f"[recipe_step] Socials post gate refused {tool_name}")
                elif _dedup_key in _composio_call_cache:
                    result_text, call_ok = _composio_call_cache[_dedup_key]
                    exec_ms = 0
                    logger.info(f"[recipe_step] Composio dedup hit: {tool_name} (skipped repeat call)")
                else:
                    _temp_files: list = []
                    try:
                        t0 = time.time()

                        # --- WORKAROUND: Composio LinkedIn image upload is broken (#3094, #3113) ---
                        # Remove when Composio fixes — see linkedin_image_workaround.py
                        _tool_upper = tool_name.upper()
                        if _tool_upper == "LINKEDIN_CREATE_LINKED_IN_POST":
                            from core.composio.linkedin_image_workaround import has_image_params, execute_linkedin_image_post
                            if has_image_params(tool_args):
                                logger.info("[LinkedInWorkaround] Intercepting %s in recipe", _tool_upper)
                                try:
                                    exec_result = await execute_linkedin_image_post(
                                        params=tool_args,
                                        workspace_id=workspace_id,
                                        entity_id=composio_result.entity_id,
                                        composio_client=get_composio_client(),
                                    )
                                except Exception as wa_exc:
                                    logger.error("[LinkedInWorkaround] Exception: %s", wa_exc, exc_info=True)
                                    exec_result = {"success": False, "data": None, "error": str(wa_exc)}
                                exec_ms = int((time.time() - t0) * 1000)
                                success = exec_result.get("success", False)
                                data = exec_result.get("data")
                                error = exec_result.get("error")
                                if success:
                                    result_text = json.dumps(data, default=str) if isinstance(data, (dict, list)) else str(data or "")
                                    logger.info(f"[recipe_step] LinkedIn workaround OK in {exec_ms}ms")
                                else:
                                    result_text = f"Error executing {tool_name}: {error or 'unknown error'}"
                                    logger.warning(f"[recipe_step] LinkedIn workaround failed: {error}")
                                call_ok = bool(success)
                                _composio_call_cache[_dedup_key] = (result_text, call_ok)
                                all_tool_calls.append({
                                    "action": tool_name, "params": tool_args,
                                    "result": result_text[:8000], "duration_ms": exec_ms,
                                    "composio_direct": True, "success": call_ok,
                                })
                                messages.append({
                                    "role": "tool", "tool_call_id": tool_id,
                                    "content": result_text[:20000],
                                })
                                continue

                        tool_args, _temp_files = await resolve_file_uploads(
                            action=tool_name,
                            params=tool_args,
                            workspace_id=workspace_id,
                        )
                        # PRD-192 S5: dispatch through the spine — the per-action
                        # name rides in `action`, so the gate's effective-name
                        # resolution, the tracker, and the audit row all see the
                        # real action; the executor enforces assigned/connected/
                        # mapped and resolves the workspace entity itself.
                        spine_result = await tool_router.execute_and_format(
                            tool_name="composio_execute",
                            tool_args={"action": tool_name, "params": tool_args},
                            agent_id=agent.id,
                            workspace_id=workspace_id,
                            original_intent=clean_prompt,
                            caller_context={
                                "playbook_execution_id": recipe_execution_id,
                                "playbook_step": step_order,
                            },
                        )
                        exec_ms = int((time.time() - t0) * 1000)

                        raw = spine_result.get("raw_result") or {}
                        success = bool(spine_result.get("success"))
                        call_ok = success
                        data = raw.get("data") if isinstance(raw, dict) else None
                        error = (
                            (raw.get("error") if isinstance(raw, dict) else None)
                            or spine_result.get("llm_context")
                        )

                        if success:
                            result_text = json.dumps(data, default=str) if isinstance(data, (dict, list)) else str(data or spine_result.get("llm_context") or "")
                            logger.info(f"[recipe_step] Composio spine OK: {tool_name} in {exec_ms}ms")
                        else:
                            result_text = f"Error executing {tool_name}: {error or 'unknown error'}"
                            logger.warning(f"[recipe_step] Composio spine failed: {tool_name} — {error}")
                    except Exception as exc:
                        exec_ms = int((time.time() - t0) * 1000)
                        result_text = f"Error executing {tool_name}: {exc}"
                        logger.error(f"[recipe_step] Composio direct exception: {tool_name}: {exc}", exc_info=True)
                    finally:
                        for tf in _temp_files:
                            try:
                                tf.unlink(missing_ok=True)
                            except Exception:
                                pass
                    _composio_call_cache[_dedup_key] = (result_text, call_ok)

                all_tool_calls.append({
                    "action": tool_name,
                    "params": tool_args,
                    "result": result_text[:8000],
                    "duration_ms": exec_ms,
                    "composio_direct": True,
                    "success": call_ok,
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
                "success": result.get("success") is not False,
            })

            messages.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "content": result.get("llm_context", "No result"),
            })

        logger.info(f"[recipe_step] Tool iteration {iteration + 1}: {len(response.tool_calls)} calls")
        if owner_ask is not None:
            break  # F140: the owner is asked; this step goes no further

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
        "owner_ask": owner_ask,
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
        from config import config
        from core.storage import ensure_bucket, get_s3_client

        bucket = config.RECIPE_LOG_S3_BUCKET
        if not bucket:
            return None

        s3_key = f"workspaces/{workspace_id}/logs/executions/{execution_id}/step_{step_order}.json"

        ensure_bucket(bucket)
        s3 = get_s3_client()

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
        # F137: the call's own success flag. B22's "Tool composio_execute failed: …"
        # holds no "error", so the text guess showed four failed drafts as (success).
        worked = tc.get("success")
        if worked is None:  # a record from before the flag: the old guess
            worked = "error" not in str(tc.get("result", "")).lower()[:100]
        tool_summaries.append(f"{action} ({'success' if worked else 'error'})")

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
                    await _fail_execution(db, recipe_execution_id, f"Task crashed: {owner_error_text(e)}")
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

    current_task = asyncio.current_task()
    if current_task is not None:
        _running_executions[recipe_execution_id] = current_task

    async with semaphore:
        logger.info(
            "[recipe_direct] Semaphore ACQUIRED for %s (execution=%s, recipe=%s, was_waiting=%s)",
            workspace_id, recipe_execution_id, recipe_id, waiting,
        )
        try:
            await _execute_recipe_inner(
                recipe_execution_id, recipe_id, workspace_id, input_data, db_url,
            )
        except asyncio.CancelledError:
            logger.info(
                "[recipe_direct] CancelledError received — execution %s aborted mid-flight",
                recipe_execution_id,
            )
            await _mark_execution_cancelled(recipe_execution_id, db_url)
            # Do not re-raise — we've handled it cleanly. Task can complete.
        finally:
            _running_executions.pop(recipe_execution_id, None)
            logger.info(
                "[recipe_direct] Semaphore RELEASED for %s (execution=%s)",
                workspace_id, recipe_execution_id,
            )


async def _mark_execution_cancelled(execution_id: str, db_url: Optional[str]) -> None:
    """Ensure the execution row reflects cancellation. The cancel endpoint
    already writes status='cancelled', but if the kill came via task.cancel()
    without going through the endpoint (or the endpoint hasn't committed yet),
    we patch it here defensively."""
    try:
        if db_url:
            _engine = create_engine(db_url)
            _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
            db = _SessionLocal()
        else:
            db = SessionLocal()
        try:
            execution = db.query(RecipeExecution).filter(
                RecipeExecution.execution_id == execution_id
            ).first()
            if execution and execution.status not in ("completed", "failed", "cancelled"):
                # The cancel endpoint writes its row before it signals the task, so
                # a run still open here was stopped by something else — a shutdown.
                # (``sa_func`` was never imported here: until now this NameError'd
                # into the warning below and the run stayed 'running'.)
                execution.status = "cancelled"
                execution.error_message = execution.error_message or RUN_STOPPED_TEXT
                execution.completed_at = datetime.now(timezone.utc)
                db.commit()
        finally:
            db.close()
    except Exception:
        logger.warning(
            "[recipe_direct] Failed to mark execution %s cancelled (best-effort)",
            execution_id, exc_info=True,
        )


def _tokens_to_usd(tokens: int, app_config) -> float:
    """Price tokens with the same convention the mission dollar-ceiling uses
    (PRD-163 S5) so a playbook's $ budget is denominated identically to a
    mission's. PRD-192 S3 (F059 finish): routed through
    ``modules.policy.pricing`` — the ONE pricing source; a playbook run spans
    per-step agents/models so pricing's documented flat last-resort applies.
    ``app_config`` kept for signature stability at the call sites."""
    from modules.policy import pricing as _pricing

    return _pricing.price_total_tokens_usd(None, None, tokens)


def _playbook_cost_ceiling_usd(exec_config: dict, app_config) -> float:
    """The playbook run's DOLLAR ceiling: an explicit ``execution_config`` value,
    else 0 (unlimited). Mirrors the mission ``config['cost_ceiling']`` convention."""
    ceiling = (exec_config or {}).get("cost_ceiling")
    if ceiling is None:
        ceiling = (exec_config or {}).get("cost_ceiling_usd")
    if isinstance(ceiling, (int, float)) and ceiling > 0:
        return float(ceiling)
    return 0.0


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
    origin = ExitStack()
    try:
        logger.info(f"[recipe_direct] Starting execution {recipe_execution_id} for recipe {recipe_id}")

        # F113: whatever the trigger stored (a string from a tool call, a retried
        # row), the steps read key-value pairs.
        from core.services.playbook_inputs import playbook_inputs, with_input_defaults

        input_data, _input_problem = playbook_inputs(input_data)
        if _input_problem:
            await _fail_execution(db, recipe_execution_id, f"This run's input could not be read: {_input_problem}")
            return

        # Load recipe and execution
        recipe = db.query(WorkflowRecipe).filter(WorkflowRecipe.id == recipe_id).first()
        if not recipe:
            await _fail_execution(db, recipe_execution_id, "Recipe not found")
            return
        # A scheduled run, a tool's run and a retry carry only what they were
        # given: the playbook's declared defaults fill the rest, as the run route's do.
        input_data = with_input_defaults(input_data, getattr(recipe, "inputs", None))

        execution = db.query(RecipeExecution).filter(
            RecipeExecution.execution_id == recipe_execution_id
        ).first()
        if not execution:
            logger.error(f"[recipe_direct] Execution record not found: {recipe_execution_id}")
            return
        # F155: a run a widget turn started (its origin is on the row, server-set)
        # runs under that turn's key scopes and team lock, retries and reruns too.
        origin.enter_context(origin_surface(execution.execution_metadata))

        # Disabled / deleted workspace gate — covers scheduled runs that
        # bypass the request-context middleware. The HTTP entry point also
        # checks this, but a scheduled job triggers execute_recipe_direct
        # directly so we must guard here too.
        from core.models.workspaces import Workspace as _Workspace
        ws_state = db.query(_Workspace).filter(_Workspace.id == workspace_id).first()
        if not ws_state or ws_state.deleted_at is not None:
            await _fail_execution(
                db, recipe_execution_id,
                "Workspace not found or deleted",
            )
            return
        if ws_state.paused_at is not None:
            logger.warning(
                "[recipe_direct] Refusing to run — workspace %s is disabled (reason=%s)",
                workspace_id, ws_state.paused_reason,
            )
            await _fail_execution(
                db, recipe_execution_id,
                "Workspace is disabled. Contact an administrator.",
            )
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

        # PRD-204 S7: merge per-execution prompt overrides (rerun tweak) at
        # execution start. The recipe row is never written back to.
        step_overrides = (execution.execution_metadata or {}).get("step_overrides")
        if step_overrides:
            steps = _apply_step_overrides(steps, step_overrides)

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
        # F135 (B67, B87): agent #303 was switched off at 14:08:11 and ran a step at 14:08:16.
        switched_off = [a for a in agents if (a.status or "active") != "active"]
        if switched_off:
            names = ", ".join(f"#{a.id} {a.name} ({a.status})" for a in switched_off)
            await _fail_execution(
                db, recipe_execution_id,
                f"Switched off: {names}. Switch the agent on, or give its steps another agent. Nothing ran.")
            return

        # F182 (night 6): what the run needs is checked once, before step 1. Run
        # 207 started with no inputs, and its first step wrote to another café's
        # contact from memory. A declared default fills in ('' never does); a
        # required input still missing stops the run and asks the owner for it
        # by name (F140's stop), and their answer's rerun is given it.
        from core.services.playbook_inputs import input_contract, inputs_question, missing_inputs, with_defaults

        contract = input_contract(recipe.inputs, steps)
        input_data = with_defaults(contract, input_data)
        needed = missing_inputs(contract, input_data)
        if needed:
            from services.playbook_owner_ask import NEEDS_YOU, stop_for_owner

            first = steps[0]
            first_agent = agent_map.get(first.get('agent_id'))
            question = inputs_question(recipe.name, needed, contract)
            asked = await stop_for_owner(
                db, execution=execution, recipe=recipe, step_order=first.get('order', 1),
                agent_id=first.get('agent_id'), agent_name=getattr(first_agent, "name", None),
                ask={"question": question, "options": None}, step_results=[], step_calls=[], inputs=needed,
            )
            if not asked:
                await _fail_execution(db, recipe_execution_id, f"{NEEDS_YOU} {question}")
            return

        # --- Initialize scratchpad ---
        from core.services.playbook_scratchpad import PlaybookScratchpad
        scratchpad = PlaybookScratchpad(recipe_execution_id)
        scratchpad.write_inputs(input_data)
        scratchpad.write_meta(recipe_id, total_steps)

        # --- Pre-execution: load Mem0 memories ---
        # F155: none for a run a widget turn started — they are the owner's runs.
        recipe_memories = None
        if not widget_turn():
            try:
                from core.services.playbook_memory_service import PlaybookMemoryService
                memory_svc = PlaybookMemoryService(db=db)
                # F159: the parameter is playbook_id. The call passed recipe_id=,
                # raised TypeError on every run, and was logged as "skipped", so
                # no run ever recalled its playbook's past runs.
                recipe_memories = await memory_svc.retrieve_relevant_memories(
                    playbook_id=recipe.id,
                    context={"workspace_id": str(workspace_id), "input_data": input_data}
                )
                if recipe_memories and recipe_memories.get("total_memories", 0) > 0:
                    logger.info(
                        "[recipe_direct] Loaded %d Mem0 memories for recipe %d",
                        recipe_memories["total_memories"], recipe.id,
                    )
            except Exception as exc:
                logger.warning("[recipe_direct] Playbook memory recall failed for recipe %s: %s",
                               recipe.id, exc, exc_info=True)

        # F125: execution_config holds seconds. No unit is guessed from the size;
        # only the floors apply (core/services/playbook_timeouts.py).
        exec_config = recipe.execution_config or {}

        from config import config as app_config
        from core.services.playbook_timeouts import step_timeout_seconds, total_timeout_seconds

        raw_step = step_timeout_seconds(exec_config, app_config.PLAYBOOK_DEFAULT_STEP_TIMEOUT_SECONDS)
        raw_total = total_timeout_seconds(exec_config, app_config.PLAYBOOK_DEFAULT_TOTAL_TIMEOUT_SECONDS)

        step_timeout_sec = max(raw_step, app_config.PLAYBOOK_MIN_STEP_TIMEOUT_SECONDS)
        total_timeout_sec = max(raw_total, app_config.PLAYBOOK_MIN_TOTAL_TIMEOUT_SECONDS)
        logger.info(
            f"[recipe_direct] Timeouts: step={step_timeout_sec:.0f}s, "
            f"total={total_timeout_sec:.0f}s (configured: step={raw_step:.0f}s, total={raw_total:.0f}s)"
        )

        # F140: a rerun after the owner answered carries the answers in every step's prompt.
        from services.playbook_owner_ask import NEEDS_YOU, owner_answers_block, owner_question, stop_for_owner

        owner_answers = owner_answers_block(execution.execution_metadata)

        # Execute each step sequentially
        step_results: List[Dict[str, Any]] = []
        step_result: Dict[str, Any] = {}  # the last step's full dict (a budget stop reads its output)
        answers: Dict[str, str] = {}      # output_key -> that step's whole answer (F130 placeholders)
        # PRD-251 US-117: what each finished step offers a fixed step's {{ step_N.key }}.
        finished_steps: Dict[int, Dict[str, Any]] = {}
        execution_start = time.time()

        for idx, step in enumerate(steps):
            # Cancellation check — re-read execution row to detect external cancel
            db.expire(execution)
            db.refresh(execution)
            if execution.status == "cancelled":
                logger.info(
                    "[recipe_direct] Cancelled before step %d (execution=%s)",
                    idx + 1, recipe_execution_id,
                )
                _persist_step_results(db, execution, step_results)
                return

            # Total timeout check
            elapsed = time.time() - execution_start
            if elapsed > total_timeout_sec:
                msg = f"Total execution timeout ({total_timeout_sec}s) exceeded after {int(elapsed)}s at step {idx + 1}"
                logger.warning(f"[recipe_direct] {msg}")
                _persist_step_results(db, execution, step_results)
                # F123: out of time after finished work: say so, and keep the work.
                stop = _budget_stop(
                    f"Out of time after step {idx} of {total_steps} "
                    f"({_span(elapsed)}; budget {_span(total_timeout_sec)})",
                    idx + 1, step_results, step_result,
                )
                await _fail_execution(db, recipe_execution_id, stop[0] if stop else msg,
                                      step_results=step_results, review_card=stop[1] if stop else None)
                return

            # PRD-181 S2 (F060): playbook DOLLAR-CEILING admission gate — the same
            # $ ceiling missions enforce, generalised via services.budget_ceiling.
            # A ceiling of 0/absent means unlimited. A next step that would push
            # cumulative spend over the ceiling stops the run here (fail honestly),
            # exactly like the mission dispatcher's 'block' rule.
            ceiling_usd = _playbook_cost_ceiling_usd(exec_config, app_config)
            if ceiling_usd > 0:
                used_usd = _tokens_to_usd(
                    sum(s.get("tokens_used", 0) for s in step_results), app_config
                )
                from services.budget_ceiling import playbook_can_afford

                # Estimate this step at the average per-step spend so far (or a
                # single step's floor when nothing has run yet).
                per_step_est = (used_usd / max(1, idx)) if idx > 0 else 0.0
                if not playbook_can_afford(
                    ceiling_usd=ceiling_usd, used_usd=used_usd, next_step_usd=per_step_est
                ):
                    msg = (
                        f"Playbook budget ceiling ${ceiling_usd:.2f} reached "
                        f"(${used_usd:.4f} spent) before step {idx + 1}"
                    )
                    logger.warning(f"[recipe_direct] {msg}")
                    _persist_step_results(db, execution, step_results)
                    # F123: the dollar ceiling stops the same honest way.
                    stop = _budget_stop(
                        f"Budget ceiling ${ceiling_usd:.2f} reached after step {idx} of "
                        f"{total_steps} (${used_usd:.4f} spent)",
                        idx + 1, step_results, step_result,
                    )
                    await _fail_execution(db, recipe_execution_id, stop[0] if stop else msg,
                                          step_results=step_results, review_card=stop[1] if stop else None)
                    return

            step_id = step.get('step_id', f'step-{idx + 1}')
            step_order = step.get('order', idx + 1)
            agent_id = step.get('agent_id')
            prompt_template = (step.get('prompt_template') or '').strip()
            error_handling = step.get('error_handling', 'stop')
            max_retries = step.get('max_retries', 1)
            output_key = step.get('output_key', f'step_{step_order}')
            agent = agent_map.get(agent_id)
            agent_name = agent.name if agent else f"Agent {agent_id}"
            step_type = step.get("type", "agent")

            # A fixed generate_document step has no agent and no prompt (PRD-251 US-117).
            if not prompt_template and step_type != PLAYBOOK_DOCUMENT_STEP:
                msg = f"Step {step_order} ({agent_name}) has no prompt_template — skipping"
                logger.warning(f"[recipe_direct] {msg}")
                step_result = {
                    "step_id": step_id, "order": step_order,
                    "agent_id": agent_id, "agent_name": agent_name,
                    "prompt_template": "", "output_key": output_key,
                    "status": "skipped", "output": msg,
                    "tool_calls": [], "duration_ms": 0, "tokens_used": 0,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "error": msg, "retries": 0,
                }
                step_results.append(step_result)
                if error_handling == 'stop':
                    await _fail_execution(db, recipe_execution_id, msg, step_results=step_results)
                    return
                continue

            agent_cfg = getattr(agent, 'configuration', None) or {}
            step_max_iter = step.get(
                "max_iterations",
                agent_cfg.get("max_iterations", app_config.RECIPE_DEFAULT_MAX_ITERATIONS),
            )
            logger.info(f"[recipe_direct] Step {step_order}/{total_steps}: {agent_name} (max_turns={step_max_iter}) — {prompt_template[:200]}")

            # Update execution progress. The stamp tells the reconciler the run is
            # alive: a run of several steps outlives TASK_STALL_TIMEOUT_SECONDS.
            execution.current_step = idx + 1
            execution.execution_metadata = _progress_stamped(execution.execution_metadata)
            db.commit()

            # PRD-239 S3b: a session agent's step is a Claude Code session on the
            # operator's machine — minutes, not seconds. It gets its own bound and
            # the total budget grows by the same allowance; while it waits, the run
            # stamps progress so the stall watchdog leaves it alone.
            step_deadline = _step_deadline(step_timeout_sec, _is_session_step(db, agent), app_config)
            total_timeout_sec += max(0.0, step_deadline - step_timeout_sec)

            def _mark_progress(_execution=execution) -> None:
                _stamp_progress(db, _execution)

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
            clean_step_prompt = fill_step_placeholders(prompt_template, input_data, answers)

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

            # A fixed generate_document step (PRD-63): a PDF/DOCX/XLSX, or a social
            # image/video rendered by media-render (PRD-251 US-117).
            if step_type == PLAYBOOK_DOCUMENT_STEP:
                try:
                    from modules.documents.generation_service import (
                        DocumentGenerationService,
                        deliverables_app_url,
                    )
                    gen = _document_step_config(step)
                    # {{ step_N.key }} in the title and data: what earlier steps produced.
                    gen_title, gen_data = _resolved_document_step(gen, finished_steps)

                    gen_service = DocumentGenerationService(db, workspace_id)
                    gen_result = await _stamping_progress(
                        gen_service.generate(
                            title=gen_title,
                            format=gen["format"],
                            data=gen_data,
                            workspace_id=workspace_id,
                            template_name=gen["template_name"],
                            template_id=gen["template_id"],
                        ),
                        _mark_progress,
                        app_config.PLAYBOOK_PROGRESS_STAMP_SECONDS,
                    )
                    # PRD-242 S4: a playbook-rendered document is a Deliverable —
                    # it used to live only inside the step's output JSON.
                    registration = gen_service.register_as_deliverable(
                        gen_result,
                        title=gen_title,
                        source_type="playbook",
                        source_id=str(getattr(execution, "execution_id", "") or ""),
                        agent_name=getattr(recipe, "name", None),
                        template_id=gen["template_id"],
                    )

                    step_result["status"] = "completed"
                    step_result["output"] = json.dumps({
                        "document_url": gen_result.download_url,
                        "filename": gen_result.filename,
                        "format": gen_result.format,
                        "size_kb": gen_result.size // 1024,
                        "deliverable_id": (registration or {}).get("deliverable_id"),
                        "app_url": deliverables_app_url(),
                        "share_url": gen_service.share_link(gen_result),
                        "template_id": gen_result.template_id,
                        "template_name": gen_result.template_name,
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
                    finished_steps[step_order] = step_values(step_result["output"])

                    logger.info(f"[recipe_direct] Step {step_order} (generate_document) completed: {gen_result.filename}")
                    compact = _build_compact_step_result(step_result)
                    step_results.append(compact)
                    _persist_step_results(db, execution, step_results)
                    continue

                except Exception as e:
                    step_result["status"] = "failed"
                    step_result["error"] = owner_error_text(e)
                    step_result["duration_ms"] = int((time.time() - step_start) * 1000)
                    step_result["completed_at"] = datetime.now(timezone.utc).isoformat()
                    logger.error(f"[recipe_direct] generate_document step failed: {e}", exc_info=True)

                    if error_handling == "stop":
                        step_results.append(_build_compact_step_result(step_result))
                        _persist_step_results(db, execution, step_results)
                        await _fail_execution(db, recipe_execution_id, f"Document generation step failed: {owner_error_text(e)}", step_results=step_results)
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
                        step_result["error"] = f"pre_exec error: {owner_error_text(pre_exc)}"
                        step_result["duration_ms"] = int((time.time() - step_start) * 1000)
                        step_result["completed_at"] = datetime.now(timezone.utc).isoformat()
                        step_results.append(_build_compact_step_result(step_result))
                        _persist_step_results(db, execution, step_results)
                        await _fail_execution(db, recipe_execution_id, f"Step {step_order} pre_exec error: {owner_error_text(pre_exc)}", step_results=step_results)
                        return

            # F055: a placeholder the run did not supply must never reach an
            # agent. Ticket #387's agent did the right thing — refused to invent
            # a document and asked — but it should never have been handed a
            # template variable to puzzle over. The step fails, saying which
            # placeholder and why, so the owner can fix the playbook or rerun it
            # with input.
            missing_input = unresolved_input_placeholders(clean_step_prompt)
            if missing_input:
                reason = (
                    f"Step {step_order} uses {', '.join(missing_input)}, but this run was "
                    f"{'given no input' if not input_data else 'not given that field'}. "
                    "Rerun the playbook with the input it expects, or change the step's "
                    "prompt. Nothing was sent to the agent."
                )
                logger.warning(f"[recipe_direct] {reason}")
                step_result["status"] = "failed"
                step_result["error"] = reason
                step_result["duration_ms"] = int((time.time() - step_start) * 1000)
                step_result["completed_at"] = datetime.now(timezone.utc).isoformat()
                step_results.append(_build_compact_step_result(step_result))
                _persist_step_results(db, execution, step_results)
                if error_handling == "stop":
                    await _fail_execution(db, recipe_execution_id, reason, step_results=step_results)
                    return
                continue

            if owner_answers:  # after the F055 check: an answer is the owner's text, not a template
                clean_step_prompt = f"{clean_step_prompt}\n\n{owner_answers}"

            # Execute with retries
            attempt = 0
            success = False
            last_error = None
            exec_messages = []
            owner_ask = None

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
                            recipe_execution_id=recipe_execution_id,
                            progress=_mark_progress,
                        ),
                        timeout=step_deadline,
                    )

                    if result.get("status") == "cancelled":
                        logger.info(
                            "[recipe_direct] Step %d returned cancelled — halting execution %s",
                            step_order, recipe_execution_id,
                        )
                        step_result["status"] = "cancelled"
                        step_result["error"] = "Cancelled by user"
                        step_result["duration_ms"] = int((time.time() - step_start) * 1000)
                        step_result["completed_at"] = datetime.now(timezone.utc).isoformat()
                        step_results.append(_build_compact_step_result(step_result))
                        _persist_step_results(db, execution, step_results)
                        # Cancel endpoint already wrote status='cancelled' + completed_at
                        return

                    if result.get("status") == "success":
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

                        # F140: a step that asks the owner stops the run and asks them,
                        # never retried. Before F131's rule: B4's step 1 failed to read
                        # the file AND asked for its path; the owner can answer that.
                        owner_ask = owner_question(step_result["output"], result, prompt_template)
                        if owner_ask is not None:
                            break

                        # F131: a step can run to its end and still have failed.
                        failed = step_failure(step_result["tool_calls"], result)
                        if failed:
                            last_error = failed
                            logger.warning(f"[recipe_direct] Step {step_order} failed: {failed}")
                            attempt += 1
                            continue

                        step_result["status"] = "completed"
                        success = True

                        # Write to scratchpad (auto-extract). The step's own saved keys
                        # first: write_step_results replaces them with every export so far.
                        finished_steps[step_order] = step_values(
                            step_result["output"], scratchpad.step_exports(step_order) if scratchpad else {}
                        )
                        agent_exports = scratchpad.get_exports() if scratchpad else {}
                        scratchpad.write_step_results(
                            step_order=step_order,
                            tool_calls=step_result["tool_calls"],
                            agent_output=step_result["output"],
                            agent_exports=agent_exports,
                        )

                        answers[output_key] = step_result["output"]
                        logger.info(f"[recipe_direct] Step {step_order} completed → output_key={output_key} ({step_result['tokens_used']} tokens)")
                    else:
                        last_error = result.get("error", "Agent returned non-success status")
                        logger.warning(f"[recipe_direct] Step {step_order} failed: {last_error}")

                except asyncio.TimeoutError:
                    last_error = f"Step timed out after {step_deadline:.0f}s"
                    logger.warning(f"[recipe_direct] Step {step_order} timed out ({step_deadline:.0f}s)")
                except Exception as e:
                    last_error = owner_error_text(e)
                    logger.error(f"[recipe_direct] Step {step_order} exception: {e}", exc_info=True)

                attempt += 1

            # Finalize step result
            step_result["duration_ms"] = int((time.time() - step_start) * 1000)
            step_result["completed_at"] = datetime.now(timezone.utc).isoformat()

            if owner_ask is not None:
                step_result["status"] = "failed"
                step_result["error"] = f"{NEEDS_YOU} {owner_ask['question']}"
                step_results.append(_build_compact_step_result(step_result))
                _persist_step_results(db, execution, step_results)
                asked = await stop_for_owner(
                    db, execution=execution, recipe=recipe, step_order=step_order, agent_id=agent_id,
                    agent_name=agent_name, ask=owner_ask, step_results=step_results,
                    step_calls=step_result["tool_calls"],
                )
                if not asked:
                    await _fail_execution(db, recipe_execution_id, step_result["error"], step_results=step_results)
                return

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

            # PRD-128: dispatch playbook_step_complete (default pref silent)
            await _dispatch_playbook_event(
                db=db,
                workspace_id=workspace_id,
                recipe_execution_id=recipe_execution_id,
                event_type="playbook_step_complete",
                title=f"Playbook step: {recipe.name} ({step_order}/{total_steps})",
                message=str(step_result.get("output", ""))[:500] or None,
                agent_id=agent_id,
                agent_name=agent_name,
                status="ok",
            )

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
        final_output = _final_output(step_results, step_result)

        execution.status = 'completed'
        execution.completed_at = datetime.now(timezone.utc)
        execution.step_results = step_results
        execution.output_data = {
            "final_output": final_output,
            "total_duration_ms": total_duration,
            "total_tokens": total_tokens,
            "steps_completed": len(step_results),
        }

        # PRD-142 W3-S12: playbooks primitive heartbeat at the COMPLETED
        # boundary. tick is a one-shot terminal transition; the wrapper
        # swallows any emit failure so a broken heartbeat writer cannot
        # fail playbook completion.
        from services.playbook_engine_heartbeat import _emit_playbooks_primitive
        _emit_playbooks_primitive(
            execution.workspace_id,
            success=True,
            detail=(
                f"exec={recipe_execution_id} steps={len(step_results)} "
                f"duration_ms={total_duration} tokens={total_tokens}"
            ),
        )

        # PRD-128: dispatch playbook_complete before final commit so the
        # notification row persists in the same transaction as the status
        # update.
        await _dispatch_playbook_event(
            db=db,
            workspace_id=workspace_id,
            recipe_execution_id=recipe_execution_id,
            event_type="playbook_complete",
            title=f"Playbook complete: {recipe.name}",
            message=(
                f"{len(step_results)} steps completed in {total_duration // 1000}s"
            ),
            status="ok",
        )

        # PRD-204 S3: playbook terminal choke point (success) -- joins the
        # same transaction as the status update; fail-soft.
        _ingest_playbook_terminal_watch(
            db,
            execution,
            terminal_state="completed",
            summary=f"{len(step_results)} steps completed in {total_duration // 1000}s",
        )

        db.commit()

        # Complete the board task
        try:
            from services.board_task_bridge import complete_recipe_board_task
            last_log_url = next((sr.get("log_url") for sr in reversed(step_results)
                                 if sr.get("status") == "completed"), None)
            complete_recipe_board_task(db, recipe_execution_id, success=True,
                                       result=_card_text(str(final_output), last_log_url))
        except Exception:
            db.rollback()
            logger.warning("Board task completion failed (non-blocking)", exc_info=True)

        logger.info(
            f"[recipe_direct] Execution {recipe_execution_id} COMPLETED — "
            f"{len(step_results)} steps, {total_duration}ms, {total_tokens} tokens"
        )

        # --- Auto-report (mirrors heartbeat / task auto-report) ---
        try:
            await _auto_create_playbook_report(
                db=db,
                workspace_id=workspace_id,
                recipe=recipe,
                recipe_execution_id=recipe_execution_id,
                execution=execution,
                step_results=step_results,
                total_duration_ms=total_duration,
                total_tokens=total_tokens,
                final_output=final_output,
                success=True,
            )
        except Exception as report_err:
            logger.warning(
                "[recipe_direct] Playbook auto-report failed (non-blocking) for %s: %s",
                recipe_execution_id, report_err,
            )

        # --- Post-execution: update agent performance_metrics ---
        _update_agent_performance_metrics(db, step_results, success=True)

        # --- Post-execution: learning + memory storage ---
        # F155: a run a widget turn started teaches the playbook nothing and
        # leaves nothing in memory (a widget turn stores none, F154); later runs
        # would recall it.
        post_exec_config = recipe.execution_config or {}
        learning_result = None
        remembered = not widget_turn()
        if remembered and (post_exec_config.get('auto_learning') or post_exec_config.get('auto_learn', False)):
            try:
                from core.services.playbook_learning_service import PlaybookLearningService
                learning_svc = PlaybookLearningService(db=db)
                learning_result = learning_svc.analyze_execution(recipe_execution_id)
                logger.info(f"[recipe_direct] Auto-learning completed for {recipe_execution_id}")
            except Exception as e:
                logger.warning(f"[recipe_direct] Auto-learning failed (non-blocking): {e}")

        # Store execution memories in Mem0 + L2 short-term
        if remembered:
            try:
                from core.services.playbook_memory_service import PlaybookMemoryService
                memory_svc = PlaybookMemoryService(db=db)
                await memory_svc.store_execution_memory(
                    recipe_execution_id,
                    learnings=learning_result,
                )
                logger.info(f"[recipe_direct] Stored playbook memories for {recipe_execution_id}")
            except Exception as e:
                logger.warning(f"[recipe_direct] Playbook memory storage skipped: {e}", exc_info=True)

    except Exception as e:
        logger.error(f"[recipe_direct] Fatal error in execution {recipe_execution_id}: {e}", exc_info=True)
        try:
            await _fail_execution(db, recipe_execution_id, owner_error_text(e))
        except Exception as err:
            logger.exception(
                f"[recipe_direct] _fail_execution itself failed for {recipe_execution_id}: {err}"
            )
    finally:
        origin.close()
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
# Prompt resolution: the run's input and the earlier steps' answers
# ---------------------------------------------------------------------------

# `{input}` and `{input.<field>}` — the placeholders a playbook step may use to
# reach what the run was given. Nothing else in a prompt is touched: a brace in
# a JSON example or a code sample is the author's text, not a variable.
_INPUT_PLACEHOLDER_RE = re.compile(r"\{input(?:\.[A-Za-z0-9_]+)?\}")


def substitute_playbook_input(template: str, input_data: Optional[Dict[str, Any]]) -> str:
    """Fill a step's input placeholders from the run's input.

    F055 (night 2, ticket #387): a step reading "…every price and tasting note
    in {input}" reached its agent verbatim. The engine understood only
    ``{input.<field>}``; the bare ``{input}`` — the thing a person naturally
    writes — passed straight through. It now means the whole input: the one
    field when there is one, the ``content`` of a trigger or upload when there
    is that, otherwise every field as ``key: value`` lines.
    """
    resolved = template
    if not input_data:
        return resolved
    for key, value in input_data.items():
        resolved = resolved.replace(f"{{input.{key}}}", str(value))
    if "{input}" in resolved:
        resolved = resolved.replace("{input}", _whole_input(input_data))
    return resolved


def _whole_input(input_data: Dict[str, Any]) -> str:
    if len(input_data) == 1:
        return str(next(iter(input_data.values())))
    if input_data.get("content"):
        return str(input_data["content"])
    return "\n".join(f"{key}: {value}" for key, value in input_data.items())


def unresolved_input_placeholders(text: str) -> List[str]:
    """Input placeholders still in ``text`` after substitution — each one is a
    variable the run did not supply."""
    return sorted(set(_INPUT_PLACEHOLDER_RE.findall(text or "")))


# `{{name}}` — a named blank (F130). Inner spaces allowed: `{{ name }}`.
_NAMED_BLANK_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


def fill_step_placeholders(
    template: str,
    input_data: Optional[Dict[str, Any]],
    answers: Optional[Dict[str, str]] = None,
) -> str:
    """Fill a step's placeholders: F055's ``{input}`` and ``{input.<field>}``,
    then (F130) ``{{name}}`` from the run's input of that name, else from the
    answer an earlier step stored under that ``output_key``, and ``{output_key}``
    from that answer.

    Night 4: ``{{date}}``, ``{{month}} {{year}}`` and ``{{roast_log_filename}}``
    reached the agents literally although the run supplied exactly those keys
    (B8, B27, B57), and step 2's ``{{price_per_kilo}}`` stayed blank after step 1
    stored "£22.00" under that key (B68). A blank that names neither is left as
    written: a prompt may be asking for a template.
    """
    resolved = substitute_playbook_input(template, input_data)
    answers = {key: answer_for_next_step(text) for key, text in (answers or {}).items() if text}
    values = {**answers, **{key: str(value) for key, value in (input_data or {}).items()}}
    resolved = _NAMED_BLANK_RE.sub(lambda m: values.get(m.group(1), m.group(0)), resolved)
    for key, text in answers.items():
        resolved = resolved.replace(f"{{{key}}}", text)
    return resolved


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _document_step_config(step: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise a ``generate_document`` playbook step (PRD-242 S4). Pure.

    The step may carry its settings under ``config`` or inline. ``template_id``
    (a UUID string — the id ``platform_list_templates`` hands out) is parsed
    here so an invalid one fails the step with a clear message instead of a
    stack trace deep in the renderer; it takes precedence over ``template_name``.
    ``data`` is an object, or (PRD-251 US-117) one ``{{ step_N.key }}`` string
    naming an object an earlier step produced.
    """
    cfg = step.get("config", step) if isinstance(step, dict) else {}
    raw_id = cfg.get("template_id")
    template_id = None
    if raw_id:
        try:
            template_id = UUID(str(raw_id))
        except (ValueError, TypeError):
            raise ValueError(f"generate_document step: template_id {raw_id!r} is not a UUID")
    data = cfg.get("data", {})
    return {
        "title": cfg.get("title", "Document"),
        "format": cfg.get("format", "pdf"),
        "data": data if isinstance(data, (dict, str)) else {},
        "template_name": cfg.get("template_name"),
        "template_id": template_id,
    }


def _resolved_document_step(gen: Dict[str, Any], finished_steps: Dict[int, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """The step's title and data with every ``{{ step_N.key }}`` resolved (PRD-251 US-117). Pure.

    What each reference reads is ``core/services/playbook_step_refs.py``: an
    earlier step's output, its scratchpad_write keys, or the keys of its JSON
    answer. A reference no earlier step answers fails the step, naming it, and
    nothing is rendered. The data must come out an object.
    """
    title = resolve_step_references(gen["title"], finished_steps)
    data = resolve_step_references(gen["data"], finished_steps)
    if not isinstance(data, dict):
        raise ValueError(
            f"generate_document step: data must be an object, or a reference to one, not {type(data).__name__}"
        )
    return str(title), data


def step_failure(tool_calls: List[Dict[str, Any]], result: Dict[str, Any]) -> Optional[str]:
    """Why a step that ran to its end failed, from deterministic signals, or None.

    F131 (night 4): the run's status came from the loop finishing, never from what
    the steps did. B33, B25: a step's tool call failed in words while the turn
    ended normally, so the step's 'stop' never fired and the run said complete.
    B47: a session step never reached Automatos ("No Automatos tools this
    session") and the run said "Playbook complete". Whether the step's answer
    MEANS it failed is the PRD-204 watch's job (Gerard's wiring), not this.
    """
    if result.get("session_connected") is False:
        return "the session never reached Automatos, so it ran without any of its tools"
    if tool_calls and tool_calls[-1].get("success") is False:
        last = tool_calls[-1]
        said = " ".join(str(last.get("result") or "").split())[:200]
        return f"its last tool call, {last.get('action') or 'a tool'}, failed" + (f": {said}" if said else "")
    return None


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
                    "success": call.get("success"),
                })
        return normalized
    return []


# The board card's result cap, on the success path and on a budget stop's card.
BOARD_RESULT_MAX_CHARS = 4000


def _final_output(step_results: List[dict], last_step: Dict[str, Any]) -> Optional[str]:
    """The run's output: the last step's full text when it completed (its dict is
    still in scope), else the newest completed step's stored preview."""
    final_output = None
    if last_step.get("status") == "completed":
        final_output = last_step.get("output", "")
    if not final_output:
        for sr in reversed(step_results):
            if sr.get("status") == "completed":
                final_output = sr.get("output_preview", "")
                break
    return final_output


def _card_text(text: str, log_url: Optional[str]) -> str:
    """A board card's text at the cap. A cut says so where it happens and where
    the rest is: a silent cut hides work the way a missing output does (F123)."""
    if len(text) <= BOARD_RESULT_MAX_CHARS:
        return text
    rest = f" The full output: {log_url}" if log_url else ""
    return f"{text[:BOARD_RESULT_MAX_CHARS]}\n\n[Cut at {BOARD_RESULT_MAX_CHARS:,} characters.{rest}]"


def _span(seconds: float) -> str:
    """A duration as an owner reads it: whole minutes, or seconds under one."""
    return f"{int(seconds // 60)} min" if seconds >= 60 else f"{int(seconds)} s"


def _step_numbers(orders: List[Any]) -> str:
    """Step numbers as prose: '1–3' when consecutive, else '1, 3 and 4'."""
    if all(isinstance(o, int) for o in orders) and orders == list(range(orders[0], orders[0] + len(orders))):
        return f"{orders[0]}–{orders[-1]}"
    return ", ".join(str(o) for o in orders[:-1]) + f" and {orders[-1]}"


def _budget_stop(
    reason: str,
    next_step: int,
    step_results: List[dict],
    last_step: Dict[str, Any],
) -> Optional[Tuple[str, str]]:
    """F123: a run out of budget after finished work fails honestly. Returns
    (error_message, card_result): the reason first, then every completed step,
    then the last one's output. None when no step completed: today's failure
    stands, with no work to show."""
    done = [s for s in step_results if s.get("status") == "completed"]
    if not done:
        return None
    orders = [s.get("order") for s in done]
    if len(orders) == 1:
        finished = f"Step {orders[0]} completed; its output is below."
    else:
        finished = f"Steps {_step_numbers(orders)} completed; the last one's output is below."
    message = f"{reason}. Step {next_step} never started. {finished}"
    lines = [message, "", "Completed steps:"]
    for s in done:
        line = (f"- Step {s.get('order')} ({s.get('agent_name')}): completed in "
                f"{_span((s.get('duration_ms') or 0) / 1000)}, {s.get('tokens_used') or 0:,} tokens")
        lines.append(f"{line}, log {s['log_url']}" if s.get("log_url") else line)
    lines += ["", f"Step {done[-1].get('order')}'s output:", _final_output(step_results, last_step) or ""]
    return message, _card_text("\n".join(lines), done[-1].get("log_url"))


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
    step_results: Optional[List[dict]] = None,
    review_card: Optional[str] = None,
):
    """Mark an execution as failed with an error message. ``review_card`` is
    the finished work a budget stop leaves (F123): the board card goes to review
    with it instead of failed."""
    try:
        execution = db.query(RecipeExecution).filter(
            RecipeExecution.execution_id == execution_id
        ).first()
        if execution:
            # F197: a credit failure is said in plain words (the raw text is in the
            # log), and a scheduled run that hit it is marked to go again once
            # credit is back.
            from core.llm.credit import is_out_of_credit, mark_for_rerun, plain_failure

            if is_out_of_credit(error_message):
                logger.warning(f"[F197] {execution_id} stopped on credit: {error_message}")
                error_message = plain_failure(error_message, scheduled=mark_for_rerun(execution))
            else:
                error_message = plain_failure(error_message)
            execution.status = 'failed'
            execution.error_message = error_message
            execution.completed_at = datetime.now(timezone.utc)
            if step_results is not None:
                execution.step_results = step_results

            # PRD-204 S3: playbook terminal choke point (failure) -- joins the
            # failure-status transaction below; fail-soft.
            _ingest_playbook_terminal_watch(
                db,
                execution,
                terminal_state="failed",
                summary=(error_message or "Playbook execution failed")[:500],
            )

            db.commit()

            # PRD-142 W3-S12: playbooks primitive heartbeat at the FAILED
            # boundary. Paired with the user-visible error_message + the
            # auto-report below so the tile flips down at the same instant
            # the failure is recorded (§H DoD #2 Failure path visible).
            from services.playbook_engine_heartbeat import _emit_playbooks_primitive
            _emit_playbooks_primitive(
                execution.workspace_id,
                success=False,
                detail=f"exec={execution_id} error={error_message}",
            )

            # Fail the board task, or put the finished work in front of a human
            try:
                from services.board_task_bridge import complete_recipe_board_task as _complete_board
                if review_card:
                    _complete_board(db, execution_id, success=False, result=review_card, review=True)
                else:
                    _complete_board(db, execution_id, success=False, error_message=error_message)
            except Exception:
                db.rollback()

            # PRD-185 S4: dispatch a playbook_failed notification so a human sees
            # the failure — mirrors playbook_complete on the success path. Its
            # absence made a ~17-day OpenRouter 402 outage silent (board closed
            # 'done', no event, nobody notified).
            try:
                await _dispatch_playbook_event(
                    db=db,
                    workspace_id=execution.workspace_id,
                    recipe_execution_id=execution_id,
                    event_type="playbook_failed",
                    title="Playbook failed",
                    message=(error_message[:200] if error_message else "Playbook execution failed"),
                    status="error",
                )
            except Exception:
                logger.warning("[recipe_direct] playbook_failed dispatch failed for %s", execution_id)

            logger.info(f"[recipe_direct] Execution {execution_id} marked FAILED: {error_message}")
            # Update agent performance_metrics for failure
            _update_agent_performance_metrics(db, step_results or [], success=False)

            # Auto-report on failure too — system admin needs to see these
            try:
                recipe = db.query(WorkflowRecipe).filter(WorkflowRecipe.id == execution.recipe_id).first()
                if recipe:
                    duration_ms = 0
                    if execution.started_at and execution.completed_at:
                        duration_ms = int(
                            (execution.completed_at - execution.started_at).total_seconds() * 1000
                        )
                    total_tokens = sum((s.get("tokens_used", 0) for s in (step_results or [])))
                    await _auto_create_playbook_report(
                        db=db,
                        workspace_id=str(execution.workspace_id),
                        recipe=recipe,
                        recipe_execution_id=execution_id,
                        execution=execution,
                        step_results=step_results or [],
                        total_duration_ms=duration_ms,
                        total_tokens=total_tokens,
                        final_output=error_message,
                        success=False,
                    )
            except Exception as rep_err:
                logger.warning(
                    "[recipe_direct] Failure auto-report skipped for %s: %s",
                    execution_id, rep_err,
                )

            # Capture failure context into memory so future runs can learn from it
            try:
                from core.services.playbook_memory_service import PlaybookMemoryService
                memory_svc = PlaybookMemoryService(db=db)
                await memory_svc.store_execution_memory(
                    execution_id,
                    learnings={"failure_reason": error_message},
                )
                logger.info(f"[recipe_direct] Stored failure memory for {execution_id}")
            except Exception as mem_err:
                logger.warning(
                    f"[recipe_direct] Failure memory storage skipped for {execution_id}: {mem_err}",
                    exc_info=True,
                )
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
