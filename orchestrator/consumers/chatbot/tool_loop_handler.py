"""
Tool Loop Handler
=================

Tool loop execution logic extracted from StreamingChatService:
- run_tool_loop: the main iterative tool execution loop
- Composio error recovery
- Search spiral detection
- Loop prevention injection
"""

import json
import logging
import time
from types import SimpleNamespace
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from .tool_loop import ToolExecutionTracker
from .tool_integration import (
    build_assistant_tool_message,
    extract_user_text_from_messages,
    score_composio_candidates,
    build_composio_param_recovery,
)

logger = logging.getLogger(__name__)


async def run_tool_loop(
    service,
    response,
    llm_messages: List[Dict[str, Any]],
    agent_runtime,
    tool_data: Dict[str, Any],
    use_tools: Optional[List[Dict[str, Any]]],
    composio_result: Any = None,
) -> AsyncGenerator[Any, None]:
    """
    Unified tool execution loop with dedup, retry limits, and Composio recovery.

    Yields SSE chunks and a final {'_final_response': response} dict.

    Args:
        service: The StreamingChatService instance.
    """
    import asyncio

    max_iterations = 10
    iteration = 0
    current_response = response
    tracker = ToolExecutionTracker()

    # Recovery budgets
    action_not_mapped_retry_budget = 1
    invalid_parameters_retry_budget = 1

    # Search spiral detection
    last_tool_name: Optional[str] = None
    empty_same_tool_streak = 0

    # Per-tool attempt tracking for loop prevention
    tool_attempts: Dict[str, int] = {}

    # Multi-step tools that get a higher retry cap
    _MULTI_STEP_TOOLS = {
        "composio_execute",
        "read_file", "write_file", "list_directory", "create_directory", "delete_file",
        "generate_document",
        "workspace_read_file", "workspace_grep", "workspace_list_dir",
        "workspace_write_file", "workspace_create_directory",
    }

    while current_response.tool_calls and iteration < max_iterations:
        iteration += 1
        logger.info(f"Tool iteration {iteration}: {len(current_response.tool_calls)} tool calls")

        start_times: Dict[str, float] = {}
        tool_calls_prepared: List[Tuple[str, str, Dict]] = []
        fatal_errors: List[Dict[str, Any]] = []
        followup_system_messages: List[Dict[str, Any]] = []
        executed_call_key_repeat = False

        # Phase 1: Emit tool-start events
        for tool_call in current_response.tool_calls:
            tool_name = tool_call.get('function', {}).get('name', 'unknown')
            tool_id = tool_call.get('id', f'call_{int(time.time() * 1000)}')

            try:
                args_str = tool_call.get('function', {}).get('arguments', '{}')
                tool_input = json.loads(args_str) if isinstance(args_str, str) else (args_str or {})
            except Exception:
                tool_input = {}

            yield service.streaming_handler.format_aisdk_tool_start(tool_id, tool_name, tool_input=tool_input)
            await asyncio.sleep(0)

            start_times[tool_id] = time.time()
            tool_calls_prepared.append((tool_id, tool_name, tool_call))

        # Phase 2: Execute each tool
        tool_results: List[Dict[str, Any]] = []
        for tool_id, tool_name, tool_call in tool_calls_prepared:
            try:
                args_str = tool_call.get('function', {}).get('arguments', '{}')
                tool_args = json.loads(args_str or '{}') if isinstance(args_str, str) else (args_str or {})

                # Dedup check via ToolExecutionTracker
                should_skip, skip_reason = tracker.should_skip_execution(tool_name, tool_args)
                if should_skip:
                    executed_call_key_repeat = True
                    llm_context = f"Skipped: {skip_reason}"
                    tool_results.append({
                        "tool_call_id": tool_id,
                        "role": "tool",
                        "name": tool_name,
                        "content": llm_context,
                    })
                    yield service.streaming_handler.format_aisdk_data(
                        "tool-result",
                        {"toolCallId": tool_id, "toolName": tool_name, "result": llm_context},
                    )
                    await asyncio.sleep(0)
                    continue

                # Record execution
                tracker.record_execution(tool_name, tool_args)

                # Execute via tool_router.execute_and_format
                user_text = extract_user_text_from_messages(llm_messages)

                # Direct Composio action execution for per-action tools
                _is_composio_action = (
                    composio_result and composio_result.entity_id and (
                        tool_name in composio_result.action_set
                        or any(tool_name.startswith(f"{app}_") for app in composio_result.app_names)
                    )
                )
                if _is_composio_action:
                    llm_context = await _execute_composio_action(
                        service.db, tool_name, tool_args, composio_result
                    )
                else:
                    result = await service.tool_router.execute_and_format(
                        tool_name=tool_name,
                        tool_args=tool_args,
                        agent_id=agent_runtime.agent_id if hasattr(agent_runtime, 'agent_id') else 1,
                        workspace_id=service.workspace_id,
                        original_intent=user_text,
                    )

                    # Track empty search results for spiral detection
                    empty_same_tool_streak = _track_search_spiral(
                        result, tool_name, last_tool_name, empty_same_tool_streak
                    )
                    last_tool_name = tool_name

                    # Track per-tool attempts
                    tool_attempts[tool_name] = tool_attempts.get(tool_name, 0) + 1

                    llm_context = result.get('llm_context', str(result.get('raw_result', '')))
                    if len(llm_context) > 6000:
                        llm_context = llm_context[:6000] + f"\n... (truncated {len(llm_context) - 6000} chars)"

                    # Emit frontend data (tool-data for widgets)
                    frontend_data = result.get("frontend_data", {})
                    if result.get("success") and frontend_data:
                        tool_data.update(frontend_data)
                        yield service.streaming_handler.format_aisdk_tool_data(frontend_data)
                        await asyncio.sleep(0)

                    # Emit workflow-update for recipe/workflow tools
                    _WORKFLOW_TOOL_PREFIXES = ("platform_list_recipes", "platform_create_recipe", "platform_execute_recipe")
                    if tool_name.startswith(_WORKFLOW_TOOL_PREFIXES) or "workflow" in tool_name.lower():
                        _raw = result.get("raw_result") or {}
                        _wf_id = str(_raw.get("id") or _raw.get("workflow_id") or _raw.get("recipe_id") or tool_id)
                        _wf_status = "completed" if result.get("success") else "failed"
                        yield service.streaming_handler.format_aisdk_workflow_update(
                            workflow_id=_wf_id, status=_wf_status, current_step=tool_name,
                        )
                        await asyncio.sleep(0)

                    # Composio error recovery
                    recovery_result = _handle_composio_error_recovery(
                        service.db, result, tool_name, llm_messages, agent_runtime,
                        action_not_mapped_retry_budget, invalid_parameters_retry_budget,
                        followup_system_messages, service.workspace_id,
                    )
                    if recovery_result is not None:
                        if recovery_result.get("_early_return"):
                            yield {"_final_response": SimpleNamespace(
                                content=recovery_result["message"],
                                tool_calls=None, usage=None,
                            )}
                            return
                        action_not_mapped_retry_budget = recovery_result.get(
                            "action_not_mapped_retry_budget", action_not_mapped_retry_budget
                        )
                        invalid_parameters_retry_budget = recovery_result.get(
                            "invalid_parameters_retry_budget", invalid_parameters_retry_budget
                        )

                    if result.get('fatal_error'):
                        fatal_errors.append(result)

                # Store tool result
                tool_results.append({
                    'tool_call_id': tool_id,
                    'role': 'tool',
                    'name': tool_name,
                    'content': llm_context,
                })

                # Emit tool-end + tool-result events
                duration_ms = int((time.time() - start_times.get(tool_id, time.time())) * 1000)
                yield service.streaming_handler.format_aisdk_tool_end(
                    tool_call_id=tool_id, tool_name=tool_name, success=True, duration_ms=duration_ms,
                )
                yield service.streaming_handler.format_aisdk_data('tool-result', {
                    'toolCallId': tool_id,
                    'toolName': tool_name,
                    'result': llm_context[:500],
                })
                await asyncio.sleep(0)

                # Loop prevention: inject proceed instructions
                _inject_loop_prevention(
                    llm_messages, tool_name, tool_attempts,
                    empty_same_tool_streak, result if not _is_composio_action else {"success": True},
                    agent_runtime, _MULTI_STEP_TOOLS,
                )
                # Database tool: force synthesis after first success
                if (not _is_composio_action
                        and tool_name in {"query_database", "smart_query_database"}
                        and result.get("success")):
                    llm_messages.append({
                        "role": "system",
                        "content": (
                            "You now have the database result. Do NOT call the database tool again. "
                            "Write the final answer using the tool output above."
                        ),
                    })
                    # Append tool exchange before forcing synthesis
                    llm_messages.append(build_assistant_tool_message(tool_calls_prepared))
                    llm_messages.extend(tool_results)
                    final = await agent_runtime.llm_manager.generate_response(messages=llm_messages, tools=None)
                    yield {"_final_response": SimpleNamespace(
                        content=final.content or "", tool_calls=None,
                        usage=getattr(final, "usage", None),
                    )}
                    return

            except Exception as e:
                logger.error(f"Tool {tool_name} failed: {e}")
                error_msg = f"Error executing {tool_name}: {str(e)}"
                tool_results.append({
                    'tool_call_id': tool_id,
                    'role': 'tool',
                    'name': tool_name,
                    'content': error_msg,
                })
                yield service.streaming_handler.format_aisdk_data('tool-result', {
                    'toolCallId': tool_id,
                    'toolName': tool_name,
                    'result': error_msg,
                })
                await asyncio.sleep(0)

        # Phase 3: Append tool exchange to message history
        llm_messages.append(build_assistant_tool_message(tool_calls_prepared))
        llm_messages.extend(tool_results)

        if followup_system_messages:
            llm_messages.extend(followup_system_messages)

        # Phase 4: Force synthesis if duplicate or exhausted
        _any_tool_exhausted = any(v >= 8 for v in tool_attempts.values())
        if executed_call_key_repeat or _any_tool_exhausted:
            if _any_tool_exhausted:
                logger.warning(f"[tool-loop] Tool hard cap reached -- forcing synthesis (attempts: {dict(tool_attempts)})")
            llm_messages.append({
                "role": "system",
                "content": (
                    "You now have the tool results needed. "
                    "Do NOT call any more tools. "
                    "Write the final answer for the user using the tool output above."
                ),
            })
            final = await agent_runtime.llm_manager.generate_response(messages=llm_messages, tools=None)
            yield {"_final_response": SimpleNamespace(
                content=final.content or "", tool_calls=None,
                usage=getattr(final, "usage", None),
            )}
            return

        if fatal_errors:
            yield {"_final_response": SimpleNamespace(
                content=(
                    "I ran into a server configuration issue while executing that tool. "
                    "Please restart the backend and try again."
                ),
                tool_calls=None, usage=None,
            )}
            return

        # Phase 5: Next LLM call
        current_response = await agent_runtime.llm_manager.generate_response(
            messages=llm_messages, tools=use_tools,
        )
        logger.info(f"Iteration {iteration} complete. More tool calls: {bool(current_response.tool_calls)}, Has content: {bool(current_response.content)}")

        if not current_response.tool_calls:
            yield {'_final_response': current_response}
            return

    # Max iterations reached
    if iteration >= max_iterations:
        logger.warning(f"Max tool iterations ({max_iterations}) reached. Forcing final response.")
        final = await agent_runtime.llm_manager.generate_response(
            messages=llm_messages, tools=None,
        )
        yield {'_final_response': final}


async def _execute_composio_action(
    db_session,
    tool_name: str,
    tool_args: Dict[str, Any],
    composio_result: Any,
) -> str:
    """Execute a Composio per-action tool directly."""
    try:
        from modules.tools.services.composio_tool_service import ComposioToolService
        _exec_svc = ComposioToolService(db_session)
        exec_result = _exec_svc.execute_action(
            action_name=tool_name,
            params=tool_args,
            entity_id=composio_result.entity_id,
        )
        success = exec_result.get("success", False)
        data = exec_result.get("data")
        error = exec_result.get("error")
        if success:
            llm_context = json.dumps(data, default=str) if isinstance(data, (dict, list)) else str(data or "")
        else:
            llm_context = f"Error executing {tool_name}: {error or 'unknown error'}"
        logger.info(f"[Composio direct] {tool_name}: success={success}")
    except Exception as exc:
        llm_context = f"Error executing {tool_name}: {exc}"
        logger.error(f"[Composio direct] {tool_name} exception: {exc}", exc_info=True)

    if len(llm_context) > 4000:
        llm_context = llm_context[:4000] + "\n... (truncated)"
    return llm_context


def _track_search_spiral(
    result: Dict[str, Any],
    tool_name: str,
    last_tool_name: Optional[str],
    empty_same_tool_streak: int,
) -> int:
    """Track consecutive empty search results for spiral detection."""
    try:
        raw = result.get("raw_result") or {}
        count = raw.get("count")
        if count is None:
            rr = raw.get("results")
            if isinstance(rr, list):
                count = len(rr)
        is_search_tool = tool_name.startswith("search_") or tool_name in {"semantic_search"}
        is_empty = isinstance(count, int) and count == 0
        if is_search_tool and is_empty:
            if last_tool_name == tool_name:
                return empty_same_tool_streak + 1
            return 1
    except Exception:
        pass
    return 0


def _handle_composio_error_recovery(
    db_session,
    result: Dict[str, Any],
    tool_name: str,
    llm_messages: List[Dict[str, Any]],
    agent_runtime,
    action_not_mapped_retry_budget: int,
    invalid_parameters_retry_budget: int,
    followup_system_messages: List[Dict[str, Any]],
    workspace_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    """
    Handle Composio-specific error recovery (action-not-mapped, invalid-parameters).
    Returns None if no recovery needed, dict with recovery state otherwise.
    """
    if result.get("success"):
        return None

    error_type = result.get("error_type")
    raw_error = (result.get("raw_result") or {}).get("error") if isinstance(result.get("raw_result"), dict) else None

    # Recovery for action_not_mapped
    if error_type == "composio_action_not_mapped" and action_not_mapped_retry_budget > 0:
        user_text = extract_user_text_from_messages(llm_messages)
        if raw_error and "Examples of mapped actions:" in raw_error:
            examples = raw_error.split("Examples of mapped actions:", 1)[1].strip()
            top = score_composio_candidates(user_text, examples)
            if not top:
                return {
                    "_early_return": True,
                    "message": (
                        "That action is not available in the local integrations cache for this workspace/agent. "
                        "The system won't guess a different action (to avoid doing the wrong thing). "
                        "Please run a Composio sync to refresh `composio_actions_cache` for this app, then retry."
                    ),
                }
            followup_system_messages.append({
                "role": "system",
                "content": (
                    "The previous Composio action name was not mapped. "
                    "Retry using ONE of these exact mapped action names that best matches the user's request:\n"
                    f"{', '.join(top)}\n"
                    "Use `composio_execute` again with the corrected `action`."
                ),
            })
        else:
            followup_system_messages.append({
                "role": "system",
                "content": (
                    "The previous Composio action name was not mapped. "
                    "Retry using a valid mapped action from `composio_actions_cache`."
                ),
            })
        return {"action_not_mapped_retry_budget": action_not_mapped_retry_budget - 1}

    # Recovery for invalid_parameters on composio_execute
    if (
        tool_name == "composio_execute"
        and error_type == "invalid_parameters"
        and invalid_parameters_retry_budget > 0
    ):
        build_composio_param_recovery(
            db_session, llm_messages, agent_runtime, followup_system_messages, workspace_id,
        )
        return {"invalid_parameters_retry_budget": invalid_parameters_retry_budget - 1}

    # Deterministic errors -- stop immediately (unless we have followup instructions)
    deterministic_error_types = {
        "composio_not_assigned",
        "composio_not_connected",
        "composio_action_not_allowed",
        "composio_missing_workspace",
        "invalid_parameters",
    }
    if followup_system_messages:
        return None
    if error_type in deterministic_error_types:
        raw_error_msg = raw_error or result.get('llm_context', '') or "That tool is not available for this agent/workspace."
        return {"_early_return": True, "message": raw_error_msg}

    return None


def _inject_loop_prevention(
    llm_messages: List[Dict[str, Any]],
    tool_name: str,
    tool_attempts: Dict[str, int],
    empty_same_tool_streak: int,
    result: Dict[str, Any],
    agent_runtime,
    multi_step_tools: set,
) -> None:
    """Inject system messages to prevent tool loops."""
    # Search spiral: 2+ consecutive empty results from same tool
    if empty_same_tool_streak >= 2 and (
        tool_name.startswith("search_") or tool_name in {"semantic_search"}
    ):
        llm_messages.append({
            "role": "system",
            "content": (
                f"The tool `{tool_name}` returned no results after multiple attempts. "
                "STOP calling search tools. Use the information you already have "
                "and proceed to fulfill the user's request with your other available tools."
            ),
        })
        logger.info(f"[tool-loop] Search spiral detected for {tool_name} -- injecting proceed instruction")

    # Per-tool retry limits
    _is_multi_step = (
        tool_name in multi_step_tools
        or tool_name.startswith("composio_")
        or tool_name.startswith("workspace_")
    )
    _attempts = tool_attempts.get(tool_name, 0)

    if _is_multi_step and _attempts >= 8:
        llm_messages.append({
            "role": "system",
            "content": (
                f"STOP: `{tool_name}` has been called {_attempts} times. "
                "You MUST now synthesize a response from the results you have. "
                "Do NOT call any more tools."
            ),
        })
        logger.warning(f"[tool-loop] Multi-step tool {tool_name} hit hard cap ({_attempts} calls) -- forcing synthesis")
    elif not _is_multi_step and _attempts >= 2:
        llm_messages.append({
            "role": "system",
            "content": (
                f"You have already called `{tool_name}` multiple times. "
                f"Do NOT call `{tool_name}` again. "
                "Use the results you already have and proceed to fulfill the user's request "
                "with your other available tools."
            ),
        })
        logger.info(f"[tool-loop] Tool {tool_name} hit retry limit -- injecting proceed instruction")
