"""
File operation executors -- read, write, list, create, delete files.

All operations are proxied to the workspace-worker service via
exec_workspace.execute_workspace_action(). This guarantees files land in
the actual workspace volume (shared with the frontend file browser and
agent workspace views) rather than the API container's ephemeral /tmp.

The public tool names (read_file, write_file, list_directory,
create_directory, delete_file) are preserved so existing agent prompts
and skill definitions keep working. Internally each call is translated
to the corresponding workspace_* action and dispatched to the worker.
"""

import logging
from typing import Any, Dict, Optional

from modules.tools.execution import exec_workspace

logger = logging.getLogger(__name__)


def _pick_path(parameters: Dict[str, Any]) -> str:
    """Extract the path param, accepting the multiple aliases LLMs emit."""
    return (
        parameters.get("path")
        or parameters.get("file_path")
        or parameters.get("dir_path")
        or ""
    )


async def execute_file_op(
    executor,
    tool_name: str,
    parameters: Dict[str, Any],
    agent_id: int,
    workspace_id: Optional[Any] = None,
    trace_id: Optional[str] = None,
    caller_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute a file operation by delegating to the workspace worker.

    Translates legacy tool names + param aliases to workspace_* equivalents.
    """
    # The dispatch layer (unified_executor) passes workspace_id via kwargs on
    # newer paths but older paths call without it. Resolve from the executor
    # state when possible so we never silently fall back to the API container.
    ws_id = workspace_id or getattr(executor, "_current_workspace_id", None)

    path = _pick_path(parameters)
    logger.info(
        f"[file-op] tool={tool_name} path={path} workspace={ws_id} "
        f"(proxying to workspace-worker)"
    )

    if tool_name == "read_file":
        return await exec_workspace.execute_workspace_action(
            executor,
            "workspace_read_file",
            {"path": path},
            workspace_id=ws_id,
            trace_id=trace_id,
            agent_id=agent_id,
            caller_context=caller_context,
        )

    if tool_name == "write_file":
        return await exec_workspace.execute_workspace_action(
            executor,
            "workspace_write_file",
            {"path": path, "content": parameters.get("content", "")},
            workspace_id=ws_id,
            trace_id=trace_id,
            agent_id=agent_id,
            caller_context=caller_context,
        )

    if tool_name == "list_directory":
        return await exec_workspace.execute_workspace_action(
            executor,
            "workspace_list_dir",
            {"path": path or "."},
            workspace_id=ws_id,
            trace_id=trace_id,
            agent_id=agent_id,
            caller_context=caller_context,
        )

    if tool_name == "create_directory":
        if not path:
            return {"success": False, "error": "dir_path is required", "tool": tool_name}
        # Worker has no mkdir endpoint — use exec. Path is shell-quoted.
        quoted = "'" + path.replace("'", "'\\''") + "'"
        return await exec_workspace.execute_workspace_action(
            executor,
            "workspace_exec",
            {"command": f"mkdir -p {quoted}"},
            workspace_id=ws_id,
            trace_id=trace_id,
            agent_id=agent_id,
            caller_context=caller_context,
        )

    if tool_name == "delete_file":
        if not path:
            return {"success": False, "error": "file_path is required", "tool": tool_name}
        quoted = "'" + path.replace("'", "'\\''") + "'"
        return await exec_workspace.execute_workspace_action(
            executor,
            "workspace_exec",
            {"command": f"rm -rf {quoted}"},
            workspace_id=ws_id,
            trace_id=trace_id,
            agent_id=agent_id,
            caller_context=caller_context,
        )

    return {
        "success": False,
        "error": f"Unknown file operation: {tool_name}",
        "tool": tool_name,
    }
