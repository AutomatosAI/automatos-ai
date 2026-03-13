"""
Workspace tool executor -- proxy calls to the workspace worker via WorkspaceClient.
Extracted from unified_executor.py.
"""

import logging
from typing import Any, Dict, Optional
from uuid import UUID

logger = logging.getLogger(__name__)


async def resolve_repo_dir(client) -> Optional[str]:
    """Auto-detect the git repo directory inside a workspace.

    Workspaces have repos cloned under ``repos/<name>/``.  This helper
    lists that directory and returns the path to the first repo found
    (e.g. ``repos/automatos-ai``), or *None* if nothing is there.
    Result is **not** cached -- workspaces are short-lived so the cost
    of one extra ``list_dir`` per execution is negligible.
    """
    try:
        result = await client.list_dir("repos")
        entries = result.get("entries", [])
        for entry in entries:
            if entry.get("type") == "directory":
                return f"repos/{entry['name']}"
    except Exception:
        pass
    return None


async def execute_workspace_action(
    executor,
    tool_name: str,
    parameters: Dict[str, Any],
    workspace_id: Optional[UUID] = None,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a workspace tool via WorkspaceClient proxy to the worker."""
    if not workspace_id:
        return {
            "success": False,
            "error": "workspace_id required for workspace tools",
            "tool": tool_name,
        }

    try:
        from core.workspace_client import WorkspaceClient
        client = WorkspaceClient(str(workspace_id))

        # Auto-detect repo directory for tools that need it
        repo_dir: Optional[str] = None
        needs_repo = tool_name in ("workspace_git", "workspace_exec")
        needs_path_prefix = tool_name in (
            "workspace_read_file", "workspace_write_file",
            "workspace_grep", "workspace_list_dir",
        )
        if (needs_repo or needs_path_prefix) and not parameters.get("cwd") and not parameters.get("path", "").startswith("repos/"):
            repo_dir = await resolve_repo_dir(client)

        if tool_name == "workspace_read_file":
            path = parameters.get("path", "")
            if not path:
                return {"success": False, "error": "path is required", "tool": tool_name}
            # Auto-prefix with repo dir if path is relative and doesn't already include it
            if repo_dir and not path.startswith("repos/") and not path.startswith("/"):
                path = f"{repo_dir}/{path}"
            result = await client.read_file(path)

        elif tool_name == "workspace_write_file":
            path = parameters.get("path", "")
            content = parameters.get("content")
            if not path:
                return {"success": False, "error": "path is required", "tool": tool_name}
            if content is None:
                return {"success": False, "error": "content is required", "tool": tool_name}
            if repo_dir and not path.startswith("repos/") and not path.startswith("/"):
                path = f"{repo_dir}/{path}"
            result = await client.write_file(path, content)

        elif tool_name == "workspace_list_dir":
            path = parameters.get("path", ".")
            if repo_dir and path == ".":
                path = repo_dir
            result = await client.list_dir(path)

        elif tool_name == "workspace_grep":
            pattern = parameters.get("pattern", "")
            if not pattern:
                return {"success": False, "error": "pattern is required", "tool": tool_name}
            grep_path = parameters.get("path", ".")
            if repo_dir and grep_path == ".":
                grep_path = repo_dir
            result = await client.grep(
                pattern=pattern,
                path=grep_path,
                include=parameters.get("include", ""),
                max_results=parameters.get("max_results", 50),
            )

        elif tool_name == "workspace_exec":
            command = parameters.get("command", "")
            if not command:
                return {"success": False, "error": "command is required", "tool": tool_name}
            result = await client.exec_command(
                command=command,
                cwd=parameters.get("cwd") or repo_dir,
                timeout=parameters.get("timeout", 120),
            )

        elif tool_name == "workspace_git":
            operation = parameters.get("operation", "")
            if not operation:
                return {"success": False, "error": "operation is required", "tool": tool_name}
            result = await client.git(
                operation=operation,
                cwd=parameters.get("cwd") or repo_dir,
                args=parameters.get("args", ""),
            )

        else:
            return {"success": False, "error": f"Unknown workspace tool: {tool_name}", "tool": tool_name}

        # Worker returned an error
        if result.get("success") is False or result.get("error"):
            logger.warning(
                f"[tool-trace {trace_id or 'no-trace'}] Workspace action {tool_name} "
                f"error: {result.get('error', 'unknown')}"
            )
            result.setdefault("success", False)
            return result

        # Ensure success=True so tool_router recognizes it
        result["success"] = True
        logger.info(
            f"[tool-trace {trace_id or 'no-trace'}] Workspace action {tool_name} completed"
        )
        return result

    except Exception as e:
        logger.error(f"[tool-trace {trace_id or 'no-trace'}] Workspace action error: {e}", exc_info=True)
        return {"success": False, "error": str(e), "tool": tool_name}
