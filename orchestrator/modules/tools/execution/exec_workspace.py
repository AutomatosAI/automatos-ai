"""
Workspace tool executor -- proxy calls to the workspace worker via WorkspaceClient.
Extracted from unified_executor.py.
"""

import logging
from typing import Any, Dict, Optional
from uuid import UUID

logger = logging.getLogger(__name__)


# Keys in caller_context that indicate the source of the execution. First
# non-None wins. Defaults to "chat" if none present.
_SOURCE_TYPE_KEYS = (
    ("heartbeat_id", "heartbeat"),
    ("mission_id", "mission"),
    ("task_id", "task"),
    ("playbook_id", "playbook"),
    ("trigger_id", "trigger"),
)


def _derive_source(caller_context: Optional[Dict[str, Any]]) -> tuple[str, Optional[str]]:
    """Derive (source_type, source_id) from caller_context."""
    if not caller_context:
        return "chat", None
    # Explicit override wins
    if caller_context.get("source_type"):
        return (
            str(caller_context.get("source_type")),
            str(caller_context.get("source_id")) if caller_context.get("source_id") else None,
        )
    for key, stype in _SOURCE_TYPE_KEYS:
        val = caller_context.get(key)
        if val:
            return stype, str(val)
    return "chat", None


def _auto_register_deliverable(
    *,
    workspace_id: UUID | str,
    file_path: str,
    write_result: Dict[str, Any],
    agent_id: Optional[int],
    caller_context: Optional[Dict[str, Any]],
    trace_id: Optional[str],
) -> None:
    """Register a freshly-written file as a deliverable.

    Failure MUST NOT break the write flow — all exceptions are caught and
    logged. See PRD-129 US-004.
    """
    try:
        from services.deliverable_service import (
            DeliverableService,
            AGENT_REGISTERABLE_ARTIFACT_TYPES,
            _infer_artifact_type,
            _humanize_basename,
        )

        artifact_type = _infer_artifact_type(file_path)
        if artifact_type not in AGENT_REGISTERABLE_ARTIFACT_TYPES:
            return

        # Try to read file_size_bytes from worker response to avoid a follow-up
        # HTTP round-trip. Workers may return `size`, `bytes_written`, etc.
        file_size_bytes: Optional[int] = None
        for key in ("file_size_bytes", "size", "bytes_written", "bytes"):
            raw = write_result.get(key) if isinstance(write_result, dict) else None
            if isinstance(raw, int):
                file_size_bytes = raw
                break

        # Resolve agent name from DB (cheap LEFT JOIN fallback exists, but
        # setting it avoids a join at read time and keeps soft-deleted agents
        # attributable).
        agent_name: Optional[str] = None
        if agent_id:
            try:
                from core.database.database import SessionLocal
                from core.models.core import Agent as AgentModel
                with SessionLocal() as lookup_db:
                    agent_row = lookup_db.query(AgentModel).filter(
                        AgentModel.id == agent_id
                    ).first()
                    if agent_row:
                        agent_name = agent_row.name
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "[tool-trace %s] Could not resolve agent_name for agent_id=%s: %s",
                    trace_id or "no-trace", agent_id, exc,
                )

        source_type, source_id = _derive_source(caller_context)

        from core.database.database import SessionLocal
        with SessionLocal() as db:
            service = DeliverableService(db=db, workspace_id=workspace_id)
            service.register(
                file_path=file_path,
                title=_humanize_basename(file_path),
                source_type=source_type,
                source_id=source_id,
                agent_id=agent_id,
                agent_name=agent_name,
                artifact_type=artifact_type,
                storage_type="workspace",
                file_size_bytes=file_size_bytes,
            )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "[tool-trace %s] Auto-register deliverable failed path=%s: %s",
            trace_id or "no-trace", file_path, exc, exc_info=True,
        )


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
    agent_id: Optional[int] = None,
    caller_context: Optional[Dict[str, Any]] = None,
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
        is_git_clone = tool_name == "workspace_git" and parameters.get("operation") == "clone"
        needs_repo = tool_name in ("workspace_git", "workspace_exec") and not is_git_clone
        needs_path_prefix = tool_name in (
            "workspace_read_file", "workspace_write_file",
            "workspace_grep", "workspace_list_dir",
        )
        # Paths starting with these prefixes are workspace-root relative, not repo-relative
        _WORKSPACE_ROOT_PREFIXES = ("repos/", "artifacts/", "content/", "reports/", "logs/")
        param_path = parameters.get("path", "")
        param_cwd = parameters.get("cwd", "")
        path_is_workspace_root = any(param_path.startswith(p) for p in _WORKSPACE_ROOT_PREFIXES)
        if (needs_repo or needs_path_prefix) and not param_cwd and not path_is_workspace_root:
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

        elif tool_name == "workspace_html_to_png":
            # Render an HTML page to a PNG inside the workspace. The resulting
            # file is auto-registered as a deliverable below (artifact_type
            # 'image' — inferred from .png).
            url = parameters.get("url", "")
            output_path = parameters.get("output_path", "")
            viewport = parameters.get("viewport") or {}
            if not url:
                return {"success": False, "error": "url is required", "tool": tool_name}
            if not output_path:
                return {"success": False, "error": "output_path is required", "tool": tool_name}
            try:
                viewport_w = int(viewport.get("w", 0))
                viewport_h = int(viewport.get("h", 0))
            except (TypeError, ValueError):
                return {
                    "success": False,
                    "error": "viewport.w and viewport.h must be integers",
                    "tool": tool_name,
                }
            result = await client.html_to_png(
                url=url,
                viewport_w=viewport_w,
                viewport_h=viewport_h,
                output_path=output_path,
                wait_for=parameters.get("wait_for", "[data-render-ready='true']"),
                full_page=bool(parameters.get("full_page", False)),
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

        # PRD-129 US-004: auto-register deliverable on successful write.
        # Registration failure MUST NOT break the file write.
        if tool_name == "workspace_write_file" and workspace_id:
            _auto_register_deliverable(
                workspace_id=workspace_id,
                file_path=path,
                write_result=result,
                agent_id=agent_id,
                caller_context=caller_context,
                trace_id=trace_id,
            )

        # workspace_html_to_png writes a PNG into the workspace; register it as
        # a deliverable using the path the worker actually wrote (post-validation).
        # Same contract as workspace_write_file — failure must not break the render.
        if tool_name == "workspace_html_to_png" and workspace_id:
            written_path = result.get("file_path")
            if written_path:
                _auto_register_deliverable(
                    workspace_id=workspace_id,
                    file_path=written_path,
                    write_result=result,
                    agent_id=agent_id,
                    caller_context=caller_context,
                    trace_id=trace_id,
                )

        return result

    except Exception as e:
        logger.error(f"[tool-trace {trace_id or 'no-trace'}] Workspace action error: {e}", exc_info=True)
        return {"success": False, "error": str(e), "tool": tool_name}
