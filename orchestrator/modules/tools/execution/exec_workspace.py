"""
Workspace tool executor -- proxy calls to the workspace worker via WorkspaceClient.
Extracted from unified_executor.py.
"""

import base64
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



# How many sibling names an error lists before it stops being readable.
_PATH_HINT_ENTRIES = 25


async def _with_directory_hint(client: Any, path: str, result: dict) -> dict:
    """``result`` plus what actually exists near ``path``.

    Walks up to the nearest directory that lists, and names its entries, so the
    caller can correct itself in one step instead of guessing again.
    """
    from pathlib import PurePosixPath

    try:
        probe = PurePosixPath(path)
        for _ in range(4):
            parent = probe.parent
            listing = await client.list_dir(str(parent) if str(parent) != "." else ".")
            if listing.get("success"):
                entries = listing.get("entries") or listing.get("files") or listing.get("result") or []
                names = []
                for entry in entries:
                    name = entry.get("name") if isinstance(entry, dict) else str(entry)
                    if name:
                        names.append(str(name))
                shown = sorted(names)[:_PATH_HINT_ENTRIES]
                more = max(0, len(names) - len(shown))
                where = str(parent) if str(parent) != "." else "the workspace root"
                hint = (
                    f" Nothing at {path!r}. {where} contains: "
                    + (", ".join(shown) + (f" (+{more} more)" if more else "") if shown else "nothing")
                    + ". Paths are relative to the workspace root — use one of these, "
                      "or workspace_list_dir to look further."
                )
                return {**result, "error": f"{result.get('error', 'read failed')}.{hint}"}
            if str(parent) in ("", ".", "/"):
                break
            probe = parent
    except Exception:  # noqa: BLE001 — a hint must never replace the real error
        logger.debug("[workspace] could not build a path hint for %r", path, exc_info=True)
    return result


# F179 (A): only a raster image is ever made public, recognised by its first bytes,
# never by its name. Anything else (a customer CSV, a page, an SVG) is refused.
PUBLIC_IMAGE_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)
ONLY_IMAGES_ARE_PUBLIC = (
    "Only images can be made public; share a document through its Deliverables link."
)


def public_image_type(data: bytes) -> Optional[str]:
    """The raster image type these bytes are (png, jpeg, gif, webp), or None."""
    for signature, mime in PUBLIC_IMAGE_SIGNATURES:
        if data.startswith(signature):
            return mime
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def public_image_url(image_id: str) -> str:
    """Where the public image store serves ``image_id``: anyone with the link opens it."""
    from config import config

    return f"{(config.BACKEND_URL or '').rstrip('/')}/api/generated-images/{image_id}"


# F179 (B): a confirmation card names what it acts on. A public link's id is minted
# only when the image is stored, so the card names the file and where it will be.
PATH_ON_CARD_CHARS = 160


def _public_url_card(params: Dict[str, Any]) -> str:
    path = str(params.get("path") or "").strip()[:PATH_ON_CARD_CHARS]
    return f" on {path!r}, published at {public_image_url('<new id>')} for anyone with the link"


_CARD_SUBJECTS = {"workspace_get_public_url": _public_url_card}


def clear_declared_gates(
    db,
    tool_name: str,
    parameters: Dict[str, Any],
    *,
    workspace_id: Optional[UUID],
    agent_id: Optional[int] = None,
    caller_context: Optional[Dict[str, Any]] = None,
):
    """F179 (B): a workspace tool clears the gates its definition declares, the
    platform actions' own (PlatformActionExecutor.clear): super admin, admin and
    confirmation. None when it declares none, as the file tools do, so they run
    as before without touching the database; otherwise the refusal or card to
    return, or how the call cleared (for ``marked``)."""
    from modules.tools.discovery import get_action_registry

    action_def = get_action_registry().get(tool_name)
    declared = action_def is not None and (
        action_def.requires_confirmation or action_def.admin_only or action_def.super_admin_only
    )
    if not declared or not workspace_id:
        return None
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    # The actor is the runtime's agent, never a parameter (exec_platform's rule).
    params = {
        k: v for k, v in (parameters if isinstance(parameters, dict) else {}).items()
        if k not in ("_agent_id", "_agent_name")
    }
    if agent_id:
        params["_agent_id"] = agent_id
    card = _CARD_SUBJECTS.get(tool_name)
    return PlatformActionExecutor(db=db, workspace_id=workspace_id).clear(
        tool_name, params, caller_context, card_subject=card(params) if card else "",
    )


async def _get_public_url(client, path: str, workspace_id: UUID, trace_id: Optional[str]) -> Dict[str, Any]:
    """Download a workspace IMAGE and upload it to the public image store.

    Returns a publicly accessible URL that external services (Instagram,
    Twitter, etc.) can fetch without authentication. Only a raster image
    (png, jpeg, gif, webp, recognised by its bytes) is published (F179).
    """
    result = await client.download_file(path)
    if result.get("success") is False:
        return {"success": False, "error": f"Could not read file: {result.get('error')}", "tool": "workspace_get_public_url"}

    file_bytes: bytes = result["content"]
    if not file_bytes:
        return {"success": False, "error": "File is empty", "tool": "workspace_get_public_url"}

    content_type = public_image_type(file_bytes)
    if content_type is None:
        return {"success": False, "error": ONLY_IMAGES_ARE_PUBLIC, "tool": "workspace_get_public_url"}

    b64_data = base64.b64encode(file_bytes).decode("ascii")

    from core.services.image_store import get_image_store
    store = get_image_store()
    image_id = await store.save_image(b64_data, mime_type=content_type, workspace_id=str(workspace_id))

    public_url = public_image_url(image_id)

    logger.info(
        "[tool-trace %s] workspace_get_public_url: %s -> %s (%d bytes)",
        trace_id or "no-trace", path, public_url, len(file_bytes),
    )
    return {
        "success": True,
        "public_url": public_url,
        "file_path": path,
        "size_bytes": len(file_bytes),
        "content_type": content_type,
    }


async def execute_gated_workspace_action(
    executor,
    tool_name: str,
    parameters: Dict[str, Any],
    workspace_id: Optional[UUID] = None,
    trace_id: Optional[str] = None,
    agent_id: Optional[int] = None,
    caller_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """F179 (B): how every workspace tool is dispatched, by its own name or a file
    tool's (exec_file_ops): it clears the gates its definition declares, then runs."""
    async def run() -> Dict[str, Any]:
        return await execute_workspace_action(
            executor, tool_name, parameters,
            workspace_id=workspace_id, trace_id=trace_id,
            agent_id=agent_id, caller_context=caller_context,
        )

    cleared = clear_declared_gates(
        getattr(executor, "db", None), tool_name, parameters,
        workspace_id=workspace_id, agent_id=agent_id, caller_context=caller_context,
    )
    if cleared is None:
        return await run()
    from modules.tools.discovery.platform_executor import Cleared, marked

    if not isinstance(cleared, Cleared):
        return cleared
    result = await run()
    from modules.tools.execution.tool_grants import give_back_unused

    give_back_unused(getattr(executor, "db", None), cleared.approved_via_grant_id, result)
    return marked(result, cleared)


def exec_failure(result: Dict[str, Any]) -> Optional[str]:
    """F191 (night 6): why a command the worker ran failed, read from its exit
    code, or None.

    python3 scripts/profile.py exited 2 ("can't open file") and the call said
    success. The worker answers /exec with HTTP 200 and the exit code in the
    body, and nothing read it, so telemetry, F137's step summary and F131's
    failed-last-call rule all saw a success. Any non-zero exit fails the call,
    except 1 with nothing on stderr: that is a command's "no" (grep found
    nothing, test was false, diff found a difference), and a step that ends on
    one has not failed.
    """
    code = result.get("exit_code")
    if not isinstance(code, int) or code == 0 or result.get("error"):
        return None
    stderr = str(result.get("stderr") or "").strip()
    if code == 1 and not stderr:
        return None
    return f"the command exited {code}: {stderr.splitlines()[-1] if stderr else 'nothing on stderr'}"


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

        # Auto-detect repo directory ONLY for git/exec which need a cwd.
        # File operations (read, write, list, grep) treat paths as
        # workspace-root relative — agents can write to any directory
        # without needing a hardcoded allowlist.
        repo_dir: Optional[str] = None
        is_git_clone = tool_name == "workspace_git" and parameters.get("operation") == "clone"
        needs_repo = tool_name in ("workspace_git", "workspace_exec") and not is_git_clone
        param_cwd = parameters.get("cwd", "")
        if needs_repo and not param_cwd:
            repo_dir = await resolve_repo_dir(client)

        if tool_name == "workspace_read_file":
            path = parameters.get("path", "")
            if not path:
                return {"success": False, "error": "path is required", "tool": tool_name}
            result = await client.read_file(path)
            # F057: workspace_read_file failed 9 of 13 calls across three agents
            # with a bare "Path is not a file" / "File not found", and they
            # retried the same guess. A path error should hand back what IS
            # there — the same lesson as the agent roster in the "Agent not
            # found" message. Best effort: never turn a read error into a
            # different error.
            if not result.get("success"):
                result = await _with_directory_hint(client, path, result)

        elif tool_name == "workspace_write_file":
            path = parameters.get("path", "")
            content = parameters.get("content")
            if not path:
                return {"success": False, "error": "path is required", "tool": tool_name}
            if content is None:
                return {"success": False, "error": "content is required", "tool": tool_name}
            result = await client.write_file(path, content)

        elif tool_name == "workspace_list_dir":
            path = parameters.get("path", ".")
            result = await client.list_dir(path)

        elif tool_name == "workspace_grep":
            pattern = parameters.get("pattern", "")
            if not pattern:
                return {"success": False, "error": "pattern is required", "tool": tool_name}
            result = await client.grep(
                pattern=pattern,
                path=parameters.get("path", "."),
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
            url = parameters.get("url", "").strip()
            output_path = parameters.get("output_path", "").strip()
            viewport = parameters.get("viewport") or {}
            if not url:
                return {"success": False, "error": "url is required", "tool": tool_name}
            if not output_path:
                return {"success": False, "error": "output_path is required", "tool": tool_name}

            # Auto-append .png if the agent forgot the extension
            if not output_path.lower().endswith(".png"):
                output_path = output_path.rstrip(".") + ".png"

            # If agent passed a workspace-relative path instead of file:// URL,
            # auto-prefix so the worker can resolve it.
            if not url.startswith(("file://", "http://", "https://")):
                ws_id = str(workspace_id)
                url = f"file:///workspaces/{ws_id}/{url.lstrip('/')}"

            # Reject non-HTML file:// URLs (agents sometimes pass .md or .json)
            if url.startswith("file://"):
                from urllib.parse import urlparse
                parsed_path = urlparse(url).path
                if not parsed_path.lower().endswith((".html", ".htm")):
                    ext = parsed_path.rsplit(".", 1)[-1] if "." in parsed_path else "none"
                    return {
                        "success": False,
                        "error": (
                            f"url points to a .{ext} file — this tool renders HTML pages "
                            f"to PNG. Pass the URL of an HTML page in the workspace, e.g. "
                            f"file:///workspaces/{{id}}/deliverables/charts/revenue.html"
                        ),
                        "tool": tool_name,
                    }

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

        elif tool_name == "workspace_get_public_url":
            path = parameters.get("path", "").strip()
            if not path:
                return {"success": False, "error": "path is required", "tool": tool_name}
            result = await _get_public_url(client, path, workspace_id, trace_id)

        else:
            return {"success": False, "error": f"Unknown workspace tool: {tool_name}", "tool": tool_name}

        # F191: a command that failed says so; its output stays for the model.
        failed = exec_failure(result) if tool_name == "workspace_exec" else None
        if failed:
            result = {**result, "success": False, "error": failed}

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
