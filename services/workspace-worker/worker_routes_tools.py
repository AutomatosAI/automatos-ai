"""Tool routes of the workspace worker: exec, git and html-to-png.

Moved unchanged out of ``WorkspaceWorker._health_server`` (PRD-256 O2b); see
``worker_http``.
"""
from __future__ import annotations

from aiohttp import web

from worker_http import open_workspace


async def exec_handler(request):
    """POST /workspaces/{workspace_id}/exec — run a sandboxed command."""
    from executor import WorkspaceToolExecutor

    ws_manager, refused = open_workspace(request)
    if refused is not None:
        return refused

    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 — any unreadable body is a 400, as before the move (O2b)
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    command = body.get("command", "").strip()
    if not command:
        return web.json_response({"error": "command is required"}, status=400)

    cwd = body.get("cwd")
    timeout = min(int(body.get("timeout", 120)), 300)

    executor = WorkspaceToolExecutor(ws_manager)
    result = await executor.execute_command(command, timeout=timeout, cwd=cwd)
    return web.json_response(result)


async def git_handler(request):
    """POST /workspaces/{workspace_id}/git — execute a git operation."""
    from executor import WorkspaceToolExecutor

    ws_manager, refused = open_workspace(request)
    if refused is not None:
        return refused

    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 — any unreadable body is a 400, as before the move (O2b)
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    operation = body.get("operation", "").strip()
    allowed_ops = {
        "status", "diff", "add", "commit", "push", "pull",
        "log", "branch", "checkout", "stash", "show", "blame", "fetch",
        "clone",
    }
    if not operation:
        return web.json_response({"error": "operation is required"}, status=400)
    if operation not in allowed_ops:
        return web.json_response(
            {"error": f"Operation '{operation}' not allowed. Allowed: {', '.join(sorted(allowed_ops))}"},
            status=400,
        )

    executor = WorkspaceToolExecutor(ws_manager)

    if operation == "clone":
        repo_url = body.get("args", "").strip()
        if not repo_url:
            return web.json_response({"error": "args must contain the repo URL"}, status=400)
        result = await executor._git_clone(
            repo_url=repo_url,
            branch=body.get("branch"),
            shallow=True,
        )
    else:
        cwd = body.get("cwd")
        args = body.get("args", "")
        command = f"git {operation} {args}".strip()
        result = await executor.execute_command(command, timeout=120, cwd=cwd)
    return web.json_response(result)


async def html_to_png_handler(request):
    """POST /workspaces/{workspace_id}/html-to-png — render HTML → PNG.

    Body:
        {
          "url": "file:///workspaces/{id}/deliverables/charts/revenue.html",
          "viewport": {"w": 1080, "h": 1350},
          "output_path": "deliverables/charts/2026-04-29/revenue.png",
          "wait_for": "[data-render-ready='true']",   # optional
          "full_page": false                            # optional
        }

    Response: see WorkspaceToolExecutor.html_to_png().
    """
    from executor import WorkspaceToolExecutor

    ws_manager, refused = open_workspace(request)
    if refused is not None:
        return refused

    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 — any unreadable body is a 400, as before the move (O2b)
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    url = (body.get("url") or "").strip()
    output_path = (body.get("output_path") or "").strip()
    viewport = body.get("viewport") or {}
    wait_for = body.get("wait_for", "[data-render-ready='true']")
    full_page = bool(body.get("full_page", False))

    if not url:
        return web.json_response({"error": "url is required"}, status=400)
    if not output_path:
        return web.json_response({"error": "output_path is required"}, status=400)

    try:
        viewport_w = int(viewport.get("w", 0))
        viewport_h = int(viewport.get("h", 0))
    except (TypeError, ValueError):
        return web.json_response(
            {"error": "viewport.w and viewport.h must be integers"}, status=400,
        )

    executor = WorkspaceToolExecutor(ws_manager)
    result = await executor.html_to_png(
        url=url,
        viewport_w=viewport_w,
        viewport_h=viewport_h,
        output_path=output_path,
        wait_for=wait_for,
        full_page=full_page,
    )
    if not result.get("success"):
        # Validation errors → 400. Render errors (timeout, browser
        # crash) also → 400 since they're caller-fixable in practice.
        return web.json_response(result, status=400)
    return web.json_response(result)
