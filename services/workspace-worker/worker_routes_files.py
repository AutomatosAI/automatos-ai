"""File routes of the workspace worker: list, read, write, grep and download.

Moved unchanged out of ``WorkspaceWorker._health_server`` (PRD-256 O2b); see
``worker_http``. Sensitive paths are never served: one list for every file
route (``workspace_manager.SENSITIVE_NAMES``, F178).
"""
from __future__ import annotations

import mimetypes

from aiohttp import web

from worker_http import WORKER_HTTP, open_workspace
from workspace_manager import SecurityError, is_sensitive_name

MAX_FILE_SIZE = 2 * 1024 * 1024  # 2 MB
MAX_DIR_ENTRIES = 500

# Language detection map (Monaco-compatible)
_lang_map = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".json": "json",
    ".yaml": "yaml", ".yml": "yaml", ".md": "markdown",
    ".html": "html", ".htm": "html", ".css": "css", ".scss": "scss",
    ".sql": "sql", ".sh": "shell", ".bash": "shell", ".zsh": "shell",
    ".rs": "rust", ".go": "go", ".java": "java", ".c": "c",
    ".cpp": "cpp", ".h": "c", ".hpp": "cpp", ".rb": "ruby",
    ".php": "php", ".xml": "xml", ".toml": "toml", ".ini": "ini",
    ".cfg": "ini", ".env": "dotenv", ".dockerfile": "dockerfile",
    ".r": "r", ".swift": "swift", ".kt": "kotlin", ".lua": "lua",
}


def _guess_language(filename: str) -> str:
    from pathlib import Path as P
    return _lang_map.get(P(filename).suffix.lower(), "plaintext")


async def list_files_handler(request):
    """GET /workspaces/{workspace_id}/files?path=. — directory listing.

    Provisions the workspace directory tree on first read so non-coder
    clients see the default folder structure (reports/, content/, etc.)
    even before any mission has run.
    """
    volume_path = request.app[WORKER_HTTP].volume_path
    from pathlib import Path as P

    ws_manager, refused = open_workspace(request)
    if refused is not None:
        return refused
    rel_path = request.query.get("path", ".")
    ws_dir = P(volume_path) / ws_manager.workspace_id

    try:
        target = ws_manager.resolve_safe_path(rel_path)
    except SecurityError as exc:
        return web.json_response({"error": str(exc)}, status=403)

    # Block access to sensitive paths
    if ws_manager.is_sensitive_path(target):
        return web.json_response({"error": "Access denied"}, status=403)

    if not target.exists():
        return web.json_response({"error": "Path not found"}, status=404)
    if not target.is_dir():
        return web.json_response({"error": "Path is not a directory"}, status=400)

    entries = []
    truncated = False
    try:
        for i, item in enumerate(
            sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        ):
            # Skip sensitive entries
            if is_sensitive_name(item.name):
                continue
            if i >= MAX_DIR_ENTRIES:
                truncated = True
                break
            stat = item.stat()
            rel = str(item.relative_to(ws_dir))
            entries.append({
                "name": item.name,
                "path": rel,
                "type": "directory" if item.is_dir() else "file",
                "size": stat.st_size if item.is_file() else 0,
                "modified_at": stat.st_mtime,
            })
    except PermissionError:
        return web.json_response({"error": "Permission denied"}, status=403)

    return web.json_response({
        "path": rel_path,
        "entries": entries,
        "truncated": truncated,
    })


async def file_content_handler(request):
    """GET /workspaces/{workspace_id}/files/content?path=file.py — file content."""
    import mimetypes

    rel_path = request.query.get("path")
    if not rel_path:
        return web.json_response({"error": "path query param required"}, status=400)

    ws_manager, refused = open_workspace(request)
    if refused is not None:
        return refused

    try:
        target = ws_manager.resolve_safe_path(rel_path)
    except SecurityError as exc:
        return web.json_response({"error": str(exc)}, status=403)

    # Block access to sensitive paths
    if ws_manager.is_sensitive_path(target):
        return web.json_response({"error": "Access denied"}, status=403)

    if not target.exists():
        return web.json_response({"error": "File not found"}, status=404)
    if not target.is_file():
        return web.json_response({"error": "Path is not a file"}, status=400)

    file_size = target.stat().st_size
    if file_size > MAX_FILE_SIZE:
        return web.json_response(
            {"error": f"File too large ({file_size} bytes, max {MAX_FILE_SIZE})"},
            status=413,
        )

    try:
        content = target.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError):
        return web.json_response({"error": "Unable to read file as text"}, status=422)

    mime_type, _ = mimetypes.guess_type(target.name)

    return web.json_response({
        "path": rel_path,
        "name": target.name,
        "content": content,
        "size": file_size,
        "language": _guess_language(target.name),
        "mime_type": mime_type or "text/plain",
    })


async def write_file_handler(request):
    """POST /workspaces/{workspace_id}/files/write — write a file."""
    from executor import WorkspaceToolExecutor

    ws_manager, refused = open_workspace(request)
    if refused is not None:
        return refused

    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 — any unreadable body is a 400, as before the move (O2b)
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    path = body.get("path", "").strip()
    content = body.get("content")
    if not path:
        return web.json_response({"error": "path is required"}, status=400)
    if content is None:
        return web.json_response({"error": "content is required"}, status=400)

    executor = WorkspaceToolExecutor(ws_manager)
    result = await executor.write_file(path, content)
    if result.get("error"):
        return web.json_response(result, status=400)
    return web.json_response(result)


async def grep_handler(request):
    """GET /workspaces/{workspace_id}/files/grep — search file contents."""
    from executor import WorkspaceToolExecutor

    ws_manager, refused = open_workspace(request)
    if refused is not None:
        return refused

    pattern = request.query.get("pattern", "").strip()
    if not pattern:
        return web.json_response({"error": "pattern query param required"}, status=400)

    search_path = request.query.get("path", ".")
    include = request.query.get("include", "")
    max_results = min(int(request.query.get("max_results", "50")), 200)

    executor = WorkspaceToolExecutor(ws_manager)

    # Build grep command
    cmd_parts = ["grep", "-rn"]
    if include:
        cmd_parts.extend(["--include", include])
    cmd_parts.append("--")
    cmd_parts.append(pattern)
    cmd_parts.append(".")

    import shlex
    cmd = " ".join(shlex.quote(p) for p in cmd_parts)

    try:
        ws_manager.resolve_safe_path(search_path)   # refuses a path outside the workspace
    except SecurityError:
        return web.json_response({"error": "Invalid search path"}, status=403)

    result = await executor.execute_command(cmd, timeout=30, cwd=search_path)

    # Parse grep output into structured matches
    matches = []
    if result.get("stdout"):
        for line in result["stdout"].splitlines():
            if len(matches) >= max_results:
                break
            # grep -rn output: file:line:content
            parts = line.split(":", 2)
            if len(parts) >= 3:
                matches.append({
                    "file": parts[0],
                    "line": int(parts[1]) if parts[1].isdigit() else 0,
                    "content": parts[2],
                })

    total = len(result.get("stdout", "").splitlines()) if result.get("stdout") else 0
    return web.json_response({
        "matches": matches,
        "total": total,
        "truncated": total > max_results,
        "pattern": pattern,
    })


async def download_file_handler(request):
    """GET /workspaces/{workspace_id}/files/download — raw binary download."""
    ws_manager, refused = open_workspace(request)
    if refused is not None:
        return refused

    rel_path = request.query.get("path", "").strip()
    if not rel_path:
        return web.json_response({"error": "path parameter required"}, status=400)

    # The same traversal and sensitive-name guards as the listing and
    # content routes (F178: download served .ssh/, .canvas/ and the rest).
    try:
        target = ws_manager.resolve_safe_path(rel_path)
    except SecurityError:
        return web.json_response({"error": "Path traversal denied"}, status=403)
    if ws_manager.is_sensitive_path(target):
        return web.json_response({"error": "Access denied"}, status=403)
    if not target.is_file():
        return web.json_response({"error": "File not found"}, status=404)

    max_download_size = 100 * 1024 * 1024  # 100 MB
    file_size = target.stat().st_size
    if file_size > max_download_size:
        return web.json_response(
            {"error": f"File too large ({file_size} bytes, max {max_download_size})"},
            status=413,
        )

    mime_type, _ = mimetypes.guess_type(target.name)
    return web.FileResponse(
        target,
        headers={
            "Content-Disposition": f'attachment; filename="{target.name}"',
            "Content-Type": mime_type or "application/octet-stream",
        },
    )
