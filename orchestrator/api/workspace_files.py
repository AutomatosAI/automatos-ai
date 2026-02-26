"""
Workspace File Browser API
============================
PRD-66 Phase 1: Code Viewer Widget

Read-only filesystem access to physical workspace directories.
The backend container mounts workspace_data:/workspaces:ro.

  GET /api/workspaces/{workspace_id}/files          — directory listing
  GET /api/workspaces/{workspace_id}/files/content   — file content
"""

import logging
import mimetypes
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext

logger = logging.getLogger(__name__)

WORKSPACE_ROOT = Path(os.environ.get("WORKSPACE_VOLUME_PATH", "/workspaces"))

# Safety limits
MAX_FILE_SIZE = 2 * 1024 * 1024  # 2 MB for content endpoint
MAX_DIR_ENTRIES = 500

router = APIRouter(
    prefix="/api/workspaces/{workspace_id}/files",
    tags=["workspace-files"],
)


# ---------------------------------------------------------------------------
# Path security — inline from WorkspaceManager.resolve_safe_path()
# ---------------------------------------------------------------------------

class PathSecurityError(Exception):
    """Raised when a path escapes the workspace sandbox."""


def _resolve_safe_path(workspace_root: Path, relative_path: str) -> Path:
    """Resolve *relative_path* inside *workspace_root*, blocking escapes.

    Blocks: ../../ traversal, symlink escape, absolute paths, null bytes.
    """
    if "\x00" in relative_path:
        raise PathSecurityError("Null byte in path")

    if relative_path.startswith("/"):
        raise PathSecurityError("Absolute path not allowed")

    resolved = (workspace_root / relative_path).resolve()
    base_resolved = workspace_root.resolve()

    if not str(resolved).startswith(str(base_resolved)):
        raise PathSecurityError("Path traversal blocked")

    return resolved


def _workspace_dir(workspace_id: str) -> Path:
    """Return the root directory for a given workspace, verifying it exists."""
    ws_dir = WORKSPACE_ROOT / workspace_id
    if not ws_dir.is_dir():
        raise HTTPException(status_code=404, detail="Workspace directory not found")
    return ws_dir


def _guess_language(filename: str) -> str:
    """Map file extension to a Monaco-compatible language id."""
    ext = Path(filename).suffix.lower()
    return {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".md": "markdown",
        ".html": "html",
        ".htm": "html",
        ".css": "css",
        ".scss": "scss",
        ".sql": "sql",
        ".sh": "shell",
        ".bash": "shell",
        ".zsh": "shell",
        ".rs": "rust",
        ".go": "go",
        ".java": "java",
        ".c": "c",
        ".cpp": "cpp",
        ".h": "c",
        ".hpp": "cpp",
        ".rb": "ruby",
        ".php": "php",
        ".xml": "xml",
        ".toml": "toml",
        ".ini": "ini",
        ".cfg": "ini",
        ".env": "dotenv",
        ".dockerfile": "dockerfile",
        ".r": "r",
        ".swift": "swift",
        ".kt": "kotlin",
        ".lua": "lua",
    }.get(ext, "plaintext")


# ---------------------------------------------------------------------------
# GET /api/workspaces/{workspace_id}/files — directory listing
# ---------------------------------------------------------------------------
@router.get("")
async def list_files(
    workspace_id: str,
    path: str = Query(".", description="Relative path inside workspace"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """List files and directories at *path* inside the workspace."""
    # Verify workspace ownership
    if str(ctx.workspace_id) != workspace_id:
        raise HTTPException(status_code=403, detail="Workspace access denied")

    ws_dir = _workspace_dir(workspace_id)

    try:
        target = _resolve_safe_path(ws_dir, path)
    except PathSecurityError as exc:
        raise HTTPException(status_code=403, detail=str(exc))

    if not target.exists():
        raise HTTPException(status_code=404, detail="Path not found")

    if not target.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")

    entries = []
    try:
        for i, item in enumerate(sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))):
            if i >= MAX_DIR_ENTRIES:
                break

            # Skip hidden files starting with . (optional — keep for now)
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
        raise HTTPException(status_code=403, detail="Permission denied reading directory")

    return {
        "path": path,
        "entries": entries,
        "truncated": len(entries) >= MAX_DIR_ENTRIES,
    }


# ---------------------------------------------------------------------------
# GET /api/workspaces/{workspace_id}/files/content — file content
# ---------------------------------------------------------------------------
@router.get("/content")
async def get_file_content(
    workspace_id: str,
    path: str = Query(..., description="Relative file path inside workspace"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Return the text content of a file for the code viewer."""
    # Verify workspace ownership
    if str(ctx.workspace_id) != workspace_id:
        raise HTTPException(status_code=403, detail="Workspace access denied")

    ws_dir = _workspace_dir(workspace_id)

    try:
        target = _resolve_safe_path(ws_dir, path)
    except PathSecurityError as exc:
        raise HTTPException(status_code=403, detail=str(exc))

    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")

    if not target.is_file():
        raise HTTPException(status_code=400, detail="Path is not a file")

    # Size guard
    file_size = target.stat().st_size
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({file_size} bytes, max {MAX_FILE_SIZE})",
        )

    # Read as text — binary files will fail gracefully
    try:
        content = target.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.warning("Failed to read %s: %s", target, exc)
        raise HTTPException(status_code=422, detail="Unable to read file as text")

    mime_type, _ = mimetypes.guess_type(target.name)

    return {
        "path": path,
        "name": target.name,
        "content": content,
        "size": file_size,
        "language": _guess_language(target.name),
        "mime_type": mime_type or "text/plain",
    }
