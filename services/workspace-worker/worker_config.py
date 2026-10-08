"""
Worker-local config seam (PRD-203 C·S8)
=======================================

The workspace-worker is a self-contained image (``COPY . .`` / ``PYTHONPATH=/app``)
— at runtime it has NO access to the orchestrator's ``config.py``. This module
centralises the worker's environment reads so call sites never do ``os.getenv``
inline (mirrors the orchestrator's config discipline).

Named ``worker_config`` — deliberately NOT ``config`` — so it can never collide
with the orchestrator's ``config.py`` that ``main.py`` transiently places on
``sys.path`` (``sys.path.insert(0, "../../orchestrator")``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

# Workspace mount root inside the worker container (PRD-233 S1). Compose
# bind-mounts the host directory ``${AUTOMATOS_WORKSPACE_DIR:-./workspaces}``
# here; Railway mounts its persistent volume here. Per-workspace roots are
# ``workspace_root() / <workspace_id>`` (``WorkspaceManager.root``) — the
# boundary every canvas tool call is re-bound against (canvas_confinement).
WORKSPACE_ROOT_ENV = "WORKSPACE_VOLUME_PATH"
DEFAULT_WORKSPACE_ROOT = "/workspaces"


def model_auth_env() -> Dict[str, str]:
    """The model credential for the headless Claude Agent SDK subprocess.

    Returns the subset of ``{CLAUDE_CODE_OAUTH_TOKEN, ANTHROPIC_API_KEY}`` that is
    actually set — merged into ``ClaudeAgentOptions.env`` by ``start_session`` so
    the SDK subprocess is authenticated. An OAuth token (subscription) is
    preferred when both are present; either alone is sufficient.

    Empty dict → no credential configured. ``start_session`` treats that as a
    fail-fast condition on the real SDK path (never a silent idle).
    """
    env: Dict[str, str] = {}
    token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "").strip()
    if token:
        env["CLAUDE_CODE_OAUTH_TOKEN"] = token
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        env["ANTHROPIC_API_KEY"] = key
    return env


def has_model_auth() -> bool:
    """True iff a model credential is configured for the SDK subprocess."""
    return bool(model_auth_env())


def workspace_root() -> Path:
    """The workspace mount root (``WORKSPACE_VOLUME_PATH``, default ``/workspaces``).

    Read at call time, not import time, so it always reflects the process
    environment. Blank/whitespace counts as unset: a relative root silently
    anchored to the process cwd is the one thing this must never return.
    """
    raw = os.environ.get(WORKSPACE_ROOT_ENV, "").strip()
    return Path(raw or DEFAULT_WORKSPACE_ROOT)


# The largest binary file the platform may write through /files/write a piece at a
# time (the Socials copies of approved pictures and videos, executor._write_chunk).
MAX_BINARY_WRITE_ENV = "WORKER_MAX_BINARY_WRITE_BYTES"
DEFAULT_MAX_BINARY_WRITE_BYTES = 500 * 1024 * 1024


def max_binary_write_bytes() -> int:
    """The cap on one binary file written in pieces (``WORKER_MAX_BINARY_WRITE_BYTES``)."""
    raw = os.environ.get(MAX_BINARY_WRITE_ENV, "").strip()
    return int(raw) if raw.isdigit() else DEFAULT_MAX_BINARY_WRITE_BYTES


# The internal HTTP server (worker_http.py): the shared secret the platform sends in
# ``X-Internal-Token`` (empty = not enforced), and the address it binds.
INTERNAL_TOKEN_ENV = "WORKER_INTERNAL_TOKEN"
BIND_HOST_ENV = "WORKER_BIND_HOST"
DEFAULT_BIND_HOST = "0.0.0.0"


def worker_internal_token() -> str:
    """``WORKER_INTERNAL_TOKEN``; empty when unset."""
    return os.environ.get(INTERNAL_TOKEN_ENV, "")


def worker_bind_host() -> str:
    """``WORKER_BIND_HOST``, default ``0.0.0.0``."""
    return os.environ.get(BIND_HOST_ENV, DEFAULT_BIND_HOST)
