"""Codegraph over the canvas workspace clone (PRD-170 S6 / Q38).

The canvas session should navigate its workspace by call graph, not just grep.
The PRD-165 codegraph tool family (``platform_codegraph_*`` — search / get_symbol
/ call_graph / dependencies / architecture) already exists and is WORKSPACE-
SCOPED. S6 points it at the *local workspace clone* (Q38: local-path scope) and
keeps the index fresh: index on session start + reindex on commit.

This module holds the pure, DB-free glue that defines that binding:

  * ``workspace_clone_project_name`` — the deterministic codegraph project name
    for a workspace clone (so the session's tools resolve to the right index);
  * ``clone_index_path`` — the local path of the clone to index, under the
    workspace mount (never outside it — tenancy);
  * ``IndexTrigger`` + ``should_reindex`` — the "on session start + on commit"
    trigger policy.

The DB-backed indexing itself (walk → parse → store) and the session getting
codegraph tools are NOT wired here — and doing so needs two pieces of net-new
infrastructure this pure module deliberately does not invent (a scope decision
for the PRD owner, per CLAUDE.md §12, not a silent defer):

  1. A LOCAL-DIRECTORY index entrypoint on ``CodeGraphService``. Its only public
     indexer is ``index_github_project`` — it CLONES from a GitHub URL to a temp
     dir. Indexing an already-present workspace clone needs a new public method
     (its internal ``_discover_code_files`` directory walk exists; the public
     seam does not) OR a workspace→repo-URL resolution to reuse the URL path.
  2. An MCP bridge so the HEADLESS ``claude`` CLI session (worker container) can
     CALL codegraph. The ``platform_codegraph_*`` family runs in the platform's
     OWN agent loop (``agent_platform_tools``); the CLI session cannot reach it.
     "the agent navigates by call-graph" needs an SDK MCP server exposing
     codegraph to the session — no MCP surface exists in the codebase yet.

What IS landed + tested here is the pure, security-critical glue that BINDS the
PRD-165 codegraph family to the local clone once that infrastructure exists: the
deterministic project name, the tenancy-fenced clone path, and the on-start +
on-commit reindex trigger policy. Pure stdlib.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Optional

# Codegraph project name for a workspace clone. Deterministic + collision-free
# across workspaces: the workspace id is a stable key. The tool family resolves
# a project by (name, workspace_id), so this name scopes cleanly.
_CLONE_PROJECT_PREFIX = "canvas-workspace"


def workspace_clone_project_name(workspace_id: str, repo_subpath: str = "") -> str:
    """The codegraph project name for a workspace's clone.

    ``repo_subpath`` distinguishes multiple repos under one workspace (e.g.
    ``repos/app`` vs ``repos/lib``); empty means the workspace root clone.
    """
    slug = re.sub(r"[^A-Za-z0-9]+", "-", (repo_subpath or "").strip("/")).strip("-")
    base = f"{_CLONE_PROJECT_PREFIX}:{workspace_id}"
    return f"{base}:{slug}" if slug else base


def clone_index_path(workspace_root: str, repo_subpath: str = "") -> str:
    """Absolute local path of the clone to index, guaranteed inside the mount.

    Tenancy: ``repo_subpath`` is normalised and must stay under
    ``workspace_root`` — a traversal (``..``) or absolute escape raises
    ValueError, so the codegraph indexer can never be pointed outside the
    workspace mount.
    """
    root = PurePosixPath(workspace_root)
    sub = (repo_subpath or "").strip()
    if not sub:
        return str(root)
    if sub.startswith("/") or "\x00" in sub:
        raise ValueError(f"repo_subpath must be workspace-relative: {sub!r}")
    candidate = PurePosixPath(*[p for p in sub.split("/") if p not in ("", ".")])
    if any(part == ".." for part in candidate.parts):
        raise ValueError(f"repo_subpath escapes the workspace mount: {sub!r}")
    resolved = root / candidate
    # PurePosixPath comparison — resolved must be under root.
    if root not in resolved.parents and resolved != root:
        raise ValueError(f"repo_subpath escapes the workspace mount: {sub!r}")
    return str(resolved)


class IndexTrigger(str, enum.Enum):
    """What caused a (re)index request for the workspace clone."""

    SESSION_START = "session_start"
    COMMIT = "commit"
    MANUAL = "manual"


@dataclass(frozen=True)
class IndexDecision:
    """Whether to (re)index, and why."""

    reindex: bool
    reason: str


def should_reindex(
    trigger: IndexTrigger,
    already_indexed: bool,
    changed_files: int = 0,
) -> IndexDecision:
    """Trigger policy for the workspace-clone index (Q38 — on start + on commit).

      * SESSION_START indexes only if the clone isn't already indexed (warm
        sessions reuse the existing index — cheap start);
      * COMMIT always reindexes when files changed (the graph must reflect the
        new tree the session just wrote); a no-op commit skips;
      * MANUAL always reindexes.
    """
    if trigger is IndexTrigger.SESSION_START:
        if already_indexed:
            return IndexDecision(False, "warm session — clone already indexed")
        return IndexDecision(True, "cold session — index the clone")
    if trigger is IndexTrigger.COMMIT:
        if changed_files > 0:
            return IndexDecision(True, f"commit changed {changed_files} file(s) — reindex")
        return IndexDecision(False, "commit changed no files — index unchanged")
    return IndexDecision(True, "manual reindex")
