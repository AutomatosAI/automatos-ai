"""
Workspace Manager
=================
PRD-56 Phase 2: Physical Workspace Architecture

Manages persistent workspace directories on the worker volume.
Each workspace gets its own directory tree with repos, artifacts,
and ephemeral task execution dirs.

Filesystem layout per workspace:
    /workspaces/{workspace_id}/
    ├── repos/          ← Cloned repos (persistent, git pull on revisit)
    ├── reports/        ← Agent reports (platform_submit_report)
    ├── content/        ← Long-form content (posts, articles, drafts)
    ├── artifacts/      ← Test reports, build outputs (persistent)
    ├── analytics/      ← Analytics exports, KPI snapshots
    ├── graph/          ← Knowledge graph exports
    ├── tasks/          ← Ephemeral per-task execution dirs (cleaned up)
    ├── .ssh/           ← Deploy keys (injected from credential store)
    ├── .gitconfig      ← Per-workspace git identity
    └── .workspace_meta.json
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

VOLUME_PATH = os.environ.get("WORKSPACE_VOLUME_PATH", "/workspaces")
DEFAULT_QUOTA_GB = int(os.environ.get("WORKSPACE_DEFAULT_QUOTA_GB", "5"))


class WorkspaceManager:
    """Manages a single workspace's filesystem on the persistent volume.

    Handles:
    - Directory provisioning (first-use setup)
    - Storage quota enforcement
    - Ephemeral task dir creation/cleanup
    - Credential injection (SSH keys, git config)
    - Safe path resolution (traversal prevention)
    """

    def __init__(self, workspace_id: str, volume_path: Optional[str] = None) -> None:
        self.workspace_id = workspace_id
        self.volume_path = volume_path or VOLUME_PATH
        self.root = Path(self.volume_path) / workspace_id
        self.quota_bytes = DEFAULT_QUOTA_GB * (1024 ** 3)
        self._current_usage: int = 0

    # =========================================================================
    # Directory provisioning
    # =========================================================================

    DEFAULT_SUBDIRS: tuple[str, ...] = (
        "repos",
        "reports",
        "content",
        "artifacts",
        "analytics",
        "graph",
        "tasks",
    )

    def ensure_workspace_exists(self) -> bool:
        """Create workspace directory tree if first use. Returns True if newly created."""
        created = not self.root.exists()

        for subdir in self.DEFAULT_SUBDIRS:
            (self.root / subdir).mkdir(parents=True, exist_ok=True)

        meta_path = self.root / ".workspace_meta.json"
        if not meta_path.exists():
            meta_path.write_text(json.dumps({
                "workspace_id": self.workspace_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "plan_tier": "pilot",
                "storage_quota_bytes": self.quota_bytes,
                "total_tasks_run": 0,
                "repos_cached": [],
            }, indent=2))
            logger.info("Workspace %s provisioned at %s", self.workspace_id[:8], self.root)

        return created

    # =========================================================================
    # Storage quota
    # =========================================================================

    def get_usage_bytes(self) -> int:
        """Calculate current disk usage for this workspace."""
        if not self.root.exists():
            return 0
        total = 0
        for f in self.root.rglob("*"):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
        self._current_usage = total
        return total

    def check_quota(self) -> bool:
        """Check if workspace is under storage quota. Updates _current_usage."""
        usage = self.get_usage_bytes()
        under = usage < self.quota_bytes
        if not under:
            logger.warning(
                "Workspace %s over quota: %s / %s",
                self.workspace_id[:8], self.usage_human, self.quota_human,
            )
        return under

    @property
    def usage_human(self) -> str:
        return f"{self._current_usage / (1024 ** 3):.1f}GB"

    @property
    def quota_human(self) -> str:
        return f"{self.quota_bytes / (1024 ** 3):.0f}GB"

    # =========================================================================
    # Task directory lifecycle
    # =========================================================================

    def create_task_dir(self, task_id: str) -> Path:
        """Create ephemeral task execution directory."""
        task_dir = self.root / "tasks" / f"task_{task_id}"
        task_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Created task dir: %s", task_dir)
        return task_dir

    def cleanup_task(self, task_id: str) -> None:
        """Remove ephemeral task directory + task-specific credentials."""
        task_dir = self.root / "tasks" / f"task_{task_id}"
        if task_dir.exists():
            shutil.rmtree(task_dir, ignore_errors=True)
            logger.debug("Cleaned up task dir: %s", task_dir)

        # Remove task-specific env file
        task_env = self.root / f".task_env_{task_id}"
        if task_env.exists():
            task_env.unlink(missing_ok=True)

    def cleanup_all_stale_tasks(self, max_age_hours: int = 24) -> int:
        """Remove task dirs older than max_age_hours. Returns count removed."""
        tasks_dir = self.root / "tasks"
        if not tasks_dir.exists():
            return 0

        cutoff = datetime.now(timezone.utc).timestamp() - (max_age_hours * 3600)
        removed = 0

        for task_dir in tasks_dir.iterdir():
            if task_dir.is_dir():
                try:
                    mtime = task_dir.stat().st_mtime
                    if mtime < cutoff:
                        shutil.rmtree(task_dir, ignore_errors=True)
                        removed += 1
                except OSError:
                    pass

        if removed:
            logger.info("Cleaned %d stale task dirs in workspace %s", removed, self.workspace_id[:8])
        return removed

    # =========================================================================
    # Credential injection
    # =========================================================================

    def inject_credentials(self, task_id: str, credentials: dict) -> None:
        """Inject SSH keys and git config for this task.

        Credentials dict may contain:
        - ssh_private_key: PEM-format private key for repo cloning
        - git_name: Git author name for commits
        - git_email: Git author email for commits
        - env_vars: Dict of additional env vars to inject
        """
        # SSH key
        if credentials.get("ssh_private_key"):
            ssh_dir = self.root / ".ssh"
            ssh_dir.mkdir(exist_ok=True)
            key_file = ssh_dir / "id_ed25519"
            key_file.write_text(credentials["ssh_private_key"])
            key_file.chmod(0o600)

            # SSH config to skip host key checking for github.com
            ssh_config = ssh_dir / "config"
            if not ssh_config.exists():
                ssh_config.write_text(
                    "Host github.com\n"
                    "  StrictHostKeyChecking no\n"
                    "  UserKnownHostsFile /dev/null\n"
                    "Host gitlab.com\n"
                    "  StrictHostKeyChecking no\n"
                    "  UserKnownHostsFile /dev/null\n"
                )
                ssh_config.chmod(0o600)

            logger.debug("Injected SSH key for workspace %s", self.workspace_id[:8])

        # Git identity
        if credentials.get("git_name"):
            gitconfig = self.root / ".gitconfig"
            gitconfig.write_text(
                f'[user]\n'
                f'    name = {credentials["git_name"]}\n'
                f'    email = {credentials.get("git_email", "agent@automatos.app")}\n'
            )

        # Task-specific env vars
        if credentials.get("env_vars"):
            task_env = self.root / f".task_env_{task_id}"
            lines = [f"{k}={v}" for k, v in credentials["env_vars"].items()]
            task_env.write_text("\n".join(lines))
            task_env.chmod(0o600)

    def clear_credentials(self) -> None:
        """Remove all injected credentials from workspace."""
        ssh_dir = self.root / ".ssh"
        if ssh_dir.exists():
            shutil.rmtree(ssh_dir, ignore_errors=True)

        gitconfig = self.root / ".gitconfig"
        if gitconfig.exists():
            gitconfig.unlink(missing_ok=True)

    # =========================================================================
    # Path safety
    # =========================================================================

    def resolve_safe_path(self, relative_path: str) -> Path:
        """Resolve a path and guarantee it stays within the workspace.

        Blocks: ../../ traversal, symlink escape, absolute paths, null bytes.

        Raises:
            SecurityError: If the resolved path escapes the workspace root.
        """
        if "\x00" in relative_path:
            raise SecurityError(f"Null byte in path: workspace {self.workspace_id[:8]}")

        if relative_path.startswith("/"):
            raise SecurityError(f"Absolute path not allowed: {relative_path}")

        resolved = (self.root / relative_path).resolve()
        base_resolved = self.root.resolve()

        try:
            resolved.relative_to(base_resolved)
        except ValueError:
            raise SecurityError(
                f"Path traversal blocked: '{relative_path}' resolves outside "
                f"workspace {self.workspace_id[:8]}"
            )

        return resolved

    # =========================================================================
    # Repo management
    # =========================================================================

    def get_repo_path(self, repo_name: str) -> Path:
        """Get the path for a cached repo. Does NOT create it."""
        safe_name = repo_name.replace("/", "_").replace("\\", "_")
        return self.root / "repos" / safe_name

    def repo_exists(self, repo_name: str) -> bool:
        """Check if a repo is already cloned."""
        return self.get_repo_path(repo_name).exists()

    def list_repos(self) -> list[str]:
        """List all cached repos in this workspace."""
        repos_dir = self.root / "repos"
        if not repos_dir.exists():
            return []
        return [d.name for d in repos_dir.iterdir() if d.is_dir()]

    def get_artifacts_path(self) -> Path:
        """Get the artifacts directory for this workspace."""
        return self.root / "artifacts"

    # =========================================================================
    # Metadata
    # =========================================================================

    def update_metadata(self, **kwargs) -> None:
        """Update workspace metadata file with given fields."""
        meta_path = self.root / ".workspace_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        else:
            meta = {}

        meta.update(kwargs)
        meta["updated_at"] = datetime.now(timezone.utc).isoformat()
        meta_path.write_text(json.dumps(meta, indent=2))

    def increment_task_count(self) -> None:
        """Increment total_tasks_run in metadata."""
        meta_path = self.root / ".workspace_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            meta["total_tasks_run"] = meta.get("total_tasks_run", 0) + 1
            meta["last_task_at"] = datetime.now(timezone.utc).isoformat()
            meta_path.write_text(json.dumps(meta, indent=2))


class SecurityError(Exception):
    """Raised when a workspace security boundary is violated."""
    pass
