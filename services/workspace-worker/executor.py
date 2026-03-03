"""
Workspace Tool Executor
=======================
PRD-56 Phase 2: Sandboxed command execution in physical workspaces.

All shell commands and file operations run through this executor,
which enforces:
- Command whitelist (only approved binaries)
- Path containment (all paths must stay within workspace)
- Output limits (stdout/stderr capped)
- Timeout enforcement
- Sandboxed environment variables (stripped PATH)

This is the security boundary for agent code execution on the worker.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional

from workspace_manager import SecurityError, WorkspaceManager

logger = logging.getLogger(__name__)

# =============================================================================
# Command whitelist — only these binaries can execute
# =============================================================================

ALLOWED_COMMANDS: set[str] = {
    # Shell builtins / interpreters
    "sh", "bash",
    "cd", "pwd", "export", "source", "test", "true", "false",

    # Version control
    "git",

    # Python ecosystem
    "python", "python3", "pip", "pip3", "uv",
    "pytest", "ruff", "black", "mypy", "isort", "flake8",
    "coverage", "tox", "python3.12",

    # Node.js ecosystem
    "node", "npm", "npx", "pnpm", "yarn",
    "vitest", "jest", "tsc", "eslint", "prettier",

    # General tools
    "ls", "cat", "grep", "egrep", "fgrep", "rg",
    "find", "tree", "wc", "sort", "uniq", "cut", "tr",
    "head", "tail", "diff", "patch", "jq", "sed", "awk",
    "xargs", "tee", "less", "more",
    "curl", "wget",
    "make", "cmake",
    "tar", "gzip", "gunzip", "zip", "unzip", "bzip2",
    "touch", "mkdir", "cp", "mv", "rm", "ln", "chmod",
    "echo", "printf", "env", "which", "whoami", "id",
    "date", "basename", "dirname", "realpath", "readlink",
    "stat", "file", "du", "df",
    "ps", "kill", "sleep", "timeout",
    "clear", "reset",

    # Language runtimes (polyglot repos)
    "cargo", "go", "ruby", "java", "javac", "mvn", "gradle",
    "rustc", "gcc", "g++",

    # Docker (read-only inspection, not container escape)
    "docker-compose",
}

# Patterns that are ALWAYS blocked, even if the binary is whitelisted
BLOCKED_PATTERNS: list[str] = [
    r"rm\s+-rf\s+/\s*$",        # rm -rf /
    r"rm\s+-rf\s+/[^w]",        # rm -rf /anything (but not /workspaces)
    r"\bsudo\b",                 # privilege escalation
    r"\bsu\s",                   # user switching
    r"\bchmod\s+777\b",          # dangerous permissions
    r"\bkubectl\b",              # k8s access
    r">\s*/dev/",                # device access
    r"\bmkfs\b",                 # filesystem formatting
    r"\bdd\s+if=",              # raw disk operations
    r"\biptables\b",            # firewall manipulation
    r"\bsystemctl\b",           # service management
    r"\bpasswd\b",              # password changes
    r"\buseradd\b",             # user creation
    r"\buserdel\b",             # user deletion
    r"\bmount\b",               # filesystem mounting
    r"\bumount\b",              # filesystem unmounting
    r"`",                        # backtick execution
    r"\n",                       # embedded newlines
]

# Compiled blocked patterns for performance
_BLOCKED_RE = [re.compile(p) for p in BLOCKED_PATTERNS]

# Maximum output sizes
MAX_STDOUT_BYTES = 100_000    # 100KB
MAX_STDERR_BYTES = 50_000     # 50KB

# Default command timeout
DEFAULT_TIMEOUT = 120         # 2 minutes


class WorkspaceToolExecutor:
    """Sandboxed executor for shell commands and file operations.

    All operations are confined to a single workspace directory.
    Uses WorkspaceManager for path validation and security.
    """

    def __init__(self, workspace_manager: WorkspaceManager) -> None:
        self.ws = workspace_manager

    # =========================================================================
    # Shell command execution
    # =========================================================================

    async def execute_command(
        self,
        command: str,
        timeout: int = DEFAULT_TIMEOUT,
        cwd: Optional[str] = None,
        env_extras: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Execute a shell command within the workspace.

        Args:
            command: Shell command string to execute.
            timeout: Max seconds before kill.
            cwd: Working directory relative to workspace root. None = workspace root.
            env_extras: Additional env vars to set for this command.

        Returns:
            Dict with exit_code, stdout, stderr, duration_ms, truncated.
        """
        # 1. Validate command against whitelist
        validation = self._validate_command(command)
        if validation:
            return {"error": validation, "exit_code": -1, "stdout": "", "stderr": ""}

        # 2. Resolve working directory
        if cwd:
            work_dir = self.ws.resolve_safe_path(cwd)
        else:
            work_dir = self.ws.root

        if not work_dir.exists():
            return {"error": f"Working directory does not exist: {cwd}", "exit_code": -1}

        # 3. Build sandboxed environment
        env = self._build_sandboxed_env(env_extras)

        # 4. Execute
        logger.info("Executing in %s: %s", self.ws.workspace_id[:8], command[:100])

        import time
        start = time.monotonic()

        try:
            # Use shell mode for compound commands (pipes, &&, etc).
            # SECURITY: _validate_command already verified each segment against
            # the command whitelist and blocked patterns before reaching here.
            has_shell_operators = any(op in command for op in ("|", "&&", "||", ";", ">", "<"))
            if has_shell_operators:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    cwd=str(work_dir),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                )
            else:
                argv = shlex.split(command)
                proc = await asyncio.create_subprocess_exec(
                    *argv,
                    cwd=str(work_dir),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                elapsed = int((time.monotonic() - start) * 1000)
                return {
                    "exit_code": -1,
                    "stdout": "",
                    "stderr": f"Command timed out after {timeout}s",
                    "duration_ms": elapsed,
                    "timed_out": True,
                }

            elapsed = int((time.monotonic() - start) * 1000)

            # Truncate output if needed
            stdout = stdout_bytes[:MAX_STDOUT_BYTES].decode("utf-8", errors="replace")
            stderr = stderr_bytes[:MAX_STDERR_BYTES].decode("utf-8", errors="replace")
            truncated = len(stdout_bytes) > MAX_STDOUT_BYTES or len(stderr_bytes) > MAX_STDERR_BYTES

            return {
                "exit_code": proc.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "duration_ms": elapsed,
                "truncated": truncated,
            }

        except Exception as e:
            logger.error("Command execution error in %s: %s", self.ws.workspace_id[:8], e)
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": str(e),
                "error": str(e),
            }

    # =========================================================================
    # File operations (sandboxed)
    # =========================================================================

    async def read_file(self, path: str, max_bytes: int = 500_000) -> Dict[str, Any]:
        """Read a file from the workspace."""
        try:
            safe_path = self.ws.resolve_safe_path(path)
        except SecurityError as e:
            return {"error": str(e)}

        if not safe_path.exists():
            return {"error": f"File not found: {path}"}
        if not safe_path.is_file():
            return {"error": f"Not a file: {path}"}

        try:
            content = safe_path.read_bytes()
            truncated = len(content) > max_bytes
            return {
                "content": content[:max_bytes].decode("utf-8", errors="replace"),
                "size_bytes": len(content),
                "truncated": truncated,
                "path": str(safe_path.relative_to(self.ws.root)),
            }
        except Exception as e:
            return {"error": f"Read error: {e}"}

    async def write_file(self, path: str, content: str) -> Dict[str, Any]:
        """Write a file to the workspace."""
        try:
            safe_path = self.ws.resolve_safe_path(path)
        except SecurityError as e:
            return {"error": str(e)}

        try:
            safe_path.parent.mkdir(parents=True, exist_ok=True)
            safe_path.write_text(content)
            return {
                "written": True,
                "path": str(safe_path.relative_to(self.ws.root)),
                "size_bytes": len(content.encode()),
            }
        except Exception as e:
            return {"error": f"Write error: {e}"}

    async def list_directory(self, path: str = ".") -> Dict[str, Any]:
        """List directory contents within the workspace."""
        try:
            safe_path = self.ws.resolve_safe_path(path)
        except SecurityError as e:
            return {"error": str(e)}

        if not safe_path.exists():
            return {"error": f"Directory not found: {path}"}
        if not safe_path.is_dir():
            return {"error": f"Not a directory: {path}"}

        entries = []
        for item in sorted(safe_path.iterdir()):
            try:
                stat = item.stat()
                entries.append({
                    "name": item.name,
                    "type": "dir" if item.is_dir() else "file",
                    "size": stat.st_size if item.is_file() else None,
                })
            except OSError:
                entries.append({"name": item.name, "type": "unknown"})

        return {
            "path": str(safe_path.relative_to(self.ws.root)),
            "entries": entries,
            "count": len(entries),
        }

    async def create_directory(self, path: str) -> Dict[str, Any]:
        """Create a directory within the workspace."""
        try:
            safe_path = self.ws.resolve_safe_path(path)
        except SecurityError as e:
            return {"error": str(e)}

        try:
            safe_path.mkdir(parents=True, exist_ok=True)
            return {
                "created": True,
                "path": str(safe_path.relative_to(self.ws.root)),
            }
        except Exception as e:
            return {"error": f"Create directory error: {e}"}

    # =========================================================================
    # High-level task steps (called by worker main)
    # =========================================================================

    async def execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single task step. Dispatches to the right method."""
        action = step.get("action", "")

        if action == "execute_command" or action == "shell":
            return await self.execute_command(
                command=step.get("command", ""),
                timeout=step.get("timeout", DEFAULT_TIMEOUT),
                cwd=step.get("cwd"),
            )

        elif action == "git_clone":
            return await self._git_clone(
                repo_url=step["repo"],
                branch=step.get("branch"),
                shallow=step.get("shallow", True),
            )

        elif action == "git_pull":
            return await self._git_pull(
                repo_name=step.get("repo_name", ""),
                branch=step.get("branch"),
            )

        elif action == "read_file":
            return await self.read_file(step["path"])

        elif action == "write_file":
            return await self.write_file(step["path"], step["content"])

        elif action == "list_directory":
            return await self.list_directory(step.get("path", "."))

        elif action == "create_directory":
            return await self.create_directory(step["path"])

        else:
            return {"error": f"Unknown action: {action}"}

    # =========================================================================
    # Git operations (high-level)
    # =========================================================================

    # Branch name pattern — alphanumeric + . / _ - only, no leading dash.
    _BRANCH_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._/\-]{0,254}$")

    async def _git_clone(
        self,
        repo_url: str,
        branch: Optional[str] = None,
        shallow: bool = True,
    ) -> Dict[str, Any]:
        """Clone a repo into the workspace repos/ directory. Uses cache if exists.

        PRD-70 FIX-01: Validates branch name and uses ``--`` separator to
        prevent argument injection in the worker container.
        """
        # PRD-70 FIX-01: Validate branch to prevent --upload-pack injection
        if branch:
            if branch.startswith("-") or not self._BRANCH_RE.match(branch):
                return {
                    "exit_code": 1,
                    "stdout": "",
                    "stderr": f"Invalid branch name: {branch}",
                }

        # Extract repo name from URL
        repo_name = repo_url.rstrip("/").split("/")[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]

        repo_path = self.ws.get_repo_path(repo_name)

        if repo_path.exists():
            # Already cloned — do git pull instead
            logger.info("Repo %s already cached, pulling updates", repo_name)
            return await self._git_pull(repo_name, branch)

        # Build clone command with -- separator (PRD-70 FIX-01)
        cmd_parts = ["git", "clone"]
        if shallow:
            cmd_parts.extend(["--depth", "1"])
        if branch:
            cmd_parts.extend(["--branch", branch])
        cmd_parts.append("--")  # End of options — positional args only after this
        cmd_parts.extend([repo_url, str(repo_path)])

        cmd = " ".join(shlex.quote(p) for p in cmd_parts)
        result = await self.execute_command(cmd, timeout=300)

        if result.get("exit_code") == 0:
            # Update workspace metadata
            repos = self.ws.list_repos()
            self.ws.update_metadata(repos_cached=repos)
            result["repo_path"] = str(repo_path.relative_to(self.ws.root))
            result["cached"] = False

        return result

    async def _git_pull(
        self,
        repo_name: str,
        branch: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Pull latest changes for a cached repo."""
        repo_path = self.ws.get_repo_path(repo_name)
        if not repo_path.exists():
            return {"error": f"Repo not cached: {repo_name}. Clone it first."}

        cmd = "git pull"
        if branch:
            cmd = f"git checkout {shlex.quote(branch)} && git pull"

        result = await self.execute_command(
            cmd,
            timeout=120,
            cwd=f"repos/{repo_name}",
        )
        result["repo_path"] = str(repo_path.relative_to(self.ws.root))
        result["cached"] = True
        return result

    # =========================================================================
    # Command validation
    # =========================================================================

    def _validate_command(self, command: str) -> Optional[str]:
        """Validate command against whitelist and blocked patterns.

        Returns None if valid, error message string if blocked.
        """
        if not command or not command.strip():
            return "Empty command"

        # Check blocked patterns first (highest priority)
        for pattern in _BLOCKED_RE:
            if pattern.search(command):
                return f"Command blocked by security policy: matches pattern '{pattern.pattern}'"

        # Extract the first binary from the command
        # Handle: pipes, &&, ||, semicolons, subshells
        # We check each command segment
        # Split on actual command separators (&&, ||, ;, |) — NOT single &
        # which appears in shell redirects like 2>&1.
        segments = re.split(r'&&|\|\||[;|]', command)
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue

            # Handle env var prefix (e.g., "FOO=bar python script.py")
            parts = segment.split()
            binary = None
            for part in parts:
                if "=" in part and not part.startswith("-"):
                    continue  # Skip env var assignments
                binary = part
                break

            if binary is None:
                continue

            # Reject binaries specified with path separators or relative paths
            # (e.g., /usr/bin/python, ./malicious, ../escape)
            if "/" in binary or "\\" in binary or binary.startswith("."):
                return (
                    f"Path-based binary '{binary}' not allowed. "
                    f"Use plain binary names only (e.g., 'python', not '/usr/bin/python')."
                )

            binary_name = binary

            if binary_name not in ALLOWED_COMMANDS:
                return (
                    f"Command '{binary_name}' not in whitelist. "
                    f"Allowed: {', '.join(sorted(ALLOWED_COMMANDS))}"
                )

        return None

    # =========================================================================
    # Environment sandboxing
    # =========================================================================

    def _build_sandboxed_env(self, extras: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Build a stripped-down environment for subprocess execution.

        Only includes essential vars. Removes any sensitive host env vars.
        """
        env = {
            # Minimal PATH — only standard locations
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            # Workspace identity
            "WORKSPACE_ID": self.ws.workspace_id,
            "HOME": str(self.ws.root),
            # Git config location
            "GIT_CONFIG_GLOBAL": str(self.ws.root / ".gitconfig"),
            # SSH config
            "GIT_SSH_COMMAND": f"ssh -F {self.ws.root / '.ssh' / 'config'} -i {self.ws.root / '.ssh' / 'id_ed25519'} -o StrictHostKeyChecking=no",
            # Locale
            "LANG": "en_US.UTF-8",
            "LC_ALL": "en_US.UTF-8",
            # Python
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            # Node
            "NODE_ENV": "test",
            "npm_config_cache": str(self.ws.root / ".npm_cache"),
        }

        # Add any task-specific extras
        if extras:
            env.update(extras)

        return env
