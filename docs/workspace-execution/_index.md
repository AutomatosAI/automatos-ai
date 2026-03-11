# Workspace Execution

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



## Purpose and Scope

This document covers the **Workspace Execution** subsystem, which provides sandboxed code execution, file operations, and repository management in isolated workspace environments. Each workspace gets its own persistent directory on a shared volume where agents can clone repos, run commands, read/write files, and execute development workflows.

**Related pages:**
- For agent-driven workflow execution, see [Recipe Execution Engine](#4.2)
- For task submission and management APIs, see [Task Management](#9.6)
- For security policies and multi-tenancy, see [Data Isolation](#12.3)

The workspace execution system consists of three main components:

1. **Workspace Worker Service** — Long-running process that consumes tasks from Redis queues and executes them in isolated directories
2. **Orchestrator APIs** — REST endpoints for task submission, file browsing, and GitHub integration
3. **WorkspaceClient** — HTTP client library that proxies orchestrator requests to the worker

---

## System Architecture

The workspace execution system follows a **task queue** pattern with strict orchestrator-worker separation:

```mermaid
graph TB
    subgraph "Orchestrator Layer (Port 8000)"
        TaskAPI["POST /api/tasks/submit<br/>orchestrator/api/tasks.py"]
        FilesAPI["GET /api/workspaces/{id}/files<br/>orchestrator/api/workspace_files.py"]
        GitHubAPI["POST /api/workspaces/{id}/github/clone<br/>orchestrator/api/workspace_github.py"]
        Client["WorkspaceClient<br/>orchestrator/core/workspace_client.py"]
    end
    
    subgraph "Queue Layer"
        RedisQueues["Redis Priority Queues<br/>workspace:tasks:critical<br/>workspace:tasks:high<br/>workspace:tasks:normal<br/>workspace:tasks:low"]
        RedisStatus["Redis Status Hashes<br/>workspace:task:{id}:status"]
        RedisPubSub["Redis Pub/Sub<br/>workspace:task:{id}:events"]
    end
    
    subgraph "Worker Service (Port 8081)"
        Worker["WorkspaceWorker<br/>services/workspace-worker/main.py"]
        Executor["WorkspaceToolExecutor<br/>services/workspace-worker/executor.py"]
        Manager["WorkspaceManager<br/>services/workspace-worker/workspace_manager.py"]
        HealthAPI["HTTP Server<br/>GET /health<br/>GET /workspaces/{id}/files<br/>POST /workspaces/{id}/exec"]
    end
    
    subgraph "Storage Layer"
        Volume["/workspaces/{workspace_id}/<br/>repos/<br/>tasks/<br/>artifacts/<br/>.ssh/<br/>.gitconfig"]
        DB[(PostgreSQL<br/>task_executions table)]
    end
    
    TaskAPI -->|"1. Enqueue task"| RedisQueues
    GitHubAPI -->|"1. Enqueue git_clone"| RedisQueues
    
    Worker -->|"2. RPOP from queues"| RedisQueues
    Worker -->|"3. Update status"| RedisStatus
    Worker -->|"4. Publish events"| RedisPubSub
    Worker -->|"5. Persist state"| DB
    
    Worker --> Executor
    Executor --> Manager
    Manager --> Volume
    
    FilesAPI --> Client
    Client -->|"HTTP GET/POST"| HealthAPI
    HealthAPI --> Manager
    
    TaskAPI -.->|"Track submission"| DB
```

**Sources:** [orchestrator/api/tasks.py:1-404](), [services/workspace-worker/main.py:1-832](), [orchestrator/core/workspace_client.py:1-185]()

---

## Workspace Worker Service

### Main Loop

The `WorkspaceWorker` class implements an ARQ-style queue consumer without the ARQ library dependency. It polls Redis priority queues in strict order (critical → high → normal → low) and executes tasks concurrently up to `WORKER_CONCURRENCY` (default: 3).

```mermaid
stateDiagram-v2
    [*] --> Starting
    Starting --> Connecting: "Connect to Redis"
    Connecting --> HealthCheck: "Verify Redis ready"
    HealthCheck --> Polling: "Start consume loop"
    
    Polling --> CheckSemaphore: "Check concurrency"
    CheckSemaphore --> Polling: "At max, sleep 0.5s"
    CheckSemaphore --> Dequeue: "Semaphore available"
    
    Dequeue --> Polling: "All queues empty"
    Dequeue --> AcquireSem: "Got task payload"
    
    AcquireSem --> SpawnTask: "Acquire semaphore"
    SpawnTask --> Polling: "asyncio.create_task"
    
    SpawnTask --> Provision: "Task: Provision workspace"
    Provision --> CheckQuota: "WorkspaceManager"
    CheckQuota --> Failed: "Over quota"
    CheckQuota --> CreateTaskDir: "Under quota"
    CreateTaskDir --> InjectCreds: "tasks/task_{id}/"
    InjectCreds --> ExecuteSteps: "SSH keys + git config"
    
    ExecuteSteps --> StepLoop: "For each step"
    StepLoop --> CheckCancel: "Check Redis status"
    CheckCancel --> Cancelled: "status == cancelled"
    CheckCancel --> PublishProgress: "Not cancelled"
    PublishProgress --> ExecuteStep: "Publish event"
    ExecuteStep --> StepResult: "WorkspaceToolExecutor"
    StepResult --> StepLoop: "Continue if more steps"
    StepResult --> Completed: "All steps done"
    
    Completed --> CleanupTask: "Write result to Redis"
    Failed --> CleanupTask
    Cancelled --> CleanupTask
    CleanupTask --> ReleaseSem: "Remove ephemeral dir"
    ReleaseSem --> Polling: "Release semaphore"
    
    Polling --> Shutdown: "SIGTERM/SIGINT"
    Shutdown --> WaitActive: "_running = False"
    WaitActive --> Cleanup: "await active tasks"
    Cleanup --> [*]
```

**Sources:** [services/workspace-worker/main.py:60-226]()

### Concurrency Control

The worker uses an `asyncio.Semaphore` to limit concurrent task execution. Each task execution:

1. Acquires the semaphore (blocks if at limit)
2. Spawns an `asyncio.Task` for execution
3. Releases the semaphore in the `finally` block

This prevents resource exhaustion when many tasks are queued simultaneously.

**Sources:** [services/workspace-worker/main.py:71-180]()

### Health Check Server

The worker runs an `aiohttp` HTTP server on port 8081 (configurable via `WORKER_HEALTH_PORT`) for two purposes:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Kubernetes liveness probe, returns `{"status": "healthy", "active_tasks": N}` |
| `/workspaces/{id}/files` | GET | Directory listing for code viewer widget (proxied from orchestrator) |
| `/workspaces/{id}/files/content` | GET | File content for code viewer (max 2MB) |
| `/workspaces/{id}/exec` | POST | Direct command execution (bypasses queue) |
| `/workspaces/{id}/files/write` | POST | Direct file write (bypasses queue) |
| `/workspaces/{id}/files/grep` | GET | Search file contents via grep |
| `/workspaces/{id}/git` | POST | Git operations (status, diff, commit, push) |

The HTTP endpoints use `WORKER_INTERNAL_TOKEN` for authentication (header: `X-Internal-Token`). This prevents public internet access while allowing orchestrator-to-worker communication.

**Sources:** [services/workspace-worker/main.py:461-818]()

---

## Workspace Filesystem

### Directory Layout

Each workspace gets a persistent directory tree on the worker volume:

```
/workspaces/{workspace_id}/
├── repos/                    ← Cloned repositories (persistent, git pull on revisit)
│   └── {repo_name}/
│       ├── .git/
│       └── src/
├── tasks/                    ← Ephemeral per-task execution directories
│   └── task_{task_id}/
├── artifacts/                ← Build outputs, test reports (persistent)
│   └── test-results.json
├── .ssh/                     ← Deploy keys for private repo access
│   ├── id_ed25519
│   └── config
├── .gitconfig                ← Git author identity
├── .task_env_{task_id}       ← Task-specific env vars (deleted after task)
└── .workspace_meta.json      ← Workspace metadata
```

**Sources:** [services/workspace-worker/workspace_manager.py:10-18]()

### WorkspaceManager API

The `WorkspaceManager` class provides the core filesystem abstraction:

| Method | Purpose |
|--------|---------|
| `ensure_workspace_exists()` | Create directory tree + metadata on first use |
| `get_usage_bytes()` | Calculate current disk usage recursively |
| `check_quota()` | Enforce `DEFAULT_QUOTA_GB` storage limit |
| `create_task_dir(task_id)` | Create ephemeral `tasks/task_{id}/` directory |
| `cleanup_task(task_id)` | Remove ephemeral dir + task-specific credentials |
| `inject_credentials(task_id, creds)` | Write SSH keys, git config, env vars |
| `resolve_safe_path(rel_path)` | Validate path stays within workspace (security boundary) |
| `get_repo_path(repo_name)` | Return path for a cached repo |
| `list_repos()` | List all cached repos in `repos/` |

**Sources:** [services/workspace-worker/workspace_manager.py:36-303]()

### Path Safety

All path operations go through `resolve_safe_path()`, which enforces these security rules:

1. **No null bytes** — `\x00` in path raises `SecurityError`
2. **No absolute paths** — Paths starting with `/` are rejected
3. **No directory traversal** — Resolved path must stay within `{workspace_id}/` (checked via `Path.relative_to()`)
4. **Symlinks resolved** — Uses `Path.resolve()` to follow symlinks and verify containment

```python
# Example: orchestrator/workspace_manager.py:228-253
def resolve_safe_path(self, relative_path: str) -> Path:
    if "\x00" in relative_path:
        raise SecurityError("Null byte in path")
    if relative_path.startswith("/"):
        raise SecurityError("Absolute path not allowed")
    
    resolved = (self.root / relative_path).resolve()
    base_resolved = self.root.resolve()
    
    try:
        resolved.relative_to(base_resolved)  # Raises ValueError if escapes
    except ValueError:
        raise SecurityError(f"Path traversal blocked: '{relative_path}'")
    
    return resolved
```

**Sources:** [services/workspace-worker/workspace_manager.py:228-253]()

### Storage Quotas

The worker enforces per-workspace storage quotas to prevent disk exhaustion:

1. Default quota: `DEFAULT_QUOTA_GB` (default: 5GB, configurable via env var)
2. Quota check before each task execution — task fails immediately if over quota
3. Usage calculation via recursive `rglob("*")` (cached in `_current_usage`)

Quota enforcement happens in the task execution flow:

```mermaid
sequenceDiagram
    participant Worker as "WorkspaceWorker"
    participant Manager as "WorkspaceManager"
    participant Volume as "/workspaces/ Volume"
    
    Worker->>Manager: "check_quota()"
    Manager->>Volume: "rglob('*') all files"
    Volume-->>Manager: "file list"
    Manager->>Manager: "sum(f.stat().st_size)"
    
    alt "Usage < quota_bytes"
        Manager-->>Worker: "True"
        Worker->>Manager: "create_task_dir(task_id)"
        Worker->>Worker: "Execute task steps"
    else "Usage >= quota_bytes"
        Manager-->>Worker: "False"
        Worker->>Worker: "Fail task with quota error"
        Worker->>Worker: "Skip execution, publish error event"
    end
```

**Sources:** [services/workspace-worker/main.py:251-264](), [services/workspace-worker/workspace_manager.py:83-114]()

---

## Task Lifecycle

### State Machine

Tasks flow through the following states:

```mermaid
stateDiagram-v2
    [*] --> Queued: "submit_task() enqueues"
    
    Queued --> Running: "Worker dequeues + starts"
    Queued --> Cancelled: "User cancels via API"
    Queued --> Failed: "Redis enqueue error"
    
    Running --> Completed: "All steps succeed"
    Running --> Failed: "Step error / exception"
    Running --> TimedOut: "Exceeds timeout_seconds"
    Running --> Cancelled: "User cancels mid-execution"
    
    Completed --> [*]
    Failed --> [*]
    TimedOut --> [*]
    Cancelled --> [*]
```

### Submission Flow

Tasks are submitted via `POST /api/tasks/submit` with this atomic two-phase protocol:

**Phase 1: Database Insert**
```sql
INSERT INTO task_executions (
    id, workspace_id, task_type, status, priority,
    runner_backend, configuration, submitted_at
) VALUES (?, ?, 'background_job', 'queued', ?, 'queued', ?, NOW())
```

**Phase 2: Redis Enqueue** (only if DB insert succeeds)
```python
# Status hash
redis.hset(f"workspace:task:{task_id}:status", {
    "status": "queued",
    "workspace_id": workspace_id,
    "submitted_at": now.isoformat(),
})

# Active tasks set
redis.sadd(f"workspace:ws:{workspace_id}:active_tasks", task_id)

# Priority queue (critical > high > normal > low)
redis.lpush(f"workspace:tasks:{priority}", json.dumps(payload))
```

If Phase 2 fails, the DB row is immediately marked as `failed` with `error_message = "Redis enqueue failed: {error}"`.

**Sources:** [orchestrator/api/tasks.py:62-173]()

### Task Payload Structure

The JSON payload enqueued to Redis contains all execution metadata:

```json
{
  "task_id": "uuid-v4",
  "task_type": "background_job",
  "workspace_id": "workspace-uuid",
  "agent_id": null,
  "priority": "normal",
  "timeout_seconds": 300,
  "steps": [
    {
      "action": "git_clone",
      "repo": "https://x-access-token:TOKEN@github.com/org/repo.git",
      "branch": "main",
      "description": "Clone repo"
    },
    {
      "action": "execute_command",
      "command": "pytest tests/",
      "cwd": "repos/repo",
      "timeout": 120
    }
  ],
  "credentials": {
    "ssh_private_key": "-----BEGIN OPENSSH PRIVATE KEY-----\n...",
    "git_name": "Agent",
    "git_email": "agent@automatos.app"
  },
  "created_at": "2025-01-15T10:30:00Z"
}
```

**Sources:** [orchestrator/api/tasks.py:89-99]()

### Event Streaming

Tasks publish real-time events via Redis pub/sub to channel `workspace:task:{task_id}:events`. The orchestrator exposes this as Server-Sent Events (SSE) via `GET /api/tasks/{task_id}/events`:

| Event Type | Data | When |
|------------|------|------|
| `status_changed` | `{"status": "running"}` | Transitions between states |
| `progress_update` | `{"step": 2, "total_steps": 5, "description": "..."}` | Before each step execution |
| `error` | `{"error": "..."}` | Step failure or exception |

```mermaid
sequenceDiagram
    participant Client as "Frontend<br/>(EventSource)"
    participant API as "GET /api/tasks/{id}/events<br/>orchestrator/api/tasks.py"
    participant Redis as "Redis Pub/Sub"
    participant Worker as "WorkspaceWorker"
    
    Client->>API: "Connect SSE stream"
    API->>Redis: "SUBSCRIBE workspace:task:{id}:events"
    
    Worker->>Redis: "PUBLISH status_changed"
    Redis->>API: "Event message"
    API->>Client: "event: status_changed\ndata: {...}"
    
    Worker->>Redis: "PUBLISH progress_update"
    Redis->>API: "Event message"
    API->>Client: "event: progress_update\ndata: {...}"
    
    Worker->>Redis: "PUBLISH status_changed (completed)"
    Redis->>API: "Event message"
    API->>Client: "event: status_changed\ndata: {status: completed}"
    API->>Client: "Close stream"
```

**Sources:** [orchestrator/api/tasks.py:352-403](), [services/workspace-worker/main.py:422-434]()

---

## Command Execution

### Security Model

The `WorkspaceToolExecutor` enforces a **five-layer security model** for command execution:

```mermaid
graph TB
    Input["User Command:<br/>'pytest tests/ && npm test'"]
    
    Layer1["Layer 1: Command Whitelist<br/>ALLOWED_COMMANDS set"]
    Layer2["Layer 2: Blocked Patterns<br/>_BLOCKED_RE regexes"]
    Layer3["Layer 3: Path Safety<br/>WorkspaceManager.resolve_safe_path()"]
    Layer4["Layer 4: Sandboxed Environment<br/>Stripped PATH, no host vars"]
    Layer5["Layer 5: Output Limits<br/>MAX_STDOUT_BYTES, MAX_STDERR_BYTES"]
    
    Result["Execute via asyncio.create_subprocess_shell"]
    
    Input --> Layer1
    Layer1 -->|"Binary not in whitelist"| Reject1["Return {error: 'Command not in whitelist'}"]
    Layer1 -->|"Binary allowed"| Layer2
    
    Layer2 -->|"Matches blocked pattern"| Reject2["Return {error: 'Blocked by security policy'}"]
    Layer2 -->|"No match"| Layer3
    
    Layer3 -->|"Path escapes workspace"| Reject3["Return {error: 'Working directory invalid'}"]
    Layer3 -->|"Path safe"| Layer4
    
    Layer4 --> Layer5
    Layer5 --> Result
    
    Result --> Output["Return {exit_code, stdout, stderr, duration_ms, truncated}"]
```

**Sources:** [services/workspace-worker/executor.py:1-537]()

### Command Whitelist

Only these binaries are allowed (exact name matching, no path components):

| Category | Commands |
|----------|----------|
| **Shell** | `sh`, `bash`, `cd`, `pwd`, `export`, `source`, `test` |
| **VCS** | `git` |
| **Python** | `python`, `python3`, `pip`, `pip3`, `uv`, `pytest`, `ruff`, `black`, `mypy`, `isort`, `flake8`, `coverage`, `tox` |
| **Node** | `node`, `npm`, `npx`, `pnpm`, `yarn`, `vitest`, `jest`, `tsc`, `eslint`, `prettier` |
| **General** | `ls`, `cat`, `grep`, `find`, `tree`, `wc`, `sort`, `diff`, `jq`, `sed`, `awk`, `curl`, `wget`, `make`, `cmake`, `tar`, `gzip`, `zip`, `touch`, `mkdir`, `cp`, `mv`, `rm`, `chmod`, `echo`, `env`, `which` |
| **Other Languages** | `cargo`, `go`, `ruby`, `java`, `javac`, `mvn`, `gradle`, `rustc`, `gcc`, `g++` |

Commands with path separators (e.g., `/usr/bin/python`, `./malicious`, `../escape`) are rejected to prevent binary injection via relative/absolute paths.

**Sources:** [services/workspace-worker/executor.py:35-73]()

### Blocked Patterns

These regex patterns are **always blocked**, even if the binary is whitelisted:

| Pattern | Why Blocked |
|---------|-------------|
| `rm\s+-rf\s+/\s*$` | Deletes entire filesystem |
| `rm\s+-rf\s+/[^w]` | Deletes anything except `/workspaces/` |
| `\bsudo\b` | Privilege escalation |
| `\bsu\s` | User switching |
| `\bchmod\s+777\b` | Dangerous permissions |
| `\bkubectl\b` | Kubernetes cluster access |
| `>\s*/dev/` | Device file access |
| `\bmkfs\b` | Filesystem formatting |
| `\bdd\s+if=` | Raw disk operations |
| `\biptables\b` | Firewall manipulation |
| `\bsystemctl\b` | Service management |
| `\bpasswd\b` | Password changes |
| `\buseradd\b`, `\buserdel\b` | User management |
| `\bmount\b`, `\bumount\b` | Filesystem mounting |
| `` ` `` | Backtick command substitution |
| `\n` | Embedded newlines |

**Sources:** [services/workspace-worker/executor.py:76-98]()

### Validation Algorithm

Commands are validated by splitting on shell operators (`&&`, `||`, `;`, `|`) and checking each segment:

```python
# services/workspace-worker/executor.py:448-500
def _validate_command(self, command: str) -> Optional[str]:
    # 1. Check blocked patterns first (highest priority)
    for pattern in _BLOCKED_RE:
        if pattern.search(command):
            return f"Command blocked: matches pattern '{pattern.pattern}'"
    
    # 2. Split on shell operators: &&, ||, ;, |
    segments = re.split(r'&&|\|\||[;|]', command)
    
    for segment in segments:
        # 3. Extract binary name (skip env var assignments like FOO=bar)
        parts = segment.strip().split()
        binary = None
        for part in parts:
            if "=" in part and not part.startswith("-"):
                continue  # Skip env assignments
            binary = part
            break
        
        # 4. Reject path-based binaries (/, \, or leading .)
        if "/" in binary or "\\" in binary or binary.startswith("."):
            return "Path-based binary not allowed"
        
        # 5. Check whitelist
        if binary not in ALLOWED_COMMANDS:
            return f"Command '{binary}' not in whitelist"
    
    return None  # Valid
```

**Sources:** [services/workspace-worker/executor.py:448-500]()

### Sandboxed Environment

Subprocesses run with a **stripped environment** that prevents host variable leakage:

```python
# services/workspace-worker/executor.py:506-536
env = {
    # Minimal PATH — only standard locations
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    
    # Workspace identity
    "WORKSPACE_ID": workspace_id,
    "HOME": f"/workspaces/{workspace_id}",
    
    # Git config
    "GIT_CONFIG_GLOBAL": f"/workspaces/{workspace_id}/.gitconfig",
    "GIT_SSH_COMMAND": f"ssh -F .ssh/config -i .ssh/id_ed25519 -o StrictHostKeyChecking=no",
    
    # Locale
    "LANG": "en_US.UTF-8",
    "LC_ALL": "en_US.UTF-8",
    
    # Python
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUNBUFFERED": "1",
    
    # Node
    "NODE_ENV": "test",
    "npm_config_cache": f"/workspaces/{workspace_id}/.npm_cache",
}
```

Host environment variables (e.g., `AWS_ACCESS_KEY_ID`, `DATABASE_URL`) are **not** inherited.

**Sources:** [services/workspace-worker/executor.py:506-536]()

### Output Limits

Command output is capped to prevent memory exhaustion:

- **STDOUT**: 100KB (`MAX_STDOUT_BYTES`)
- **STDERR**: 50KB (`MAX_STDERR_BYTES`)

Truncated output sets `truncated: true` in the result dict.

**Sources:** [services/workspace-worker/executor.py:100-106]()

---

## File Operations

The `WorkspaceToolExecutor` provides sandboxed file operations that agents call via `workspace_*` tools:

### Read File

```python
# services/workspace-worker/executor.py:230-252
async def read_file(self, path: str, max_bytes: int = 500_000) -> Dict[str, Any]:
    safe_path = self.ws.resolve_safe_path(path)  # Path safety check
    if not safe_path.exists():
        return {"error": "File not found"}
    if not safe_path.is_file():
        return {"error": "Not a file"}
    
    content = safe_path.read_bytes()
    truncated = len(content) > max_bytes
    return {
        "content": content[:max_bytes].decode("utf-8", errors="replace"),
        "size_bytes": len(content),
        "truncated": truncated,
    }
```

**Sources:** [services/workspace-worker/executor.py:230-252]()

### Write File

```python
# services/workspace-worker/executor.py:254-270
async def write_file(self, path: str, content: str) -> Dict[str, Any]:
    safe_path = self.ws.resolve_safe_path(path)  # Path safety check
    safe_path.parent.mkdir(parents=True, exist_ok=True)  # Create parent dirs
    safe_path.write_text(content)
    return {
        "written": True,
        "path": str(safe_path.relative_to(self.ws.root)),
        "size_bytes": len(content.encode()),
    }
```

**Sources:** [services/workspace-worker/executor.py:254-270]()

### List Directory

```python
# services/workspace-worker/executor.py:272-300
async def list_directory(self, path: str = ".") -> Dict[str, Any]:
    safe_path = self.ws.resolve_safe_path(path)
    if not safe_path.is_dir():
        return {"error": "Not a directory"}
    
    entries = []
    for item in sorted(safe_path.iterdir()):
        stat = item.stat()
        entries.append({
            "name": item.name,
            "type": "dir" if item.is_dir() else "file",
            "size": stat.st_size if item.is_file() else None,
        })
    
    return {"path": path, "entries": entries, "count": len(entries)}
```

**Sources:** [services/workspace-worker/executor.py:272-300]()

### HTTP Endpoints for File Browsing

The worker's HTTP server exposes file operations for the code viewer widget:

| Endpoint | Method | Purpose | Max Size | Filter |
|----------|--------|---------|----------|--------|
| `/workspaces/{id}/files` | GET | Directory listing | 500 entries | Hides `.ssh`, `.gitconfig`, `.aws`, `.task_env_*` |
| `/workspaces/{id}/files/content` | GET | File content | 2 MB | Hides sensitive paths |
| `/workspaces/{id}/files/grep` | GET | Search via `grep -rn` | 200 matches | User-provided pattern + include glob |
| `/workspaces/{id}/files/write` | POST | Direct file write | No limit | Path safety enforced |

Sensitive paths (`.ssh`, `.gitconfig`, `.aws`, `.workspace_meta.json`, `.task_env_*`) are **blocked** from file browsing to prevent credential leakage.

**Sources:** [services/workspace-worker/main.py:525-759]()

---

## GitHub Integration

### OAuth-Authenticated Cloning

The GitHub integration uses Composio to retrieve OAuth tokens and inject them into clone URLs:

```mermaid
sequenceDiagram
    participant UI as "Frontend<br/>RepoSelector.tsx"
    participant API as "POST /api/workspaces/{id}/github/clone<br/>orchestrator/api/workspace_github.py"
    participant Composio as "Composio SDK<br/>get_app_access_token()"
    participant Redis as "Redis Queue"
    participant Worker as "WorkspaceWorker"
    participant Git as "git clone"
    
    UI->>API: "Clone request {repo_url, branch}"
    
    API->>API: "Validate URL (HTTPS only, allowed hosts)"
    API->>API: "Validate branch (no --, @{, ..)"
    
    API->>Composio: "get_app_access_token(entity_id, 'GITHUB')"
    Composio-->>API: "OAuth token"
    
    API->>API: "Inject token into URL:<br/>https://x-access-token:TOKEN@github.com/..."
    
    API->>Redis: "Enqueue git_clone task"
    Redis-->>API: "task_id"
    API-->>UI: "{task_id, status: queued, events_url}"
    
    Worker->>Redis: "Dequeue task"
    Worker->>Worker: "WorkspaceManager.get_repo_path()"
    
    alt "Repo already cached"
        Worker->>Git: "git pull (in repos/ dir)"
    else "First clone"
        Worker->>Git: "git clone --depth 1 --branch BRANCH -- URL DEST"
    end
    
    Git-->>Worker: "exit_code, stdout, stderr"
    Worker->>Worker: "Update workspace metadata (repos_cached list)"
    Worker->>Redis: "Publish status_changed event"
```

**Sources:** [orchestrator/api/workspace_github.py:167-293](), [services/workspace-worker/executor.py:368-419]()

### URL Validation (PRD-66)

Clone URLs are validated with strict rules to prevent injection attacks:

```python
# orchestrator/api/workspace_github.py:69-91
@field_validator("repo_url")
@classmethod
def validate_repo_url(cls, v: str) -> str:
    parsed = urlparse(v)
    
    # 1. HTTPS only
    if parsed.scheme != "https":
        raise ValueError("Only HTTPS clone URLs are allowed")
    
    # 2. Allowed hosts only
    if parsed.hostname not in {"github.com", "gitlab.com", "bitbucket.org"}:
        raise ValueError(f"Host not allowed: {parsed.hostname}")
    
    # 3. No embedded credentials
    if parsed.username or parsed.password:
        raise ValueError("Clone URL must not contain embedded credentials")
    
    return v

@field_validator("branch")
@classmethod
def validate_branch(cls, v: Optional[str]) -> Optional[str]:
    if not v:
        return None
    
    # 4. No dangerous branch names (PRD-70 FIX-01)
    if ".." in v or "@{" in v or not re.match(r"^[A-Za-z0-9._/\-]+$", v):
        raise ValueError("Invalid branch name")
    
    return v
```

**Sources:** [orchestrator/api/workspace_github.py:69-91]()

### Git Clone Command Construction (PRD-70 FIX-01)

The worker uses `--` separator to prevent argument injection:

```python
# services/workspace-worker/executor.py:400-410
cmd_parts = ["git", "clone"]
if shallow:
    cmd_parts.extend(["--depth", "1"])
if branch:
    cmd_parts.extend(["--branch", branch])
cmd_parts.append("--")  # End of options — positional args only after this
cmd_parts.extend([repo_url, str(repo_path)])

cmd = " ".join(shlex.quote(p) for p in cmd_parts)
# Result: git clone --depth 1 --branch main -- https://... /workspaces/.../repos/repo
```

This prevents attacks like `--upload-pack=malicious-script` being injected via branch names.

**Sources:** [services/workspace-worker/executor.py:368-419]()

### Repository Caching

Cloned repos are cached in `repos/` for the lifetime of the workspace:

1. **First clone**: `git clone --depth 1` (shallow clone for speed)
2. **Subsequent access**: `git pull` (updates existing clone)
3. **Metadata tracking**: Workspace `.workspace_meta.json` stores `repos_cached: ["repo-name", ...]`

The worker checks `WorkspaceManager.repo_exists(repo_name)` before cloning to decide between clone vs pull.

**Sources:** [services/workspace-worker/executor.py:393-398]()

---

## Agent Tools

Agents interact with workspaces via **platform tools** registered in the `ActionRegistry`. These tools route through the orchestrator's `WorkspaceClient` to the worker:

### Tool Definitions

| Tool Name | Action | Permission | Parameters |
|-----------|--------|------------|------------|
| `workspace_read_file` | Read file content | `read` | `path` (relative to workspace root) |
| `workspace_write_file` | Write/create file | `write` | `path`, `content` |
| `workspace_list_dir` | Directory listing | `read` | `path` (default: `.`) |
| `workspace_grep` | Search file contents | `read` | `pattern`, `path`, `include` (glob), `max_results` |
| `workspace_exec` | Run shell command | `write` | `command`, `cwd`, `timeout` |
| `workspace_git` | Git operations | `write` | `operation` (enum), `cwd`, `args` |

**Sources:** [orchestrator/modules/tools/discovery/workspace_actions.py:15-248]()

### Tool Routing Flow

```mermaid
sequenceDiagram
    participant Agent as "Agent Runtime<br/>(LLM tool call)"
    participant Router as "ToolRouter<br/>orchestrator/modules/tools/tool_router.py"
    participant Executor as "UnifiedToolExecutor<br/>orchestrator/modules/tools/execution/unified_executor.py"
    participant Client as "WorkspaceClient<br/>orchestrator/core/workspace_client.py"
    participant Worker as "Worker HTTP API<br/>Port 8081"
    
    Agent->>Router: "execute_tool('workspace_exec', {command: 'pytest'})"
    Router->>Executor: "execute(action_def, params, context)"
    
    Executor->>Executor: "Detect category == 'workspace_exec'"
    Executor->>Client: "exec_command(command='pytest', cwd=None, timeout=120)"
    
    Client->>Worker: "POST /workspaces/{id}/exec<br/>X-Internal-Token header"
    Worker->>Worker: "WorkspaceToolExecutor.execute_command()"
    Worker->>Worker: "Validate command, sanitize env, run subprocess"
    Worker-->>Client: "{exit_code: 0, stdout: '...', stderr: '...', duration_ms: 1234}"
    
    Client-->>Executor: "Result dict"
    Executor-->>Router: "Format for LLM consumption"
    Router-->>Agent: "Tool result in next message"
```

**Sources:** [orchestrator/modules/tools/tool_router.py:1-575](), [orchestrator/core/workspace_client.py:56-185]()

### WorkspaceClient Methods

The `WorkspaceClient` is a thin HTTP wrapper around the worker's endpoints:

```python
# orchestrator/core/workspace_client.py:56-185
class WorkspaceClient:
    def __init__(self, workspace_id: str):
        self.workspace_id = workspace_id
    
    async def read_file(self, path: str) -> Dict[str, Any]:
        url = f"{WORKER_URL}/workspaces/{self.workspace_id}/files/content"
        resp = await httpx_client.get(url, params={"path": path})
        return resp.json()
    
    async def write_file(self, path: str, content: str) -> Dict[str, Any]:
        url = f"{WORKER_URL}/workspaces/{self.workspace_id}/files/write"
        resp = await httpx_client.post(url, json={"path": path, "content": content})
        return resp.json()
    
    async def list_dir(self, path: str = ".") -> Dict[str, Any]:
        url = f"{WORKER_URL}/workspaces/{self.workspace_id}/files"
        resp = await httpx_client.get(url, params={"path": path})
        return resp.json()
    
    async def grep(self, pattern: str, path: str = ".", ...) -> Dict[str, Any]:
        url = f"{WORKER_URL}/workspaces/{self.workspace_id}/files/grep"
        resp = await httpx_client.get(url, params={...})
        return resp.json()
    
    async def exec_command(self, command: str, cwd: str = None, ...) -> Dict[str, Any]:
        url = f"{WORKER_URL}/workspaces/{self.workspace_id}/exec"
        resp = await httpx_client.post(url, json={...})
        return resp.json()
    
    async def git(self, operation: str, cwd: str = None, ...) -> Dict[str, Any]:
        url = f"{WORKER_URL}/workspaces/{self.workspace_id}/git"
        resp = await httpx_client.post(url, json={...})
        return resp.json()
```

All methods return `{"success": False, "error": "..."}` on connection errors.

**Sources:** [orchestrator/core/workspace_client.py:56-185]()

---

## Security & Sandboxing

The workspace execution system implements **defense-in-depth** with five security layers:

### Layer 1: URL Validation (PRD-66)

GitHub clone URLs are validated before task submission:

- **Scheme check**: Only `https://` is allowed (no `git://`, `ssh://`, `file://`)
- **Host allowlist**: Only `github.com`, `gitlab.com`, `bitbucket.org`
- **No embedded credentials**: URL must not contain `username:password@`
- **Branch name validation**: No `..`, `@{`, leading dashes (prevents `--upload-pack` injection)

**Sources:** [orchestrator/api/workspace_github.py:69-91]()

### Layer 2: Path Safety

All file/directory operations go through `WorkspaceManager.resolve_safe_path()`:

```python
# services/workspace-worker/workspace_manager.py:228-253
def resolve_safe_path(self, relative_path: str) -> Path:
    # 1. Block null bytes
    if "\x00" in relative_path:
        raise SecurityError("Null byte in path")
    
    # 2. Block absolute paths
    if relative_path.startswith("/"):
        raise SecurityError("Absolute path not allowed")
    
    # 3. Resolve symlinks + verify containment
    resolved = (self.root / relative_path).resolve()
    base_resolved = self.root.resolve()
    
    try:
        resolved.relative_to(base_resolved)  # Raises ValueError if escapes
    except ValueError:
        raise SecurityError(f"Path traversal blocked: '{relative_path}'")
    
    return resolved
```

This prevents attacks like:
- `../../../etc/passwd` (directory traversal)
- `/etc/passwd` (absolute path)
- `symlink-to-root` (symlink escape)

**Sources:** [services/workspace-worker/workspace_manager.py:228-253]()

### Layer 3: Command Whitelist

Only approved binaries from `ALLOWED_COMMANDS` can execute. Path-based binaries are rejected:

```python
# Allowed:
"pytest tests/"
"python3 -m pytest tests/"
"npm test && npm run lint"

# Rejected:
"/usr/bin/python script.py"  # Path-based binary
"./malicious-binary"          # Relative path binary
"../escape/binary"            # Directory traversal
"rm -rf /"                    # Blocked pattern
"sudo apt install malware"    # 'sudo' blocked
```

**Sources:** [services/workspace-worker/executor.py:35-73](), [services/workspace-worker/executor.py:448-500]()

### Layer 4: Environment Sandboxing

Subprocesses run with a **stripped environment**:

- **PATH**: Limited to `/usr/local/bin:/usr/bin:/bin` (no `/sbin`, no user paths)
- **HOME**: Set to workspace root (not host user's home)
- **No host variables**: `AWS_*`, `DATABASE_URL`, `SECRET_KEY` are not inherited
- **Git isolation**: `GIT_CONFIG_GLOBAL` points to workspace `.gitconfig`

**Sources:** [services/workspace-worker/executor.py:506-536]()

### Layer 5: Storage Quotas

Per-workspace disk usage is enforced to prevent exhaustion attacks:

1. Default quota: 5GB (`DEFAULT_QUOTA_GB`)
2. Checked before each task execution
3. Task fails immediately if over quota (no execution)
4. Usage calculation via `rglob("*")` + `sum(f.stat().st_size)`

**Sources:** [services/workspace-worker/workspace_manager.py:83-114]()

### Sensitive Path Filtering

The worker's HTTP server **blocks** access to sensitive paths:

```python
# services/workspace-worker/main.py:473-481
_SENSITIVE_NAMES = {
    ".ssh",           # SSH private keys
    ".gitconfig",     # Git credentials
    ".aws",           # AWS credentials
    ".gcp",           # GCP credentials
    ".workspace_meta.json",  # Workspace metadata
}

def _is_sensitive(name: str) -> bool:
    if name in _SENSITIVE_NAMES:
        return True
    if name.startswith(".task_env_"):  # Task-specific env vars
        return True
    return False
```

File browsing endpoints return 403 Forbidden if a path component matches.

**Sources:** [services/workspace-worker/main.py:473-481](), [services/workspace-worker/main.py:546-548](), [services/workspace-worker/main.py:609-611]()

---

## API Reference

### Task Management API

**Base URL:** `/api/tasks`

| Endpoint | Method | Purpose | Auth |
|----------|--------|---------|------|
| `/submit` | POST | Submit workspace task | Workspace JWT |
| `` (list) | GET | List recent tasks | Workspace JWT |
| `/{task_id}` | GET | Get task detail + result | Workspace JWT |
| `/{task_id}/cancel` | POST | Cancel queued/running task | Workspace JWT |
| `/{task_id}/events` | GET | SSE stream of task events | Workspace JWT |

**Request: POST /api/tasks/submit**
```json
{
  "steps": [
    {
      "action": "git_clone",
      "repo": "https://github.com/org/repo.git",
      "branch": "main"
    },
    {
      "action": "execute_command",
      "command": "pytest tests/",
      "cwd": "repos/repo",
      "timeout": 120
    }
  ],
  "priority": "normal",
  "timeout_seconds": 300
}
```

**Response: POST /api/tasks/submit**
```json
{
  "task_id": "uuid-v4",
  "status": "queued",
  "queue": "workspace:tasks:normal",
  "steps": 2,
  "events_url": "/api/tasks/{task_id}/events"
}
```

**Sources:** [orchestrator/api/tasks.py:1-404]()

### Workspace Files API

**Base URL:** `/api/workspaces/{workspace_id}`

| Endpoint | Method | Purpose | Auth |
|----------|--------|---------|------|
| `/files` | GET | Directory listing | Workspace JWT |
| `/files/content` | GET | Read file content | Workspace JWT |
| `/exec` | POST | Run shell command | Workspace JWT |

**Request: GET /files?path=repos/my-app**
```json
{
  "path": "repos/my-app",
  "entries": [
    {"name": "src", "type": "directory", "size": 0},
    {"name": "package.json", "type": "file", "size": 1234}
  ],
  "truncated": false
}
```

**Request: GET /files/content?path=repos/my-app/src/main.py**
```json
{
  "path": "repos/my-app/src/main.py",
  "name": "main.py",
  "content": "#!/usr/bin/env python3\n...",
  "size": 5678,
  "language": "python",
  "mime_type": "text/x-python"
}
```

**Sources:** [orchestrator/api/workspace_files.py:1-108]()

### GitHub Integration API

**Base URL:** `/api/workspaces/{workspace_id}/github`

| Endpoint | Method | Purpose | Auth |
|----------|--------|---------|------|
| `/repos` | GET | List user's GitHub repos | Workspace JWT |
| `/clone` | POST | Clone repo into workspace | Workspace JWT |

**Request: POST /clone**
```json
{
  "repo_url": "https://github.com/org/repo.git",
  "branch": "main"
}
```

**Response: POST /clone**
```json
{
  "task_id": "uuid-v4",
  "status": "queued",
  "events_url": "/api/tasks/{task_id}/events"
}
```

**Sources:** [orchestrator/api/workspace_github.py:1-294]()

### Worker HTTP API (Internal)

**Base URL:** `http://workspace-worker:8081` (internal only, requires `X-Internal-Token`)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check (public) |
| `/workspaces/{id}/files` | GET | Directory listing |
| `/workspaces/{id}/files/content` | GET | File content |
| `/workspaces/{id}/files/write` | POST | Write file |
| `/workspaces/{id}/files/grep` | GET | Search files |
| `/workspaces/{id}/exec` | POST | Run command |
| `/workspaces/{id}/git` | POST | Git operation |

**Sources:** [services/workspace-worker/main.py:798-804]()

---

## Configuration

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection for queues + pub/sub |
| `DATABASE_URL` | (required) | PostgreSQL connection for task_executions table |
| `WORKSPACE_VOLUME_PATH` | `/workspaces` | Persistent volume mount path |
| `WORKSPACE_DEFAULT_QUOTA_GB` | `5` | Default storage quota per workspace |
| `WORKER_CONCURRENCY` | `3` | Max concurrent task executions |
| `WORKER_HEALTH_PORT` | `8081` | HTTP server port for health checks + file API |
| `WORKER_BIND_HOST` | `0.0.0.0` | HTTP server bind address |
| `WORKER_INTERNAL_TOKEN` | (optional) | Bearer token for worker API auth |
| `WORKER_INTERNAL_URL` | `http://workspace-worker:8081` | Worker HTTP URL (orchestrator config) |

**Sources:** [services/workspace-worker/main.py:14-19](), [services/workspace-worker/workspace_manager.py:32-33](), [orchestrator/core/workspace_client.py:20-44]()

### Docker Compose

The workspace worker runs as a separate service in the `docker-compose.yml`:

```yaml
workspace-worker:
  build:
    context: ./services/workspace-worker
    dockerfile: Dockerfile
  ports:
    - "8081:8081"
  volumes:
    - workspace_data:/workspaces  # Persistent volume for all workspaces
  environment:
    - REDIS_URL=redis://redis:6379/0
    - DATABASE_URL=postgresql://...
    - WORKER_CONCURRENCY=3
    - WORKER_HEALTH_PORT=8081
    - WORKER_INTERNAL_TOKEN=${WORKER_INTERNAL_TOKEN}
  depends_on:
    redis:
      condition: service_healthy
    postgres:
      condition: service_healthy
  profiles:
    - workers
    - all
```

**Sources:** [docker-compose.yml:1-282]()

---