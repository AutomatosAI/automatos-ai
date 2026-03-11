# Workspace API Reference

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [frontend/app/admin/plugins/page.tsx](frontend/app/admin/plugins/page.tsx)
- [frontend/components/widgets/CodingCanvasWidget/RepoSelector.tsx](frontend/components/widgets/CodingCanvasWidget/RepoSelector.tsx)
- [frontend/lib/api-client.ts](frontend/lib/api-client.ts)
- [orchestrator/.env.example](orchestrator/.env.example)
- [orchestrator/api/agent_plugins.py](orchestrator/api/agent_plugins.py)
- [orchestrator/api/tasks.py](orchestrator/api/tasks.py)
- [orchestrator/api/widgets/cors.py](orchestrator/api/widgets/cors.py)
- [orchestrator/api/widgets/rate_limit.py](orchestrator/api/widgets/rate_limit.py)
- [orchestrator/api/workspace_files.py](orchestrator/api/workspace_files.py)
- [orchestrator/api/workspace_github.py](orchestrator/api/workspace_github.py)
- [orchestrator/config.py](orchestrator/config.py)
- [orchestrator/core/database/load_seed_data.py](orchestrator/core/database/load_seed_data.py)
- [orchestrator/core/seeds/seed_personas.py](orchestrator/core/seeds/seed_personas.py)
- [orchestrator/core/seeds/seed_plugin_categories.py](orchestrator/core/seeds/seed_plugin_categories.py)
- [orchestrator/core/services/plugin_cache.py](orchestrator/core/services/plugin_cache.py)
- [orchestrator/core/workspace_client.py](orchestrator/core/workspace_client.py)
- [orchestrator/main.py](orchestrator/main.py)
- [orchestrator/modules/tools/discovery/action_registry.py](orchestrator/modules/tools/discovery/action_registry.py)
- [orchestrator/modules/tools/discovery/workspace_actions.py](orchestrator/modules/tools/discovery/workspace_actions.py)
- [scripts/ralph/prd.json](scripts/ralph/prd.json)
- [services/workspace-worker/executor.py](services/workspace-worker/executor.py)
- [services/workspace-worker/main.py](services/workspace-worker/main.py)
- [services/workspace-worker/workspace_manager.py](services/workspace-worker/workspace_manager.py)

</details>



This page provides a complete REST API reference for workspace operations, including file management, command execution, GitHub integration, and task orchestration. For conceptual overviews of these features, see [Workspace Worker Architecture](#9.1), [GitHub Integration](#9.2), [File Operations](#9.3), [Command Execution](#9.4), and [Task Management](#9.6).

---

## API Architecture

The workspace API is organized into two tiers: a **public orchestrator API** that handles authentication and request routing, and an **internal worker API** that executes operations on the persistent volume. Most operations are synchronous proxies (file I/O, command exec), while long-running tasks (git clone, multi-step workflows) use an async queue-based system.

### Two-Tier API Design

```mermaid
graph TB
    subgraph "Client Layer"
        Frontend["Frontend<br/>(Next.js)"]
        Agent["Agent Runtime<br/>(LLM Tool Calls)"]
    end
    
    subgraph "Orchestrator API<br/>(Port 8000)"
        FilesRouter["workspace_files.py<br/>/api/workspaces/{id}/files<br/>/api/workspaces/{id}/exec"]
        GithubRouter["workspace_github.py<br/>/api/workspaces/{id}/github/*"]
        TasksRouter["tasks.py<br/>/api/tasks/*"]
        WorkspaceClient["WorkspaceClient<br/>(httpx async proxy)"]
    end
    
    subgraph "Worker HTTP API<br/>(Port 8081)"
        HealthServer["Health Server<br/>/health<br/>/workspaces/{id}/*"]
        Executor["WorkspaceToolExecutor<br/>(sandboxed execution)"]
        Manager["WorkspaceManager<br/>(filesystem isolation)"]
    end
    
    subgraph "Persistent Volume"
        Volume["/workspaces/{workspace_id}/<br/>repos/, tasks/, artifacts/"]
    end
    
    Frontend -->|"Clerk JWT<br/>X-Workspace-ID"| FilesRouter
    Frontend -->|"Clerk JWT<br/>X-Workspace-ID"| GithubRouter
    Frontend -->|"Clerk JWT<br/>X-Workspace-ID"| TasksRouter
    Agent -->|"Platform Tools"| FilesRouter
    Agent -->|"Platform Tools"| TasksRouter
    
    FilesRouter -->|"HTTP Proxy"| WorkspaceClient
    GithubRouter -->|"HTTP Proxy"| WorkspaceClient
    TasksRouter -->|"Redis Queue"| TasksRouter
    
    WorkspaceClient -->|"X-Internal-Token"| HealthServer
    HealthServer --> Executor
    Executor --> Manager
    Manager --> Volume
```

**Sources:** [orchestrator/api/workspace_files.py:1-108](), [orchestrator/api/workspace_github.py:1-294](), [orchestrator/api/tasks.py:1-396](), [orchestrator/core/workspace_client.py:1-191](), [services/workspace-worker/main.py:461-700]()

---

## File Operations API

Endpoints for browsing workspace files, reading content, and executing commands. All operations are scoped to a single workspace identified by `{workspace_id}` in the URL path.

### GET /api/workspaces/{workspace_id}/files

List directory contents at a given path within the workspace.

**Request:**
```http
GET /api/workspaces/{workspace_id}/files?path=repos/my-app/src
Authorization: Bearer {clerk_jwt}
X-Workspace-ID: {workspace_id}
```

**Query Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `path` | string | No | `"."` | Relative path inside workspace (workspace-root-relative) |

**Response (200 OK):**
```json
{
  "path": "repos/my-app/src",
  "entries": [
    {
      "name": "main.py",
      "type": "file",
      "size": 4829
    },
    {
      "name": "utils",
      "type": "dir",
      "size": null
    }
  ],
  "count": 2
}
```

**Error Responses:**
- `403 Forbidden` - Workspace access denied (mismatched workspace_id)
- `404 Not Found` - Directory does not exist
- `503 Service Unavailable` - Worker unreachable

**Sources:** [orchestrator/api/workspace_files.py:30-51](), [services/workspace-worker/executor.py:272-300]()

---

### GET /api/workspaces/{workspace_id}/files/content

Read the text content of a file for code viewing or processing.

**Request:**
```http
GET /api/workspaces/{workspace_id}/files/content?path=repos/my-app/README.md
Authorization: Bearer {clerk_jwt}
X-Workspace-ID: {workspace_id}
```

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | Yes | Relative file path inside workspace |

**Response (200 OK):**
```json
{
  "content": "# My Application\n\nThis is a sample...",
  "size_bytes": 2456,
  "truncated": false,
  "path": "repos/my-app/README.md"
}
```

**Truncation:** Content is capped at 500KB by default. If `truncated: true`, only the first 500KB is returned.

**Error Responses:**
- `400 Bad Request` - Path traversal detected (e.g. `../../../etc/passwd`)
- `403 Forbidden` - Workspace access denied
- `404 Not Found` - File does not exist
- `503 Service Unavailable` - Worker unreachable

**Sources:** [orchestrator/api/workspace_files.py:54-74](), [services/workspace-worker/executor.py:230-252]()

---

### POST /api/workspaces/{workspace_id}/exec

Execute a sandboxed shell command in the workspace. Only whitelisted commands are allowed (see [Security & Sandboxing](#9.5)).

**Request:**
```http
POST /api/workspaces/{workspace_id}/exec
Authorization: Bearer {clerk_jwt}
X-Workspace-ID: {workspace_id}
Content-Type: application/json

{
  "command": "pytest tests/ --verbose",
  "cwd": "repos/my-app",
  "timeout": 120
}
```

**Request Body (ExecRequest):**
| Field | Type | Required | Default | Constraints | Description |
|-------|------|----------|---------|-------------|-------------|
| `command` | string | Yes | - | 1-4096 chars | Shell command to execute |
| `cwd` | string | No | workspace root | - | Working directory (relative to workspace) |
| `timeout` | integer | No | 120 | 1-300 | Max execution time in seconds |

**Response (200 OK):**
```json
{
  "exit_code": 0,
  "stdout": "===== test session starts =====\ncollected 15 items\n...",
  "stderr": "",
  "duration_ms": 3421,
  "truncated": false
}
```

**Output Limits:**
- `stdout`: Max 100KB
- `stderr`: Max 50KB
- If output exceeds limits, `truncated: true` and content is trimmed

**Command Whitelist:** Only approved binaries can execute: `git`, `python`, `node`, `npm`, `pytest`, `ls`, `cat`, `grep`, etc. See [ALLOWED_COMMANDS in executor.py:35-73]() for the full list.

**Blocked Patterns:** Certain dangerous patterns are blocked even if the binary is whitelisted: `sudo`, `rm -rf /`, `chmod 777`, etc. See [BLOCKED_PATTERNS in executor.py:76-95]().

**Error Responses:**
- `400 Bad Request` - Command not whitelisted or blocked pattern detected
- `403 Forbidden` - Workspace access denied
- `503 Service Unavailable` - Worker unreachable

**Sources:** [orchestrator/api/workspace_files.py:77-107](), [services/workspace-worker/executor.py:122-224]()

---

## GitHub Integration API

Endpoints for browsing GitHub repositories via Composio OAuth and cloning them into the workspace.

### GitHub Repository Object

All GitHub endpoints return repositories in this format:

```json
{
  "name": "automatos-ai",
  "full_name": "AutomatosAI/automatos-ai",
  "url": "https://github.com/AutomatosAI/automatos-ai.git",
  "description": "Multi-agent AI orchestration platform",
  "default_branch": "main",
  "private": false,
  "language": "Python",
  "updated_at": "2024-01-15T10:30:00Z"
}
```

---

### GET /api/workspaces/{workspace_id}/github/repos

List GitHub repositories accessible to the authenticated user via Composio.

**Request:**
```http
GET /api/workspaces/{workspace_id}/github/repos?page=1&per_page=30
Authorization: Bearer {clerk_jwt}
X-Workspace-ID: {workspace_id}
```

**Query Parameters:**
| Parameter | Type | Required | Default | Constraints | Description |
|-----------|------|----------|---------|-------------|-------------|
| `page` | integer | No | 1 | ≥1 | Page number for pagination |
| `per_page` | integer | No | 30 | 1-100 | Repositories per page |

**Response (200 OK):**
```json
{
  "repos": [
    {
      "name": "my-app",
      "full_name": "user/my-app",
      "url": "https://github.com/user/my-app.git",
      "description": "My application",
      "default_branch": "main",
      "private": true,
      "language": "TypeScript",
      "updated_at": "2024-01-15T10:30:00Z"
    }
  ],
  "page": 1,
  "per_page": 30
}
```

**Prerequisites:**
- Workspace must have GitHub connected via Composio (`EntityConnection` with `composio_entity_id`)
- User must have authorized GitHub OAuth via Composio app assignment

**Error Responses:**
- `403 Forbidden` - Workspace access denied
- `404 Not Found` - No Composio entity found for workspace (GitHub not connected)
- `501 Not Implemented` - Composio SDK not installed
- `502 Bad Gateway` - GitHub API error

**Sources:** [orchestrator/api/workspace_github.py:97-161](), [orchestrator/core/composio/entity_manager.py]()

---

### POST /api/workspaces/{workspace_id}/github/clone

Clone a GitHub repository into the workspace via task submission. Supports both public and private repos (private requires GitHub OAuth token from Composio).

**Request:**
```http
POST /api/workspaces/{workspace_id}/github/clone
Authorization: Bearer {clerk_jwt}
X-Workspace-ID: {workspace_id}
Content-Type: application/json

{
  "repo_url": "https://github.com/user/my-app.git",
  "branch": "develop"
}
```

**Request Body (CloneRequest):**
| Field | Type | Required | Validation | Description |
|-------|------|----------|------------|-------------|
| `repo_url` | string | Yes | HTTPS only, allowed hosts | Git clone URL (HTTPS) |
| `branch` | string | No | Safe branch pattern | Branch to clone (default: repo default) |

**URL Validation:**
- Scheme must be `https://`
- Host must be in allowlist: `github.com`, `gitlab.com`, `bitbucket.org`
- No embedded credentials (username:password in URL)

**Branch Validation:**
- Must match pattern: `^[A-Za-z0-9._/\-]+$`
- Cannot contain `..`, `@{`, or leading `-`

**Response (200 OK):**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "events_url": "/api/tasks/550e8400-e29b-41d4-a716-446655440000/events"
}
```

**Execution Flow:**
1. Orchestrator attempts to retrieve GitHub OAuth token from Composio
2. If token found, injects into HTTPS URL: `https://x-access-token:{token}@github.com/...`
3. Submits `git_clone` task to Redis queue with injected URL
4. Worker executes: `git clone --depth 1 --branch {branch} -- {url} /workspaces/{id}/repos/{name}`
5. Task transitions: `queued` → `running` → `completed`

**Token Injection Security (PRD-70):**
- OAuth token is injected server-side (never exposed to client)
- URL uses `x-access-token` authentication format
- Worker executes with `--` separator to prevent argument injection
- Branch name validated to prevent `--upload-pack` injection attacks

**Error Responses:**
- `400 Bad Request` - Invalid URL, branch, or worker backend not `queued`
- `403 Forbidden` - Workspace access denied
- `404 Not Found` - No Composio entity for workspace
- `503 Service Unavailable` - Redis enqueue failed

**Sources:** [orchestrator/api/workspace_github.py:167-293](), [services/workspace-worker/executor.py:368-419]()

---

## Task Management API

Endpoints for submitting workspace tasks, monitoring execution, and streaming real-time events.

### Task Lifecycle State Machine

```mermaid
stateDiagram-v2
    [*] --> queued: POST /api/tasks/submit
    queued --> running: Worker dequeues
    queued --> cancelled: POST /cancel
    running --> completed: Success
    running --> failed: Error
    running --> timed_out: Timeout exceeded
    running --> cancelled: POST /cancel
    completed --> [*]
    failed --> [*]
    timed_out --> [*]
    cancelled --> [*]
    
    note right of queued
        Status in Redis:
        workspace:task:{id}:status
    end note
    
    note right of completed
        Result in Redis (1h TTL):
        workspace:task:{id}:result
    end note
```

---

### POST /api/tasks/submit

Submit a workspace task with explicit steps for immediate execution. Bypasses LLM agent — runs steps directly on the worker.

**Request:**
```http
POST /api/tasks/submit
Authorization: Bearer {clerk_jwt}
X-Workspace-ID: {workspace_id}
Content-Type: application/json

{
  "steps": [
    {
      "action": "execute_command",
      "command": "pytest tests/ --cov",
      "cwd": "repos/my-app",
      "timeout": 180,
      "description": "Run tests with coverage"
    },
    {
      "action": "read_file",
      "path": "repos/my-app/coverage.xml"
    }
  ],
  "priority": "normal",
  "timeout_seconds": 300
}
```

**Request Body (SubmitTaskRequest):**
| Field | Type | Required | Default | Constraints | Description |
|-------|------|----------|---------|-------------|-------------|
| `steps` | array | Yes | - | ≥1 step | Steps to execute in order |
| `priority` | string | No | `"normal"` | low, normal, high, critical | Queue priority |
| `timeout_seconds` | integer | No | 300 | 10-3600 | Max execution time |

**Step Actions:**

| Action | Required Fields | Optional Fields | Description |
|--------|----------------|-----------------|-------------|
| `execute_command` | `command` | `cwd`, `timeout` | Run shell command |
| `git_clone` | `repo` | `branch`, `shallow` | Clone git repository |
| `git_pull` | `repo_name` | `branch` | Pull latest changes |
| `read_file` | `path` | - | Read file content |
| `write_file` | `path`, `content` | - | Write or create file |
| `list_directory` | - | `path` | List directory entries |
| `create_directory` | `path` | - | Create directory |

**Response (200 OK):**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "queue": "workspace:tasks:normal",
  "steps": 2,
  "events_url": "/api/tasks/550e8400-e29b-41d4-a716-446655440000/events"
}
```

**Queue Priority:** Tasks are enqueued to priority-specific Redis lists:
- `workspace:tasks:critical` — dequeued first
- `workspace:tasks:high`
- `workspace:tasks:normal` (default)
- `workspace:tasks:low`

**Atomic Submission:** Task row is inserted into `task_executions` table **before** Redis enqueue. If Redis enqueue fails, the DB row is marked `failed` with error message.

**Prerequisites:**
- `TASK_RUNNER_BACKEND=queued` (returns 400 if local/kubernetes backend)

**Error Responses:**
- `400 Bad Request` - Invalid steps, wrong backend, or validation failure
- `503 Service Unavailable` - Redis queue unavailable

**Sources:** [orchestrator/api/tasks.py:62-173](), [services/workspace-worker/main.py:227-357]()

---

### GET /api/tasks

List recent tasks for the current workspace with optional filtering.

**Request:**
```http
GET /api/tasks?status=running&limit=20&offset=0
Authorization: Bearer {clerk_jwt}
X-Workspace-ID: {workspace_id}
```

**Query Parameters:**
| Parameter | Type | Required | Default | Constraints | Description |
|-----------|------|----------|---------|-------------|-------------|
| `status` | string | No | - | queued, running, completed, failed, timed_out, cancelled | Filter by status |
| `task_type` | string | No | - | - | Filter by task type |
| `limit` | integer | No | 20 | 1-100 | Results per page |
| `offset` | integer | No | 0 | ≥0 | Pagination offset |

**Response (200 OK):**
```json
{
  "tasks": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "task_type": "background_job",
      "agent_id": null,
      "status": "completed",
      "priority": "normal",
      "runner_backend": "queued",
      "submitted_at": "2024-01-15T10:30:00Z",
      "started_at": "2024-01-15T10:30:02Z",
      "completed_at": "2024-01-15T10:30:15Z",
      "tokens_used": null,
      "execution_time_ms": 13421,
      "error_message": null,
      "correlation_id": null,
      "worker_id": "worker-12345-1705318200"
    }
  ],
  "total": 156,
  "limit": 20,
  "offset": 0
}
```

**Ordering:** Tasks are returned in reverse chronological order (newest first) by `submitted_at`.

**Sources:** [orchestrator/api/tasks.py:176-247]()

---

### GET /api/tasks/{task_id}

Get full task detail including result payload and execution metadata.

**Request:**
```http
GET /api/tasks/550e8400-e29b-41d4-a716-446655440000
Authorization: Bearer {clerk_jwt}
X-Workspace-ID: {workspace_id}
```

**Response (200 OK):**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "task_type": "background_job",
  "agent_id": null,
  "prompt": null,
  "configuration": {
    "steps": [
      {"action": "execute_command", "command": "pytest tests/"}
    ]
  },
  "status": "completed",
  "priority": "normal",
  "runner_backend": "queued",
  "resources_requested": null,
  "resources_used": null,
  "submitted_at": "2024-01-15T10:30:00Z",
  "started_at": "2024-01-15T10:30:02Z",
  "completed_at": "2024-01-15T10:30:15Z",
  "result": {
    "status": "completed",
    "result": {
      "steps": [
        {
          "step": 1,
          "action": "execute_command",
          "result": {
            "exit_code": 0,
            "stdout": "===== test session starts =====\n...",
            "stderr": "",
            "duration_ms": 3421
          }
        }
      ],
      "workspace_id": "ws-123",
      "repos_cached": ["my-app"]
    },
    "execution_time_ms": 13421,
    "completed_at": "2024-01-15T10:30:15Z"
  },
  "error_message": null,
  "tokens_used": null,
  "execution_time_ms": 13421,
  "parent_execution_id": null,
  "correlation_id": null,
  "worker_id": "worker-12345-1705318200",
  "workspace_path": "/workspaces/ws-123"
}
```

**Result Structure:** The `result` field contains:
- `status`: Final status (completed, failed, timed_out)
- `result.steps`: Array of step results with stdout/stderr/exit codes
- `result.repos_cached`: List of repositories cached in workspace
- `execution_time_ms`: Total execution time in milliseconds
- `error`: Error message if any step failed

**Error Responses:**
- `404 Not Found` - Task not found or workspace mismatch

**Sources:** [orchestrator/api/tasks.py:250-298]()

---

### POST /api/tasks/{task_id}/cancel

Cancel a queued or running task. Cancellation is best-effort: queued tasks are guaranteed to cancel, running tasks check between steps.

**Request:**
```http
POST /api/tasks/550e8400-e29b-41d4-a716-446655440000/cancel
Authorization: Bearer {clerk_jwt}
X-Workspace-ID: {workspace_id}
```

**Response (200 OK):**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "cancelled",
  "message": "Task cancelled successfully"
}
```

**Cancellation Behavior:**

| State | Behavior |
|-------|----------|
| `queued` | Status updated to `cancelled` in Redis. Worker skips on dequeue. |
| `running` | Status updated to `cancelled` in Redis. Worker checks between steps and aborts. |
| `completed`, `failed`, `timed_out` | Returns 400 (already terminal state) |

**Inter-Step Cancellation:** Workers check Redis status between each step execution. If status is `cancelled`, the worker publishes a `status_changed` event and stops execution.

**Error Responses:**
- `400 Bad Request` - Task already in terminal state
- `404 Not Found` - Task not found or workspace mismatch

**Sources:** [orchestrator/api/tasks.py:304-355](), [services/workspace-worker/main.py:282-285]()

---

### GET /api/tasks/{task_id}/events

Server-Sent Events (SSE) stream of real-time task execution events. Subscribe to this endpoint to receive progress updates, status changes, and error notifications.

**Request:**
```http
GET /api/tasks/550e8400-e29b-41d4-a716-446655440000/events
Authorization: Bearer {clerk_jwt}
X-Workspace-ID: {workspace_id}
Accept: text/event-stream
```

**Response (200 OK - SSE Stream):**
```
data: {"event_type": "status_changed", "task_id": "550e8400-...", "data": {"status": "running"}, "timestamp": "2024-01-15T10:30:02Z"}

data: {"event_type": "progress_update", "task_id": "550e8400-...", "data": {"step": 1, "total_steps": 3, "description": "Run tests with coverage"}, "timestamp": "2024-01-15T10:30:02Z"}

data: {"event_type": "status_changed", "task_id": "550e8400-...", "data": {"status": "completed", "execution_time_ms": 13421}, "timestamp": "2024-01-15T10:30:15Z"}
```

**Event Types:**

| Event Type | Data Fields | Description |
|------------|-------------|-------------|
| `status_changed` | `status` | Task transitioned to new status |
| `progress_update` | `step`, `total_steps`, `description` | Step execution started |
| `error` | `error`, `step` (optional) | Error occurred during step |

**Event Schema:**
```typescript
{
  event_type: "status_changed" | "progress_update" | "error",
  task_id: string,
  data: {
    // Event-specific fields
  },
  timestamp: string  // ISO 8601
}
```

**Stream Lifecycle:**
1. Client connects, receives `connected` SSE comment
2. Worker publishes events to Redis channel `workspace:task:{task_id}:events`
3. Orchestrator subscribes to channel and forwards as SSE
4. On task completion/failure, stream sends final event and closes
5. If task already complete when client connects, sends final state immediately

**Timeout:** Stream auto-closes after 10 minutes of inactivity or when task reaches terminal state.

**Error Responses:**
- `404 Not Found` - Task not found or workspace mismatch

**Sources:** [orchestrator/api/tasks.py:358-396](), [services/workspace-worker/main.py:422-434]()

---

## Request/Response Models

### Pydantic Schemas

All request bodies are validated using Pydantic models defined in the endpoint files.

**ExecRequest** ([workspace_files.py:80-83]()):
```python
class ExecRequest(BaseModel):
    command: str = Field(..., min_length=1, max_length=4096)
    cwd: Optional[str] = None
    timeout: int = Field(default=120, ge=1, le=300)
```

**CloneRequest** ([workspace_github.py:65-91]()):
```python
class CloneRequest(BaseModel):
    repo_url: str = Field(..., description="Git clone URL (HTTPS)")
    branch: Optional[str] = Field(None, description="Branch to clone")

    @field_validator("repo_url")
    @classmethod
    def validate_repo_url(cls, v: str) -> str:
        parsed = urlparse(v)
        if parsed.scheme != "https":
            raise ValueError("Only HTTPS clone URLs are allowed")
        if parsed.hostname not in {"github.com", "gitlab.com", "bitbucket.org"}:
            raise ValueError(f"Host not allowed: {parsed.hostname}")
        return v
```

**TaskStep** ([tasks.py:39-49]()):
```python
class TaskStep(BaseModel):
    action: str = Field(..., description="Action type: execute_command, git_clone, ...")
    command: Optional[str] = None
    repo: Optional[str] = None
    branch: Optional[str] = None
    path: Optional[str] = None
    content: Optional[str] = None
    cwd: Optional[str] = None
    timeout: Optional[int] = None
    description: Optional[str] = None
```

**SubmitTaskRequest** ([tasks.py:52-56]()):
```python
class SubmitTaskRequest(BaseModel):
    steps: List[TaskStep] = Field(..., min_length=1)
    priority: str = Field("normal", description="Priority: low, normal, high, critical")
    timeout_seconds: int = Field(300, ge=10, le=3600)
```

**Sources:** [orchestrator/api/workspace_files.py:80-83](), [orchestrator/api/workspace_github.py:65-91](), [orchestrator/api/tasks.py:39-56]()

---

## Error Handling

All endpoints follow a consistent error response format.

### HTTP Status Codes

| Code | Meaning | Common Causes |
|------|---------|---------------|
| 200 | OK | Successful operation |
| 400 | Bad Request | Invalid parameters, validation failure, wrong backend |
| 403 | Forbidden | Workspace access denied (X-Workspace-ID mismatch) |
| 404 | Not Found | Resource not found (task, file, directory, Composio entity) |
| 501 | Not Implemented | Composio SDK not installed |
| 502 | Bad Gateway | External API error (GitHub, Composio) |
| 503 | Service Unavailable | Worker unreachable, Redis queue unavailable |

### Error Response Format

```json
{
  "detail": "Human-readable error message"
}
```

**FastAPI HTTPException:** All errors are raised as `HTTPException` with appropriate status code and detail message.

**Worker Connection Errors:** When the workspace worker is unreachable, the `WorkspaceClient` returns:
```json
{
  "success": false,
  "error": "Workspace worker unreachable (read_file): ConnectError(...)",
  "status_code": 503
}
```

**Sources:** [orchestrator/core/workspace_client.py:47-53](), [orchestrator/api/workspace_files.py:42-49]()

---

## Authentication & Authorization

### Clerk JWT Authentication

All workspace endpoints require Clerk JWT authentication via the `Authorization: Bearer {token}` header.

**Request Context Extraction:**
```python
from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext

@router.get("/files")
async def list_files(
    workspace_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    if str(ctx.workspace_id) != workspace_id:
        raise HTTPException(status_code=403, detail="Workspace access denied")
```

**Workspace Isolation:** Every endpoint validates that `ctx.workspace_id` matches the `{workspace_id}` path parameter before executing operations. This ensures users can only access their own workspaces.

**Internal Worker Authentication:** Requests from orchestrator to worker use `X-Internal-Token` header (not Clerk JWT). This token is configured via `WORKER_INTERNAL_TOKEN` environment variable.

**Sources:** [orchestrator/api/workspace_files.py:19-42](), [orchestrator/core/workspace_client.py:33-38]()

---

## Internal Worker HTTP API

While most operations proxy through the orchestrator API, the workspace worker also exposes a direct HTTP API on port 8081. This is **internal-only** and should not be exposed to public internet.

### Worker Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check with worker stats |
| GET | `/workspaces/{id}/files` | List directory (direct, no auth check) |
| GET | `/workspaces/{id}/files/content` | Read file content (direct) |
| POST | `/workspaces/{id}/exec` | Execute command (direct) |
| GET | `/workspaces/{id}/grep` | Search files with regex pattern |

**Authentication:** All worker endpoints check for `X-Internal-Token` header. If configured, unauthorized requests return 403.

**Health Endpoint Response:**
```json
{
  "status": "healthy",
  "worker_id": "worker-12345-1705318200",
  "active_tasks": 2,
  "concurrency_limit": 3
}
```

**Grep Endpoint:** Provides regex search across workspace files:
```http
GET /workspaces/{id}/grep?pattern=def%20handle_error&include=*.py&max_results=50
X-Internal-Token: {token}
```

Response:
```json
{
  "matches": [
    {
      "file": "repos/my-app/src/main.py",
      "line": 42,
      "content": "def handle_error(exc: Exception) -> None:",
      "line_number": 42
    }
  ],
  "total_matches": 3,
  "truncated": false
}
```

**Sources:** [services/workspace-worker/main.py:461-700]()

---

## WorkspaceClient Proxy

The orchestrator uses `WorkspaceClient` to proxy requests to the worker HTTP API. This provides a typed, async interface for backend code.

### Usage Example

```python
from core.workspace_client import WorkspaceClient

client = WorkspaceClient(workspace_id)

# Read file
result = await client.read_file("repos/my-app/README.md")
if result.get("success") is False:
    raise HTTPException(status_code=503, detail=result["error"])

# Execute command
result = await client.exec_command(
    command="pytest tests/",
    cwd="repos/my-app",
    timeout=180
)

# Search files
result = await client.grep(
    pattern="TODO",
    include="*.py",
    max_results=20
)
```

### Available Methods

| Method | Parameters | Returns |
|--------|------------|---------|
| `read_file(path)` | `path: str` | `Dict[str, Any]` |
| `write_file(path, content)` | `path: str, content: str` | `Dict[str, Any]` |
| `list_dir(path)` | `path: str` | `Dict[str, Any]` |
| `exec_command(command, cwd, timeout)` | `command: str, cwd: Optional[str], timeout: int` | `Dict[str, Any]` |
| `grep(pattern, path, include, max_results)` | `pattern: str, ...` | `Dict[str, Any]` |

**Connection Pooling:** `WorkspaceClient` uses a singleton `httpx.AsyncClient` shared across all instances. Connection is reused for performance.

**Timeout Configuration:**
- Connect: 10 seconds
- Read: 130 seconds (allows for long command execution)
- Write: 30 seconds
- Pool: 10 seconds

**Error Handling:** All methods return `{"success": False, "error": "..."}` on connection/timeout errors. Callers should check for this pattern.

**Sources:** [orchestrator/core/workspace_client.py:1-191]()

---

## Rate Limiting (Widget API)

Widget SDK endpoints (`/api/widgets/*`) have per-API-key rate limiting enforced by `WidgetRateLimitMiddleware`.

**Limits:**
- Public keys (`ak_pub_*`): 30 requests/minute
- Server keys (`ak_srv_*`): 1000 requests/minute
- Unauthenticated (IP-based): 30 requests/minute

**Rate Limit Headers:**
```http
X-RateLimit-Limit: 30
X-RateLimit-Remaining: 25
X-RateLimit-Reset: 45
```

**429 Response:**
```http
HTTP/1.1 429 Too Many Requests
Retry-After: 45
Content-Type: application/json

{
  "detail": "Rate limit exceeded",
  "retry_after": 45
}
```

**Sliding Window:** Uses in-memory sliding window counter with 60-second window size. Requests older than 60 seconds are dropped from the window.

**Sources:** [orchestrator/api/widgets/rate_limit.py:1-166]()

---

## Request Flow Examples

### File Read Flow

```mermaid
sequenceDiagram
    participant Client as "Frontend"
    participant Router as "workspace_files.py"
    participant WSClient as "WorkspaceClient"
    participant Worker as "Worker HTTP API"
    participant Executor as "WorkspaceToolExecutor"
    participant Manager as "WorkspaceManager"
    
    Client->>Router: GET /api/workspaces/{id}/files/content?path=main.py
    Note over Client,Router: Authorization: Bearer {jwt}<br/>X-Workspace-ID: {id}
    
    Router->>Router: get_request_context_hybrid()
    Router->>Router: Validate ctx.workspace_id == path.workspace_id
    
    Router->>WSClient: read_file("main.py")
    WSClient->>Worker: GET /workspaces/{id}/files/content?path=main.py
    Note over WSClient,Worker: X-Internal-Token: {token}
    
    Worker->>Executor: read_file("main.py")
    Executor->>Manager: resolve_safe_path("main.py")
    Manager-->>Executor: /workspaces/{id}/main.py
    
    Executor->>Manager: Check path is inside workspace
    Manager-->>Executor: ✓ Valid
    
    Executor->>Executor: Read file, decode UTF-8
    Executor-->>Worker: {content, size, truncated, path}
    Worker-->>WSClient: JSON response
    WSClient-->>Router: JSON response
    Router-->>Client: JSON response
```

**Sources:** [orchestrator/api/workspace_files.py:54-74](), [orchestrator/core/workspace_client.py:68-78](), [services/workspace-worker/executor.py:230-252]()

---

### GitHub Clone Flow

```mermaid
sequenceDiagram
    participant Client as "Frontend"
    participant Router as "workspace_github.py"
    participant Composio as "EntityManager"
    participant TaskRunner as "TaskRunner (queued)"
    participant Redis as "Redis Queue"
    participant Worker as "Workspace Worker"
    participant Git as "Git Clone"
    
    Client->>Router: POST /api/workspaces/{id}/github/clone
    Note over Client,Router: {repo_url, branch}
    
    Router->>Router: Validate repo_url (HTTPS, allowed host)
    Router->>Router: Validate branch pattern
    
    Router->>Composio: get_entity_by_workspace(workspace_id)
    Composio-->>Router: {composio_entity_id}
    
    Router->>Composio: get_app_access_token(entity_id, "GITHUB")
    Composio-->>Router: {oauth_token}
    
    Router->>Router: Inject token into URL
    Note over Router: https://x-access-token:{token}@github.com/...
    
    Router->>Router: Build task payload
    Note over Router: {task_id, workspace_id, steps: [git_clone]}
    
    Router->>Router: Insert task_executions row (status: queued)
    Router->>Redis: LPUSH workspace:tasks:normal
    Redis-->>Router: OK
    
    Router-->>Client: {task_id, status: queued, events_url}
    
    Note over Worker: Worker polling loop
    Worker->>Redis: RPOP workspace:tasks:*
    Redis-->>Worker: {task payload}
    
    Worker->>Worker: Update status: running
    Worker->>Git: git clone --depth 1 --branch {branch} -- {url} /workspaces/{id}/repos/{name}
    Git-->>Worker: Exit code 0
    
    Worker->>Worker: Update status: completed
    Worker->>Redis: PUBLISH workspace:task:{id}:events
    Worker->>Redis: SET workspace:task:{id}:result
```

**Sources:** [orchestrator/api/workspace_github.py:167-293](), [services/workspace-worker/main.py:227-357](), [services/workspace-worker/executor.py:368-419]()

---

## See Also

- [Workspace Worker Architecture](#9.1) - Internal architecture of the workspace worker service
- [GitHub Integration](#9.2) - Conceptual overview of GitHub OAuth and cloning
- [File Operations](#9.3) - File system security and operations
- [Command Execution](#9.4) - Command whitelisting and sandboxing
- [Security & Sandboxing](#9.5) - Five-layer security model
- [Task Management](#9.6) - Task lifecycle and queue system

---