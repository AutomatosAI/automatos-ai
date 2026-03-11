# GitHub Integration

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

The GitHub Integration subsystem enables agents and users to browse and clone GitHub repositories directly into workspace directories via Composio's OAuth authentication. This provides agents with the ability to read source code, execute tests, and make modifications to repositories within the sandboxed workspace environment.

This document covers repository listing, authenticated cloning, and security validation. For general workspace file operations once a repository is cloned, see [File Operations](#9.3). For executing commands within cloned repositories, see [Command Execution](#9.4).

---

## Architecture Overview

GitHub integration follows a three-tier architecture: the frontend initiates requests, the orchestrator validates and queues them, and the workspace worker executes the actual git operations with injected OAuth credentials.

### System Flow Diagram

```mermaid
flowchart TB
    subgraph "Frontend Layer"
        RepoSelector["RepoSelector.tsx<br/>GitHub Repo Browser"]
    end
    
    subgraph "Orchestrator Layer"
        GitHubAPI["workspace_github.py<br/>GET /repos<br/>POST /clone"]
        EntityManager["EntityManager<br/>Composio Entity Lookup"]
        ComposioClient["get_composio_client()<br/>OAuth Token Retrieval"]
        TaskRunner["TaskRunner<br/>Redis Queue Submission"]
    end
    
    subgraph "Composio Service"
        ComposioAPI["Composio API<br/>GITHUB_LIST_REPOSITORIES<br/>get_app_access_token"]
    end
    
    subgraph "Data Layer"
        Redis["Redis<br/>workspace:tasks:normal"]
        PostgreSQL["PostgreSQL<br/>task_executions table"]
    end
    
    subgraph "Worker Layer"
        WorkerMain["main.py<br/>WorkspaceWorker._execute_task"]
        Executor["executor.py<br/>WorkspaceToolExecutor._git_clone"]
        WSManager["workspace_manager.py<br/>WorkspaceManager.get_repo_path"]
    end
    
    RepoSelector -->|"1. List repos"| GitHubAPI
    RepoSelector -->|"2. Clone request"| GitHubAPI
    
    GitHubAPI -->|"Resolve entity_id"| EntityManager
    GitHubAPI -->|"Fetch GitHub token"| ComposioClient
    GitHubAPI -->|"Submit clone task"| TaskRunner
    
    ComposioClient --> ComposioAPI
    EntityManager --> PostgreSQL
    
    TaskRunner -->|"Insert row"| PostgreSQL
    TaskRunner -->|"Enqueue payload"| Redis
    
    WorkerMain -->|"Dequeue task"| Redis
    WorkerMain -->|"Execute steps"| Executor
    
    Executor -->|"Resolve repo path"| WSManager
    Executor -->|"git clone --branch -- url"| Executor
```

**Sources**: [orchestrator/api/workspace_github.py:1-294](), [services/workspace-worker/main.py:227-358](), [services/workspace-worker/executor.py:368-419]()

---

## Repository Browsing

Repository listing is proxied through Composio's GitHub integration to retrieve the authenticated user's accessible repositories with OAuth scope enforcement.

### Composio Entity Resolution

Before making any GitHub API calls, the system resolves the workspace's Composio `entity_id` via the `EntityManager`:

```mermaid
sequenceDiagram
    participant API as workspace_github.py
    participant EM as EntityManager
    participant DB as PostgreSQL
    participant Composio as Composio Client
    
    API->>EM: get_entity_by_workspace(workspace_id)
    EM->>DB: SELECT composio_entity_id FROM entity_connections
    DB-->>EM: entity_id or None
    EM-->>API: entity dict or 404
    
    API->>Composio: execute_action(GITHUB_LIST_REPOSITORIES, entity_id)
    Composio-->>API: repos list
    API-->>Frontend: JSON response with repos
```

The `_get_entity_id` helper function [orchestrator/api/workspace_github.py:47-58]() throws HTTP 404 if the workspace has not connected GitHub via Composio, ensuring early failure with a clear error message:

```
"No Composio entity found for this workspace. Connect GitHub first."
```

**Sources**: [orchestrator/api/workspace_github.py:47-58](), [orchestrator/core/composio/entity_manager.py]()

### List Repositories Endpoint

**Endpoint**: `GET /api/workspaces/{workspace_id}/github/repos`

**Query Parameters**:
- `page` (default: 1, min: 1)
- `per_page` (default: 30, min: 1, max: 100)

**Response Structure**:

| Field | Type | Description |
|-------|------|-------------|
| `repos` | array | List of repository objects |
| `page` | int | Current page number |
| `per_page` | int | Items per page |

**Repository Object Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Repository short name |
| `full_name` | string | Owner/repo format |
| `url` | string | Clone URL (HTTPS) |
| `description` | string | Repository description |
| `default_branch` | string | Default branch (e.g., "main") |
| `private` | boolean | Private repository flag |
| `language` | string | Primary language |
| `updated_at` | string | Last update timestamp (ISO 8601) |

The implementation [orchestrator/api/workspace_github.py:97-161]() handles Composio's nested response structure, extracting the repository list from `result.data.data.repositories` and normalizing it into a flat array.

**Sources**: [orchestrator/api/workspace_github.py:97-161]()

---

## Repository Cloning

Repository cloning is implemented as an asynchronous task executed by the workspace worker with OAuth token injection for private repository access.

### Clone Request Validation

The `CloneRequest` model [orchestrator/api/workspace_github.py:65-92]() enforces strict validation rules via Pydantic validators:

#### URL Validation

**Allowed Hosts**: `github.com`, `gitlab.com`, `bitbucket.org`

Validation rules enforced by `validate_repo_url` [orchestrator/api/workspace_github.py:69-79]():

```python
# Only HTTPS allowed (no git://, ssh://)
if parsed.scheme != "https":
    raise ValueError("Only HTTPS clone URLs are allowed")

# Restricted to allowed hosts
if parsed.hostname not in _ALLOWED_CLONE_HOSTS:
    raise ValueError(f"Host not allowed: {parsed.hostname}")

# No embedded credentials (security risk)
if parsed.username or parsed.password:
    raise ValueError("Clone URL must not contain embedded credentials")
```

#### Branch Name Validation

The `validate_branch` method [orchestrator/api/workspace_github.py:81-91]() prevents injection attacks via branch names:

```python
# Dangerous patterns blocked:
_BRANCH_RE = re.compile(r"^[A-Za-z0-9._/\-]+$")

if ".." in branch or "@{" in branch or not _BRANCH_RE.match(branch):
    raise ValueError("Invalid branch name")
```

This prevents attacks like `--upload-pack=/tmp/malicious` being passed to git commands.

**Sources**: [orchestrator/api/workspace_github.py:65-92]()

### OAuth Token Injection

For private repositories, the clone endpoint attempts to retrieve the user's GitHub OAuth token from Composio and inject it into the HTTPS clone URL:

```mermaid
flowchart TD
    Start["POST /clone request"] --> GetToken["client.get_app_access_token(entity_id, 'GITHUB')"]
    GetToken --> HasToken{"Token retrieved?"}
    
    HasToken -->|Yes| InjectToken["Inject into URL:<br/>https://x-access-token:TOKEN@github.com/..."]
    HasToken -->|No| WarnFallback["Log warning:<br/>Public clone only"]
    
    InjectToken --> SubmitTask["Submit git_clone task"]
    WarnFallback --> SubmitTask
    
    SubmitTask --> QueueToWorker["Enqueue to Redis"]
```

The token injection code [orchestrator/api/workspace_github.py:193-211]() modifies the clone URL before task submission:

```python
token = await asyncio.to_thread(
    client.get_app_access_token, entity_id, "GITHUB"
)
if token:
    if clone_url.startswith("https://github.com"):
        clone_url = clone_url.replace(
            "https://github.com",
            f"https://x-access-token:{token}@github.com",
        )
        logger.info("Injected GitHub token for authenticated clone")
```

If token retrieval fails (e.g., OAuth not connected), the system logs a warning and proceeds with unauthenticated cloning, which works for public repositories.

**Sources**: [orchestrator/api/workspace_github.py:193-211]()

### Clone Endpoint Implementation

**Endpoint**: `POST /api/workspaces/{workspace_id}/github/clone`

**Request Body**:

```json
{
  "repo_url": "https://github.com/owner/repo.git",
  "branch": "develop"
}
```

**Response**:

```json
{
  "task_id": "a7f3c...uuid",
  "status": "queued",
  "events_url": "/api/tasks/a7f3c.../events"
}
```

The clone operation is **not executed synchronously**. Instead, it creates a background task and returns immediately, allowing the frontend to poll or stream events via SSE.

**Sources**: [orchestrator/api/workspace_github.py:167-293]()

---

## Task Submission Flow

Clone requests follow an atomic two-phase submission pattern to prevent race conditions between the database and Redis queue.

### Atomic Task Submission Sequence

```mermaid
sequenceDiagram
    participant API as workspace_github.py
    participant DB as PostgreSQL
    participant Redis as Redis
    participant Worker as WorkspaceWorker
    
    Note over API: Generate task_id (UUID)
    API->>DB: INSERT INTO task_executions<br/>(id, status='queued', config)
    DB-->>API: Row inserted
    API->>DB: COMMIT
    
    API->>Redis: HSET workspace:task:{id}:status
    API->>Redis: SADD workspace:ws:{ws_id}:active_tasks
    API->>Redis: LPUSH workspace:tasks:normal
    
    alt Redis enqueue fails
        API->>DB: UPDATE status='failed'
        API->>DB: COMMIT
        API-->>Frontend: 503 Service Unavailable
    end
    
    Worker->>Redis: RPOP workspace:tasks:normal
    Redis-->>Worker: Task payload
    Worker->>Worker: Execute git clone
```

**Critical Ordering**: The database row is inserted **before** Redis enqueue [orchestrator/api/workspace_github.py:238-256](). This prevents a race condition where a worker picks up a task that has no corresponding database record.

If Redis enqueue fails after the database commit, the endpoint marks the task as failed in the database and returns HTTP 503 [orchestrator/api/workspace_github.py:275-282]():

```python
except Exception as enqueue_err:
    logger.error("Redis enqueue failed for task %s: %s", task_id[:8], enqueue_err)
    db.execute(text(
        "UPDATE task_executions SET status = 'failed', error_message = :err WHERE id = :id"
    ), {"id": task_id, "err": f"Enqueue failed: {enqueue_err}"})
    db.commit()
    raise HTTPException(status_code=503, detail="Failed to enqueue task to worker")
```

**Sources**: [orchestrator/api/workspace_github.py:236-283]()

### Task Payload Structure

The clone task payload [orchestrator/api/workspace_github.py:216-234]() follows the standard workspace task format:

```json
{
  "task_id": "uuid",
  "task_type": "background_job",
  "workspace_id": "uuid",
  "agent_id": null,
  "priority": "normal",
  "timeout_seconds": 300,
  "steps": [
    {
      "action": "git_clone",
      "repo": "https://x-access-token:TOKEN@github.com/owner/repo.git",
      "branch": "develop",
      "description": "Clone https://github.com/owner/repo.git"
    }
  ],
  "created_at": "2024-01-15T10:30:00Z"
}
```

The OAuth token (if retrieved) is embedded directly in the `repo` URL, not in a separate credentials field. This allows the worker to clone the repository without additional Composio API calls.

**Sources**: [orchestrator/api/workspace_github.py:216-234]()

---

## Worker-Side Execution

The workspace worker executes the `git_clone` action with additional security validations and intelligent caching.

### Git Clone Implementation

The `_git_clone` method [services/workspace-worker/executor.py:368-419]() implements the following flow:

```mermaid
flowchart TD
    Start["git_clone step"] --> ValidateBranch{"Branch name<br/>valid?"}
    ValidateBranch -->|Invalid| ErrorReturn["Return error:<br/>Invalid branch name"]
    ValidateBranch -->|Valid| ExtractName["Extract repo name from URL"]
    
    ExtractName --> CheckCache{"Repo already<br/>cached?"}
    CheckCache -->|Yes| GitPull["Execute git pull<br/>(update cache)"]
    CheckCache -->|No| BuildCmd["Build clone command"]
    
    BuildCmd --> AddDepth["Add --depth 1<br/>(shallow clone)"]
    AddDepth --> AddBranch["Add --branch {branch}<br/>(if specified)"]
    AddBranch --> AddSeparator["Add -- separator<br/>(PRD-70 security)"]
    AddSeparator --> ExecuteClone["Execute git clone"]
    
    ExecuteClone --> UpdateMeta["Update .workspace_meta.json<br/>(repos_cached list)"]
    GitPull --> Return["Return result"]
    UpdateMeta --> Return
    ErrorReturn --> Return
```

**Sources**: [services/workspace-worker/executor.py:368-419]()

### PRD-70 Security Fix: Argument Injection Prevention

The clone implementation includes a critical security fix documented in PRD-70 FIX-01. The vulnerability was that unvalidated branch names could be used to inject git arguments like `--upload-pack=/tmp/malicious`.

**Defense-in-Depth Strategy**:

1. **Orchestrator-side validation** [orchestrator/api/workspace_github.py:81-91](): Reject invalid branch names before task submission
2. **Worker-side validation** [services/workspace-worker/executor.py:380-386](): Redundant check in case orchestrator validation is bypassed
3. **Git separator `--`** [services/workspace-worker/executor.py:406](): Marks end of options, treating all subsequent arguments as positional

```python
# PRD-70 FIX-01: Validate branch to prevent --upload-pack injection
if branch:
    if branch.startswith("-") or not self._BRANCH_RE.match(branch):
        return {
            "exit_code": 1,
            "stdout": "",
            "stderr": f"Invalid branch name: {branch}",
        }

# Build clone command with -- separator (PRD-70 FIX-01)
cmd_parts = ["git", "clone"]
if shallow:
    cmd_parts.extend(["--depth", "1"])
if branch:
    cmd_parts.extend(["--branch", branch])
cmd_parts.append("--")  # End of options — positional args only after this
cmd_parts.extend([repo_url, str(repo_path)])
```

**Sources**: [services/workspace-worker/executor.py:368-419]()

### Repository Caching Strategy

Cloned repositories are stored persistently in the workspace's `repos/` directory [services/workspace-worker/workspace_manager.py:259-262]():

```
/workspaces/{workspace_id}/
├── repos/
│   ├── my-app/          ← Cached clone
│   └── other-repo/      ← Another cached clone
├── tasks/               ← Ephemeral task dirs
└── artifacts/           ← Build outputs
```

If a repository with the same name already exists, the worker automatically switches from `git clone` to `git pull` [services/workspace-worker/executor.py:395-398](), updating the cached repository instead of re-cloning. This optimization reduces bandwidth and speeds up repeated executions.

**Sources**: [services/workspace-worker/executor.py:395-398](), [services/workspace-worker/workspace_manager.py:259-274]()

---

## Frontend Integration

The `RepoSelector` component provides a modal dialog for browsing and cloning GitHub repositories.

### Component Architecture

```mermaid
graph TB
    subgraph "RepoSelector Component"
        State["Component State:<br/>repos, isLoading, error,<br/>filter, cloningRepo"]
        ListAPI["apiClient.listGithubRepos()"]
        CloneAPI["apiClient.cloneGithubRepo()"]
        UI["UI Layer:<br/>Search input, repo list,<br/>clone buttons"]
    end
    
    subgraph "Backend"
        ListEndpoint["GET /github/repos"]
        CloneEndpoint["POST /github/clone"]
    end
    
    State --> ListAPI
    State --> CloneAPI
    State --> UI
    
    ListAPI --> ListEndpoint
    CloneAPI --> CloneEndpoint
    
    CloneEndpoint -->|"task_id"| Callback["onCloneStarted callback"]
    Callback -->|"Monitor task"| TaskSSE["SSE /tasks/{id}/events"]
```

**Sources**: [frontend/components/widgets/CodingCanvasWidget/RepoSelector.tsx:1-182]()

### Repository Display

Repositories are displayed with the following visual indicators [frontend/components/widgets/CodingCanvasWidget/RepoSelector.tsx:144-176]():

- **Lock icon** (`Lock`): Private repository
- **Globe icon** (`Globe`): Public repository
- **Language badge**: Primary programming language
- **Loader spinner**: Indicates cloning in progress

The filter input [frontend/components/widgets/CodingCanvasWidget/RepoSelector.tsx:90-94]() searches both `full_name` and `description` fields case-insensitively:

```typescript
const filtered = repos.filter(
  (r) =>
    r.full_name.toLowerCase().includes(filter.toLowerCase()) ||
    (r.description || '').toLowerCase().includes(filter.toLowerCase())
)
```

**Sources**: [frontend/components/widgets/CodingCanvasWidget/RepoSelector.tsx:90-176]()

### Clone Callback Pattern

When a clone is initiated, the component calls the `onCloneStarted` callback with the task ID [frontend/components/widgets/CodingCanvasWidget/RepoSelector.tsx:69-88]():

```typescript
const result = (await apiClient.cloneGithubRepo(
  workspaceId,
  repo.url,
  repo.default_branch
)) as { task_id: string }

onCloneStarted?.(result.task_id)
onOpenChange(false)  // Close dialog
```

The parent component (typically `CodingCanvasWidget`) is responsible for:
1. Subscribing to task events via SSE
2. Updating UI with clone progress
3. Refreshing the file browser when clone completes

**Sources**: [frontend/components/widgets/CodingCanvasWidget/RepoSelector.tsx:69-88]()

---

## Security Considerations

The GitHub integration implements multiple layers of security validation:

### URL Security

| Validation | Location | Purpose |
|------------|----------|---------|
| HTTPS-only | `CloneRequest.validate_repo_url` | Prevents protocol downgrade attacks |
| Host allowlist | `_ALLOWED_CLONE_HOSTS` | Restricts to known safe hosts |
| No embedded credentials | URL parser check | Prevents credential leakage in logs |

**Sources**: [orchestrator/api/workspace_github.py:36-79]()

### Branch Name Security

| Validation | Pattern | Blocked Examples |
|------------|---------|------------------|
| Alphanumeric + `./_-` only | `[A-Za-z0-9._/\-]+` | `--upload-pack`, `../escape` |
| No path traversal | `..` detection | `../../etc/passwd` |
| No git reflog syntax | `@{` detection | `@{-1}`, `@{push}` |

**Sources**: [orchestrator/api/workspace_github.py:39-91](), [services/workspace-worker/executor.py:366-386]()

### Worker Environment Sandboxing

The workspace worker runs with a stripped environment [services/workspace-worker/executor.py:506-536]():

- **Restricted PATH**: Only standard system paths
- **Isolated HOME**: Set to workspace root
- **Custom SSH config**: Points to workspace `.ssh/` directory
- **No host environment leakage**: Parent process env vars are not inherited

Git operations use the sandboxed environment, preventing access to host credentials or configurations.

**Sources**: [services/workspace-worker/executor.py:506-536]()

---

## Error Handling

### Orchestrator Error Cases

| Error Condition | HTTP Status | Error Message |
|-----------------|-------------|---------------|
| Workspace access denied | 403 | "Workspace access denied" |
| No Composio entity | 404 | "No Composio entity found for this workspace. Connect GitHub first." |
| Composio SDK not installed | 501 | "Composio SDK not installed" |
| GitHub API error | 502 | "GitHub API error: {error_msg}" |
| Redis enqueue failure | 503 | "Failed to enqueue task to worker" |

**Sources**: [orchestrator/api/workspace_github.py:97-293]()

### Worker Error Cases

| Error Condition | Behavior | Exit Code |
|-----------------|----------|-----------|
| Invalid branch name | Return error immediately | 1 |
| Git clone fails | Return stderr output | Non-zero |
| Timeout (300s) | Kill process, return timeout error | -1 |

Worker errors are stored in both Redis (`workspace:task:{id}:result`) and PostgreSQL (`task_executions.error_message`) for reliable retrieval.

**Sources**: [services/workspace-worker/main.py:342-353](), [services/workspace-worker/executor.py:368-419]()

---

## API Reference

### List GitHub Repositories

**Endpoint**: `GET /api/workspaces/{workspace_id}/github/repos`

**Authentication**: Requires valid workspace context (JWT + `X-Workspace-ID` header)

**Query Parameters**:

| Parameter | Type | Default | Constraints | Description |
|-----------|------|---------|-------------|-------------|
| `page` | integer | 1 | ≥ 1 | Page number for pagination |
| `per_page` | integer | 30 | 1-100 | Items per page |

**Success Response** (200 OK):

```json
{
  "repos": [
    {
      "name": "repo-name",
      "full_name": "owner/repo-name",
      "url": "https://github.com/owner/repo-name.git",
      "description": "Repository description",
      "default_branch": "main",
      "private": false,
      "language": "Python",
      "updated_at": "2024-01-15T10:30:00Z"
    }
  ],
  "page": 1,
  "per_page": 30
}
```

**Error Responses**:

- **403 Forbidden**: Workspace access denied
- **404 Not Found**: No Composio entity configured (GitHub not connected)
- **501 Not Implemented**: Composio SDK not available
- **502 Bad Gateway**: GitHub API returned an error

**Sources**: [orchestrator/api/workspace_github.py:97-161]()

### Clone GitHub Repository

**Endpoint**: `POST /api/workspaces/{workspace_id}/github/clone`

**Authentication**: Requires valid workspace context + queued task runner backend

**Request Body**:

```json
{
  "repo_url": "https://github.com/owner/repo.git",
  "branch": "develop"
}
```

**Request Body Fields**:

| Field | Type | Required | Validation |
|-------|------|----------|------------|
| `repo_url` | string | Yes | HTTPS only, allowed hosts, no embedded credentials |
| `branch` | string | No | Alphanumeric + `./_-`, no `..` or `@{` |

**Success Response** (200 OK):

```json
{
  "task_id": "a7f3c2d1-8b4e-4f5a-9c3e-1d2e3f4a5b6c",
  "status": "queued",
  "events_url": "/api/tasks/a7f3c2d1-8b4e-4f5a-9c3e-1d2e3f4a5b6c/events"
}
```

The response includes a `task_id` that can be used to:
- Poll task status: `GET /api/tasks/{task_id}`
- Stream live updates: `GET /api/tasks/{task_id}/events` (SSE)
- Cancel task: `POST /api/tasks/{task_id}/cancel`

**Error Responses**:

- **400 Bad Request**: Invalid URL or branch name, or wrong task runner backend
- **403 Forbidden**: Workspace access denied
- **404 Not Found**: No Composio entity configured
- **503 Service Unavailable**: Redis queue unavailable

**Sources**: [orchestrator/api/workspace_github.py:167-293]()

### Task Event Stream Format

Once a clone task is queued, subscribe to its event stream:

**Endpoint**: `GET /api/tasks/{task_id}/events`

**Response Type**: `text/event-stream`

**Event Types**:

```
event: status_changed
data: {"status": "running"}

event: progress_update
data: {"step": 1, "total_steps": 1, "description": "Clone https://github.com/..."}

event: status_changed
data: {"status": "completed", "execution_time_ms": 3245}
```

For complete task management API documentation, see [Workspace API Reference](#9.7).

**Sources**: [orchestrator/api/tasks.py:349-403]()

---