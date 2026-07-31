# Security & Sandboxing

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [frontend/components/widgets/CodingCanvasWidget/RepoSelector.tsx](frontend/components/widgets/CodingCanvasWidget/RepoSelector.tsx)
- [frontend/components/widgets/TerminalWidget/InteractiveTerminal.tsx](frontend/components/widgets/TerminalWidget/InteractiveTerminal.tsx)
- [frontend/components/widgets/TerminalWidget/index.tsx](frontend/components/widgets/TerminalWidget/index.tsx)
- [orchestrator/api/tasks.py](orchestrator/api/tasks.py)
- [orchestrator/api/widgets/cors.py](orchestrator/api/widgets/cors.py)
- [orchestrator/api/widgets/rate_limit.py](orchestrator/api/widgets/rate_limit.py)
- [orchestrator/api/workspace_exec.py](orchestrator/api/workspace_exec.py)
- [orchestrator/api/workspace_github.py](orchestrator/api/workspace_github.py)
- [orchestrator/core/workspace_client.py](orchestrator/core/workspace_client.py)
- [orchestrator/modules/tools/discovery/workspace_actions.py](orchestrator/modules/tools/discovery/workspace_actions.py)
- [orchestrator/modules/tools/execution/exec_workspace.py](orchestrator/modules/tools/execution/exec_workspace.py)
- [services/workspace-worker/Dockerfile](services/workspace-worker/Dockerfile)
- [services/workspace-worker/entrypoint.sh](services/workspace-worker/entrypoint.sh)
- [services/workspace-worker/executor.py](services/workspace-worker/executor.py)
- [services/workspace-worker/main.py](services/workspace-worker/main.py)
- [services/workspace-worker/requirements.txt](services/workspace-worker/requirements.txt)
- [services/workspace-worker/workspace_manager.py](services/workspace-worker/workspace_manager.py)

</details>



Automatos AI implements a multi-layered security architecture designed to isolate agent execution, protect the host filesystem, and govern external access via widgets. The system relies on physical workspace boundaries, strict command whitelisting, and per-workspace rate limiting to ensure that autonomous agents operate within safe constraints.

## Workspace Isolation & Path Traversal Prevention

The core of the sandboxing strategy is the `WorkspaceWorker`, which manages isolated directories for each workspace on a persistent volume [services/workspace-worker/main.py:6-9](). Security is enforced at the `WorkspaceManager` and `WorkspaceToolExecutor` levels to prevent agents from accessing data outside their designated environment.

### Path Containment
All file operations and command executions are passed through a validation layer. The `WorkspaceToolExecutor` uses `WorkspaceManager.resolve_safe_path` to ensure that any path provided by an agent (or user) is resolved relative to the workspace root and does not use `..` or symlinks to escape the boundary [services/workspace-worker/executor.py:115-116]().

### Command Whitelisting
Agents do not have unrestricted shell access. The system maintains a strict `ALLOWED_COMMANDS` set, including essential tools for development:
*   **Interpreters:** `sh`, `bash`, `python3`, `node` [services/workspace-worker/executor.py:35-49]().
*   **Package Managers:** `pip`, `uv`, `npm`, `pnpm` [services/workspace-worker/executor.py:44-49]().
*   **Dev Tools:** `git`, `pytest`, `ruff`, `tsc`, `jq`, `curl` [services/workspace-worker/executor.py:41-58]().
*   **System Utils:** `ls`, `grep`, `cat`, `find`, `tar`, `chmod` [services/workspace-worker/executor.py:53-60]().

Furthermore, even whitelisted commands are subject to regex-based `BLOCKED_PATTERNS` to prevent dangerous operations such as:
*   `rm -rf /` (Root deletion) [services/workspace-worker/executor.py:77-78]().
*   `sudo` or `su` (Privilege escalation) [services/workspace-worker/executor.py:79-80]().
*   `kubectl` (Unauthorized cluster access) [services/workspace-worker/executor.py:82]().
*   Device access via `> /dev/` [services/workspace-worker/executor.py:83]().
*   Backtick execution and embedded newlines [services/workspace-worker/executor.py:93-94]().

### Data Flow: Secure Command Execution
The following diagram illustrates how a command from an agent is validated before execution on the worker.

**Figure 1: Command Validation and Execution Flow**
```mermaid
graph TD
    subgraph "Orchestrator [orchestrator/core/workspace_client.py]"
        "Agent/User" -- "POST /api/workspaces/{ws_id}/exec" --> "WorkspaceExecAPI"["api.workspace_exec:exec_command"]
        "WorkspaceExecAPI" -- "Proxy Request" --> "WorkspaceClient"["core.workspace_client:WorkspaceClient.exec_command"]
    end

    subgraph "Workspace Worker [services/workspace-worker/main.py]"
        "WorkspaceClient" -- "HTTP POST /exec" --> "WorkspaceWorker"["main.py:WorkspaceWorker"]
        "WorkspaceWorker" -- "Validate Command" --> "WTE_Validate"["executor.py:WorkspaceToolExecutor._validate_command"]
        "WTE_Validate" -- "Check Whitelist" --> "Allowed?"{"Allowed?"}
        "Allowed?" -- "No" --> "SecurityError"["Return SecurityError"]
        "Allowed?" -- "Yes" --> "BuildEnv"["executor.py:WorkspaceToolExecutor._build_sandboxed_env"]
        "BuildEnv" -- "Subprocess" --> "AsyncSubprocess"["asyncio.create_subprocess_exec"]
        "AsyncSubprocess" -- "Output Limit" --> "Truncate"["executor.py:MAX_STDOUT_BYTES"]
    end
```
Sources: [orchestrator/core/workspace_client.py:153-171](), [services/workspace-worker/executor.py:122-143](), [services/workspace-worker/executor.py:101-102](), [services/workspace-worker/main.py:59-68]()

## Storage Quotas & Resource Limits

To prevent resource exhaustion, the `WorkspaceWorker` enforces limits on storage and process output:
*   **Storage Quotas:** The default storage per workspace is configured via `WORKSPACE_DEFAULT_QUOTA_GB` (defaulting to 5GB) [services/workspace-worker/main.py:17](). 
*   **Output Capping:** Command output is truncated if it exceeds `MAX_STDOUT_BYTES` (100KB) or `MAX_STDERR_BYTES` (50KB) to prevent memory bloat in the orchestrator [services/workspace-worker/executor.py:101-102]().
*   **Timeouts:** Every command has a default timeout of 120 seconds, and the `WorkspaceClient` caps requests at 600 seconds [services/workspace-worker/executor.py:105](), [orchestrator/core/workspace_client.py:162]().
*   **Concurrency:** The worker uses an `asyncio.Semaphore` to limit concurrent task execution based on `WORKER_CONCURRENCY` [services/workspace-worker/main.py:78]().

Sources: [services/workspace-worker/main.py:17-18](), [services/workspace-worker/executor.py:101-105](), [orchestrator/core/workspace_client.py:162]()

## Widget Security & Rate Limiting

The widget system allows embedding Automatos capabilities into external sites. This requires specialized security measures to prevent abuse and ensure cross-origin safety.

### Rate Limiting via `WidgetRateLimitMiddleware`
The widget subsystem includes rate limiting using a thread-safe in-memory sliding-window counter [orchestrator/api/widgets/rate_limit.py:46-51]().
*   **Public Keys:** Default 30 requests per minute [orchestrator/api/widgets/rate_limit.py:37]().
*   **Server Keys:** Default 1000 requests per minute for keys starting with `ak_srv_` [orchestrator/api/widgets/rate_limit.py:38](), [orchestrator/api/widgets/rate_limit.py:127]().
*   **Headers:** The system injects standard headers (`X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`) into every widget response [orchestrator/api/widgets/rate_limit.py:152-156]().
*   **Exceeding Limits:** If the limit is exceeded, it returns a `429 Too Many Requests` status with a `Retry-After` header [orchestrator/api/widgets/rate_limit.py:136-149]().

### CORS Policy
Widget endpoints (`/api/widgets/*`) use a dedicated `WidgetCORSMiddleware`. This middleware is ASGI-native to avoid buffering `StreamingResponse` (SSE) data, ensuring real-time performance for chat streams [orchestrator/api/widgets/cors.py:5-8]().
*   **Origin Allowlist:** There is no global widget origin env var. A merchant storefront is authorised from the per-key `SdkApiKey.allowed_domains` the merchant maintains — the same list `widget_auth` enforces on the real request. Preflights carry no `Authorization` header, so the middleware asks whether the origin is named on *any* active public key [orchestrator/api/widgets/cors.py]().
*   **First-party Origins:** Our own dashboard and marketing site come from `config.CORS_ALLOW_ORIGINS`, the same list the app-wide `CORSMiddleware` uses. `/api/sites/*` (dashboard Sites CRUD, JWT-cookie auth) resolves from this list only and never consults merchant keys — otherwise any merchant could open CORS on the admin surface by naming that origin on their own key.
*   **Preflight Handling:** The middleware handles `OPTIONS` requests by validating the `Origin` and returning appropriate `Access-Control-Allow-*` headers [orchestrator/api/widgets/cors.py:57-77]().
*   **Credentials:** If the origin is allowed, `access-control-allow-credentials: true` is set to support authenticated widget interactions [orchestrator/api/widgets/cors.py:88]().

**Figure 2: Widget Authentication and Rate Limiting Architecture**
```mermaid
graph LR
    subgraph "External Webpage"
        "JS_SDK"["JS Widget SDK"]
    end

    subgraph "Automatos Backend [orchestrator/api/widgets/]"
        "CORS_MW"["cors.py:WidgetCORSMiddleware"]
        "RL_MW"["rate_limit.py:WidgetRateLimitMiddleware"]
        "RL_Store"["rate_limit.py:RateLimitStore"]
    end

    "JS_SDK" -- "POST /api/widgets/chat" --> "CORS_MW"
    "CORS_MW" -- "Next" --> "RL_MW"
    "RL_MW" -- "check(key_id)" --> "RL_Store"
    "RL_MW" -- "Inject Headers" --> "Response"["HTTP Response"]
```
Sources: [orchestrator/api/widgets/cors.py:36-42](), [orchestrator/api/widgets/rate_limit.py:85-103](), [orchestrator/api/widgets/rate_limit.py:134-156]()

## GitHub Integration Security

The GitHub integration allows agents to clone repositories into their workspace using Composio actions. To prevent exploitation via malicious URLs:
*   **Host Validation:** Only `github.com`, `gitlab.com`, and `bitbucket.org` are permitted hosts for HTTPS clone URLs [orchestrator/api/workspace_github.py:37](), [orchestrator/api/workspace_github.py:91-92]().
*   **Credential Scrubbing:** Clone URLs must not contain embedded credentials (username or password) [orchestrator/api/workspace_github.py:93-94]().
*   **Branch Sanitization:** Branch names are validated against a strict regex (`_BRANCH_RE`) to prevent shell injection or directory traversal within the `.git` directory [orchestrator/api/workspace_github.py:40](), [orchestrator/api/workspace_github.py:105-106]().

Sources: [orchestrator/api/workspace_github.py:37-40](), [orchestrator/api/workspace_github.py:81-95]()

---