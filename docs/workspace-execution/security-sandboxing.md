# Security & Sandboxing

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [frontend/components/widgets/CodingCanvasWidget/RepoSelector.tsx](frontend/components/widgets/CodingCanvasWidget/RepoSelector.tsx)
- [orchestrator/api/api_keys.py](orchestrator/api/api_keys.py)
- [orchestrator/api/main.py](orchestrator/api/main.py)
- [orchestrator/api/tasks.py](orchestrator/api/tasks.py)
- [orchestrator/api/widgets/__init__.py](orchestrator/api/widgets/__init__.py)
- [orchestrator/api/widgets/auth.py](orchestrator/api/widgets/auth.py)
- [orchestrator/api/widgets/chat.py](orchestrator/api/widgets/chat.py)
- [orchestrator/api/widgets/cors.py](orchestrator/api/widgets/cors.py)
- [orchestrator/api/widgets/data.py](orchestrator/api/widgets/data.py)
- [orchestrator/api/widgets/documents.py](orchestrator/api/widgets/documents.py)
- [orchestrator/api/widgets/rate_limit.py](orchestrator/api/widgets/rate_limit.py)
- [orchestrator/api/widgets/router.py](orchestrator/api/widgets/router.py)
- [orchestrator/api/widgets/session.py](orchestrator/api/widgets/session.py)
- [orchestrator/api/workspace_files.py](orchestrator/api/workspace_files.py)
- [orchestrator/api/workspace_github.py](orchestrator/api/workspace_github.py)
- [orchestrator/core/database/migrations/043_team_based_document_scoping.sql](orchestrator/core/database/migrations/043_team_based_document_scoping.sql)
- [orchestrator/core/models/sdk_api_keys.py](orchestrator/core/models/sdk_api_keys.py)
- [orchestrator/core/services/api_key_service.py](orchestrator/core/services/api_key_service.py)
- [orchestrator/core/workspace_client.py](orchestrator/core/workspace_client.py)
- [orchestrator/modules/tools/discovery/workspace_actions.py](orchestrator/modules/tools/discovery/workspace_actions.py)
- [services/workspace-worker/executor.py](services/workspace-worker/executor.py)
- [services/workspace-worker/main.py](services/workspace-worker/main.py)
- [services/workspace-worker/workspace_manager.py](services/workspace-worker/workspace_manager.py)

</details>



Automatos AI implements a multi-layered security architecture designed to isolate agent execution, protect the host filesystem, and govern external access via widgets. The system relies on physical workspace boundaries, strict command whitelisting, and per-workspace rate limiting to ensure that autonomous agents operate within safe constraints.

## Workspace Isolation & Path Traversal Prevention

The core of the sandboxing strategy is the `WorkspaceWorker`, which manages isolated directories for each workspace on a persistent volume. Security is enforced at the `WorkspaceManager` and `WorkspaceToolExecutor` levels to prevent agents from accessing data outside their designated environment.

### Path Containment
All file operations and command executions are passed through a validation layer. The `WorkspaceToolExecutor` uses `WorkspaceManager.resolve_safe_path` to ensure that any path provided by an agent (or user) is resolved relative to the workspace root and does not use `..` or symlinks to escape the boundary [services/workspace-worker/executor.py:146-149]().

### Command Whitelisting
Agents do not have unrestricted shell access. The system maintains a strict `ALLOWED_COMMANDS` set, including essential tools for development (e.g., `git`, `python`, `npm`, `ls`, `grep`, `uv`, `docker-compose` for inspection) [services/workspace-worker/executor.py:35-73](). 

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
    subgraph "Orchestrator [orchestrator/api/]"
        "Agent/User" -- "POST /api/workspaces/{ws_id}/exec" --> "WorkspaceFilesRouter"["workspace_files.py:exec_command"]
        "WorkspaceFilesRouter" -- "Proxy Request" --> "WorkspaceClient"["core/workspace_client.py:WorkspaceClient.exec_command"]
    end

    subgraph "Workspace Worker [services/workspace-worker/]"
        "WorkspaceClient" -- "HTTP POST /exec" --> "WorkspaceWorkerHandler"["main.py:WorkspaceWorker"]
        "WorkspaceWorkerHandler" -- "Validate Command" --> "WTE_Validate"["executor.py:WorkspaceToolExecutor._validate_command"]
        "WTE_Validate" -- "Check Whitelist" --> "Allowed?"{"Allowed?"}
        "Allowed?" -- "No" --> "SecurityError"["Return SecurityError"]
        "Allowed?" -- "Yes" --> "BuildEnv"["executor.py:WorkspaceToolExecutor._build_sandboxed_env"]
        "BuildEnv" -- "Subprocess" --> "AsyncSubprocess"["asyncio.create_subprocess_exec"]
        "AsyncSubprocess" -- "Output Limit" --> "Truncate"["executor.py:MAX_STDOUT_BYTES"]
    end
```
Sources: [orchestrator/api/workspace_files.py:86-107](), [services/workspace-worker/executor.py:122-143](), [services/workspace-worker/executor.py:101-102](), [orchestrator/api/main.py:95]()

## Storage Quotas & Resource Limits

To prevent resource exhaustion, the `WorkspaceWorker` enforces limits on storage and process output:
*   **Storage Quotas:** The default storage per workspace is configured via `WORKSPACE_DEFAULT_QUOTA_GB` (defaulting to 5GB) [services/workspace-worker/main.py:17](). The `WorkspaceManager` calculates usage via `get_usage_bytes` by walking the workspace root [services/workspace-worker/workspace_manager.py:83-95]().
*   **Output Capping:** Command output is truncated if it exceeds `MAX_STDOUT_BYTES` (100KB) or `MAX_STDERR_BYTES` (50KB) to prevent memory bloat in the orchestrator [services/workspace-worker/executor.py:101-102]().
*   **Timeouts:** Every command has a default timeout of 120 seconds, configurable up to a maximum of 300 seconds [orchestrator/api/workspace_files.py:83](), [services/workspace-worker/executor.py:105]().

## Widget Security & SDK Access Control

The widget system allows embedding Automatos capabilities into external sites. This requires specialized security measures to prevent abuse and ensure cross-origin safety.

### API Key Types & Session Exchange
The system supports two types of SDK API keys defined in `SdkApiKey` [orchestrator/core/models/sdk_api_keys.py:47]():
1.  **Public Keys (`ak_pub_`):** Used in browser environments. Must have a defined `allowed_domains` list [orchestrator/api/api_keys.py:122-127]().
2.  **Server Keys (`ak_srv_`):** Used in backend environments. Can be exchanged for short-lived JWT session tokens via the `/widgets/auth` endpoint [orchestrator/api/widgets/session.py:74-79]().

The `ApiKeyService` handles hashing (SHA-256) so plaintext keys are never stored [orchestrator/core/services/api_key_service.py:23-25]().

### Widget Authentication Flow
Requests to widget endpoints are processed by the `widget_auth` dependency [orchestrator/api/widgets/auth.py:112](). It checks for a Bearer token and attempts to:
1.  Decode it as a JWT session token [orchestrator/api/widgets/auth.py:141]().
2.  Fall back to raw API key validation and origin/domain checks [orchestrator/api/widgets/auth.py:174-187]().

**Figure 2: Widget Authentication and Permission Resolution**
```mermaid
graph TD
    "Request" -- "Bearer Token" --> "widget_auth"["api/widgets/auth.py:widget_auth"]
    "widget_auth" -- "Check JWT" --> "JWT_Valid?"{"JWT Valid?"}
    
    "JWT_Valid?" -- "Yes" --> "ExtractContext"["Populate WidgetAuthContext"]
    "JWT_Valid?" -- "No" --> "DB_Lookup"["core/services/api_key_service.py:validate_api_key"]
    
    "DB_Lookup" -- "Match Found" --> "DomainCheck"["api_key_service.py:check_domain"]
    "DomainCheck" -- "Allowed" --> "ExtractContext"
    
    "ExtractContext" -- "Includes" --> "WorkspaceID"["workspace_id"]
    "ExtractContext" -- "Includes" --> "Perms"["permissions"]
    "ExtractContext" -- "Includes" --> "AgentLock"["default_agent_id"]
```
Sources: [orchestrator/api/widgets/auth.py:112-198](), [orchestrator/core/services/api_key_service.py:97-124](), [orchestrator/api/widgets/session.py:124-147]()

### CORS & Rate Limiting
*   **CORS:** Widget endpoints use `WidgetCORSMiddleware` which validates the `Origin` or `Referer` headers against the allowed domains of the API key [orchestrator/api/widgets/auth.py:73-82]().
*   **Rate Limiting:** Managed via `WidgetRateLimitMiddleware` (in `rate_limit.py`). It enforces request quotas per workspace and key type to prevent DoS attacks on the LLM orchestration layer [orchestrator/api/widgets/rate_limit.py:1-150]().

## GitHub Integration Security

The GitHub integration (`workspace_github.py`) allows agents to clone repositories into their workspace using Composio actions. To prevent exploitation via malicious URLs:
*   **Host Validation:** Only `github.com`, `gitlab.com`, and `bitbucket.org` are permitted hosts for HTTPS clone URLs [orchestrator/api/workspace_github.py:37](), [orchestrator/api/workspace_github.py:75-76]().
*   **Credential Scrubbing:** Clone URLs must not contain embedded credentials (username or password) [orchestrator/api/workspace_github.py:77-78]().
*   **Branch Sanitization:** Branch names are validated against a strict regex (`_BRANCH_RE`) to prevent shell injection or directory traversal within the `.git` directory [orchestrator/api/workspace_github.py:40](), [orchestrator/api/workspace_github.py:89-91]().

Sources: [orchestrator/api/workspace_github.py:37-40](), [orchestrator/api/workspace_github.py:65-91]()

---