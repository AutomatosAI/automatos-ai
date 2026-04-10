# GitHub Integration

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [frontend/components/widgets/CodingCanvasWidget/RepoSelector.tsx](frontend/components/widgets/CodingCanvasWidget/RepoSelector.tsx)
- [orchestrator/api/main.py](orchestrator/api/main.py)
- [orchestrator/api/tasks.py](orchestrator/api/tasks.py)
- [orchestrator/api/widgets/cors.py](orchestrator/api/widgets/cors.py)
- [orchestrator/api/widgets/rate_limit.py](orchestrator/api/widgets/rate_limit.py)
- [orchestrator/api/workspace_files.py](orchestrator/api/workspace_files.py)
- [orchestrator/api/workspace_github.py](orchestrator/api/workspace_github.py)
- [orchestrator/core/workspace_client.py](orchestrator/core/workspace_client.py)
- [orchestrator/modules/tools/discovery/workspace_actions.py](orchestrator/modules/tools/discovery/workspace_actions.py)
- [services/workspace-worker/executor.py](services/workspace-worker/executor.py)
- [services/workspace-worker/main.py](services/workspace-worker/main.py)
- [services/workspace-worker/workspace_manager.py](services/workspace-worker/workspace_manager.py)

</details>



The GitHub Integration subsystem enables AI agents and users to interact with remote repositories directly within sandboxed workspace environments. This integration supports repository discovery, automated cloning into persistent workspace volumes, and a suite of tools for code manipulation and version control backed by the Composio SDK.

## Overview

GitHub integration is implemented as a bridge between the **Orchestrator API**, the **Workspace Worker**, and external VCS providers. It leverages the `Composio` platform to handle OAuth authentication and entity management, allowing agents to act on behalf of users with fine-grained permissions.

### Key Components
*   **`workspace_github.py`**: The primary API router for repository listing and clone task submission [orchestrator/api/workspace_github.py:31-34]().
*   **`Workspace Worker`**: An ARQ-style consumer that executes the physical `git clone` and file operations on a persistent volume [services/workspace-worker/main.py:58-67]().
*   **`RepoSelector`**: A frontend component allowing users to browse and select repositories for their workspace [frontend/components/widgets/CodingCanvasWidget/RepoSelector.tsx:39-44]().
*   **`WorkspaceClient`**: An asynchronous proxy used by the Orchestrator to communicate with the worker's HTTP API for file and command operations [orchestrator/core/workspace_client.py:56-61]().

## Implementation & Data Flow

The integration follows a decoupled architecture where the API server manages metadata and permissions, while the worker service handles heavy I/O and shell execution.

### Repository Discovery and Cloning
When a user or agent requests a repository list, the system uses the `EntityManager` to resolve the workspace's Composio identity [orchestrator/api/workspace_github.py:47-58](). It then fetches repository metadata via the `GITHUB_LIST_REPOSITORIES_FOR_THE_AUTHENTICATED_USER` action [orchestrator/api/workspace_github.py:118-122]().

**GitHub Repository Operations Flow**

```mermaid
sequenceDiagram
    participant FE as "Frontend (RepoSelector.tsx)"
    participant API as "Orchestrator API (workspace_github.py)"
    participant CMP as "Composio SDK / GitHub API"
    participant DB as "PostgreSQL (task_executions)"
    participant RED as "Redis (workspace:tasks:normal)"
    participant WRK as "Workspace Worker (main.py)"

    FE->>API: "GET /api/workspaces/{id}/github/repos"
    API->>CMP: "execute_action(GITHUB_LIST_REPOSITORIES)"
    CMP-->>API: "List of Repositories"
    API-->>FE: "JSON Repo List"

    FE->>API: "POST /api/workspaces/{id}/github/clone"
    Note over API: "Validate HTTPS URL & Branch"
    API->>DB: "INSERT task_executions (status='queued')"
    API->>RED: "LPUSH workspace:tasks:normal {payload}"
    API-->>FE: "202 Accepted (task_id)"

    WRK->>RED: "RPOP workspace:tasks:normal"
    WRK->>WRK: "git clone --branch {b} {url}"
    WRK->>DB: "UPDATE task_executions (status='completed')"
```
Sources: [orchestrator/api/workspace_github.py:97-161](), [orchestrator/api/workspace_github.py:167-200](), [orchestrator/api/tasks.py:106-151](), [services/workspace-worker/main.py:179-193]()

## API Reference: workspace_github.py

The GitHub integration provides two main endpoints scoped by `workspace_id`.

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/workspaces/{id}/github/repos` | `GET` | Lists GitHub repositories accessible via the authenticated Composio entity [orchestrator/api/workspace_github.py:97-98](). |
| `/api/workspaces/{id}/github/clone` | `POST` | Enqueues a background task to clone a repository into the workspace volume [orchestrator/api/workspace_github.py:167-168](). |

### URL Validation
To prevent SSRF and injection attacks, the `CloneRequest` model enforces strict validation:
*   **Scheme**: Must be `https` [orchestrator/api/workspace_github.py:73-74]().
*   **Allowed Hosts**: Limited to `github.com`, `gitlab.com`, and `bitbucket.org` [orchestrator/api/workspace_github.py:37-38](), [orchestrator/api/workspace_github.py:75-76]().
*   **Credentials**: No embedded usernames or passwords allowed in the URL [orchestrator/api/workspace_github.py:77-78]().
*   **Branch**: Validated against a safe regex `^[A-Za-z0-9._/\-]+$` [orchestrator/api/workspace_github.py:40-41](), [orchestrator/api/workspace_github.py:89-91]().

## Agent-Facing Workspace Tools

Agents interact with GitHub repositories using a set of "Platform Actions" defined in the `ActionRegistry`. These actions are proxied to the `Workspace Worker` via the `WorkspaceClient`.

**Code Entity Mapping: Natural Language to Tool Execution**

```mermaid
graph TD
    subgraph "Natural_Language_Space"
        NL["'Search for the login logic in the repo'"]
    end

    subgraph "Code_Entity_Space_Orchestrator"
        AR["ActionRegistry (action_registry.py)"]
        WA["workspace_grep (workspace_actions.py)"]
        WC["WorkspaceClient.grep() (workspace_client.py)"]
    end

    subgraph "Execution_Space_Worker"
        WE["WorkspaceToolExecutor (executor.py)"]
        CMD["/usr/bin/rg (ripgrep)"]
    end

    NL -->|Intent_Matching| AR
    AR -->|Resolves| WA
    WA -->|Calls| WC
    WC -->|HTTP_GET_files_grep| WE
    WE -->|Executes| CMD
```
Sources: [orchestrator/modules/tools/discovery/workspace_actions.py:121-157](), [orchestrator/core/workspace_client.py:130-149](), [services/workspace-worker/executor.py:108-116]()

### Key Workspace Actions
*   **`workspace_read_file`**: Retrieves text content, size, and language of a specific file [orchestrator/modules/tools/discovery/workspace_actions.py:18-50]().
*   **`workspace_write_file`**: Creates or overwrites files, including automatic parent directory creation [orchestrator/modules/tools/discovery/workspace_actions.py:52-88]().
*   **`workspace_list_dir`**: Explores project structure [orchestrator/modules/tools/discovery/workspace_actions.py:90-118]().
*   **`workspace_grep`**: Performs regex searches across the repository [orchestrator/modules/tools/discovery/workspace_actions.py:121-157]().
*   **`workspace_exec`**: Runs whitelisted commands (e.g., `pytest`, `npm test`) in the repository context [orchestrator/modules/tools/discovery/workspace_actions.py:160-200]().

## Security and Sandboxing

GitHub integration adheres to strict security boundaries to prevent unauthorized access or system compromise during code execution.

### Command Whitelisting
The `WorkspaceToolExecutor` maintains an `ALLOWED_COMMANDS` set. Only binaries in this list (e.g., `git`, `python`, `npm`, `ls`) are permitted [services/workspace-worker/executor.py:35-73]().

### Blocked Patterns
Even if a binary is whitelisted, specific argument patterns are blocked via regex (e.g., `rm -rf /`, `sudo`, `chmod 777`) [services/workspace-worker/executor.py:76-95]().

### Path Containment
The `WorkspaceManager` ensures all file operations are confined to the `/workspaces/{workspace_id}` directory using `resolve_safe_path`, which prevents directory traversal attacks [services/workspace-worker/workspace_manager.py:47-51](), [services/workspace-worker/executor.py:146-149]().

### Rate Limiting and CORS
Widget-based access to workspace files is protected by:
*   **`WidgetCORSMiddleware`**: Restricts access to configured origins [orchestrator/api/widgets/cors.py:36-37]().
*   **`WidgetRateLimitMiddleware`**: Implements a sliding-window counter per API key to prevent abuse [orchestrator/api/widgets/rate_limit.py:85-98]().

## Workspace Filesystem Layout

Each workspace is provisioned with a structured directory tree on the persistent volume:

| Directory | Purpose | Persistence |
| :--- | :--- | :--- |
| `/repos/` | Cloned GitHub repositories | Persistent [services/workspace-worker/workspace_manager.py:12]() |
| `/tasks/` | Ephemeral execution directories for specific tasks | Cleaned up after 24h [services/workspace-worker/workspace_manager.py:13](), [services/workspace-worker/workspace_manager.py:139]() |
| `/artifacts/` | Test reports, build outputs, and logs | Persistent [services/workspace-worker/workspace_manager.py:14]() |
| `.ssh/` | Deploy keys and SSH configurations | Persistent [services/workspace-worker/workspace_manager.py:15]() |

**System Interaction Map**

```mermaid
graph LR
    subgraph "Frontend_Layer"
        RS["RepoSelector.tsx"]
    end
    
    subgraph "Orchestrator_Layer"
        WG["workspace_github.py"]
        WCL["WorkspaceClient (workspace_client.py)"]
    end
    
    subgraph "Worker_Layer"
        WW["WorkspaceWorker (main.py)"]
        WTE["WorkspaceToolExecutor (executor.py)"]
        WM["WorkspaceManager (workspace_manager.py)"]
    end
    
    RS -->|API_Call| WG
    WG -->|Enqueue_Task| WW
    WW -->|Execute_Command| WTE
    WTE -->|Path_Validation| WM
    WCL -->|Proxy_Request| WTE
```

Sources: [services/workspace-worker/workspace_manager.py:10-18](), [services/workspace-worker/workspace_manager.py:58-77](), [orchestrator/api/workspace_github.py:7-8](), [orchestrator/core/workspace_client.py:9-11]()

---