# Tool Assignment

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [docs/DoctorsNotes.docx](docs/DoctorsNotes.docx)
- [orchestrator/api/tools.py](orchestrator/api/tools.py)
- [orchestrator/consumers/chatbot/tool_router.py](orchestrator/consumers/chatbot/tool_router.py)
- [orchestrator/core/composio/client.py](orchestrator/core/composio/client.py)
- [orchestrator/modules/tools/execution/unified_executor.py](orchestrator/modules/tools/execution/unified_executor.py)
- [orchestrator/modules/tools/registry/tool_registry.py](orchestrator/modules/tools/registry/tool_registry.py)
- [orchestrator/modules/tools/services/composio_hint_service.py](orchestrator/modules/tools/services/composio_hint_service.py)
- [orchestrator/modules/tools/services/composio_tool_service.py](orchestrator/modules/tools/services/composio_tool_service.py)
- [orchestrator/modules/tools/tool_router.py](orchestrator/modules/tools/tool_router.py)
- [orchestrator/services/metadata_sync_service.py](orchestrator/services/metadata_sync_service.py)

</details>



## Purpose and Scope

This document explains how Composio tools (external app integrations) are assigned to agents in the Automatos AI platform. It covers the `AgentAppAssignment` table, stable tool ID generation, connection filtering, and the full lifecycle of tool assignments.

For information about the Composio integration architecture and entity management, see [Composio Integration](#6.1). For details on how tools are resolved and executed at runtime, see [Tool Resolution Strategies](#6.2) and [Tool Router & Execution](#6.3). For information about establishing OAuth connections, see [Connecting Apps](#6.4).

---

## Assignment Architecture

Tool assignment in Automatos AI follows a database-backed model where each agent-to-app relationship is stored as an `AgentAppAssignment` record. This approach provides:

- **Persistence**: Assignments survive agent restarts and system redeployments
- **Configuration**: Per-assignment config overrides (priority, custom settings)
- **Activation Control**: Assignments can be enabled/disabled without deletion
- **Audit Trail**: Track who assigned tools and when
- **Connection Filtering**: Only connected apps can be assigned

### High-Level Assignment Flow

```mermaid
graph TB
    subgraph "Frontend"
        UI["Agent Configuration UI"]
        ToolSelector["Tool Selector Component"]
    end
    
    subgraph "API Layer"
        CreateAgent["POST /api/agents"]
        UpdateAgent["PUT /api/agents/{id}"]
        ResolveIDs["_resolve_tool_ids_to_app_names()"]
    end
    
    subgraph "Connection Layer"
        EntityManager["EntityManager"]
        GetConnections["get_entity_connections()"]
    end
    
    subgraph "Database"
        AgentTable[("Agent table")]
        AssignmentTable[("agent_app_assignments")]
        CacheTable[("composio_app_cache")]
    end
    
    subgraph "Composio"
        ComposioAPI["Composio API"]
        EntityConnections["Entity Connections"]
    end
    
    UI --> ToolSelector
    ToolSelector --> CreateAgent
    ToolSelector --> UpdateAgent
    
    CreateAgent --> ResolveIDs
    UpdateAgent --> ResolveIDs
    
    ResolveIDs --> EntityManager
    EntityManager --> GetConnections
    GetConnections --> ComposioAPI
    
    ResolveIDs --> CacheTable
    
    CreateAgent --> AgentTable
    UpdateAgent --> AgentTable
    
    CreateAgent --> AssignmentTable
    UpdateAgent --> AssignmentTable
    
    ComposioAPI --> EntityConnections
```

**Sources**: [orchestrator/api/agents.py:63-110](), [orchestrator/api/agents.py:362-438](), [orchestrator/api/agents.py:608-698]()

---

## Stable Tool IDs

The frontend and backend use a **stable hashing function** to generate consistent integer IDs for Composio apps based on their names. This allows the UI to work with predictable IDs even before apps are cached in the database.

### Hash Algorithm

```mermaid
graph LR
    AppName["App Name<br/>(e.g. 'GITHUB')"]
    Hash["Hash Function<br/>h = sum(ord(ch) * 31^i)"]
    ToSigned["Convert to signed 32-bit"]
    Negate["Make negative"]
    StableID["Stable Tool ID<br/>(e.g. -1234567)"]
    
    AppName --> Hash
    Hash --> ToSigned
    ToSigned --> Negate
    Negate --> StableID
```

The hash function in `_stable_tool_id()` matches the frontend's `stableId()` function, ensuring consistency:

```python
def _stable_tool_id(name: str) -> int:
    """Match frontend stableId() hash (negative int)."""
    h = 0
    for ch in (name or ""):
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
        # convert to signed 32-bit
        if h & 0x80000000:
            h = -((~h + 1) & 0xFFFFFFFF)
    if h == 0:
        return -1
    return -abs(int(h))
```

**Key Properties**:
- Always returns a **negative integer** (to distinguish from database IDs)
- Deterministic: same app name always produces the same ID
- Case-sensitive: "GITHUB" and "github" produce different IDs

**Sources**: [orchestrator/api/agents.py:34-44]()

---

## Assignment Table Schema

The `agent_app_assignments` table (represented by the `AgentAppAssignment` model) stores the many-to-many relationship between agents and Composio apps.

### Table Structure

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key |
| `agent_id` | INTEGER | Foreign key to `agents.id` |
| `app_name` | VARCHAR | Composio app name (uppercase, e.g. "GITHUB") |
| `app_type` | VARCHAR | Always "EXTERNAL" for Composio apps |
| `assigned_by` | INTEGER | User ID who made the assignment (if available) |
| `assigned_at` | TIMESTAMP | When the assignment was created |
| `is_active` | BOOLEAN | Whether the assignment is currently active |
| `priority` | INTEGER | Assignment priority (default 0) |
| `config` | JSONB | Per-assignment configuration overrides |

### Entity Relationships

```mermaid
erDiagram
    Agent ||--o{ AgentAppAssignment : "has"
    AgentAppAssignment }o--|| ComposioAppCache : "references"
    ComposioAppCache ||--|| ComposioAPI : "cached from"
    
    Agent {
        int id PK
        string name
        string workspace_id FK
    }
    
    AgentAppAssignment {
        int id PK
        int agent_id FK
        string app_name
        string app_type
        int assigned_by
        timestamp assigned_at
        boolean is_active
        int priority
        jsonb config
    }
    
    ComposioAppCache {
        int id PK
        string app_name UK
        string description
        string logo_url
        jsonb categories
    }
```

**Sources**: [orchestrator/api/agents.py:13](), [orchestrator/api/agents.py:146-175]()

---

## Connection Filtering

A critical feature of tool assignment is **connection filtering**: only apps that are connected for the current workspace can be assigned to agents. This prevents configuration errors where an agent expects a tool but no OAuth connection exists.

### Connection Validation Flow

```mermaid
sequenceDiagram
    participant API as "Agent API"
    participant Resolver as "_resolve_tool_ids_to_app_names()"
    participant EM as "EntityManager"
    participant Composio as "Composio API"
    participant DB as "Database"
    
    API->>Resolver: tool_ids = [123, -456, 789]
    Resolver->>EM: get_entity_by_workspace(workspace_id)
    EM->>Composio: GET /entities?workspace_id=...
    Composio-->>EM: entity_id
    
    Resolver->>EM: get_entity_connections(entity_id)
    EM->>Composio: GET /entities/{id}/connections
    Composio-->>EM: [{"app_name": "GITHUB", "status": "active"}, ...]
    
    Resolver->>Resolver: Filter connections by status<br/>(active, added, pending only)
    
    Resolver->>DB: Query ComposioAppCache<br/>WHERE app_name IN (connected_apps)
    DB-->>Resolver: Cached app records
    
    Resolver->>Resolver: Build id_to_app mapping<br/>(DB IDs + stable IDs)
    
    Resolver->>Resolver: Resolve tool_ids to app_names<br/>Filter out unconnected apps
    
    Resolver-->>API: ["GITHUB", "SLACK"]
```

The `_resolve_tool_ids_to_app_names()` function performs this validation:

**Allowed Connection Statuses**:
- `active` - OAuth connection is active
- `added` - Connection added but not yet authorized
- `pending` - Connection authorization in progress

**Rejected Connection Statuses**:
- `disabled` - Connection was disabled
- `failed` - Connection authorization failed
- Any other status

**Sources**: [orchestrator/api/agents.py:63-110](), [orchestrator/core/composio/entity_manager.py]()

---

## Creating Tool Assignments

When a new agent is created via `POST /api/agents`, the frontend sends a list of `tool_ids` that may include:
- Database IDs from `composio_app_cache.id`
- Stable hash IDs (negative integers)

### Creation Process

```mermaid
graph TB
    RequestBody["Request Body<br/>{name, description, tool_ids: [123, -456]}"]
    
    Validate["Validate agent_data"]
    CreateAgent["Create Agent record"]
    
    ResolveIDs["_resolve_tool_ids_to_app_names()"]
    CheckConnections["Filter by workspace connections"]
    
    LoopApps{"For each<br/>resolved app"}
    CreateAssignment["Create AgentAppAssignment<br/>agent_id, app_name, app_type='EXTERNAL'"]
    
    Commit["db.commit()"]
    LoadAgent["Load agent with tools"]
    BuildResponse["_build_agent_response()"]
    Response["Return AgentResponse<br/>with tools array"]
    
    RequestBody --> Validate
    Validate --> CreateAgent
    CreateAgent --> ResolveIDs
    ResolveIDs --> CheckConnections
    CheckConnections --> LoopApps
    
    LoopApps -->|Yes| CreateAssignment
    CreateAssignment --> LoopApps
    LoopApps -->|No| Commit
    
    Commit --> LoadAgent
    LoadAgent --> BuildResponse
    BuildResponse --> Response
```

### Code Implementation

The assignment creation logic in `POST /api/agents`:

```python
# Add tools (NEW: agent_app_assignments)
if agent_data.tool_ids:
    desired_apps = _resolve_tool_ids_to_app_names(db, ctx, agent_data.tool_ids)
    for app_name in desired_apps:
        db.add(
            AgentAppAssignment(
                agent_id=agent.id,
                app_name=app_name,
                app_type="EXTERNAL",
                assigned_by=_assigned_by_user_id(ctx),
                is_active=True,
                priority=0,
                config={},
            )
        )
```

**Sources**: [orchestrator/api/agents.py:407-421](), [orchestrator/api/agents.py:362-438]()

---

## Updating Tool Assignments

When updating an agent via `PUT /api/agents/{id}`, the backend uses a **diff-and-sync** strategy:
1. Resolve new tool IDs to app names
2. Query existing assignments
3. Disable assignments no longer selected
4. Re-enable or create assignments for selected tools

### Update Process

```mermaid
graph TB
    UpdateRequest["PUT /api/agents/{id}<br/>tool_ids: [123, -789]"]
    
    ResolveNew["Resolve tool_ids to app_names<br/>(with connection filtering)"]
    DesiredSet["desired_set = {'GITHUB', 'SLACK'}"]
    
    QueryCurrent["Query AgentAppAssignment<br/>WHERE agent_id = {id}"]
    CurrentMap["current_map = {'GITHUB': row1, 'JIRA': row2}"]
    
    LoopCurrent{"For each<br/>current assignment"}
    CheckDesired{"Is app_name<br/>in desired_set?"}
    Disable["Set is_active = False"]
    
    LoopDesired{"For each<br/>desired app"}
    CheckCurrent{"Already<br/>assigned?"}
    Reactivate["Set is_active = True"]
    CreateNew["Create new AgentAppAssignment"]
    
    Commit["db.commit()"]
    
    UpdateRequest --> ResolveNew
    ResolveNew --> DesiredSet
    DesiredSet --> QueryCurrent
    QueryCurrent --> CurrentMap
    
    CurrentMap --> LoopCurrent
    LoopCurrent -->|Yes| CheckDesired
    CheckDesired -->|No| Disable
    CheckDesired -->|Yes| LoopCurrent
    Disable --> LoopCurrent
    
    LoopCurrent -->|No| LoopDesired
    LoopDesired -->|Yes| CheckCurrent
    CheckCurrent -->|Yes| Reactivate
    CheckCurrent -->|No| CreateNew
    Reactivate --> LoopDesired
    CreateNew --> LoopDesired
    
    LoopDesired -->|No| Commit
```

### Soft Deletion Pattern

The backend uses **soft deletion** by setting `is_active = False` rather than deleting rows. This provides:
- **Audit Trail**: History of what was assigned and when
- **Re-activation**: Easy to re-enable previously assigned tools
- **Analytics**: Track tool assignment patterns over time

**Sources**: [orchestrator/api/agents.py:651-683](), [orchestrator/api/agents.py:608-698]()

---

## Reading Tool Assignments

When reading agent details via `GET /api/agents/{id}`, the backend loads assignments and enriches them with cached app metadata.

### Response Building Process

```mermaid
graph TB
    GetAgent["GET /api/agents/{id}"]
    
    LoadAgent["Load Agent from DB<br/>(with relationships)"]
    
    QueryAssignments["Query AgentAppAssignment<br/>WHERE agent_id = {id}<br/>AND is_active = True"]
    
    ExtractNames["Extract app_names from assignments"]
    QueryCache["Query ComposioAppCache<br/>WHERE app_name IN (names)"]
    BuildMap["Build cache_map = {app_name: cached_row}"]
    
    LoopAssignments{"For each<br/>assignment"}
    LookupCache["Lookup cached app data"]
    BuildTool["Build tool object<br/>{id, name, description, icon, ...}"]
    
    AddToList["Add to tools array"]
    
    Response["Return AgentResponse<br/>with tools: [...]"]
    
    GetAgent --> LoadAgent
    LoadAgent --> QueryAssignments
    QueryAssignments --> ExtractNames
    ExtractNames --> QueryCache
    QueryCache --> BuildMap
    
    BuildMap --> LoopAssignments
    LoopAssignments -->|Yes| LookupCache
    LookupCache --> BuildTool
    BuildTool --> AddToList
    AddToList --> LoopAssignments
    
    LoopAssignments -->|No| Response
```

### Tool Object Structure

Each tool in the response includes:

```json
{
  "id": 123,                          // ComposioAppCache.id (or null)
  "assignment_id": 456,               // AgentAppAssignment.id
  "name": "GITHUB",                   // App name (uppercase)
  "description": "GitHub integration", // From cache
  "provider": "Composio",             // Always "Composio"
  "category": "developer-tools",      // From cache categories[0]
  "icon": "https://...",              // Logo URL from cache
  "permissions": {},                  // Reserved for future use
  "configuration": {},                // From assignment.config
  "assigned_at": "2025-01-15T10:30:00Z" // Timestamp
}
```

**Sources**: [orchestrator/api/agents.py:146-175](), [orchestrator/api/agents.py:140-240]()

---

## Assignment Lifecycle States

Assignments follow a lifecycle managed through the `is_active` boolean field:

### State Diagram

```mermaid
stateDiagram-v2
    [*] --> NotAssigned
    
    NotAssigned --> Active : Create Assignment<br/>(is_active=True)
    
    Active --> Inactive : Update agent.tool_ids<br/>(remove app)
    Inactive --> Active : Update agent.tool_ids<br/>(re-add app)
    
    Active --> Active : Update agent<br/>(app still selected)
    
    Active --> Deleted : Delete agent
    Inactive --> Deleted : Delete agent
    
    Deleted --> [*]
    
    note right of Active
        Visible in GET /api/agents/{id}
        Used at runtime for tool resolution
    end note
    
    note right of Inactive
        Hidden from API responses
        Preserved in database
        Can be reactivated
    end note
```

### State Transitions

| From State | To State | Trigger | Database Action |
|------------|----------|---------|-----------------|
| Not Assigned | Active | Create agent with tool_ids | `INSERT` with `is_active=True` |
| Not Assigned | Active | Update agent, add tool | `INSERT` with `is_active=True` |
| Active | Active | Update agent, keep tool | No change |
| Active | Inactive | Update agent, remove tool | `UPDATE SET is_active=False` |
| Inactive | Active | Update agent, re-add tool | `UPDATE SET is_active=True` |
| Active/Inactive | Deleted | Delete agent | `DELETE` (cascade) |

**Sources**: [orchestrator/api/agents.py:651-683]()

---

## User ID Tracking

The `assigned_by` field attempts to track which user created the assignment. However, due to the current authentication system using Clerk user IDs (strings like `"user_..."`), while the database column is an `INTEGER`, the field is often `NULL`.

### Current Implementation

```python
def _assigned_by_user_id(ctx: RequestContext) -> Optional[int]:
    """
    `agent_app_assignments.assigned_by` is an INTEGER in Postgres.
    Our RequestContext `user.id` is currently a Clerk user id string (e.g. "user_..."),
    so we must never write it into this column.
    """
    try:
        raw = getattr(getattr(ctx, "user", None), "id", None)
        if raw is None:
            return None
        # Only accept numeric values
        return int(raw)
    except Exception:
        return None
```

This function returns `None` for most requests since Clerk IDs are non-numeric. Future work could:
- Create a separate user mapping table
- Change the column type to VARCHAR
- Use a different identifier

**Sources**: [orchestrator/api/agents.py:47-60]()

---

## Runtime Tool Resolution

At agent execution time, the system loads active assignments and resolves them to Composio tool definitions. This process is separate from assignment management and is covered in detail in [Tool Resolution Strategies](#6.2).

### Quick Overview

```mermaid
graph LR
    AgentExecution["Agent Execution Request"]
    LoadAssignments["Load active AgentAppAssignment<br/>WHERE agent_id = X AND is_active = True"]
    ExtractNames["Extract app_names array"]
    ResolveTools["ComposioToolService<br/>or ComposioHintService"]
    ToolDefinitions["Tool Definitions<br/>(OpenAI function schemas)"]
    ExecuteLLM["Pass to LLM for tool calling"]
    
    AgentExecution --> LoadAssignments
    LoadAssignments --> ExtractNames
    ExtractNames --> ResolveTools
    ResolveTools --> ToolDefinitions
    ToolDefinitions --> ExecuteLLM
```

**Sources**: See [Tool Resolution Strategies](#6.2) for complete details.

---

## Best Practices

### For API Consumers

1. **Always send stable IDs**: Use the frontend's `stableId()` hash function or query `GET /api/tools/connected` to get valid IDs
2. **Respect connection status**: Don't attempt to assign apps that aren't connected
3. **Use full updates**: Send the complete list of desired tool_ids in update requests (the backend handles the diff)
4. **Check tool visibility**: After updates, verify tools appear in `GET /api/agents/{id}` response

### For Backend Developers

1. **Preserve soft deletion**: Never hard-delete assignments unless cascading from agent deletion
2. **Filter by is_active**: Always filter `WHERE is_active = True` when loading for runtime
3. **Validate connections**: Always call `_resolve_tool_ids_to_app_names()` to enforce connection filtering
4. **Handle cache misses**: Tools may not have cached metadata; handle `null` gracefully

**Sources**: [orchestrator/api/agents.py:63-110](), [orchestrator/api/agents.py:146-175](), [orchestrator/api/agents.py:651-683]()

---

## API Integration Examples

### Creating an Agent with Tools

```http
POST /api/agents
Content-Type: application/json
X-Workspace-ID: <workspace-uuid>
Authorization: Bearer <jwt>

{
  "name": "GitHub Assistant",
  "description": "Helps with GitHub operations",
  "agent_type": "custom",
  "tool_ids": [-1234567, 98765],  // Stable hash + DB ID
  "configuration": {}
}
```

### Updating Agent Tools

```http
PUT /api/agents/42
Content-Type: application/json
X-Workspace-ID: <workspace-uuid>
Authorization: Bearer <jwt>

{
  "tool_ids": [-1234567]  // Remove all but GITHUB
}
```

### Reading Agent Tools

```http
GET /api/agents/42
X-Workspace-ID: <workspace-uuid>
Authorization: Bearer <jwt>

Response:
{
  "id": 42,
  "name": "GitHub Assistant",
  "tools": [
    {
      "id": 123,
      "assignment_id": 456,
      "name": "GITHUB",
      "description": "GitHub integration",
      "category": "developer-tools",
      "icon": "https://...",
      "assigned_at": "2025-01-15T10:30:00Z"
    }
  ]
}
```

**Sources**: [orchestrator/api/agents.py:362-438](), [orchestrator/api/agents.py:608-698](), [orchestrator/api/agents.py:537-555]()

---