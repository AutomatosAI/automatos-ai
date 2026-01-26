# PRD-35: Tool Catalog & Registry Architecture

**Version:** 2.0  
**Status:** 🟡 Design Phase  
**Priority:** HIGH - Core Platform Architecture  
**Author:** Automatos AI Platform Team  
**Last Updated:** 2026-01-17  
**Dependencies:** PRD-33 (MCP Gateway Integration), PRD-34 (Unified Integrations Adapter), PRD-18 (Credential Management)

---

## Executive Summary

This PRD defines a production-grade architecture for tool discovery, catalog, assignment, and execution across the Automatos platform. The architecture supports three consumer types:

1. **Automatos UI Users** (Hosted Mode) - Agents, Chat, Workflows with stored credentials
2. **Widgets** (BYO Mode) - Embedded experiences with bring-your-own credentials
3. **Third-Party Integrations** (BYO Mode) - API access with bring-your-own credentials

**Key Principles:**
- **Adapter** owns tool definitions (single source of truth)
- **Context Forge** is the unified MCP gateway (single entry point)
- **Automatos** owns enablement, assignments, and credentials
- **BYO credentials** for widgets/third-parties - no credential storage for external consumers

---

## 1. Problem Statement

### Current Pain
- Tool definitions exist in multiple places (Automatos DB + Adapter DB)
- No clear agent-tool assignment model
- ToolRegistry is just a lookup with no access control
- Unclear credential resolution flow between systems
- Context Forge role not properly integrated

### Desired State
- Single source of truth for tool definitions (Adapter)
- Clear agent-tool assignment with explicit enablement
- 1:1 tool:credential mapping at enablement time
- Support for both hosted credentials and BYO credentials
- Context Forge as the unified execution gateway

---

## 2. Scope

### In Scope
- Tool catalog ownership and data flow
- Agent-tool assignment model
- Credential resolution (hosted vs BYO)
- Execution via Context Forge → Adapter
- Consumer models (Automatos, Widgets, Third-Party)

### Out of Scope
- Widget UI implementation details
- Third-party API client libraries
- Multi-region deployment
- Advanced observability

---

## 3. Architecture Overview

### 3.1 Component Responsibilities

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    TOOL PLATFORM (Standalone)                             │
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  CONTEXT FORGE GATEWAY (mcp.automatos.app)                         │  │
│  │  - Single public MCP endpoint for ALL consumers                    │  │
│  │  - JWT authentication (validates tenant_id)                        │  │
│  │  - Gateway routing to Adapter                                      │  │
│  │  - Tool discovery endpoint                                         │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                    │                                      │
│  ┌────────────────────────────────▼───────────────────────────────────┐  │
│  │  UNIFIED ADAPTER                                                   │  │
│  │  - Tool definitions (adapter_tools) = SOURCE OF TRUTH              │  │
│  │  - REST passthrough (OpenAPI-based)                                │  │
│  │  - MCP proxy (upstream MCP servers)                                │  │
│  │  - Credential modes:                                               │  │
│  │    - "byo": credentials in request payload                         │  │
│  │    - "hosted": callback to Automatos API to resolve                │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Consumer Model

```
┌──────────────────────────────────────────────────────────────────────────┐
│  CONSUMER TYPE 1: Automatos UI Users (Hosted Mode)                       │
│                                                                          │
│  - Use Automatos UI (Chat, Agents, Workflows)                            │
│  - Credentials stored encrypted in Automatos                             │
│  - Agent-tool assignments managed in Automatos                           │
│  - credential_mode: "hosted"                                             │
│  - Flow: Automatos → CF → Adapter → (callback to resolve creds) → SaaS   │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│  CONSUMER TYPE 2: Widgets (BYO Mode)                                     │
│                                                                          │
│  - Embedded on customer's website                                        │
│  - Customer provides credentials at runtime                              │
│  - NO credential storage in Automatos                                    │
│  - credential_mode: "byo"                                                │
│  - Flow: Widget → CF → Adapter → (creds from request) → SaaS             │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│  CONSUMER TYPE 3: Third-Party API Integration (BYO Mode)                 │
│                                                                          │
│  - Direct API access to Context Forge                                    │
│  - Bring own credentials in every request                                │
│  - credential_mode: "byo"                                                │
│  - Flow: API Call → CF → Adapter → (creds from request) → SaaS           │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Full Architecture Diagram

```
                              ┌─────────────────────┐
                              │   External SaaS     │
                              │ (GitHub, Slack...)  │
                              └──────────▲──────────┘
                                         │
                                         │ REST/MCP
                                         │
┌────────────────────────────────────────┼────────────────────────────────┐
│                    TOOL PLATFORM       │                                 │
│                                        │                                 │
│  ┌─────────────────────────────────────┴──────────────────────────────┐ │
│  │                    UNIFIED ADAPTER                                  │ │
│  │  - adapter_tools table = SOURCE OF TRUTH for tool definitions      │ │
│  │  - RestExecutor: OpenAPI passthrough                               │ │
│  │  - McpExecutor: MCP proxy                                          │ │
│  │  - credential_mode: "byo" → use from request                       │ │
│  │  - credential_mode: "hosted" → callback to Automatos API           │ │
│  └─────────────────────────────────────▲──────────────────────────────┘ │
│                                        │                                 │
│  ┌─────────────────────────────────────┴──────────────────────────────┐ │
│  │              CONTEXT FORGE (mcp.automatos.app)                      │ │
│  │  - Gateway routing                                                  │ │
│  │  - JWT auth (validates tenant_id)                                   │ │
│  │  - Tool discovery (from Adapter)                                    │ │
│  └─────────────────────────────────────▲──────────────────────────────┘ │
│                                        │                                 │
└────────────────────────────────────────┼────────────────────────────────┘
                                         │
          ┌──────────────────────────────┼──────────────────────────────┐
          │                              │                              │
          ▼                              ▼                              ▼
┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐
│   AUTOMATOS UI   │         │     WIDGETS      │         │   THIRD-PARTY    │
│   (Hosted Mode)  │         │   (BYO Mode)     │         │    (BYO Mode)    │
│                  │         │                  │         │                  │
│ - Agents         │         │ - Embedded       │         │ - API access     │
│ - Tool assigns   │         │ - Customer creds │         │ - Customer creds │
│ - Credentials    │◀────────│ - Widget config  │         │                  │
│ - Tenant config  │ callback│                  │         │                  │
└──────────────────┘         └──────────────────┘         └──────────────────┘

         │
         │ Adapter calls back to resolve credentials (hosted mode)
         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         AUTOMATOS DATABASE                                │
│                                                                           │
│  ┌────────────────┐  ┌────────────────┐  ┌─────────────────────────────┐ │
│  │ tenants        │  │ credentials    │  │ agent_tool_assignments      │ │
│  │ - subscription │  │ - encrypted    │  │ - agent_id                  │ │
│  │ - settings     │  │ - tenant_id    │  │ - adapter_tool_id           │ │
│  └────────────────┘  │ - cred_type    │  │ - enabled                   │ │
│                      └────────────────┘  └─────────────────────────────┘ │
│  ┌────────────────┐                                                      │
│  │ agents         │  ┌─────────────────────────────────────────────────┐ │
│  │ - tenant_id    │  │ tenant_tool_config                              │ │
│  │ - skills       │  │ - tenant_id                                     │ │
│  │ - config       │  │ - adapter_tool_id                               │ │
│  └────────────────┘  │ - enabled                                       │ │
│                      │ - credential_id (1:1 with tool)                 │ │
│                      └─────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Model

### 4.1 Data Ownership

| Data | Owner | Description |
|------|-------|-------------|
| Tool definitions | **Adapter** | Name, description, capabilities, OpenAPI URL, auth config |
| Tool enablement | **Automatos** | Which tools are enabled for which tenant |
| Tool:Credential mapping | **Automatos** | 1:1 mapping when tool is enabled |
| Agent-tool assignments | **Automatos** | Which agents can use which tools |
| Credentials | **Automatos** | Encrypted storage for hosted mode |

### 4.2 Adapter Data Model (Source of Truth)

```sql
-- ADAPTER DATABASE (adapter_tools)
CREATE TABLE adapter_tools (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    provider VARCHAR(100) NOT NULL,
    category VARCHAR(100) NOT NULL,
    adapter_type VARCHAR(50) NOT NULL,  -- 'rest' or 'mcp'
    enabled BOOLEAN DEFAULT TRUE,
    
    -- REST passthrough config
    openapi_url TEXT,
    base_url TEXT,
    operation_ids JSONB,  -- Allowed operations
    
    -- MCP config
    mcp_server_url TEXT,
    
    -- Auth config
    auth_config JSONB,  -- { type: "api_key"|"bearer"|"oauth2", ... }
    credential_type VARCHAR(100),  -- Maps to Automatos credential_types
    
    -- Metadata
    tags JSONB,
    metadata JSONB,
    created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ
);
```

### 4.3 Automatos Data Model (Enablement & Assignment)

```sql
-- AUTOMATOS DATABASE

-- Tool enablement per tenant (with 1:1 credential mapping)
CREATE TABLE tenant_tool_config (
    id SERIAL PRIMARY KEY,
    tenant_id UUID NOT NULL,
    adapter_tool_id VARCHAR(255) NOT NULL,  -- Reference to Adapter's tool name
    adapter_tool_name VARCHAR(255) NOT NULL, -- Cached for display
    enabled BOOLEAN DEFAULT TRUE,
    credential_id INTEGER REFERENCES credentials(id),  -- 1:1 mapping
    configuration JSONB DEFAULT '{}',  -- Tenant-specific overrides
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(tenant_id, adapter_tool_id)
);

-- Agent-tool assignments (like skills, 1:many)
CREATE TABLE agent_tool_assignments (
    id SERIAL PRIMARY KEY,
    agent_id INTEGER NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    adapter_tool_id VARCHAR(255) NOT NULL,  -- Reference to Adapter's tool
    enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(agent_id, adapter_tool_id)
);

-- Index for fast lookups
CREATE INDEX idx_tenant_tool_config_tenant ON tenant_tool_config(tenant_id);
CREATE INDEX idx_agent_tool_assignments_agent ON agent_tool_assignments(agent_id);
```

---

## 5. Execution Flows

### 5.1 Tool Discovery Flow

```
1. Adapter exposes MCP tools/list endpoint
2. Context Forge registers Adapter as a gateway
3. Context Forge discovers tools via MCP protocol
4. Automatos UI queries Adapter API for available tools
5. User enables tool → creates tenant_tool_config with credential
6. User assigns tool to agent → creates agent_tool_assignment
```

### 5.2 Tool Execution (Hosted Mode - Automatos)

```
1. Agent requests tool execution
   └─▶ UnifiedToolExecutor.execute_tool("github_repos_list", params)

2. Validate agent has tool assignment
   └─▶ Check agent_tool_assignments table

3. Get tenant's tool config
   └─▶ Get credential_id from tenant_tool_config

4. Build MCP request
   └─▶ MCPToolExecutor builds JSON-RPC payload
   └─▶ Include: tenant_id, credential_mode: "hosted"

5. Send to Context Forge
   └─▶ POST mcp.automatos.app/mcp

6. Context Forge routes to Adapter

7. Adapter sees credential_mode: "hosted"
   └─▶ Calls back to Automatos: POST /api/credentials/resolve
   └─▶ Automatos returns decrypted credential data

8. Adapter executes tool against SaaS API
   └─▶ Injects credentials into REST/MCP request

9. Result returns through chain
   └─▶ Adapter → CF → Automatos → Agent
```

### 5.3 Tool Execution (BYO Mode - Widget/Third-Party)

```
1. Widget/Third-party sends request to Context Forge
   └─▶ POST mcp.automatos.app/mcp
   └─▶ Include: tenant_id, credential_mode: "byo", credentials: {...}

2. Context Forge validates JWT (tenant auth)

3. Context Forge routes to Adapter

4. Adapter sees credential_mode: "byo"
   └─▶ Uses credentials directly from request payload

5. Adapter executes tool against SaaS API

6. Result returns to caller
```

### 5.4 Credential Resolution Callback (Hosted Mode)

```
Adapter calls Automatos API:

POST /api/credentials/resolve
Authorization: Bearer <service-token>
Content-Type: application/json

{
    "tenant_id": "uuid",
    "tool_name": "github",
    "service_name": "unified-adapter"
}

Response:
{
    "success": true,
    "data": {
        "api_key": "ghp_xxx...",
        "base_url": "https://api.github.com"
    }
}
```

---

## 6. API Contracts

### 6.1 Adapter APIs

```yaml
# Tool Discovery
GET /admin/tools
Response: [{ id, name, description, provider, category, adapter_type, auth_config, ... }]

# Tool Execution (via MCP)
POST /mcp
Content-Type: application/json
{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
        "name": "github_repos_list",
        "arguments": { "org": "automatos-ai" }
    },
    "id": "call-1",
    "meta": {
        "tenant_id": "uuid",
        "credential_mode": "hosted" | "byo",
        "credentials": { ... }  // Only for BYO mode
    }
}
```

### 6.2 Automatos APIs

```yaml
# Credential Resolution (called by Adapter)
POST /api/credentials/resolve
Authorization: Bearer <service-token>
{
    "tenant_id": "uuid",
    "tool_name": "github",
    "service_name": "unified-adapter"
}
Response: { "success": true, "data": { "api_key": "..." } }

# Tool Enablement
POST /api/tools/enable
{
    "adapter_tool_id": "github",
    "credential_id": 123
}

# Agent Tool Assignment
POST /api/agents/{agent_id}/tools
{
    "adapter_tool_id": "github",
    "enabled": true
}

# Get Agent's Available Tools
GET /api/agents/{agent_id}/tools
Response: [{ adapter_tool_id, name, enabled, ... }]
```

### 6.3 Context Forge APIs

```yaml
# MCP Gateway (routes to Adapter)
POST /mcp
# Standard MCP JSON-RPC format

# Tool Discovery
GET /tools
Response: [{ name, description, inputSchema, ... }]

# Health Check
GET /health
Response: { "status": "ok" }
```

---

## 7. Tool Enablement & Assignment Flow

### 7.1 User Enables a Tool (Settings > Tools)

```
1. User opens Tools page in Automatos UI
2. UI fetches available tools from Adapter: GET /admin/tools
3. User searches/selects a tool (e.g., GitHub)
4. UI shows credential form based on tool.credential_type
5. User enters credentials (API key, OAuth token, etc.)
6. Submit:
   - Create/update credential in credentials table
   - Create tenant_tool_config with credential_id
   - Tool is now "enabled" for tenant
```

### 7.2 User Assigns Tool to Agent (Agent Config)

```
1. User opens Agent configuration modal
2. UI shows "Tools" section (similar to Skills)
3. UI fetches tenant's enabled tools: GET /api/tenant/tools
4. User toggles tools on/off for this agent
5. Submit:
   - Create/update agent_tool_assignments
   - Agent can now use assigned tools
```

### 7.3 Runtime: Get Tools for Agent

```python
async def get_tools_for_agent(agent_id: int, tenant_id: UUID) -> List[Tool]:
    """
    Get all tools available to an agent.
    
    Requirements:
    1. Tool must be enabled for tenant (tenant_tool_config.enabled = true)
    2. Tool must be assigned to agent (agent_tool_assignments.enabled = true)
    """
    # Get agent's tool assignments
    assignments = db.query(AgentToolAssignment).filter(
        AgentToolAssignment.agent_id == agent_id,
        AgentToolAssignment.enabled == True
    ).all()
    
    assigned_tool_ids = {a.adapter_tool_id for a in assignments}
    
    # Get tenant's enabled tools (with credentials)
    tenant_tools = db.query(TenantToolConfig).filter(
        TenantToolConfig.tenant_id == tenant_id,
        TenantToolConfig.enabled == True,
        TenantToolConfig.adapter_tool_id.in_(assigned_tool_ids)
    ).all()
    
    # Fetch tool definitions from Adapter
    adapter_client = AdapterClient()
    available_tools = []
    
    for tenant_tool in tenant_tools:
        tool_def = await adapter_client.get_tool(tenant_tool.adapter_tool_id)
        available_tools.append({
            "tool": tool_def,
            "credential_id": tenant_tool.credential_id,
            "tenant_tool_config_id": tenant_tool.id
        })
    
    return available_tools
```

---

## 8. Migration Plan

### Phase 1: Data Model Updates
- [ ] Add `tenant_tool_config` table to Automatos
- [ ] Add `agent_tool_assignments` table to Automatos
- [ ] Migrate existing `mcp_tools` data to reference Adapter tools

### Phase 2: API Implementation
- [ ] Implement credential resolution callback endpoint
- [ ] Update Adapter to call back for hosted credentials
- [ ] Add tool enablement API endpoints
- [ ] Add agent assignment API endpoints

### Phase 3: UI Updates
- [ ] Update Tools settings page to enable tools with credentials
- [ ] Add tool assignment section to Agent configuration modal
- [ ] Update ToolRegistry to use new data model

### Phase 4: Runtime Integration
- [ ] Update UnifiedToolExecutor to validate assignments
- [ ] Update MCPToolExecutor to pass credential_mode
- [ ] End-to-end testing with hosted and BYO modes

---

## 9. Security Considerations

### Credential Security
- Credentials stored encrypted in Automatos (existing)
- BYO credentials never stored - used and discarded
- Credential resolution requires service token authentication
- No credentials in logs (redaction in Adapter)

### Access Control
- Tenant isolation via tenant_id in all queries
- Agent can only use explicitly assigned tools
- Tool must be enabled at tenant level first
- JWT authentication at Context Forge gateway

---

## 10. Success Metrics

- **SM-1:** 100% of tool executions route through CF → Adapter
- **SM-2:** Agent-tool assignments enforced at runtime
- **SM-3:** Credential resolution callback latency < 100ms
- **SM-4:** Zero credential leakage in logs
- **SM-5:** Widget BYO mode functional end-to-end

---

## 11. Open Questions (Resolved)

| Question | Resolution |
|----------|------------|
| Should Automatos have a local tool table? | YES - for enablement/assignment only, NOT definitions |
| Who resolves hosted credentials? | Adapter calls back to Automatos API |
| Widget/Third-party auth? | JWT with tenant_id, BYO credentials |
| Tool:Credential mapping? | 1:1 at tool enablement time (tenant_tool_config) |
| Agent:Tool mapping? | 1:many (agent_tool_assignments), like skills |

---

## 12. References

- **PRD-33:** MCP Gateway Integration (Context Forge)
- **PRD-34:** Unified Integrations Adapter
- **Code:** `automatos-ai/orchestrator/modules/tools/`
- **Code:** `automatos-unified-adapter/src/unified_adapter/`
