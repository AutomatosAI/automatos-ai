# PRD-35: Technical Specification

**Version:** 1.0  
**Status:** 🟡 Design Phase  
**Parent:** PRD-35 (Tool Catalog & Registry Architecture)  
**Last Updated:** 2026-01-17

---

## Part 1: API Contracts

### 1.1 Automatos Credential Resolution API

This API is called by the Unified Adapter to resolve credentials for hosted mode executions.

#### Endpoint: POST /api/credentials/resolve

**Purpose:** Resolve and return decrypted credentials for a tool execution.

**Authentication:** Service token (Bearer token issued to Adapter)

**Request:**
```json
{
    "tenant_id": "550e8400-e29b-41d4-a716-446655440000",
    "tool_name": "github",
    "credential_type": "github_api",  // Optional, can resolve by tool_name
    "service_name": "unified-adapter",
    "environment": "production"  // Optional, defaults to "production"
}
```

**Response (Success):**
```json
{
    "success": true,
    "data": {
        "api_key": "ghp_xxxxxxxxxxxxxxxxxxxx",
        "base_url": "https://api.github.com"
    },
    "meta": {
        "credential_id": 123,
        "credential_type": "github_api",
        "environment": "production",
        "resolved_at": "2026-01-17T10:30:00Z"
    }
}
```

**Response (Not Found):**
```json
{
    "success": false,
    "error": {
        "code": "CREDENTIAL_NOT_FOUND",
        "message": "No credential found for tool 'github' in tenant"
    }
}
```

**Response (Unauthorized):**
```json
{
    "success": false,
    "error": {
        "code": "UNAUTHORIZED",
        "message": "Invalid service token"
    }
}
```

**Implementation Notes:**
- Lookup order: `tenant_tool_config.credential_id` → `credential_types` fallback
- Credentials are decrypted server-side and returned
- Audit log entry created for each resolution
- Rate limiting: 100 req/min per service_name

---

### 1.2 Automatos Tool Enablement APIs

#### Endpoint: GET /api/tools/available

**Purpose:** List all tools available from Adapter (for enablement UI).

**Authentication:** User JWT (tenant context)

**Response:**
```json
{
    "data": [
        {
            "adapter_tool_id": "github",
            "name": "GitHub",
            "description": "GitHub repository management",
            "provider": "github",
            "category": "dev",
            "credential_type": "github_api",
            "auth_config": {
                "type": "api_key",
                "fields": [
                    { "name": "api_key", "label": "Personal Access Token", "type": "password", "required": true }
                ]
            },
            "icon": "github.svg",
            "enabled_for_tenant": true,  // From tenant_tool_config
            "credential_configured": true
        }
    ],
    "pagination": {
        "total": 25,
        "page": 1,
        "per_page": 50
    }
}
```

---

#### Endpoint: POST /api/tools/enable

**Purpose:** Enable a tool for the tenant with credentials.

**Authentication:** User JWT (tenant context, admin role)

**Request:**
```json
{
    "adapter_tool_id": "github",
    "credentials": {
        "api_key": "ghp_xxxxxxxxxxxxxxxxxxxx"
    },
    "configuration": {
        "default_org": "automatos-ai"  // Optional tool-specific config
    }
}
```

**Response:**
```json
{
    "success": true,
    "data": {
        "id": 1,
        "tenant_id": "550e8400-e29b-41d4-a716-446655440000",
        "adapter_tool_id": "github",
        "adapter_tool_name": "GitHub",
        "enabled": true,
        "credential_id": 123,
        "configuration": { "default_org": "automatos-ai" },
        "created_at": "2026-01-17T10:30:00Z"
    }
}
```

**Logic:**
1. Validate adapter_tool_id exists in Adapter
2. Create or update credential in `credentials` table
3. Create or update `tenant_tool_config` with credential_id
4. Return success

---

#### Endpoint: DELETE /api/tools/{adapter_tool_id}/disable

**Purpose:** Disable a tool for the tenant.

**Authentication:** User JWT (tenant context, admin role)

**Response:**
```json
{
    "success": true,
    "message": "Tool 'github' disabled for tenant"
}
```

**Logic:**
1. Set `tenant_tool_config.enabled = false`
2. Optionally: Remove agent assignments (or leave orphaned)
3. Do NOT delete credentials (user may re-enable)

---

### 1.3 Agent Tool Assignment APIs

#### Endpoint: GET /api/agents/{agent_id}/tools

**Purpose:** Get all tools assigned to an agent.

**Authentication:** User JWT (tenant context)

**Response:**
```json
{
    "data": [
        {
            "id": 1,
            "adapter_tool_id": "github",
            "name": "GitHub",
            "description": "GitHub repository management",
            "category": "dev",
            "icon": "github.svg",
            "enabled": true,
            "assigned_at": "2026-01-17T10:30:00Z"
        },
        {
            "id": 2,
            "adapter_tool_id": "slack",
            "name": "Slack",
            "description": "Slack messaging",
            "category": "messaging",
            "icon": "slack.svg",
            "enabled": true,
            "assigned_at": "2026-01-17T10:35:00Z"
        }
    ]
}
```

---

#### Endpoint: POST /api/agents/{agent_id}/tools

**Purpose:** Assign a tool to an agent.

**Authentication:** User JWT (tenant context)

**Request:**
```json
{
    "adapter_tool_id": "github",
    "enabled": true
}
```

**Response:**
```json
{
    "success": true,
    "data": {
        "id": 1,
        "agent_id": 14,
        "adapter_tool_id": "github",
        "enabled": true,
        "created_at": "2026-01-17T10:30:00Z"
    }
}
```

**Validation:**
1. Tool must be enabled for tenant (`tenant_tool_config.enabled = true`)
2. Agent must belong to same tenant
3. Idempotent: update if assignment exists

---

#### Endpoint: DELETE /api/agents/{agent_id}/tools/{adapter_tool_id}

**Purpose:** Remove a tool assignment from an agent.

**Authentication:** User JWT (tenant context)

**Response:**
```json
{
    "success": true,
    "message": "Tool 'github' removed from agent 14"
}
```

---

#### Endpoint: PUT /api/agents/{agent_id}/tools/batch

**Purpose:** Batch update tool assignments (for UI toggles).

**Authentication:** User JWT (tenant context)

**Request:**
```json
{
    "assignments": [
        { "adapter_tool_id": "github", "enabled": true },
        { "adapter_tool_id": "slack", "enabled": true },
        { "adapter_tool_id": "notion", "enabled": false }
    ]
}
```

**Response:**
```json
{
    "success": true,
    "data": {
        "updated": 3,
        "assignments": [
            { "adapter_tool_id": "github", "enabled": true },
            { "adapter_tool_id": "slack", "enabled": true },
            { "adapter_tool_id": "notion", "enabled": false }
        ]
    }
}
```

---

### 1.4 Adapter MCP Execution API

#### Endpoint: POST /mcp (on Adapter, via Context Forge)

**Purpose:** Execute a tool via MCP protocol.

**Request (Hosted Mode):**
```json
{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
        "name": "mcp_github_repos_list",
        "arguments": {
            "org": "automatos-ai"
        }
    },
    "id": "call-123",
    "meta": {
        "tenant_id": "550e8400-e29b-41d4-a716-446655440000",
        "credential_mode": "hosted"
    }
}
```

**Request (BYO Mode):**
```json
{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
        "name": "mcp_github_repos_list",
        "arguments": {
            "org": "automatos-ai"
        }
    },
    "id": "call-123",
    "meta": {
        "tenant_id": "550e8400-e29b-41d4-a716-446655440000",
        "credential_mode": "byo",
        "credentials": {
            "api_key": "ghp_xxxxxxxxxxxxxxxxxxxx"
        }
    }
}
```

**Response (Success):**
```json
{
    "jsonrpc": "2.0",
    "id": "call-123",
    "result": {
        "content": [
            {
                "type": "json",
                "json": {
                    "repositories": [
                        { "name": "automatos-ai", "full_name": "automatos-ai/automatos-ai" }
                    ]
                }
            }
        ]
    }
}
```

**Response (Error):**
```json
{
    "jsonrpc": "2.0",
    "id": "call-123",
    "result": {
        "content": [
            { "type": "text", "text": "Credential resolution failed: No credential found" }
        ],
        "is_error": true
    }
}
```

---

## Part 2: Agent-Tool Assignment Flow

### 2.1 UI Flow: Tools Settings Page

**Location:** Settings > Tools (existing page, enhanced)

**User Journey:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│  TOOLS SETTINGS PAGE                                                     │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │  Search: [_________________________] [Category: All ▼]              ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │  ENABLED TOOLS (3)                                                  ││
│  │                                                                      ││
│  │  ┌──────────────────────┐ ┌──────────────────────┐                  ││
│  │  │ 🐙 GitHub            │ │ 💬 Slack             │                  ││
│  │  │ Repository mgmt      │ │ Team messaging       │                  ││
│  │  │ [✓ Enabled] [Config] │ │ [✓ Enabled] [Config] │                  ││
│  │  └──────────────────────┘ └──────────────────────┘                  ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │  AVAILABLE TOOLS (22)                                               ││
│  │                                                                      ││
│  │  ┌──────────────────────┐ ┌──────────────────────┐                  ││
│  │  │ 📝 Notion            │ │ 📧 Gmail             │                  ││
│  │  │ Workspace docs       │ │ Email management     │                  ││
│  │  │ [+ Enable]           │ │ [+ Enable]           │                  ││
│  │  └──────────────────────┘ └──────────────────────┘                  ││
│  └─────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
```

**Enable Tool Modal:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ENABLE GITHUB                                                     [X]  │
│─────────────────────────────────────────────────────────────────────────│
│                                                                          │
│  🐙 GitHub                                                               │
│  Repository management, issues, pull requests                            │
│                                                                          │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  CREDENTIALS                                                             │
│                                                                          │
│  Personal Access Token *                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ ghp_xxxxxxxxxxxxxxxxxxxx                                            ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                                                          │
│  ℹ️ Create a token at github.com/settings/tokens with repo scope        │
│                                                                          │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  CONFIGURATION (Optional)                                                │
│                                                                          │
│  Default Organization                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ automatos-ai                                                        ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                                                          │
│                                          [Cancel]  [Test & Enable]       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### 2.2 UI Flow: Agent Configuration

**Location:** Agent modal > Tools tab (new section, similar to Skills)

**User Journey:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│  EDIT AGENT: Research Assistant                                    [X]  │
│─────────────────────────────────────────────────────────────────────────│
│  [General] [Skills] [Tools] [Memory] [Settings]                         │
│─────────────────────────────────────────────────────────────────────────│
│                                                                          │
│  TOOLS TAB                                                               │
│                                                                          │
│  Assign tools this agent can use. Tools must be enabled in Settings     │
│  first.                                                                  │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │  ASSIGNED TOOLS                                                      ││
│  │                                                                      ││
│  │  ┌────────────────────────────────────────────────────────────────┐ ││
│  │  │ [✓] 🐙 GitHub         Repository management              [−]  │ ││
│  │  │ [✓] 💬 Slack          Team messaging                     [−]  │ ││
│  │  │ [ ] 📧 Gmail          Email management                   [+]  │ ││
│  │  └────────────────────────────────────────────────────────────────┘ ││
│  │                                                                      ││
│  │  ℹ️ Only tools enabled for your organization are shown              ││
│  │                                                                      ││
│  │  Don't see a tool? [Enable in Settings →]                           ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                                                          │
│                                          [Cancel]  [Save Changes]        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### 2.3 Runtime Flow: Tool Execution

```python
# automatos-ai/orchestrator/modules/tools/services/tool_access_service.py

class ToolAccessService:
    """Service for managing tool access and execution."""
    
    def __init__(self, db: Session, adapter_client: AdapterClient):
        self.db = db
        self.adapter_client = adapter_client
    
    async def get_agent_tools(
        self,
        agent_id: int,
        tenant_id: UUID
    ) -> List[AgentTool]:
        """
        Get all tools available to an agent.
        
        Returns tools that are:
        1. Enabled for tenant (tenant_tool_config.enabled = true)
        2. Assigned to agent (agent_tool_assignments.enabled = true)
        """
        # Get agent's assignments
        assignments = self.db.query(AgentToolAssignment).filter(
            AgentToolAssignment.agent_id == agent_id,
            AgentToolAssignment.enabled == True
        ).all()
        
        if not assignments:
            return []
        
        assigned_tool_ids = [a.adapter_tool_id for a in assignments]
        
        # Get tenant's enabled tools that are assigned
        tenant_tools = self.db.query(TenantToolConfig).filter(
            TenantToolConfig.tenant_id == tenant_id,
            TenantToolConfig.enabled == True,
            TenantToolConfig.adapter_tool_id.in_(assigned_tool_ids)
        ).all()
        
        # Fetch tool definitions from Adapter
        result = []
        for tenant_tool in tenant_tools:
            try:
                tool_def = await self.adapter_client.get_tool(
                    tenant_tool.adapter_tool_id
                )
                result.append(AgentTool(
                    adapter_tool_id=tenant_tool.adapter_tool_id,
                    name=tool_def["name"],
                    description=tool_def["description"],
                    category=tool_def["category"],
                    credential_id=tenant_tool.credential_id,
                    methods=tool_def.get("capabilities", {}).get("methods", [])
                ))
            except Exception as e:
                logger.warning(f"Failed to fetch tool {tenant_tool.adapter_tool_id}: {e}")
        
        return result
    
    async def validate_tool_access(
        self,
        agent_id: int,
        tenant_id: UUID,
        adapter_tool_id: str
    ) -> Tuple[bool, Optional[str], Optional[int]]:
        """
        Validate if an agent can use a tool.
        
        Returns:
            (has_access, error_message, credential_id)
        """
        # Check agent assignment
        assignment = self.db.query(AgentToolAssignment).filter(
            AgentToolAssignment.agent_id == agent_id,
            AgentToolAssignment.adapter_tool_id == adapter_tool_id,
            AgentToolAssignment.enabled == True
        ).first()
        
        if not assignment:
            return False, f"Tool '{adapter_tool_id}' not assigned to agent", None
        
        # Check tenant enablement
        tenant_config = self.db.query(TenantToolConfig).filter(
            TenantToolConfig.tenant_id == tenant_id,
            TenantToolConfig.adapter_tool_id == adapter_tool_id,
            TenantToolConfig.enabled == True
        ).first()
        
        if not tenant_config:
            return False, f"Tool '{adapter_tool_id}' not enabled for tenant", None
        
        if not tenant_config.credential_id:
            return False, f"Tool '{adapter_tool_id}' has no configured credential", None
        
        return True, None, tenant_config.credential_id
    
    async def execute_tool(
        self,
        agent_id: int,
        tenant_id: UUID,
        adapter_tool_id: str,
        method: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a tool with access validation.
        """
        # Validate access
        has_access, error, credential_id = await self.validate_tool_access(
            agent_id, tenant_id, adapter_tool_id
        )
        
        if not has_access:
            return {
                "success": False,
                "error": error,
                "tool": adapter_tool_id
            }
        
        # Build MCP request
        tool_name = f"mcp_{adapter_tool_id}_{method}"
        
        request_payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": params
            },
            "id": str(uuid.uuid4()),
            "meta": {
                "tenant_id": str(tenant_id),
                "credential_mode": "hosted"
            }
        }
        
        # Execute via Context Forge
        result = await self.context_forge_client.execute(request_payload)
        
        return result
```

---

### 2.4 Integration with UnifiedToolExecutor

```python
# Update to automatos-ai/orchestrator/modules/tools/execution/unified_executor.py

class UnifiedToolExecutor:
    def __init__(self, db_session: Session, ...):
        # ... existing init ...
        self._tool_access_service = None
    
    @property
    def tool_access_service(self):
        if self._tool_access_service is None:
            from modules.tools.services.tool_access_service import ToolAccessService
            self._tool_access_service = ToolAccessService(
                self.db,
                AdapterClient()
            )
        return self._tool_access_service
    
    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        agent_id: int = 0,
        tenant_id: UUID = None
    ) -> Dict[str, Any]:
        """Execute a tool with proper access control."""
        
        # Check if this is an MCP/external tool
        if tool_name.startswith("mcp_"):
            # Parse tool name: mcp_{adapter_tool_id}_{method}
            parts = tool_name.split("_", 2)
            if len(parts) >= 3:
                adapter_tool_id = parts[1]
                method = parts[2]
                
                # Execute via ToolAccessService (validates assignment)
                return await self.tool_access_service.execute_tool(
                    agent_id=agent_id,
                    tenant_id=tenant_id,
                    adapter_tool_id=adapter_tool_id,
                    method=method,
                    params=parameters
                )
        
        # ... existing routing for internal tools ...
```

---

### 2.5 Updated ToolRegistry

```python
# Update to automatos-ai/orchestrator/modules/tools/registry/tool_registry.py

class ToolRegistry:
    """
    Tool registry that combines:
    - Core platform tools (in-memory, always available)
    - External tools (from Adapter, filtered by agent assignment)
    """
    
    def __init__(self, db_session: Optional[Session] = None):
        self.db = db_session
        self.tools: Dict[str, ToolSpec] = {}
        self._adapter_client = None
        
        # Register core platform tools
        self._register_core_tools()
    
    @property
    def adapter_client(self):
        if self._adapter_client is None:
            from modules.tools.services.adapter_client import AdapterClient
            self._adapter_client = AdapterClient()
        return self._adapter_client
    
    async def get_tools_for_agent(
        self,
        agent_id: int,
        tenant_id: UUID
    ) -> List[ToolSpec]:
        """
        Get all tools available to an agent.
        
        Combines:
        - Core platform tools (always available)
        - External tools (from Adapter, filtered by assignment)
        """
        # Start with core tools
        available_tools = list(self.tools.values())
        
        # Add external tools from Adapter (filtered by assignment)
        tool_access = ToolAccessService(self.db, self.adapter_client)
        external_tools = await tool_access.get_agent_tools(agent_id, tenant_id)
        
        for ext_tool in external_tools:
            # Convert to ToolSpec
            for method in ext_tool.methods:
                tool_name = f"mcp_{ext_tool.adapter_tool_id}_{method}"
                available_tools.append(ToolSpec(
                    name=tool_name,
                    category=ToolCategory.MCP_TOOLS,
                    description=f"{ext_tool.description} - Method: {method}",
                    executor_class="MCPToolExecutor",
                    executor_method="execute_tool",
                    parameters=[
                        ToolParameter(
                            name="params",
                            type="object",
                            description=f"Parameters for {method}",
                            required=False,
                            default={}
                        )
                    ],
                    metadata={
                        "adapter_tool_id": ext_tool.adapter_tool_id,
                        "method": method,
                        "credential_id": ext_tool.credential_id
                    }
                ))
        
        return available_tools
```

---

## Part 3: Database Migrations

### Migration: Add Tool Assignment Tables

```python
# automatos-ai/orchestrator/alembic/versions/xxx_add_tool_assignment_tables.py

"""Add tenant_tool_config and agent_tool_assignments tables

Revision ID: xxx
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

def upgrade():
    # Tenant tool configuration (1:1 with credential)
    op.create_table(
        'tenant_tool_config',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('tenant_id', UUID(as_uuid=True), nullable=False),
        sa.Column('adapter_tool_id', sa.String(255), nullable=False),
        sa.Column('adapter_tool_name', sa.String(255), nullable=False),
        sa.Column('enabled', sa.Boolean(), default=True),
        sa.Column('credential_id', sa.Integer(), sa.ForeignKey('credentials.id')),
        sa.Column('configuration', JSONB, default={}),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.UniqueConstraint('tenant_id', 'adapter_tool_id', name='uq_tenant_tool_config')
    )
    op.create_index('idx_tenant_tool_config_tenant', 'tenant_tool_config', ['tenant_id'])
    
    # Agent tool assignments
    op.create_table(
        'agent_tool_assignments',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('agent_id', sa.Integer(), sa.ForeignKey('agents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('adapter_tool_id', sa.String(255), nullable=False),
        sa.Column('enabled', sa.Boolean(), default=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.UniqueConstraint('agent_id', 'adapter_tool_id', name='uq_agent_tool_assignment')
    )
    op.create_index('idx_agent_tool_assignments_agent', 'agent_tool_assignments', ['agent_id'])

def downgrade():
    op.drop_table('agent_tool_assignments')
    op.drop_table('tenant_tool_config')
```

---

## Part 4: Summary Checklist

### API Implementation
- [x] `POST /api/credentials/resolve` - Credential callback for Adapter ✅ (BYO mode working)
- [x] `GET /api/mcp-tools/` - List tools from database ✅
- [x] `POST /api/credentials/` - Create credential ✅
- [x] `PUT /api/credentials/{id}` - Update credential ✅
- [ ] `GET /api/tools/available` - List tools from Adapter (sync with Adapter catalog)
- [ ] `POST /api/tools/enable` - Enable tool with credentials (unified flow)
- [ ] `DELETE /api/tools/{id}/disable` - Disable tool
- [x] `GET /api/agents/{id}/tools` - Get agent's tools ✅
- [x] `POST /api/agents/{id}/tools` - Assign tool to agent ✅
- [x] `DELETE /api/agents/{id}/tools/{tool_id}` - Remove assignment ✅
- [ ] `PUT /api/agents/{id}/tools/batch` - Batch update assignments

### Database
- [x] `credentials` table - Working ✅
- [x] `credential_types` table - 400+ types seeded ✅
- [x] `mcp_tools` table - Working ✅
- [x] `agent_tool_assignments` table - Working ✅
- [ ] `tenant_tool_config` table - For multi-tenant tool enablement
- [ ] Add indexes for performance (verify existing)

### Backend Services
- [x] `UnifiedToolExecutor` - Routes MCP tools correctly ✅ (fixed operation parameter handling)
- [x] `MCPToolExecutor` - Executes via Unified Adapter ✅
- [x] `RestExecutor` (Adapter) - POST body handling fixed ✅
- [x] `CredentialStore` - Create/update/encrypt working ✅
- [ ] `ToolAccessService` - Tool access validation and execution
- [ ] Update `ToolRegistry` - Combine core + external tools

### Frontend
- [x] Tools catalog page - Browse/search tools ✅
- [x] Tool config modal - Configure credentials ✅ (credential update bug fixed)
- [x] Credential management UI - Working ✅
- [x] DynamicCredentialForm - Dynamic form generation ✅
- [ ] Tools Settings page - Enable tools with credentials (unified flow)
- [ ] Agent modal - Tools tab for assignment UI polish

### Adapter Updates
- [x] REST passthrough working ✅
- [x] OpenAPI spec loading ✅
- [x] POST body parameters (form-urlencoded) ✅
- [x] Slack API error detection (ok: false) ✅
- [ ] Support `credential_mode` in MCP request meta (for hosted mode)
- [ ] Implement credential callback to Automatos (for hosted mode)

### Testing
- [x] E2E test: Slack `chat.postMessage` working ✅
- [ ] Unit tests for ToolAccessService
- [ ] Integration tests for credential callback
- [ ] E2E tests for more tools (GitHub, Notion, etc.)

---

## Part 5: Next Steps - Bulk Tool Registration

### Phase 1: Tool Catalog Population (Priority: HIGH)

**Goal:** Register 50+ tools in the catalog with proper metadata and credential mappings.

#### 1.1 Tool Categories to Add:
| Category | Example Tools | Credential Type |
|----------|--------------|-----------------|
| **Communication** | Slack, Discord, Telegram, Twilio | OAuth/API Key |
| **DevOps** | GitHub, GitLab, Jira, Linear | OAuth/PAT |
| **CRM** | Salesforce, HubSpot, Pipedrive | OAuth |
| **Productivity** | Notion, Google Docs, Airtable | OAuth |
| **Database** | PostgreSQL, MongoDB, Supabase | Connection String |
| **AI/ML** | OpenAI, Anthropic, HuggingFace | API Key |
| **Analytics** | Google Analytics, Mixpanel | OAuth/API Key |
| **Storage** | S3, GCS, Cloudflare R2 | Access Key |

#### 1.2 Bulk Registration Script
```bash
# Run tool registration from OpenAPI specs
python scripts/register_tools_from_openapi.py \
  --spec-dir ./tool-specs/ \
  --adapter-url https://adapter.automatos.app
```

#### 1.3 Registration Data Structure
```json
{
  "tool_name": "github",
  "display_name": "GitHub",
  "provider": "github",
  "category": "devops",
  "icon": "github.svg",
  "credential_type": "github_api",
  "openapi_spec_url": "https://raw.githubusercontent.com/.../github-openapi.json",
  "operations": ["repos_list", "issues_create", "pr_merge", ...],
  "metadata": {
    "documentation": "https://docs.github.com/en/rest",
    "rate_limits": {"requests_per_hour": 5000}
  }
}
```

### Phase 2: Credential Type Expansion

**Current:** 400+ credential types seeded (most are placeholders)

**Action Items:**
1. Verify schema_definition for top 20 tools
2. Add OAuth flow support for tools requiring it
3. Add validation patterns (e.g., GitHub PAT format: `ghp_*`)
4. Add test_endpoint configurations for credential testing

### Phase 3: Hosted Credential Mode

**Goal:** Enable Automatos to store credentials and resolve them on demand.

**Implementation:**
1. Add `credential_mode` parameter to tool execution
2. Implement callback API from Adapter → Automatos
3. Add tenant_tool_config for multi-tenant credential isolation
4. Add credential rotation and expiry handling

### Phase 4: Tool Discovery & Recommendation

**Goal:** Smart tool suggestions based on agent type and task.

**Features:**
- "You might also need" suggestions
- Tool bundles (e.g., "DevOps Bundle" = GitHub + Jira + Slack)
- Usage analytics for tool recommendations

---

## Appendix: Bugs Fixed in This Iteration

### Bug 1: Operation Parameter Not Passed to Adapter
**File:** `unified_executor.py`  
**Issue:** `_execute_mcp_tool` was looking for `method` key, but tool calls used `operation`  
**Fix:** Support both `method` and `operation` keys, with fallback logic

### Bug 2: POST Parameters Sent as Query String
**File:** `executors.py` (Unified Adapter)  
**Issue:** RestExecutor sent all params as query params, but Slack API requires body  
**Fix:** Detect POST/PUT/PATCH and send remaining params in request body (form-urlencoded)

### Bug 3: Slack API Errors Not Detected
**File:** `executors.py` (Unified Adapter)  
**Issue:** Slack returns 200 OK with `ok: false` in body for errors  
**Fix:** Check for `ok: false` in JSON response and log warning

### Bug 4: Credential Update Creating Duplicates
**File:** `tool-config-modal.tsx`  
**Issue:** Config modal always called create API, not update  
**Fix:** Look up existing credentials by type and pass `credentialId` for updates

### Bug 5: Query Params Not Sent to Credentials API
**File:** `credentials.ts`  
**Issue:** `listCredentials` passed `query` in options, but apiClient ignored it  
**Fix:** Build query params directly into URL string
