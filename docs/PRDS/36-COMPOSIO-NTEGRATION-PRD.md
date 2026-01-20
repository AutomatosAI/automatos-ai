# Composio Integration PRD - Automatos

**Version:** 1.0  
**Status:** 🟡 Planning Phase  
**Date:** January 20, 2026  
**Author:** DeepAgent

---

## Executive Summary

This PRD outlines the integration of **Composio** (500+ tools) and **AIML API** (400+ LLMs) into Automatos, creating a truly provider-agnostic unified gateway system. The integration will:

1. **Replace** custom MCP tool execution with Composio's managed tool infrastructure
2. **Add** AIML API as the primary LLM gateway for 400+ models
3. **Redesign** the Tools page to match the "Manage Apps" UI pattern (see screenshot)
4. **Simplify** OAuth management by leveraging Composio's built-in authentication

### Key Benefits
- **500+ pre-built tools** via Composio (vs. current ~160k individual methods)
- **400+ LLMs** via AIML API (vs. current 7 provider-specific clients)
- **Simplified OAuth** - Composio handles auth flows for all integrations
- **Reduced maintenance** - Tool definitions managed by Composio
- **Faster time-to-value** - New integrations available immediately

---

## Table of Contents

1. [Current State Analysis](#current-state-analysis)
2. [Composio Integration Strategy](#composio-integration-strategy)
3. [Provider-Agnostic Architecture](#provider-agnostic-architecture)
4. [Tools Page Redesign](#tools-page-redesign)
5. [Backend Refactoring Plan](#backend-refactoring-plan)
6. [Frontend Updates Plan](#frontend-updates-plan)
7. [Implementation Roadmap](#implementation-roadmap)
8. [Migration Strategy](#migration-strategy)
9. [Success Metrics](#success-metrics)

---

## Current State Analysis

### Backend Architecture

#### Core Components
| Component | Location | Purpose |
|-----------|----------|---------|
| Tool Catalog API | `api/tool_catalog.py` | Tenant-level tool enablement, agent assignments |
| Tools API | `api/tools.py` | CRUD operations for tool marketplace |
| MCP Tools API | `api/mcp_tools.py` | MCP-specific tool management |
| Unified Executor | `modules/tools/execution/unified_executor.py` | Routes tool calls to executors |
| MCP Executor | `modules/tools/execution/mcp_executor.py` | Executes MCP protocol tools |
| Credential Store | `core/credentials/service.py` | Encrypted credential management |
| Tool Access Service | `modules/tools/services/tool_access_service.py` | Access validation |

#### Database Models
| Model | Table | Purpose |
|-------|-------|---------|
| `Tool` | `tools` | Tool registry with MCP config |
| `TenantToolConfig` | `tenant_tool_config` | Per-tenant tool enablement |
| `AgentToolAssignment` | `agent_tool_assignments` | Agent-to-tool mappings |
| `Credential` | `credentials` | Encrypted credential storage |
| `MCPServer` | `mcp_servers` | MCP server connections |

#### LLM Client Architecture
Current provider-specific implementations:
```
core/llm/clients/
├── base.py              # Abstract base class
├── openai_client.py     # OpenAI GPT models
├── anthropic_client.py  # Claude models
├── google_client.py     # Gemini models
├── azure_client.py      # Azure OpenAI
├── bedrock_client.py    # AWS Bedrock
├── grok_client.py       # xAI Grok
└── huggingface_client.py # HuggingFace models
```

### Frontend Architecture

#### Key Components
| Component | Location | Purpose |
|-----------|----------|---------|
| Tools Dashboard | `components/tools/tools-dashboard.tsx` | Main tools management UI |
| Agent Configuration | `components/agents/agent-configuration.tsx` | Agent settings with tool assignment |
| Tool Config Modal | `components/tools/tool-config-modal.tsx` | Tool configuration dialog |
| Tool Details Modal | `components/tools/tool-details-modal.tsx` | Tool information display |

#### API Hooks
| Hook | File | Purpose |
|------|------|---------|
| `useMCPTools` | `use-mcp-tools-api.ts` | Fetch/manage MCP tools |
| `useAgentTools` | `use-mcp-tools-api.ts` | Get tools assigned to agent |
| `useAssignToolToAgent` | `use-mcp-tools-api.ts` | Tool-agent assignment |
| `useCredentials` | `use-credentials-api.ts` | Credential management |

### Current Tool Management Flow

```
1. Adapter Discovery
   └── Unified Adapter exposes tools via REST/MCP
   
2. Tool Enablement (Settings > Tools)
   └── User enables tool → Provides credentials → Stored encrypted
   
3. Agent Assignment (Agents > Configure)
   └── Admin assigns tools to agents → Stored in agent_tool_assignments
   
4. Execution (Chat/Workflow)
   └── Agent calls tool → UnifiedExecutor routes → MCPExecutor executes
   └── Credentials resolved via CredentialStore
```

### Identified Issues

1. **160k individual methods** registered as tools - overwhelms LLM decision-making
2. **7 separate LLM clients** with duplicated logic
3. **Custom OAuth flows** for each integration
4. **MCP complexity** - requires Context Forge gateway
5. **No semantic grouping** - tools not organized by capability

---

## Composio Integration Strategy

### What to REMOVE

| Component | Reason |
|-----------|--------|
| `modules/tools/execution/mcp_executor.py` | Replaced by Composio SDK |
| `modules/tools/services/adapter_client.py` | Composio handles tool discovery |
| `modules/tools/services/adapter_tools_client.py` | Composio handles tool definitions |
| `modules/tools/services/mcp_tool_executor.py` | Composio handles execution |
| `modules/tools/executors/jit_mcp_client.py` | MCP no longer needed |
| MCP-specific models in `core/models/tools.py` | `MCPServer`, `MCPToolConnection` |
| Individual LLM clients (6 of 7) | AIML API provides unified access |

**Files to DELETE:**
- [ ] `modules/tools/execution/mcp_executor.py`
- [ ] `modules/tools/services/adapter_client.py`
- [ ] `modules/tools/services/adapter_tools_client.py`
- [ ] `modules/tools/services/mcp_tool_executor.py`
- [ ] `modules/tools/executors/jit_mcp_client.py`
- [ ] `core/llm/clients/anthropic_client.py`
- [ ] `core/llm/clients/google_client.py`
- [ ] `core/llm/clients/azure_client.py`
- [ ] `core/llm/clients/bedrock_client.py`
- [ ] `core/llm/clients/grok_client.py`
- [ ] `core/llm/clients/huggingface_client.py`

### What to REUSE

| Component | Adaptation Needed |
|-----------|-------------------|
| `core/credentials/service.py` | Keep encryption, adapt for Composio entity storage |
| `core/credentials/encryption.py` | No changes - continue using for local secrets |
| `core/models/tool_assignments.py` | Keep `AgentToolAssignment`, adapt `TenantToolConfig` |
| `modules/tools/execution/unified_executor.py` | Adapt routing to Composio |
| `modules/tools/registry/tool_registry.py` | Sync with Composio apps |
| `modules/tools/services/tool_access_service.py` | Keep access validation logic |
| `api/tool_catalog.py` | Adapt endpoints for Composio |
| `api/credentials.py` | Keep for local credential management |
| Frontend hooks structure | Adapt for Composio APIs |
| Agent configuration flow | Add Composio tool selection |

### What to CHANGE

| Component | Changes |
|-----------|---------|
| `TenantToolConfig` model | Add `composio_connection_id`, remove MCP fields |
| `UnifiedToolExecutor` | Route MCP-category tools to Composio SDK |
| `LLMConfig`/`BaseLLMProvider` | Add AIML API as primary provider |
| Tool routes map | Add Composio tool categories |
| Frontend Tools Dashboard | Redesign for "Manage Apps" pattern |
| Agent Configuration | Add app-level feature toggles |
| Credential resolution flow | Integrate Composio entity management |

### What to ADD

| Component | Purpose |
|-----------|---------|
| `core/composio/client.py` | Composio SDK wrapper |
| `core/composio/entity_manager.py` | Manage Composio entities (user connections) |
| `core/composio/tool_executor.py` | Execute tools via Composio |
| `core/aiml/client.py` | AIML API unified LLM client |
| `api/composio.py` | Composio-specific endpoints |
| `ComposioConnection` model | Track entity-app connections |
| Frontend: `ManageAppsModal` | App feature toggles UI |
| Frontend: `ComposioOAuthButton` | OAuth connection component |
| Redis caching layer | Cache Composio tool metadata |

---

## Provider-Agnostic Architecture

### AIML API Integration

**AIML API** provides access to 400+ LLMs through a single OpenAI-compatible endpoint:

```python
# core/aiml/client.py
from openai import OpenAI

class AIMLClient:
    """
    Provider-agnostic LLM client using AIML API.
    Supports 400+ models: GPT-4, Claude, Gemini, Llama, Mistral, etc.
    """
    
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.aimlapi.com/v1"
        )
    
    async def generate_response(
        self,
        messages: List[Dict],
        model: str = "gpt-4o",  # Or claude-3-5-sonnet, gemini-1.5-pro, etc.
        tools: List[Dict] = None,
        **kwargs
    ) -> LLMResponse:
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            **kwargs
        )
        return LLMResponse(
            content=response.choices[0].message.content,
            tool_calls=response.choices[0].message.tool_calls,
            usage=response.usage.model_dump(),
            model=model,
            provider="aiml"
        )
```

**Supported Model Families:**
- OpenAI: GPT-4o, GPT-4-turbo, GPT-3.5-turbo, o1-preview
- Anthropic: Claude 3.5 Sonnet, Claude 3 Opus/Haiku
- Google: Gemini 1.5 Pro/Flash, Gemini 2.0
- Meta: Llama 3.1 405B/70B/8B
- Mistral: Mixtral, Mistral Large/Medium
- And 380+ more...

### Composio Integration

```python
# core/composio/client.py
from composio_openai import ComposioToolSet, Action, App

class ComposioClient:
    """
    Composio SDK wrapper for tool execution.
    Manages 500+ apps with built-in OAuth.
    """
    
    def __init__(self, api_key: str):
        self.toolset = ComposioToolSet(api_key=api_key)
    
    def get_tools_for_apps(
        self,
        apps: List[str],
        actions: List[str] = None
    ) -> List[Dict]:
        """Get tool definitions for specified apps."""
        if actions:
            return self.toolset.get_tools(actions=actions)
        return self.toolset.get_tools(apps=apps)
    
    async def execute_action(
        self,
        action: str,
        params: Dict,
        entity_id: str
    ) -> Dict:
        """Execute a Composio action."""
        return await self.toolset.execute_action(
            action=action,
            params=params,
            entity_id=entity_id
        )
    
    def get_entity(self, entity_id: str):
        """Get or create a Composio entity for user."""
        return self.toolset.get_entity(id=entity_id)
    
    def initiate_connection(
        self,
        entity_id: str,
        app: str,
        redirect_url: str = None
    ) -> str:
        """Initiate OAuth connection for an app."""
        entity = self.get_entity(entity_id)
        connection = entity.initiate_connection(
            app_name=app,
            redirect_url=redirect_url
        )
        return connection.redirectUrl
```

### Updated LLM Manager

```python
# core/llm/manager.py (updated)

class LLMManager:
    """
    Unified LLM manager using AIML API as primary provider.
    Falls back to direct provider clients if needed.
    """
    
    def __init__(self):
        self.aiml_client = AIMLClient(os.getenv("AIML_API_KEY"))
        self.composio_client = ComposioClient(os.getenv("COMPOSIO_API_KEY"))
        
        # Model routing (all go through AIML API)
        self.model_routing = {
            "gpt-4o": "gpt-4o",
            "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
            "gemini-1.5-pro": "gemini-1.5-pro",
            "llama-3.1-405b": "meta-llama/Llama-3.1-405B-Instruct",
            # ... 400+ models
        }
    
    async def chat_with_tools(
        self,
        messages: List[Dict],
        model: str,
        apps: List[str],
        entity_id: str
    ) -> LLMResponse:
        """Execute chat with Composio tools."""
        # Get tools from Composio
        tools = self.composio_client.get_tools_for_apps(apps)
        
        # Call LLM via AIML API
        response = await self.aiml_client.generate_response(
            messages=messages,
            model=self.model_routing.get(model, model),
            tools=tools
        )
        
        # Handle tool calls
        if response.tool_calls:
            tool_results = []
            for tool_call in response.tool_calls:
                result = await self.composio_client.execute_action(
                    action=tool_call.function.name,
                    params=json.loads(tool_call.function.arguments),
                    entity_id=entity_id
                )
                tool_results.append(result)
            
            # Continue conversation with results
            messages.append({"role": "assistant", "tool_calls": response.tool_calls})
            for i, result in enumerate(tool_results):
                messages.append({
                    "role": "tool",
                    "tool_call_id": response.tool_calls[i].id,
                    "content": json.dumps(result)
                })
            
            return await self.chat_with_tools(messages, model, apps, entity_id)
        
        return response
```

---

## Tools Page Redesign

### Current vs. New Design

**Current Design:**
- Flat list of 160k+ individual tools
- Category filtering
- Per-tool install/configure

**New Design (Manage Apps Pattern):**
- App-level organization (GitHub, Slack, etc.)
- Per-app feature toggles
- OAuth connection status
- Grouped by provider

### UI Components

Based on the screenshot provided, the new "Manage Apps" modal should have:

```
┌─────────────────────────────────────────────────────────────┐
│  Manage Apps                                            ✕  │
│  Enable or disable which features your agent can use       │
├─────────────────────────────────────────────────────────────┤
│  [GitHub (beta)] 2/17    [Slack] 2/20    [Gmail] 0/15      │
│                                                             │
│  ✓ Enable all    ✕ Disable all  │  🗑 Remove this app      │
├─────────────────────────────────────────────────────────────┤
│  🔍 Search features...                                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────┐  ┌─────────────────────────┐
│  │ List Repository Issues  [●]│  │ Get Pull Request Det[●]│
│  │ Lists issues (which incl...│  │ Retrieves a specific ...│
│  └─────────────────────────────┘  └─────────────────────────┘
│  ┌─────────────────────────────┐  ┌─────────────────────────┐
│  │ List Pull Request Review[○]│  │ List PR Review Comm..[○]│
│  │ Lists submitted reviews ...│  │ Lists all review comm...│
│  └─────────────────────────────┘  └─────────────────────────┘
│  ... more features ...                                      │
├─────────────────────────────────────────────────────────────┤
│                                          [NEXT STEP →]      │
└─────────────────────────────────────────────────────────────┘
```

### React Component Structure

```tsx
// components/tools/manage-apps-modal.tsx

interface ManageAppsModalProps {
  agentId: number
  open: boolean
  onClose: () => void
}

export function ManageAppsModal({ agentId, open, onClose }: ManageAppsModalProps) {
  const [selectedApp, setSelectedApp] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  
  // Fetch connected apps from Composio
  const { data: connectedApps } = useConnectedApps(agentId)
  
  // Fetch app features (actions) for selected app
  const { data: appFeatures } = useAppFeatures(selectedApp)
  
  // Fetch enabled features for this agent
  const { data: enabledFeatures } = useAgentEnabledFeatures(agentId, selectedApp)
  
  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl">
        <DialogHeader>
          <DialogTitle>Manage Apps</DialogTitle>
          <DialogDescription>
            Enable or disable which features your agent can use for each app
          </DialogDescription>
        </DialogHeader>
        
        {/* App Tabs */}
        <AppTabList 
          apps={connectedApps}
          selectedApp={selectedApp}
          onSelectApp={setSelectedApp}
        />
        
        {/* Bulk Actions */}
        <div className="flex gap-4">
          <Button variant="ghost" onClick={() => enableAll()}>
            <Check /> Enable all
          </Button>
          <Button variant="ghost" onClick={() => disableAll()}>
            <X /> Disable all
          </Button>
          <Button variant="ghost" className="text-red-500">
            <Trash /> Remove this app
          </Button>
        </div>
        
        {/* Feature Search */}
        <Input 
          placeholder="Search features..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
        
        {/* Feature Grid */}
        <FeatureGrid 
          features={appFeatures}
          enabledFeatures={enabledFeatures}
          searchQuery={searchQuery}
          onToggle={toggleFeature}
        />
        
        <DialogFooter>
          <Button onClick={onClose}>Done</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
```

### Feature Card Component

```tsx
// components/tools/feature-card.tsx

interface FeatureCardProps {
  feature: {
    id: string
    name: string
    description: string
    enabled: boolean
  }
  onToggle: (id: string, enabled: boolean) => void
}

export function FeatureCard({ feature, onToggle }: FeatureCardProps) {
  return (
    <Card className={cn(
      "p-4 cursor-pointer transition-all",
      feature.enabled && "border-green-500 bg-green-500/10"
    )}>
      <div className="flex justify-between items-start">
        <div>
          <h4 className="font-medium">{feature.name}</h4>
          <p className="text-sm text-muted-foreground line-clamp-2">
            {feature.description}
          </p>
        </div>
        <Switch 
          checked={feature.enabled}
          onCheckedChange={(checked) => onToggle(feature.id, checked)}
        />
      </div>
    </Card>
  )
}
```

---

## Backend Refactoring Plan

### Phase 1: Core Integration (Week 1-2)

#### 1.1 Add Composio SDK
```python
# requirements.txt additions
composio-core>=0.4.0
composio-openai>=0.4.0
openai>=1.0.0  # For AIML API (OpenAI-compatible)
```

#### 1.2 Create Composio Client Module
```
core/composio/
├── __init__.py
├── client.py           # ComposioToolSet wrapper
├── entity_manager.py   # Entity/connection management
├── tool_executor.py    # Action execution
└── models.py           # Pydantic models
```

#### 1.3 Create AIML Client Module
```
core/aiml/
├── __init__.py
├── client.py           # AIML API client
├── models.py           # Request/response models
└── model_catalog.py    # 400+ model definitions
```

### Phase 2: Database Changes (Week 2)

#### 2.1 New Migration: Add Composio Fields

```python
# alembic/versions/20260120_add_composio_integration.py

def upgrade():
    # Add Composio entity tracking
    op.create_table(
        'composio_entities',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('composio_entity_id', sa.String(255), nullable=False, unique=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
    )
    
    # Add Composio connection tracking
    op.create_table(
        'composio_connections',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('entity_id', sa.Integer(), sa.ForeignKey('composio_entities.id')),
        sa.Column('app_name', sa.String(100), nullable=False),
        sa.Column('connection_id', sa.String(255)),
        sa.Column('status', sa.String(50), default='pending'),
        sa.Column('connected_at', sa.DateTime()),
    )
    
    # Update tenant_tool_config for Composio
    op.add_column('tenant_tool_config',
        sa.Column('composio_app_name', sa.String(100))
    )
    op.add_column('tenant_tool_config',
        sa.Column('enabled_actions', postgresql.JSONB(), default=list)
    )
    
    # Add agent feature toggles
    op.create_table(
        'agent_app_features',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('agent_id', sa.Integer(), sa.ForeignKey('agents.id')),
        sa.Column('app_name', sa.String(100), nullable=False),
        sa.Column('action_name', sa.String(255), nullable=False),
        sa.Column('enabled', sa.Boolean(), default=True),
    )
```

### Phase 3: API Updates (Week 3)

#### 3.1 New Composio Endpoints

```python
# api/composio.py

router = APIRouter(prefix="/api/composio", tags=["Composio"])

@router.get("/apps")
async def list_available_apps():
    """List all available Composio apps."""
    pass

@router.post("/connect/{app_name}")
async def initiate_connection(app_name: str, tenant_id: UUID):
    """Initiate OAuth connection for an app."""
    pass

@router.get("/connections")
async def list_connections(tenant_id: UUID):
    """List all connected apps for tenant."""
    pass

@router.delete("/connections/{app_name}")
async def disconnect_app(app_name: str, tenant_id: UUID):
    """Disconnect an app."""
    pass

@router.get("/apps/{app_name}/actions")
async def list_app_actions(app_name: str):
    """List all available actions for an app."""
    pass
```

#### 3.2 Updated Agent Tools Endpoints

```python
# api/tool_catalog.py (updates)

@router.get("/agents/{agent_id}/apps")
async def get_agent_apps(agent_id: int):
    """Get apps assigned to an agent with feature status."""
    pass

@router.put("/agents/{agent_id}/apps/{app_name}/features")
async def update_agent_features(
    agent_id: int,
    app_name: str,
    features: List[FeatureToggle]
):
    """Update enabled features for an agent-app combination."""
    pass

@router.post("/agents/{agent_id}/apps/{app_name}")
async def assign_app_to_agent(agent_id: int, app_name: str):
    """Assign a connected app to an agent."""
    pass
```

### Phase 4: Executor Integration (Week 3-4)

#### 4.1 Update Unified Executor

```python
# modules/tools/execution/unified_executor.py (updated)

class UnifiedToolExecutor:
    def __init__(self, db_session: Session):
        self.db = db_session
        self._composio_executor = None
        self._aiml_client = None
        
        # Updated tool routes
        self.tool_routes = {
            # Keep platform tools
            'search_knowledge': self._execute_platform_tool,
            'search_codebase': self._execute_platform_tool,
            
            # Keep file/shell operations
            'read_file': self._execute_file_op,
            'write_file': self._execute_file_op,
            'execute_command': self._execute_shell,
            
            # Route Composio apps
            'github_*': self._execute_composio_tool,
            'slack_*': self._execute_composio_tool,
            'gmail_*': self._execute_composio_tool,
            # ... all Composio apps
        }
    
    @property
    def composio_executor(self):
        if self._composio_executor is None:
            from core.composio import ComposioToolExecutor
            self._composio_executor = ComposioToolExecutor(self.db)
        return self._composio_executor
    
    async def _execute_composio_tool(
        self,
        tool_name: str,
        arguments: Dict,
        agent_id: int,
        tenant_id: UUID
    ) -> Dict:
        """Execute a Composio tool action."""
        # Validate agent has access to this action
        if not await self._validate_agent_action_access(agent_id, tool_name):
            raise PermissionError(f"Agent {agent_id} not authorized for {tool_name}")
        
        # Get entity ID for tenant
        entity = await self.composio_executor.get_entity_for_tenant(tenant_id)
        
        # Execute via Composio
        return await self.composio_executor.execute(
            action=tool_name,
            params=arguments,
            entity_id=entity.composio_entity_id
        )
```

---

## Frontend Updates Plan

### Phase 1: New Hooks (Week 2-3)

```typescript
// hooks/use-composio-api.ts

// Connected apps management
export function useConnectedApps(tenantId: string) {
  return useQuery({
    queryKey: ['composio', 'connections', tenantId],
    queryFn: () => apiClient.getComposioConnections(tenantId)
  })
}

export function useInitiateConnection() {
  return useMutation({
    mutationFn: ({ appName, tenantId }: { appName: string; tenantId: string }) =>
      apiClient.initiateComposioConnection(appName, tenantId)
  })
}

export function useDisconnectApp() {
  return useMutation({
    mutationFn: ({ appName, tenantId }: { appName: string; tenantId: string }) =>
      apiClient.disconnectComposioApp(appName, tenantId)
  })
}

// App actions/features
export function useAppActions(appName: string) {
  return useQuery({
    queryKey: ['composio', 'apps', appName, 'actions'],
    queryFn: () => apiClient.getComposioAppActions(appName),
    enabled: !!appName
  })
}

// Agent feature management
export function useAgentAppFeatures(agentId: number, appName: string) {
  return useQuery({
    queryKey: ['agents', agentId, 'apps', appName, 'features'],
    queryFn: () => apiClient.getAgentAppFeatures(agentId, appName),
    enabled: !!agentId && !!appName
  })
}

export function useUpdateAgentFeatures() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: ({ 
      agentId, 
      appName, 
      features 
    }: { 
      agentId: number
      appName: string
      features: FeatureToggle[]
    }) => apiClient.updateAgentAppFeatures(agentId, appName, features),
    onSuccess: (_, { agentId, appName }) => {
      queryClient.invalidateQueries({ 
        queryKey: ['agents', agentId, 'apps', appName, 'features'] 
      })
    }
  })
}
```

### Phase 2: New Components (Week 3-4)

```
components/
├── composio/
│   ├── app-connection-button.tsx   # OAuth connection button
│   ├── connected-apps-list.tsx     # List of connected apps
│   ├── app-feature-grid.tsx        # Feature toggle grid
│   └── manage-apps-modal.tsx       # Main management modal
├── tools/
│   └── tools-dashboard.tsx         # Updated for Composio
└── agents/
    └── agent-configuration.tsx     # Updated with app management
```

### Phase 3: Page Updates (Week 4)

#### Tools Page Redesign

```tsx
// components/tools/tools-dashboard.tsx (rewritten)

export function ToolsDashboard() {
  const { data: availableApps } = useAvailableApps()
  const { data: connectedApps } = useConnectedApps(tenantId)
  const initiateConnection = useInitiateConnection()
  
  return (
    <div className="p-6 space-y-6">
      <header>
        <h1>App Integrations</h1>
        <p>Connect and manage your app integrations</p>
      </header>
      
      {/* Connected Apps Section */}
      <section>
        <h2>Connected Apps ({connectedApps?.length || 0})</h2>
        <div className="grid grid-cols-4 gap-4">
          {connectedApps?.map(app => (
            <ConnectedAppCard 
              key={app.name}
              app={app}
              onManage={() => openManageModal(app.name)}
              onDisconnect={() => disconnectApp(app.name)}
            />
          ))}
        </div>
      </section>
      
      {/* Available Apps Section */}
      <section>
        <h2>Available Apps</h2>
        <div className="grid grid-cols-4 gap-4">
          {availableApps?.map(app => (
            <AvailableAppCard 
              key={app.name}
              app={app}
              onConnect={() => initiateConnection.mutate({ 
                appName: app.name, 
                tenantId 
              })}
            />
          ))}
        </div>
      </section>
      
      {/* Manage Apps Modal */}
      <ManageAppsModal 
        open={!!selectedApp}
        appName={selectedApp}
        onClose={() => setSelectedApp(null)}
      />
    </div>
  )
}
```

---

## Implementation Roadmap

### Task Checklist

#### Phase 1: Foundation (Week 1-2)

**Backend Setup**
- [ ] Install Composio SDK (`composio-core`, `composio-openai`)
- [ ] Create `core/composio/` module structure
- [ ] Implement `ComposioClient` class
- [ ] Implement `EntityManager` class
- [ ] Create unit tests for Composio client

**AIML API Integration**
- [ ] Create `core/aiml/` module structure
- [ ] Implement `AIMLClient` class
- [ ] Create model catalog with 400+ models
- [ ] Add environment variable: `AIML_API_KEY`
- [ ] Create unit tests for AIML client

**Database Preparation**
- [ ] Create migration for `composio_entities` table
- [ ] Create migration for `composio_connections` table
- [ ] Update `tenant_tool_config` with Composio fields
- [ ] Create `agent_app_features` table
- [ ] Run migrations on dev environment

#### Phase 2: API Layer (Week 2-3)

**New Endpoints**
- [ ] Create `api/composio.py` router
- [ ] Implement `GET /api/composio/apps`
- [ ] Implement `POST /api/composio/connect/{app_name}`
- [ ] Implement `GET /api/composio/connections`
- [ ] Implement `DELETE /api/composio/connections/{app_name}`
- [ ] Implement `GET /api/composio/apps/{app_name}/actions`

**Updated Endpoints**
- [ ] Update `api/tool_catalog.py` for Composio
- [ ] Add `GET /api/agents/{id}/apps` endpoint
- [ ] Add `PUT /api/agents/{id}/apps/{app}/features` endpoint
- [ ] Add `POST /api/agents/{id}/apps/{app}` endpoint

**Authentication**
- [ ] Add Composio API key to secrets management
- [ ] Implement OAuth callback handler for Composio
- [ ] Add webhook endpoint for Composio connection events

#### Phase 3: Execution Layer (Week 3-4)

**Executor Updates**
- [ ] Create `core/composio/tool_executor.py`
- [ ] Update `UnifiedToolExecutor` routing
- [ ] Implement Composio action execution
- [ ] Add entity ID resolution from tenant
- [ ] Implement feature access validation

**LLM Integration**
- [ ] Update `LLMManager` to use AIML client
- [ ] Implement tool-call handling with Composio
- [ ] Add model routing configuration
- [ ] Remove deprecated provider clients (Phase 5)

**Caching**
- [ ] Implement Redis caching for Composio app metadata
- [ ] Cache action definitions per app
- [ ] Add cache invalidation on connection changes

#### Phase 4: Frontend (Week 4-5)

**New Hooks**
- [ ] Create `hooks/use-composio-api.ts`
- [ ] Implement `useConnectedApps` hook
- [ ] Implement `useAvailableApps` hook
- [ ] Implement `useAppActions` hook
- [ ] Implement `useAgentAppFeatures` hook
- [ ] Implement connection mutation hooks

**New Components**
- [ ] Create `ManageAppsModal` component
- [ ] Create `FeatureCard` component
- [ ] Create `FeatureGrid` component
- [ ] Create `AppConnectionButton` component
- [ ] Create `ConnectedAppCard` component
- [ ] Create `AvailableAppCard` component

**Page Updates**
- [ ] Redesign Tools Dashboard page
- [ ] Update Agent Configuration with app management
- [ ] Add "Manage Apps" button to agent card
- [ ] Update chat input for tool autocomplete

#### Phase 5: Cleanup & Migration (Week 5-6)

**Remove Deprecated Code**
- [ ] Delete `mcp_executor.py`
- [ ] Delete `adapter_client.py`
- [ ] Delete `adapter_tools_client.py`
- [ ] Delete `mcp_tool_executor.py`
- [ ] Delete `jit_mcp_client.py`
- [ ] Delete individual LLM provider clients (keep `base.py`)
- [ ] Remove MCP-specific database models

**Data Migration**
- [ ] Create script to migrate existing tool assignments
- [ ] Map MCP tools to Composio actions
- [ ] Migrate credential references
- [ ] Update agent configurations

**Testing**
- [ ] End-to-end test: Connect GitHub app
- [ ] End-to-end test: Enable/disable features
- [ ] End-to-end test: Agent uses Composio tool
- [ ] Performance test: Tool execution latency
- [ ] Load test: Multiple concurrent tool calls

#### Phase 6: Documentation & Launch (Week 6)

**Documentation**
- [ ] Update API documentation
- [ ] Create Composio integration guide
- [ ] Document AIML API model catalog
- [ ] Update agent configuration docs
- [ ] Create migration guide for existing users

**Monitoring**
- [ ] Add Composio execution metrics
- [ ] Add AIML API usage tracking
- [ ] Create dashboard for tool usage
- [ ] Set up alerts for connection failures

**Launch**
- [ ] Feature flag for gradual rollout
- [ ] Beta testing with select users
- [ ] Monitor error rates
- [ ] Full rollout

---

## Migration Strategy

### For Existing Tool Configurations

```python
# scripts/migrate_to_composio.py

async def migrate_tenant_tools(tenant_id: UUID):
    """Migrate existing tool configurations to Composio."""
    
    # 1. Get existing tool configs
    existing_configs = db.query(TenantToolConfig).filter(
        TenantToolConfig.tenant_id == tenant_id
    ).all()
    
    # 2. Create Composio entity for tenant
    composio = ComposioClient()
    entity = composio.get_entity(str(tenant_id))
    
    # 3. Map old tools to Composio apps
    tool_mapping = {
        'github': 'GITHUB',
        'slack': 'SLACK',
        'gmail': 'GMAIL',
        # ... more mappings
    }
    
    # 4. For each existing config, initiate Composio connection
    for config in existing_configs:
        composio_app = tool_mapping.get(config.tool_id)
        if composio_app:
            # User will need to re-authenticate via OAuth
            connection = entity.initiate_connection(app_name=composio_app)
            
            # Store pending connection
            db.add(ComposioConnection(
                entity_id=entity.id,
                app_name=composio_app,
                status='pending_migration',
                legacy_config_id=config.id
            ))
    
    db.commit()
```

### For Agent Tool Assignments

```python
async def migrate_agent_assignments(agent_id: int):
    """Migrate agent tool assignments to feature toggles."""
    
    # 1. Get existing assignments
    assignments = db.query(AgentToolAssignment).filter(
        AgentToolAssignment.agent_id == agent_id
    ).all()
    
    # 2. Map to Composio actions
    for assignment in assignments:
        composio_app = get_composio_app_for_tool(assignment.tool_id)
        
        if composio_app:
            # Get all actions for this app
            actions = composio.get_app_actions(composio_app)
            
            # Enable all actions by default (user can disable later)
            for action in actions:
                db.add(AgentAppFeature(
                    agent_id=agent_id,
                    app_name=composio_app,
                    action_name=action.name,
                    enabled=True
                ))
    
    db.commit()
```

### Rollback Plan

1. **Feature Flag**: All Composio features behind `ENABLE_COMPOSIO` flag
2. **Parallel Systems**: Keep MCP executor until migration complete
3. **Data Backup**: Full backup of tool configs before migration
4. **Gradual Rollout**: 10% → 50% → 100% user rollout

---

## Success Metrics

### Quantitative

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Available Tools | 160k methods | 500+ apps | Composio catalog |
| LLM Providers | 7 clients | 400+ models | AIML API catalog |
| Tool Selection Accuracy | ~60% | 95%+ | User feedback |
| OAuth Setup Time | Manual per-tool | One-click | Time to connect |
| New Integration Time | Weeks | Immediate | Composio availability |
| Maintenance Burden | High | Low | Dev hours/month |

### Qualitative

- [ ] Users can connect apps in < 30 seconds
- [ ] Agents correctly select tools 95%+ of the time
- [ ] No manual credential management required for Composio apps
- [ ] Feature toggles provide granular control
- [ ] UI matches "Manage Apps" design pattern

### Technical Health

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Tool execution latency | < 2s | > 5s |
| OAuth success rate | > 99% | < 95% |
| Connection uptime | > 99.9% | < 99% |
| API error rate | < 0.1% | > 1% |

---

## Appendix

### A. Composio App Categories

| Category | Example Apps | Count |
|----------|--------------|-------|
| Developer Tools | GitHub, GitLab, Jira, Linear | 50+ |
| Communication | Slack, Discord, Teams, Email | 30+ |
| CRM & Sales | Salesforce, HubSpot, Pipedrive | 40+ |
| Productivity | Notion, Asana, Trello, Todoist | 60+ |
| Cloud & Infrastructure | AWS, GCP, Azure, Vercel | 30+ |
| Data & Analytics | Snowflake, BigQuery, Mixpanel | 40+ |
| Marketing | Mailchimp, SendGrid, Intercom | 35+ |
| Finance | Stripe, QuickBooks, Plaid | 25+ |
| Social Media | Twitter, LinkedIn, Instagram | 20+ |
| Other | 100+ more apps | 100+ |

### B. AIML API Model Examples

```python
# Popular models available via AIML API
AIML_MODELS = {
    # OpenAI
    "gpt-4o": "gpt-4o-2024-11-20",
    "gpt-4o-mini": "gpt-4o-mini",
    "o1-preview": "o1-preview",
    
    # Anthropic
    "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3-opus": "claude-3-opus-20240229",
    
    # Google
    "gemini-2.0-flash": "gemini-2.0-flash-exp",
    "gemini-1.5-pro": "gemini-1.5-pro",
    
    # Meta
    "llama-3.1-405b": "meta-llama/Llama-3.1-405B-Instruct",
    "llama-3.1-70b": "meta-llama/Llama-3.1-70B-Instruct",
    
    # Mistral
    "mistral-large": "mistral-large-latest",
    "mixtral-8x7b": "mistral/Mixtral-8x7B-Instruct-v0.1",
    
    # ... 380+ more
}
```

### C. Environment Variables

```bash
# New environment variables required
COMPOSIO_API_KEY=your_composio_api_key
AIML_API_KEY=your_aiml_api_key

# Optional
COMPOSIO_WEBHOOK_SECRET=your_webhook_secret
AIML_DEFAULT_MODEL=gpt-4o
```

---

**Document Version History**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-20 | DeepAgent | Initial PRD |
