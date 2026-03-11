# Authentication & Multi-Tenancy

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [docker-compose.yml](docker-compose.yml)
- [frontend/.dockerignore](frontend/.dockerignore)
- [frontend/Dockerfile](frontend/Dockerfile)
- [frontend/app/admin/plugins/page.tsx](frontend/app/admin/plugins/page.tsx)
- [frontend/lib/api-client.ts](frontend/lib/api-client.ts)
- [orchestrator/.env.example](orchestrator/.env.example)
- [orchestrator/Dockerfile](orchestrator/Dockerfile)
- [orchestrator/api/agent_plugins.py](orchestrator/api/agent_plugins.py)
- [orchestrator/config.py](orchestrator/config.py)
- [orchestrator/core/database/load_seed_data.py](orchestrator/core/database/load_seed_data.py)
- [orchestrator/core/redis/client.py](orchestrator/core/redis/client.py)
- [orchestrator/core/seeds/seed_personas.py](orchestrator/core/seeds/seed_personas.py)
- [orchestrator/core/seeds/seed_plugin_categories.py](orchestrator/core/seeds/seed_plugin_categories.py)
- [orchestrator/core/services/plugin_cache.py](orchestrator/core/services/plugin_cache.py)
- [orchestrator/main.py](orchestrator/main.py)
- [orchestrator/requirements.txt](orchestrator/requirements.txt)
- [scripts/ralph/prd.json](scripts/ralph/prd.json)

</details>



## Purpose & Scope

This document describes Automatos AI's authentication and multi-tenancy architecture. It covers Clerk JWT-based authentication, workspace-scoped data isolation, and secure credential management across the platform.

For workspace creation and management UI, see [Workspace Management](#12.2). For credential storage and resolution strategies, see [Credentials Management](#12.4). For API security and rate limiting, see [Backend Architecture](#13.1).

---

## Authentication Architecture

### Clerk JWT Integration

Automatos AI uses Clerk for authentication, providing secure JWT-based identity verification without requiring password management infrastructure. Authentication is optional during local development but required in production.

**Configuration** ([config.py:168-175]()):

```python
CLERK_SECRET_KEY: str = os.getenv("CLERK_SECRET_KEY")
CLERK_PUBLISHABLE_KEY: str = os.getenv("NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY")
CLERK_JWKS_URL: str = os.getenv("CLERK_JWKS_URL")
CLERK_AUDIENCE: str = os.getenv("CLERK_AUDIENCE", "")
```

The system enforces authentication via `REQUIRE_AUTH` flag ([config.py:87]()), which defaults to `true` in production but can be disabled for local development.

**Sources:** [orchestrator/config.py:168-175](), [orchestrator/config.py:87]()

---

### Authentication Flow

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant ClerkSDK["Clerk SDK"]
    participant FastAPI["FastAPI Backend"]
    participant Auth["get_request_context_hybrid"]
    participant DB["PostgreSQL"]

    User->>Frontend: Visit /dashboard
    Frontend->>ClerkSDK: Check auth state
    
    alt User not signed in
        ClerkSDK->>Frontend: Redirect to /sign-in
        User->>ClerkSDK: Login with email/OAuth
        ClerkSDK->>User: Issue JWT token
    end
    
    Frontend->>ClerkSDK: getToken()
    ClerkSDK-->>Frontend: JWT token
    
    Frontend->>FastAPI: API request + Authorization: Bearer {JWT} + X-Workspace-ID
    FastAPI->>Auth: Middleware validates request
    Auth->>ClerkSDK: Verify JWT signature (JWKS)
    ClerkSDK-->>Auth: Valid / Invalid
    
    alt JWT valid
        Auth->>DB: Query user workspace membership
        DB-->>Auth: Workspace confirmed
        Auth-->>FastAPI: RequestContext(user_id, workspace_id)
        FastAPI->>FastAPI: Route handler processes request
    else JWT invalid or workspace denied
        Auth-->>FastAPI: HTTPException 401/403
        FastAPI-->>Frontend: Error response
    end
```

**Authentication Steps:**

1. **Frontend Authentication**: User authenticates through Clerk's UI components (`/sign-in`, `/sign-up`)
2. **JWT Issuance**: Clerk issues a JWT token to the authenticated user
3. **Token Retrieval**: Frontend calls `getToken()` from Clerk's React SDK
4. **Request Headers**: Frontend includes `Authorization: Bearer {JWT}` and `X-Workspace-ID` headers
5. **JWT Verification**: Backend validates JWT signature using Clerk's JWKS endpoint
6. **Workspace Validation**: Backend confirms user's membership in the specified workspace
7. **Request Context**: Successful validation creates a `RequestContext` with `user_id` and `workspace_id`

**Sources:** [orchestrator/main.py:118-122](), [orchestrator/config.py:168-175](), [frontend/lib/api-client.ts:160-163]()

---

## Request Context & Middleware

### Middleware Pipeline

Every API request flows through a standardized middleware pipeline that enforces authentication, workspace isolation, and security policies.

```mermaid
graph TB
    Request["HTTP Request"]
    CORS["CORSMiddleware"]
    RateLimit["Rate Limiting<br/>(SlowAPI 60/min)"]
    BodySize["Body Size Limit<br/>(10MB default, 50MB uploads)"]
    Security["Security Headers<br/>(CSP, X-Frame-Options)"]
    ReqID["Request ID Assignment<br/>(X-Request-ID)"]
    Tracking["API Call Tracking<br/>(api_call_stats)"]
    Auth["Authentication Middleware<br/>(get_request_context_hybrid)"]
    Router["API Routers<br/>(/api/agents, /api/chat, etc.)"]

    Request --> CORS
    CORS --> RateLimit
    RateLimit --> BodySize
    BodySize --> Security
    Security --> ReqID
    ReqID --> Tracking
    Tracking --> Auth
    Auth --> Router

    style Auth fill:#333
    style CORS fill:#111
    style RateLimit fill:#111
```

**Middleware Execution Order** ([main.py:555-642]()):

1. **CORS Middleware** (line 560): Validates origin against allowed list
2. **Rate Limiting** (line 583): Enforces 60 requests/minute per IP (configurable)
3. **Body Size Limit** (line 603): Prevents payload too large (10MB default, 50MB for uploads)
4. **Security Headers** (line 617): Adds CSP, X-Frame-Options, HSTS in production
5. **Request ID** (line 632): Assigns unique ID for distributed tracing
6. **API Tracking** (line 643): Records metrics to `api_call_stats` dict
7. **Authentication** (implicit): Enforced by dependency injection in route handlers

**Sources:** [orchestrator/main.py:555-688](), [orchestrator/main.py:583-596]()

---

### Request Context Resolution

The `get_request_context_hybrid` dependency provides workspace-aware authentication to route handlers.

```mermaid
graph TB
    Handler["Route Handler<br/>(depends on get_request_context_hybrid)"]
    Context["get_request_context_hybrid"]
    ClerkJWT["Clerk JWT Validation"]
    Header["X-Workspace-ID Header"]
    Default["DEFAULT_WORKSPACE_ID env var"]
    TenantID["DEFAULT_TENANT_ID constant<br/>(UUID: 00000000-0000-0000-0000-000000000000)"]
    Result["RequestContext<br/>(user_id, workspace_id)"]

    Handler --> Context
    Context --> ClerkJWT
    Context --> Header
    
    ClerkJWT -->|Valid| ExtractUserID["Extract user_id from JWT"]
    ClerkJWT -->|Invalid & REQUIRE_AUTH=true| Reject401["HTTPException 401"]
    ClerkJWT -->|Missing & REQUIRE_AUTH=false| Anonymous["user_id = None (dev mode)"]
    
    ExtractUserID --> WorkspaceRes["Resolve workspace_id"]
    Anonymous --> WorkspaceRes
    
    WorkspaceRes --> Header
    Header -->|Present| ValidateWorkspace["Validate user workspace membership"]
    Header -->|Missing| Default
    Default -->|Set| UseDefault["Use DEFAULT_WORKSPACE_ID"]
    Default -->|Missing| TenantID
    TenantID --> UseDefault
    
    ValidateWorkspace -->|Authorized| Result
    ValidateWorkspace -->|Denied| Reject403["HTTPException 403"]
    UseDefault --> Result
    
    Result --> Handler

    style ClerkJWT fill:#333
    style Result fill:#333
```

**Workspace Resolution Priority** (from highest to lowest):

1. **`X-Workspace-ID` Header**: Explicitly specified workspace (validated against user membership)
2. **`DEFAULT_WORKSPACE_ID` Environment Variable**: Fallback for requests without workspace header
3. **`DEFAULT_TENANT_ID` Constant**: UUID `00000000-0000-0000-0000-000000000000` for single-tenant mode

**Sources:** [orchestrator/core/auth/hybrid.py:1-200](), [orchestrator/config.py:21-23](), [orchestrator/config.py:173-174]()

---

## Multi-Tenancy Architecture

### Five-Layer Data Isolation

Automatos AI implements workspace isolation across five infrastructure layers to ensure complete tenant separation.

```mermaid
graph TB
    subgraph "Layer 1: Database"
        DBFilter["workspace_id Column<br/>(All major tables)"]
        OwnerType["owner_type Field<br/>(marketplace vs workspace)"]
    end
    
    subgraph "Layer 2: S3 Storage"
        S3Paths["Path Prefix:<br/>workspaces/{workspace_id}/"]
        S3Vectors["Vector Bucket:<br/>automatos-vectors-{workspace_id}"]
    end
    
    subgraph "Layer 3: Redis"
        RedisKeys["Key Prefix:<br/>workspace:ws:{workspace_id}:*"]
        PubSub["PubSub Channels:<br/>workflow:{id}:execution:{id}"]
    end
    
    subgraph "Layer 4: Filesystem"
        WorkspaceFS["Volume Path:<br/>/workspaces/{workspace_id}/"]
        PathSafety["resolve_safe_path()<br/>symlink & traversal protection"]
    end
    
    subgraph "Layer 5: Application Logic"
        QueryFilter["SQLAlchemy Filters<br/>(workspace_id = ctx.workspace_id)"]
        PermChecks["Permission Checks<br/>(AgentAppAssignment, EntityConnection)"]
    end

    DBFilter --> OwnerType
    S3Paths --> S3Vectors
    RedisKeys --> PubSub
    WorkspaceFS --> PathSafety
    QueryFilter --> PermChecks

    style DBFilter fill:#333
    style S3Paths fill:#333
    style RedisKeys fill:#333
    style WorkspaceFS fill:#333
    style QueryFilter fill:#333
```

**Sources:** [Diagram 8 from high-level system diagrams]()

---

### Database Isolation

All major tables include a `workspace_id` column with foreign key constraints to the `workspaces` table. Queries automatically filter by the authenticated user's workspace.

**Example Table Structure** ([core/models/core.py:1-500]()):

```python
class Agent(Base):
    __tablename__ = "agents"
    
    id = Column(Integer, primary_key=True)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id"), nullable=False, index=True)
    name = Column(String, nullable=False)
    status = Column(String, default="active")
    # ... other fields
```

**Query Pattern** ([api/agents.py:50-200]()):

```python
@router.get("/api/agents")
async def list_agents(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    agents = db.query(Agent).filter(
        Agent.workspace_id == ctx.workspace_id,
        Agent.status == "active"
    ).all()
    return {"items": agents}
```

**Marketplace vs Workspace Items:**

The `owner_type` field distinguishes shared marketplace items from workspace-private items:

- **`owner_type = "marketplace"`**: Globally available (e.g., community plugins, personas)
- **`owner_type = "workspace"`**: Private to the owning workspace

**Sources:** [orchestrator/core/models/core.py:1-500](), [orchestrator/api/agents.py:1-300]()

---

### S3 Storage Isolation

All S3 objects use workspace-prefixed keys to prevent cross-tenant access. Vector embeddings are stored in per-workspace buckets.

**S3 Key Structure** ([config.py:246-260]()):

| Resource | Key Pattern | Example |
|----------|------------|---------|
| Documents | `workspaces/{workspace_id}/documents/{doc_id}/` | `workspaces/abc123/documents/report.pdf` |
| Vectors | Separate bucket per workspace | `automatos-vectors-abc123` |
| Recipe Logs | `workspaces/{workspace_id}/logs/{execution_id}/` | `workspaces/abc123/logs/exec-456.json` |
| Generated Images | `workspaces/{workspace_id}/images/{image_id}` | `workspaces/abc123/images/chart.png` |

**Configuration** ([config.py:252-260]()):

```python
S3_VECTORS_ENABLED: bool = os.getenv("S3_VECTORS_ENABLED", "false").lower() == "true"
S3_VECTORS_BUCKET: str = os.getenv("S3_VECTORS_BUCKET")  # Per-workspace bucket
S3_DOCUMENTS_BUCKET: str = os.getenv("S3_DOCUMENTS_BUCKET", "automatos-ai")
```

**Sources:** [orchestrator/config.py:246-260](), [orchestrator/modules/rag/service.py:1-550]()

---

### Redis Isolation

Redis keys use workspace-scoped prefixes to prevent cache leakage between tenants.

**Key Naming Conventions** ([core/redis/client.py:1-200]()):

```python
# Workspace-specific keys
workspace:ws:{workspace_id}:cache:{key}
workspace:ws:{workspace_id}:session:{session_id}

# Task queue keys
workspace:task:{task_id}:status
workspace:task:{task_id}:result

# PubSub channels (workflow-scoped)
workflow:{workflow_id}:execution:{execution_id}
```

**Redis Configuration with Security** ([docker-compose.yml:48-73]()):

```yaml
redis:
  command: >
    redis-server
    --requirepass ${REDIS_PASSWORD}
    --rename-command FLUSHDB ""
    --rename-command FLUSHALL ""
    --rename-command DEBUG ""
```

The Redis instance is hardened by disabling dangerous commands (`FLUSHDB`, `FLUSHALL`, `DEBUG`) that could wipe data if exposed.

**Sources:** [orchestrator/core/redis/client.py:66-119](), [docker-compose.yml:48-73]()

---

### Filesystem Isolation

The workspace worker service isolates code execution in per-workspace directories with quota enforcement and path safety validation.

**Volume Configuration** ([docker-compose.yml:174-217]()):

```yaml
workspace-worker:
  environment:
    WORKSPACE_VOLUME_PATH: /workspaces
    WORKSPACE_DEFAULT_QUOTA_GB: 5
    WORKER_CONCURRENCY: 3
  volumes:
    - workspace_data:/workspaces
```

**Path Safety Validation** (workspace-worker service):

```python
def resolve_safe_path(workspace_id: str, relative_path: str) -> Path:
    """Resolve path within workspace directory, preventing traversal attacks."""
    base = Path(f"/workspaces/{workspace_id}").resolve()
    target = (base / relative_path).resolve()
    
    # Reject if target escapes workspace directory
    if not str(target).startswith(str(base)):
        raise ValueError("Path traversal attempt blocked")
    
    # Reject symlinks
    if target.is_symlink():
        raise ValueError("Symlinks not allowed")
    
    return target
```

**Security Policies:**

1. **Path Safety**: Blocks `..`, symlinks, absolute paths
2. **Command Whitelist**: Only approved binaries can execute
3. **Storage Quotas**: Default 5GB per workspace (configurable)
4. **Environment Sandbox**: Stripped `PATH`, no host variables
5. **Resource Limits**: CPU and memory limits enforced by Docker

**Sources:** [docker-compose.yml:174-217](), [orchestrator/config.py:219-226]()

---

## Credential Management

### Credential Types & Storage

Automatos AI supports multiple credential types with secure encrypted storage. Credentials are scoped to workspaces and encrypted at rest.

```mermaid
graph TB
    subgraph "Credential Types"
        OpenAI["OpenAI API Key"]
        Anthropic["Anthropic API Key"]
        Google["Google/Gemini API Key"]
        Azure["Azure OpenAI<br/>(key + endpoint)"]
        OpenRouter["OpenRouter API Key"]
        Composio["Composio API Key"]
        Custom["Custom API Keys"]
    end
    
    subgraph "Storage Layer"
        DB["credentials Table<br/>(encrypted value)"]
        CredType["credential_types Table<br/>(schema definitions)"]
    end
    
    subgraph "Encryption"
        Key["CREDENTIAL_ENCRYPTION_KEY<br/>(from environment)"]
        Encrypt["encrypt_credential()"]
        Decrypt["decrypt_credential()"]
    end
    
    OpenAI --> DB
    Anthropic --> DB
    Google --> DB
    Azure --> DB
    OpenRouter --> DB
    Composio --> DB
    Custom --> DB
    
    DB --> CredType
    DB --> Encrypt
    Encrypt --> Key
    Key --> Decrypt

    style DB fill:#333
    style Key fill:#333
```

**Credential Table Schema:**

```sql
CREATE TABLE credentials (
    id UUID PRIMARY KEY,
    workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,
    credential_type_id UUID REFERENCES credential_types(id),
    name VARCHAR(255) NOT NULL,
    value TEXT NOT NULL,  -- Encrypted JSON
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_credentials_workspace ON credentials(workspace_id);
```

**Encryption Configuration** ([config.py:175]()):

```python
CREDENTIAL_ENCRYPTION_KEY: str = os.getenv("CREDENTIAL_ENCRYPTION_KEY")
```

**Sources:** [orchestrator/config.py:175](), [orchestrator/core/database/load_seed_data.py:62-103]()

---

### Six-Level Credential Resolution

When agents request LLM credentials, the system follows a six-level fallback strategy to maximize availability while respecting workspace boundaries.

```mermaid
graph TB
    AgentRequest["Agent requests LLM credentials<br/>(provider: openai)"]
    
    L1["Level 1: Agent-Specific Credential"]
    L2["Level 2: Workspace Default Credential"]
    L3["Level 3: System Settings Credential"]
    L4["Level 4: User API Key (BYOK)"]
    L5["Level 5: Environment Variable"]
    L6["Level 6: Fallback Provider"]
    
    Found["Credential Found"]
    Fallback["Try Secondary Provider"]
    Error["No Valid Credentials"]
    
    AgentRequest --> L1
    L1 -->|Not found| L2
    L1 -->|Found| Found
    
    L2 -->|Not found| L3
    L2 -->|Found| Found
    
    L3 -->|Not found| L4
    L3 -->|Found| Found
    
    L4 -->|Not found| L5
    L4 -->|Found| Found
    
    L5 -->|Not found| L6
    L5 -->|Found| Found
    
    L6 -->|Primary failed| Fallback
    L6 -->|Found| Found
    
    Fallback -->|Secondary found| Found
    Fallback -->|Secondary failed| Error

    style L1 fill:#333
    style L2 fill:#333
    style L5 fill:#333
    style Found fill:#333
```

**Resolution Strategy** (implemented in `LLMManager`):

1. **Agent-Specific Credential**: Check `agent.model_config.credentials`
2. **Workspace Default Credential**: Query `credentials` table filtered by `workspace_id`
3. **System Settings Credential**: Check `system_settings` table (admin-configured defaults)
4. **User API Key (BYOK)**: Check `user_api_keys` table for bring-your-own-key
5. **Environment Variable**: Fall back to `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.
6. **Fallback Provider**: If primary provider fails, try secondary (e.g., GPT-4 → Claude)

**Sources:** [orchestrator/core/llm/manager.py:200-650](), [orchestrator/modules/agents/factory/agent_factory.py:200-650]()

---

### Composio OAuth Connections

External tool integrations via Composio use workspace-scoped OAuth connections stored in the `entity_connections` table.

**Entity Connection Storage:**

```sql
CREATE TABLE entity_connections (
    id UUID PRIMARY KEY,
    workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,
    composio_entity_id VARCHAR(255) NOT NULL,  -- Composio's internal ID
    app_name VARCHAR(100) NOT NULL,
    connection_status VARCHAR(50) DEFAULT 'active',
    access_token TEXT,  -- Encrypted OAuth token
    refresh_token TEXT,  -- Encrypted refresh token
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

**Connection Flow:**

1. User clicks "Connect" on a Composio app in the marketplace
2. Frontend redirects to Composio OAuth consent page with `entity_id = workspace_{workspace_id}`
3. User grants permissions
4. Composio callback stores encrypted tokens in `entity_connections`
5. Agents can now use the connected app via `ComposioToolService`

**Permission Validation** ([modules/tools/execution/unified_executor.py:1-800]()):

```python
def validate_action_for_intent(
    agent_id: int,
    app_name: str,
    workspace_id: UUID,
    db: Session
) -> bool:
    """Validate agent has permission to use app in workspace."""
    # Check agent assignment
    assignment = db.query(AgentAppAssignment).filter(
        AgentAppAssignment.agent_id == agent_id,
        AgentAppAssignment.app_name == app_name,
        AgentAppAssignment.is_active == True
    ).first()
    
    # Check workspace connection
    connection = db.query(EntityConnection).filter(
        EntityConnection.workspace_id == workspace_id,
        EntityConnection.app_name == app_name,
        EntityConnection.connection_status == 'active'
    ).first()
    
    return assignment is not None and connection is not None
```

**Sources:** [orchestrator/modules/tools/services/composio_tool_service.py:1-360](), [orchestrator/modules/tools/execution/unified_executor.py:1-800]()

---

## Frontend Integration

### Clerk React SDK Setup

The frontend uses Clerk's React SDK to handle authentication UI and token management.

**Provider Configuration** ([frontend/app/layout.tsx or similar]()):

```tsx
<ClerkProvider publishableKey={process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY}>
  <SignedIn>
    {/* Authenticated app content */}
  </SignedIn>
  <SignedOut>
    <RedirectToSignIn />
  </SignedOut>
</ClerkProvider>
```

**Token Injection** ([frontend/lib/api-client.ts:157-163]()):

```typescript
public setClerkTokenGetter(getter: () => Promise<string | null>) {
  this.getClerkToken = getter
  console.log('✅ Clerk token getter configured')
}
```

The `ApiClient` class accepts a token getter function from components with access to Clerk's `useAuth()` hook, then includes the JWT in all API requests.

**Sources:** [frontend/lib/api-client.ts:157-163](), [orchestrator/config.py:168-171]()

---

### Workspace Selection

The frontend maintains workspace context in `localStorage` and provides admin override capabilities for support scenarios.

**Workspace Header Injection** ([frontend/lib/api-client.ts:803-816]()):

```typescript
async getAuthHeaders(): Promise<Record<string, string>> {
  const headers: Record<string, string> = {}
  
  // Admin override takes priority over localStorage
  const workspaceId = _adminWorkspaceOverride || 
                      localStorage.getItem('last_active_workspace') ||
                      localStorage.getItem('last_active_org')
  
  if (workspaceId) {
    headers['X-Workspace-ID'] = workspaceId
  }
  
  if (this.getClerkToken) {
    const token = await this.getClerkToken()
    if (token) headers['Authorization'] = `Bearer ${token}`
  }
  
  return headers
}
```

**Admin Override Functions** ([frontend/lib/api-client.ts:83-91]()):

```typescript
let _adminWorkspaceOverride: string | null = null

export function setAdminWorkspaceOverride(wsId: string | null) {
  _adminWorkspaceOverride = wsId
}

export function getAdminWorkspaceOverride(): string | null {
  return _adminWorkspaceOverride
}
```

This allows admin users to temporarily impersonate workspaces for debugging without changing `localStorage`.

**Sources:** [frontend/lib/api-client.ts:803-816](), [frontend/lib/api-client.ts:83-91]()

---

## Security Considerations

### Production Hardening

**Security Headers** ([main.py:617-627]()):

```python
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
    
    if config.ENVIRONMENT == "production":
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"
    
    return response
```

**Database SSL Enforcement** ([config.py:46-58]()):

```python
@staticmethod
def get_database_url() -> str:
    """Return DATABASE_URL with sslmode=require enforced for non-local hosts."""
    url = Config.DATABASE_URL
    if not url:
        return url
    
    _local_hosts = ("localhost", "127.0.0.1", "postgres", "db")
    if any(h in url for h in _local_hosts):
        return url
    
    if "sslmode" not in url:
        url += "?sslmode=require" if "?" not in url else "&sslmode=require"
    
    return url
```

Production database connections enforce SSL mode unless connecting to local development databases.

**Sources:** [orchestrator/main.py:617-627](), [orchestrator/config.py:46-58]()

---

### Rate Limiting

SlowAPI provides IP-based rate limiting with configurable limits per endpoint.

**Configuration** ([main.py:583-596]()):

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

def _get_real_client_ip(request) -> str:
    """Extract real client IP, respecting X-Forwarded-For behind reverse proxy."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return get_remote_address(request)

limiter = Limiter(key_func=_get_real_client_ip, default_limits=["60/minute"])
app.state.limiter = limiter
```

The rate limiter respects `X-Forwarded-For` headers when behind reverse proxies (e.g., Railway, Cloudflare) to correctly identify client IPs.

**Sources:** [orchestrator/main.py:583-596]()

---

### Request Body Size Limits

The backend enforces different body size limits based on endpoint type to prevent resource exhaustion.

**Middleware** ([main.py:598-614]()):

```python
MAX_BODY_SIZE = 10 * 1024 * 1024  # 10MB
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB
UPLOAD_PATHS = ("/api/documents/upload", "/api/admin/plugins/upload", "/api/documents/templates/upload")

@app.middleware("http")
async def limit_request_body(request, call_next):
    content_length = request.headers.get("content-length")
    limit = MAX_UPLOAD_SIZE if any(request.url.path.startswith(p) for p in UPLOAD_PATHS) else MAX_BODY_SIZE
    
    if content_length:
        if int(content_length) > limit:
            return JSONResponse(status_code=413, content={"detail": "Payload too large"})
    
    return await call_next(request)
```

**Limits:**

- **Default**: 10MB for API requests
- **Uploads**: 50MB for document/plugin uploads

**Sources:** [orchestrator/main.py:598-614]()

---

### Redis Security Hardening

The Redis instance is hardened by disabling dangerous administrative commands.

**Disabled Commands** ([docker-compose.yml:54-62]()):

```yaml
command: >
  redis-server
  --requirepass ${REDIS_PASSWORD}
  --rename-command FLUSHDB ""
  --rename-command FLUSHALL ""
  --rename-command DEBUG ""
```

This prevents accidental or malicious data wipes if the Redis instance is exposed (e.g., misconfigured firewall). The `CONFIG` command is also renamed, breaking `redis-cli CONFIG` but preventing runtime configuration changes.

**Sources:** [docker-compose.yml:54-62]()

---

## Environment Configuration

### Required Variables

**Minimum Configuration** ([.env.example:1-65]()):

```bash
# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=orchestrator_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_database_password_here

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password_here

# Authentication
CLERK_SECRET_KEY=sk_test_...
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_...
CLERK_JWKS_URL=https://your-clerk-domain.clerk.accounts.dev/.well-known/jwks.json

# Workspace
DEFAULT_WORKSPACE_ID=00000000-0000-0000-0000-000000000000

# Encryption
CREDENTIAL_ENCRYPTION_KEY=your_32_char_encryption_key_here

# API Security
API_KEY=your_secure_api_key_here
REQUIRE_AUTH=true  # false for local dev, true for production

# CORS
CORS_ALLOW_ORIGINS=http://localhost:3000,https://yourdomain.com
```

**Sources:** [orchestrator/.env.example:1-65]()

---

### Docker Compose Configuration

**Backend Service** ([docker-compose.yml:78-138]()):

```yaml
backend:
  environment:
    # Database
    DATABASE_URL: postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
    
    # Redis
    REDIS_HOST: redis
    REDIS_PASSWORD: ${REDIS_PASSWORD}
    
    # Authentication
    CLERK_SECRET_KEY: ${CLERK_SECRET_KEY}
    NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY: ${NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY}
    CLERK_JWKS_URL: ${CLERK_JWKS_URL}
    
    # Security
    API_KEY: ${API_KEY}
    
  volumes:
    - backend_logs:/app/logs
    - workspace_data:/workspaces:ro  # Read-only for code viewer
```

**Frontend Service** ([docker-compose.yml:146-170]()):

```yaml
frontend:
  environment:
    NEXT_PUBLIC_API_URL: ${NEXT_PUBLIC_API_URL:-http://localhost:8000}
    NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY: ${NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY}
```

The frontend only receives public-safe environment variables (`NEXT_PUBLIC_*` prefix). Secrets like `CLERK_SECRET_KEY` stay server-side only.

**Sources:** [docker-compose.yml:78-170]()

---

## Development Workflow

### Local Development Setup

1. **Copy Environment Template**:
   ```bash
   cp orchestrator/.env.example .env
   ```

2. **Configure Clerk** (optional for local dev):
   - Create a Clerk application at https://clerk.com
   - Copy API keys to `.env`
   - Set `REQUIRE_AUTH=false` to bypass authentication locally

3. **Start Services**:
   ```bash
   docker-compose up --build
   ```

4. **Access Application**:
   - Frontend: http://localhost:3000
   - Backend: http://localhost:8000
   - API Docs: http://localhost:8000/docs

**Development Mode Authentication:**

When `REQUIRE_AUTH=false`, the system allows unauthenticated requests and assigns a default workspace ID. This simplifies local testing without requiring Clerk setup.

**Sources:** [orchestrator/config.py:87-88](), [orchestrator/.env.example:33]()

---

### Testing Multi-Tenancy

**Manual Workspace Testing:**

1. Create multiple workspaces via API or UI
2. Switch workspace context by changing `X-Workspace-ID` header
3. Verify data isolation:
   - Agents from workspace A should not appear in workspace B
   - Document uploads should use workspace-prefixed S3 keys
   - Redis cache keys should include workspace ID

**Admin Override Testing** ([frontend/lib/api-client.ts:83-91]()):

```typescript
// In browser console
import { setAdminWorkspaceOverride } from '@/lib/api-client'

// Switch to workspace UUID
setAdminWorkspaceOverride('abc-123-def-456')

// Reset to user's actual workspace
setAdminWorkspaceOverride(null)
```

**Sources:** [frontend/lib/api-client.ts:83-91]()

---

## Migration from Single-Tenant

### Adding Workspace Support to Existing Tables

**Migration Template**:

```sql
-- Add workspace_id column
ALTER TABLE your_table 
ADD COLUMN workspace_id UUID 
REFERENCES workspaces(id) ON DELETE CASCADE;

-- Backfill existing rows with default workspace
UPDATE your_table 
SET workspace_id = '00000000-0000-0000-0000-000000000000' 
WHERE workspace_id IS NULL;

-- Make column non-nullable
ALTER TABLE your_table 
ALTER COLUMN workspace_id SET NOT NULL;

-- Add index for query performance
CREATE INDEX idx_your_table_workspace ON your_table(workspace_id);
```

**Update Query Logic**:

```python
# Before (single-tenant)
agents = db.query(Agent).filter(Agent.status == "active").all()

# After (multi-tenant)
agents = db.query(Agent).filter(
    Agent.workspace_id == ctx.workspace_id,
    Agent.status == "active"
).all()
```

**Sources:** [orchestrator/core/models/core.py:1-500]()

---

## Summary

| Component | Isolation Method | Key Configuration |
|-----------|-----------------|-------------------|
| **Authentication** | Clerk JWT validation | `CLERK_SECRET_KEY`, `CLERK_JWKS_URL` |
| **Database** | `workspace_id` column + filters | Foreign key to `workspaces` table |
| **S3 Storage** | Path prefix `workspaces/{id}/` | Per-workspace vector buckets |
| **Redis Cache** | Key prefix `workspace:ws:{id}:` | Workspace-scoped keys |
| **Filesystem** | Volume path `/workspaces/{id}/` | `WORKSPACE_VOLUME_PATH` env var |
| **Credentials** | Encrypted per workspace | `CREDENTIAL_ENCRYPTION_KEY` |
| **Rate Limiting** | IP-based (60/min default) | SlowAPI middleware |
| **Request Context** | `get_request_context_hybrid` | `X-Workspace-ID` header |

Automatos AI's multi-tenancy architecture ensures complete data isolation through layered security controls, allowing thousands of workspaces to operate securely on shared infrastructure.

**Sources:** [orchestrator/main.py:1-800](), [orchestrator/config.py:1-423](), [frontend/lib/api-client.ts:1-1500]()

---