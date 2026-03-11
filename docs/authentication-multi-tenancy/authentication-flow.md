# Authentication Flow

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [frontend/app/admin/plugins/page.tsx](frontend/app/admin/plugins/page.tsx)
- [frontend/lib/api-client.ts](frontend/lib/api-client.ts)
- [orchestrator/.env.example](orchestrator/.env.example)
- [orchestrator/api/agent_plugins.py](orchestrator/api/agent_plugins.py)
- [orchestrator/config.py](orchestrator/config.py)
- [orchestrator/core/database/load_seed_data.py](orchestrator/core/database/load_seed_data.py)
- [orchestrator/core/seeds/seed_personas.py](orchestrator/core/seeds/seed_personas.py)
- [orchestrator/core/seeds/seed_plugin_categories.py](orchestrator/core/seeds/seed_plugin_categories.py)
- [orchestrator/core/services/plugin_cache.py](orchestrator/core/services/plugin_cache.py)
- [orchestrator/main.py](orchestrator/main.py)
- [scripts/ralph/prd.json](scripts/ralph/prd.json)

</details>



## Purpose and Scope

This document describes the authentication and authorization mechanisms in the Automatos AI platform, covering how incoming HTTP requests are validated, how user identity and workspace context are established, and how this information flows through the system. For information about workspace isolation and multi-tenancy data filtering, see [Data Isolation](#12.3). For credential management and storage, see [Credentials Management](#12.4).

---

## Authentication Architecture Overview

The Automatos AI platform implements a **hybrid authentication system** that supports two authentication methods:

1. **Clerk JWT tokens** - For user sessions via the web frontend
2. **API keys** - For programmatic access and service-to-service communication

All authenticated endpoints receive a `RequestContext` object containing the validated user identity, workspace ID, and authorization metadata. This context flows through the entire request lifecycle, ensuring workspace isolation and proper access control.

**Sources:** [orchestrator/main.py:1-50](), [Diagram 7: Multi-Tenancy and Security Architecture]()

---

## Authentication Methods

### Clerk JWT Authentication

The primary authentication method for web users is Clerk JWT validation. The backend validates JWT tokens against Clerk's JWKS endpoint to verify authenticity.

**Key characteristics:**
- Tokens provided in `Authorization: Bearer <token>` header
- Validated against Clerk JWKS (JSON Web Key Set)
- Workspace ID extracted from JWT claims or resolved via org_id
- Session-based authentication for interactive users

**Configuration:**
- Backend requires `CLERK_SECRET_KEY` and optionally `CLERK_JWKS_URL` for JWT validation
- Frontend requires `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` for Clerk SDK initialization
- Optional `CLERK_AUDIENCE` for JWT audience validation
- JWT validation happens on every request via `get_request_context_hybrid` dependency

**Sources:** [orchestrator/config.py:169-175]()

### API Key Authentication

For programmatic access (scripts, integrations, automation), the platform supports API key authentication:

**Key characteristics:**
- Provided in `X-API-Key` header
- Matched against `ORCHESTRATOR_API_KEY` (with fallbacks to `AUTOMATOS_API_KEY` or `API_KEY`)
- Grants admin-level access across all workspaces
- Bypasses Clerk JWT validation entirely

**Configuration:**
```bash
ORCHESTRATOR_API_KEY=your_secure_api_key_here
REQUIRE_API_KEY=true  # Enable/disable API key requirement
REQUIRE_AUTH=true     # Master auth switch (default: true)
AUTH_DEBUG=false      # Enable auth debug logging
```

The configuration uses a fallback chain: `ORCHESTRATOR_API_KEY` → `AUTOMATOS_API_KEY` → `API_KEY`, allowing flexibility in naming conventions.

**Sources:** [orchestrator/config.py:84-88]()

---

## Hybrid Authentication System

### Request Authentication Flow

```mermaid
graph TD
    Request["Incoming HTTP Request"] --> Middleware["CORS + Security Headers Middleware"]
    Middleware --> RateLimiter["Rate Limiter<br/>(slowapi: 60/min per IP)"]
    RateLimiter --> BodySizeLimit["Body Size Limit Middleware<br/>10MB default, 50MB uploads"]
    BodySizeLimit --> SecurityHeaders["Security Headers Middleware<br/>OWASP headers"]
    SecurityHeaders --> RequestID["Request ID Middleware<br/>X-Request-ID injection"]
    RequestID --> APITracking["API Tracking Middleware<br/>Performance metrics"]
    APITracking --> HybridAuth["get_request_context_hybrid<br/>FastAPI Depends"]
    
    HybridAuth --> CheckAPIKey{"X-API-Key<br/>header present?"}
    
    CheckAPIKey -->|Yes| ValidateAPIKey["Validate against<br/>config.API_KEY"]
    ValidateAPIKey --> APIKeyValid{"Valid?"}
    APIKeyValid -->|Yes| CreateAdminContext["Create RequestContext<br/>auth_type='api_key'<br/>admin privileges"]
    APIKeyValid -->|No| Reject401["HTTP 401<br/>Invalid API key"]
    
    CheckAPIKey -->|No| CheckJWT{"Authorization<br/>Bearer header?"}
    CheckJWT -->|Yes| ValidateJWT["Validate JWT<br/>against Clerk JWKS"]
    ValidateJWT --> JWTValid{"Valid?"}
    JWTValid -->|Yes| ExtractClaims["Extract workspace_id<br/>from JWT claims"]
    ExtractClaims --> ResolveWorkspace["Resolve workspace<br/>validate user access"]
    ResolveWorkspace --> CreateUserContext["Create RequestContext<br/>auth_type='jwt'<br/>workspace_id set"]
    JWTValid -->|No| Reject401
    
    CheckJWT -->|No| CheckRequired{"REQUIRE_API_KEY<br/>config?"}
    CheckRequired -->|true| Reject401
    CheckRequired -->|false| CreateAnonymous["Create RequestContext<br/>auth_type=None<br/>development mode"]
    
    CreateAdminContext --> InjectContext["Inject RequestContext<br/>into FastAPI Depends"]
    CreateUserContext --> InjectContext
    CreateAnonymous --> InjectContext
    
    InjectContext --> RouteHandler["Route Handler<br/>receives ctx: RequestContext"]
    Reject401 --> ErrorResponse["Return 401 response<br/>to client"]
```

**Sources:** [orchestrator/main.py:560-688](), [Diagram 2: Request Processing Pipeline]()

---

## RequestContext Structure

The `RequestContext` object is the central data structure for authentication and authorization throughout the platform:

```python
@dataclass
class RequestContext:
    workspace_id: UUID          # Active workspace for request
    user: User                  # Authenticated user object (or None for API key)
    auth_type: str              # 'jwt' | 'api_key' | None
    admin_all_workspaces: bool  # True if admin override enabled
```

**Key properties:**

| Field | Type | Description |
|-------|------|-------------|
| `workspace_id` | `UUID` | The workspace context for the request. All database queries are filtered by this ID. |
| `user` | `User \| None` | The authenticated user object from the database. None for API key auth. |
| `auth_type` | `str \| None` | Authentication method used: `'jwt'`, `'api_key'`, or `None` (dev mode). |
| `admin_all_workspaces` | `bool` | If True, enables cross-workspace admin queries (see Admin Override Pattern). |

**Sources:** [orchestrator/main.py:17](), [orchestrator/core/auth/hybrid.py]() (imported but not provided in context)

---

## Workspace Resolution

### Workspace ID Injection

The workspace context is established through a two-step process:

1. **Frontend injection**: The `X-Workspace-ID` header is injected by the frontend
2. **Backend validation**: The backend validates the user has access to the workspace

```mermaid
graph TD
    Frontend["Frontend API Client"] --> CheckOverride{"Admin workspace<br/>override active?"}
    
    CheckOverride -->|Yes| UseOverride["Use _adminWorkspaceOverride<br/>from module state"]
    CheckOverride -->|No| CheckLocalStorage{"X-Workspace-ID<br/>in request options?"}
    
    CheckLocalStorage -->|Yes| UseProvided["Use provided<br/>workspace ID"]
    CheckLocalStorage -->|No| ReadStorage["Read from localStorage:<br/>1. last_active_workspace<br/>2. last_active_org"]
    
    UseOverride --> InjectHeader["Inject X-Workspace-ID<br/>header in request"]
    UseProvided --> InjectHeader
    ReadStorage --> InjectHeader
    
    InjectHeader --> SendRequest["Send HTTP request<br/>to backend"]
    
    SendRequest --> BackendAuth["Backend: get_request_context_hybrid"]
    BackendAuth --> ExtractHeader["Extract X-Workspace-ID<br/>from headers"]
    
    ExtractHeader --> ValidateAccess["Query workspace_members table<br/>validate user membership"]
    ValidateAccess --> AccessValid{"User has<br/>access?"}
    
    AccessValid -->|Yes| SetContext["Set workspace_id<br/>in RequestContext"]
    AccessValid -->|No| CheckDefault{"Has default<br/>workspace?"}
    
    CheckDefault -->|Yes| UseDefault["Use user's default<br/>workspace from org_id"]
    CheckDefault -->|No| Return403["HTTP 403<br/>Forbidden"]
    
    UseDefault --> SetContext
    SetContext --> ContinueRequest["Request continues<br/>with workspace context"]
```

**Special case: Bootstrap mode**

For new installations with ≤2 active workspaces, the first user is automatically promoted to admin with cross-workspace access. This enables initial setup without manual admin assignment.

**Single-Tenant Mode:**

The platform supports a single-tenant deployment mode with a default tenant UUID:
```python
DEFAULT_TENANT_ID = UUID("00000000-0000-0000-0000-000000000000")
```

This simplifies deployment for organizations that don't require multi-tenancy.

**Sources:** [frontend/lib/api-client.ts:854-862](), [orchestrator/config.py:19-22](), [Diagram 8: Multi-Tenancy & Security Model]()

---

## Admin Workspace Override

Administrators can view platform-wide analytics and data across all workspaces using a special **`__all__` sentinel value** in the `X-Workspace-ID` header.

### Override Mechanism

```mermaid
graph TD
    AdminUI["Admin Dashboard"] --> SetOverride["User selects '__all__'<br/>in workspace dropdown"]
    SetOverride --> CallSetter["setAdminWorkspaceOverride('__all__')"]
    
    CallSetter --> UpdateModule["Update module-level<br/>_adminWorkspaceOverride"]
    UpdateModule --> InvalidateCache["Invalidate React Query<br/>cache for affected queries"]
    
    InvalidateCache --> NextRequest["Next API request"]
    NextRequest --> ApiClient["ApiClient.request()"]
    
    ApiClient --> CheckOverride{"_adminWorkspaceOverride<br/>set?"}
    CheckOverride -->|Yes| UseOverride["Use override value<br/>('__all__')"]
    CheckOverride -->|No| UseDefault["Use localStorage<br/>workspace ID"]
    
    UseOverride --> InjectHeader["Inject X-Workspace-ID: __all__<br/>in request headers"]
    UseDefault --> InjectHeaderNormal["Inject normal<br/>workspace ID"]
    
    InjectHeader --> BackendAuth["Backend: get_request_context_hybrid"]
    BackendAuth --> CheckSentinel{"X-Workspace-ID<br/>== '__all__'?"}
    
    CheckSentinel -->|Yes| CheckAdmin{"User is<br/>system admin?"}
    CheckAdmin -->|Yes| SetAdminContext["Set admin_all_workspaces=True<br/>Use admin's home workspace"]
    CheckAdmin -->|No| Return403["HTTP 403<br/>Admin privileges required"]
    
    CheckSentinel -->|No| NormalFlow["Normal workspace<br/>validation"]
    
    SetAdminContext --> QueryHandler["Route handler<br/>skips workspace filters<br/>if admin_all_workspaces=True"]
    NormalFlow --> QueryHandlerNormal["Route handler<br/>applies workspace filter"]
    
    InjectHeaderNormal --> BackendAuth
```

**Implementation details:**

- **Frontend**: `_adminWorkspaceOverride` module-level variable in `api-client.ts` [frontend/lib/api-client.ts:83-91]()
- **Backend**: Checks `admin_all_workspaces` flag in `RequestContext` to skip workspace filters
- **Security**: Only users with `system_role='admin'` or bootstrap bypass can use this feature
- **Cache invalidation**: React Query cache is invalidated when override changes to prevent stale data

**Sources:** [frontend/lib/api-client.ts:83-91, 854-862](), [Diagram 7: Multi-Tenancy and Security Architecture]()

---

## Frontend Authentication Integration

### API Client Configuration

The frontend `ApiClient` class in `api-client.ts` handles authentication injection for all API requests.

**Clerk Token Injection Flow:**

```mermaid
graph TD
    Component["React Component"] --> UseAuth["useAuth() hook<br/>from @clerk/nextjs"]
    UseAuth --> GetToken["getToken() function<br/>provided by Clerk"]
    
    GetToken --> ConfigClient["ApiClient.setClerkTokenGetter()<br/>Register getter function"]
    
    ConfigClient --> ApiRequest["apiClient.request(endpoint)"]
    ApiRequest --> CheckGetter{"getClerkToken<br/>function set?"}
    
    CheckGetter -->|Yes| CallGetter["await getClerkToken()"]
    CallGetter --> Timeout["Promise.race with<br/>2s timeout"]
    
    Timeout --> GotToken{"Token received<br/>within 2s?"}
    GotToken -->|Yes| InjectBearer["Inject Authorization:<br/>Bearer {token}"]
    GotToken -->|No| WarnTimeout["Log timeout warning<br/>proceed without auth"]
    
    CheckGetter -->|No| WarnNoGetter["Log 'no token available'<br/>warning"]
    
    InjectBearer --> InjectWorkspace["Inject X-Workspace-ID<br/>from localStorage or override"]
    WarnTimeout --> InjectWorkspace
    WarnNoGetter --> InjectWorkspace
    
    InjectWorkspace --> FetchAPI["fetch(url, headers)"]
    FetchAPI --> Response["Response from backend"]
    
    Response --> Check401{"Status 401?"}
    Check401 -->|Yes| ThrowAuthError["Throw error:<br/>'HTTP 401: Unauthorized'"]
    Check401 -->|No| ReturnData["Return response data"]
```

**Key implementation details:**

- **Token getter**: Set via `setClerkTokenGetter()` from a component with `useAuth()` access [frontend/lib/api-client.ts:160-163]()
- **Timeout protection**: 2-second timeout prevents hanging if Clerk is slow [frontend/lib/api-client.ts:832-835]()
- **Workspace injection**: Always includes `X-Workspace-ID` header for multi-tenancy [frontend/lib/api-client.ts:854-862]()
- **Error handling**: 401 responses throw descriptive errors prompting sign-in [frontend/lib/api-client.ts:896-902]()

**Sources:** [frontend/lib/api-client.ts:93-154, 819-909]()

---

## Middleware Stack

The authentication flow is part of a larger middleware stack that processes every request:

```mermaid
graph TD
    Request["HTTP Request"] --> CORS["CORSMiddleware<br/>Allow configured origins<br/>Expose routing headers"]
    
    CORS --> RateLimit["Rate Limiter<br/>slowapi: 60/min per IP<br/>X-Forwarded-For aware"]
    
    RateLimit --> SecurityHeaders["Security Headers Middleware<br/>X-Content-Type-Options: nosniff<br/>X-Frame-Options: DENY<br/>Content-Security-Policy<br/>HSTS (production only)"]
    
    SecurityHeaders --> RequestID["Request ID Middleware<br/>Generate or extract X-Request-ID<br/>Set request context for logging"]
    
    RequestID --> APITracking["API Tracking Middleware<br/>Record call counts<br/>Track response times<br/>Monitor error rates"]
    
    APITracking --> RouteMatch["FastAPI Route Matching<br/>URL pattern matching"]
    
    RouteMatch --> AuthDepends["get_request_context_hybrid<br/>FastAPI Depends injection<br/>Validate authentication<br/>Build RequestContext"]
    
    AuthDepends --> RouteHandler["Route Handler<br/>ctx: RequestContext parameter<br/>Business logic execution"]
    
    RouteHandler --> ResponseHeaders["Response Headers<br/>X-Request-ID<br/>X-Routing-* (if routed)<br/>Security headers"]
    
    ResponseHeaders --> Response["HTTP Response"]
```

**Middleware responsibilities:**

| Middleware | Purpose | Configuration |
|------------|---------|---------------|
| **CORS** | Allow cross-origin requests from configured domains | `CORS_ALLOW_ORIGINS` env var (comma-separated list) |
| **Widget CORS** | Special CORS handling for embeddable widgets | Optional `WidgetCORSMiddleware` if widget API available |
| **Widget Rate Limit** | Separate rate limiting for widget SDK requests | Optional `WidgetRateLimitMiddleware` |
| **Rate Limiter** | Prevent abuse with per-IP rate limiting (60/min default) | `slowapi` with `_get_real_client_ip` respecting `X-Forwarded-For` |
| **Body Size Limit** | Enforce request size limits | 10MB default, 50MB for upload endpoints |
| **Security Headers** | OWASP-recommended security headers | `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, `Permissions-Policy`, CSP, HSTS (production only) |
| **Request ID** | Unique ID for tracing and log correlation | Auto-generated UUID or from `X-Request-ID` header, exposed in response |
| **API Tracking** | Performance metrics and endpoint health monitoring | In-memory deque (last 100 calls/endpoint), max 500 endpoints tracked |

**Implementation details:**

- **CORS origins parsing**: Splits comma-separated origins and strips whitespace [orchestrator/main.py:557-558]()
- **Real IP extraction**: Uses `X-Forwarded-For` header when behind reverse proxy [orchestrator/main.py:587-592]()
- **Request ID context**: Uses contextvars for thread-safe request ID tracking [orchestrator/main.py:632-641]()
- **Stats capping**: Limits stats dictionary growth to prevent memory exhaustion [orchestrator/main.py:670-673]()

**Sources:** [orchestrator/main.py:560-688](), [orchestrator/config.py:98-99](), [Diagram 2: Request Processing Pipeline]()

---

## Configuration Reference

### Environment Variables

Authentication behavior is controlled by these configuration settings:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ORCHESTRATOR_API_KEY` | string | None | Primary secret key for API key authentication (fallback chain: `AUTOMATOS_API_KEY` → `API_KEY`) |
| `REQUIRE_API_KEY` | boolean | `true` | Enforce API key requirement |
| `REQUIRE_AUTH` | boolean | `true` | Master authentication switch |
| `AUTH_DEBUG` | boolean | `false` | Enable detailed auth logging |
| `CORS_ALLOW_ORIGINS` | string | `http://localhost:3000` | Comma-separated list of allowed origins |
| `CLERK_SECRET_KEY` | string | None | Clerk backend API key for JWT validation |
| `CLERK_JWKS_URL` | string | None | Clerk JWKS endpoint URL (optional) |
| `CLERK_AUDIENCE` | string | Empty | JWT audience claim validation (optional) |
| `DEFAULT_WORKSPACE_ID` | string | None | Default workspace for single-tenant mode |

**Development mode:**

For local development without Clerk setup, set `REQUIRE_AUTH=false` to bypass authentication:

```bash
# .env
REQUIRE_AUTH=false
```

This creates anonymous `RequestContext` objects with `auth_type=None`. **Never use in production.**

**Production security:**

The configuration enforces SSL for database connections in production (non-local hosts):

```python
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

This ensures secure connections to remote databases (PRD-70 FIX-05).

**Sources:** [orchestrator/config.py:84-88, 169-175, 47-58](), [orchestrator/.env.example:33-35]()

---

## Security Considerations

### Authentication Security

1. **JWT Validation**: All Clerk JWTs are validated against public JWKS on every request - no local storage of tokens
2. **API Key Security**: API keys grant full admin access - rotate regularly and store securely
3. **Rate Limiting**: 60 requests/minute per IP prevents brute force attacks
4. **Security Headers**: OWASP-recommended headers protect against XSS, clickjacking, and MIME sniffing

### Workspace Isolation

1. **Workspace ID filtering**: All database queries include `WHERE workspace_id = :workspace_id` clause
2. **Access validation**: Backend validates user membership in requested workspace
3. **Admin override logging**: All admin cross-workspace queries are logged in `routing_decisions` table
4. **Bootstrap bypass**: Limited to deployments with ≤2 workspaces (pilot mode)

**Sources:** [Diagram 7: Multi-Tenancy and Security Architecture](), [orchestrator/main.py:464-474]()

---

## Error Handling

### Authentication Failures

| Error Code | Condition | Response |
|------------|-----------|----------|
| **401 Unauthorized** | Missing or invalid JWT token | `{"detail": "HTTP 401: Unauthorized (missing/invalid Clerk token)"}` |
| **401 Unauthorized** | Invalid API key | `{"detail": "Invalid API key"}` |
| **403 Forbidden** | User lacks access to requested workspace | `{"detail": "Access denied: workspace mismatch"}` |
| **403 Forbidden** | Non-admin attempts to use `__all__` override | `{"detail": "Admin privileges required"}` |
| **429 Too Many Requests** | Rate limit exceeded (>60/min) | Standard slowapi rate limit response |

**Frontend error handling:**

The `ApiClient` class throws errors with descriptive messages for authentication failures:

```typescript
if (response.status === 401) {
  throw new Error(
    'HTTP 401: Unauthorized (missing/invalid Clerk token). ' +
    'Make sure you are signed in and the API client is configured with Clerk.'
  )
}
```

**Sources:** [frontend/lib/api-client.ts:896-902](), [orchestrator/main.py:449-461]()

---

## Code Entity Reference

### Key Functions and Classes

| Entity | Location | Purpose |
|--------|----------|---------|
| `get_request_context_hybrid` | `orchestrator/core/auth/hybrid.py` | FastAPI dependency that validates auth and builds `RequestContext` |
| `RequestContext` | `orchestrator/core/auth/dependencies.py` | Dataclass containing authenticated user, workspace, and auth metadata |
| `ApiClient` | `frontend/lib/api-client.ts` | Frontend HTTP client with Clerk integration and workspace injection |
| `setAdminWorkspaceOverride` | `frontend/lib/api-client.ts:85-87` | Module-level function to set admin workspace override |
| `Config.API_KEY` | `orchestrator/config.py:67` | API key configuration property |
| `Config.REQUIRE_API_KEY` | `orchestrator/config.py:68` | Toggle for authentication requirement |
| `Config.CORS_ALLOW_ORIGINS` | `orchestrator/config.py:73-79` | CORS origin configuration |

**Sources:** [orchestrator/main.py:18](), [orchestrator/config.py:66-79](), [frontend/lib/api-client.ts:83-163, 819-909]()

---