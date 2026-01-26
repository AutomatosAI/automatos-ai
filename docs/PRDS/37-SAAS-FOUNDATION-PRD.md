# PRD-37: SaaS Foundation - Authentication, Workspaces & Billing

**Version:** 2.0  
**Status:** 🟡 Planning Phase  
**Date:** January 20, 2026  
**Author:** DeepAgent  
**Prerequisites:** None (Foundation Layer)  
**Blocks:** PRD-36 (Composio Integration)

---

## Executive Summary

This PRD establishes the foundational SaaS infrastructure for Automatos using a **simplified workspace model**:

1. **Clerk Authentication** - User auth, SSO, MFA, organizations
2. **Clerk Billing** - Subscription plans and payments (beta, ready by launch)
3. **Workspaces** - Auto-created on signup, team member invites
4. **Usage Tracking** - Token and tool call logging
5. **API Keys** - For widget/embed integration

### Architecture Decision

Every user gets their own **workspace** on signup:
- **Solo users** → Personal workspace (Starter plan)
- **Small business** → Team workspace with members (Business plan)
- **Enterprise** → Large team workspace (Enterprise plan)

### Key Simplifications from v1.0

| v1.0 (Complex) | v2.0 (Simplified) |
|----------------|-------------------|
| Tenants → Multi-tenant | Workspaces → Simple |
| 4 roles (owner/admin/member/viewer) | 2 roles (owner/member) |
| Stripe + webhooks | Clerk Billing (unified) |
| Sub-organizations | Removed (future) |
| Per-seat billing | Flat plan pricing |

### Timeline

**~9 days** implementation (down from 14 days)

---

## Table of Contents

1. [Architecture](#architecture)
2. [Database Schema](#database-schema)
3. [Backend Implementation](#backend-implementation)
4. [Frontend Implementation](#frontend-implementation)
5. [Clerk Billing Setup](#clerk-billing-setup)
6. [Implementation Roadmap](#implementation-roadmap)

---

## Architecture

### User Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           USER SIGNS UP                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  1. User clicks "Sign Up" → Clerk handles (email, Google, GitHub)       │
│  2. On first login → Workspace auto-created                             │
│  3. User lands in their personal workspace                               │
│  4. User can invite team members (becomes team workspace)               │
│  5. Clerk Billing handles plan upgrades                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Model

```
┌──────────────────┐       ┌──────────────────┐
│    workspaces    │       │       users      │
├──────────────────┤       ├──────────────────┤
│ id (UUID) PK     │       │ id SERIAL PK     │
│ name             │       │ clerk_user_id    │◄── From Clerk
│ slug             │       │ email            │
│ owner_id FK      │──────►│ name             │
│ clerk_org_id     │       │ avatar_url       │
│ plan             │       │ last_sign_in     │
│ plan_limits      │       │ created_at       │
│ settings JSONB   │       └──────────────────┘
│ created_at       │                │
└──────────────────┘                │
         │                          │
         ▼                          ▼
┌──────────────────────────────────────────────┐
│           workspace_members                   │
├──────────────────────────────────────────────┤
│ workspace_id UUID FK                          │
│ user_id INTEGER FK                            │
│ role (owner | member)                         │
│ joined_at                                     │
└──────────────────────────────────────────────┘
```

### Pricing Plans

| Plan | Price | Target | Limits |
|------|-------|--------|--------|
| **Starter** | Free or $9/mo | Solo users | 3 agents, 10 workflows, 5 docs |
| **Business** | $29/mo | Small teams | 10 agents, 50 workflows, 10 members |
| **Enterprise** | $99/mo | Large teams | Unlimited, priority support |

---

## Database Schema

### New Tables (Add to init_complete_schema.sql)

```sql
-- ================================================================
-- SAAS FOUNDATION TABLES (PRD-37 v2.0)
-- ================================================================

-- Workspaces (simplified from tenants)
CREATE TABLE IF NOT EXISTS workspaces (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) UNIQUE,
    owner_id INTEGER NOT NULL REFERENCES users(id),
    clerk_org_id VARCHAR(255) UNIQUE,
    plan VARCHAR(50) DEFAULT 'starter' CHECK (plan IN ('starter', 'business', 'enterprise')),
    plan_limits JSONB DEFAULT '{
        "max_agents": 3,
        "max_workflows": 10,
        "max_documents": 5,
        "max_members": 1
    }'::jsonb,
    settings JSONB DEFAULT '{}'::jsonb,
    is_personal BOOLEAN DEFAULT TRUE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_workspaces_owner ON workspaces(owner_id);
CREATE INDEX idx_workspaces_clerk_org ON workspaces(clerk_org_id);
CREATE INDEX idx_workspaces_slug ON workspaces(slug);
CREATE INDEX idx_workspaces_plan ON workspaces(plan);

-- Workspace Members
CREATE TABLE IF NOT EXISTS workspace_members (
    id SERIAL PRIMARY KEY,
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL DEFAULT 'member' CHECK (role IN ('owner', 'member')),
    invited_by INTEGER REFERENCES users(id),
    invited_at TIMESTAMP,
    joined_at TIMESTAMP DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(workspace_id, user_id)
);

CREATE INDEX idx_members_workspace ON workspace_members(workspace_id);
CREATE INDEX idx_members_user ON workspace_members(user_id);

-- Extend Users table for Clerk
ALTER TABLE users ADD COLUMN IF NOT EXISTS clerk_user_id VARCHAR(255) UNIQUE;
ALTER TABLE users ADD COLUMN IF NOT EXISTS name VARCHAR(255);
ALTER TABLE users ADD COLUMN IF NOT EXISTS avatar_url VARCHAR(500);
ALTER TABLE users ADD COLUMN IF NOT EXISTS last_sign_in TIMESTAMP;
ALTER TABLE users ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE;

CREATE INDEX IF NOT EXISTS idx_users_clerk ON users(clerk_user_id);

-- API Keys (for widget integration)
CREATE TABLE IF NOT EXISTS api_keys (
    id SERIAL PRIMARY KEY,
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    key_hash VARCHAR(255) NOT NULL,
    key_prefix VARCHAR(12) NOT NULL,
    scopes JSONB DEFAULT '["read", "write"]'::jsonb,
    last_used_at TIMESTAMP,
    expires_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_api_keys_workspace ON api_keys(workspace_id);
CREATE INDEX idx_api_keys_prefix ON api_keys(key_prefix);
CREATE INDEX idx_api_keys_hash ON api_keys(key_hash);

-- Usage Tracking (tokens, tool calls)
CREATE TABLE IF NOT EXISTS usage_logs (
    id SERIAL PRIMARY KEY,
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(id),
    metric_type VARCHAR(50) NOT NULL CHECK (metric_type IN (
        'llm_tokens_input', 'llm_tokens_output', 'tool_call', 
        'agent_run', 'workflow_run', 'document_upload'
    )),
    quantity INTEGER NOT NULL DEFAULT 1,
    model VARCHAR(100),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_usage_workspace ON usage_logs(workspace_id);
CREATE INDEX idx_usage_type ON usage_logs(metric_type);
CREATE INDEX idx_usage_created ON usage_logs(created_at DESC);

-- Usage Summary (daily rollups for dashboard)
CREATE TABLE IF NOT EXISTS usage_summary (
    id SERIAL PRIMARY KEY,
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    metric_type VARCHAR(50) NOT NULL,
    total_quantity INTEGER NOT NULL DEFAULT 0,
    UNIQUE(workspace_id, date, metric_type)
);

CREATE INDEX idx_usage_summary_workspace ON usage_summary(workspace_id, date);

-- Add workspace_id to existing tables
ALTER TABLE agents ADD COLUMN IF NOT EXISTS workspace_id UUID REFERENCES workspaces(id);
ALTER TABLE workflows ADD COLUMN IF NOT EXISTS workspace_id UUID REFERENCES workspaces(id);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS workspace_id UUID REFERENCES workspaces(id);
ALTER TABLE chats ADD COLUMN IF NOT EXISTS workspace_id UUID REFERENCES workspaces(id);
ALTER TABLE credentials ADD COLUMN IF NOT EXISTS workspace_id UUID REFERENCES workspaces(id);
ALTER TABLE mcp_tools ADD COLUMN IF NOT EXISTS workspace_id UUID REFERENCES workspaces(id);
ALTER TABLE skills ADD COLUMN IF NOT EXISTS workspace_id UUID REFERENCES workspaces(id);

CREATE INDEX IF NOT EXISTS idx_agents_workspace ON agents(workspace_id);
CREATE INDEX IF NOT EXISTS idx_workflows_workspace ON workflows(workspace_id);
CREATE INDEX IF NOT EXISTS idx_documents_workspace ON documents(workspace_id);
CREATE INDEX IF NOT EXISTS idx_chats_workspace ON chats(workspace_id);
CREATE INDEX IF NOT EXISTS idx_credentials_workspace ON credentials(workspace_id);
```

### Migration File

Create: `orchestrator/core/database/migrations/037_saas_foundation.sql`

```sql
-- Migration 037: SaaS Foundation (Workspaces)
-- Simplified multi-workspace support with Clerk auth

BEGIN;

-- [Include all CREATE TABLE and ALTER TABLE from above]

-- Insert default system workspace (for migration of existing data)
INSERT INTO workspaces (id, name, slug, owner_id, plan, is_personal, is_active)
SELECT 
    '00000000-0000-0000-0000-000000000000',
    'System Default',
    'system',
    1,  -- Assumes user ID 1 exists
    'enterprise',
    FALSE,
    TRUE
WHERE EXISTS (SELECT 1 FROM users WHERE id = 1)
ON CONFLICT DO NOTHING;

-- Migrate existing data to system workspace
UPDATE agents SET workspace_id = '00000000-0000-0000-0000-000000000000' WHERE workspace_id IS NULL;
UPDATE workflows SET workspace_id = '00000000-0000-0000-0000-000000000000' WHERE workspace_id IS NULL;
UPDATE documents SET workspace_id = '00000000-0000-0000-0000-000000000000' WHERE workspace_id IS NULL;
UPDATE chats SET workspace_id = '00000000-0000-0000-0000-000000000000' WHERE workspace_id IS NULL;
UPDATE credentials SET workspace_id = '00000000-0000-0000-0000-000000000000' WHERE workspace_id IS NULL;

COMMIT;
```

---

## Backend Implementation

### Requirements.txt Additions

```txt
# PRD-37: SaaS Foundation

# Clerk JWT Verification
PyJWT>=2.8.0
cryptography>=41.0.0

# API Key Generation (stdlib, no install needed)
# secrets, hashlib

# Rate Limiting
slowapi>=0.1.9
```

### Core Modules

```
orchestrator/
├── core/
│   ├── auth/
│   │   ├── __init__.py
│   │   ├── clerk.py             # Clerk JWT verification
│   │   ├── dependencies.py      # FastAPI dependencies
│   │   └── api_key.py           # API key auth
│   ├── workspaces/
│   │   ├── __init__.py
│   │   ├── service.py           # Workspace CRUD
│   │   ├── models.py            # SQLAlchemy models
│   │   └── limits.py            # Plan limit checks
│   └── usage/
│       ├── __init__.py
│       └── service.py           # Usage tracking
├── api/
│   ├── auth.py                  # Auth endpoints
│   ├── workspaces.py            # Workspace endpoints
│   └── api_keys.py              # API key endpoints
```

### Clerk JWT Verification

```python
# core/auth/clerk.py

from typing import Optional, Dict, Any
import jwt
from jwt import PyJWKClient
import os

class ClerkAuth:
    """Clerk JWT verification."""
    
    def __init__(self):
        self.jwks_url = os.getenv("CLERK_JWKS_URL")
        # Extract issuer/audience from JWKS URL (e.g., https://app.clerk.accounts.dev)
        # The audience is typically the issuer URL for Clerk tokens
        self.audience = os.getenv("CLERK_AUDIENCE")
        if not self.audience and self.jwks_url:
            # Derive from JWKS URL: https://app.clerk.accounts.dev/.well-known/jwks.json -> https://app.clerk.accounts.dev
            self.audience = self.jwks_url.replace("/.well-known/jwks.json", "")
        self._jwks_client = None
    
    @property
    def jwks_client(self) -> PyJWKClient:
        if self._jwks_client is None:
            self._jwks_client = PyJWKClient(self.jwks_url)
        return self._jwks_client
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify Clerk JWT and return claims."""
        try:
            signing_key = self.jwks_client.get_signing_key_from_jwt(token)
            return jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                audience=self.audience,
                options={"verify_aud": True, "verify_exp": True}
            )
        except Exception:
            return None
```

### Request Context

```python
# core/auth/dependencies.py

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer
from dataclasses import dataclass
from uuid import UUID

@dataclass
class RequestContext:
    user_id: int
    clerk_user_id: str
    email: str
    workspace_id: UUID
    workspace_name: str
    role: str  # owner | member
    plan: str  # starter | business | enterprise

async def get_context(
    request: Request,
    credentials = Depends(HTTPBearer())
) -> RequestContext:
    """Get authenticated request context with workspace."""
    # Verify JWT
    claims = clerk_auth.verify_token(credentials.credentials)
    if not claims:
        raise HTTPException(401, "Invalid token")
    
    # Get/create user
    user = get_or_create_user(claims)
    
    # Get workspace (from header or default)
    workspace_id = request.headers.get("X-Workspace-ID")
    workspace = get_user_workspace(user.id, workspace_id)
    
    if not workspace:
        raise HTTPException(404, "Workspace not found")
    
    membership = get_membership(workspace.id, user.id)
    
    return RequestContext(
        user_id=user.id,
        clerk_user_id=user.clerk_user_id,
        email=user.email,
        workspace_id=workspace.id,
        workspace_name=workspace.name,
        role=membership.role,
        plan=workspace.plan,
    )
```

### Workspace Service

```python
# core/workspaces/service.py

class WorkspaceService:
    """Workspace management."""
    
    def create_personal_workspace(self, user_id: int, email: str) -> Workspace:
        """Create personal workspace on first login."""
        slug = email.split("@")[0].lower()
        return Workspace(
            name=f"{slug}'s Workspace",
            slug=self._unique_slug(slug),
            owner_id=user_id,
            plan="starter",
            is_personal=True,
        )
    
    def invite_member(self, workspace_id: UUID, email: str, inviter_id: int):
        """Invite a team member."""
        # Check plan limits
        if not self._can_add_member(workspace_id):
            raise LimitExceeded("Member limit reached. Upgrade plan.")
        # Send invite via Clerk
        pass
    
    def check_limit(self, workspace_id: UUID, resource: str) -> bool:
        """Check if workspace has capacity for resource."""
        workspace = self.get(workspace_id)
        limits = workspace.plan_limits
        current = self._count_resource(workspace_id, resource)
        max_allowed = limits.get(f"max_{resource}", 0)
        return max_allowed == -1 or current < max_allowed
```

### Usage Tracking

```python
# core/usage/service.py

class UsageService:
    """Track usage for analytics and limits."""
    
    def log(
        self,
        workspace_id: UUID,
        metric_type: str,
        quantity: int = 1,
        user_id: int = None,
        model: str = None,
        metadata: dict = None
    ):
        """Log a usage event."""
        self.db.add(UsageLog(
            workspace_id=workspace_id,
            user_id=user_id,
            metric_type=metric_type,
            quantity=quantity,
            model=model,
            metadata=metadata or {},
        ))
        self.db.commit()
    
    def get_summary(self, workspace_id: UUID, days: int = 30) -> dict:
        """Get usage summary for dashboard."""
        # Aggregate from usage_logs or usage_summary
        pass
```

---

## Frontend Implementation

### Package.json Additions

```json
{
  "dependencies": {
    "@clerk/nextjs": "^4.29.0"
  }
}
```

### Clerk Setup

```tsx
// app/layout.tsx
import { ClerkProvider } from '@clerk/nextjs'

export default function RootLayout({ children }) {
  return (
    <ClerkProvider>
      <html><body>{children}</body></html>
    </ClerkProvider>
  )
}
```

```typescript
// middleware.ts
import { authMiddleware } from "@clerk/nextjs";

export default authMiddleware({
  publicRoutes: ["/", "/sign-in(.*)", "/sign-up(.*)", "/api/webhooks(.*)"],
});

export const config = {
  matcher: ["/((?!.+\\.[\\w]+$|_next).*)", "/", "/(api|trpc)(.*)"],
};
```

### API Client with Auth

```typescript
// lib/api.ts
import { useAuth } from "@clerk/nextjs";

export function useApi() {
  const { getToken } = useAuth();

  return async (url: string, options: RequestInit = {}) => {
    const token = await getToken();
    return fetch(`${process.env.NEXT_PUBLIC_API_URL}${url}`, {
      ...options,
      headers: {
        ...options.headers,
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
    });
  };
}
```

### Workspace Context

```tsx
// contexts/workspace.tsx
import { createContext, useContext } from 'react';

interface WorkspaceContext {
  id: string;
  name: string;
  plan: string;
  role: 'owner' | 'member';
}

const WorkspaceContext = createContext<WorkspaceContext | null>(null);

export function useWorkspace() {
  const ctx = useContext(WorkspaceContext);
  if (!ctx) throw new Error("useWorkspace must be used within WorkspaceProvider");
  return ctx;
}
```

---

## Clerk Billing Setup

### Clerk Dashboard Configuration

When Clerk Billing is available:

1. **Create Plans in Clerk Dashboard:**
   - Starter: $9/mo (or free tier)
   - Business: $29/mo
   - Enterprise: $99/mo

2. **Configure Features per Plan:**
   ```
   starter:   max_agents=3, max_workflows=10, max_members=1
   business:  max_agents=10, max_workflows=50, max_members=10
   enterprise: unlimited
   ```

3. **Enable Billing Portal** for self-service upgrades

### Sync Plan to Backend

```python
# Clerk webhook handler (when billing launches)
@router.post("/api/webhooks/clerk")
async def clerk_webhook(request: Request):
    event = await verify_clerk_webhook(request)
    
    if event["type"] == "organization.updated":
        org = event["data"]
        # Sync plan from Clerk to our DB
        workspace = get_by_clerk_org(org["id"])
        workspace.plan = org["subscription"]["plan"]
        workspace.plan_limits = PLAN_LIMITS[workspace.plan]
        db.commit()
```

### Fallback: Minimal Stripe (if Clerk Billing delayed)

If needed before Clerk Billing GA, implement minimal Stripe:

```python
# Simple checkout only, no complex webhooks
@router.post("/api/billing/checkout")
async def create_checkout(plan: str, ctx: RequestContext = Depends(get_context)):
    session = stripe.checkout.Session.create(
        customer_email=ctx.email,
        line_items=[{"price": PRICE_IDS[plan], "quantity": 1}],
        mode="subscription",
        success_url=f"{APP_URL}/settings/billing?success=true",
        cancel_url=f"{APP_URL}/settings/billing",
        metadata={"workspace_id": str(ctx.workspace_id)}
    )
    return {"url": session.url}
```

---

## Implementation Roadmap

### Phase 1: Clerk Auth (Days 1-2)

- [ ] Install `@clerk/nextjs` and `PyJWT`
- [ ] Create Clerk application in dashboard
- [ ] Add ClerkProvider to Next.js layout
- [ ] Create auth middleware
- [ ] Implement `ClerkAuth` class for JWT verification
- [ ] Create `get_context` dependency
- [ ] Test login/signup flow

### Phase 2: Workspaces (Days 3-4)

- [ ] Add database tables to `init_complete_schema.sql`
- [ ] Create migration `037_saas_foundation.sql`
- [ ] Run migration
- [ ] Create SQLAlchemy models
- [ ] Implement `WorkspaceService`
- [ ] Create workspace on first login
- [ ] Add `workspace_id` filtering to existing queries
- [ ] Create workspace settings page

### Phase 3: Team Members (Days 5-6)

- [ ] Implement member invite flow
- [ ] Create member management UI
- [ ] Test owner/member permissions
- [ ] Add plan limit checks

### Phase 4: API Keys & Usage (Days 7-8)

- [ ] Create API key generation endpoint
- [ ] Implement API key authentication
- [ ] Create usage tracking service
- [ ] Add usage logging to LLM calls
- [ ] Add usage logging to tool calls
- [ ] Create usage dashboard component

### Phase 5: Polish & Deploy (Day 9)

- [ ] Update environment variables
- [ ] Deploy to Railway staging
- [ ] End-to-end testing
- [ ] Documentation

---

## Environment Variables

```bash
# Clerk
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_xxx
CLERK_SECRET_KEY=sk_test_xxx
CLERK_JWKS_URL=https://your-app.clerk.accounts.dev/.well-known/jwks.json

# App URLs
NEXT_PUBLIC_APP_URL=https://your-app.railway.app
NEXT_PUBLIC_API_URL=https://api.your-app.railway.app
```

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Auth Success Rate | > 99.9% |
| Workspace Creation | < 500ms |
| API Key Validation | < 20ms |
| Data Isolation | 100% |

---

## Dependencies for PRD-36

After PRD-37, Composio can use:

```python
# Composio entity ID from workspace
entity_id = f"workspace_{ctx.workspace_id}"
```

---

**Document Version History**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-20 | DeepAgent | Initial PRD (complex) |
| 2.0 | 2026-01-20 | DeepAgent | Simplified: workspaces, Clerk billing, 2 roles |
