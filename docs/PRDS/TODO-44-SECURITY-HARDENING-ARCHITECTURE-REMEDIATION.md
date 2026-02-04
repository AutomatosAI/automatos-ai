# PRD-43: Platform Security Hardening & Architecture Remediation

**Version:** 1.0
**Status:** Planning Phase
**Date:** February 3, 2026
**Author:** Claude Opus 4.5 (Architect Review + API Security Audit)
**Prerequisites:** None
**Blocks:** Go-live readiness

---

## Introduction

A comprehensive architect review and API security audit of the Automatos AI Platform identified **3 critical, 7 high, 7 medium, and 5 low severity** security vulnerabilities alongside **12 architectural issues**. This PRD addresses all findings through focused, individually-implementable user stories organized into six phases.

The platform currently **fails 8 of 10 OWASP API Security Top 10** categories. Auth is disabled by default, 14+ routers have zero authentication, SQL injection vectors exist in raw queries, and several endpoints crash at runtime due to undefined references.

This is pre-launch hardening. Secret rotation is deferred to go-live preparation. The focus is code-level fixes, architectural improvements, and establishing secure defaults.

---

## Goals

- Achieve OWASP API Security Top 10 compliance across all endpoints
- Enforce authentication on 100% of mutation and data-access endpoints
- Eliminate all SQL injection vectors
- Fix all runtime crashes (NameError, undefined references)
- Implement rate limiting, security headers, and proper CORS
- Decompose god files and introduce a service layer
- Establish consistent API response patterns
- Remove mock/simulated data from production code paths

---

## Non-Goals

- Secret rotation (deferred to go-live preparation)
- Full test suite creation (separate PRD)
- Frontend redesign or UX changes
- New feature development
- Performance optimization beyond fixing N+1 queries
- CI/CD pipeline changes

---

## User Stories

### Phase 1: Critical Security Fixes

#### US-001: Flip REQUIRE_AUTH Default to Secure-by-Default

**Description:** As a platform operator, I want authentication required by default so that deployments are secure without manual configuration.

**Acceptance Criteria:**
- [ ] In `orchestrator/core/auth/hybrid.py`, change `REQUIRE_AUTH` default from empty string (falsy) to `True`
- [ ] Anonymous fallback only activates when `REQUIRE_AUTH` is explicitly set to `false`/`0`/`off`
- [ ] The hardcoded fallback UUID `00000000-0000-0000-0000-000000000001` is removed or only used in explicit dev mode
- [ ] Add `REQUIRE_AUTH=false` to `orchestrator/.env` for local development
- [ ] Typecheck/lint passes

---

#### US-002: Add Auth to System Settings Router

**Description:** As a security engineer, I want the system-settings endpoints protected so that unauthorized users cannot modify platform configuration.

**Acceptance Criteria:**
- [ ] All endpoints in the system-settings router use `get_request_context_hybrid` dependency
- [ ] Requests without valid auth receive 401
- [ ] Workspace scoping applied where applicable
- [ ] Typecheck/lint passes

---

#### US-003: Add Auth to Permissions Router

**Description:** As a security engineer, I want permission management endpoints protected so that unauthorized users cannot assign/revoke agent tool permissions.

**Acceptance Criteria:**
- [ ] All endpoints in the permissions router use `get_request_context_hybrid` dependency
- [ ] Requests without valid auth receive 401
- [ ] Typecheck/lint passes

---

#### US-004: Add Auth to Memory Router

**Description:** As a security engineer, I want memory storage/retrieval endpoints protected.

**Acceptance Criteria:**
- [ ] All endpoints in `/api/v1/memory` use `get_request_context_hybrid` dependency
- [ ] Requests without valid auth receive 401
- [ ] Typecheck/lint passes

---

#### US-005: Add Auth to Analytics, Benchmarking, and Evaluation Routers

**Description:** As a security engineer, I want analytics, benchmarking, and evaluation endpoints protected.

**Acceptance Criteria:**
- [ ] All endpoints in `/analytics`, `/api/v1/benchmarking`, `/api/evaluation`, `/api/analytics` use `get_request_context_hybrid`
- [ ] Requests without valid auth receive 401
- [ ] Typecheck/lint passes

---

#### US-006: Add Auth to Remaining Unprotected Routers

**Description:** As a security engineer, I want all remaining unprotected routers secured: templates, playbooks, chatbot, execution-history, field-theory, database analytics.

**Acceptance Criteria:**
- [ ] All endpoints in `/api/templates`, `/api/playbooks`, `/api/chatbot`, `/api/execution-history`, `/api/field-theory`, `/api/database/analytics` use `get_request_context_hybrid`
- [ ] Requests without valid auth receive 401
- [ ] Typecheck/lint passes

---

#### US-007: Lock Down Credential Resolve Endpoint

**Description:** As a security engineer, I want the credential resolve endpoint restricted to admin-only access so that decrypted secrets cannot be accessed by regular users.

**Acceptance Criteria:**
- [ ] `POST /api/credentials/resolve` requires authenticated admin role
- [ ] `GET /api/credentials/{credential_id}` that returns decrypted data requires admin role
- [ ] Non-admin users receive 403 Forbidden
- [ ] Typecheck/lint passes

---

#### US-008: Fix SQL Injection in Documents Embedding Query

**Description:** As a security engineer, I want the embedding vector query in documents.py to use parameterized queries instead of f-string interpolation.

**Acceptance Criteria:**
- [ ] `orchestrator/api/documents.py` lines 736-756: replace f-string embedding interpolation with `text().bindparams()`
- [ ] Query results remain functionally identical
- [ ] Typecheck/lint passes

---

#### US-009: Fix SQL Injection in Database Knowledge Module

**Description:** As a security engineer, I want SQL injection vectors in database_knowledge.py fixed.

**Acceptance Criteria:**
- [ ] `orchestrator/api/database_knowledge.py` line 537: `SET LOCAL statement_timeout` uses parameterized value
- [ ] All other f-string SQL patterns in this file converted to parameterized queries
- [ ] Typecheck/lint passes

---

#### US-010: Fix SQL Injection in CodeGraph Service

**Description:** As a security engineer, I want SQL injection vectors in codegraph_service.py fixed.

**Acceptance Criteria:**
- [ ] `orchestrator/modules/codegraph/codegraph_service.py` lines 727, 733, 1010, 1204: all f-string SQL converted to parameterized queries
- [ ] Typecheck/lint passes

---

#### US-011: Fix SQL Injection in Knowledge Multimodal and NL2SQL

**Description:** As a security engineer, I want SQL injection vectors in knowledge_multimodal.py and introspection.py fixed.

**Acceptance Criteria:**
- [ ] `orchestrator/modules/rag/services/knowledge_multimodal.py` lines 576, 597: parameterized queries
- [ ] `orchestrator/modules/nl2sql/schema/introspection.py` lines 105, 122, 204, 216: table/column names properly escaped or parameterized
- [ ] `orchestrator/api/skills.py` line 659: parameterized query
- [ ] Typecheck/lint passes

---

#### US-012: Remove Browser-Exposed API Key

**Description:** As a security engineer, I want the API key removed from the browser bundle so that it cannot be extracted by end users.

**Acceptance Criteria:**
- [ ] Remove `NEXT_PUBLIC_API_KEY` from `frontend/.env.local`
- [ ] Remove hardcoded fallback key `test_api_key_for_backend_validation_2025` from `frontend/app/api/chat/route.ts`
- [ ] Frontend authenticates exclusively via Clerk JWT tokens for API requests
- [ ] Verify API calls still work with Clerk auth only
- [ ] Typecheck/lint passes

---

### Phase 2: Runtime Bug Fixes

#### US-013: Fix NameError in update_workflow Endpoint

**Description:** As a developer, I want the update_workflow endpoint to work without crashing so that workflows can be updated.

**Acceptance Criteria:**
- [ ] `orchestrator/api/workflows.py` line 360: add `ctx: RequestContext = Depends(get_request_context_hybrid)` parameter
- [ ] Endpoint uses `ctx.workspace_id` for workspace-scoped query
- [ ] Typecheck/lint passes

---

#### US-014: Fix NameError in duplicate_workflow Endpoint

**Description:** As a developer, I want the duplicate_workflow endpoint to work without crashing.

**Acceptance Criteria:**
- [ ] `orchestrator/api/workflows.py` line 617: add `ctx: RequestContext = Depends(get_request_context_hybrid)` parameter
- [ ] Endpoint uses `ctx.workspace_id` for workspace-scoped query
- [ ] Typecheck/lint passes

---

#### US-015: Remove Dead manager.broadcast() Calls

**Description:** As a developer, I want the dead WebSocket manager references removed so that orchestrator endpoints don't crash on success.

**Acceptance Criteria:**
- [ ] `orchestrator/api/orchestrator.py` lines 93, 213, 290: remove `await manager.broadcast(...)` calls
- [ ] Replace with appropriate logging or SSE event if needed
- [ ] Endpoints complete successfully without NameError
- [ ] Typecheck/lint passes

---

#### US-016: Fix Undefined websocket_connections in Health Check

**Description:** As a developer, I want the health check endpoint to work without crashing.

**Acceptance Criteria:**
- [ ] `orchestrator/api/main.py` lines 488, 502: remove or replace `websocket_connections` references
- [ ] Health check returns valid response
- [ ] Typecheck/lint passes

---

### Phase 3: High Severity Security Fixes

#### US-017: Implement Rate Limiting with slowapi

**Description:** As a platform operator, I want rate limiting on API endpoints to prevent brute-force attacks and cost abuse on LLM endpoints.

**Acceptance Criteria:**
- [ ] Configure `slowapi.Limiter` in `orchestrator/main.py` with sensible defaults
- [ ] Add rate limit exception handler
- [ ] Apply rate limits to auth-related endpoints (stricter: 10/min)
- [ ] Apply rate limits to LLM/chat endpoints (moderate: 30/min)
- [ ] Apply default rate limit to all other endpoints (generous: 100/min)
- [ ] Fix health check to report actual rate limiting status
- [ ] Typecheck/lint passes

---

#### US-018: Restrict CORS Configuration

**Description:** As a security engineer, I want CORS restricted to only the methods and headers actually needed.

**Acceptance Criteria:**
- [ ] `orchestrator/main.py`: change `allow_methods` from `["*"]` to `["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]`
- [ ] Change `allow_headers` from `["*"]` to explicit list: `["Content-Type", "Authorization", "X-API-Key", "X-Workspace-ID", "X-Request-ID"]`
- [ ] Verify frontend requests still work with restricted headers
- [ ] Typecheck/lint passes

---

#### US-019: Add Security Headers Middleware

**Description:** As a security engineer, I want security headers on all HTTP responses.

**Acceptance Criteria:**
- [ ] Add middleware to `orchestrator/main.py` that sets on every response:
  - `X-Content-Type-Options: nosniff`
  - `X-Frame-Options: DENY`
  - `Referrer-Policy: strict-origin-when-cross-origin`
  - `Permissions-Policy: camera=(), microphone=(), geolocation=()`
- [ ] HSTS header added when `ENVIRONMENT=production`
- [ ] Typecheck/lint passes

---

#### US-020: Make GitHub Webhook Signature Verification Mandatory

**Description:** As a security engineer, I want GitHub webhook signature verification to be required, not optional.

**Acceptance Criteria:**
- [ ] `orchestrator/api/github_webhooks.py`: if `GITHUB_WEBHOOK_SECRET` is not set, return 500 "Webhook secret not configured"
- [ ] If signature header is missing, return 401 "Missing signature"
- [ ] Signature verification always runs when both are present
- [ ] Typecheck/lint passes

---

#### US-021: Add Workspace Membership Verification

**Description:** As a security engineer, I want the platform to verify that an authenticated user belongs to the workspace they're requesting, preventing cross-workspace data access via header manipulation.

**Acceptance Criteria:**
- [ ] In `orchestrator/core/auth/hybrid.py`: after resolving workspace ID from headers, verify the authenticated user is a member of that workspace
- [ ] Return 403 if user is not a member of the requested workspace
- [ ] Skip check for API key auth (service-to-service)
- [ ] Typecheck/lint passes

---

### Phase 4: Medium & Low Severity Fixes

#### US-022: Sanitize Error Responses

**Description:** As a security engineer, I want error responses to return generic messages instead of leaking internal details.

**Acceptance Criteria:**
- [ ] Create a utility function `safe_error_response(status_code, default_message, exception, logger)` that logs details server-side and returns generic message to client
- [ ] Replace `raise HTTPException(status_code=500, detail=str(e))` patterns across: credentials.py, permissions endpoints, system settings, and other affected files
- [ ] Internal details (table names, file paths, stack traces) never appear in HTTP response bodies
- [ ] Typecheck/lint passes

---

#### US-023: Disable Swagger Docs in Production

**Description:** As a security engineer, I want API documentation hidden in production to reduce attack surface.

**Acceptance Criteria:**
- [ ] `orchestrator/main.py`: set `docs_url`, `redoc_url`, `openapi_url` to `None` when `ENVIRONMENT` is `production`
- [ ] Docs remain available in development
- [ ] Typecheck/lint passes

---

#### US-024: Add File Upload MIME Type Validation

**Description:** As a security engineer, I want uploaded files validated by actual content type, not just file extension.

**Acceptance Criteria:**
- [ ] `orchestrator/api/documents.py`: use `python-magic` to validate MIME type matches expected content
- [ ] Reject files where extension doesn't match detected MIME type
- [ ] Generate random filenames instead of using `{hash}_{original_filename}` pattern
- [ ] Typecheck/lint passes

---

#### US-025: Remove Auto-Admin Domain Logic

**Description:** As a security engineer, I want admin role assignment managed explicitly, not auto-granted by email domain.

**Acceptance Criteria:**
- [ ] `orchestrator/core/auth/clerk.py` lines 197-200: remove the `@automatos.app` auto-admin check
- [ ] Admin roles managed through Clerk public metadata or database role table
- [ ] Existing admin users retain their role through the new mechanism
- [ ] Typecheck/lint passes

---

#### US-026: Extract user_id from Auth Context

**Description:** As a security engineer, I want user_id sourced from the authenticated context, not from client-provided query parameters.

**Acceptance Criteria:**
- [ ] `orchestrator/api/credentials.py` line 159: remove `user_id` query parameter, use `ctx.user.id` instead
- [ ] Same fix in permissions endpoints (lines 353, 453, 575)
- [ ] Audit all other endpoints accepting `user_id` as input
- [ ] Typecheck/lint passes

---

#### US-027: Remove Credential Data Logging

**Description:** As a security engineer, I want credential values and encryption key fragments removed from log output.

**Acceptance Criteria:**
- [ ] `orchestrator/api/credentials.py` line 172: remove logging of `credential.credential_data` value
- [ ] `orchestrator/core/credentials/encryption.py` line 74: remove logging of encryption key characters
- [ ] Replace with safe log messages that indicate the operation without exposing values
- [ ] Typecheck/lint passes

---

#### US-028: Add Auth to /exports Static File Mount

**Description:** As a security engineer, I want the /exports directory protected from unauthenticated access.

**Acceptance Criteria:**
- [ ] `orchestrator/main.py` line 464: replace static file mount with an authenticated endpoint that serves files
- [ ] Or add middleware that checks auth before serving from /exports
- [ ] Typecheck/lint passes

---

#### US-029: Enable TypeScript Build Error Checking

**Description:** As a developer, I want TypeScript build errors caught during build so that type-safety issues don't reach production.

**Acceptance Criteria:**
- [ ] `frontend/next.config.js`: remove or set `ignoreBuildErrors: false`
- [ ] Fix any TypeScript errors that surface
- [ ] Build completes successfully
- [ ] Typecheck/lint passes

---

#### US-030: Update Outdated Dependencies

**Description:** As a developer, I want dependencies updated to address known CVEs and get security patches.

**Acceptance Criteria:**
- [ ] Update `anthropic` from `0.8.1` to latest stable
- [ ] Update `fastapi` from `0.104.1` to latest stable
- [ ] Replace `PyPDF2==3.0.1` with `pypdf` (PyPDF2 is deprecated with known CVEs)
- [ ] Update `cryptography` minimum to `>=43.0.0`
- [ ] Run `pip audit` and address any remaining vulnerabilities
- [ ] All existing functionality still works after updates
- [ ] Typecheck/lint passes

---

#### US-031: Remove Hardcoded Docker Compose Defaults

**Description:** As a developer, I want docker-compose to require explicit credentials rather than falling back to hardcoded defaults.

**Acceptance Criteria:**
- [ ] `docker-compose.yml`: remove default values from `POSTGRES_PASSWORD`, `REDIS_PASSWORD`, `API_KEY`
- [ ] Add a `.env.example` with placeholder values and instructions
- [ ] Docker compose fails with clear error if required vars are not set
- [ ] Typecheck/lint passes

---

### Phase 5: Architecture Remediation — Backend

#### US-032: Fix Hardcoded tenant_id in Database Knowledge

**Description:** As a developer, I want tenant_id sourced from the auth context instead of hardcoded to 1.

**Acceptance Criteria:**
- [ ] `orchestrator/api/database_knowledge.py` lines 85, 135, 173, 462: replace `tenant_id=1` with value from `RequestContext`
- [ ] Add `get_request_context_hybrid` dependency to affected endpoints
- [ ] Typecheck/lint passes

---

#### US-033: Decompose workflows.py — Extract CRUD Operations

**Description:** As a developer, I want workflow CRUD operations in their own module so that workflows.py is not a 2,777-line god file.

**Acceptance Criteria:**
- [ ] Create `orchestrator/api/workflow_crud.py` containing: create, read, update, delete, list, duplicate workflow endpoints
- [ ] Move relevant imports and helper functions
- [ ] Register the new router in main.py
- [ ] All moved endpoints still function correctly
- [ ] Typecheck/lint passes

---

#### US-034: Decompose workflows.py — Extract Execution Logic

**Description:** As a developer, I want workflow execution logic in its own module.

**Acceptance Criteria:**
- [ ] Create `orchestrator/api/workflow_execution.py` containing: create_execution, cancel_execution, execution status, live progress, SSE streaming endpoints
- [ ] Move relevant imports and helper functions
- [ ] Register the new router in main.py
- [ ] All moved endpoints still function correctly
- [ ] Typecheck/lint passes

---

#### US-035: Decompose workflows.py — Extract Analytics and Templates

**Description:** As a developer, I want workflow analytics and template endpoints in their own module.

**Acceptance Criteria:**
- [ ] Create `orchestrator/api/workflow_analytics.py` containing: dashboard stats, execution history, template recommendations
- [ ] Move relevant imports and helper functions
- [ ] Register the new router in main.py
- [ ] Original workflows.py reduced to under 500 lines (or removed entirely)
- [ ] Typecheck/lint passes

---

#### US-036: Separate ORM Models from Pydantic Schemas

**Description:** As a developer, I want ORM models, Pydantic schemas, and enums separated into distinct files so that core.py is not a 1,291-line mixed-concern file.

**Acceptance Criteria:**
- [ ] Create `orchestrator/core/schemas/` directory with domain-specific schema files
- [ ] Create `orchestrator/core/enums.py` for all enum classes
- [ ] Move Pydantic schemas out of `core/models/core.py` into `core/schemas/`
- [ ] Move enums out of `core/models/core.py` into `core/enums.py`
- [ ] Update all imports across the codebase
- [ ] `core/models/core.py` contains only SQLAlchemy ORM models
- [ ] Typecheck/lint passes

---

#### US-037: Replace Star Imports in core/models/__init__.py

**Description:** As a developer, I want explicit imports instead of 11 wildcard imports so that symbol origins are traceable.

**Acceptance Criteria:**
- [ ] `orchestrator/core/models/__init__.py`: replace all `from .module import *` with explicit named imports
- [ ] Only export symbols that are actually used by consumers
- [ ] Add `__all__` to each sub-module to control exports
- [ ] Typecheck/lint passes

---

#### US-038: Introduce WorkflowService Layer

**Description:** As a developer, I want a service layer between API handlers and ORM queries so that business logic is reusable and testable.

**Acceptance Criteria:**
- [ ] Create `orchestrator/services/workflow_service.py` with methods for core workflow operations
- [ ] API handlers delegate to service methods instead of containing inline ORM queries
- [ ] Service methods accept typed parameters (not raw Request objects)
- [ ] At minimum, extract the `get_active_workflows` logic (currently 148 lines inline)
- [ ] Typecheck/lint passes

---

#### US-039: Fix N+1 Query in Active Workflows Endpoint

**Description:** As a developer, I want the active workflows endpoint to use efficient queries instead of N+1 patterns.

**Acceptance Criteria:**
- [ ] `orchestrator/api/workflows.py` (or new workflow_crud.py): replace per-workflow count queries with a single aggregation query using subquery or window functions
- [ ] Endpoint returns identical data
- [ ] Query count reduced from 3N+1 to constant (2-3 queries max)
- [ ] Typecheck/lint passes

---

#### US-040: Standardize JSON Column Types to JSONB

**Description:** As a developer, I want all JSON columns using JSONB for consistent indexing and query support.

**Acceptance Criteria:**
- [ ] Audit all SQLAlchemy models for `Column(JSON)` usage
- [ ] Create Alembic migration to convert `JSON` columns to `JSONB`
- [ ] Verify no queries break after conversion
- [ ] Typecheck/lint passes

---

#### US-041: Remove Mock/Simulated Data from Production Handlers

**Description:** As a developer, I want all simulated/mock data removed from API handlers so that endpoints return real data or explicit 501 Not Implemented.

**Acceptance Criteria:**
- [ ] `orchestrator/api/workflows.py`: remove `# Simulate progress` patterns, replace with real progress tracking or 501
- [ ] `orchestrator/api/orchestrator.py`: remove `# Simulate phase execution` block, implement or return 501
- [ ] `orchestrator/api/agent_endpoints.py`: remove `# Simulate learning`
- [ ] `orchestrator/api/memory.py`: remove mock search results
- [ ] All affected endpoints either return real data or HTTP 501 with clear message
- [ ] Typecheck/lint passes

---

#### US-042: Resolve Circular Dependency Chain

**Description:** As a developer, I want the circular dependency between `core` and `modules` resolved so that lazy imports are no longer needed.

**Acceptance Criteria:**
- [ ] Identify all places where `core` imports from `modules` (reverse dependency)
- [ ] Introduce interfaces/protocols in `core` that `modules` implements
- [ ] Remove at least 50% of the 32 lazy import workarounds
- [ ] No new circular import errors introduced
- [ ] Typecheck/lint passes

---

### Phase 6: Architecture Remediation — Frontend

#### US-043: Unify Frontend API Client

**Description:** As a developer, I want a single API access pattern in the frontend instead of two competing systems.

**Acceptance Criteria:**
- [ ] Deprecate or remove the `apiClient` singleton class in `frontend/lib/api-client.ts`
- [ ] All API calls route through the `useAuthenticatedApi` hook pattern
- [ ] Clerk token injection happens in one place
- [ ] Remove `PAGE_MOCK_CONFIG` system (or move to a dev-only context)
- [ ] Typecheck/lint passes

---

#### US-044: Standardize API Response Shapes

**Description:** As a developer, I want all API endpoints to use a consistent response envelope so that frontend parsing is predictable.

**Acceptance Criteria:**
- [ ] Define a standard response envelope: `{ data: T, meta?: { total?: number, page?: number } }`
- [ ] Create Pydantic generic response model
- [ ] Migrate at minimum: workflows list, active workflows, agents list endpoints to new format
- [ ] Document the response convention
- [ ] Typecheck/lint passes

---

#### US-045: Remove MockProvider from Production Provider Tree

**Description:** As a developer, I want the MockProvider removed from the production component tree to reduce overhead.

**Acceptance Criteria:**
- [ ] `frontend/components/providers.tsx`: remove MockProvider from the provider chain
- [ ] Move mock functionality to dev-only conditional rendering if needed
- [ ] No runtime errors when MockProvider is absent
- [ ] Typecheck/lint passes

---

## Functional Requirements

- FR-1: All API endpoints MUST require authentication (Clerk JWT or API key) unless explicitly documented as public
- FR-2: Authentication MUST be enabled by default (`REQUIRE_AUTH=true`)
- FR-3: All database queries MUST use parameterized queries; no f-string interpolation in SQL
- FR-4: Error responses MUST NOT contain internal details (stack traces, table names, file paths)
- FR-5: All state-changing endpoints MUST be rate-limited
- FR-6: Security headers MUST be set on every HTTP response
- FR-7: Workspace access MUST be validated against user membership
- FR-8: Credential resolution endpoints MUST require admin role
- FR-9: GitHub webhook signature verification MUST be mandatory
- FR-10: File uploads MUST validate MIME type against actual content
- FR-11: API documentation MUST be disabled in production
- FR-12: No mock or simulated data in production response handlers
- FR-13: Frontend MUST NOT expose API keys in browser bundles
- FR-14: All JSON database columns MUST use JSONB type
- FR-15: No single API file should exceed 500 lines

---

## Technical Considerations

### Dependencies
- `slowapi>=0.1.9` (already in requirements.txt, needs wiring)
- `python-magic` (already in requirements.txt, needs usage in upload handler)
- `pypdf` (replacement for deprecated `PyPDF2`)

### Key Files
- `orchestrator/core/auth/hybrid.py` — central auth logic
- `orchestrator/core/auth/clerk.py` — Clerk JWT verification
- `orchestrator/api/main.py` — app setup, middleware, CORS
- `orchestrator/api/workflows.py` — largest API file (decomposition target)
- `orchestrator/core/models/core.py` — mixed ORM/schema file (separation target)
- `frontend/lib/api-client.ts` — largest frontend file (unification target)
- `frontend/components/providers.tsx` — provider tree

### Migration Strategy
- Phases 1-2 (Critical + Runtime) can be done independently per story
- Phase 3 (High Security) stories are independent
- Phase 4 (Medium/Low) stories are independent
- Phase 5 (Backend Architecture) has internal dependencies: US-033/034/035 should complete before US-038
- Phase 6 (Frontend Architecture) is independent of backend phases

### Risk Mitigation
- Each story is small enough to implement and revert independently
- Auth changes should be tested against Clerk dev instance
- Database migrations (US-040) should be tested on a staging copy first
- Dependency updates (US-030) should be done one package at a time

---

## Success Metrics

- 0 API endpoints accessible without authentication (excluding explicit public routes)
- 0 SQL injection vectors (f-string patterns in SQL queries)
- 0 runtime NameError crashes
- 100% of HTTP responses include security headers
- Rate limiting active on all endpoints
- No file over 500 lines in API layer after decomposition
- OWASP API Security Top 10: PASS on all 10 categories

---

## Open Questions

1. Should we implement a formal API versioning strategy (e.g., `/api/v2/`) alongside the response envelope standardization, or just evolve `/api/` in place?
2. For rate limiting, should limits be per-user or per-IP? Per-user is more precise but requires auth to be resolved first.
3. Should the `useAuthenticatedApi` hook replacement be a breaking change or a gradual migration with a compatibility wrapper?
4. For workspace membership verification (US-021), how should service-to-service API key requests handle workspace scoping?
