# PRD-34: Unified Integrations Adapter (Context Forge + REST/MCP)

**Version:** 1.0  
**Status:** 🟡 Design Phase  
**Priority:** HIGH - External Integrations  
**Author:** Automatos AI Platform Team  
**Last Updated:** 2026-01-14  
**Dependencies:** PRD-33 (MCP Gateway Integration), PRD-20 (MCP Integration), PRD-18 (Credential Management)

---

## Executive Summary

Automatos needs to expose many SaaS integrations without running hundreds of MCP servers. This PRD defines a **Unified Integrations Adapter** as a standalone service that provides a single MCP server endpoint for many tools, while still using **Context Forge** as the gateway (`mcp.automatos.app`). The MVP targets **10–20 tools across multiple categories**, prioritizing a mix of REST passthrough (where OpenAPI specs exist) and MCP for tools that already have hosted MCP endpoints. Unified.to is researched as a hosted accelerator but excluded from MVP due to pricing and vendor lock-in concerns.  
Sources: Unified MCP docs and integration catalog ([docs.unified.to](https://docs.unified.to/), [unified.to/integrations](https://unified.to/integrations)), open-source MCP server references ([github.com/modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers)).

---

## 1) Background & How We Got Here

Automatos originally imported a large MCP tool catalog (400+ entries) with `mcp://` and registry URLs (e.g., `mcp.so`). These are **catalog references**, not live servers. Context Forge was adopted to provide a single gateway and admin UI, but it **only discovers tools from live MCP servers** and **blocks manual MCP tool imports** by design. This created a mismatch: a large catalog without executable endpoints.

Key decisions to date:
- **Context Forge** runs at `mcp.automatos.app` as the single public gateway.
- **Automatos UI** remains the system of record for tool and credential CRUD.
- **Credentials** are stored encrypted in Automatos and must be synced/used for execution.
- **Manual MCP import** into Context Forge is not supported; discovery requires real MCP servers.

This PRD addresses the gap by introducing a **Unified Integrations Adapter** service so Automatos can expose many tools without deploying one MCP server per integration.

---

## 2) Current State Snapshot

### Context Forge (Gateway)
- Deployed and reachable at `mcp.automatos.app`
- Acts as the **single entry point** for MCP clients and tools
- Requires live MCP servers for tool discovery
- Includes admin UI for gateways, tools, prompts, resources

### Automatos (Core Platform)
- Stores tool catalog and credentials (encrypted)
- Routes tool execution through `MCPToolExecutor` / `UnifiedToolExecutor`
- Credential system is already used for tool resolution and testing

### Catalog Reality
- The imported tool catalog is a **registry**, not executable endpoints
- Most entries point to registry URLs, not hosted MCP servers

---

## 1) Problem Statement

### Current Pain
- The tool catalog contains **registry entries**, not live endpoints. MCP tools cannot be executed without real servers.
- Context Forge does **not** allow manual MCP tool imports; tools must be discovered from live MCP servers.
- Spinning up one MCP server per integration does not scale operationally.
- Unified.to and similar hosted options are expensive at scale.

### Impact
- External integrations cannot be executed reliably.
- Tool enablement is blocked by missing live endpoints.
- High operational cost and time-to-value for a large tool catalog.

---

## 2) Goals & Success Metrics

### Goals
- **G1 (Single MCP Entry Point):** Maintain `mcp.automatos.app` as the single gateway endpoint via Context Forge.
- **G2 (Unified Adapter):** Provide one Automatos-hosted MCP server that maps to many tools.
- **G3 (REST + MCP Hybrid):** Use REST passthrough when OpenAPI exists; MCP for hosted MCP endpoints.
- **G4 (MVP Coverage):** Enable 10–20 integrations across multiple categories.
- **G5 (Tenant-Ready):** Single-tenant dev now; record clear multi-tenant extension points.

### Success Metrics (MVP)
- **M1:** 10–20 tools executable end-to-end via Context Forge.
- **M2:** Average tool call latency < 2s added overhead.
- **M3:** Credential resolution success rate ≥ 99% (dev).
- **M4:** 0 plaintext credential leakage in logs or UI.

---

## 3) Research Findings

### Hosted MCP Providers
- **Unified.to MCP Server** provides a single hosted MCP endpoint with tools derived from Unified connections, but is paid and introduces vendor lock-in ([docs.unified.to](https://docs.unified.to/), [unified.to/integrations](https://unified.to/integrations)).

### Open Source MCP Servers
- The official MCP servers repository contains **reference servers**, not a unified adapter or gateway; it is useful for testing and building adapters but does not solve large-scale integration needs alone ([github.com/modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers)).

### API Registries for REST Passthrough
REST passthrough requires OpenAPI specs. Candidate registries:
- APIs.guru OpenAPI Directory ([apis.guru](https://apis.guru/about?utm_source=openai))
- Postman Public API Network (for published schemas and collections)
- RapidAPI Directory (public API listings and specs) ([rapidapi.com](https://rapidapi.com/blog/directory/?utm_source=openai))

---

## 4) Architecture Overview

**High-level flow:**
1. Automatos UI manages **tools + credentials**
2. Unified Adapter reads tool definitions + credentials from Automatos
3. Adapter exposes one MCP server endpoint
4. Context Forge registers a single **Gateway** to the Adapter
5. Context Forge discovers tools from the Adapter
6. LLM tool calls flow through Context Forge → Adapter → SaaS API

**Key principle:** Context Forge remains the **single entry point**; the Adapter is the **single MCP server** that scales to many tools.

---

## 5) System Context & Component Responsibilities

### Context Forge (Gateway)
- Public MCP endpoint (`mcp.automatos.app`)
- MCP tool discovery from registered gateways
- Admin UI for internal tool registry visibility
- JWT-based access control

### Unified Integrations Adapter (New Service)
- Single MCP server endpoint for many tools
- Translates MCP tool calls → REST or MCP upstreams
- Fetches tool catalog + credentials from Automatos
- Enforces tool scoping (enabled/disabled, category)
- Standardizes responses and errors

### Automatos Core
- Source of truth for tool catalog and credential management
- Credential encryption and resolution (existing capability)
- Tool enablement and metadata (existing capability)
- User/agent permissions

### External SaaS APIs
- REST/GraphQL endpoints, often with OpenAPI specs
- OAuth 2.0 or API key authentication
- Rate limits and usage policies

---

## 6) Data Flow (Request Lifecycle)

### 6.1 Tool Discovery
1. Adapter loads tools from Automatos.
2. Adapter publishes MCP tool list to Context Forge via Gateway discovery.
3. Context Forge surfaces tools in its admin UI and tool list.

### 6.2 Tool Execution
1. User chat triggers tool execution in Automatos.
2. Automatos calls Context Forge MCP endpoint.
3. Context Forge routes to Adapter Gateway.
4. Adapter resolves credentials from Automatos.
5. Adapter executes tool against REST/MCP upstream.
6. Adapter returns normalized response.
7. Context Forge returns result to Automatos UI.

---

## 7) REST vs MCP Decision Matrix

| Criteria | Prefer REST Passthrough | Prefer MCP |
|---|---|---|
| OpenAPI spec exists | ✅ | ❌ |
| Hosted MCP endpoint exists | ❌ | ✅ |
| OAuth flow required | ⚠️ (needs adapter support) | ✅ (if MCP supports OAuth) |
| Tool count per vendor | ✅ | ✅ |
| Vendor provides official MCP server | ❌ | ✅ |

Decision rule: default to REST passthrough for **public APIs with OpenAPI specs**. Use MCP only when a real MCP endpoint exists or the vendor publishes a stable MCP server.

---

## 8) MVP Scope (10–20 Tools, Cross-Category)

### Categories (mix of 2–4 each)
- **Messaging:** Slack, Gmail, Microsoft Teams
- **Productivity:** Google Drive, Notion, Jira
- **CRM:** HubSpot, Salesforce, Pipedrive
- **Dev/Repo:** GitHub, GitLab
- **Calendar:** Google Calendar, Outlook

### Selection Criteria
- Public API documentation + OpenAPI available → **REST passthrough**
- Hosted MCP endpoint available → **MCP**
- Credential type already supported in Automatos

---

## 9) Tool Shortlist (Draft MVP Set)

**Messaging:** Slack, Gmail  
**Productivity:** Google Drive, Notion  
**CRM:** HubSpot, Salesforce  
**Dev/Repo:** GitHub, GitLab  
**Calendar:** Google Calendar, Outlook  
**Support/Ticketing:** Zendesk, Intercom  
**Storage:** Dropbox, Box

Final list to be confirmed after confirming API specs and auth requirements.

---

## 10) User Stories

### US-001: Standalone Unified Adapter Service
**Description:** As a platform engineer, I want the adapter to run as its own service so I can scale and secure it independently of Automatos core.

**Acceptance Criteria:**
- [ ] Adapter service runs in a separate container
- [ ] MCP endpoint exposed for Context Forge discovery
- [ ] Environment-based config for base URLs and auth
- [ ] Typecheck/lint passes

### US-002: Tool Discovery From Automatos DB
**Description:** As a developer, I want the adapter to discover tools from Automatos so I can manage tools in one place.

**Acceptance Criteria:**
- [ ] Adapter loads tool definitions from Automatos database or API
- [ ] Tools are filtered by enabled status and category
- [ ] Tools expose JSON schema for input parameters
- [ ] Typecheck/lint passes

### US-003: Credential Resolution via Automatos
**Description:** As an admin, I want credentials to be managed in Automatos but used by the adapter for tool calls.

**Acceptance Criteria:**
- [ ] Adapter retrieves credentials securely (encrypted) via Automatos services
- [ ] No credentials logged in plaintext
- [ ] Credential lookup supports environment (dev/staging/prod)
- [ ] Typecheck/lint passes

### US-004: REST Passthrough Tool Execution
**Description:** As a user, I want REST-backed tools to execute without a dedicated MCP server.

**Acceptance Criteria:**
- [ ] Adapter can call REST APIs using OpenAPI-derived schemas
- [ ] Errors are returned in a consistent MCP tool response format
- [ ] Typecheck/lint passes

### US-005: Context Forge Integration
**Description:** As a platform engineer, I want Context Forge to discover tools from the adapter so Automatos uses a single gateway.

**Acceptance Criteria:**
- [ ] Adapter is registered as a single MCP gateway in Context Forge
- [ ] Tools are auto-discovered and visible in Context Forge UI
- [ ] Tool execution works via `mcp.automatos.app`
- [ ] Typecheck/lint passes

---

## 11) Authentication & Security

### Current Auth Model (Automatos)
- Credentials are stored encrypted in Automatos and resolved via the credential store.
- Tool execution uses resolved credentials and injects into tool execution context.
- API access is controlled via existing JWT/API key mechanisms.

### Context Forge Auth
- Context Forge uses JWT bearer tokens for admin/API access.
- Context Forge should only expose a **single gateway endpoint** publicly.

### Adapter Auth
- Adapter should authenticate to Automatos using a **service token** or internal JWT.
- Adapter should validate incoming requests from Context Forge.

### IAM Considerations
IAM should be evaluated for a future hardening phase:
- Per-tenant scoped credentials
- Per-tool and per-method permissions
- Role-based access policies (admin, developer, tenant admin)
- Audit trails for tool access by user/agent

For MVP, keep authentication simple (service-level JWT + existing Automatos auth) while documenting upgrade paths for IAM.

---

## 12) Functional Requirements

- **FR-1:** Adapter must expose MCP endpoints compatible with Context Forge discovery.
- **FR-2:** Adapter must load tool metadata from Automatos (DB or API).
- **FR-3:** Adapter must resolve credentials from Automatos secure store.
- **FR-4:** REST tools must be supported using OpenAPI schemas.
- **FR-5:** Adapter must support tool scoping (enabled/disabled, category).
- **FR-6:** Adapter must emit structured errors and usage logs.
- **FR-7:** Context Forge remains the single external entry point.

---

## 13) Non-Goals (Out of Scope)

- Full coverage of 400+ integrations in MVP.
- Multi-tenant isolation enforcement (MVP is single-tenant dev).
- Usage-based billing or quota enforcement in adapter.
- Complex OAuth flows beyond existing credential types.

---

## 14) Design Considerations

- Automatos UI remains the **tool catalog + credential UX**.
- Adapter provides **execution only**, not configuration UI.
- Context Forge remains the **external gateway** for all MCP access.
- Tool naming should remain compatible with Automatos (`mcp_<tool>_<method>`).

---

## 15) Technical Considerations

- **Service Placement:** Dedicated container/service with isolated scaling.
- **MCP Transport:** Streamable HTTP preferred; SSE optional.
- **REST Schema Sources:** APIs.guru + Postman + RapidAPI registries.
- **Security:** JWT between Automatos ↔ Adapter; JWT between Automatos ↔ Context Forge.
- **Tenant-Ready Notes:** Add tenant_id fields in tool execution context; plan for scoped credential resolution.

---

## 16) Implementation Plan (Phased)

### Phase 0: Preparation
- Confirm adapter service repo location and deployment target
- Ensure Context Forge gateway stable at `mcp.automatos.app`
- Validate credential storage and audit logging

### Phase 1: Adapter MVP
- Implement adapter service with MCP discovery + execution
- Integrate Automatos tool catalog + credential resolution
- Support REST passthrough for 5–10 tools

### Phase 2: Context Forge Integration
- Register adapter as a single gateway
- Validate discovery and tool listing
- End-to-end tool calls via Automatos chat

### Phase 3: Expand Coverage
- Add 10–20 tools (mix of REST/MCP)
- Add retry, rate limit handling, and error normalization

---

## 17) Testing Plan

- **Unit:** Tool execution mapping, credential resolution, schema validation
- **Integration:** Context Forge discovery + tool invocation
- **E2E:** Chat → tool call → external API → response
- **Security:** Credential redaction tests, JWT auth validation

---

## 18) Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| OAuth flow complexity | Slower onboarding | Start with API-key tools; add OAuth later |
| API spec gaps | REST passthrough failure | Use Postman/OpenAPI registries and manual specs |
| Vendor rate limits | Tool failures | Built-in backoff + caching |
| Tool overload in LLM | Poor tool selection | Limit tool exposure per agent or task |

---

## 19) Success Metrics

- **SM-1:** 10–20 tools operational via Context Forge within MVP.
- **SM-2:** Time to add a new tool < 1 day for REST-backed tools.
- **SM-3:** Adapter uptime ≥ 99% in dev environment.
- **SM-4:** 0 credential leak incidents.

---

## 20) Open Questions

- Should Automatos store OpenAPI specs internally or reference external registries?
- Which 10–20 tools should be first (final list and owners)?
- Do we allow direct REST passthrough in Context Forge without the adapter?
- How will tenant scoping be enforced once multi-tenant is enabled?

