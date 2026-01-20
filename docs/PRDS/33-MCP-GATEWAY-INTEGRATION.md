# PRD-33: MCP Gateway Integration (Context Forge)

**Version:** 1.0  
**Status:** 🟡 Design Phase  
**Priority:** HIGH - External Integrations  
**Author:** Automatos AI Platform Team  
**Last Updated:** 2026-01-14  
**Dependencies:** PRD-20 (MCP Integration), PRD-18 (Credential Management)

---

## Executive Summary

Adopt **IBM Context Forge** as a unified MCP Gateway to centralize tool execution, authentication, and protocol translation for Automatos AI. This PRD defines the MVP integration for a **single-tenant dev environment** that uses **Context Forge as the credential authority** while Automatos continues to provide the user experience for credential and tool management.

References:
- Context Forge repo: https://github.com/IBM/mcp-context-forge  
- Context Forge docs: https://ibm.github.io/mcp-context-forge/  
- Overview article: https://medium.com/@crivetimihai/mcp-gateway-the-missing-proxy-for-ai-tools-2b16d3b018d5

---

## 1) Problem Statement

### Current Pain
- MCP tools require **live MCP server endpoints**; catalog URLs are not executable.
- Managing 400+ integrations individually is slow and brittle.
- Auth and tool execution are scattered across multiple servers with inconsistent protocols.
- No unified gateway to wrap REST APIs as MCP tools.

### Impact
- Slower onboarding of external integrations.
- Higher operational complexity and inconsistent security posture.
- Delays in SaaS launch readiness.

---

## 2) Goals & Success Metrics

### Goals
- **G1 (Unified MCP Endpoint):** Automatos calls a single MCP Gateway endpoint for all tools.
- **G2 (Credential Authority):** Context Forge stores credentials; Automatos remains the UX for CRUD.
- **G3 (Tool Coverage):** Enable a small set of tools via MCP or REST wrapping in MVP.
- **G4 (Auth Alignment):** Use JWT auth compatible with Automatos API model.
- **G5 (Tenant-Ready):** Design with clear extension points for future multi-tenancy.

### Success Metrics (MVP)
- **M1:** 3 tools functional end-to-end (e.g., Slack + 2 REST-wrapped tools).
- **M2:** Tool calls succeed via gateway with < 3 retries and < 2s added latency.
- **M3:** Credential sync success rate ≥ 99% in dev.
- **M4:** Zero plaintext credential leakage in logs or UI.

---

## 3) Personas & Core User Journeys

### Personas
- **P1: Automatos Admin (Internal Team):** Manages tool catalog and connectivity.
- **P2: Automatos Developer:** Debugs tool calls and gateway integration.
- **P3: End User (Tenant Admin, future):** Adds credentials and enables tools.

### Core Journeys
#### J1: Add Credential (Automatos UI → Gateway)
1. Admin opens Automatos Credentials tab.
2. Adds/updates Slack credentials.
3. Automatos syncs credential to Context Forge.
4. Credential status shown in Automatos UI.

#### J2: Enable Tool (Automatos UI)
1. Admin enables Slack tool in Automatos.
2. Tool mapping in Automatos references Context Forge tool ID.
3. LLM calls the tool via MCP Gateway.

#### J3: Tool Execution (Chat)
1. User requests a tool action.
2. Automatos sends tool call to Context Forge MCP endpoint.
3. Context Forge handles auth and routes to MCP or REST integration.
4. Result returns to Automatos and is surfaced in chat.

---

## 4) Scope (MVP)

### In Scope
- Context Forge gateway running in Railway (Docker) using existing Postgres + Redis.
- Automatos UI continues to manage credentials; credentials are synced to Context Forge.
- Tool execution routed through Context Forge MCP endpoint.
- Support for **MCP servers** and **REST-wrapped tools**.
- Single-tenant dev configuration, with tenant-ready notes.

### Out of Scope (MVP)
- Full multi-tenant isolation enforcement.
- Automated mapping of 400+ tools.
- Advanced observability or SLA-level performance tuning.
- User-facing tool marketplace changes.

---

## 5) User Stories

### US-001: Deploy Context Forge Gateway on Railway
**Description:** As a platform engineer, I want to run Context Forge in Railway so Automatos can call a single MCP endpoint.

**Acceptance Criteria:**
- [ ] Context Forge container runs in Railway and is reachable via `mcp.automatos.app`
- [ ] Gateway uses Postgres and Redis connections from Railway
- [ ] Health check endpoint returns healthy
- [ ] Typecheck/lint passes

### US-002: Credential Sync from Automatos to Context Forge
**Description:** As an admin, I want credentials saved in Automatos to be synced to Context Forge so tool auth is managed centrally.

**Acceptance Criteria:**
- [ ] Create/update/delete in Automatos triggers corresponding sync in Context Forge
- [ ] Sync status is visible in Automatos UI
- [ ] No credential values are logged in plaintext
- [ ] Typecheck/lint passes
- [ ] Verify in browser using dev-browser skill

### US-003: Tool Catalog Mapping to Context Forge Tools
**Description:** As an admin, I want Automatos tools to map to Context Forge tool IDs so calls are routed correctly.

**Acceptance Criteria:**
- [ ] Automatos tool model stores gateway tool ID or server ID mapping
- [ ] Tool enablement in Automatos respects mapping
- [ ] Typecheck/lint passes

### US-004: Execute MCP Tool via Gateway
**Description:** As a user, I want tool calls to go through Context Forge so MCP execution is consistent.

**Acceptance Criteria:**
- [ ] Tool execution uses gateway endpoint (not direct vendor URL)
- [ ] Errors from gateway are shown clearly in Automatos logs/UI
- [ ] Typecheck/lint passes

### US-005: REST-Wrapped Tool via Gateway
**Description:** As an admin, I want to expose a REST integration as an MCP tool so LLMs can call it.

**Acceptance Criteria:**
- [ ] A REST API tool is registered in Context Forge and callable from Automatos
- [ ] Tool response is correctly formatted for chat
- [ ] Typecheck/lint passes

---

## 6) Functional Requirements

- **FR-1:** Automatos must call a single MCP gateway endpoint for tool execution.
- **FR-2:** Context Forge must be the credential authority for MCP tool execution.
- **FR-3:** Automatos must sync credential CRUD to Context Forge with status feedback.
- **FR-4:** Tool registry in Automatos must map tools to Context Forge IDs.
- **FR-5:** Gateway must support MCP tools and REST-wrapped tools.
- **FR-6:** JWT auth must be used between Automatos and the gateway.
- **FR-7:** MVP must support at least 3 functional tools.

---

## 7) Non-Goals (Out of Scope)

- Full multi-tenant isolation enforcement in gateway.
- Automated import of all 400+ tools.
- High-availability or multi-region deployment.
- Advanced observability pipelines (OpenTelemetry, Phoenix).

---

## 8) Design Considerations (Optional)

- Automatos UI remains the single UX for credential management.
- Context Forge Admin UI used by internal team for tool registration.
- Add a "Gateway Sync Status" indicator in Automatos Credentials UI.
- Keep tool enablement UX unchanged for users.

---

## 9) Technical Considerations (Optional)

- **Gateway Hosting:** Railway Docker build from Context Forge container image or Dockerfile.
- **Database:** Use existing Postgres instead of MariaDB (Context Forge supports Postgres).
- **Redis:** Use existing Redis instance for caching/session support.
- **Auth:** JWT between Automatos and gateway; plan for tenant token later.
- **Tenant-Ready:** Store tenant_id in Automatos models for future isolation.
- **Security:** No credentials in logs; use encrypted storage in Context Forge.

---

## 10) Success Metrics

- **SM-1:** MVP runs 3 tools end-to-end through gateway.
- **SM-2:** Credential sync success rate ≥ 99%.
- **SM-3:** Gateway tool call error rate < 5% (dev).
- **SM-4:** Added latency from gateway hop < 2 seconds.

---

## 11) Open Questions

- How should Automatos model gateway tool IDs (single ID vs server+tool)?
- Should Automatos cache tool schemas from Context Forge or fetch dynamically?
- Should gateway credential storage fully replace Automatos credential DB long-term?
- Which 3 MVP tools should be prioritized (Slack + 2 REST tools)?

