# PRD-150: Auth Decoupling — Pluggable AuthProvider & Open-Core Edition Split

**Status:** Draft (reviewed 2026-06-09 — Q1/Q6 resolved, Wave-5 sequencing + publication gate added)
**Author:** Gerard Kavanagh (with Auto)
**Date:** 2026-06-09
**Type:** Refactor / Consolidation (extract a provider boundary; delete the leakage layer)
**Related:** PRD-37 (SaaS Foundation — introduced Clerk), PRD-09 (Board SDK-auth unlock), PRD-151 (Storage decoupling — MinIO default), PRD-152 (mem0 & internal-services decoupling), PRD-153 (One-command local run)

---

## 1. Introduction / Overview

Automatos started as an open-source platform you could clone and run locally. It is now a full multi-tenant SaaS whose authentication is hard-wired to **Clerk** (a proprietary, paid identity service). That single dependency makes the open-source promise false: you cannot `git clone && docker compose up` without Clerk credentials and a Clerk tenant.

This PRD makes authentication **pluggable** behind a single `AuthProvider` interface so that:

- **Open-source (OSS) edition** runs locally / on Railway in a **single workspace with no login** — zero external auth dependencies.
- **SaaS / Enterprise edition** keeps Clerk exactly as it works today — multi-tenant, multi-org, invitations, the lot — by installing a **separate private package** (`automatos-saas`) that registers the Clerk provider.

The core never imports the SaaS package; the SaaS package extends the core. The request-handling contract (`RequestContext`) is identical in both editions, so the ~112 files that consume it do not change.

This is a **refactor**, not a feature. Per the platform's clean-coding rules, the Clerk-specific code is **moved and the originals deleted in the same change** — no `_legacy` shims, no dual code paths.

### Why now

Lack of contributors traces directly to "can't run it locally." Auth is the highest-leverage decoupling: it is the only dependency that blocks the *front door* (you cannot log in at all without a Clerk tenant), and the data contract is already provider-agnostic (see §7). It is not the only cloud hard-dependency — AWS S3 blocks document features at runtime (PRD-151) and several `*.railway.internal` defaults assume Railway's private network (PRD-152) — but it is the one that proves the open-core boundary (provider interface + entry-point registry + import-linter + edition config + both-provider CI) that those PRDs then repeat with smaller blast radius.

---

## 2. Current-State Coupling Map (graph-grounded)

Extracted from `graphify-out/graph.json` (87 Clerk nodes, 242 auth nodes) and cross-checked against live grep on 2026-06-09. Clerk coupling is **three layers**, and the third is the real cost.

### Layer 1 — Mechanism (cleanly contained)
- [`core/auth/clerk.py`](../../orchestrator/core/auth/clerk.py) — the `ClerkAuth` class (40 graph nodes). Public surface:
  `verify_token`, `extract_user_info`, `is_configured`, `get_org_members`, `invite_to_org`, `create_user_invitation`, `revoke_user_invitation`, `remove_from_org`, `delete_user`, `_fetch_email_from_clerk_api`.
- [`core/auth/hybrid.py`](../../orchestrator/core/auth/hybrid.py) — the resolver. Clerk-specific helpers: `_resolve_workspace_for_clerk_user`, `_user_is_workspace_member`, `_user_has_workspace_access`, auto-provision-personal-workspace. **Also** multiplexes non-Clerk auth (API key, SDK key, anonymous) which must stay in core.

### Layer 2 — Schema (DB-level coupling)
- `users.clerk_user_id` (unique constraint `users_clerk_user_id_key`, index `idx_users_clerk`)
- `workspaces.clerk_org_id`
- `workspace_invitations.clerk_invitation_id` (migration `add_clerk_invitation_id.py`)
- Model docstring: `core/models/core.py` — *"User model with Clerk authentication (PRD-37)"*

### Layer 3 — Leakage (the expensive layer)
**~19 API files independently resolve `clerk_user_id → users.id`** instead of receiving an already-resolved internal id. Confirmed examples:
- [`api/widget_marketplace.py`](../../orchestrator/api/widget_marketplace.py) — *"Resolve clerk_user_id -> users.id"*
- [`api/workspace_skills.py`](../../orchestrator/api/workspace_skills.py) — *"Look up the integer users.id for the current Clerk user"*
- [`api/admin_prompts.py`](../../orchestrator/api/admin_prompts.py) — *"Require admin role for Clerk users; allow API key auth"*

Plus the **frontend**: [`frontend/components/clerk-api-client-provider.tsx`](../../frontend/components/clerk-api-client-provider.tsx) and `frontend/lib/api-client.ts` (`setClerkTokenGetter`).

**Blast radius:** 112 files import `get_request_context_hybrid` / `RequestContext`. They are downstream of the boundary and **must not change** — that is the test of a correct refactor.

---

## 3. Goals

- **G1** — One `AuthProvider` interface; Clerk and Local are interchangeable implementations selected by config.
- **G2** — OSS edition runs with **no external auth service** and **no login**, in a single seeded workspace.
- **G3** — Clerk implementation lives in a **separate private `automatos-saas` package**; `orchestrator.core` never imports it (enforced in CI).
- **G4** — Identity resolution is **centralized**: `RequestContext.user.id` carries the internal `users.id`; the ~19 per-endpoint `clerk_user_id → users.id` helpers are **deleted**.
- **G5** — **Zero behavioural change for SaaS.** Existing multi-tenant Clerk flows (JWT, orgs, invitations, auto-provisioning, purge) pass unchanged.
- **G6** — The full auth test suite runs against **both** providers in CI, proving parity and that local mode does not bypass workspace scoping.

---

## 4. User Stories

Grouped into 6 phases. Each story is sized for one focused implementation session.

### Phase 0 — Boundary & scaffolding

#### US-001: Add edition/provider config knobs
**Description:** As a developer, I need canonical config that selects the auth provider so no code reads provider choice from scattered env vars.

**Acceptance Criteria:**
- [ ] `config.py` adds `EDITION` (`"oss" | "saas" | "enterprise"`, default `"oss"`) and `AUTH_PROVIDER` (`"local" | "clerk"`, default `"local"`).
- [ ] All existing `CLERK_*` vars remain in `config.py` and are optional (no value required when `AUTH_PROVIDER=local`).
- [ ] No `os.getenv()` added outside `config.py` (platform rule).
- [ ] Typecheck/lint passes.

#### US-002: Define the `AuthProvider` interface
**Description:** As a developer, I need a single contract every auth backend implements so the resolver depends on an abstraction, not Clerk.

**Acceptance Criteria:**
- [ ] New `core/auth/base.py` defines `AuthProvider` (typing `Protocol`, `@runtime_checkable`).
- [ ] Methods (signatures mirror current `ClerkAuth` so the Clerk impl is a move, not a rewrite):
  - `is_configured -> bool`
  - `async resolve_context(request, db) -> RequestContext` — the whole "HTTP request → principal + workspace" job
  - `async list_members(workspace_id) -> list[dict]`
  - `async invite_member(workspace_id, email, role, inviter_id) -> dict`
  - `async revoke_invitation(invitation_id) -> bool`
  - `async remove_member(workspace_id, user_id) -> None`
  - `async delete_user(user_id) -> bool`
- [ ] A `MemberManagementNotSupported` exception is defined for providers (Local) that have no org plane.
- [ ] `RequestContext` / `UserContext` are reused unchanged from `core/auth/dependencies.py` (no new context types).
- [ ] Typecheck passes.

#### US-003: Provider registry via entry points
**Description:** As a developer, I need the core to discover and select a provider that may be supplied by an external package, without importing it.

**Acceptance Criteria:**
- [ ] New `core/auth/registry.py` discovers providers via `importlib.metadata.entry_points(group="automatos.auth_providers")`.
- [ ] Core registers its own built-in `local` provider through the same entry-point group (in `orchestrator`'s own package metadata).
- [ ] `get_auth_provider()` returns the instance named by `config.AUTH_PROVIDER`, memoized per process.
- [ ] If `AUTH_PROVIDER` names a provider that is not installed (e.g. `clerk` without `automatos-saas`), boot **fails fast** with a clear, actionable error.
- [ ] Unit test: registry selects `local` by default; selecting `clerk` without the package raises the expected error.

#### US-004: Enforce core-never-imports-saas in CI
**Description:** As a maintainer, I need the dependency direction guaranteed so closed code can never leak into the OSS core.

**Acceptance Criteria:**
- [ ] `import-linter` contract added: layers/forbidden rule — `orchestrator` (core) must not import `automatos_saas`.
- [ ] CI job runs the contract and fails the build on violation.
- [ ] Contract documented in `CONTRIBUTING.md` (one paragraph: why the direction is one-way).

### Phase 1 — Local provider (OSS path)

#### US-005: `LocalAuthProvider` returns a single-workspace admin context
**Description:** As a self-hoster, I want the app to treat me as the single owner of the single workspace with no login.

**Acceptance Criteria:**
- [ ] New `core/auth/local.py` implements `AuthProvider`.
- [ ] `resolve_context` returns `RequestContext(workspace_id=<seeded default>, user=UserContext(id=<local users.id>, email="local@localhost", role="owner", system_role="admin"), auth_type="local")`.
- [ ] Member-management methods raise `MemberManagementNotSupported` (single-user edition has no invite flow).
- [ ] `is_configured` is always `True`.
- [ ] No network calls, no external service.
- [ ] Unit test asserts the returned context shape and that `auth_type == "local"`.

#### US-006: Edition-aware boot seed (one workspace + one user, idempotent)
**Description:** As a self-hoster, I need the single workspace and local user to exist on first boot without manual setup.

**Acceptance Criteria:**
- [ ] On boot, **only when `AUTH_PROVIDER=local`**, ensure exactly one workspace (id = `config.DEFAULT_WORKSPACE_ID`, generated + persisted if unset) and one local user exist.
- [ ] Seed is idempotent (safe to run every boot; no duplicates).
- [ ] SaaS boot path does **not** run this seed (Clerk auto-provisions per org).
- [ ] Seed lives in the existing boot sequence (`core/boot/`), not a new ad-hoc script.
- [ ] Integration test: two consecutive boots yield exactly one workspace + one user.

#### US-007: Add `"local"` to the auth-type contract & resolver fallthrough
**Description:** As a developer, I need the resolver to recognise the local provider as a first-class auth type.

**Acceptance Criteria:**
- [ ] `RequestContext.auth_type` docstring updated to include `"local"`.
- [ ] `get_request_context_hybrid` dispatch order (see US-010) ends in the configured provider, returning `auth_type="local"` in OSS.
- [ ] No endpoint special-cases `"local"`; workspace filtering runs identically (local resolves to a real `workspace_id`).
- [ ] Test: an arbitrary existing endpoint returns 200 in local mode and is scoped to the single workspace.

### Phase 2 — Extract Clerk into `automatos-saas` (full split)

#### US-008: Create the `automatos-saas` private package skeleton
**Description:** As a maintainer, I need a separate installable package to hold all closed-source provider code.

**Acceptance Criteria:**
- [ ] New private package `automatos-saas` (own `pyproject.toml`, own license header — not OSS).
- [ ] Declares entry point: `[project.entry-points."automatos.auth_providers"] clerk = "automatos_saas.auth.clerk_provider:ClerkAuthProvider"`.
- [ ] Depends on `orchestrator` core; core has **no** dependency on it.
- [ ] SaaS install path documented: `pip install -e orchestrator && pip install -e automatos-saas`.

#### US-009: Move `ClerkAuth` → `ClerkAuthProvider` and delete from core
**Description:** As a developer, I need the Clerk mechanism relocated behind the interface with no duplicate left in core.

**Acceptance Criteria:**
- [ ] `ClerkAuthProvider` (in `automatos_saas`) implements `AuthProvider`; wraps the existing `ClerkAuth` logic (JWT verify, email fetch, org/member ops) — **moved, not rewritten**.
- [ ] Clerk-specific workspace-resolution helpers (`_resolve_workspace_for_clerk_user`, `_user_is_workspace_member`, `_user_has_workspace_access`, auto-provision) move into the provider.
- [ ] Clerk webhook handling currently in core (`hybrid.py`, `api/notifications.py`, `api/workspaces.py`, `api/team.py`, `api/workflow_recipes.py`) is relocated or routed through the provider; **identify each Clerk webhook touchpoint and move or gate it** (see Open Questions Q4).
- [ ] `core/auth/clerk.py` is **deleted** (no `_legacy`).
- [ ] `grep -rn "import.*clerk\|ClerkAuth" orchestrator/` returns **zero** results outside tests that exercise the SaaS edition.

#### US-010: Refactor `get_request_context_hybrid` into a thin provider dispatcher
**Description:** As a developer, I need the resolver to delegate the human-user path to the selected provider while keeping edition-independent auth (API key, SDK key) in core.

**Acceptance Criteria:**
- [ ] `get_request_context_hybrid` becomes: try **SDK key** (board plane, `ak_pub_*`/`ak_srv_*`) → **API key** (service-to-service, `ORCHESTRATOR_API_KEY`) → **`provider.resolve_context()`** (Clerk or Local) → **anonymous** (only if `REQUIRE_AUTH=false`).
- [ ] SDK-key and API-key logic **stays in core** (both editions need it; not Clerk-specific).
- [ ] The inline Clerk branch is **deleted** from `hybrid.py` (now lives in the provider).
- [ ] All 112 `RequestContext` consumers compile and pass with **no signature changes**.
- [ ] `test_board_sdk_auth.py` passes unchanged (board plane behaviour preserved).

#### US-011: Route org/member management through the provider
**Description:** As a developer, I need `api/team.py` and `services/workspace_purge.py` to call the provider interface, not Clerk directly.

**Acceptance Criteria:**
- [ ] `api/team.py` invite/revoke/list/remove call `provider.invite_member / revoke_invitation / list_members / remove_member`.
- [ ] `services/workspace_purge.py` calls `provider.delete_user` (no direct `clerk.delete_user`).
- [ ] In OSS, the team-management router is **not mounted** (single-user edition); requests to it 404 cleanly.
- [ ] No direct `clerk` symbol referenced in `api/` or `services/` after this story.

### Phase 3 — Centralize identity (delete the leakage layer)

#### US-012: Resolver populates the internal `users.id`
**Description:** As a developer, I want `ctx.user.id` to already be the internal `users.id` so endpoints never resolve identity themselves.

**Acceptance Criteria:**
- [ ] `ClerkAuthProvider.resolve_context` resolves `clerk_user_id → users.id` **once** and sets `UserContext.id` to the internal id; `clerk_user_id` stays available on `UserContext.clerk_user_id` for provider-internal use only.
- [ ] `LocalAuthProvider` sets `UserContext.id` to the seeded local user's id.
- [ ] `UserContext.id` carries the internal id as **`int`** (settled — Q1 resolved: `users.id` is an `Integer` PK; the UUID assumption in `widget_marketplace.py` was a stale comment). Callers that stringify must do so at their own boundary.
- [ ] Test: `ctx.user.id` equals the DB `users.id` for both providers.

#### US-013: Delete the ~19 per-endpoint `clerk_user_id → users.id` helpers
**Description:** As a maintainer, I need the duplicated identity-resolution code gone so Clerk identity does not leak across the API surface.

**Acceptance Criteria:**
- [ ] Every endpoint that previously resolved `clerk_user_id → users.id` (incl. `widget_marketplace.py`, `workspace_skills.py`, `admin_prompts.py`, and the rest of the ~19) now uses `ctx.user.id` directly.
- [ ] The local helper functions are **deleted** (not left dormant).
- [ ] `grep -rn "clerk_user_id" orchestrator/api orchestrator/services` returns **zero** results.
- [ ] Each touched endpoint has a regression test (or an existing one) proving identical behaviour.

#### US-014: Migration — make Clerk columns optional, add provider marker
**Description:** As a developer, I need the schema to support non-Clerk identities without breaking existing Clerk rows.

**Acceptance Criteria:**
- [ ] `users.clerk_user_id` is **already `nullable=True` in the model** (verified 2026-06-09) — the migration's job is parity: confirm the live DB column matches the model (the platform has known model↔DB drift, e.g. `escalation_level`); fix only if drifted. Unique constraint retained — Postgres allows multiple NULLs. Same verify-don't-assume for `workspaces.clerk_org_id` and `workspace_invitations.clerk_invitation_id`.
- [ ] Add `users.auth_provider` (`varchar`, nullable) — backfill existing rows to `'clerk'`; seeded local user is `'local'`.
- [ ] Migration is reversible (`downgrade` defined).
- [ ] Migration runs clean on a Clerk-populated DB and a fresh OSS DB.
- [ ] Sequencing: lands **after** PRD-142 Wave 5's table folds merge (single Alembic queue — see Risk & sequencing).

### Phase 4 — Frontend (full-stack)

#### US-015: Edition flag drives Clerk mounting
**Description:** As a self-hoster, I want the OSS frontend to render with no Clerk provider and no login screen.

**Acceptance Criteria:**
- [ ] `NEXT_PUBLIC_EDITION` (`oss` | `saas`) added; sourced through the frontend's config module.
- [ ] When `oss`, `<ClerkProvider>` is **not** mounted; the app boots straight into the single workspace.
- [ ] When `saas`, current Clerk behaviour is unchanged.
- [ ] Verify in browser using dev-browser skill: OSS build loads the app with no sign-in redirect.

#### US-016: `setClerkTokenGetter` becomes a no-op in OSS
**Description:** As a developer, I need the API client to work without Clerk tokens in OSS.

**Acceptance Criteria:**
- [ ] In OSS, `api-client` sends no Clerk bearer token (or a static local token the backend accepts in local mode).
- [ ] `clerk-api-client-provider.tsx` is conditionally rendered (SaaS only) or replaced by a `LocalAuthProvider` shim component.
- [ ] No runtime error when Clerk SDK env vars are absent.
- [ ] Verify in browser using dev-browser skill: API calls succeed in OSS build.

#### US-017: Hide auth/team UI in OSS single-workspace mode
**Description:** As a self-hoster, I should not see sign-in, org-switcher, or team-invite UI that has no meaning in single-workspace mode.

**Acceptance Criteria:**
- [ ] Sign-in / sign-up / org-switcher / team-invite UI is hidden when `EDITION=oss`.
- [ ] No dead links or broken pages result.
- [ ] Verify in browser using dev-browser skill.

### Phase 5 — Parity, runnability, docs

#### US-018: Run the auth test suite against both providers
**Description:** As a maintainer, I need proof that OSS and SaaS share one code path and local mode never bypasses workspace scoping.

**Acceptance Criteria:**
- [ ] Auth/integration tests are parametrized to run under `AUTH_PROVIDER=local` **and** `AUTH_PROVIDER=clerk` (Clerk mocked).
- [ ] A test explicitly asserts that a request in local mode is still filtered by `workspace_id` (no scoping bypass).
- [ ] CI runs both parameter sets and both must pass.

#### US-019: OSS compose boots with no auth service + QUICKSTART
**Description:** As a contributor, I want one command to run the whole OSS stack locally with no proprietary services.

**Note (2026-06-09):** the compose definition itself is owned by **PRD-153** (which fixes the broken init-schema mount and consolidates the 7 drifted `infrastructure/docker-compose.*.yml` files). This story owns only the **auth slice** of that boot. Do not build a parallel compose here.

**Acceptance Criteria:**
- [ ] The PRD-153 compose boots with **no Clerk env vars set** and `AUTH_PROVIDER=local`, `EDITION=oss` — no auth service, no login screen.
- [ ] Boots green with a single required secret (one LLM API key) plus generated DB/Redis passwords.
- [ ] `QUICKSTART.md` updated: clone → set one key → `docker compose up` → working app, single workspace, no login.
- [ ] Verified end-to-end on a clean checkout (document the run).

#### US-020: Provider unit/contract tests
**Description:** As a developer, I need targeted tests for the new seam.

**Acceptance Criteria:**
- [ ] `LocalAuthProvider` context shape test; boot-seed idempotency test; registry selection + missing-package error test.
- [ ] Contract test: both providers satisfy `AuthProvider` (`runtime_checkable` assertion).
- [ ] Assertion test (grep-style or AST): no `clerk` symbol imported anywhere under `orchestrator/`.
- [ ] Coverage for new modules ≥ 80% (platform rule).

---

## 5. Functional Requirements

- **FR-1** — `config.py` exposes `EDITION` and `AUTH_PROVIDER`; defaults yield a working OSS build with no Clerk config.
- **FR-2** — A single `AuthProvider` Protocol (`core/auth/base.py`) defines identity + member-management surface.
- **FR-3** — Providers are discovered via the `automatos.auth_providers` entry-point group and selected by `AUTH_PROVIDER`.
- **FR-4** — Core (`orchestrator`) must not import `automatos_saas`; CI enforces this with `import-linter`.
- **FR-5** — `LocalAuthProvider` returns a fixed single-workspace owner context, `auth_type="local"`, with no network calls.
- **FR-6** — Boot seeds exactly one workspace + one local user when `AUTH_PROVIDER=local`, idempotently; SaaS boot does not.
- **FR-7** — `ClerkAuthProvider` lives only in `automatos-saas`; `core/auth/clerk.py` is deleted.
- **FR-8** — `get_request_context_hybrid` dispatch order: SDK key → API key → provider → anonymous; SDK/API-key paths remain in core.
- **FR-9** — `RequestContext.user.id` is the internal `users.id` for all providers; endpoints never resolve identity.
- **FR-10** — The ~19 per-endpoint `clerk_user_id → users.id` helpers are deleted; `clerk_user_id` does not appear under `orchestrator/api` or `orchestrator/services`.
- **FR-11** — Migration makes Clerk columns nullable and adds `users.auth_provider`, backfilled.
- **FR-12** — `api/team.py` and `services/workspace_purge.py` call provider methods, not Clerk directly; team router is unmounted in OSS.
- **FR-13** — Frontend mounts Clerk only when `EDITION=saas`; OSS renders single-workspace, no login.
- **FR-14** — `docker-compose.oss.yml` runs the full OSS stack with only DB, Redis, backend, frontend.
- **FR-15** — CI runs the auth suite under both `local` and `clerk` providers; both pass.

---

## 6. Non-Goals (Out of Scope)

- **Billing, plans, tier enforcement, quotas** — the SaaS wrapper's commercial layer is a later PRD.
- **SSO / SAML / SCIM / audit log** — enterprise features, not this PRD.
- **Additional providers** (Auth0, Cognito, Keycloak) — the interface must *allow* them; we ship only Local + Clerk.
- **Local multi-user accounts / login UI** — explicitly chosen out: OSS is single-workspace, no-login. (Operators exposing OSS publicly put their own proxy/auth in front.)
- **Decoupling S3/object storage** — **PRD-151** (S3-compatible factory + MinIO default; pgvector stays the vector default).
- **Decoupling mem0 + `*.railway.internal` services** — **PRD-152** (mem0 self-hosted in compose; telemetry/voice/opt-worker defaults made local-safe).
- **Compose consolidation / one-command local boot** — **PRD-153** (also fixes the broken init-schema mount and the schema-lifecycle drift).
- **Knowledge Graph and NL2SQL need NO decoupling** — verified 2026-06-09: KG lives in Postgres (`workspace_graphs`, no graph DB), NL2SQL is pure LLM. Do not invent provider interfaces for them.
- **Marketplace decoupling** — a product decision (central registry vs bundled content), not surgery; deferred until after PRD-151 (its assets ride the same S3 factory).
- **Physical split of the frontend into two repos** — frontend stays one codebase, gated by the edition flag.
- **Changing the 112 `RequestContext` consumers' signatures** — they must remain untouched; if any must change, that is a red flag to stop and reassess.

---

## 7. Technical Considerations

### The contract is already provider-agnostic
[`core/auth/dependencies.py`](../../orchestrator/core/auth/dependencies.py) already models this split:
- `UserContext` has `id`, `email`, `role`, `system_role` as the generic principal, and comments `clerk_user_id` / `org_id` as *"Auth-provider specific fields (optional)"*.
- `RequestContext.auth_type` already enumerates `"clerk" | "api_key" | "sdk_key" | "anonymous"`.

So Phase 3 is mostly about **filling `UserContext.id` correctly and removing the optional Clerk fields' use from business logic** — not redesigning the contract.

### What stays in core vs. moves to `automatos-saas`
| Concern | Lives in | Reason |
|---|---|---|
| `AuthProvider` interface, registry, `LocalAuthProvider` | **core** | OSS must work standalone |
| API-key auth (`ORCHESTRATOR_API_KEY`) | **core** | edition-independent service-to-service |
| SDK-key / board-plane auth (`ak_pub_*`/`ak_srv_*`) | **core** | edition-independent (PRD-09) |
| `RequestContext` / `UserContext` / `get_request_context_hybrid` shim | **core** | the shared contract |
| `ClerkAuthProvider`, Clerk JWT/JWKS, org ops, Clerk webhooks | **automatos-saas** | proprietary, SaaS-only |

### Platform clean-coding rules (non-negotiable here)
- **No backward-compat shims.** `core/auth/clerk.py` and the 19 helpers are **deleted** in the same change that replaces them.
- **No `os.getenv()` outside `config.py`.** `EDITION` / `AUTH_PROVIDER` go through config.
- **Reuse over build.** Reuse `UserContext` / `RequestContext`; do not invent new context types. Reuse the existing `core/boot/` sequence for seeding.
- **Delete what you replace.** Remove orphaned imports and the unmounted team router wiring in OSS.

### Risk & sequencing
- The risky stories are **US-010** (resolver dispatch) and **US-013** (deleting leakage). Land Phase 0–1 (additive, no deletion) first; the app still runs on Clerk throughout. Only switch the dispatcher and delete Clerk-from-core once Local + registry + tests are green.
- Keep `git` history clean: one phase per PR, each independently green, each deleting what it supersedes.
- **Parity gate:** US-018 (both-provider CI) must be in place *before* US-009/US-013 deletions merge, so regressions are caught immediately.
- **PRD-142 Wave 5 interaction (decided 2026-06-09):** Phases 0–1 are additive and may run in a parallel worktree *while* the Wave 5 cut list executes. Phases 2–3 must wait until Wave 5 merges — both campaigns delete across `orchestrator/api` (direct collision: `api/workflow_recipes.py` is a Q4 webhook touchpoint *and* a Wave-5 playbook-fold target), and Alembic heads must stay serialized (Wave 5 drops/folds tables; US-014 alters `users`). Refactoring auth through files Wave 5 is about to delete is throwaway work — cut first, then sweep.
- **Publication gate (separate from this refactor):** this PRD makes the core OSS-*ready*; actually publishing the repo (or the extracted core) additionally requires the open security-hygiene work to complete first — full secrets rotation and a git-history audit/scrub. Track that as its own checklist; do not flip the repo public on the strength of this PRD alone.

### Performance
- `resolve_context` does one extra DB lookup (`clerk_user_id → users.id`) that endpoints previously did N times — net **fewer** queries. No regression expected.

---

## 8. Success Metrics

- **M1 (the headline):** A fresh `git clone` + one LLM key + `docker compose -f docker-compose.oss.yml up` yields a working app, single workspace, no login — measured on a clean machine. *Scope note (2026-06-09): this PRD delivers the no-Clerk half of M1. Full M1 additionally requires PRD-151 (document features currently fail at runtime without AWS creds — `DocumentManager` is lazy, so boot is green but uploads/RAG are not) and PRD-153 (the root compose's init-schema mount points at a non-existent path today, so a clean clone gets an empty database). The three PRDs share M1 as the program-level headline.*
- **M2:** `import-linter` reports **zero** `core → automatos_saas` imports in CI.
- **M3:** `grep -rn "clerk" orchestrator/ --include=*.py` (excluding SaaS-edition tests) returns **zero** results.
- **M4:** Full auth suite green under **both** `AUTH_PROVIDER=local` and `clerk`.
- **M5 (no SaaS regression):** existing Clerk multi-tenant journeys (JWT login, org→workspace, invite/revoke, auto-provision, purge) pass unchanged.
- **M6:** Net lines deleted > net lines added in `orchestrator/api` + `orchestrator/services` (leakage removal).

---

## 9. Open Questions

- **Q1 — `users.id` type. RESOLVED 2026-06-09:** `Integer` is authoritative — `core/models/core.py` declares `id = Column(Integer, primary_key=True)`; `clerk_user_id = Column(String(255), unique=True, nullable=True)`. The "UUID" comment in `api/widget_marketplace.py` is simply wrong (delete it with the helper in US-013). `UserContext.id` carries the `int` directly — no string encoding.
- **Q2 — API-key admin context.** Today an API key yields an admin/all-workspaces context. In OSS single-workspace mode, is API-key auth still wanted, or is local-owner sufficient? (Proposed: keep it; it's edition-independent.)
- **Q3 — `DEFAULT_WORKSPACE_ID` provenance.** Generate-and-persist on first boot, or require the operator to set it? (Proposed: auto-generate, persist, log it.)
- **Q4 — Clerk webhooks.** Five files reference Clerk + webhook (`hybrid.py`, `api/notifications.py`, `api/workspaces.py`, `api/team.py`, `api/workflow_recipes.py`). Enumerate each: which are Clerk user/org sync (move to `automatos-saas`) vs. unrelated workspace `webhook_key` (stays in core)?
- **Q5 — Frontend packaging.** One build with a runtime/`NEXT_PUBLIC_EDITION` flag (proposed) vs. build-time tree-shaking of Clerk for a smaller OSS bundle?
- **Q6 — PR/worktree strategy. RESOLVED 2026-06-09:** dedicated `feature/auth-decoupling` branch off `main`, one PR per phase. Phases 0–1 may run in a parallel worktree alongside PRD-142 Wave 5; Phases 2–3 merge only after the Wave 5 cut list lands (see Risk & sequencing).

---

## 10. Phase Summary (suggested delivery order)

| Phase | Stories | Character | Reversible? |
|---|---|---|---|
| 0 — Boundary & scaffolding | US-001–004 | Additive | Yes |
| 1 — Local provider | US-005–007 | Additive (app still on Clerk) | Yes |
| 2 — Extract Clerk to package | US-008–011 | **Move + delete** | Harder |
| 3 — Centralize identity | US-012–014 | **Delete leakage** + migration | Migration reversible |
| 4 — Frontend | US-015–017 | Edition-gated UI | Yes |
| 5 — Parity, runnability, docs | US-018–020 | Tests + compose + docs | Yes |

**Gate:** Phase 5's both-provider CI (US-018) must precede the Phase 2/3 deletions in merge order.
