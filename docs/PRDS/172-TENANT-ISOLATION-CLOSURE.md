# PRD-172 — Tenant Isolation Closure (Wave 2)

**Status:** Draft v1 — pending approval
**Type:** Security / Tenancy (P0 multi-tenant isolation)
**Priority:** P0 — WS-1 done properly; tenancy is the single differentiator vs OpenClaw
**Owner:** Gerard Kavanagh
**Author:** Gerard Kavanagh + Claude (Opus 4.8)
**Date:** 2026-07-02
**Phase:** A — Coherence · **Size:** M · **Risk:** medium (per-router dependency additions, **not** a change to the 657-site shared hybrid auth) · runs in **parallel with Wave 1**
**Parent:** [PLATFORM-OS-ROADMAP.md](./PLATFORM-OS-ROADMAP.md)
**Source:** review §4 (High F002/F003/F004/F005/F006/F007; F039/F045; F019), §13 Wave 2 + §13 Tenancy/Security pillars, §14 (production unknowns + owner calls)
**Findings register pinned to:** `37fdecc4e` — re-confirm each `file:line` on current `main` before editing
**Findings in scope:** F002, F003, F004, F005, F006, F007, F039, F045, F019

---

## Operating Principle

> **Every endpoint that touches tenant data proves the tenant.** A request carries a workspace; the handler
> reads, writes, and deletes only within that workspace, or it is not that request's data. Isolation is not a
> property we assert in a design doc — it is a property a **cross-tenant matrix test** demonstrates in CI:
> workspace A cannot read, write, or delete any of workspace B's skills, documents, vectors, workflows, or
> context. This PRD does not build a new auth system. It adds the workspace-scoped dependency and the
> `ctx.workspace_id` filter to the handful of surfaces that skipped them — a per-router change, **not** a
> change to the 657-site shared hybrid auth (memory: `prd09-board-sdk-auth-unlock`).

---

## 1. Purpose

Tenancy is the one thing that makes Automatos a **multi-tenant SaaS operating system** rather than a
single-tenant agent runner — it is the single differentiator vs OpenClaw, and it is the pillar the review
grades hardest. The roadmap's two governing bars for this wave (review §13) are unambiguous pass/fail:

- **Tenancy / RBAC pillar:** "Cross-tenant matrix (workspace A → B read/write/delete on every domain) 100%
  denied … isolation enforced in CI."
- **Security pillar:** "Zero unauthenticated endpoints touching tenant data."

On `origin/main` today neither bar passes. Nine findings describe the same class of defect from three angles:
routers that authenticate and then **never use** `ctx.workspace_id` (F002, F039), routers that require **no
auth at all** on tenant data (F045, F007), a provisioning surface that is **fail-open** when its key is unset
(F004), a vector search that **silently drops** the workspace filter it accepts (F005), sync routes with **no
scope** that let a guessed UUID trigger costly bulk-ops and overwrite another tenant's graph (F003), a legacy
workflow engine that is a **cross-tenant existence oracle** (F006), and a "SELECT-only" SQL validator that
**passes `UPDATE`** (F019).

None of these needs a new subsystem. Each is a missing dependency, a missing filter, a fail-closed assertion,
or a denylist. **Wave 2 is WS-1 (tenancy) done properly** — the prior workstream did tenancy piecemeal and
left these nine surfaces behind. It runs in **parallel with Wave 1** (Execution Spine Integrity, PRD-171):
W1 makes the engine run, W2 makes the engine's data tenant-safe; they share no files and no ordering
dependency.

---

## 2. Background

### 2.1 What's working today (must not break)

- **The shared hybrid auth dependency exists and is correct.** The workspace-scoped auth context
  (`ctx.workspace_id`) is threaded through the large majority of routers already; the fix on every finding
  below is to **use** the dependency the platform already has, on the routers that skipped it. This is a
  per-router addition, **not** a change to the 657-site shared hybrid auth (memory:
  `prd09-board-sdk-auth-unlock`; review §14). Do not touch the shared dependency itself.
- **The obs tier already has a role gate.** PRD-143 established `require_super_admin` and the obs-tier
  boundary (memory: `prd143-rev2-obs-lock`); F007 is applying that existing gate to two monitoring routers
  that were mounted before it, not inventing a new one.
- **The bucket-name template carries `{workspace_id}`.** F005's isolation model — a per-workspace bucket —
  is a valid design; the defect is that nothing *enforces* the placeholder is present and nothing applies the
  post-filter on the shared/no-team path. The template stays; the assertion and the filter are added.
- **Missions' own tenancy is sound.** The mission lifecycle scopes to workspace correctly; F006 is the
  *legacy* PRD-125 workflow router (a disabled-stub engine), not the mission path. Do not touch missions.

### 2.2 What's broken / blocked

- **F002 — the skills router takes `ctx` and never uses it** (`skills.py:838,731,604`). Any workspace can
  attach another workspace's private `SKILL.md` to its own agent (prompt-injection + exfiltration), read any
  skill's content, and a single `DELETE /{skill_id}` can deactivate a **global builtin-core** skill and
  lobotomise every workspace's Auto.
- **F003 — four Shopify sync routes declare only `db=Depends(get_db)`** (`shopify.py:584-991`). Guessing a
  workspace UUID triggers costly Composio bulk-ops and **overwrites that workspace's knowledge graph** via
  `import_graph(merge=False)`.
- **F004 — the Shopify provisioning surface is fail-open** (`config.py:441`, `shopify.py:43-45`). When
  `SHOPIFY_INTERNAL_API_KEY` is unset, `config.py:441` defaults it to `""` and `_verify_internal_key` returns
  early on a falsy key, so any request carrying any `Authorization` header value is accepted. A missing header
  422s, but `Authorization: Bearer x` sails through to `/provision`, `/connect`, `/deactivate`.
- **F005 — S3 vector search ignores its `filters` param** (`s3_vectors_backend.py:123-146`,
  `rag/service.py:316-317`; gated by `S3_VECTORS_ENABLED=true`). `S3VectorsBackend.search()` accepts `filters`
  and never applies it; isolation rests entirely on a bucket-name template, and nothing validates that
  `S3_VECTORS_BUCKET` embeds the `{workspace_id}` placeholder. On the default no-team path the choke point
  applies no post-filter and `_expand_to_parent_context` hydrates chunks with **no workspace scoping**, so a
  shared bucket leaks cross-workspace chunk text into LLM context.
- **F006 — legacy workflow execute is a cross-tenant oracle** (`workflows.py:1059-1072`, ADJUSTED). The
  endpoint fetches the workflow with **no workspace filter** and selects the first active agent from *any*
  workspace. Under the default `local` runner this is a cross-tenant existence oracle; under `queued` it
  enqueues a real cross-workspace `AgentTask`.
- **F007 — `GET /api/alerts` is unconditionally unauthenticated** (ADJUSTED). The `ALERT_INGEST_TOKEN` bearer
  check guards only the ingest `POST`; the read surface (and its sibling monitoring router) expose tenant
  alert data with no auth. These are the obs tier PRD-143 defines.
- **F039 — `/api/v1/memory` authenticates but never uses `ctx.workspace_id`**. It keys a process-global store
  on caller-supplied `session_id`, so any authenticated caller can read/write another tenant's memory by
  reusing a `session_id`. (Review §14 also lists `/api/v1/memory` as a possible "placebo" to fix-or-delete —
  the fix-or-delete framing is an owner call; the **tenancy fix here is unambiguous**.)
- **F045 — `api/context.py` mounts seven endpoints with no auth dependency** and an unscoped
  `SELECT COUNT(*)`. Unauthenticated tenant-data surface plus a cross-tenant count.
- **F019 — the "SELECT-only" NL2SQL validator passes side-effecting SQL** . It passes
  `query_to_xml('UPDATE ...')`; the SELECT validator does not block side-effecting SQL functions, so a
  crafted "read" can mutate.

### 2.3 Why now

W2 has **no dependency on W1** and is startable immediately in parallel. Until it lands, the platform fails
both the Tenancy and Security enterprise bars, and **the moat pitch is unmakeable**: you cannot sell
per-tenant learned edges (the W7 moat) on a platform where workspace A can read workspace B's documents and
delete its skills. F002 alone is a single-request denial-of-service against every tenant's Auto (delete one
global builtin-core skill). W4 (the unified policy plane) is **gated on W2** in the roadmap — a policy plane
that evaluates tenancy has no correct answer to enforce until these surfaces actually carry a workspace. This
supersedes the prior review's "WS-1 tenancy is done" status for these nine surfaces (roadmap §8: "WS-1 did
tenancy piecemeal").

---

## 3. Findings in scope

| ID | Severity | Location (pinned `37fdecc4e`) | Defect | Fix |
|---|---|---|---|---|
| **F002** | High | `orchestrator/api/skills.py:838,731,604` | Router takes `ctx` and never uses it; cross-workspace skill attach/read + `DELETE /{skill_id}` can deactivate a global builtin-core skill (lobotomises every tenant's Auto) | Filter `Agent` and `Skill` by `ctx.workspace_id`; require ownership **or** super-admin for global-skill delete |
| **F003** | High | `orchestrator/api/shopify.py:584-991` | Four sync routes declare only `db=Depends(get_db)`; a guessed UUID triggers Composio bulk-ops + overwrites the graph via `import_graph(merge=False)` | Add the workspace-scoped auth dependency to the four sync routes |
| **F004** | High | `orchestrator/config.py:441`, `orchestrator/api/shopify.py:43-45` | Provisioning is fail-open: unset `SHOPIFY_INTERNAL_API_KEY` defaults to `""`, `_verify_internal_key` returns early on falsy key → any `Authorization` header accepted | Require the Shopify key **fail-closed at boot**; set it in Railway |
| **F005** | High | `orchestrator/modules/search/vector_store/backends/s3_vectors_backend.py:123-146`, `orchestrator/modules/rag/service.py:316-317` (gated `S3_VECTORS_ENABLED=true`) | `search()` accepts `filters` and never applies it; no validation that `S3_VECTORS_BUCKET` embeds `{workspace_id}`; no-team choke point applies no post-filter and `_expand_to_parent_context` is unscoped → cross-workspace chunk leak into LLM context | Fail startup if the resolved bucket lacks `{workspace_id}`; pass + apply a `workspace_id` metadata filter that drops non-matching hits |
| **F006** | High (ADJUSTED) | `orchestrator/api/workflows.py:1059-1072` | Legacy workflow execute fetches the workflow with no workspace filter and picks the first active agent from *any* workspace; existence oracle (`local`) / cross-workspace `AgentTask` (`queued`) | **Preferred: delete the PRD-125 router** (its engine is a disabled stub — CLAUDE.md §5); **or** scope workflow/agent queries to `ctx.workspace_id` (owner confirmation) |
| **F007** | High (ADJUSTED) | `GET /api/alerts` (read surface in `orchestrator/api/analytics_real.py`) | Unconditionally unauthenticated; `ALERT_INGEST_TOKEN` guards only the ingest `POST` | Wrap both monitoring routers in `require_super_admin` (the obs tier per PRD-143); keep the bearer token **only** for AlertManager ingest |
| **F039** | High | `orchestrator/api/memory.py` (`/api/v1/memory`) | Authenticates but never uses `ctx.workspace_id`; keys a process-global store on caller-supplied `session_id` | Scope memory reads/writes/deletes to `ctx.workspace_id` (fix-or-delete framing is an owner call; the tenancy fix is in scope) |
| **F045** | High | `orchestrator/api/context.py` | Seven endpoints with **no auth dependency** + an unscoped `SELECT COUNT(*)` | Add the workspace-scoped auth dependency; scope the count query to `ctx.workspace_id` |
| **F019** | High | NL2SQL SELECT-only validator (`query_to_xml`) | Passes `query_to_xml('UPDATE ...')`; SELECT validator does not block side-effecting SQL functions | Add a side-effecting-function **denylist** to the SELECT validator (read-only DB role at provisioning is an owner policy call — dependency, not in scope) |

> **Re-confirmation note.** The register is pinned to `37fdecc4e`; current `main` is a later commit. Each
> file confirmed present on `main` at authoring, **except** the NL2SQL validator symbol `query_to_xml`, which
> did not resolve on current `main` — re-confirm the validator's function name and `file:line` before editing
> F019. Confirm every other `file:line` on current `main` before touching it.

---

## 4. Changes (minimal diff, per finding)

**4.1 F002 — scope skills to the workspace; gate global-skill delete.** In `skills.py` (the three sites
`838,731,604` — attach, read, delete), filter both the `Agent` lookup and the `Skill` lookup by
`ctx.workspace_id` so a caller can only attach/read skills its workspace owns. For `DELETE /{skill_id}`:
require that the skill belongs to `ctx.workspace_id`, **or** that the caller is super-admin, before
deactivating; a global builtin-core skill is deletable **only** by super-admin. No workspace-scoped caller
can deactivate a global skill. `ctx` stops being an unused parameter and becomes the filter it was always
meant to be.

**4.2 F003 — add the workspace-scoped auth dependency to the four sync routes.** In `shopify.py:584-991`,
change the four sync route signatures from `db=Depends(get_db)` alone to also take the workspace-scoped auth
context, and scope the sync + `import_graph` target to `ctx.workspace_id`. A guessed UUID can no longer
trigger another workspace's bulk-ops or overwrite its graph. `import_graph(merge=False)` still overwrites —
but only the caller's own graph, which is the intended behaviour.

**4.3 F004 — require the Shopify key fail-closed at boot.** At `config.py:441`, stop defaulting
`SHOPIFY_INTERNAL_API_KEY` to `""` for the provisioning surface; require it at startup and **fail boot** if
unset (matching the platform's fail-closed secret pattern — memory notes the analogous S3 startup assertion in
4.5). In `_verify_internal_key` (`shopify.py:43-45`), remove the early-return-on-falsy-key branch so a falsy
configured key can never accept an arbitrary `Authorization` value; with the key mandatory at boot, the
verifier only ever compares against a real secret. **Owner input (does not change the code fix):** confirm
`SHOPIFY_INTERNAL_API_KEY` is set in Railway (see §6) — the fail-closed requirement is correct either way; if
it is currently unset, this change will (correctly) fail the deploy until it is set.

**4.4 F005 — assert the bucket placeholder at startup; apply the workspace filter.** Two changes, both gated
by `S3_VECTORS_ENABLED=true`:
- **Startup assertion:** if `S3_VECTORS_ENABLED` is true and the resolved `S3_VECTORS_BUCKET` does **not**
  contain the `{workspace_id}` placeholder, **fail startup** (a shared bucket with no placeholder is a
  cross-tenant leak by construction).
- **Filter application:** in `S3VectorsBackend.search()` (`s3_vectors_backend.py:123-146`), apply the
  `filters` param it already accepts — pass a `workspace_id` metadata filter and drop hits whose metadata
  `workspace_id` does not match. At the RAG choke point (`rag/service.py:316-317`) ensure the no-team path
  passes the `workspace_id` filter, and that `_expand_to_parent_context` hydrates only chunks scoped to the
  same workspace. **Owner input (does not change the code fix):** confirm every deploy's `S3_VECTORS_BUCKET`
  carries the placeholder (see §6).

**4.5 F006 — delete the legacy PRD-125 workflow router (preferred), or scope it.** The review's preferred
path — and CLAUDE.md §5 (delete what you supersede) — is to **delete the PRD-125 workflow execute router**
(`workflows.py:1059-1072` and the router it belongs to), because its engine is already a disabled stub and
missions are the canonical path (CLAUDE.md §10: Mission, not Workflow). Deleting it removes the oracle and the
cross-workspace `AgentTask` enqueue entirely, and removes dead surface. **This is the recommended path, framed
as an owner confirmation, not a decision taken here** (CLAUDE.md §12): if the owner confirms delete, remove the
router + engine stub + orphan imports/routes in the same PR. **Fallback** (if the owner wants it kept): scope
the workflow fetch and the agent selection to `ctx.workspace_id`, so a workflow and its agent must both belong
to the caller's workspace. No backward-compat shim either way (CLAUDE.md §4).

**4.6 F007 — role-gate the monitoring routers; keep the bearer only for ingest.** Wrap both monitoring
routers (the `GET /api/alerts` read surface and its sibling) in `require_super_admin` — they are the obs tier
PRD-143 defines (memory: `prd143-rev2-obs-lock`). Keep the `ALERT_INGEST_TOKEN` bearer check **only** on the
AlertManager ingest `POST` (a machine-to-machine path that legitimately has no user session). Reuse the
existing `require_super_admin` dependency; do not invent a new gate.

**4.7 F039 — scope memory to the workspace.** In `api/memory.py` (`/api/v1/memory`), key reads, writes, and
deletes on `ctx.workspace_id` (composed with `session_id`, not `session_id` alone), so a caller cannot reach
another tenant's memory by reusing a `session_id`. **Owner call surfaced, not taken:** review §14 lists this
route as a possible "placebo" to fix-or-delete; whether the route survives is the owner's IA call — the
tenancy fix applies regardless while it exists (CLAUDE.md §12).

**4.8 F045 — add auth + scope the count.** In `api/context.py`, add the workspace-scoped auth dependency to
all seven endpoints (they currently have none), and scope the `SELECT COUNT(*)` to `ctx.workspace_id` so the
count reflects only the caller's workspace. No unauthenticated tenant-data surface remains on this router.

**4.9 F019 — denylist side-effecting functions in the SELECT validator.** In the NL2SQL SELECT-only validator
(re-confirm the symbol on `main` — see §3 note), add a denylist that rejects side-effecting SQL constructs
inside a nominally-`SELECT` query, so `query_to_xml('UPDATE ...')` and equivalent function-wrapped mutations
are blocked. Prefer the existing AST validator path (memory: `prd160-nl2sql-accuracy` uses a `sqlglot` AST
validator) over string matching where available. **Dependency surfaced, not taken:** the **read-only DB role
at provisioning** is the owner's policy call that sets the true blast radius (review §14) — the validator
function-denylist is the in-scope code fix; the DB role is noted as a dependency in §6.

> **Scope discipline (CLAUDE.md §12).** Wave 2 fixes tenancy on the nine named surfaces. The **unified policy
> plane** that will evaluate tenancy/role/budget/approval at one choke point is **W4** (roadmap §3), gated on
> W2 — W2 makes the surfaces carry a workspace so W4 has something correct to enforce; it does not build the
> plane. That boundary is the review's dependency order, made explicit in the roadmap, **not** a silent
> descope. If re-confirmation on `main` shows a finding's blast radius reaches a sibling route this PRD does
> not name, that route is added here — not punted.

---

## 5. Test-first acceptance

Write these **failing first**, then implement to green. The headline is the **cross-tenant matrix** (the
wave's definition of done, review §13); each finding adds a focused failing test underneath it.

**5.0 The cross-tenant matrix (headline acceptance).** A single parametrized test asserts that **workspace A
cannot read, write, or delete any of workspace B's data** across every in-scope domain — **skills,
documents/vectors, workflows, memory, and context**. For each domain × operation (read / write / delete), a
request authenticated as workspace A against a resource owned by workspace B returns denied (403/404 per the
route's contract) or an empty/own-only result — never B's data and never a mutation of B's data. This is the
exact bar the roadmap names ("isolation enforced in CI") and it stays green as a permanent regression guard.

Per-finding failing tests underneath the matrix:

1. **F002.** Workspace A cannot attach or read a `SKILL.md` owned by workspace B (attach and read both denied
   / own-only). A workspace-scoped `DELETE /{skill_id}` against a **global builtin-core** skill is denied; the
   same delete as super-admin succeeds; the global skill remains active for all other workspaces after a
   denied attempt.
2. **F003.** A request from workspace A carrying a **guessed** workspace-B UUID to any of the four sync routes
   is denied by the auth dependency (no Composio bulk-op fires, workspace B's graph is untouched); the same
   sync scoped to the caller's own workspace succeeds.
3. **F004.** With `SHOPIFY_INTERNAL_API_KEY` **unset**, boot **fails** (fail-closed). With the key set, a
   request bearing an **arbitrary** `Authorization: Bearer x` to `/provision` (and `/connect`,
   `/deactivate`) is **rejected**; only the correct key is accepted.
4. **F005.** With `S3_VECTORS_ENABLED=true` and a `S3_VECTORS_BUCKET` **missing** the `{workspace_id}`
   placeholder, startup **fails**. With a valid template, `S3VectorsBackend.search()` returns **zero**
   cross-workspace hits given a shared-bucket fixture containing both workspaces' chunks, and
   `_expand_to_parent_context` hydrates only same-workspace chunks. (Test is skipped when the flag is off.)
5. **F006.** *If delete (preferred):* the legacy workflow execute route returns 404 (router removed) and no
   `AgentTask` is enqueued; a test asserts the route + engine stub are gone. *If scope (fallback):* workspace
   A executing a workflow/agent owned by workspace B is denied, and **no cross-workspace `AgentTask` is
   enqueued** under the `queued` runner.
6. **F007.** `GET /api/alerts` (and its sibling monitoring route) returns 403 for a non-super-admin caller and
   succeeds for super-admin; the AlertManager ingest `POST` still succeeds with a valid `ALERT_INGEST_TOKEN`
   and is rejected without it.
7. **F039.** Workspace A cannot read or write memory under a `session_id` owned by workspace B — a reused
   `session_id` from A returns A's data only, never B's; writes land in A's workspace scope.
8. **F045.** All seven `api/context.py` endpoints reject an unauthenticated request (401/403); the
   `SELECT COUNT(*)` returns only the caller's-workspace count given a two-workspace fixture.
9. **F019.** The SELECT-only validator **rejects** `query_to_xml('UPDATE ...')` and a representative set of
   side-effecting function wrappers, and still **accepts** a legitimate read `SELECT`.

**Wave-level bar (definition of done).** The cross-tenant matrix (5.0) is green in CI across all five domains,
and there are **zero unauthenticated endpoints touching tenant data** among the in-scope routers (F004, F007,
F045 close the unauth surfaces; F002/F003/F005/F006/F039 close the authenticated-but-unscoped ones). Both the
Tenancy/RBAC and Security enterprise bars (review §13) pass for these surfaces.

---

## 6. Risks & rollback

- **Blast radius is per-router, not the shared auth.** Every fix adds a dependency or a filter to a specific
  router; **none changes the 657-site shared hybrid auth dependency** (memory: `prd09-board-sdk-auth-unlock`).
  That is the deliberate risk boundary — the medium risk rating reflects "we are touching auth on many
  routers," not "we are rewriting auth."
- **F004 / F005 fail-closed startup can fail a deploy.** These are the two **production-unknown** dependencies
  (review §14, roadmap §6): (a) is `SHOPIFY_INTERNAL_API_KEY` set in Railway, and (b) does every deploy's
  `S3_VECTORS_BUCKET` carry the `{workspace_id}` placeholder? The code fixes (fail-closed, startup assertion)
  are **unambiguous regardless** — but if either is currently mis-configured, this PRD will (correctly) turn a
  silent leak into a **loud boot failure**. **Owner input required before deploy:** confirm both are set (§6
  is where these are surfaced, not silently assumed). Mitigation: land the assertions behind the wave's own
  branch, confirm the Railway env in the same change window, deploy together.
- **F006 delete-vs-scope is an owner decision.** The review prefers delete (dead stub engine, CLAUDE.md §5);
  the fallback is scope. This PRD **recommends delete and asks** — it does not take the decision (CLAUDE.md
  §12). Either path is a clean, independently-revertible commit.
- **F019 depth depends on an owner policy call.** The validator denylist bounds the *query* surface; the
  **read-only DB role at provisioning** bounds the *blast radius* and is the owner's call (review §14). The
  in-scope fix is correct on its own; the DB role, if adopted, is defence-in-depth layered on top later.
- **F039 fix-or-delete framing.** The tenancy fix is unconditional while the route exists; whether the route
  is later removed as a "placebo" (review §14) is an IA decision that does not block this fix.
- **Rollback.** Each finding is an independent commit; revert individually. The cross-tenant matrix test
  (5.0) must stay green as a permanent regression guard regardless of which fixes are reverted.

---

## 7. References

- Review §4 — F002/F003/F004/F005/F006/F007 (High), F039/F045/F019: `reports/PLATFORM_OS_REVIEW_2026-07-01.md`
- Review §13 Wave 2 (acceptance: cross-tenant matrix A→B read/write/delete across every domain, 100% denied)
- Review §13 pillars — Tenancy/RBAC ("cross-tenant matrix … isolation enforced in CI") and Security ("zero
  unauthenticated endpoints touching tenant data")
- Review §14 — production unknowns (`SHOPIFY_INTERNAL_API_KEY` set? `S3_VECTORS_BUCKET` placeholder?), F006
  delete-vs-scope, F019 read-only DB role, F039 fix-or-delete
- [PLATFORM-OS-ROADMAP.md](./PLATFORM-OS-ROADMAP.md) — W2 (parallel with W1); W2 gates W4 (unified policy plane)
- [PRD-171 — Execution Spine Integrity](./171-EXECUTION-SPINE-INTEGRITY.md) — the parallel Wave 1 (shares no files)
- Memory: `prd09-board-sdk-auth-unlock` (narrow scope-gated dep, not a change to shared hybrid auth),
  `prd143-rev2-obs-lock` (obs tier + `require_super_admin`), `prd160-nl2sql-accuracy` (`sqlglot` AST validator)
- CLAUDE.md §4 (no backward-compat shims), §5 (delete what you supersede — esp. F006), §10 (Mission ≠
  Workflow), §12 (no unilateral descope; owner decisions surfaced, not taken)
