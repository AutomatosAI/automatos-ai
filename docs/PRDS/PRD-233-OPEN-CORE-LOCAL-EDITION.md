# PRD-233: Open-Core Local Edition — Basic becomes worth downloading

> **Status:** BUILT 2026-08-29/30 on `ralph/prd-233-local-edition` (stacked on PRD-209's branch; S4 on its own stacked branch `ralph/prd-233-s4-storage-factory` per Q3). Local test pass green (see §Build log). Awaiting push → CI → owner merge. Grounded @ `origin/main 182cd6739` (2026-08-29); `file:line` refs may drift — confirm by grep at build. **Program:** Open-Core **Phase 1** (see `PRD-WAVE-OPEN-CORE.md`; owner decisions locked 2026-08-29: single codebase FINAL, enterprise PARKED, Basic + SaaS co-exist). **Depends on:** PRD-209 (Phase 0 — fresh-clone boot + compose local defaults; **must merge first**, shares `docker-compose.yml`). Reuses PRD-175 (`AUTH_EDITION`), PRD-176 (MinIO seam), PRD-197 S5 (pgvector-local RAG, landed `9cfbf9005`). **Absorbs PRD-151** (storage-client consolidation — S4 here; PRD-151 header updated).

---

## Framing (CLAUDE.md §3)

**Extension + consolidation — no new subsystem.** Every story extends something that already runs in SaaS (workspace-worker, tool router, seeds, boto3 call sites) or consolidates a duplication PRD-151 already enumerated. Net-new surface: one tool-routing degrade seam and one docs page. **Build size:** M overall (S4 is the only mechanical sweep). **Risk:** Low-Medium — the tool seam touches the router (fail-open history, see S2); everything else is additive or deletion.

## Overview

After PRD-209, a stranger can boot the platform. This PRD makes the booted thing **worth keeping**: agents that can act on the user's own machine (the thing SaaS structurally cannot offer), a tool surface that degrades honestly instead of dying silently without a Composio platform key, a seeded first-run that demonstrates the product in two minutes, storage that works fully against MinIO, and the intake scaffolding that points community energy at skills/tools where it never conflicts with core. North-Star: this is the contributor-acquisition wave — the local edition is the funnel into everything else.

**Non-goals (owner decisions, recorded — not deferrals):** no capability gating in Basic (paid tiers gate organizational features and hosting, never product capability); no `ee/` directory or license keys (enterprise PARKED 2026-08-29); teams-connecting-local-workspaces-via-SaaS = future funnel, out of this PRD by owner call; **session mode (subscription runtime) = PRD-234**, not here.

---

## Current reality (grounded @ `origin/main 182cd6739`)

- **Composio is woven with no seam and no fallback**: ~164 backend files; `orchestrator/core/composio/` (client, tool_executor, tool_router_manager, entity_manager), `orchestrator/modules/tools/composio_tool_router.py`, `orchestrator/modules/tools/tool_router.py`, ~18 API routers; config keys at `orchestrator/config.py:516-526, 984-998`. Without `COMPOSIO_API_KEY` the integration surface is dead and nothing tells the user why. Router history: F018 fail-open (`modules/tools/tool_router.py` ~960-996) — this codebase's disease class is *fail-open-wider*; the degrade seam must fail **honest**, not open.
- **workspace-worker exists, containerized, off by default locally**: `services/workspace-worker/` (executor, workspace_manager, canvas_session_service, canvas_confinement, canvas_approvals) — the Code Canvas runtime; compose gates it behind `profiles: ["workers"]`.
- **Storage is half-seamed**: the RAG/document leg honors `S3_ENDPOINT_URL` (PRD-176 F089), but PRD-151 §2's enumerated **12+ independent `boto3.client()` sites** remain (e.g. `orchestrator/api/documents.py:296`, `api/document_generation.py:389`, `api/recipe_executor.py:795`, `services/checkpoint_service.py:26`, `services/tool_manifest_service.py:26`, `services/workspace_purge.py:29` — re-verify list at build; #625 may have deleted the voice entries), plus **three bespoke local fallbacks** PRD-151 marked for deletion: `core/services/marketplace_s3.py:268`, `core/services/image_store.py:205`, `modules/attachments/store.py:82`.
- **Seeds exist** (`orchestrator/core/seeds/`, entrypoint wait→migrate→seed step from PRD-176) but there is no curated local first-run: no demo agents/playbook bound to the `DEFAULT_WORKSPACE_ID` workspace.
- **Community intake is generic**: `CONTRIBUTING.md` exists; nothing states the commercial-shipping deal, nothing routes capability contributions to skills/tools, no DCO check.

---

## Stories (test-first; CI is the only gate — no local runs, no servers)

### S1 · Workspace-worker joins the local profile — agents act on the user's machine — S/M
**Files:** `docker-compose.yml` (worker moves into the default/local profile **after PRD-209 S7 merges** — sequence-sensitive shared file); `services/workspace-worker/worker_config.py` + `canvas_confinement.py` (a host-access dial: confinement root defaults to a designated host directory bind-mounted into the worker, e.g. `${AUTOMATOS_WORKSPACE_DIR:-./workspaces}`); `envs/api.defaults` (worker URL default).
**Test:** worker config unit tests for the confinement-root resolution (designated-dir default; escape attempts rejected — extend existing `canvas_confinement` tests); a compose source-guard asserts the worker service is in the local profile and the bind-mount matches the config default.
**Notes:** This ships the *existing* runtime to the laptop — no new execution paths. Scope of host access is **Q1** (designated directory vs wider). The approval layer (`canvas_approvals.py`) stays on — local ≠ unguarded.

### S2 · Tool-routing degrade seam — honest without Composio — M
**Files:** `orchestrator/core/composio/client.py` (one availability predicate: key present + client constructible — config-read only, no `os.getenv` outside `config.py`); `orchestrator/modules/tools/tool_router.py` + `modules/tools/composio_tool_router.py` (routing consults availability ONCE, at a single seam: absent ⇒ Composio tools are **not offered** — excluded from discovery/schemas — rather than offered-then-erroring); frontend tools surface shows an "integrations disabled — no Composio key" state, not an empty lie.
**Test:** router tests: with key configured, Composio tools present (today's behaviour, unchanged); with key absent, (a) discovery excludes Composio tools, (b) native platform tools + agent-native tools still route, (c) a direct Composio call returns an explicit `integrations_unavailable` error — **never** a silent success or a fail-open pass-through. Regression: the F018 gate stays fail-closed.
**Notes:** Degrade order in Basic: platform-native tools → (PRD-234's session tools when that lands) → BYO `COMPOSIO_API_KEY` if the user sets one (document it, **Q2**). This story is the D2 seam from the program doc; it deliberately does NOT abstract Composio behind a provider interface — one predicate, one exclusion point, honest UI.

### S3 · First-run seed — the two-minute demo — S
**Files:** `orchestrator/core/seeds/` (an idempotent local-edition seed: the `DEFAULT_WORKSPACE_ID` workspace + a small named agent roster + one demo Playbook + a welcome Deliverable, gated on `AUTH_EDITION=local`; runs in the existing entrypoint seed step — no new lifecycle); reuse `seed_auto_agent.py` patterns (note its skip-existing behaviour at `:110` — the local seed must be idempotent-refresh, not skip-forever).
**Test:** seed unit test against a fresh schema: run twice ⇒ identical state (idempotent); assert workspace id == config `DEFAULT_WORKSPACE_ID`; assert the roster/playbook rows exist and are workspace-scoped. The PRD-209 S5 smoke probe may additionally assert the seed ran (coordinate, don't duplicate).
**Notes:** Content stays minimal and useful (a real Playbook a stranger can run with only their LLM key). No file-hacks: seed content lives in the seed module, DB is the runtime source (CLAUDE.md §4).

### S4 · One storage-client factory — absorbs PRD-151 — M
**Files:** one factory in `orchestrator/core/` honoring `S3_ENDPOINT_URL` + per-bucket config (PRD-151's design: "S3's API *is* the interface — no StorageProvider abstraction"); migrate the enumerated `boto3.client()` sites to it; **DELETE the three bespoke local fallbacks** (`marketplace_s3.py` local path, `image_store.py` local path, `modules/attachments/store.py` local path) in the same PR — MinIO is the local answer, one code path both editions.
**Test:** factory unit tests (endpoint/credential resolution: MinIO defaults locally, AWS in SaaS); a source guard asserts zero direct `boto3.client(` calls outside the factory module (today: 12+); existing attachment/marketplace/image tests keep passing against the factory.
**Notes:** Mechanical but wide — **Q3** asks whether this rides this wave or lands as its own PR lane inside it (same PRD either way; no scope loss). Re-verify PRD-151's site list at build (#625 removed voice).

### S5 · Self-host docs + community intake — S
**Files:** `docs/getting-started/self-hosting.md` (the real guide: 3 secrets + 1 LLM key, what runs where, MinIO/pgvector notes, worker host-access dial, BYO-Composio, troubleshooting — supersedes the scattered `docs/deployment-infrastructure/` compose pages, which get pointers, not duplicates); `CONTRIBUTING.md` (+2 paragraphs: the commercial-shipping deal — contributions may ship in hosted/commercial editions under Apache-2.0 §5 — and "where to contribute capability": skills/tools/MCP first, core second); `.github/workflows/` DCO check (probot/dco or equivalent lane).
**Test:** doc guard: every `${VAR:?}` in compose appears in the self-hosting guide (same shape as PRD-209 S8's guard — share the checker); DCO lane proves itself on the PR (sign-offs present).
**Notes:** Keep the tone factual — what things do, no pitch. The **PRD-210 history scrub remains the promotion gate**: docs can merge, but no launch push until 210 runs (program-level gate, owner-scheduled).

### S6 · Local profile — Auto knows who it is talking to — S
**Context (owner ask, 2026-08-29 live test):** "no login on local also means I can't add a profile, means Auto doesn't know me — make it feel more real: *Hello Gerard*." The local edition has exactly one operator; the anonymous session currently carries no identity, so every greeting, attribution, and `created_by` is faceless.
**Files:** the local-edition seed (S3) creates ONE `users` row for the operator (well-known id alongside `DEFAULT_WORKSPACE_ID`, e.g. `DEFAULT_LOCAL_USER_ID` in `envs/api.defaults`; `name`/`email` start as placeholders); `core/auth/hybrid.py` anonymous lane binds `UserContext(id=<that row>, name, email, system_role="super_admin")` in local mode (today it carries only the role); **Settings → Profile** in local mode becomes a plain form (name, role/title, email) writing that row — replacing the Clerk-managed profile surface that has nothing to show locally; the Auto context pack / greeting reads `ctx.user.name` (it already renders names for SaaS users — reuse, do not add a second greeting path).
**Test:** hybrid unit test: local anonymous ctx carries the seeded user id + name; profile API round-trip test in local mode (PUT name → GET reflects; saas untouched); context-section test asserts the operator's name appears in Auto's greeting/context when set and falls back gracefully when blank.
**Backbone already shipped by PRD-209 (live-test finding 2026-08-29: chat 500'd "No users found"):** the entrypoint seeds `users` id 1 (`LOCAL_OPERATOR_EMAIL`, config + `envs/api.defaults`), and `hybrid.py`'s anonymous local lane carries that email + `super_admin`. S6's remaining scope = the Settings → Profile form (name/role/email writing that row), the greeting/context threading, and attribution polish.
**Notes:** No accounts, no passwords, no login — a profile, not an identity system. Seed is idempotent-refresh only on placeholder values (never overwrite a name the operator typed). This also fixes the faceless `created_by`/attribution on locally created agents, Playbooks and Deliverables.

---

## Sequencing

- **After PRD-209 merges** (shared `docker-compose.yml`; the boot must be true before any of this matters).
- S1 ∥ S2 ∥ S4 are file-disjoint and parallel-safe; S3 after S1 (worker profile defines what the demo can show); S6 with S3 (same seed + hybrid lane); S5 last (documents reality).
- **Delivered early by PRD-209's live-test fixes (2026-08-29):** the marketplace catalog seeders (agents v2, Shopify agents, packages, personas, plugin categories) now run idempotently at every boot via `core/database/load_seed_data`, and the local session is the instance super-admin. S3's remaining scope is the curated first-run *content* (demo Playbook, welcome Deliverable, the prod-curated catalog refresh) and the Settings→Credentials nudge — not the seeding plumbing.

## Verification (CI only)

No local servers, no `docker compose` runs on the dev machine. Every behavioural story carries pure unit tests (confinement resolution, router degrade, seed idempotency, factory resolution) + source guards (compose profile, zero stray `boto3.client(`, doc-var coverage, DCO lane presence). The de-masked PRD-209 smoke lane is the end-to-end proof surface for anything boot-visible.

## Success metrics

- With `COMPOSIO_API_KEY` unset: tool discovery excludes Composio, native tools work, the UI says why. Today: silent dead surface.
- Worker runs in the local profile against a designated host directory with approvals on. Today: off by default, container-confined only.
- Fresh local boot lands in a seeded workspace with a runnable demo Playbook. Today: empty.
- Zero `boto3.client(` outside the factory; the three bespoke local fallbacks deleted. Today: 12+ and 3.
- A stranger can go clone → running → first Playbook executed with only the documented steps. Today: impossible.

## Decisions recorded 2026-08-29 (owner, during the live local test)

- **Marketplace = two things.** The seeded **local catalog** (agents, packages, playbooks, personas) is the product's content library — offline, no auth, stays. The **community hub** (share/pull other people's work) is a network service: **pulling needs no account and nothing is ever paywalled** (n8n community nodes / HACS / Grafana catalog model — "more people sharing makes the platform stronger"); **publishing needs a free Automatos account, and pushes are monitored and approved** (moderation + a verified tier). Subscription stays where it is: hosted workspaces + teams. Hub v1 form (GitHub-repo-backed registry first vs SaaS-hosted) = still open (Q5). Portable package manifest export/import = the primitive both need.
- **SaaS-only surfaces hidden in local by an explicit allowlist**, not by role (the local session is super_admin): Workspace Admin, Team, plan/trial pills, invitations, sign-in/up. (→ S7)
- **Composio in the local edition = bring your own key in `.env`** (`COMPOSIO_API_KEY`, env-only — `config.py`; compose passthrough added by PRD-209). Nuance for S2/S3: the catalog seeders bind agents to Composio apps via `composio_apps_cache`; on a keyless first boot those bindings are skipped ("Tool 'SLACK' not found"), and the by-name/by-slug idempotency means they are **not** re-bound when a key is added later — S2 must re-resolve seeded agents' app bindings once the cache populates (or bind lazily by intended slug). Also: nothing syncs the catalogue on boot — the only triggers are `POST /api/tools/sync` (Tools page **Sync**, `workspace:manage`) and two manual scripts (`scripts/run_tools_sync.py`, `scripts/sync_composio_metadata.py`); S2 acceptance = a boot with a key and an empty `composio_apps_cache` runs the sync itself, then re-binds the seeded agents.

### S7 · Edition-aware navigation — SaaS-only surfaces hidden in local — S
**Files:** the nav/route registry (sidebar + `app/` routes for Team, Workspace Admin, billing/plan, invitations, sign-in/up) gated on `isSaaS` via one explicit list in `lib/auth-edition.ts` (no per-page ad-hoc checks); local Settings shows Profile (S6) + Credentials + system settings only.
**Test:** a component test renders the nav in local mode and asserts none of the listed surfaces appear; saas mode unchanged.
**Notes:** role-based hiding is wrong here — the local operator is super_admin by design.

## Open questions — Gerard's call (§12)

1. **Host-access default scope (S1).** Designated directory (`./workspaces` bind-mount; recommended — safe default, one dial to widen) vs home-directory access out of the box. Confirm.
2. **BYO-Composio placement (S2/S5).** Documented-but-secondary (recommended) vs promoted as the primary local tool path?
3. **S4 lane (storage sweep).** Ride the main wave PR or its own PR inside the wave (wide mechanical diff; separate PR eases review)? Recommendation: separate PR, same wave.
4. **Local MCP surface (S2 notes).** v1 = user-configured MCP servers documented for the session runtime (PRD-234) only, or does the platform tool router also learn MCP discovery here? Recommendation: defer MCP-in-router to PRD-234's session lane — one seam at a time. Confirm (this is a scope call, not a silent deferral).

---

*Traceability: program doc `PRD-WAVE-OPEN-CORE.md` (owner decisions 2026-08-29); `reports/dossiers/deployability-open-core.md` (EXTEND verdict); `reports/dossiers/thesis-T2-repo-topology.md` (monolith-in-compose is the self-host shape); PRD-151 §2 (boto3 census — absorbed here as S4); PRD-153 §5/§10 (program framing — Phase-0 pieces absorbed by PRD-209 S7-S9); coupling census from the 2026-08-29 open-core sweep (Composio 164 files; Clerk 83/30 — untouched here; identity work remains optional hygiene, not a Basic blocker).*
5. **Hub v1 form.** GitHub-repo-backed registry (`automatos-packages`, PRs as the publish path — zero infra, matches the skills-repo doctrine; recommended) vs SaaS-hosted registry with accounts first. Browsing on by default with a visible switch + env var (recommended) vs opt-in.

## Build log (2026-08-29/30)

All seven stories built and proven on the live local stack: S1 worker in the default profile against `./workspaces`; S2 availability seam + boot catalogue sync + honest Tools card; S3 first-run seed (Auto, roster, 'Two-minute brief', welcome Deliverable) idempotent-refresh; S4 storage factory (own PR — 12 SaaS-reaching behaviour changes enumerated for the owner); S5 self-hosting guide + DCO lane + every README/doc page; S6 local profile ("I'm talking to you, <name>!" proven through the frontend proxy); S7 edition-aware nav + unrestricted local exposure profile. Findings fixed on the way: fresh databases were missing `v_workspace_outputs` (orphan-root ordering — generator gained model-column + idempotent raw-SQL passes; CI Gate 2 asserts it); every local upload had ended `status: failed` behind an "uploaded successfully" message (embeddings never saw the operator's key — `PLATFORM_KEY_WORKSPACE_ID` now points at the local workspace; message honest); the seed loader's `sys.path` hack, the dead skills seeder and the v2 marketplace seeder's id churn; a developer's `AWS_*` hijacking MinIO; uvicorn dev reloads wedging on open SSE streams; the postgres data volume keeping its initdb password across recreates.

**Owner decisions still open after the build:** Q1 host-access default = designated directory (built as recommended); Q2 BYO-Composio documented-but-secondary (built as recommended); Q3 S4 = separate PR (done); Q4 MCP-in-router deferred to PRD-234 (as recommended — confirm); Q5 hub v1 form; the `role/title` profile field (no column; add via a later migration or drop); `TASK_RUNNER_BACKEND=queued` locally (needed for clone-a-repo; diverts workflow execution to the worker — Railway's value unknown); uploaded originals stay in the container's `/tmp` (pre-existing upload-path behaviour; MinIO holds only pipeline outputs) — durability gap for the local edition.
