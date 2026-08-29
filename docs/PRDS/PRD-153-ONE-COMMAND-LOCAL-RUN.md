# PRD-153: One-Command Local Run — Compose Consolidation & Schema Lifecycle

**Status:** Superseded — its compose-consolidation slice (delete the six `infrastructure/docker-compose*.yml`) was absorbed and executed by **PRD-209 S9 (US-008, 2026-08-29)**; the local-defaults consumer by PRD-209 S7. This doc's G1/M2 compose-deletion goals are now met on the fresh-clone-boot branch.
**Author:** Gerard Kavanagh (with Auto)
**Date:** 2026-06-09
**Type:** Refactor / Consolidation (one canonical compose; one schema lifecycle; delete six drifted compose files)
**Related:** PRD-150 (Auth — supplies no-login boot), PRD-151 (Storage — supplies MinIO), PRD-152 (mem0/internal services — supplies memory + local-safe defaults). This PRD is the **capstone**: it owns the compose those PRDs plug into, and the program-level headline metric M1.

---

## 1. Introduction / Overview

The original Automatos vision: **core = frontend, api, db, redis — clone and run.** The container topology never abandoned that (the root `docker-compose.yml` default profile is exactly those four services). What rotted is everything around it, because **production runs on Railway, which consumes per-service Dockerfiles and never touches compose** — so the compose path has had no consumer, no CI, and no one to notice it breaking.

It is broken today. Code-verified 2026-06-09:

1. **Fresh clone cannot boot.** The root compose mounts `./orchestrator/database/init_complete_schema.sql` into postgres initdb — **that path does not exist** (the file lives at `orchestrator/core/database/`). Docker silently mounts an empty directory; a clean `docker compose up --build` produces a database with no schema, a seed loader that warns-and-continues, and an app pointed at a broken DB.
2. **Three competing schema mechanisms.** `init_complete_schema.sql` (header: *"Generated: October 2025"* — 8 months stale), `orchestrator/core/database/migrations/` (10 raw SQL scripts), and `orchestrator/alembic/versions/` (124 migrations, run on boot via `alembic upgrade heads` in the Dockerfile). Note **`heads` plural** — the chain has multiple heads. Known drift exists even in prod (`escalation_level`).
3. **Seven orphaned compose files** under `infrastructure/` (`core`, `data`, `landing`, `memory`, `monitoring`, `voice`, plus another `docker-compose.yml`) overlapping and disagreeing with the root file.
4. **Missing services**: no Qdrant (field memory), no mem0 (L3), no object storage (PRD-151's MinIO), no `agent-opt-worker`. `.env.example` omits `QDRANT_URL` and all `MEM0_*` vars.

One piece of **good news**, also verified: `services/workspace-worker` (and the agent-task plane generally) is **already fully local** — Redis queue + `/workspaces` volume, zero boto3 in `services/`. The "workers need replacing with local filesystem" assumption is false; they need nothing but a place in the compose.

This PRD makes **`git clone` → set 3 secrets + 1 LLM key → `docker compose up` → working platform** true, and keeps it true with a CI smoke job that boots the compose on every PR.

---

## 2. Goals

- **G1** — **One** canonical `docker-compose.yml` at repo root; the six `infrastructure/docker-compose.*.yml` files are folded in (as profiles) or deleted. No second source of compose truth.
- **G2** — **One** schema lifecycle: empty database → `alembic upgrade head` (single head) → idempotent seed. `init_complete_schema.sql` and the raw-SQL migrations directory are retired.
- **G3** — Default profile boots a **complete working platform**: postgres(pgvector), redis, backend, frontend, workspace-worker, qdrant, minio (PRD-151), mem0 + mem0-pgvector (PRD-152).
- **G4** — Optional capability via profiles: `observability` (loki/prom/promtail/grafana), `voice`, `optimizer` (agent-opt-worker), `tools` (adminer, gotenberg).
- **G5** — `.env.example` is complete and minimal: every required var present with generation instructions; every optional var documented with its feature.
- **G6** — A **CI compose-boot smoke job** proves M1 on every PR: build, up, golden-path checks (API health, frontend render, doc upload, L3 round-trip, agent task writes a workspace file), down.
- **G7** — Railway is untouched: it keeps consuming per-service Dockerfiles; compose remains a local/OSS concern only.

---

## 3. User Stories

### Phase 0 — Stop the silent breakage (hotfix-sized, land immediately)

#### US-001: Fix the init-schema mount path
- [ ] Root compose mounts the **real** path (`orchestrator/core/database/init_complete_schema.sql`) — one-line fix so today's flow works at all while the real lifecycle (Phase 1) lands.
- [ ] Smoke-verify: clean volume, `docker compose up`, backend healthcheck green, table count > 100.

#### US-002: Layered env architecture — secrets vs committed local defaults
**Design (files created 2026-06-09):** two layers, one rule — *secrets in exactly one gitignored file; everything else committed and prefilled.*

| File | Status | Contents |
|---|---|---|
| `.env` (from `.env.example`) | gitignored, user-created | **Secrets + host ports only**: `POSTGRES_PASSWORD`, `REDIS_PASSWORD`, `API_KEY`, one LLM key; service secrets (`MEM0_API_KEY`, `MEM0_PG_PASSWORD`, MinIO root creds as `AWS_ACCESS_KEY_ID`/`SECRET`); obs-profile secrets (`GF_SECURITY_ADMIN_PASSWORD`, `LOG_RELAY_SECRET`, `ALERT_INGEST_TOKEN`) |
| `envs/api.defaults` | **committed** (exists) | backend local topology, prefilled: DB/Redis hosts, `EDITION`/`AUTH_PROVIDER` (PRD-150), MinIO endpoints (PRD-151), `MEM0_API_URL`/`QDRANT_URL`, internal services set OFF (PRD-152) |
| `envs/frontend.defaults` | **committed** (exists) | `NEXT_PUBLIC_API_URL`/`WS_URL`, `NEXT_PUBLIC_EDITION=oss`, no Clerk |
| `envs/observability.defaults` | **committed** (exists) | obs-profile ports/URLs (Prometheus/Grafana/Loki/Alertmanager/exporters) |
| `envs/*.local` | gitignored (rule added) | personal overrides, listed last in `env_file:` so they win |

Naming note: files are `*.defaults`, **not** `*.env` — the repo's `block-secrets.sh` hook (correctly) refuses agent edits to `*.env`/`*.env.*` paths; compose `env_file:` accepts any filename, so nothing is lost.

**Acceptance Criteria:**
- [ ] Compose wires `env_file:` per service: backend → `[envs/api.defaults, envs/api.local?]`; frontend → `[envs/frontend.defaults, envs/frontend.local?]`; obs-profile services → `envs/observability.defaults` (backend additionally lists it after `api.defaults` when the profile is on, so `LOKI_URL`/`PROMETHEUS_URL` override the empty defaults). Explicit `environment:` blocks shrink to secrets interpolation only.
- [ ] `.env.example` updated to the secrets-only contract: add slots for `MEM0_API_KEY`, `MEM0_PG_PASSWORD`, MinIO root creds, obs-profile secrets; move topology values out (they live in `envs/`); add a pointer to `envs/`. *(Note: `block-secrets.sh` blocks agent edits to `.env.example` — this AC is either done by hand or after the hook gains an `*.example` carve-out; decide in Q6.)*
- [ ] `docker compose config` (the rendered artifact) is reviewed in the PR to prove no var was dropped in the migration from `environment:` blocks to `env_file:`.
- [ ] Boot-time config validation lists every missing *required* var in one error (not first-failure).
- [ ] QUICKSTART documents the two layers in one paragraph: "edit `.env` for secrets; never edit `envs/*.defaults`; override via `envs/*.local`."

### Phase 1 — One schema lifecycle

#### US-003: Prove (or repair) alembic-from-zero
- [ ] CI job: empty postgres → `alembic upgrade head` → assert table parity with the SQLAlchemy metadata (compare to `Base.metadata`) and with a prod-schema snapshot.
- [ ] **Merge the multiple heads into one** (`alembic merge`); CI fails on future multi-head.
- [ ] If the 124-migration chain cannot replay from zero: **squash** to a new verified base revision (one-time, documented), then the parity test guards it forever.

#### US-004: Retire the parallel mechanisms
- [ ] Audit the 10 scripts in `orchestrator/core/database/migrations/` — each is either already-represented in alembic (delete), needed (port into an alembic revision), or dead (delete).
- [ ] Delete `init_complete_schema.sql` + the compose initdb mount; fresh boot = alembic + seed only (same path as Railway — one lifecycle everywhere).
- [ ] `docker-entrypoint.sh` updated: wait-for-postgres → `alembic upgrade head` → seed → start (migration moves from Dockerfile CMD into the entrypoint so compose and Railway share it).
- [ ] Seed (`database/load_seed_data.py`) is idempotent and *required-green* (today it warns-and-continues; a failed seed must fail the boot loudly).

### Phase 2 — One compose

#### US-005: Fold the seven into one
- [ ] `infrastructure/docker-compose.{core,data,memory,monitoring,voice}.yml` + `infrastructure/docker-compose.yml` are folded into the root file as services/profiles and **deleted**. (`landing` — see Q2.)
- [ ] Default profile = G3 list, all healthchecked, correct `depends_on` ordering.
- [ ] `workspace-worker` moves from `--profile workers` into the **default** profile (agent deliverables are core platform behavior, not an extra).
- [ ] Profiles per G4; `docker compose config` is the reviewed artifact proving no env var is silently dropped.

#### US-006: Frontend production-mode option
- [ ] Compose supports `BUILD_TARGET=production` (Next.js standalone) alongside the dev hot-reload default; QUICKSTART documents both.

#### US-007: Interim no-Clerk boot (until PRD-150 lands)
- [ ] Document the `REQUIRE_AUTH=false` anonymous path as the *temporary* local-boot mode in QUICKSTART, marked for replacement by PRD-150's `AUTH_PROVIDER=local`; compose works with zero Clerk vars set.

### Phase 3 — Prove it and keep it proven

#### US-008: Clean-machine verification
- [ ] Full run on a machine (or fresh VM/runner) with no prior state: clone → `.env` → `up` → golden checks pass; the run is recorded in the PR (timings, image sizes, peak RAM — self-hosters will ask).
- [ ] Apple Silicon + x86 both verified (workspace-worker bundles Playwright/Chromium — confirm multi-arch images build).

#### US-009: CI compose-boot smoke job (the M1 ratchet)
- [ ] Required-check CI job on every PR touching compose/Dockerfiles/alembic/config: build, up, run golden-path checks (API `/health`, frontend 200, doc upload + RAG answer [PRD-151], L3 store/recall [PRD-152], agent task → workspace file appears), down.
- [ ] Budgeted < 10 min via layer caching; flake-quarantine rules per the Wave-2 test-net conventions.

#### US-010: QUICKSTART rewrite
- [ ] `QUICKSTART.md`: prerequisites, the one command, what's running where (service map), profile catalogue, troubleshooting (ports, ARM, low-RAM), and "what is NOT in OSS" (Clerk/SaaS — honest edition framing per PRD-150).

---

## 4. Functional Requirements

- **FR-1** — Exactly one compose file defines local topology; zero files under `infrastructure/` define services that root compose also defines.
- **FR-2** — Fresh-clone boot path: empty DB → single-head alembic → idempotent seed → healthy services; no SQL-dump initialization anywhere.
- **FR-3** — Default profile yields working: chat (LLM key), document upload + RAG, L3 memory, playbook execution with step logs, agent task producing a deliverable file, field memory (qdrant).
- **FR-4** — Optional profiles add observability/voice/optimizer/tools without editing the file.
- **FR-5** — Required env surface: `POSTGRES_PASSWORD`, `REDIS_PASSWORD`, `API_KEY`, one LLM key (+ MinIO/mem0 secrets auto-instructed in `.env.example`). Nothing else mandatory.
- **FR-6** — CI smoke job is a required check; breaking local boot breaks the PR.
- **FR-7** — Railway deployment artifacts (per-service Dockerfiles, railway env) are byte-unchanged by this PRD except the shared entrypoint migration step (US-004), which is verified on a Railway preview before merge.

---

## 5. Non-Goals (Out of Scope)

- **Kubernetes / Helm / the Enterprise wrapper** — later, on top of a proven compose topology.
- **Auth provider work** — PRD-150 owns it; this PRD only hosts its outcome (and the US-007 interim).
- **Storage/mem0 code changes** — PRD-151/152 own them; this PRD hosts their containers.
- **Landing site containers** — separate product surface (Q2).
- **Performance tuning of the local stack** — defaults must *run* on a 16GB laptop, not be optimal.
- **Publishing the repo / licensing** — the open-core publication gate (secrets rotation + history audit) is tracked in PRD-150 §7; this PRD just makes the artifact worth publishing.

---

## 6. Technical Considerations

- **Why compose rotted (and the fix that sticks):** Railway never reads it, so only CI can be its consumer. US-009 is the load-bearing story of this PRD — without it, every other fix decays again within months.
- **Schema lifecycle order matters:** US-003 (prove alembic-from-zero) **before** US-004 (delete init SQL). Never delete the only working bootstrap, however stale, before its replacement is CI-green. Mirrors PRD-150's parity-gate discipline.
- **`alembic upgrade heads` plural is a live smell** — merge to one head early; it also serializes cleanly with PRD-142 Wave 5's migration work and PRD-150 US-014 (single alembic queue across the program).
- **Entrypoint as the shared boot contract:** moving `alembic upgrade head` into `docker-entrypoint.sh` gives compose and Railway one boot sequence — kills the class of "works on Railway, broken locally" drift at the root.
- **Resource budget:** default stack ≈ 10 containers (incl. mem0 pair + minio + qdrant). Target < 6GB RAM idle; measure in US-008 and publish.
- **Sequencing:** Phase 0 lands now (it's a bug fix). Phase 1 is independent of PRD-150/151/152 and can run in parallel with PRD-142 Wave 5 **except** the head-merge, which coordinates with Wave 5's migrations. Phase 2 lands after PRD-151/152 define their services. Phase 3 closes the program.

---

## 7. Success Metrics

- **M1 (program headline, shared with PRD-150/151/152):** clean machine, `git clone` → `.env` (3 secrets + 1 LLM key) → `docker compose up` → chat answers, document RAG works, memory persists across sessions, an agent task produces a deliverable. No Clerk, no AWS, no Railway, no mem0 cloud.
- **M2:** `ls infrastructure/docker-compose*.yml` → at most `landing` (per Q2); everything else deleted.
- **M3:** Schema mechanisms: 1 (alembic, single head). `init_complete_schema.sql` gone; raw-SQL migrations dir gone.
- **M4:** CI smoke job required and green; time-to-first-working-platform for a new contributor ≤ 15 minutes on a laptop.
- **M5:** Railway production deploys unaffected (next deploy after merge is the proof).

---

## 8. Open Questions

- **Q1 — Squash threshold.** If alembic-from-zero fails, do we fix individual revisions (preserves history) or squash to a base (clean but loses replay)? (Proposed: time-box repair to one session; then squash.)
- **Q2 — `docker-compose.landing.yml`.** The landing site has its own repo/deploy (seven-repo topology). Delete here, or move to the landing repo? (Proposed: move, then delete here.)
- **Q3 — Default profile weight.** Is mem0-in-default too heavy for low-end machines, or is "memory works out of the box" worth ~2 containers? (PRD-152 proposes default-on; confirm after US-008 RAM measurements.)
- **Q4 — Seed data ownership.** `load_seed_data.py` + `credential_types` sentinel — is the seed set complete for a *useful* empty workspace (personas, default playbooks, marketplace starters)? May need a "starter content" pass — possibly its own small PRD.
- **Q5 — Compose project name / volume migration.** Existing local devs have volumes under current names; renaming services (e.g. `mem0-server`) orphans data. Provide a one-shot migration note or accept reset for pre-release local stacks? (Proposed: accept reset — pre-launch.)
- **Q6 — `block-secrets.sh` carve-out.** The hook blocks agent edits to `*.env`/`*.env.*` including `.env.example`, which US-002 must update. Add a narrow allow for `*.example` (keeps all real env files blocked), or keep the hook absolute and do `.env.example` edits by hand? (Proposed: allow `*.example` — it is documentation, not a secret store; Gerard to confirm since it loosens a security hook.)

---

## 9. Phase Summary

| Phase | Stories | Character | Gate |
|---|---|---|---|
| 0 — Hotfix + env truth | US-001–002 | Bug fix, additive | land immediately |
| 1 — Schema lifecycle | US-003–004 | Repair, then **delete** 2 mechanisms | US-003 CI green before US-004 deletes |
| 2 — One compose | US-005–007 | Consolidation + **delete 6 files** | after PRD-151/152 service defs |
| 3 — Proof & ratchet | US-008–010 | Verification + required CI | closes program M1 |

---

## 10. Program view (the four open-core slices)

| PRD | Slice | Depends on | Can start |
|---|---|---|---|
| 150 | Auth → `AuthProvider` + local mode | — (Ph 2–3 after Wave 5) | Ph 0–1 now, parallel worktree |
| 151 | Storage → factory + MinIO | Wave 5 merge (touches `recipe_executor`) | after Wave 5 |
| 152 | mem0 + railway.internal sweep | Ph 1–2 ride PRD-153 compose | Ph 0 now |
| **153** | Compose + schema lifecycle | Ph 2 needs 151/152 service defs | **Ph 0–1 now** |

Suggested order alongside PRD-142 Wave 5: **[Wave 5] ∥ [150-Ph0/1 + 153-Ph0/1 + 152-Ph0]** → **151** → **152-Ph1/2 + 153-Ph2** → **150-Ph2–5** → **153-Ph3 closes M1**.
