# Automatos — Development Guardrails

> The enforceable checklist every change to Automatos must pass. Derived from the
> invariants in `BRAIN-BLUEPRINT.md §6`, the project's `CLAUDE.md`, and divergences found
> in the 2026-05-29 architecture review. If a PR violates a **MUST**, it does not merge.
>
> **Status:** baseline — 2026-05-29. These are rules, not suggestions.

---

## How to read this

Each guardrail has a **rule**, a **why**, and a **check** (how to verify, ideally automatable).
`MUST` = blocking. `SHOULD` = strong default, deviation requires a note in the PR.

---

## A. Architecture invariants (the non-negotiables)

### A1 — One cognitive loop, many front doors `MUST`
- **Rule:** chat, channels, widgets, and missions all flow through the same router → runtime →
  tool-loop path. Do not build a parallel reasoning pipeline for a new surface.
- **Why:** divergent loops drift in dedup/retry/truncation behaviour (we already have two —
  `BRAIN §8 G6`). Every fork doubles the test surface and the bug surface.
- **Check:** new entry points construct a `RequestEnvelope`/`RequestContext` and call the existing
  router/runtime; they don't re-implement tool dispatch.

### A2 — One tool executor, prefix-dispatched `MUST`
- **Rule:** all tools execute through `UnifiedToolExecutor.execute_tool()`. New tool classes are new
  prefixes/branches, never a new executor.
- **Why:** the executor is the single most reused contract in the system; forking it fragments
  permissions, telemetry, and error handling.
- **Check:** grep for new tool-execution entry points; there should be none.

### A3 — DB is the system of record `MUST`
- **Rule:** durable truth lives in Postgres. In-memory dicts/caches (`active_agents`,
  `_running_executions`) are caches — the system must reconstruct correct state from the DB after a
  restart.
- **Why:** `BRAIN §8 G4` — playbook executions die on restart because state lived only in a process
  dict. This is how user work silently vanishes.
- **Check:** "if the process restarts mid-operation, does it recover from the DB?" must be answerable
  "yes" for any user-visible operation.

### A4 — workspace_id is carried, never guessed `MUST`
- **Rule:** `workspace_id` originates in `RequestContext` and is passed explicitly down the stack. No
  layer below L1 fabricates or defaults it.
- **Why:** the historical cross-tenant leak. A fabricated/defaulted workspace_id is a data-breach
  primitive.
- **Check:** no new `DEFAULT_WORKSPACE_ID`/`WORKSPACE_ID` fallbacks (`BRAIN §8 G16` is the one to
  kill, not copy). Cached runtimes take workspace_id per-execution.

### A5 — Deliverables are markdown in S3; consumers render `MUST`
- **Rule:** agent output is written as `.md` to S3. The API returns the file/reference; rendering is
  the consumer's job.
- **Why:** keeps output portable and the backend ignorant of presentation. Rendering in the API
  couples content to one UI.
- **Check:** no HTML/markdown-to-X transformers in the API layer for deliverables.

### A6 — Core is vertical-agnostic; verticals are plugins `MUST`
- **Rule:** the cognitive core (L2) and generic surfaces know nothing about Shopify or any vertical.
  Verticals live in `integrations/<vertical>/` and register in `PLUGIN_REGISTRY`.
- **Why:** the "can run any business" promise and the moat strategy both depend on it.
- **Check:** CI gate `scripts/ci/check-no-shopify-in-generic.sh` (extend it to new generic surfaces).
  New vertical = new folder + registry entry + zero core edits.

---

## B. Reuse & cleanliness (this is a mature codebase, not greenfield)

### B1 — Reuse before build `MUST`
- **Rule:** before writing a new component/hook/table/endpoint/tool, search the graph
  (`graphify-out/`), the codebase, and memory. The burden is on you to justify why the existing one
  isn't enough.
- **Why:** 116 tables and 103 routers exist because "new" kept winning over "reuse."
- **Check:** PR description names what was searched and why new code was necessary.

### B2 — Delete what you replace `MUST`
- **Rule:** when a PR replaces a surface, the old one is deleted in the same PR. No `_legacy` suffix
  that lives forever, no dual paths "just in case."
- **Why:** the dead `chatbot_router` ("kept for backward compatibility") and deprecated
  `_stream_workflow_bridge` are exactly this debt.
- **Check:** no orphaned old path; callers migrated; imports cleaned. Exception: high-risk prod-data
  migration may keep both *temporarily* with a documented sunset date.

### B3 — No backward-compat shims `MUST`
- **Rule:** fix the pattern cleanly; don't add a compatibility layer to avoid touching callers.
- **Why:** project standing rule (`feedback-no-backward-compat`). Shims become permanent.

### B4 — Many small files over few large ones `SHOULD`
- **Rule:** 200–400 lines typical, 800 max. `coordinator_service.py` (3021) and
  `agent_factory.py` (1507) and `api-client.ts` (2687) are over budget — don't add to them, extract.
- **Why:** cohesion, testability, review-ability.

---

## C. Canonical terms (drift costs the user)

### C1 — Use the canonical noun `MUST` (new code) / `SHOULD` (cleanup)
| Use | Never |
|---|---|
| **Playbook** | ~~Recipe~~ |
| **Mission** | ~~Workflow~~, ~~Job~~ |
| **Task** (`BoardTask`; mission sub-tasks `OrchestrationTask`) | — |
| **Deliverable** | ~~Output~~, ~~Artifact~~, ~~Workspace file~~ |
| **Knowledge Graph** | ~~Business Graph~~ |
| **Command Center** | ~~Activity~~ |
| **Auto** (proper noun) | "the assistant" |

- **Why:** `BRAIN §8 G9` — ~1,682 "recipe" references make the Playbook rename skin-deep. New code
  must not add to the debt; touched files should be migrated opportunistically.
- **Check:** a `canonical-term-checker` pass on the diff before PR.

---

## D. Config & secrets

### D1 — config.py is the only place env vars are read `MUST`
- **Rule:** no `os.getenv()`/`os.environ` outside `orchestrator/config.py`. Everything reads from the
  `Config` object.
- **Why:** `BRAIN §8 G7` — ~20 violations across 8 files (worst: the whole `core/monitoring/`
  module). Scattered env reads make config unauditable and break in deploy.
- **Check:** CI grep gate for `os.getenv`/`os.environ` outside `config.py` (and `alembic/env.py`,
  which also needs fixing).

### D2 — No hardcoded values `MUST`
- **Rule:** tunables go through `config.py` or `system_settings`. No magic constants, no hardcoded
  IDs (`users.id=1` in the widget path, `BRAIN §8 G14`, is the anti-pattern).
- **Why:** `feedback-no-hardcoded-values`.

### D3 — No file hacks for DB data `MUST`
- **Rule:** personas, agent definitions, configs live in the database, not loaded from files at
  runtime.
- **Why:** `feedback-no-file-hacks-for-db-data`.

### D4 — Secrets never logged, never in args `MUST`
- **Rule:** credentials decrypted at point of use, never logged; tokens in env/secret store, not
  command-line args.
- **Check:** secret-scan on diff; NL2SQL creds and Composio keys never appear in logs.

---

## E. Reliability

### E1 — No fire-and-forget for user-visible work `MUST`
- **Rule:** anything a user is waiting on (mission, playbook, long task) must be DB-backed and
  recoverable on restart. `asyncio.create_task` for user work without durable state is banned.
- **Why:** `BRAIN §8 G4`. Missions got this right (DB tick); playbooks did not.
- **Check:** startup recovery scans for stuck `running`/`pending` rows and resumes or fails them.

### E2 — No bare `except Exception: pass/continue` `MUST`
- **Rule:** catch specifically; on a broad catch, record via the error-telemetry path (PRD-141
  Phase 0 `record_error()`) and decide explicitly to degrade or fail. Never silently swallow.
- **Why:** `BRAIN §8 G8` — masked failures are unobservable failures.
- **Check:** CI gate counting bare excepts (PRD-141 sets this up); count must not increase.

### E3 — Fail closed on authz `MUST`
- **Rule:** permission/authorization checks return **deny** on error or unknown, never allow.
- **Why:** `BRAIN §8 G3` — `_check_agent_permission()` and `validate_composio_action()` currently
  "fail open for now." That's a latent privilege-escalation path.
- **Check:** every authz function's error/else branch denies.

### E4 — Retries must learn `SHOULD`
- **Rule:** when a task is retried, feed the failure/verifier critique into the next attempt's prompt.
- **Why:** `BRAIN §8 G5` — blind re-queue repeats the same failure.

---

## F. Data & migrations

### F1 — Sessions commit/rollback and close `MUST`
- **Rule:** `get_db()` and every session path must commit or roll back, then close. Never hold a
  transaction open across an `await` that does I/O.
- **Why:** `BRAIN §8 G1` — the 9hr idle-in-transaction leak that blocks DDL.
- **Check:** connection-leak test in CI (none today).

### F2 — Migrations are online-safe `MUST`
- **Rule:** set `lock_timeout`/`statement_timeout`; avoid long table locks; no blanket
  NOT-NULL/`ALTER` on large tables without a backfill plan. Per-migration transactions, not one giant
  wrap.
- **Why:** `BRAIN §8 G2` — current `alembic/env.py` wraps everything in one transaction with no
  timeouts, so the idle-in-tx leak hard-blocks deploys.
- **Check:** `migration-reviewer` pass on any `alembic/versions/` change.

### F3 — One source of truth per concern `MUST`
- **Rule:** don't store the same fact in two places with two write paths (the dual L3 memory write,
  `BRAIN §8 G12`; two graph stores, `G11`). Pick the canonical store; others derive.
- **Why:** divergent copies are silent-corruption generators.

---

## G. Tools, agents, frontend

### G1 — Tools use the 3-file registration pattern `MUST`
- **Rule:** new platform tools register via the canonical 3-file pattern (handler + registry entry +
  schema). Don't bypass `ActionRegistry`.
- **Why:** it's the proven extension point; bypassing it breaks discovery and permissions.

### G2 — No duplicate hooks `MUST`
- **Rule:** if `useDeliverables` exists, don't add `useDeliverablesV2`. Refactor the existing hook.
  (Versioned *API* surfaces like `use-memory-v1-api` are fine — that's a real API version, not a dup.)
- **Why:** `feedback-no-backward-compat` applied to the frontend.

### G3 — Frontend calls go through api-client `MUST`
- **Rule:** all backend calls route through `frontend/lib/api-client.ts` (carries Clerk JWT +
  `X-Workspace-ID`). No ad-hoc `fetch` to the backend.
- **Why:** tenancy header + auth consistency.

### G4 — Don't kill valued surfaces `MUST`
- **Rule:** Command Center, Activity, kanban board, analytics, and widgets are kept. A
  consolidation/refactor may rehouse them but must not remove them.
- **Why:** `feedback-dont-kill-valued-surfaces`.

---

## H. Definition of Done (every primitive must meet this)

Before a primitive (chat/memory/RAG/NL2SQL/graph/missions/playbooks/channels) is called "rock
solid," it satisfies its `BRAIN §3` contract **and**:

- [ ] **Golden-journey test** exists and passes (the happy path end-to-end) — see `TEST-PLAN.md`.
- [ ] **Failure path tested** — the primitive degrades or errors visibly, never silently.
- [ ] **Restart-safe** — no user-visible work lost on process restart (E1).
- [ ] **Observable** — emits the telemetry/metrics needed for the "Is it working?" dashboard.
- [ ] **Tenant-isolated** — proven by a cross-workspace test (A4).
- [ ] **One source of truth** — no dual write paths (F3).
- [ ] **Dashboard tile** — a number that answers "is this primitive working right now?"

---

## I. Process

### I1 — Plan before code for non-trivial PRDs `MUST`
Map the existing surface → identify reuse candidates → surface ambiguities → propose the plan. Never
go "request → code."

### I2 — Ask before assuming `MUST`
If "build X" looks like it already exists, or a name is ambiguous, or a config dial would do — ask.
30 seconds beats half a day.

### I3 — Update memory after learning `SHOULD`
New architectural patterns, bug root-causes, corrected mistakes, and project decisions go to the
auto-memory index. Don't save what `git log`/grep already know.

### I4 — Never push to main without explicit approval `MUST`
Main is the deploy branch. One branch per session; bundle session work.
