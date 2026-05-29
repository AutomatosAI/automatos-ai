# Playbook Engine — Consolidation Design

> The design spec for the **consolidate-and-harden** of the execution engine in
> `PRD-142-CORE-DESIGN-REVIEW.md` §6: unify the scattered-but-live recipe/playbook execution path
> into one clean DB-backed flow, and make it restart-durable by borrowing the Mission coordinator's
> proven model. **Not a rewrite** — the engine exists and works; it is scattered and not restart-safe.
>
> **Status:** Design proposal — docs phase, no code until build is green-lit.
> **Author:** Design review session 2026-05-29 (Gerard Kavanagh).
> **Decision it implements:** PRD-142 §6 ("approved as a consolidate-and-harden, as design").
> **Depends on:** PRD-142 §10 P0s (G1 sessions, G2 migrations) landed first — this engine adds a
> table and a migration, and must not deploy behind the idle-in-tx leak.
> **Verified against:** branch `feat/widget-page-context-on-regular-chat`, code reads 2026-05-29.

---

## 1. Why this is consolidate-and-harden, not a rewrite

"Recipe" and "playbook" are the **same thing** — recipe is the pre-rename name; the front end now
says playbook. The execution logic is **live and working**, just scattered: the loop lives in
`recipe_executor.py`, the launch front door in `workflow_recipes.py` (despite its name), with dead
`modules/workflows/` remnants alongside. The *contract is right*. Two things are wrong, and both are
fixable **without inventing a new engine**:

1. **Scattered, not unified.** One concept ("playbook", née recipe) is spread across files that
   diverge in dedup, retry, and streaming behaviour. The fix is **consolidate to one clean flow +
   one execution-state table** — and delete the dead `modules/workflows/` duplication.
2. **Not restart-durable (the real defect).** The launch path uses `asyncio.create_task(...)` and
   lives only in the process. Restart mid-run → the user's playbook silently dies
   (`GUARDRAILS.md` E1/G4). The fix is **HARDEN**: port the durability model in §3.

Missions already solved the durability problem and scored grade **A−** for it (PRD-82A, the
Sequential Coordinator). This is **not a rewrite** — we are not replacing a wrong contract or
building a new engine. We are **unifying scattered-but-live code and porting a proven durability
pattern** onto it. (Gerard, 2026-05-29: "is it just scattered and we need to clean up to one clean
flow?" — yes.)

---

## 2. Current state (code-verified)

### 2.1 Three engines

| Engine | File | ~Lines | Role today |
|---|---|---|---|
| Recipe executor | `orchestrator/api/recipe_executor.py` | 1,997 | Primary execution loop; `asyncio.create_task(_safe_execute())` at **:903**. |
| Workflow recipes API | `orchestrator/api/workflow_recipes.py` | 1,872 | `launch_recipe_task()` at **:905** — the API front door. |
| Workflows module | `orchestrator/modules/workflows/` + `consumers/workflows/streaming.py` | rest of ~5,400 | Step models + in-memory SSE manager. |

### 2.2 Two persistence tables, no recovery

- **`recipe_executions`** (`core.py:1468–1517`) — the richer one. Columns:
  `status` (pending/running/completed/failed), `current_stage`, `current_step`, `step_results`,
  `error_message`, `attempt_count`, `retry_of`.
- **`workflow_executions`** (`core.py:523–546`) — the execution log. **Decided 2026-05-29 (Gerard):
  migrate and drop — "workflows don't exist."** 4 writer sites (`chat.py:92`, `composio.py:899`,
  `workflows.py:983`/`:1092`) and ~20 readers (analytics, monitoring, execution-history, agent-
  performance, nl2sql) **fold onto `playbook_executions`** — live writers repoint, dead-path writers
  delete with their files, history backfills. One execution table. Columns: `status`, `execution_log`,
  `error_message`.
- **Restart recovery: NONE.** No startup scan for `running`/`pending` rows. A crash mid-execution
  orphans the row in `running` forever.

### 2.3 Streaming is in-memory only

`consumers/workflows/streaming.py:20–80` — `WorkflowStreamManager` holds per-execution event queues
**in memory**, with replay. No DB backlog → on restart, in-flight SSE streams are lost and cannot
resume. **Note:** this manager is **kept** — it is the live chat-streaming manager (`api/chat.py:65`
imports `get_stream_manager`), not workflow-only dead code. We *harden* it (persist events), we do
**not** delete it.

### 2.4 The launch surface is SIX call sites (graph-verified, then code-confirmed)

The graph showed ~36 inbound edges from ~21 files, but most are *symbol* coupling (type imports,
shared utils like `_composio_scope_message`). The **execution-launch** surface — the only thing the
migration must repoint — is **six call sites**:

| # | Caller | Site | Path |
|---|---|---|---|
| 1 | Workflow API endpoint | `api/workflow_recipes.py:905` | `launch_recipe_task()` |
| 2 | Composio trigger | `api/composio.py:886` | `execute_recipe_direct()` |
| 3 | Workspace webhook | `api/webhooks.py:682–683` | `create_task(execute_recipe_direct())` |
| 4 | Platform tool executor | `modules/tools/discovery/handlers_playbooks.py:487` | `launch_recipe_task()` |
| 5 | Scheduler (cron playbooks) | `services/playbook_scheduler.py:208` | `launch_recipe_task()` |
| 6 | Task reconciler (retry) | `services/task_reconciler.py:273` | `launch_recipe_task()` |

Confirmed **non-callers** (they only import shared utils, not execution): `channels/__init__.py`,
`core/action_registry.py`. The execution blast radius is therefore **bounded to two entry points**
(`launch_recipe_task` and `execute_recipe_direct`) behind which all six callers sit.

---

## 3. The durability pattern to borrow (Mission coordinator)

This is the proven model (`coordinator_service.py`, grade A−). The playbook engine adopts its shape.

| Mechanism | Where (mission) | What it buys |
|---|---|---|
| Scheduled tick | `coordinator_service.py:843–850` — APScheduler `add_job(self.tick, "interval", seconds=Config.COORDINATOR_TICK_INTERVAL_SECONDS)` (default 5s) | Progress is driven by a loop that re-reads the DB, not by an in-process task. |
| Re-read state each tick | tick method `coordinator_service.py:867–944` — reads all `RUNNING` runs from DB | After a restart, the next tick picks up in-flight work automatically — **the durability primitive**. |
| Stall detection / re-dispatch | `modules/coordination/reconciler.py:687–759` — ASSIGNED>60s → STALLED, RUNNING>300s → STALLED → back to ASSIGNED | A row left `running` by a crash is detected and re-dispatched, not orphaned. |
| Durable state columns | `orchestration_runs` (`orchestration.py:39–157`), `orchestration_tasks` (`:159–292`): `state` enum, `version_id` (optimistic lock), `started_at`/`completed_at`/`updated_at`, `attempt_number`/`max_retries`, `failure_reason_code` | State lives in Postgres; the process holds none of it. |

**Key insight:** mission recovery is *continuous via the tick*, not a one-shot startup call. The
playbook engine inherits the same property, and adds one improvement (below).

---

## 4. Target design

### 4.1 Structural choice (the open decision)

Three ways to "reuse the pattern." **Decided 2026-05-29 (Gerard): C-lite.** A and B stay below so the
trade-off is on record.

| Option | Shape | Pro | Con |
|---|---|---|---|
| **A — Playbook *is* a Mission** | Run a playbook as a linear-DAG mission on the existing coordinator. | Zero new durability code; literally one loop. | Forces playbook semantics into mission semantics (approval gates, verification, board tasks, agent-decomposition) that a deterministic step-DAG doesn't want. Overloads the A− subsystem. |
| **B — Separate engine, copied pattern** | New `PlaybookEngine` with its own tick, table, state machine — same *shape* as the coordinator. | Clean semantic separation. | A second durable loop to maintain; risk of the two drifting (the exact failure we're consolidating away from). |
| **C-lite — Shared durability substrate (DECIDED)** | Extract the proven primitive (DB-state + scheduled tick + stall detection + recovery) into a small shared `DurableRunner`; missions keep their semantics on top, playbooks get a thin step-executor on top. | One durability implementation, two semantic layers; honors "reuse over build" without conflating concepts. | Modest refactor of the coordinator to sit on the shared base — **mitigated by sequencing: the Wave-2 test net lands before this Wave-3 refactor, so the A− coordinator is covered before we touch it.** |

C-lite keeps a playbook a *deterministic step DAG* (distinct from a mission's adaptive,
agent-decomposed plan) while guaranteeing both recover identically.

### 4.2 One table replaces two

A single canonical **`playbook_executions`** table (resolving the recipe/workflow naming debt, G9),
seeded from the union of today's two tables and the coordinator's durability columns:

| Column | Source | Purpose |
|---|---|---|
| `id`, `playbook_id`, `workspace_id` | both | identity + tenancy (workspace_id carried, never defaulted — `GUARDRAILS.md` A4) |
| `state` (enum: PENDING/RUNNING/COMPLETED/FAILED/STALLED) | from `recipe_executions.status` + mission state-machine | durable status |
| `current_step`, `step_results` | `recipe_executions` | step progress (verbatim, for SSE rebuild) |
| `version_id` | mission `orchestration_*` | optimistic lock against double-dispatch |
| `attempt_number`, `max_retries`, `retry_of` | `recipe_executions.attempt_count`/`retry_of` + mission | retry accounting |
| `failure_reason_code`, `error_message` | mission + `recipe_executions` | typed failure for retry-with-learning |
| `started_at`, `completed_at`, `updated_at`, `heartbeat_at` | mission | stall detection + recovery |

Both of today's tables fold into `playbook_executions`. `recipe_executions` (the live, richer
execution-state table) **evolves into** it — matching the recipe→playbook rename — and
`workflow_executions` is **migrated and dropped** (decided 2026-05-29, Gerard: "workflows don't
exist"). Live writers repoint to the new table; the ~20 analytics/history/nl2sql readers (§2.2)
move onto `playbook_executions`; dead-path writers delete with their files. **One** execution
table, one source of truth per `GUARDRAILS.md` F3 — no separate analytics log survives.

### 4.3 The interface (the stable seam the six callers use)

```
PlaybookEngine.launch(playbook_id, workspace_id, inputs, *, trigger) -> execution_id   # enqueues a DB row, returns immediately
PlaybookEngine.status(execution_id) -> PlaybookExecutionState                          # reads DB
PlaybookEngine.stream(execution_id) -> AsyncIterator[StepEvent]                         # rebuild-from-DB then tail live
PlaybookEngine.cancel(execution_id) -> None                                            # state -> FAILED(cancelled)
```

`launch()` **does not** `create_task` the work — it inserts a `PENDING` row and returns. The shared
`DurableRunner` tick advances it. This is the whole fix: the process no longer owns the execution.

All six call sites collapse onto `launch()` (the two webhook/composio direct paths included).

### 4.4 Streaming becomes restart-safe

**Harden** (not replace) the in-memory `WorkflowStreamManager` — it stays as the live chat-streaming
manager (`chat.py:65`). Back it with **step events persisted to the execution row** (append to
`step_results` / a child events table). `stream()` reconnect = rebuild from DB state, then tail live
updates the tick writes. An SSE client that drops during a restart **resumes** instead of losing the
stream. The in-memory queue becomes a cache over the durable log, not the source of truth.

### 4.5 Retry that learns (closes G5/E4)

The mission side already carries `failure_reason_code`. The new engine **feeds the prior attempt's
failure into the next attempt's prompt** rather than blind re-queue. The data exists today
(`attempt_count`, `retry_of`, `error_message`); the engine must *use* it.

### 4.6 One improvement over the mission model

Missions recover *only* via the continuous tick. The playbook engine adds an **explicit startup
reconcile**: on boot, scan `playbook_executions` for `RUNNING`/`PENDING` and re-enqueue them
immediately (don't wait up to one tick + one stall window). Cheap, and it tightens recovery latency
for user-visible work. (Worth back-porting to missions later — out of scope here.)

---

## 5. Migration & deletion plan (strangler-fig)

No dual paths left alive (`GUARDRAILS.md` B2/B3). Order:

1. **Build** the shared `DurableRunner` + `PlaybookEngine` behind the §4.3 interface, with the new
   `playbook_executions` table (online-safe migration, `lock_timeout` set — `GUARDRAILS.md` F2).
2. **Backfill** today's execution rows into `playbook_executions` (one-time data migration):
   in-flight `recipe_executions` rows plus `workflow_executions` history, so the ~20 analytics/
   history/nl2sql readers keep continuity when they repoint. Low-risk — live state is
   small/ephemeral and the log is append-only.
3. **Repoint** the six call sites (§2.4) to `PlaybookEngine.launch()`, one at a time, each behind a
   code-reviewer gate. The two entry-point functions (`launch_recipe_task`, `execute_recipe_direct`)
   become thin shims that call `launch()`, then are inlined and removed.
4. **Prove parity** — the restart-durability test (§6) passes; SSE resume works; retry feeds the
   critique.
5. **Delete the dead duplication and both legacy tables:** `recipe_executor.py` +
   `workflow_recipes.py` execution loops (consolidated into the one engine) and the dead
   `modules/workflows/` execution code. Drop **both** `recipe_executions` and `workflow_executions`
   once `playbook_executions` is proven and every reader/writer is repointed (decided 2026-05-29).
   **Keep** `WorkflowStreamManager` (live chat consumer, now DB-backed per §4.4). Remove orphan
   imports.

This is Wave **3R** in PRD-142 §12 (inside primitive hardening; gated on the P0s and the test net).

---

## 6. Definition of Done

Per `GUARDRAILS.md` §H, plus the consolidation bar:

- [ ] **One engine.** `recipe_executor.py` + `workflow_recipes.py` execution loops consolidated into one flow; dead `modules/workflows/` execution code deleted; grep for the duplicate loops returns zero.
- [ ] **One execution table.** `recipe_executions` and `workflow_executions` both fold into `playbook_executions` (sole store for execution state + history); the ~20 analytics/history/nl2sql readers and all live writers repointed; both legacy tables dropped.
- [ ] **Restart-durable (the headline test).** Launch a multi-step playbook, kill the process mid-step, restart — it resumes and completes from the DB. No orphaned `running` rows.
- [ ] **SSE resumes** across a restart (rebuild-from-DB + tail).
- [ ] **Retry learns** — a forced step failure feeds `failure_reason_code` into the next attempt's prompt (not blind re-queue).
- [ ] **Tenant-isolated** — workspace_id carried end-to-end; cross-workspace test passes.
- [ ] **Six callers migrated**, both legacy entry-point functions removed.
- [ ] **Dashboard tile** — playbook success rate + stuck-execution count (feeds PRD-142 Wave 0 dashboard).

---

## 7. Open questions (for discussion before the build PRD)

1. **Structural option — RESOLVED 2026-05-29 (Gerard): C-lite** (shared `DurableRunner`). It touches
   the A− mission coordinator, so the refactor is gated on Wave-2 test coverage landing first.
2. **Tick cadence & isolation** — share the coordinator's 5s scheduler job, or a separate playbook
   tick? (Shared scheduler, separate handler is the leaning.)
3. **Events storage** — append to `step_results` JSON vs a dedicated `playbook_execution_events`
   child table for the SSE log. Child table scales better; JSON is simpler.
4. **Scheduler/webhook trigger semantics** — `playbook_scheduler` and the workspace webhook both
   launch; confirm they want the same at-least-once / dedup behaviour the mission dispatcher uses.
5. **`workflow_executions` disposition — RESOLVED 2026-05-29 (Gerard): migrate and drop.**
   "Workflows don't exist." The 4 live writers (`chat.py:92`, `composio.py:899`, `workflows.py:983`/
   `:1092`) repoint to `playbook_executions`; the ~20 readers (analytics, monitoring, execution-
   history, agent-performance, nl2sql) move with them; dead-path writers delete with their files. No
   separate analytics log survives.

---

**This is design only.** No code, no migration, no deletion until the build PRD is approved
(PRD-142 §6, Gerard 2026-05-29).
