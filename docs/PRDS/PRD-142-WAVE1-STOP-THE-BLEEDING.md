# PRD-142 Wave 1 — Stop the Bleeding (Durable Execution, Mem0 Stability & Honest Errors)

> **Parent:** `PRD-142-CORE-DESIGN-REVIEW.md` §12, Wave 1.
> **Status:** Build-PRD — drafted 2026-05-31. Wave 0 (#400 backend, #402 frontend) is merged to
> `main`; the "Is it working?" vitals are live on the **Command Centre**. This is the second wave's
> build spec.
> **Type:** Reliability hardening + telemetry **adoption**. **Backend-only** — no new UI (the Wave 0
> ERRORS tile lights up as a *side effect* of this work). **No new features.** Reuse-first per
> CLAUDE.md §2 / §5.
> **Verified against:** `origin/main` @ `4b623a3fa`, code reads 2026-05-31.
> **Depends on:** PRD-141 Platform Reliability (`PRD-141-PLATFORM-RELIABILITY.md`). Wave 1 **pulls in
> PRD-141 Phases 0–2 by reference** rather than re-specifying them, and adds two gaps PRD-141 never
> covers.
> **Ralph config:** `scripts/ralph/prd-142-wave1.json` (to add).

---

## 1. The founding question for Wave 1

Wave 0 answered *"can we measure it?"* — yes. The very first live read (prod, 2026-05-31) shows where
the platform is **bleeding**, and Wave 1 stops exactly that, nothing else.

**Pre-launch caveat (binding):** the ~20 workspaces are mostly test + a few pilots; we are **not
live**. Adoption/activation/volume numbers are **noise**. The only Wave-0 readings that are real
*signal* are **instrumentation-correctness gaps** and **code-path failures**. Wave 1 is scoped to
those — not to "move the activation number."

| Live Wave-0 reading | What it really means | Root cause (verified on `main`) | Wave 1 action |
|---|---|---|---|
| **ERRORS tile = empty** (0 rows ever) | The sink works; **nothing reports into it** | `record_error` persists to `error_events` (`exception_telemetry.py:123`) but has only **2 call sites** (`smart_tool_router.py:216`, `signal_recorder.py:368`) — both tool-routing. Mission/verification/planner/board/wizard/workflow failure paths `logger.error()` and never call it. | **WS-A** — adopt `record_error` at the real failure hot-paths |
| **Mission success 17%** (5/30) vs workflows 90% | Almost certainly **orphaned-on-restart**, not merit failures | Missions launch via fire-and-forget `asyncio.create_task` (`board_tasks.py:788`, `wizard.py:339`, `workflows.py:1019/1137`). A redeploy mid-run loses the task; no record distinguishes "crashed" from "never started". | **WS-C** — durable execution + boot reaper |
| **Widget `callback_failed` ×3 / 7d** | A real code-path failure on the widget callback | Uninstrumented except path; failure is invisible beyond a log line | **WS-A** — instrument the widget callback path |
| (background) **DDL/migrations stall** | Idle-in-transaction holds | `get_db()` (`database.py:105`) `finally` only `close()`s — no rollback; long-lived background `SessionLocal` ticks hold a tx open (9 hr idle SELECT on `agents` observed) | **WS-D** — idle-tx lifecycle fix |
| **Mem0 stalls under load** | THE production crash source | `Mem0Client` uses sync `requests` + `time.sleep` in `run_in_executor` (35 sites) → thread-pool starvation stalls *all* async work | **WS-B** — Mem0 async (= PRD-141 Phase 1) |

**Goal:** stop runs dying silently, and make every failure visible. After Wave 1, the success-rate
and ERRORS tiles tell the *truth* — which is the precondition for every later wave's before/after.

---

## 2. What Wave 1 **is** — and is **not**

**Is:** a curated reliability bundle —
- **PRD-141 Phase 0 *adoption*** (the hot-path `record_error` calls PRD-141 deliberately left
  "opportunistic") + the bare-`except:` CI gate.
- **PRD-141 Phase 1 (Mem0 async)** — pulled in by reference. THE crash fix.
- **The cheap, safe half of PRD-141 Phase 2** — report-to-user on limits (US-009 only).
- **Two gaps PRD-141 never covers:** durable mission/wizard execution (Mission Zero P1) and the
  idle-in-transaction leak.

**Is not:**
- A re-spec of PRD-141 — we **reference** its stories by ID; the canonical spec stays in
  `PRD-141-PLATFORM-RELIABILITY.md`.
- Any new feature, endpoint-for-its-own-sake, or **UI** (the dashboard already exists — it just
  starts showing truth).
- The ~1,941 `except Exception` mass rewrite (regression risk > value; out of scope, per PRD-141 §2.3).
- PRD-141 Phases 3–5 (unified tool pipeline, negative signals, HARNESS self-management) — those map
  to **later PRD-142 waves**.

---

## 3. Relationship to PRD-141 (the mapping)

| PRD-141 unit | Wave 1? | How |
|---|---|---|
| **Phase 0 / US-001** `record_error` util + queryable sink | **Already done** | Shipped early via PRD-142 **Wave 0** (`be08b7e6a`: persist to `error_events`). Wave 1 builds on it. |
| **Phase 0 / US-002** kill 24 bare `except:` + CI gate | **In — WS-A (W1-S2)** | Pulled verbatim; cheap and safe. |
| **Phase 0 adoption** (hot-path `record_error`) | **In — WS-A (W1-S1)** | PRD-141 left this "opportunistic"; Wave 1 makes it **deliberate** at enumerated hot-paths. |
| **Phase 1 / US-003–008** Mem0 async stability | **In — WS-B (W1-S3)** | By reference; see PRD-141 §5. The gate (US-008 load test + soak) is Wave 1's gate too. |
| **Phase 2 / US-009** report to user on limits | **In — WS-E (W1-S10)** | Cheap, user-visible; safe half of Phase 2. |
| **Phase 2 / US-010–012** dynamic caps + context-budget maths | **Deferred** | Medium-risk budget maths; not "bleeding." → later wave. |
| **Phase 3** unified tool selection pipeline (US-013–016) | **Out** | Code health, not stability → PRD-142 Wave 2/3. |
| **Phase 4** negative routing signals (US-017–019) | **Out** | Accuracy, additive → later wave. (Its drain pattern is *reused* by WS-C — see §4.) |
| **Phase 5** HARNESS self-management (US-020–026) | **Out** | High-risk, flag-gated, last → final PRD-142 wave. |
| **Gap: durable mission/wizard execution** | **In — WS-C (W1-S4–S7)** | **Not in PRD-141.** New to Wave 1. |
| **Gap: idle-in-transaction leak** | **In — WS-D (W1-S8–S9)** | **Not in PRD-141.** New to Wave 1. |

---

## 4. Reuse map (read before writing a line of code)

Everything below already exists. Wave 1 **adopts / extends / fixes** it; it does not rebuild.

| Concern | Reuse this | Verdict |
|---|---|---|
| Error sink | `ErrorEvent` (`core/models/error_event.py:23`), migration `prd142_wave0_error_events.py`, persist path `exception_telemetry.py:123`, endpoint `analytics_real.py:146` | **Reuse as-is** — Wave 1 only adds **callers**; do **not** change `record_error`'s signature or never-raises contract |
| Durable background executor | The **`queued` workflow backend** already branched in `workflows.py` (`runner.backend_name == "queued"` is the durable path; `!= "queued"` is today's fire-and-forget) | **Reuse** — route missions/board through the existing queued executor before inventing a queue |
| Batched background drain | PRD-141 **US-019** pattern: single background drain task, **one DB session per flush** (NOT a session per item, NOT bare `ensure_future`) | **Reuse the pattern** for any new background writes (reaper, telemetry flush) |
| Safe session lifecycle | `get_db_session()` context manager (`database.py:114`) — commits on success, rolls back on error, closes in `finally` | **Reuse** — make background services adopt it; fix `get_db()` to match |
| Limit/budget events | existing chat/coordinator event emit path (`consumers/chatbot/service.py`, `services/coordinator_service.py`) | **Extend** — emit `limit_reached` / `BUDGET_WARNING` (PRD-141 US-009) |
| Mem0 async spec | PRD-141 §5 (US-003–008) — fully specified | **Reuse the spec** — implement as written |

**Canonical-term note:** the live tables are still `Workflow`/`WorkflowExecution`; the table
migration to **Mission** is Wave 3 (out of scope here). Wave 1 reads/writes them only where the
existing union already does. Any **user-facing** string stays **Mission**.

---

## 5. Definition of Done (the whole wave)

- [ ] A forced failure in **each** instrumented subsystem (mission, verification, planner, board,
      wizard, workflow, widget-callback, memory) produces an `error_events` row with the correct
      `subsystem`; the live ERRORS tile reflects it. **Zero** silent failure paths on the hot path.
- [ ] **24 bare `except:`** → 0, enforced by a CI gate (PRD-141 US-002).
- [ ] A redeploy **mid-mission** results in the mission being **resumed** or **cleanly marked failed**
      (`reason="orphaned_on_restart"`, `record_error(subsystem="mission")`) — **never** silently stuck
      in `running`. After the boot reaper runs, **no** orphaned `running`/`in_progress` rows older
      than the reaper threshold remain.
- [ ] **No** fire-and-forget `asyncio.create_task` launches mission/board/workflow work without a
      durable record (the bare launches are **deleted** in the same PR that adds the durable path).
- [ ] `pg_stat_activity` shows **no idle-in-transaction > 60 s**; a migration/DDL during normal
      operation does not block on the `agents` table.
- [ ] Mem0: 50 concurrent searches complete with **no thread-pool starvation** (PRD-141 US-008);
      24 h canary soak shows no error-rate regression.
- [ ] Agents that hit an iteration/budget wall emit a **user-visible** `limit_reached` / `BUDGET_WARNING`
      event instead of stopping silently.
- [ ] Every story: `pytest` green, **type checks pass**, no `os.getenv()` outside `config.py`, no
      hardcoded values, no backward-compat shims (delete what you replace in the same PR).

---

## 6. Workstreams & user stories

Story IDs are wave-local (`W1-Sn`) to avoid colliding with the PRD-141 `US-###` they reference.

### WS-A — Honest errors (telemetry adoption)
*The one real Wave-0 signal: the sink works, nothing reports into it.*

**W1-S1 — Adopt `record_error()` at the failure hot-paths.**
- Add a `record_error(subsystem=…, operation=…, error=…, workspace_id=…)` call inside the existing
  `except` blocks that today only `logger.error()`:
  - mission dispatch — `modules/coordination/dispatcher.py` (`subsystem="mission"`)
  - verification — `modules/coordination/verification.py` (`subsystem="verification"`)
  - planning — `modules/coordination/planner.py` (`subsystem="planner"`)
  - board task execution — `api/board_tasks.py:766` (`subsystem="board"`)
  - wizard scrape/Mission-Zero pipeline — `api/wizard.py` (`subsystem="wizard"`)
  - workflow execution error handler — `api/workflows.py` (`subsystem="workflow"`)
  - widget callback — the path emitting `callback_failed` (`subsystem="widget"`)
- **Constraint:** additive only — do not change control flow, do not swallow/raise differently,
  preserve `record_error`'s never-raises contract. Exact line numbers confirmed at implementation
  time (files unchanged by Wave 0).
- **AC:** induced failure in each subsystem → exactly one `error_events` row with the right
  `subsystem`; the ERRORS-by-subsystem endpoint returns it; original exception behaviour unchanged;
  `pytest` green.

**W1-S2 — Replace all 24 bare `except:` + add CI gate** *(= PRD-141 US-002, verbatim).*
- Widen every bare `except:` to `except Exception:` across `orchestrator/` (pure widening, no logic
  change). Add `scripts/ci/check-no-bare-except.sh` failing on any remaining bare `except:`.
- **AC:** grep gate returns zero; existing tests green; type checks pass.

### WS-B — Mem0 async stability (THE crash fix)
**W1-S3 — Implement PRD-141 Phase 1 (US-003–008) as specified** in `PRD-141-PLATFORM-RELIABILITY.md`
§5: `Mem0Client` → `httpx.AsyncClient`; per-workspace circuit breakers; drop `run_in_executor`
wrappers in `UnifiedMemoryService`; proactive health probe; tighten timeouts/cooldown via
`config.py`.
- **AC (= PRD-141 US-008 gate):** `grep "import requests" orchestrator/modules/memory/` → 0;
  `grep "run_in_executor.*mem0"` → 0; 50-concurrent load test passes with no thread starvation;
  24 h canary soak clean; code-reviewer agent on the Phase-1 diff.
- **Deletions:** `import requests`, module-level `_breaker`, all Mem0 `run_in_executor` wrappers,
  `time.sleep` in memory retry.

### WS-C — Durable mission / wizard / workflow execution
*The gap PRD-141 misses, and the likely cause of the 17% mission success.*

**W1-S4 — Persist a durable launch record + status transition *before* dispatch.**
- For missions, board tasks (`board_tasks.py:788`), and the wizard scrape pipeline (`wizard.py:339`):
  write the run/launch row and set state → `running` (or `queued`) **before** any background task is
  created, so a row always exists the instant work starts and "crashed mid-run" is distinguishable
  from "never started."
- **AC:** a row in `running` exists synchronously at dispatch; tests assert the transition precedes
  the background launch.

**W1-S5 — Replace fire-and-forget launches with the durable executor.**
- Route mission/board/workflow background work through the **existing `queued` backend**
  (`workflows.py` `backend_name == "queued"`) / the PRD-141 US-019 single-drain pattern, instead of
  bare `asyncio.create_task(...)` (`board_tasks.py:788`, `wizard.py:339`, `workflows.py:1019/1137`).
- **Delete** the bare `create_task` launch in the same PR (no dual path — CLAUDE.md §5).
- **AC:** no mission/board/workflow path launches via bare `asyncio.create_task`; work survives a
  process restart (resumed or recoverable); tests green.

**W1-S6 — Boot reaper for orphaned runs.**
- On startup, scan for runs left in `running`/`in_progress`/`awaiting_*` with no live executor.
  Resume if resumable; otherwise mark `failed` with `reason="orphaned_on_restart"` and call
  `record_error(subsystem="mission", operation="boot_reap")`. Runs as a guarded startup task (not
  fire-and-forget — wrap with failure capture).
- **AC:** simulate orphaned rows + restart → each is resumed or failed-with-reason; **no** `running`
  row older than the reaper threshold survives; the success-rate tile reflects the corrected state.

**W1-S7 — Make startup tasks observable.**
- `main.py:397` (`_embed_all_agents_on_startup`) and `main.py:416`
  (`_ensure_field_memory_collection`) are fire-and-forget at boot — wrap each so failure is captured
  and `record_error(subsystem="startup")` fires instead of failing silently.
- **AC:** induced startup-task failure → an `error_events` row; boot still proceeds; tests green.

### WS-D — Idle-in-transaction lifecycle fix
*The gap PRD-141 misses; the 9 hr idle SELECT on `agents` that blocks DDL.*

**W1-S8 — Fix `get_db()` to not linger idle-in-transaction.**
- `get_db()` (`database.py:105`) `finally` currently only `close()`s. Add a `rollback()` (or commit
  on the success path) before close so a request that opened a transaction never leaves the
  connection idle-in-transaction. Align with the safe `get_db_session()` pattern (`database.py:114`).
- **AC:** a read-only request handler leaves **no** idle-in-transaction connection after return;
  existing endpoint tests green; no behavioural change for handlers that already commit.

**W1-S9 — Background services commit/close per tick + `pool_pre_ping`.**
- Audit long-lived `SessionLocal()` holders (`services/harness_service.py`,
  `modules/coordination/reconciler.py`, heartbeat) so each tick uses `get_db_session()` or commits
  and releases per loop — never one session held across a multi-phase tick. Enable
  `pool_pre_ping=True` on the engine (`database.py:83`).
- **AC:** under a normal background tick, `pg_stat_activity` shows no idle-in-transaction > 60 s; a
  DDL/migration during operation does not block on `agents`; reproduce-then-verify documented.

### WS-E — Report to user on limits *(safe half of PRD-141 Phase 2)*
**W1-S10 — Emit `limit_reached` / `BUDGET_WARNING`** *(= PRD-141 US-009).*
- `consumers/chatbot/service.py`: on `iteration >= max_iterations`, emit a user-visible
  `limit_reached` event. `services/coordinator_service.py`: on budget exceeded, emit `BUDGET_WARNING`
  before pausing.
- **AC:** iteration-limit and budget-event tests green; no silent stop on the chat/coordinator paths.

---

## 7. Sequencing & gates

Land in this order — each is independently shippable:

1. **WS-A** (telemetry adoption) **first** — so every fix after it is *observable* on the live tiles.
2. **WS-B** (Mem0 async) — THE crash fix; gated by the PRD-141 US-008 load test + 24 h soak.
3. **WS-C** (durable execution) — the success-rate fix; **canary soak** (deploy, force a redeploy
   mid-mission, confirm resume/clean-fail).
4. **WS-D** (idle-tx) — verify via `pg_stat_activity` + a live DDL.
5. **WS-E** (limits) — small, lands any time after WS-A.

**Every story:** `pytest` green + type checks pass. **Every workstream:** code-reviewer agent on the
diff; CRITICAL/HIGH addressed before merge. **Risky workstreams (B, C):** canary soak on one
workspace, rollback path documented.

---

## 8. Deletions (delete what you replace — CLAUDE.md §5)

- Bare `asyncio.create_task(...)` launches for mission/board/workflow work (replaced by the durable
  executor in WS-C).
- 24 bare `except:` blocks (WS-A / PRD-141 US-002).
- Mem0 `import requests`, module-level `_breaker`, `run_in_executor` wrappers, `time.sleep` retry
  (WS-B / PRD-141 §5).

---

## 9. Out of scope

- PRD-141 Phases 3 (unified tool pipeline), 4 (negative signals), 5 (HARNESS self-management) →
  later PRD-142 waves.
- PRD-141 US-010–012 (dynamic caps + context-budget maths) → later wave.
- The ~1,941 `except Exception` mass rewrite (opportunistic adoption only).
- US-006 per-primitive health tile (deferred to **Wave 3** — no honest data source yet).
- `WorkflowExecution` → `Mission` table migration (Wave 3).
- Any new UI, route, page, LLM provider, or feature.

---

## 10. Success metrics

| Metric | Current (live, 2026-05-31) | Target | How measured |
|---|---|---|---|
| `error_events` rows after induced subsystem failures | 0 | ≥1 per instrumented subsystem | WS-A tests + ERRORS endpoint |
| Bare `except:` blocks | 24 | 0 | CI gate (W1-S2) |
| Orphaned `running` runs after a deploy | unknown (≈ the 17% gap) | 0 older than reaper threshold | W1-S6 reaper + DB check |
| Mission success rate **truthfulness** | misleading (orphans counted as failures) | reflects merit (orphans resumed or labelled) | success-rate tile post-WS-C |
| Mem0 thread-starvation under 50 concurrent | crashes | 0 | PRD-141 US-008 load test |
| Idle-in-transaction > 60 s | present (9 hr observed) | 0 | `pg_stat_activity` |
| Silent agent bails | unknown | measurable (`limit_reached` count) | W1-S10 event |
| Widget `callback_failed` | ×3 / 7 d, uninstrumented | instrumented + root-caused | WS-A widget telemetry |

---

## 11. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Durable-execution refactor (WS-C) touches the live dispatch hot path | Medium | **Reuse** the existing `queued` backend rather than inventing a queue; canary soak; delete fire-and-forget only after the durable path is proven |
| `get_db()` rollback change (WS-D) is global to every request | Medium | rollback-then-close is the standard safe pattern; covered by endpoint tests; no change for handlers that already commit |
| Mem0 async (WS-B) is a large diff | Medium | Already fully specified in PRD-141 §5; ships behind the US-008 load-test + soak gate |
| Telemetry adoption (WS-A) accidentally changes control flow | Low | Additive-only constraint; `record_error` never raises; reviewer confirms each is a pure addition |

---

**End of PRD-142 Wave 1 (build spec).**
