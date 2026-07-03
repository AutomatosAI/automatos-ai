# PRD-178: Wave 8 — Field Memory Correctness and Promotion

**Phase:** C — Moat Compounding (weeks 16–24)
**Branch:** `feat/w8-field-memory-promotion` · **Worktree:** `automatos-ai-prd178`
**Dependencies:** Wave 1 (F001 spine fix) — **merged to main (`5768c2d5b`)**
**Build size:** M · **Risk:** Low–Medium
**OS Review refs:** §5 (field/codegraph correctness), §12.5, roadmap Phase C, top-risk #4 (memory-poisoning)

---

## Overview

Field memory is live on the **write** side and genuinely reaches agent turns, but three correctness defects and one missing arm block the compounding claim. Patterns **decay and hard-delete before they can promote** to durable memory — the missing moat arm. This wave fixes the binding, makes the trace inspector read-only, scopes compaction, and adds field→durable promotion **with a taint guard** (promotion is otherwise an unguarded memory-poisoning surface — top-risk #4).

---

## Ownership boundary (parallel-safe)

Runs concurrently with W7 (PRD-177) and W9 (PRD-179).

- **W8 OWNS:** `modules/tools/discovery/platform_executor.py` (the `~842-856` field-binding region), `modules/context/adapters/vector_field.py`, the field-trace API endpoint, and a **new** promotion job under `jobs/`, plus the durable-memory (mem0) write path.
- **W8 MUST NOT TOUCH:** `modules/context/modes.py` and `modules/context/service.py` context-assembly (W9 owns the planning/heartbeat digest), `graph_router.py`/`edge_builder.py`/`telemetry.py` (W7 owns).

**Note on F020/F021 overlap:** `platform_executor.py:~842-856` is the `.first()`-on-running-mission bug. It causes **both** F020 (wrong field binding, W8) **and** F021's "running mission shadows workspace recall" symptom (W9). **W8 owns this file and fixes it once.** W9's F021 is only the `modes.py` memory-read-half — W9 will not touch `platform_executor.py`.

---

## Findings & Scope

| Finding | Issue (verified) | Fix |
|---|---|---|
| **F020** | Field auto-injection binds via `.first()` on any `state=='running'` mission — no ordering, no link to the calling task | Thread `mission_id` + `run_id` from dispatch context; bind to the calling task's run |
| **F062** | The retrieval-trace inspector calls the **writing** `field.query` path, mutating the field it observes | Split a read-only trace path that inspects without mutation |
| **F063** | Field compaction sweep has no workspace scope and no resume cursor — re-scans everything, unscoped | Add `workspace_id` scoping + a persistent resume cursor |
| **Promotion** | Compaction hard-deletes patterns before they promote (`vector_field.py:~390-445`) → field never becomes durable memory | Add a promotion job: distill strong patterns to durable mem0 (provenance preserved) **before** delete, **taint-gated** |

---

## Stories (test-first)

### S1 · Correct field binding to the calling task (F020) — S
**Files:** `modules/tools/discovery/platform_executor.py:~842-856`.
**Test:** `test_field_binding_to_task` dispatches a task with `run_id=X`, `mission_id=Y` and asserts the field binds to that task (not an arbitrary running mission); a second concurrent task does not see the first's field entries.
**Notes:** The dispatch context already carries `run_id`/`mission_id` — thread them in and drop the `.first()` lookup. This also removes the F021 shadowing symptom, so W9's heartbeat digest reads a correctly-bound field.

### S2 · Read-only retrieval-trace inspector (F062) — S
**Files:** the field-trace endpoint (grep `field` under `api/` for the trace/inspect route), `modules/context/adapters/vector_field.py`.
**Test:** `test_field_trace_readonly` snapshots field state (vector count, access counters), runs the trace/inspect request, asserts state is byte-identical after.
**Notes:** Provide a `query(..., record_access=False)` (or a distinct read path) so the inspector never writes access patterns back into the field it is observing.

### S3 · Workspace-scoped compaction with resume cursor (F063) — M
**Files:** `modules/context/adapters/vector_field.py` (compaction sweep, `~405-445`); persist the cursor via an existing checkpoint table if one fits (do not add a table if one exists).
**Test:** `test_field_compaction_workspace_scope` runs compaction for workspace A and asserts B's entries are untouched. `test_field_compaction_resume` asserts a second run resumes from the cursor and does not re-scan compacted entries across a simulated restart.

### S4 · Field→durable promotion with taint guard (the moat arm) — M
**Files:** new `jobs/promote_field_memory.py` (scheduled like `nightly_edge_recompute`), `modules/context/adapters/vector_field.py`, the durable mem0 write path (reuse PRD-159 `platform_create_memory` patterns — do not fork a parallel writer).
**Test:** `test_field_promotion_to_durable` drives a pattern to high strength + access_count, runs the job, asserts it is distilled into a typed durable memory **with provenance preserved**, and a later task retrieves it from durable memory. `test_promotion_taint_guard` asserts a pattern whose provenance carries **untrusted external content** (e.g. inbound email/web) is **NOT** promoted.
**Notes:**
1. Scan field entries above strength + access thresholds (thresholds via `config.py`).
2. **Taint gate first:** never promote a trajectory tagged with untrusted external content (top-risk #4 — promotion is the poisoning surface). Provenance/data-subject tags travel with the entry.
3. Distill survivors into typed mem0 memories (preserve provenance).
4. Only then delete from field. Promotion happens **before** the hard-delete, not after.

---

## Acceptance & Verification

**Acceptance (W8 gate):** `test_field_promotion_to_durable` + `test_promotion_taint_guard` prove strong, clean patterns survive as durable memory and tainted ones do not. (The field→planning-pack integration test lives in **W9**, which owns the planning digest that reads what W8 promotes.)

**Verification (no servers/Docker/browser):** `py_compile` every changed file + **pure pytest** with Qdrant/mem0 mocked at the boundary. CI is the integration gate.

```
python -m py_compile <changed files>
python -m pytest orchestrator/modules/context -k "field" -q
python -m pytest orchestrator/... <new promotion tests> -q
```

---

## Conventions (see automatos-ai/CLAUDE.md)
- No `os.getenv()` outside `config.py`; no new tables if one fits; reuse the mem0/`platform_create_memory` path (no parallel durable writer).
- No backward-compat shims; delete what you replace. Immutable patterns; small functions; full error handling.
- Commit to `feat/w8-field-memory-promotion`. **Do not push or open a PR** — stop after local verify and report.

## Success metrics
- Field binds to the calling task's run, not an arbitrary mission.
- Trace inspector leaves field state unchanged.
- Compaction is workspace-scoped and resumable.
- Strong, **untainted** patterns promote to durable mem0 before hard-delete; tainted ones are blocked with provenance intact.
