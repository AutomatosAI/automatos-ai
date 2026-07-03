# PRD-179: Wave 9 — Planning Intelligence Completion

**Phase:** C — Moat Compounding (weeks 16–24)
**Branch:** `feat/w9-planning-completion` · **Worktree:** `automatos-ai-prd179`
**Dependencies:** Wave 1 (F001 spine fix) + Wave 4 (policy plane, for HARNESS actuation) — **both merged to main (`5768c2d5b`)**
**Build size:** M · **Risk:** Low–Medium
**OS Review refs:** §5, §12.6, roadmap Phase C

---

## Overview

PRD-164 (planning-intelligence-seams) is **fully merged** (PR #457, all four stories). This wave is **not** an unbuilt PRD — it lands the verified remainders that never merged and repairs what shipped broken: the heartbeat/planning memory read-half, the starved mission-synthesis flywheel, HARNESS prescriptions that escalate instead of actuate, and `rag_feedback` that writes without reading back. All four are small, named diffs verified absent from the #457 diff.

---

## Ownership boundary (parallel-safe)

Runs concurrently with W7 (PRD-177) and W8 (PRD-178).

- **W9 OWNS:** `modules/context/modes.py` **heartbeat (~76–88) and planning (~141–149)** regions, `modules/context/service.py` (context assembly), `services/coordinator_service.py:~1136-1152`, `services/harness_service.py:~1679`, `api/board_tasks.py` (dispatch of prescriptions), `core/services/approval_policy.py` (read-only reuse), `api/rag_feedback.py`, `modules/rag/service.py`.
- **W9 MUST NOT TOUCH:** `platform_executor.py` (**W8 owns** — W8's F020 fix removes the F021 shadowing symptom; W9's F021 is only the `modes.py` read-half), `graph_router.py` / `edge_builder.py` / `telemetry.py` (**W7 owns** — for F070 use the **ranking path in `rag/service.py`**, not affinity edges), `modes.py` lines ~40–46 (**W7 owns** the chain-hints gate).

`modes.py` is edited by both W7 (~40–46) and W9 (~76–88, ~141–149) in **different regions** — git auto-merges. Grep to confirm regions before editing.

---

## Findings & Scope

| Finding | Issue (verified) | Fix |
|---|---|---|
| **F021** | Heartbeat agents are memory-blind by design (`modes.py:~76-88`); planning mode has no field/memory section (`modes.py:~141-149`). The Q60 memory read-half never merged in #457 | Add a workspace-scoped field/durable-memory digest to the **HEARTBEAT** and **PLANNING** context modes |
| **F049** | Mission-synthesis flywheel ingests 3 arbitrary `COMPLETED` runs — no `ORDER BY`, no exclusion of already-ingested runs → starves once >3 accumulate (`coordinator_service.py:~1136-1152`) | Add `ORDER BY created_at DESC`, SQL-side already-ingested exclusion (marker), and failure markers |
| **F048** | Human-approved HARNESS prescriptions escalate admins to a surface that returns HTTP 409 — they never actuate. Flag is off by default (`config.py:~583`) | Route approved prescriptions as **board tasks through the Wave-4 policy plane / ask verdict**; enable behind the flag |
| **F070** | `rag_feedback` stores signals that feed nothing back into ranking — write-only bucket (`api/rag_feedback.py:~50-70`) | Feed feedback into retrieval ranking so marked-unhelpful docs de-rank |

---

## Stories (test-first)

### S1 · Memory digest in heartbeat + planning modes (F021 read-half) — S/M
**Files:** `modules/context/modes.py` (HEARTBEAT ~76-88 and PLANNING ~141-149), `modules/context/service.py` (assembly, ~55-120).
**Test (this is the W8+W9 integration gate):** `test_planning_reads_field` asserts a completed mission's field distillation appears in the **next code-touching mission's planning pack**. `test_heartbeat_memory_read` asserts a heartbeat-mode agent's context includes the workspace-scoped memory digest.
**Notes:** Add a budgeted, workspace-scoped field/durable-memory digest section alongside documents + knowledge graph. Reuse the existing digest builder used elsewhere — do not invent a second one. Thread `workspace_id` so only scoped entries appear. This reads what W8 promotes to durable; both can build in parallel and meet at merge.

### S2 · Mission-synthesis flywheel ordering + dedup (F049) — S
**Files:** `services/coordinator_service.py:~1136-1152`.
**Test:** `test_flywheel_dedup_and_order` creates 10 completed missions, runs the ingest sweep, asserts the correct most-recent set is ingested (`ORDER BY created_at DESC`), and a second run ingests only new missions (SQL-side already-ingested exclusion works).
**Notes:** Prefer a marker/`last_ingested_at` column check or `NOT IN (SELECT ... FROM <ingested marker>)` on the SQL side — do not pull all rows and filter in Python. Record failure markers so a failed ingest doesn't silently re-loop.

### S3 · HARNESS prescription actuation via the policy plane (F048) — M
**Files:** `services/harness_service.py:~1679`, `api/board_tasks.py`, `core/services/approval_policy.py` (reuse the Wave-4 plane — do not build a parallel approval path).
**Test:** `test_harness_prescription_actuates` approves a prescription, asserts it becomes a board task routed through the policy plane's ask verdict, dispatches, executes, and completes (`status=done`, result ≠ null) — no HTTP 409 escalation.
**Notes:** This is **governed activation**, not an auto-apply. Approved prescription → board task → same ask verdict every other governed action passes through (Wave 4). Enable behind the existing config flag (via `config.py`).

### S4 · rag_feedback as a ranking feature (F070) — M
**Files:** `api/rag_feedback.py`, `modules/rag/service.py`.
**Test:** `test_rag_feedback_to_ranking` marks a retrieved doc unhelpful via `POST /rag/feedback`, runs a follow-up retrieval on the same query, asserts the marked doc de-ranks (lower score or absent from top-K).
**Notes:** Integrate the feedback signal into `RAGService` retrieval ranking on the **live hot path** (not offline eval). **Use the ranking path, not tool-affinity edges** — affinity edges live in `edge_builder.py`, which W7 owns; keep W9 off that file.

---

## Acceptance & Verification

**Acceptance (W9 gate):** `test_planning_reads_field` — a completed mission's field distillation appears in the next code-touching mission's planning pack (spans W8's promotion + W9's digest; W9 owns the assertion).

**Verification (no servers/Docker/browser):** `py_compile` every changed file + **pure pytest** with DB/RAG mocked at the boundary; seed `workspaces` rows in any DB-touching test (known FK trap). CI is the integration gate.

```
python -m py_compile <changed files>
python -m pytest orchestrator/services -k "flywheel or harness" -q
python -m pytest orchestrator/modules/context -k "planning or heartbeat" -q
python -m pytest orchestrator/... <rag feedback tests> -q
```

---

## Conventions (see automatos-ai/CLAUDE.md)
- No `os.getenv()` outside `config.py`; reuse the Wave-4 policy plane and existing digest builder — no parallel implementations.
- No backward-compat shims; delete what you replace. Immutable patterns; small functions; full error handling. SQLAlchemy 2.0 raw-SQL: use `CAST(:p AS type)` not `:p::type`.
- Commit to `feat/w9-planning-completion`. **Do not push or open a PR** — stop after local verify and report.

## Success metrics
- Heartbeat + planning contexts include the workspace-scoped memory digest.
- Flywheel ingests in order and deduplicates — no starvation past 3 missions.
- Approved HARNESS prescriptions actuate through the policy plane (no 409).
- RAG feedback de-ranks unhelpful docs on the live retrieval path.
