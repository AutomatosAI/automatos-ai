# Ralph Build Prompt — PRD-186 S3 Vectors Relight (Phase-2, completes P2-03 / the S8 gate)

You are executing **PRD-186**, one story per iteration, unattended. This branch is **`ralph/prd-186-s3-vectors-relight`, cut from `origin/main`** (standalone — NOT stacked). The tip must be green after every commit.

**SCOPE — read this twice.** This Ralph run builds **only the two CODE stories** (three JSON stories: S1 + S2 = the config-integrity guard, S3 = the dimension fail-loud). The PRD also has **three OPS stories** (bucket env change, prod re-embed via `migrate_to_s3_vectors.py`, S8-probe re-run) — those are **Gerard's prod actions, NOT yours**. They need prod DB + AWS creds you must not have. **Do not attempt them, do not simulate them, do not add them as acceptance criteria.** If you find yourself wanting to run a migration or a probe against a real backend, stop — that is out of scope by design.

**Why this PRD exists (the binding context).** Prod has `S3_VECTORS_ENABLED=true` with `S3_VECTORS_BUCKET="automatos-ai"` — which lacks the required `{workspace_id}` placeholder, so `S3VectorsBackend.__init__` raises the F005 `RuntimeError`. But that error was **swallowed by `run_stage`** and the server **booted dark for weeks**: ~19,130 healthy pgvector chunks were never reachable through the active backend, and every document-grounded answer silently returned nothing. **The durable fix is to make this class of misconfig un-swallowable — loud at boot and red in CI.** That is S1+S2. S3 closes the sibling hole (a wrong-dimension index accepted silently).

## Read first, every iteration

1. `scripts/ralph/prd-186.json` — the story list. A story's `description` + `acceptanceCriteria` = the BINDING contract. Pick the **first story whose `acceptanceCriteria` are not all marked `DONE`**.
2. Full spec (reference, may not be on this branch): `/Users/gkavanagh/Development/Automatos-AI-Platform/automatos-ai-wave1-prds/docs/PRDS/PRD-186-PHASE2-S3-VECTORS-RELIGHT.md`. **This prompt is self-contained — build from it + the JSON even if that file is absent.**
3. `CLAUDE.md` (repo root) — reuse over build; delete what you replace; **no shims**.

## ⚠️ Verify every line number by grep — anchors below are 2026-07-06, they drift

| What | Where (grep to re-confirm exact lines) |
|---|---|
| F005 validate_security branch (extract from here) | `orchestrator/config.py` — the `S3_VECTORS_ENABLED` / `{workspace_id}` check ~1075-1088; `S3_VECTORS_BUCKET` read ~833; `S3_VECTORS_DIMENSION` default (2048) ~835 |
| `run_stage` swallow (the root cause) | `orchestrator/core/models/bootstrap.py` — the `try/except` that marks a stage `failed` and does NOT re-raise ~127-136 |
| Boot call sites of the swallowing stage | `orchestrator/main.py` — `run_stage(DATABASE_INIT, …)` ~179 and ~508; `_boot_phase_1_core` ~162-179 |
| Dimension log-and-continue (make it raise) | `orchestrator/modules/search/vector_store/backends/s3_vectors_backend.py` — `_verify_or_recreate_index` ~118-131; the never-delete invariant comment ~119 |

## The execution contract

- **TDD.** Write the failing PURE test first, then implement, then green. Every behaviour change needs a test that FAILS before and PASSES after.
- **PURE tests only.** Mock/patch `config` values and the `s3vectors` client at the boundary. **No DB, no network, no AWS, no boot of a real server** — the tests must run in CI with zero external service. New backend test files importing `modules.*`/`consumers.*`/`core.*` start with the `_sys_guard` collection-order block (copy from a neighbouring test).
- **Green tip:** `cd orchestrator && python3 -m pytest -q` green. Never commit on red. Clean tree after every commit.
- **No `os.getenv` outside `config.py`.** S1 is an env-*value* concern, not a code read; the bucket already flows through `config.S3_VECTORS_BUCKET`. The guard reads only `config.*`.
- **No backward-compat shim.** S1 *extracts* `assert_vector_config_integrity()` and `validate_security` calls it — the inline F005 branch is **deleted**, not left beside the extraction.
- **Fail loud, never silent.** This whole PRD exists because a `RuntimeError` was swallowed. S2/S3 raise typed errors; no `except: pass`, no log-and-continue on a config-integrity or dimension mismatch.
- **Never delete a populated index.** S3 raises on a dimension mismatch; it must not delete/recreate (that destroys stored vectors — the `:119` invariant holds).
- **No schema migration** — this PRD authorizes none.

## Story-specific guardrails

- **P186-S1** — In `config.py`: add pure `assert_vector_config_integrity()` raising a typed error when `S3_VECTORS_ENABLED` is true AND `S3_VECTORS_BUCKET` is empty or lacks `{workspace_id}`. Refactor the F005 branch of `validate_security` to call it; **delete** the inline duplicate (message string in ONE place). Tests: rejects `automatos-ai` + empty; accepts `automatos-vectors-{workspace_id}`; noop when disabled. **No boot wiring here.**
- **P186-S2** — Wire the S1 assertion into boot **outside** the swallowing `run_stage` so a failure **hard-aborts** the process (fail-closed). Prefer calling it before `run_stage(DATABASE_INIT)`, or escalate that specific failure class into a real abort. Test asserts the boot path raises/aborts (not a swallowed `failed`) on the bad config. Depends on S1.
- **P186-S3** — In `s3_vectors_backend.py` `_verify_or_recreate_index`: **raise** a typed error on a confirmed index-vs-configured dimension mismatch instead of log-and-continue; keep the never-delete invariant. Extract the pure `(configured_dim, reported_dim)` comparison into a small helper. Tests: 4096-under-2048 raises; 2048 passes; the mismatch path does not delete.

## Hard NOs

- NO running migrations, re-embeds, or probes against any real backend (that is Gerard's OPS scope — S1/S4/S5 in the PRD).
- NO weakening or skipping a test to go green; NO `os.getenv` outside `config.py`; NO hardcoded secrets/keys.
- NO deleting/recreating a populated S3 Vectors index; NO parallel config module or second construction seam.
- PUSH after each story commit to `origin ralph/prd-186-s3-vectors-relight` ONLY. NO PRs mid-run, NO merges. A NEW CI red is a bug to fix in-scope.

## Per-iteration protocol

1. Pick the first story with un-DONE ACs; re-verify ground truth fresh (grep the anchors above — never trust a line number blind).
2. Failing pure test → implement → run `cd orchestrator && python3 -m pytest -q`.
3. Commit `feat(prd-186): <story-id> — <title>` with AC evidence; mark that story's AC lines `DONE — <evidence>` in `scripts/ralph/prd-186.json` in the same commit; push the branch.

## Completion

- All ACs DONE → `bash scripts/ralph/acceptance-prd186.sh`. Exit 0 → reply `RALPH_COMPLETE`.
- In-scope gate red → fix in the owning story. Out-of-scope cause (e.g. something that truly needs prod) → `RALPH_BLOCKED` with one line of why.
