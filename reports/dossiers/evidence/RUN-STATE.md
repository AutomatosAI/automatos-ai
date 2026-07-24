# Phase 2 Module Deep-Review — Run State

Brief: `reports/PHASE2-MODULE-DEEP-REVIEW-PROMPT.md` · Baseline: `reports/PLATFORM_OS_REVIEW_2026-07-01.md`
Date: 2026-07-04 · Operator: Claude Fable 5 (session model claude-fable-5, every workflow agent pinned `model: 'fable'`)

## Review target
- **Pinned source tree (read-only worktree):** `/private/tmp/claude-501/-Users-gkavanagh-Development-Automatos-AI-Platform/9e36dcac-822b-4c58-87a4-6e6ac2981e3f/scratchpad/p2-src`
- **Commit:** origin/main @ `77bc9c6d5` (Merge PR #503, 2026-07-04). All 14 hardening waves are merged on this commit (W1–W6 PRs #462-465/#481-484; W7–W14 PRs #494–#502).
- Gerard's working checkout is untouched (it sits on `chore/remove-ralph-from-git`). Remove the worktree after the run: `git -C automatos-ai worktree remove <path>`.

## Output layout (all uncommitted, in Gerard's checkout)
- `reports/dossiers/<module>.md` — one dossier per capability (sections A–J per the brief)
- `reports/dossiers/thesis-T{1,2,3}*.md` — cross-cutting thesis verdicts
- `reports/dossiers/evidence/` — capability-map.md, phase0-residual-map.md, real-data-inventory.md, per-module lens packs (`<module>--<lens>.md`), `data/` live-store samples
- `reports/PLATFORM_MODULE_DEEP_REVIEW_2026-07-04.md` — the final synthesized report

## Phase plan & status
| Phase | Workflow | Status | Run ID |
|---|---|---|---|
| W1: capability map + Phase-0 fix verification + real-data recon | p2-map-phase0 | **DONE** (16 agents, 2.5M tok) | wf_aaa8645d-1e1 |
| Checkpoint: eyeball map | inline | **DONE** — 28 modules (15 deep/13 standard); Phase0 = 38 fixed / 22 partial / 27 not-done / 0 regressed / 3 unverifiable | — |
| §5.F scope decision | inline | **DONE** — Gerard: run full defensive lens on EVERY module (ignore the "skip §5.F" filter-workaround text) | — |
| W2: per-module dossier fan-out | p2-dossiers-fable | **DONE — 28/28 banked** (session 921d8e8a). ~20 written on Fable + 8 via per-module Opus fallback when the **Fable-5 credit pool** ran dry mid-run (distinct from the account session-limit). ~87 agent launches / ~7.3M tok across 2 launches (wpo7l80ha 51ag/1.9M, wg14ahp27 36ag/5.4M). | wf_9797e7bc-097 |
| Checkpoint: dossier sanity pass | inline | pending | — |
| W3+W4+Security: theses (T1/T2/T3) + Opus security appendix (parallel) → synthesis final report | p2-finish-opus | **DONE** (all Opus, 5 agents/1.3M tok/~18min) → `reports/PLATFORM_MODULE_DEEP_REVIEW_2026-07-04.md` (261 lines) + thesis-T1/T2/T3 + security-hardening-appendix | wf_2d8f2aaa-cd2 |

## ✅ PHASE 2 COMPLETE — 2026-07-04
32 files in `reports/dossiers/` (28 module dossiers + 3 theses + security appendix) + final report `reports/PLATFORM_MODULE_DEEP_REVIEW_2026-07-04.md`. Central finding: **"good bones, open loops"** — all 28 score 2–3 (pilot unproven-not-broken). Verdicts: **T1** HOLD graph-migration (fix loops + eval first) / **T2** stay modular-monolith (3.0% real coupling) / **T3** adopt Langfuse + feed loops. #1 fix = telemetry type-poison one-liner (`composio_cache.py:215`) un-starves the whole learning plane. Cleanup TODO: remove the pinned review worktree `git -C automatos-ai worktree remove <p2-src path>` (see §Review target).

## W1 headline findings (feed the dossiers)
- **Phase 0:** 90 enumerable July findings graded on default config @ 77bc9c6d5 — 38 fixed, 22 partial, 27 not-done, 0 regressed, 3 unverifiable. Execution-spine + tenant-isolation criticals all closed. Dominant PARTIAL = W4 policy plane fully built but gated behind `AUTOMATOS_POLICY_PLANE` default OFF (config.py:645) → default deploys keep July behavior on F040/F014/F042/F043. Channels (F026-29/F066/F081) = largest fully-untouched cluster. PRD-184 kill-list never authored → dead-code backlog orphaned. Two CI lanes green-but-failing-inside (fresh-clone boot + from-zero alembic replay masked by continue-on-error).
- **Real data (prod, read-only):** memory = 87% operational chatter, **0 L3 promotions ever**, recall last used 2026-03-11; operating graph = **100% synthetic** (2026-05-05 seed), 0 organic edges for 21 real workspaces (tool_execution_logs unfed); playbook LLM steps failing daily on OpenRouter 402 since mid-June, board closes them **done** (194 done-with-error / 484 done-no-result); missions engine unused since 2026-06-13 (8/17 stuck awaiting_approval); 152 live tables vs 109 in July; alembic clean single head. **mem0 + Qdrant unreachable from this machine** (dead Railway hostname) → durable+field memory halves unaudited, flagged as gap.

## Dossier scope — §5.F DROPPED (final decision 02:59 Lisbon)
Gerard's final call: **drop the §5.F defensive/enterprise/security lens entirely** — dossiers are **A–E + G–J only**. Security/robustness is a **separate dedicated Opus pass later**, not this run. Deep = 4 lens agents (audit/comp/eval/ux), standard = 2 (audit/comp). Writers instructed to omit section F and all security/robustness/PII/ACL analysis; maturity.enterprise + maturity.defensive returned null. (This supersedes the earlier "run full defensive lens" instruction.) Fan-out is now ~142 agents.

## Session-quota resume chain + auto-continue loop
Account-wide session quota exhausts mid-run (both Fable AND Opus attempts returned identical "hit your session limit · resets 7:50am Lisbon" — it is account-level, not model-specific). On each completion notification: if failures show "hit your session limit" and dossiers are incomplete, **resume again** with `Workflow({scriptPath, resumeFromRunId: <latest runId>})` — completed dossiers/agents return cached, only the unfinished re-run. Repeat until `count == 28`.

**Auto-continue loop — PAUSED 2026-07-04.** Cron `6b919f14` **DELETED** after Gerard flagged the Fable credit burn. Dossiers are complete (28/28); the remaining phases (theses T1–T3, synthesis, Opus security pass) await his go and will run on **Opus, not Fable** (~5–8 agents total, a fraction of the dossier fan-out). *(historical:)* the cron (fired :38) advanced the pipeline one stage per tick — dossiers→theses→synthesis→Opus-security→final report — idempotent (skips anything on disk), overlap-guarded, quota-aware (notes reset time + retries next tick). Script `p2-dossiers-fable-wf_9797e7bc-097.js`. **Caveat:** cron is session-only (dies if session 921d8e8a closes); resume-cache also needs same session. If the session closes, a fresh session can still re-run the same script — agents skip the on-disk dossiers, so nothing banked is lost.

## Standing constraints (every agent prompt carries these)
Analysis only (no servers/builds/tests/installs — CI is the only gate per Gerard); read-only git/gh/jq/curl-GET; live-store access strictly read-only (SELECT/GET, LIMIT ≤50, timeouts); no moat/pitch framing — North-Star judgement only; file:line for internal claims, cited URLs for external; honest negatives required; canonical vocab (Playbook/Mission/Task/Deliverable/Knowledge Graph/Command Center/Auto).

## Recovery
Scripts persist at `~/.claude/projects/-Users-gkavanagh-Development-Automatos-AI-Platform/9e36dcac-822b-4c58-87a4-6e6ac2981e3f/workflows/scripts/`. Resume a stopped run: `Workflow({scriptPath, resumeFromRunId})` — completed agents return cached. W1 note: `args` must be hardcoded literals in the script (passing via the args param arrived undefined and crashed the first launch, wf_3f21ce9d-098).

## Environment facts
Web access confirmed working (competitive lenses viable). psql/jq/curl present. Env files with DATABASE_URL / MEM0_API_URL at `automatos-ai/orchestrator/.env` (recon uses them read-only). graph.json is stale (2026-06-09) — agents told to treat as shape-only.
