# Ralph Build Prompt — PRD-232 The Intent Graph

You are executing **PRD-232**, one story per iteration, unattended. Branch **`ralph/prd-232-intent-graph` ← `fix/main-ci-wave-drift`** (STACKED on unmerged PR #643 — its two test fixes are IN your tree; build ON them; NEVER touch its two files except through rebase). Tip green after every commit.

**CONTEXT.** The 2026-08-29 deep review (spec §2, file:line evidence) found the tool-selection pipeline ~80% built with four wiring inversions: the `platform_execute` dispatcher is STRIPPED on the graph branch; the graph runs under the wrong flag; the signal recorder is dark; intent clusters are write-only and affinities cluster-blind. You fix the wiring, author the synthetic utterance corpus, seed the graph, make learning real, and prep the eval. Decisions §6 are **LOCKED** — do not re-litigate them.

**RVW-2 RULING (2026-08-29 evening):** Gerard chose option (C) — the two-layer graph (global text-free prior + per-tenant overlay), now PRD-232 §6.5, amending PRD-177's per-tenant lock. P232-RVW-2 is UNBLOCKED and rewritten in prd-232.json — build it; do not re-raise the §12 block.

## Read first, every iteration

1. `scripts/ralph/prd-232.json` — the BINDING contract (anchors, traps, locked decisions). First story with un-DONE ACs.
2. `docs/PRDS/PRD-232-INTENT-GRAPH.md` — the spec; §2 has the evidence, §7 the traps.
3. `CLAUDE.md` — reuse over build; delete-what-you-replace; canonical terms.

## The execution contract

- **RE-VERIFY every anchor by grep before building on it** — the review evidence is from 2026-08-29 but the tree moves. If an anchor is gone or different, adapt with evidence in the commit body; if a story's premise is broken, `RALPH_BLOCKED` with the grep.
- **You are the utterance generator** (US-005): author the corpus YAML yourself. NOTHING in this build may call an external LLM API or any network service.
- **NO DB, NO servers, NO eval runs**: tests are PURE or fixture-based (`@integration` skips cleanly without local Postgres; real Postgres is CI). `seed_tool_routing_graph.py` and the uplift eval are HUMAN-APPLIED — you extend them, you never execute them against a database.
- **Flags:** `TOOL_ROUTING_GRAPH` stays default **false** (flip is a post-merge human step, US-013). `TOOL_SIGNAL_RECORDER_ENABLED` flips default **true** (US-009, locked). No `os.getenv` outside `config.py`. No hardcoded values — thresholds/pins in config.
- **NO new tables; alembic revisions only as explicitly authorized:** US-007 is AUTHORIZED for exactly ONE revision (`prd232_cluster_provenance`, nullable provenance column — see the story). Anything else needing schema: `RALPH_BLOCKED` with the evidence — never create schema silently.
- **No new ranking systems.** You wire and feed `ActionSemanticIndex` + `GraphRouter`. A second ranker or a parallel pins mechanism is an automatic review CRITICAL.
- **DeterministicEmbeddingProvider never appears outside explicit test fixtures** (PRD-185 S3).
- **Green tip:** `cd orchestrator && python3 -m pytest -q` after every commit; never commit on red. Pre-existing env failures (DB-bound) are known — your gate is the branch-scoped set; do not "fix" unrelated red.
- **STAGING DISCIPLINE:** explicit paths only. **NEVER `git add -A`/`.`/`-u`** (node_modules is untracked and NOT gitignored). **Never `git stash -u`.**

## Hard NOs

- NO flipping `TOOL_ROUTING_GRAPH` default; NO running seeds/evals against any DB; NO external LLM/API calls.
- NO new tables or routes/routers; alembic ONLY the single authorized US-007 revision; route manifest byte-untouched.
- NO edits to `orchestrator/tests/test_prd222_w2s1_plan_tiers.py` or `orchestrator/tests/authz_sweep_probe.py` (they belong to #643 beneath you).
- NO second always-include/pins mechanism; NO duplicate ranking path (PRD-141 US-016's rule stands).
- NO weakening tier gating (`super_admin_only`/admin fail-closed suites stay green untouched).
- NO `git add -A`/`.`/`-u`; NO `git stash -u`. PUSH after each story commit to `origin ralph/prd-232-intent-graph` ONLY. NO PRs mid-run, NO merges.

## Per-iteration protocol

1. Pick the first story with un-DONE ACs; re-verify its anchors fresh.
2. Implement → `cd orchestrator && python3 -m pytest -q` (branch-scoped focus, full run for sanity).
3. Commit `feat(prd-232): <US-id> — <title>` with evidence in the body; mark that story's AC lines `DONE — <evidence>` in `scripts/ralph/prd-232.json` in the same commit; push.

## Completion

- All ACs DONE → `bash scripts/ralph/acceptance-prd232.sh`. Exit 0 → reply `RALPH_COMPLETE`.
- A story cannot be built without violating a Hard NO → `RALPH_BLOCKED` with one line of why + the grep evidence in the last commit.
