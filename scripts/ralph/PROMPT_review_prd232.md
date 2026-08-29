# Ralph Review Prompt — PRD-232 The Intent Graph

You are a fresh-context **adversarial reviewer**. The build claims PRD-232 is complete. Your job: find where the wiring fix quietly regresses the surface, where the seeding lies about coverage, or where "learning" is still write-only. You fix NOTHING yourself.

## Scope

```
BASE=$(git merge-base HEAD fix/main-ci-wave-drift)
git diff --stat $BASE..HEAD
git diff $BASE..HEAD
```

**STACKED:** diff against `fix/main-ci-wave-drift` (#643), NOT main. Read `scripts/ralph/prd-232.json` (binding, locked decisions) + `docs/PRDS/PRD-232-INTENT-GRAPH.md` (§2 evidence, §7 traps).

## Hunt list — every item is a confirmed-risk class

1. **Dispatcher survival is the whole point (US-001).** Trace every `route()` return path in the DIFFED code: any branch that can return a non-empty surface without `platform_execute` = CRITICAL. The replay test must use a REALISTIC assembled tool list (dispatcher + 47-or-pinned promoted + core), not a toy fixture — a fixture without the dispatcher proves nothing (that was the original test hole, test_us014_graph_router_delegation.py:130-181).
2. **Flag honesty (US-002).** With `TOOL_ROUTING_GRAPH=false`: grep+test that NO GraphRouter query runs on any path (schema, catalog, shadow). With true: both surfaces consult it. A path still gated on `SEMANTIC_TOOL_ROUTING` calling the graph = CRITICAL (the original inversion reborn).
3. **The single ranking pass (US-003).** Spy-count in the test must cover the FULL turn including catalog render and shadow log. A second embedding computation hiding in a fallback branch = HIGH. Cross-request result caching keyed without workspace/agent = CRITICAL (tenant leak).
4. **Corpus honesty (US-005).** Run the coverage linter yourself. Spot-check 10 random actions for utterance QUALITY (register variants present, not 20 trivial rephrasings of the description). `platform_update_task_status` must carry close/ticket/blocked. Phrase-map + examples provenance complete. su-only actions present in the corpus = CRITICAL (selection-side leak).
5. **Seeded clusters vs the nightly (US-007).** The delete-and-reinsert at edge_builder.py:440-470 is the trap: prove via the test that seeded rows SURVIVE build_edges and the live FK `ToolExecutionLog.intent_cluster_id` cannot be orphaned mid-transaction. Seeds outranking higher-Wilson organic rows = HIGH. Any migration/table created silently = CRITICAL (should have been RALPH_BLOCKED).
6. **Learning reads (US-010).** Cluster-blind affinity application surviving anywhere = HIGH. `failed_after` still write-only = HIGH. Threshold hardcoded = MEDIUM.
7. **Gap/decay plumbing (US-011).** The VECTOR replay fixture must produce a gap row through the REAL recorder path (not a hand-inserted row). Decay must floor at seeded confidence, never zero-out the bootstrap. Per-call DB sessions in any new signal path = CRITICAL (the PRD-141 US-019 contract).
8. **Promotion-as-prior (US-014).** Reachability: EVERY formerly-promoted action must be provably callable (pinned, ranked-in, or dispatcher enum) — enumerate and test, not assert. Tier fail-closed suites untouched and green. The before/after token numbers must be real measurements in the PR body, not estimates. Pins hardcoded outside config = MEDIUM.
9. **Eval prep (US-012A).** Abstain rows scoreable; no network/DB in tests; eval_set count consistent with seed yaml. Any code path that RUNS the uplift eval or seed script against a DB = CRITICAL.
10. **Conventions.** ZERO new alembic files/tables/routes; route manifest byte-identical to base; no `os.getenv` outside config.py; no edits to #643's two test files; no staging poison (`node_modules` — CRITICAL); DeterministicEmbeddingProvider only in fixtures; no second ranking/pins mechanism.

## Verification

- Run the **code-review** skill (or code-reviewer agent) on the diff — any CRITICAL/HIGH it reports is a finding.
- `gh run list --branch ralph/prd-232-intent-graph --workflow test.yml --limit 3`: a NEW failure vs the base branch's known state = finding.
- Run `bash scripts/ralph/acceptance-prd232.sh`. **Non-zero = automatic CRITICAL.**

## Verdict protocol

- **No CRITICAL/HIGH/MEDIUM** → reply exactly `REVIEW_PASS` + a 5-line summary noting: (a) merge order — #643 → this branch's PR (retarget to main after #643); (b) post-merge human steps: seed script with --yes, uplift eval run, ≥5pt gate, THEN TOOL_ROUTING_GRAPH=true in Railway; (c) rollback lines (both flags).
- **Findings** → append `P232-RVW-1..n` fix stories to `scripts/ralph/prd-232.json` (title, `file:line` evidence, mechanical ACs). Commit `chore(prd-232): review findings → fix stories`. Reply `REVIEW_FINDINGS`.
- A finding that requires Gerard's judgment (not mechanics) → story marked `DECISION (Gerard, §12)` — the builder must BLOCK on it, not resolve it.
- Do not fix code. Push only the fix-story commit to `ralph/prd-232-intent-graph`. Do not re-run the build.
