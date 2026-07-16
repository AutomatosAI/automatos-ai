# Runbook — Freeze the S10 Memory-Recall Baseline (PRD-198's ⏸ gate)

**Why:** PRD-198's own §8 box blocks the Graphiti build (S2–S6) until two
numbers are frozen. The retrieval half landed with PRD-186 (#547 —
`evals/baseline/kg_retrieval_2026-07.json`, pilot-a, tenant-aliased). The
**S10 memory-recall baseline is the remaining half**, and it must come from
a live workspace snapshot with the production retriever — the bundled
synthetic snapshot is a harness fixture, not a baseline.

**Who runs it:** Gerard (live data access is human-run by policy; agents
prepare, never execute against prod).

**⚠️ Public repo rules:** the live gold set and raw memory corpus contain
tenant content — they stay **LOCAL, never committed**. Only the
tenant-aliased numbers artifact (`pilot-a` style, zero identifiers) is
committed, exactly like the retrieval freeze.

## Steps

1. **Corpus snapshot.** The 2026-07-16 capture (231 durable memories) is at
   `_worktree-backup-2026-07-16/baseline-freeze/` (local). To re-capture,
   export the durable store per workspace (`modules/memory/durable_store.py`
   namespaces `mem:{workspace}`) to a local jsonl in the harness's
   `MemoryDoc` shape (`evals/memory_recall.py` docstring).
2. **Gold set.** Author ~50 labelled queries over that corpus (LongMemEval
   category shape — the harness's `GoldQuery`), locally. Aliased tenants
   only in any file that might ever be shared.
3. **Run the harness against the snapshot with the production retriever:**

   ```bash
   cd orchestrator
   python -m evals.memory_recall \
     --corpus /LOCAL/path/memory_corpus.jsonl \
     --gold /LOCAL/path/memory_gold.jsonl \
     --json /LOCAL/path/memory_recall_raw.json
   ```

   (`--corpus` + the `retriever_factory` injection point drive the REAL
   retriever; the offline bag-of-words proxy is only the CI stand-in. If the
   run must happen against prod Qdrant, use the established `railway ssh`
   heredoc-in-arg wrapper pattern — agent-driven railway ssh is blocked.)
4. **Alias + freeze.** Strip identifiers, alias tenants (pilot-a/b), and
   commit the numbers as `orchestrator/evals/baseline/memory_recall_2026-07.json`
   with the same `{<alias>: {variants: {<variant>: {mean_recall_at_5: …}}}}`
   shape as the retrieval freeze (plus `frozen_at`/`provenance`/`caveats`
   header keys). An honest sub-target number is still the baseline — the
   trial must beat *the real number*, not a hoped-for one.
5. **Verify the gate sees it:**

   ```bash
   python -m evals.graphiti_vs_baseline
   ```

   The verdict should move from `PENDING (memory_baseline_s10 …)` to
   pending only on the graphiti treatment. That is the state in which the
   PRD-198 S2–S6 build may start (margin + credit + cohort boxes permitting).
