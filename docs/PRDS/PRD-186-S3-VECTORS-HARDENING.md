# PRD-186 (Revised 2026-07-14) — S3 Vectors **Hardening**: guard the live shared-bucket plane (tenant-isolation completeness + dimension fail-loud + parity eval + config-integrity CI gate) — P2-16-adjacent

> **Status:** DRAFT for review — spec only, **no build yet**. **This REVISES + supersedes the original `PRD-186-PHASE2-S3-VECTORS-RELIGHT.md`** (the per-workspace-bucket relight), which is now obsolete: main relit S3 a *different* way (a shared bucket + query-time isolation, `prd-172` commit `e2c86f6bd`), so the original's S1/S2 would **break the live prod config**. The RELIGHT file is retired on approval (delete-what-you-replace, §5).
> **Grounded @ `main` b4748414a** (refs re-confirmed against live code, NOT the 121-behind `ralph/prd-186` branch). Feeds/aligns with the reslimmed PRD-197.

---

## 0. What changed since the original 186 (the reconciliation)

The original 186 assumed the S8-probe world: *S3 dark because the bucket lacked `{workspace_id}`; fix = per-workspace templated buckets, fail-loud if absent.* **Main solved it differently and shipped it:** a **shared** bucket is now explicitly allowed (`config.py:1176-1177`), and tenant isolation moved from *physical* (separate buckets) to *logical* (a query-time metadata filter in `search()`). So:

| Original 186 story | Verdict vs live main | Disposition |
|---|---|---|
| S1 per-workspace templated bucket | **obsolete** — main deliberately allows the shared bucket | **drop** |
| S2 fail-loud on placeholder-less bucket | **inverted** — reviving it hard-aborts the live prod config | **drop** (main already fails on an *unset* bucket) |
| S3 dimension-mismatch fail-loud | **still valid** — `_verify_or_recreate_index:117` is *still log-only* on main | **revive → S2 here** |
| S4 re-embed / S5 probe-LIVE | **done** — S3 is populated and serving | **closed** (S5-here re-verifies coverage) |

This revision keeps the one surviving guard and adds the **broader hardening** the plane now needs — grounded in what actually shipped.

---

## 1. What this is

A **hardening** PRD for a plane that is **live, not dark**. Documents retrieve from S3 Vectors today (`_get_candidates → S3VectorsBackend`); `document_chunks` is now the content-hydration table, not the vector plane. The relight worked — but it shipped with the guards *off*: a dimension mismatch only logs, the tenant-isolation filter has a `None`-shaped hole, one of two write paths may not stamp `workspace_id`, there is no number proving S3 retrieval matches the pre-revert quality, and no CI gate stops the config drifting dark again. This PRD closes those — it is the "make the live S3 plane un-silently-breakable" work.

**Framing (CLAUDE.md §3):** **Refactor / hardening**, not net-new — guards + one eval variant + one verification probe. The net-new is a `None`-fail-closed branch, a raise-on-mismatch, one `assert_vector_config_integrity()`, and an S3 eval variant. **Build size:** S–M · **Risk:** Medium — S1 changes the **live retrieval path** (a stricter isolation filter), so it is measured against the frozen baseline (§7) and the `None`-drop is surfaced as a decision (§8-Q1), not taken silently.

**Why now / why measured:** S3 is the grounding plane for every agent answer over client documents, and it is running unguarded. The security half (tenant isolation) and the quality half (parity vs the pre-revert baseline) are both **numbers** — isolation-leak count and recall@k delta — so this is measurement-forward: prove isolation is complete and recall is flat-or-better, then the plane is trustworthy.

---

## 2. Current reality (grounded @ `main` b4748414a)

- **S3 is the live vector plane, on a shared bucket.** `config.py:1176-1177` allows a bucket with no `{workspace_id}` placeholder; `S3VectorsBackend.__init__:55` templates the (optional) placeholder; `_get_candidates` (`modules/rag/service.py:1017`) constructs the backend per workspace. `document_chunks` is content hydration only (`service.py:1176 FROM document_chunks`).
- **Isolation is a query-time filter — with a `None` hole.** `search()` (`s3_vectors_backend.py:132-207`) is bound to one `workspace_id`: it returns `[]` if an explicit `filters['workspace_id']` disagrees (`:158-168`), and drops any hit whose `metadata.workspace_id != required_ws` (`:193-195`). **But the drop is `if hit_ws is not None and hit_ws != required_ws`** — a hit with **no** `workspace_id` in metadata (`None`) is **passed through**. On a shared bucket that is a latent cross-tenant path for any unstamped chunk.
- **Two write paths; only one is proven to stamp `workspace_id`.** `S3VectorsBackend.add_documents` **does** set `metadata["workspace_id"] = self.workspace_id` (`:245-250`). But `modules/rag/ingestion/processor.py:308-309` writes via `add_vectors(vectors, metadata_list)` with a **caller-supplied** metadata list whose `workspace_id` stamping is not guaranteed at that seam — the exact source of a `None`-workspace chunk.
- **Dimension mismatch only logs.** `_verify_or_recreate_index` (`s3_vectors_backend.py:117-130`) logs `reported vs config` dimension and proceeds; it never raises. A stale comment at `:266` still says "4096-dim" while the live corpus + `config.S3_VECTORS_DIMENSION` are **2048**; the method docstring (`:82-84`) still claims delete-and-recreate although the code correctly never deletes — a misnomer + drift.
- **No config-integrity function or CI gate.** Only `validate_security()` exists (`config.py:1164`); the extracted `assert_vector_config_integrity()` the original 186-S2 built was never merged, and there is no CI test pinning the (now shared-bucket) config rules.
- **Re-embed + liveness are done.** S3 is populated and serving (the `prd-172` restore); `scripts/probe_document_vectors.py` (PRD-185 S8) remains the read-only liveness/coverage oracle. This PRD does **not** re-embed; it *verifies* coverage (S5).

---

## 3. Findings → fix → story

| # | Finding (grounded) | Fix | Story |
|---|---|---|---|
| **isolation `None`-hole** | `search():194` passes hits with `workspace_id=None` on the shared bucket; `processor.py:308` may write unstamped chunks. | Make `search()` **fail-closed on `None`** (drop unlabeled hits on a shared bucket), and stamp `workspace_id` at **both** write paths so no unlabeled chunk is ever written. | **S1** |
| **dimension log-only** | `_verify_or_recreate_index:117-130` logs a mismatch and proceeds; stale 4096 comment + delete-and-recreate docstring drift. | **Raise** (typed) on a confirmed index-vs-config dimension mismatch (never delete/mutate a populated index); rename the misnomer; fix the `:266`/`:82` drift. (Revives original 186-S3.) | **S2** |
| **no config-integrity gate** | No `assert_vector_config_integrity()`, no CI test; the config can drift dark again silently (as it did for weeks pre-relight). | A pure `assert_vector_config_integrity()` for the **shared-bucket rules** (enabled ⇒ bucket set; dimension coherent), wired to hard-abort boot **outside** the swallowing `run_stage`, + a CI test. | **S3** |
| **no parity number** | Nothing proves S3 retrieval matches the pre-revert (pgvector-fallback era) quality. | An **S3 recall/latency parity** variant in `evals/retrieval_recall.py`, asserted **flat-or-better** vs the frozen baseline (§7). | **S4** |
| **coverage unverified** | No confirmation every live chunk is in S3 with `workspace_id` + dim 2048 (the S1 `None`-drop assumes coverage). | Extend `scripts/probe_document_vectors.py` to report per-workspace **`workspace_id`-stamped %** + dimension; **OPS**, Gerard runs; remediate gaps via the existing migrator. | **S5** |

---

## 4. Stories (test-first; CI is the only gate — no local runs)

### S1 · Close the tenant-isolation gap — fail-closed on `None` + stamp both write paths — S · _security_
Make the shared-bucket isolation total. (a) In `search()` (`s3_vectors_backend.py:193-195`) drop a hit whose `metadata.workspace_id` is **missing or `None`**, not only one that mismatches — on a shared bucket an unlabeled chunk must never reach another tenant's context (§8-Q1 confirms this behavior change). (b) Guarantee the write side: `add_documents` already stamps `workspace_id` (`:250`); audit `modules/rag/ingestion/processor.py:308` `add_vectors` and ensure its `metadata_list` carries `workspace_id` for every vector, at the seam, so no `None`-workspace chunk is ever written. 
**Test:** `test_search_drops_unlabeled_hit_on_shared_bucket` (a hit with no `workspace_id` is excluded); `test_search_drops_mismatched_hit` (unchanged behavior held); `test_add_vectors_stamps_workspace_id` (the processor path writes `workspace_id` on every vector). Pure — mock the `s3vectors` client at the boundary; no AWS.
**Notes:** This is the one story that changes the live retrieval path — measured flat-or-better on recall (§7) so the stricter filter doesn't silently drop *legitimate* results (it shouldn't: legitimate chunks are stamped). No `os.getenv`. 

### S2 · Dimension-mismatch fails loud + kill the `_verify_or_recreate` misnomer — S · _revives original 186-S3_
`_verify_or_recreate_index` (`s3_vectors_backend.py:117-130`) **raises** a typed error on a confirmed `get_index` dimension ≠ `config.S3_VECTORS_DIMENSION`, instead of logging and proceeding — so 2048-dim vectors can never be silently written to/queried against a wrong-dim index. **Never delete/recreate** a populated index (the `:118` invariant holds). Rename the method to its real behavior (e.g. `_assert_index_dimension`); fix the stale `4096` comment (`:266`) and the delete-and-recreate docstring (`:82-84`) to match live dim 2048.
**Test:** `test_index_dimension_mismatch_raises` (mocked `get_index` returns 4096 under config 2048 ⇒ typed raise); `test_index_dimension_match_passes` (2048 ⇒ no raise). Pure — mock the client; no AWS.

### S3 · Config-integrity CI gate for the shared-bucket rules — S · _durable fix_
Add a pure `config.assert_vector_config_integrity()` that raises when `S3_VECTORS_ENABLED=true` **and** the bucket is unset **or** the configured dimension is incoherent — the **shared-bucket** rule set (a placeholder-less bucket is **valid**; do NOT reinstate the original 186-S2 placeholder requirement). Wire it to hard-abort boot **outside** the swallowing `run_stage` (the original 186 lesson: `bootstrap.py` swallowed the F005 `RuntimeError` for weeks), and add a CI test so the config can't ship dark again.
**Test:** `test_vector_config_integrity_rejects_unset_bucket_when_enabled`; `test_vector_config_integrity_accepts_shared_bucket` (`"automatos-vectors"`, no placeholder ⇒ passes — the live prod shape); `test_vector_config_integrity_noop_when_disabled` (open-core local); `test_boot_aborts_on_bad_vector_config` (fatal, not a swallowed `failed` stage). Pure — patch `config` at the boundary; no DB/AWS.
**Notes:** Extract the shared assertion so `validate_security` calls it (no duplicate string, §5). 

### S4 · S3 recall/latency parity eval — S · _quality guard_
Add an `s3` variant to `evals/retrieval_recall.py` that drives the real S3-backed `RAGService.retrieve` and records recall@k + latency, asserted **flat-or-better** vs the frozen pre-revert baseline (§7) using the existing honest-gate shape (published, exit-0, non-required lane). This is the number that proves the S3 relight held retrieval quality — and it is the same baseline PRD-197/198/199/201 consume.
**Test:** `test_s3_retrieval_variant_registered` (the variant is A/B-able and defaults to the bundled snapshot); `test_parity_gate_flat_or_better` (the honest-gate arithmetic; a drop is a reported regression, not a red build). Pure — bundled jsonl + stdlib; no live S3/LLM.

### S5 · Backfill/coverage verification — OPS (Gerard runs), reuse the probe — XS
Extend `scripts/probe_document_vectors.py` (read-only) to report, per workspace, the **fraction of vectors carrying a `workspace_id` in metadata** and the index **dimension**, confirming S1's `None`-drop is safe (i.e. coverage is ~100% so the stricter filter removes nothing legitimate) and the corpus is uniformly dim 2048. If any workspace shows unstamped vectors, remediate by re-embedding those docs via the existing `scripts/migrate_to_s3_vectors.py` (no new migrator).
**Deliverable (OPS, not CI):** the probe's per-workspace `workspace_id`-coverage + dimension finding, attached to close the PRD. **Gerard runs against prod** (needs AWS + DB); explicitly not a CI test.

---

## 5. Sequencing
- **S5 (coverage probe) ideally first** — it tells us whether the `None`-drop (S1) is a no-op (100% stamped) or a live fix (gaps exist), and whether a remediation re-embed is needed before S1's stricter filter goes live.
- **S2 + S3 are independent pure code** (different files) — parallel-safe, land any time.
- **S1** lands after S5 confirms coverage (so the stricter filter can't strand legitimate chunks) and is measured by **S4's parity eval** on the same PR.
- **S4** needs the frozen baseline (§7); its harness can land first and sit green until the baseline exists.
- No document-data migration here (S3 is populated); the only prod motion is S5's optional remediation re-embed, Gerard's to run.

## 6. Verification (CI is the only gate — no local runs)
Per `feedback-no-local-servers`: **no servers, builds, `pytest`, `tsc`, or installs on the dev machine.** All code tests are pure — mock the `s3vectors` client and `config` at the boundary; no AWS/DB/network. The parity eval (S4) stays a **non-required, exit-0** lane (the number is the deliverable). S5 is an **OPS** probe Gerard runs against prod, not CI. No new routes (backend-internal) ⇒ no `route-manifest.json` change. Any config-integrity boot change is covered by S3's `test_boot_aborts_on_bad_vector_config`.

## 7. Baseline capture / measurement
Consumes the **same frozen retrieval baseline** PRD-197/198/199/201 use — freeze once, serve the wave. **Capture (Gerard):** `evals/retrieval_recall.py --live --workspace <id>` against the real S3-backed retrieve → `baseline/s3_retrieval_2026-07.json`. **Success = the delta:** tenant isolation **complete** (S1 — 0 unlabeled hits reachable, coverage ~100% per S5); recall@k **flat-or-better** vs baseline (S4 — the stricter filter costs no legitimate results); dimension mismatch **fails loud** (S2, was silent); config drift **caught at boot + CI** (S3, was swallowed for weeks); `workspace_id`-coverage **measured** (S5, was unknown).

## 8. Open questions — Gerard's call (decide, don't let me defer — CLAUDE.md §12)
1. **`None`-workspace fail-closed (the security behavior change).** S1 makes `search()` drop hits with missing/`None` `workspace_id` on the shared bucket. **Recommendation: yes, fail-closed** — isolation must not depend on every writer remembering to stamp; a stray unlabeled hit reaching a tenant is worse than dropping it (S5 confirms coverage so nothing legitimate is lost). Confirm.
2. **Keep the shared bucket, or restore physical per-workspace buckets later?** **Recommendation: keep shared + the query-time filter** (simpler ops, isolation enforced fail-closed at query; per-workspace buckets were belt-and-suspenders that broke a working deploy). If you ever want physical separation it's a *new* PRD, not a revival of the dropped 186-S1.
3. **Parity margin (S4).** **Recommendation: flat-or-better** (S3 recall@5 ≥ frozen baseline − a tiny epsilon); a real drop is a regression to investigate, not a ship. Confirm the epsilon.
4. **Backfill remediation (S5).** If the probe finds unstamped/wrong-dim vectors, re-embed those docs via the existing migrator (Gerard-run) **before** S1's filter goes live. Confirm this is the remediation path (vs a metadata-only patch).
5. **Baseline metric set (§7).** Confirm recall@k on the live gold set + `workspace_id`-coverage + dimension are the right before-numbers.

---

*Supersedes `PRD-186-PHASE2-S3-VECTORS-RELIGHT.md` (retire on approval, §5). Revives that PRD's still-valid S3 (dimension fail-loud) and its config-integrity/fail-loud lesson (`run_stage` swallowing), drops its obsolete per-workspace-bucket premise, and adds the tenant-isolation completeness + parity-eval + coverage-verification hardening the shipped shared-bucket design needs. All `file:line` refs grep-confirmed against `main @ b4748414a` (S3 live via shared bucket per `prd-172` `e2c86f6bd`; `search()` `None`-hole at `:194`; dimension log-only at `:117`; two write paths `add_documents:250` / `processor.py:308`). Aligns with the reslimmed PRD-197 (which drops dimension-authority → owned here, and drops retire-S3 → wrong). Reuses `scripts/probe_document_vectors.py` (S5) + `scripts/migrate_to_s3_vectors.py` (remediation) — no new probe/migrator. PILOT lens (the plane is live, cold-start ≠ defect); measurement-forward; security framed as isolation-completeness, not moat.*
