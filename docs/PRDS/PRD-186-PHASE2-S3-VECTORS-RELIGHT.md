# PRD-186: Phase 2 · S3 Vectors Proper Relight — completes P2-03, closes the S8 gate

**Phase:** Phase 2 — Module Deep-Review remediation (completes the Wave-0 P2-03 gate before Wave 1)
**Branch:** `feat/p2-s3-vectors-relight` · **Worktree:** `automatos-ai-wave1-prds`
**Dependencies:** PRD-185 (Wave 0) merged to `main` (`649482aa3`) — specifically the S8 read-only probe (`orchestrator/scripts/probe_document_vectors.py`, `4975a4488`) whose finding is the premise of this PRD, plus the F005 backend/boot guard it exercises.
**Build size:** S–M (two small code stories + one OPS re-embed that reuses an existing script; the durable fix is one config-integrity guard) · **Risk:** Low code / Medium ops (the re-embed is a prod data operation Gerard runs — it re-embeds ~19,130 chunks; idempotent-ish, dry-run first)
**Source:** `reports/PLATFORM_MODULE_DEEP_REVIEW_2026-07-04.md` §6 Wave-0 **P2-03** (+ the Wave-3 **P2-16** conflict this PRD surfaces); dossiers `reports/dossiers/vector-substrate.md` (§C.1, J1) and `reports/dossiers/rag-retrieval.md` (§C.2); live S8 probe finding (below).

---

## Overview

Wave 0's S8 probe (PRD-185) was a decision gate, not a fix: *is the document-vector plane live, and at what dimension?* The probe has now been **run against prod**, and the answer is unambiguous — **the plane is DARK, and has been since S3 Vectors was switched on (~W2).** This PRD is the relight: it does exactly the four things S8 said would follow a DARK verdict, in the way Gerard has chosen to do them.

Judged against the **North Star** — *does this make Auto more autonomously capable and its output higher-quality for clients?* — relighting the document plane is the single highest-leverage grounding fix left in the program. Every agent answer over workspace documents (Knowledge search, the planning pack, the widget, NL2SQL schema grounding) has been silently ungrounded: retrieval returns "No relevant context found" because the backend never constructs. The embeddings themselves are **healthy** — pgvector holds ~19,130 chunks across 4 workspaces at dim 2048 — so this is not a rebuild of the corpus; it is pointing prod at a **constructable** S3 Vectors backend and moving the existing, healthy vectors into it.

**The S8 probe finding (verbatim, the premise of this PRD):**
- Prod has `S3_VECTORS_ENABLED=true`; active backend = **s3_vectors**.
- `S3_VECTORS_BUCKET="automatos-ai"` — **missing the required `{workspace_id}` placeholder** → `S3VectorsBackend.__init__` raises `ValueError: S3_VECTORS_BUCKET must contain the '{workspace_id}' placeholder for per-workspace isolation`. The backend is **NOT constructable**. Probe verdict = **DARK** — the document-vector plane has been dark since S3 Vectors was switched on (~W2); agent answers over workspace documents have **not** been grounded via S3 Vectors.
- Forced onto pgvector, the same probe returns **LIVE**: pgvector holds **~19,130 embedded chunks across 4 workspaces, all at dimension 2048** (matches configured). The embeddings exist and are healthy; prod is simply pointed at the wrong (broken) backend.

**Why it stayed dark for weeks — the durable lesson this PRD encodes.** The F005 boot guard (`config.validate_security`, `config.py:1075-1088`) *does* raise on a placeholder-less bucket. But it runs inside `run_stage(DATABASE_INIT, …)` (`main.py:179,508`), and `run_stage` **catches every exception, marks the stage `failed`, and does not re-raise** (`core/models/bootstrap.py:127-136`). So the `RuntimeError` was logged, buried in the bootstrap report, and the server booted anyway — after which every `S3VectorsBackend()` raised *per request*, swallowed into an empty candidate list. A config error that should have been a loud boot abort became weeks of silent, ungrounded retrieval with zero dashboard signal. **The config-integrity guard (S2) is the real fix**: a placeholder-less bucket while enabled must fail LOUD — hard-abort boot, and fail CI — not fail quietly per-request.

**PILOT lens (locked):** this is **wiring**, not a usage push. The corpus is real and healthy; the plane being empty is a *config defect*, not cold-start. Scope is: make the backend constructable, make its misconfiguration loud, move the existing vectors in, and prove LIVE with the S8 probe. No moat framing; no new capability — the deliverable is that *the agent's answer over a client's documents is grounded again, and the platform can never silently lose that plane the same way twice.*

---

## Findings & Scope (all `file:line` confirmed by grep against the live tree; refreshed from the dossier where it drifted)

| Finding | Issue (verified in code) | Fix | Story |
|---|---|---|---|
| **P2-03 (config)** | Prod `S3_VECTORS_BUCKET` carries no `{workspace_id}` placeholder, so `S3VectorsBackend.__init__` raises (`s3_vectors_backend.py:51-55`) and the plane is DARK. The construction seam is `RAGService._get_s3_backend` (`modules/rag/service.py:941-956`) and the ingest gate is `get_document_manager` (`api/documents.py:77-86`) — both consume `config.S3_VECTORS_ENABLED`/`config.S3_VECTORS_BUCKET`. | Set the prod env's `S3_VECTORS_BUCKET` to a **templated** value carrying `{workspace_id}` (e.g. `automatos-vectors-{workspace_id}`). This is an **OPS/env change** (no `os.getenv` in code — the value already flows through `config.py:833`). Verified by S2 + S4. | **S1 (OPS)** |
| **P2-03 (durable fix)** | The F005 boot guard raises but is **swallowed** by `run_stage` (`bootstrap.py:127-136`; called at `main.py:179,508`) → dark-by-construction became silent-per-request for weeks. There is **no CI gate** and no *fatal* boot path for this class of misconfig. | Promote config-integrity to a **first-class guard**: a pure `config.assert_vector_config_integrity()` that raises on `S3_VECTORS_ENABLED=true` + placeholder-less/empty bucket, wired to **hard-abort boot** (outside the swallowing `run_stage`) **and** run as a CI check so the placeholder-less bucket can never ship green again. | **S2** |
| **P2-03 (dimension truth)** | `S3_VECTORS_DIMENSION` default is now **2048** (`config.py:835`) and matches the healthy pgvector corpus (dim 2048). But `_verify_or_recreate_index` only **logs** a dimension discrepancy (`s3_vectors_backend.py:118-131`); a fresh index created at a mismatched dimension would silently accept 2048-dim vectors into a wrong-dim index. | Assert index-vs-configured dimension agreement at construction/first-use and **fail loud** on mismatch (raise, not log), so relight can never populate a wrong-dimension index. Folds into the same integrity guard family as S2 (pure, testable). | **S3** |
| **P2-03 (re-embed)** | The existing vectors live in pgvector, not S3 Vectors. There is **no bucket-to-bucket mover**; the canonical path is a **re-embed from Postgres** via the existing `orchestrator/scripts/migrate_to_s3_vectors.py` (reads `documents`/`document_chunks`, re-runs extract→chunk→embed→S3 through `DocumentManager`, gated on `S3_VECTORS_ENABLED`). | **OPS step (Gerard runs against prod).** Dry-run, then run `migrate_to_s3_vectors.py` after S1's templated bucket is set. Do **not** write a new migrator — reuse this one. | **S4 (OPS)** |
| **P2-03 (verify LIVE)** | S8's probe (`scripts/probe_document_vectors.py`) is the accepted liveness oracle: it reports `constructable` + per-workspace `populated` + `dimension` and classifies `live|dark|degraded|unknown`. | **OPS step (Gerard runs against prod).** Re-run the S8 probe post-relight and confirm the plane flips to **LIVE at dim 2048** across the 4 populated workspaces. Attach the finding to close the gate. | **S5 (OPS)** |

---

## Stories (test-first — write the failing test, make it green, refactor)

> Two code stories carry **pure tests** and run in CI. Three stories are **OPS** (env change + re-embed + probe re-run) that Gerard runs against prod — clearly marked, **not CI tests**. The durable, code-gated fix is **S2** (with **S3** in the same guard family); S1/S4/S5 are the operational relight around it.

### S1 · Point prod at a constructable, per-workspace bucket — **OPS (env change), no code**

**What:** Change the deployed `S3_VECTORS_BUCKET` from the placeholder-less `automatos-ai` to a **templated** value that carries `{workspace_id}` — e.g. `automatos-vectors-{workspace_id}` (the shape the F005 error message and `config.py` both name). Every workspace then resolves to its own physical bucket, satisfying the F005 tenancy guard so `S3VectorsBackend.__init__` constructs.
**Files:** deploy environment only (Railway/prod env for the orchestrator). **No source edit** — the value already flows through `config.S3_VECTORS_BUCKET` (`config.py:833`); adding an `os.getenv` anywhere else would violate the config convention.
**Test:** none in CI (env change). Its correctness is **enforced by S2** (the integrity guard now rejects the old value) and **proven by S5** (the probe flips to LIVE). Sequenced **before** S4 (the re-embed writes into the templated buckets).
**Notes:** OPS-only, reversible. This is the literal "fix env" half of S8's DARK→relight branch. Coordinate with S4: re-embed only after the templated bucket is live, or vectors land in the wrong bucket. Dossier `vector-substrate.md` §C.1; report **P2-03**.

### S2 · Config-integrity guard — a placeholder-less bucket fails LOUD at boot and in CI — **S · _the durable fix_**

**What:** The reason the plane stayed dark for weeks is that the F005 `RuntimeError` was **swallowed** by `run_stage` (`bootstrap.py:127-136`) and the server booted anyway. Make the check un-swallowable and CI-visible:
1. Add a pure `config.assert_vector_config_integrity()` that raises a clear, typed error when `S3_VECTORS_ENABLED=true` **and** `S3_VECTORS_BUCKET` is empty or lacks `{workspace_id}` (reuse the exact F005 message shape already in `validate_security`, `config.py:1075-1088` — do not duplicate the string; extract the shared assertion so `validate_security` calls it too).
2. Wire it into boot **outside the swallowing `run_stage`** so a failure **hard-aborts** the process (a fail-closed retrieval plane must not serve traffic silently). The existing `validate_security()` call lives inside `run_stage(DATABASE_INIT)` (`main.py:179`); the integrity assertion must escalate a DATABASE_INIT `failed` status into an actual boot abort for this class (or run before `run_stage`), not merely record it in the report.
3. Add a **CI check** (a pure test asserting the guard's behaviour) so a placeholder-less-while-enabled config can never ship green again.
**Files:** `config.py` (extract `assert_vector_config_integrity()` from the F005 branch of `validate_security`, `:1075-1088`); `main.py` (`_boot_phase_1_core` `:162-179` / the `run_stage(DATABASE_INIT)` call `:508`) — make the integrity failure fatal, not swallowed; a new pure test module under `orchestrator/tests/`.
**Test:** `test_vector_config_integrity_rejects_placeholderless_bucket` asserts `assert_vector_config_integrity()` **raises** when enabled + bucket `="automatos-ai"` (today's prod value) and when the bucket is empty; `test_vector_config_integrity_accepts_templated_bucket` asserts it **passes** for `"automatos-vectors-{workspace_id}"`; `test_vector_config_integrity_noop_when_disabled` asserts it is silent when `S3_VECTORS_ENABLED=false` (open-core local). `test_boot_aborts_on_bad_vector_config` asserts the boot path treats the integrity failure as **fatal**, not a swallowed `failed` stage. All pure — mock/patch `config` values at the boundary; **no DB / network / AWS**.
**Notes:** This is the one-way ratchet that makes S1's mistake unrepeatable and turns S8's finding from "we caught it by probing" into "boot/CI catch it automatically." No backward-compat shim: the extracted assertion replaces the inline F005 branch, it does not live beside it. Dossier `vector-substrate.md` §C.1 + G.2 (config-integrity CI gate); report **P2-03** (and the durable half of **P2-16**'s "config-integrity CI gate").

### S3 · Fail loud on an index-vs-configured dimension mismatch — **S**

**What:** Relight must never populate a wrong-dimension index. Today `_verify_or_recreate_index` (`s3_vectors_backend.py:118-131`) only **logs** when an existing index's dimension differs from `config.S3_VECTORS_DIMENSION` (now 2048, `config.py:835`) — and `search`/`add_documents` proceed. Make a confirmed mismatch **raise** (typed) at first-use so 2048-dim vectors can't be silently written to, or queried against, a differently-dimensioned index. (It must never *delete/recreate* a populated index — that destroys stored vectors; the existing comment at `:119` already forbids deletion. Fail loud, don't mutate.)
**Files:** `orchestrator/modules/search/vector_store/backends/s3_vectors_backend.py:118-131` (raise on confirmed mismatch instead of log-and-continue); keep the "never delete" invariant. Consider extracting the pure comparison (`configured_dim`, `reported_dim`) → a small helper so it is unit-testable without boto3.
**Test:** `test_index_dimension_mismatch_raises` feeds a mocked `get_index` returning `dimension=4096` under `config.S3_VECTORS_DIMENSION=2048` and asserts a typed error is raised (not a warning log); `test_index_dimension_match_passes` returns `dimension=2048` and asserts no raise. Pure — mock the `s3vectors` client at the boundary; **no AWS**. (The 4096↔2048 confusion is exactly the historical dimension-knob incoherence the dossier flags — this test pins the correct value.)
**Notes:** Aligns the runtime with S8's `degraded` verdict (populated-but-wrong-dimension). Small, in the same "make misconfig loud" family as S2. Dossier `vector-substrate.md` §C.3 (four dimension knobs) + J1 (dimension); report **P2-03**.

### S4 · Re-embed the healthy pgvector corpus into S3 Vectors — **OPS (Gerard runs against prod), reuse the existing script**

**What:** With S1's templated bucket live and S2/S3's guards green, move the existing ~19,130 chunks (4 workspaces, dim 2048) into per-workspace S3 Vectors indexes by **re-embedding from Postgres** using the **existing** `orchestrator/scripts/migrate_to_s3_vectors.py`. That script already: validates `S3_VECTORS_ENABLED=true` (`:354-357`), enumerates S3-backed `documents` per workspace (`phase3`, `:177-205`), and reprocesses each through `DocumentManager._process_document` → S3 Vectors (`phase4`, `:208-333`), with `--dry-run` (`:341`) and `--doc-ids` (`:344`) modes. **Do not write a new migrator.**
**Files:** `orchestrator/scripts/migrate_to_s3_vectors.py` (run as-is). No code change expected; if the script needs any touch-up to target the templated bucket, that is a **separate, surfaced** change — do not silently fork it.
**Test / deliverable (OPS, not CI):** run `python -m scripts.migrate_to_s3_vectors --dry-run` first (confirm the per-workspace document counts match the probe's 4 populated workspaces / ~19,130 chunks), then run the real migration. Capture the processed/failed/skipped summary the script prints (`:399-413`).
**Notes:** OPS step against prod data — **Gerard's to run**, explicitly not a CI test (it needs the prod DB + AWS creds). Re-embed cost is trivial (~$0.08 one-time at qwen3-embedding-8b $0.01/M, per the dossier §H). Sequenced **after** S1 (templated bucket) and after S2/S3 land (so a misconfig can't corrupt the write). Dossier `vector-substrate.md` §C.1 ("reuse `migrate_to_s3_vectors.py`'s re-embed skeleton"); report **P2-03**.

### S5 · Re-run the S8 probe and confirm LIVE at dim 2048 — **OPS (Gerard runs against prod), closes the gate**

**What:** Re-run the read-only S8 probe (`orchestrator/scripts/probe_document_vectors.py`) against prod after S4 and confirm the plane flips **DARK → LIVE**: `constructable=true`, every populated workspace `populated=true` at `dimension=2048`, verdict `live`. This is the acceptance oracle for the whole PRD and the thing that formally closes the S8 gate (unblocking Wave-1 P2-07 / PRD-188).
**Files:** `orchestrator/scripts/probe_document_vectors.py` (run as-is: `python -m scripts.probe_document_vectors --all` for full coverage, or `--json` for a machine-readable finding). No code change.
**Test / deliverable (OPS, not CI):** the probe's own exit code is the gate — it returns `0` only on verdict `LIVE` (`:330-331`). Attach the `--json` finding (per-workspace `populated`/`dimension` + verdict `live`) to close P2-03. If it returns `degraded` (dimension mismatch), S3's guard should already have prevented the write — investigate before declaring done.
**Notes:** OPS step — **Gerard's to run** (needs prod env + AWS + DB), explicitly not CI. The probe already exists and is import-safe/read-only; **do not re-implement it**. This story is the difference between "we changed some config" and "grounding is proven live." Dossier `vector-substrate.md` J1; report **P2-03**.

---

## Sequencing

Strict spine (this PRD is mostly a pipeline, not parallel):

1. **S2 + S3 land first (code, CI-gated).** The integrity + dimension guards must be green *before* any prod motion, so the relight can't repeat the silent-dark failure or write a wrong-dimension index. S2 and S3 are independent code stories (different files) and can be built in parallel.
2. **S1 (OPS env change)** — set the templated bucket. Note: with S2 merged, a deploy still carrying `automatos-ai` will now **fail boot loud** (by design) — so S1 and the S2 deploy are coordinated: ship the templated bucket at or before the boot where S2's guard goes live.
3. **S4 (OPS re-embed)** — only after S1's templated bucket is live and S2/S3 are green. Dry-run, then real.
4. **S5 (OPS probe)** — after S4; the `live` verdict closes the gate.

The three OPS stories (S1, S4, S5) are **Gerard's to run against prod**; the two code stories (S2, S3) go through the normal PR/CI path.

---

## Verification (CI is the only code gate — no local runs)

Per project convention (`feedback-no-local-servers`): **do not run servers, builds, `next dev`, headless Chromium, `pytest`, `tsc`, or installs on the dev machine.** For the **code** stories (S2, S3): write the code + **pure** tests (no DB / network / AWS — mock `config`/the `s3vectors` client at the boundary), commit, push, and let **CI (the PR checks) verify.** CI is the only code gate.

The **OPS** stories (S1 env change, S4 re-embed, S5 probe re-run) are **operational actions Gerard runs against prod** — they are explicitly **not** CI tests and must not be simulated locally by the agent. S4 and S5 require the prod DB + AWS credentials the agent does not (and should not) have. Their "verification" is the re-embed summary (S4) and the probe's `LIVE` verdict / exit-0 (S5).

---

## Conventions (non-negotiable — see `CLAUDE.md`)

- **No `os.getenv()` outside `config.py`.** S1 is an *env value* change, not a code change; the `S3_VECTORS_BUCKET` read already lives at `config.py:833`. The integrity guard (S2) reads only `config.*`.
- **No new migrator, no new probe, no parallel seam.** Reuse `scripts/migrate_to_s3_vectors.py` (S4) and `scripts/probe_document_vectors.py` (S5) exactly; extend the existing `_get_s3_backend` construction seam, never a second one.
- **No backward-compat shim.** S2 *extracts* the F005 assertion so `validate_security` calls the shared function — it does not leave the inline branch and a duplicate living side by side.
- **Fail loud, never silent.** The whole PRD exists because a `RuntimeError` was swallowed; S2/S3 raise typed errors on misconfig — no `except: pass`, no log-and-continue on a config integrity or dimension mismatch.
- **Never delete a populated index.** S3 raises on a dimension mismatch; it must not delete/recreate (that destroys stored vectors — the `:119` invariant holds).
- **Immutable patterns; small focused functions.** The extracted `assert_vector_config_integrity()` and the pure dimension comparator are small and side-effect-free.
- Canonical vocab: **Deliverable**, **Knowledge Graph**, **Command Center**, **Auto**, **Playbook** (not Recipe). S3 Vectors and pgvector are the substrate; "the document-vector plane."
- Branch `feat/p2-s3-vectors-relight`; commit, push, open a PR; CI is the gate for S2/S3.

## Success metrics (the definition of "relit")

- **The active backend constructs.** `S3VectorsBackend.__init__` no longer raises for prod (templated bucket carrying `{workspace_id}`) — S1, proven by S5's `constructable=true`.
- **A placeholder-less-while-enabled config fails LOUD.** Boot hard-aborts and CI goes red on `S3_VECTORS_ENABLED=true` + `automatos-ai` — never again silent-per-request — S2.
- **No wrong-dimension writes.** An index-vs-configured dimension mismatch raises at first use, not logs — S3.
- **The corpus is in S3 Vectors.** The existing ~19,130 chunks (4 workspaces, dim 2048) are re-embedded via the existing migrator — S4.
- **The plane is proven LIVE.** The S8 probe returns verdict `live` at dim 2048 across the populated workspaces (exit 0) — S5, closing the P2-03 gate.

## What this gates

This PRD **completes Wave-0 P2-03** (the S8 decision gate) and thereby **gates Wave-1 P2-07 (PRD-188 — the RAG quality stack: rerank + contextual chunk annotations + BM25 leg + retrieval eval).** P2-07's entire premise is improving grounding quality; there is nothing to improve while retrieval returns empty. Grounding must be **proven live** (S5's `LIVE` verdict) before any RAG-quality work is worth doing — exactly the S8 gate condition. Do not start PRD-188 until S5 is green.

---

## Decision / open question — for Gerard (§12: surfaced, not resolved)

Two calls are Gerard's, and this PRD deliberately does **not** settle them.

**1. This PRD conflicts with Wave-3 P2-16 — S3 Vectors is being *relit*, but P2-16 plans to *retire* it.**
The review's Wave-3 item **P2-16** (`reports/PLATFORM_MODULE_DEEP_REVIEW_2026-07-04.md:221`) reads: *"Consolidate vector stores onto Qdrant + pgvector (**retire S3 Vectors**) + Qdrant snapshots + config-integrity CI gate."* The dossier's E.2 verdict is the same direction (REPLACE S3 Vectors with a Qdrant collection), primarily because it makes **open-core local RAG work** (compose defaults `S3_VECTORS_ENABLED=false` with no fallback store) and drops an engine.

By fixing S3 Vectors *properly* here, this PRD makes S3 Vectors the **healthy document backend for the SaaS deployment**. That does not kill P2-16, but it **rescopes** it. The neutral framing of the choice P2-16 now faces:
- **(a) Keep S3 Vectors for documents (SaaS), Qdrant for memory only.** P2-16 shrinks to: Qdrant snapshots (the C.7 backup gap is real and independent) + the config-integrity CI gate (**already delivered by S2 here**) + the open-core document-RAG story (Qdrant or pgvector as the *local-edition* document store, S3 Vectors staying the SaaS path). No re-migration of the just-relit SaaS plane.
- **(b) Still consolidate onto Qdrant later.** Treat this relight as the *stopgap* the dossier explicitly allows ("the S3 stay is defensible only as a stopgap"), and P2-16 later moves documents to Qdrant anyway — meaning the S4 re-embed here is thrown away in a few weeks.
This PRD assumes **(a)-shaped intent** (relight, don't fold) because that is the decision that spawned it — but whether P2-16 is *rescoped* (a) or *deferred-then-executed* (b) is **Gerard's call**, not this PRD's. Flagging so P2-16's PRD is written against the right premise; **not deferring or descoping P2-16 unilaterally.**

**2. The OPS re-embed (S4) is a go/no-go against prod — Gerard's to run.**
S4 re-embeds ~19,130 chunks in prod and S1 changes the deployed bucket env. These are production data/config operations the agent does not execute. The go/no-go — *when* to flip the bucket (S1) and run the re-embed (S4), and confirming the S5 probe reads `LIVE` — is **Gerard's**. If (b) above is chosen, Gerard may prefer to **skip S4 entirely** and jump to the Qdrant consolidation instead of re-embedding into a backend slated for retirement. The code stories (S2, S3) stand regardless of which way (1) goes — a loud config-integrity guard and a loud dimension check are correct under either backend.

---

*Traceability: P2-03 (Wave-0 gate) and its conflict with P2-16 (Wave-3) are from `reports/PLATFORM_MODULE_DEEP_REVIEW_2026-07-04.md` (§6, lines 190 & 221); every finding cites `reports/dossiers/vector-substrate.md` (§C.1/C.3, G.2, J1) and `reports/dossiers/rag-retrieval.md` (§C.2). The S8 finding is the live prod probe result from `orchestrator/scripts/probe_document_vectors.py` (PRD-185, merged `649482aa3`). All `file:line` refs confirmed by grep against the live tree on 2026-07-06 and refreshed where the dossier drifted (notably `S3_VECTORS_DIMENSION` default is now **2048**, `config.py:835`; F005 boot guard at `config.py:1075-1088`; `run_stage` swallow at `bootstrap.py:127-136`). North-Star framed; PILOT lens applied; no moat framing. Descoping of P2-16 is surfaced, not decided (§12).*
