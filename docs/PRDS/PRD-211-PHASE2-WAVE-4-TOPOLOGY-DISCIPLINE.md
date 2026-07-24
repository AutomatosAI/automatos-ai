# PRD-211: Phase 2 · Wave 4 — In-repo topology discipline (import-linter contract) — P2-25

> **Status:** DRAFT — spec only, no build yet. **Grounded @ `origin/main 9dd4c848a`** (all `file:line`/path claims re-confirmed against that tree, not the working branch). **This PRD is thinner than the review row implied:** two of P2-25's three sub-items already shipped (see §0). What remains is the one load-bearing deliverable that does **not** exist yet — the `import-linter`/`grimp` CI contract that locks the healthy coupling in — plus deleting the dead mem0 residue the un-split left behind. Pairs with **PRD-184** (kill-list) which, by agreement, does **not** touch mem0 residue — this PRD owns it, to avoid duplication.

---

## 0. What's already done (P2-25 was three sub-items; two are shipped — do not re-spec, do not silently drop)

| P2-25 sub-item (as written in the review) | State on `main 9dd4c848a` | Evidence |
|---|---|---|
| `/api/tasks` under the policy gate | **DONE — stronger than gating: the lane was DELETED.** PRD-192 S6 removed the ungoverned direct-step ingress entirely (zero product callers, grep-proven). Router file gone; `main.py:83` carries the tombstone comment; no registration. | commit `44da83941` "feat(prd-192): S6 — delete the /api/tasks direct-step lane"; `git ls-tree origin/main -- orchestrator/api/tasks.py` = empty |
| `/api/tasks`-stays-gone guard | **DONE — a comprehensive guard already exists.** `orchestrator/tests/test_p2w2_tasks_lane_deleted.py` asserts router-file-deleted + main-no-register + manifest-clean + dead-frontend-client-gone + worker-sandbox-stays. | that file |
| Un-split mem0 (memory back in-process) | **DONE.** The live memory path no longer hits an external HTTP mem0 service — `orchestrator/modules/memory/durable_store.py` (`DurableMemoryStore`) is in-process Qdrant (PRD-187). **No mem0 files remain under `orchestrator/modules/memory/` at all.** | `git grep "MEM0_API_URL\|mem0_client\|httpx" origin/main -- orchestrator/modules/memory/` = 0 hits |

**Consequence for scope:** code-canvas **C7**'s concern ("the `/api/tasks` direct-step lane is ungoverned", F060) is **moot** — the dossier snapshot predates PRD-192 S6 and describes a lane that no longer exists. Memory **E1**'s architectural verdict (un-split) is satisfied; what E1 leaves behind is **dead residue to delete**, not a re-architecture. So P2-25 reduces to **one contract + one deletion + one verify.**

---

## 1. Framing & size

**Framing (CLAUDE.md §3):** **Refactor / hardening + subtraction** — no new capability. Add one CI contract that makes the *already-healthy* topology non-regressable; delete files that reference a service that no longer exists. **Build size: S–M** (one contract config + one CI lane + one source-grep guard + a bounded deletion). **Risk: Low** — the contract is authored to be **green on current `main`** (it locks the measured state, it does not demand a refactor to pass); the deletions are grep-proven-zero-importers before they land.

---

## 2. Overview

**North Star** (*does this make Auto more autonomously capable / agent output higher-quality?*): indirectly but durably — thesis **T2** measured that the platform is a **healthy modular monolith** (only **3.0%** true lateral peer-coupling between feature modules; **80.5%** of cross-module coupling is everyone sharing one kernel — ORM/DB-session/config/LLM-client) and its verdict is **stay a monolith, harden the boundaries in-repo**. That verdict is a claim until it is *enforced*. This PRD makes it enforceable: a compile-time contract that fails loud when a new lateral edge is introduced — the exact benefit a repo/service split would claim (enforced boundaries, controlled blast radius) at **zero runtime cost and with no silent network failure mode**. It protects every other module's quality by keeping the seams the review depends on from eroding under future work.

**No moat framing; no new user-facing capability.** **PILOT lens:** this is second-order plumbing — it changes nothing a pilot user sees; it protects the codebase they are piloting *on*. Empty/low counts are not in scope to "fix by driving usage."

**Not a duplicate of the route-manifest sweep.** PRD-195/207 already ship a boundary sweep (`test_route_manifest.py` + `dump_routes.py` + `authz_sweep_probe.py`) that guards **app-routes ↔ the committed route-manifest** — i.e. *routes*. The import-linter contract guards **module import coupling** — a different axis. They are complementary, not overlapping.

---

## 3. Current reality (grounded @ `main 9dd4c848a`)

- **No import-boundary contract exists.** No `import-linter`/`grimp` config anywhere (`git grep -li "importlinter\|import-linter\|grimp"` = 0), no `pyproject.toml`, no `setup.cfg`. Nothing stops a future PR from wiring `modules/rag` directly into `modules/nl2sql` and quietly raising the 3.0% lateral coupling. CI lanes today: `test.yml` (+ dedicated `check-shopify-isolation.yml`, `nl2sql-eval-scheduled.yml`, `smoke-fresh-clone.yml`, `gitleaks`, `codeql`, `malware-scan`) — the dedicated-isolation-lane pattern (`check-shopify-isolation.yml`) is the precedent this contract follows.
- **The feature-module set** (`orchestrator/modules/*`, the "independent" tier): `agents`, `attachments`, `codegraph`, `context`, `coordination`, `documents`, `evaluation`, `intake`, `knowledge`, `learning`, `memory`, `nl2sql`, `policy`, `rag`, `search`, `voice`, `widgets`, `workflows`. **`modules/tools`** is the tool-routing **aggregator** (T2: imports from 28 modules) — deliberately *not* in the independent set, so features may route through it. **Shared kernel** (`core/*`, `config`, `services`, `integrations`, `contracts`) and **aggregation layer** (`api`, `consumers`, `channels`, `jobs`) are importable by everyone.
- **Known-good lateral edges that already exist** (the 3.0% the contract must not falsely fail): e.g. `modules/rag` → `modules/search` ×5, `modules/rag` → `modules/knowledge` ×1 (T2 §3); `modules/context/sections/memory.py` → `modules/memory` (the MemorySection reader); `modules/agents` and `modules/workflows` → `modules/codegraph` (T2 §6.5 names codegraph's inbound as `api`/`tools`/`agents`/`workflows`). These are what forces the **strictness decision** in §10.
- **mem0 residue — 7 dead files still tracked, 0 code importers** (verified: the only references are the residue docs themselves + archival review snapshots `PRD-154`, `PLATFORM_DEEP_REVIEW_2026-06.md`, `dossiers/evidence/data/mem0.md`; no live import):
  `orchestrator/mem0_openapi.json` · `orchestrator/scripts/probe_mem0_endpoints.py` · `orchestrator/scripts/seed_mem0_user.py` · `scripts/test_mem0_railway.py` · `docs/PRDS/39-MEM0-MIGRATION-PRD.md` · `docs/PRDS/PRD-152-MEM0-INTERNAL-SERVICES-DECOUPLING.md` · `docs/memory-system/phase1-mem0-async-rollback.md`.

---

## 4. Findings → fix → story

| # | Finding (grounded) | Fix | Story |
|---|---|---|---|
| **T2 §6.2** | The 3.0% lateral-coupling that makes the monolith healthy is enforced by nothing; a future PR can raise it silently. | Author an `import-linter` **independence contract** over the feature-module set (kernel + `modules/tools`/`api` allowed), tuned green on current `main`; run it as a dedicated CI lane. | **S1** |
| **memory E1** | mem0 was un-split (PRD-187) but 7 dead files referencing the retired HTTP service are still tracked. | Grep-prove zero importers, **delete** all 7, add a source-grep guard that they (and any mem0 HTTP client) stay gone. | **S2** |
| **code-canvas C7 / Security §4.1** | The ungoverned `/api/tasks` ingress — closed by deletion (PRD-192 S6). | **Verify only** — the guard `test_p2w2_tasks_lane_deleted.py` already covers it; adopt it as P2-25's governance-ingress criterion, add nothing (re-speccing would duplicate). | **S3** |

---

## 5. Stories (test-first — write the failing check, make it green; CI is the only gate)

### S1 · The import-linter contract — M · _the load-bearing deliverable · T2 §6.2_
Add `import-linter` (pulls `grimp`; light, pure-Python) pinned in `orchestrator/requirements.txt`, a contract config at **`orchestrator/.importlinter`** (import-linter's native default; run with `grimp` root_packages = the app's top-level packages `modules`,`core`,`api`,`config`,`services`,… — the same sys.path the app boots on, so run **from `orchestrator/`**), and a dedicated non-required CI lane **`.github/workflows/import-linter.yml`** (mirrors `check-shopify-isolation.yml`) that runs `lint-imports --config orchestrator/.importlinter`.

The contract is an **`independence`** contract listing the 18 feature modules (§3) — none may import another. `modules/tools` and `api` are **excluded** from the list (so features route through them — the "except through `modules/tools` or `api`" allowance); the shared kernel is allowed because it is not in the list. The exact rule, in words: **"no feature module (`modules/*`) may import another feature module except through `modules/tools` or `api`."** As config:

```ini
# orchestrator/.importlinter  (run from orchestrator/ so grimp sees the app's sys.path)
[importlinter]
root_packages = modules core api config services consumers channels jobs integrations contracts

[importlinter:contract:feature-module-independence]
name = Feature modules stay laterally decoupled (T2: lock the 3.0%)
type = independence
modules =
    modules.agents
    modules.rag
    modules.memory
    modules.nl2sql
    modules.search
    modules.knowledge
    modules.context
    modules.documents
    modules.codegraph
    modules.coordination
    modules.evaluation
    modules.intake
    modules.learning
    modules.policy
    modules.voice
    modules.widgets
    modules.workflows
    modules.attachments
# modules.tools and api are NOT listed → the permitted routing layer.
# NOTE: modules.learning + modules.evaluation are PRD-184 S1 kill targets — once 184 deletes them,
#       drop them from this list (a contract cannot reference a deleted package). Sequence 184 first.
# ignore_imports = today's measured known-good edges (re-traced on main at build).
ignore_imports =
    modules.rag -> modules.search
    modules.context.sections.memory -> modules.memory
    # …the rest of the ~91 current lateral edges, enumerated at build (§10-Q1)
```

The currently-measured known-good lateral edges (§3) are carried as explicit `ignore_imports` exceptions so the contract is **green on `9dd4c848a`** — i.e. it **ratchets from the measured state**, forbidding the *next* new lateral edge rather than demanding an immediate refactor. Strictness (ratchet-from-current vs. refactor-to-zero) and the `codegraph` named exception are Gerard's calls (§10).
**Files:** `orchestrator/.importlinter` (new), `orchestrator/requirements.txt`, `.github/workflows/import-linter.yml` (new).
**Test:** the CI lane itself is the check — `lint-imports` exits 0 on `main` (green baseline) and **non-zero** when a deliberate `modules/rag`→`modules/nl2sql` edge is introduced (prove the fixture in the PR description; do not commit the violating edge). Optional thin `orchestrator/tests/test_import_contract_present.py` asserts the config file exists and parses (pure, no network).
**Notes:** Config file only — no `os.getenv`, no code path changed. Keep the contract in the canonical config file, not scattered. Complementary to the route-manifest sweep (§2), not a replacement. **Cross-PRD:** `modules.learning` + `modules.evaluation` are deletion targets in **PRD-184 S1** — if 184 lands first, omit them from the module list; if this PRD lands first, 184 drops them from the contract in its deletion commit. Re-trace the live module set at build regardless.

### S2 · Delete the dead mem0 residue + guard — S · _memory E1 · subtraction_
Grep-prove **zero importers** of each of the 7 files (§3) on `main`, then **delete all 7** in one commit. Add `orchestrator/tests/test_no_mem0_residue.py` (source-grep guard, PRD-185 S5 / PRD-197 S1 shape) asserting: (a) none of the 7 paths exist; (b) no live HTTP mem0 client returns under `orchestrator/modules/memory/` (`MEM0_API_URL`/`mem0_client`/`httpx` grep = 0) — locking the un-split. Repoint/scrub any in-repo doc reference in the **same commit** the file moves.
**Files:** delete the 7 listed; add `orchestrator/tests/test_no_mem0_residue.py`.
**Test:** `test_no_mem0_residue` (pure, filesystem + source-grep only).
**Notes:** The only surviving references are archival review snapshots (`PRD-154`, `PLATFORM_DEEP_REVIEW_2026-06.md`, `dossiers/evidence/data/mem0.md`) — evidence of history, not live wiring. Whether to also scrub those citations, and whether the three stale mem0 **PRD/doc** files are deleted vs archived, is §10-Q3. **PRD-184 (kill-list) does not touch mem0 residue — this story owns it.**

### S3 · `/api/tasks` stays gone — XS (verify-only) — · _code-canvas C7 / Security §4.1_
The guard **already exists and is comprehensive** (`test_p2w2_tasks_lane_deleted.py`). This story adds **no new test**; it records that guard as P2-25's governed-ingress success criterion and confirms it is green on `main`. Extend it **only** if a coverage gap is found (none at authoring — it already pins router-deleted, main-no-register, manifest-clean, frontend-client-gone, worker-stays).
**Files:** none (verify-only).
**Test:** existing `orchestrator/tests/test_p2w2_tasks_lane_deleted.py`.
**Notes:** Honest scope — the ungoverned-ingress risk is *closed by deletion*, not gated; nothing to build.

---

## 6. Sequencing
- **S1, S2, S3 are independent and parallel-safe** (disjoint files). S3 is a no-op verify.
- S1's config authoring is the long pole (enumerating the current known-good edges accurately — re-trace on `main` at build, do **not** take the edge list from this doc as final; the §3 list is illustrative).
- No migration, no data motion, no route change → the committed route-manifest is untouched.

## 7. Verification (CI is the only gate — no local runs)
Per `feedback-no-local-servers`: author the contract + pure tests, push, let CI verify. The **import-linter lane runs in CI** (dedicated workflow, non-required at first — flipping it to required in branch protection is the repo-admin's call, tracked under **P2-24**, not this PRD). Deletions are **grep-proven zero-importers** before landing; the source-grep guard is **repointed in the same commit** the symbol/file moves (the PRD-185 S5 discipline). No test touches DB/network/Qdrant/mem0.

## 8. Conventions (non-negotiable — see `automatos-ai/CLAUDE.md`)
- No `os.getenv()` outside `config.py` (this PRD adds none — the contract is a static config file).
- **No backward-compat shims** — the 7 residue files are deleted, not `_legacy`-suffixed; delete what you replace in the same commit.
- Many small files; the contract lives in one canonical config file, not scattered per-module.
- Canonical vocab: **Playbook**, **Deliverable**, **Knowledge Graph**, **Command Center**, **Auto**.
- Branch `feat/p2-w4-topology-discipline`; commit, push, open a PR; CI is the gate.

## 9. Success metrics
- **The import-linter contract lane is live and green on `main`** — and demonstrably **red** on an injected lateral edge (S1).
- **Lateral feature↔feature coupling is provably ≤ the locked bound** — no new lateral edge can merge silently; the 3.0% is a floor the contract defends (S1).
- **mem0 residue = 0 files / 0 importers**, with a guard that keeps it that way and keeps the un-split (no HTTP mem0 client returns) (S2).
- **The `/api/tasks` guard is green** — the ungoverned ingress stays deleted (S3).

## 10. Open questions — Gerard's call (§12 — surfaced, not deferred)
1. **Contract strictness.** Ship the contract as a **ratchet** (enumerate today's ~91 known-good lateral edges as `ignore_imports`, forbid only *new* ones — green immediately, zero refactor) **[recommended]**, or as **clean-to-zero** (refactor the existing feature→feature edges to route through `modules/tools`/`api` first, then a stricter contract)? The latter is real work on `modules/context`→`modules/memory`, `modules/rag`→`modules/search`, `agents`/`workflows`→`codegraph`.
2. **`codegraph` explicit exception.** T2 names `codegraph` the one pre-cleared clean seam (zero feature deps out; inbound via `api`/`tools`/`agents`/`workflows`). Grant `agents`/`workflows`→`codegraph` a **permanent named exception** in the contract, or hold it to the same ratchet as everything else?
3. **Stale mem0 docs — delete or archive?** The 3 doc files (`39-MEM0-MIGRATION-PRD.md`, `PRD-152-…`, `phase1-mem0-async-rollback.md`) and the incidental `mem0_openapi.json` citations in archival reviews (`PRD-154`, `PLATFORM_DEEP_REVIEW_2026-06.md`, `dossiers/…/mem0.md`): hard-delete (this PRD's default for the 3 residue docs) or move to a `docs/archive/`? Recommendation: delete the 3 residue docs; leave the archival review snapshots' citations as historical record.

---

*Traceability: thesis **T2 §6.2** (in-repo boundary hardening — the `import-linter`/grimp contract; the AST-measured 3.0% lateral / 80.5% shared-kernel finding), memory **E1** (un-split — done PRD-187; residue owned here), code-canvas **C7** (ungoverned `/api/tasks` — closed by PRD-192 S6 deletion), under review id **P2-25** (`reports/PLATFORM_MODULE_DEEP_REVIEW_2026-07-04.md` §6 Wave 4, §3 T2 verdict; Security §4.1). All paths/commits re-confirmed @ `origin/main 9dd4c848a`. Complements the PRD-195/207 route-manifest sweep (routes) on the coupling axis; pairs with PRD-184 (kill-list, which excludes mem0 residue) and P2-24 (branch-protection arming that would flip this lane to required). North-Star framed; PILOT lens; no moat framing.*
