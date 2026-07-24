# PRD-209: Phase 2 · Wave 4 — Fresh-clone boot & the honest-CI floor — P2-23

> **Status:** DRAFT — spec only, no build yet. Grounded @ `origin/main 9dd4c848a`. **Phase:** Phase 2 — Module Deep-Review remediation, **Wave 4** (deployability + topology hardening). **Depends on:** W6 (PRD-176 deployability), W12 (PRD-182 CI bar), W5 (PRD-175 auth decoupling — `AUTH_EDITION`, now landed), and **PRD-197 S5** (open-core local-RAG backend — the "serves far enough" hook; this PRD asserts the boot *reaches* it, it does not re-spec it).

---

## Framing (CLAUDE.md §3)

**Refactor / arming — right-shaped scaffolding is already here; arm the last mile.** W6/W12 built a nine-service compose stack, a fail-closed wait→migrate→seed entrypoint, a single-head invariant, a from-zero replay lane, and a fresh-clone smoke lane. None of it is net-new. The two things this scaffolding exists to guarantee — *a stranger can clone and boot* and *CI blocks a bad merge* — are both false on main today, each one small fix and one arming action from real. This wave makes them true. **Build size:** S–M (one mode change, one migration action, two CI lanes, one deletion — the alembic squash is the only M). **Risk:** Low–Medium (surgical; nothing touches runtime intelligence — the single Medium is the schema squash, fenced by a new drift check).

---

## Overview

Open-core is a **claim, not a fact, until a stranger can clone and boot** — and until the CI guarding `main` actually bites. Judged against the **North Star** (*does this make Auto more autonomously capable and the agents' output higher-quality for clients?*), this wave's value is honestly **second-order**: deployability is infrastructure, not intelligence. It makes Auto neither smarter nor the Deliverables better *directly*. Its leverage is that **a platform that cannot be stood up cannot be adopted, evaluated, or self-hosted by anyone but the owner on the one machine that already works**, and **CI that cannot block a bad merge is the mechanism by which every other Phase-2 module silently regresses** (the review's own root cause for "twelve alembic heads shipped undetected"). No moat framing; no new capability — the deliverable is that the one-command story stops lying.

**PILOT lens (locked):** this is a build-and-boot property, not a usage property. The running Railway prod is fine; there are no self-hosters yet, and that absence is **not** the problem to fix by "driving usage." In scope is the *wiring* that means, when an outsider does arrive with a fresh clone, `docker compose up` yields a running platform and the smoke lane proves it. See `feedback-pilot-usage-not-quality-signal`.

---

## Current reality (grounded @ `origin/main 9dd4c848a` — confirm `file:line` by grep at build; numbers drift)

- **The entrypoint is non-executable in git.** `docker-entrypoint.sh` (repo root) is tracked mode **`100644`** (verified: `git ls-tree origin/main -- docker-entrypoint.sh`). The Dockerfile sets `ENTRYPOINT ["docker-entrypoint.sh"]` and compose bind-mounts `./docker-entrypoint.sh` over the image's `chmod +x` stub (`docker-compose.yml:188`). The kernel cannot exec a non-executable file → the backend container dies at start *before the lifecycle runs*. (Contrast: `services/workspace-worker/entrypoint.sh` is tracked **`100755`** — the bit commits fine in this repo; the root script simply never got it.)
- **The schema head is not single.** 158 version files under `orchestrator/alembic/versions/`, but **12 leaf heads** on main (verified by AST leaf-count). The entrypoint runs `alembic upgrade heads` **fail-closed**; `init_complete_schema.sql` seeds the compose Postgres (`docker-compose.yml:35`) with **no `alembic_version` stamp** → a fresh volume must replay the whole forest. The `alembic-from-zero` lane's "assert exactly one head" gate is therefore **red today** (12 ≠ 1), and its replay step crashes on an ALTER-ed-but-never-CREATE-d table — both masked.
- **The smoke lane reports green while broken.** `.github/workflows/smoke-fresh-clone.yml` runs `scripts/ci/smoke-fresh-clone.sh` under **`continue-on-error: true`**; the `alembic-from-zero` replay step (`test.yml`, confirm line at build) is likewise `continue-on-error`. A boot death concludes **"success."**
- **The auth precondition is resolved.** The smoke lane's header says it is "blocked on W5"; but **W5 has landed** — `config.py:199-207` reads `AUTH_EDITION` (`local`⇒`REQUIRE_AUTH` forced false) and `validate_auth_edition()` is live. The smoke script already sets `AUTH_EDITION=local`. So the *only* remaining boot blockers are the exec-bit and the replay — de-masking is now unblocked.
- **No schema-drift check exists.** Grep finds only *route-manifest* drift wiring; nothing diffs the declared schema against the migration head (the "four writers of schema truth" — initdb SQL, alembic, `create_all` at boot, inline `ALTER`s — ship drift undetected).
- **Three frontend lockfiles are tracked** — `frontend/package-lock.json`, `frontend/pnpm-lock.yaml`, `frontend/yarn.lock` (verified) → nondeterministic fresh-clone builds.

---

## Findings → fix → story

| # | Finding (grounded @ 9dd4c848a) | Fix | Story |
|---|---|---|---|
| **F009 (boot)** | `docker-entrypoint.sh` tracked `100644` → container dies at exec before the lifecycle runs. | Commit the exec bit (`git update-index --chmod=+x`, mode → `100755`); add a tracked-mode guard. | **S1** |
| **F010/F051 (schema)** | 12 heads + unstamped initdb → fail-closed `alembic upgrade heads` on a fresh volume replays a forest that crashes. Fixing S1 alone converts silent drift into a **hard boot abort**. | Reach one clean head on a fresh clone: **stamp** `alembic_version` in initdb SQL **or** author the Step-2 single-baseline squash (Q1). Must land **with** S1. | **S2** |
| **F009 (mask)** | Smoke + from-zero lanes are `continue-on-error` → boot death reports green. | Drop `continue-on-error` on both so a boot/replay failure is **red**; assert the flag stays absent. | **S3** |
| **F051 / J4** | No lane catches models ↔ migration-head divergence (Supabase-style). | A CI lane that diffs the declared schema (or a fresh `create_all` dump) against the applied-migrations shadow DB; **red on drift**. | **S4** |
| **F009 (serve)** | The smoke lane asserts only `/health` 200 — not that the clone comes up far enough to *serve*. | Extend the assertion to a readiness probe that exercises PRD-197 S5's local-edition path (confirm, don't re-build). | **S5** |
| **F084** | Three tracked lockfiles → nondeterministic builds. | Delete two; keep the one the `frontend-ci` lane uses (Q5); assert exactly one remains. | **S6** |

---

## Stories (test-first — write the failing test/guard, make it green; CI is the only gate)

### S1 · Commit the entrypoint exec bit — XS · _deployability F009 / J2_
**Files:** `docker-entrypoint.sh` (mode `100644`→`100755`, a tracked mode change — no content edit); a guard under `orchestrator/tests/`.
**Test:** `test_entrypoint_is_executable` asserts the git-tracked mode of `docker-entrypoint.sh` is `100755` (source/mode guard, pure — reads `git ls-files -s`, no boot). Today: fails (`100644`).
**Notes:** Inert — and *harmful* — without S2: once executable, the fail-closed `alembic upgrade heads` runs on a fresh volume. Ship as the coupled pair (see Sequencing).

### S2 · One clean head for a fresh clone — M · _deployability F010/F051 / J2_
**Files:** `orchestrator/alembic/versions/` (a merge/baseline revision collapsing the 12 heads); **either** `orchestrator/core/database/init_complete_schema.sql` (add the `alembic_version` stamp) **or** the Step-2 single-baseline squash (Q1).
**Test:** the `alembic-from-zero` lane's "assert exactly one head" gate returns `1`, and the from-zero replay exits `0` from an empty pgvector DB (both currently red/masked). A pure unit guard asserts `len(heads)==1` by AST over the versions dir.
**Notes:** 12 heads on main today (not one). Stamp fixes the *compose* fresh-clone; the squash fixes *both* compose **and** the empty-DB from-zero lane and satisfies the single-head invariant honestly — recommend the squash, surface as **Q1**. No backward-compat: the merge revision is the new head, no `_legacy` parallel.

### S3 · De-mask the boot lanes (fail red on boot death) — S · _deployability F009 / J2_
**Files:** `.github/workflows/smoke-fresh-clone.yml` (drop `continue-on-error`); the `alembic-from-zero` replay step in `.github/workflows/test.yml` (drop `continue-on-error` — confirm line at build).
**Test:** a guard asserts `continue-on-error` is absent from the smoke boot step and the from-zero replay step (source-grep guard, matching the PRD-185 S5 shape — a workflow cannot be unit-run). Acceptance: a deliberately-broken boot concludes **failure**, not success.
**Notes:** De-masking has teeth only when the lane is also **required** in branch protection — an owner action (**Q2**). Sequence after S1+S2 land green, or the lane goes honestly red on the known-broken boot.

### S4 · Schema-drift CI check (Supabase-style diff) — M · _deployability J4 / F051_
**Files:** a new `.github/workflows/` job (or `scripts/ci/schema_drift_check.py`) that builds a shadow DB from applied migrations and diffs it against the declared `init_complete_schema.sql` (or a fresh `Base.metadata.create_all` dump); **red on drift**.
**Test:** `test_schema_drift_detects_divergent_column` feeds a fixture where a model column is absent from the migration head and asserts the check reports drift (pure — the diff logic runs against fixture DDL, no live DB).
**Notes:** This is the structural net for "four writers of schema truth" — the check that would have caught the ALTER-ed-but-never-CREATE-d table. Converging the four writers to one (retire `create_all`-at-boot + inline `ALTER`s) is the larger follow-on the check makes safe — surfaced as **Q3**, not silently deferred.

### S5 · Fresh clone comes up far enough to serve — S · _deployability F009 / reuses PRD-197 S5_
**Files:** `scripts/ci/smoke-fresh-clone.sh` (extend the assertion past `/health` 200 to a readiness probe that the local edition constructs a document backend, `AUTH_EDITION=local` / `S3_VECTORS_ENABLED=false`).
**Test:** the smoke script asserts a readiness signal (e.g. a `/health` payload field or a lightweight readiness route) that is true only when the local-edition RAG path constructs — the PRD-197 S5 hook ("The P2-23 fresh-clone lane confirms").
**Notes:** **Does not re-spec PRD-197 S5** — that PRD owns the local backend; this story only asserts the boot *reaches* it. If S5's backend is not yet merged, this asserts `/health` + records the readiness gap; it flips green when 197 S5 lands. Dependency, not duplication.

### S6 · One frontend lockfile (deterministic builds) — XS · _deployability F084_
**Files:** delete `frontend/yarn.lock` + `frontend/pnpm-lock.yaml` (keep `frontend/package-lock.json` — confirm the `frontend-ci` package manager at build, **Q5**); a guard.
**Test:** `test_single_frontend_lockfile` asserts exactly one lockfile is tracked under `frontend/` (source guard). Today: fails (three).
**Notes:** Only the lockfile relic belongs here (a fresh-clone determinism concern). The rest of the F084 cluster (`/api-control`, `/styleguide`, dead `api/anthropic_client.py`) is the un-authored PRD-184 dead-code kill-list — surfaced (**Q4**), not folded in.

---

## Sequencing

- **S1 + S2 are a coupled pair and land together.** The exec bit alone turns July's *silent schema drift* into a *hard boot abort* on the very fresh-clone path the wave opens — S2's clean head must be in the same PR (or land first). This is the single most important ordering constraint in the wave.
- **S3 (de-mask) lands after S1+S2 are green** — de-masking a known-broken boot just makes the lane honestly red (correct, but do it once the fix is in so the lane goes green, not red).
- **S4 (drift check) and S6 (lockfiles) are independent and parallel-safe.**
- **S5 depends on S3** (extends the same smoke assertion) **and on PRD-197 S5** (the backend it exercises) — it degrades gracefully to a `/health` assertion until 197 S5 merges.

---

## Verification (CI is the only gate — no local runs)

Per `feedback-no-local-servers`: **do not run `docker compose`, servers, `pytest`, or installs on the dev machine.** The mode change (S1), the migration (S2), the workflow edits (S3), the drift lane (S4), the smoke-script edit (S5), and the deletions (S6) are all committed and verified by **CI**. Workflow-level changes cannot be unit-run, so each ships with a **source/mode guard** (entrypoint mode, `continue-on-error` absence, single-head count, single-lockfile count) that *is* pure and runs in CI — the same shape as the PRD-185 S5 import-regression guard. The alembic and drift lanes are their own proof: they go green when S2 lands and red on the next divergent head.

---

## Conventions (non-negotiable — see `automatos-ai/CLAUDE.md`)

- No `os.getenv()` outside `config.py`; the `AUTH_EDITION` / `S3_VECTORS_ENABLED` edition switches are already canonical there — reuse, don't add inline reads.
- No backward-compat shims — the merge/baseline revision *is* the head; deleted lockfiles are gone, no `_legacy` twin.
- Immutable patterns; small focused guards; comprehensive error handling; no silent `except` swallows (this wave exists because two CI lanes swallowed their own red).
- No new tables; reuse the existing entrypoint, the existing smoke script, and the existing `alembic-from-zero` job — extend, don't fork.
- Canonical vocab: **Command Center**, **Auto**, **Playbook**, **Deliverable**, **Knowledge Graph**.
- Branch `feat/p2-w4-fresh-clone-boot-honest-ci` · worktree `automatos-ai-prd209`; commit, push, open a PR; CI is the gate.

## Success metrics (the definition of "boots honest, blocks honest")

- **A fresh `docker compose up`** (no external creds, `AUTH_EDITION=local`) **reaches `/health` 200 + the readiness probe**, on a **hard** (non-`continue-on-error`) lane. Today: **0 (masked green)** → **1**.
- **`docker-entrypoint.sh` tracked mode = `100755`**, asserted by a guard. Today: `100644`.
- **`alembic heads` = exactly 1**, and the from-zero replay exits 0 as a **de-masked** lane. Today: **12 heads**, replay masked red.
- **The schema-drift lane goes red on a divergent head/column** (catches the next ALTER-ed-but-never-CREATE-d table). Today: no such lane.
- **Exactly one frontend lockfile tracked.** Today: three.

---

## Open questions — Gerard's call (§12)

1. **Stamp vs squash (S2).** Stamp `alembic_version` in `init_complete_schema.sql` (light; fixes the *compose* fresh-clone only, leaves the empty-DB `alembic-from-zero` lane still replaying) **or** the Step-2 single-baseline squash (heavier; fixes **both** lanes and satisfies the single-head invariant honestly). **Recommendation: squash** — the only path that makes both boot lanes honest. Confirm.
2. **Arm the floor — branch protection (F057 / deployability J1).** De-masking (S3) has teeth only when the lane is **required**. Flip `strict:true` and require `smoke-fresh-clone`, `alembic-from-zero`, and the S4 drift lane once green (ready command in `docs/runbooks/W12-BRANCH-PROTECTION.md`) — an owner repo-admin action. **Recommendation: require all three once green.** This is the highest-leverage-per-second action adjacent to the wave; do it here or note as owner follow-up?
3. **Converge schema truth to one writer (F051 / J5).** Retire `create_all`-at-boot + inline `ALTER`s in favour of alembic-only, now that S4 makes drift a red check. Larger separable refactor — do it in this wave or a follow-on? (Surfaced, not deferred.)
4. **PRD-184 dead-code kill-list.** The rest of the F084 cluster (`/api-control`, `/styleguide`, dead `api/anthropic_client.py`, `usePageAPI` no-ops) is un-authored. S6 takes only the lockfile relic; author 184 separately? (Owner call.)
5. **Lockfile winner (S6).** Keep `package-lock.json` (npm) — confirm the `frontend-ci` lane's package manager at build so the kept lockfile matches the CI installer.

---

*Traceability: `reports/dossiers/deployability-open-core.md` (Defects 1–6; §J upgrade path J1/J2/J4/J5; F009/F010/F051/F057/F084/F092) and `reports/dossiers/thesis-T2-repo-topology.md` §6 (harden boundaries in-repo; "single compose bundle is the self-host shape"), under review id **P2-23** (`reports/PLATFORM_MODULE_DEEP_REVIEW_2026-07-04.md` §6 Wave 4, §5 relics). Reuses PRD-176/182 (deploy + CI scaffolding), PRD-175 (`AUTH_EDITION`, landed), and **PRD-197 S5** (local-RAG backend — the "serves far enough" hook, not re-spec'd here). All facts re-confirmed @ `origin/main 9dd4c848a` (entrypoint `100644`; worker entrypoint `100755`; 12 alembic heads / 158 versions; smoke + from-zero `continue-on-error`; three lockfiles; no schema-drift lane). `file:line` refs may have drifted — confirm by grep at build. North-Star framed (second-order/precondition); PILOT lens applied; no moat framing.*
