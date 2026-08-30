# PRD-209: Phase 2 · Wave 4 — Fresh-clone boot & the honest-CI floor — P2-23

> **Status:** DRAFT — spec only, no build yet. Grounded @ `origin/main 9dd4c848a`; **re-confirmed @ `origin/main 182cd6739` (2026-08-29)** — entrypoint still `100644`, smoke lane still `continue-on-error: true`, compose still edition-blind, QUICKSTART still false. **Phase:** Phase 2 — Module Deep-Review remediation, **Wave 4** (deployability + topology hardening). **Adopted 2026-08-29 as OPEN-CORE PHASE 0** (owner decision: single codebase, Basic/SaaS co-exist — see `PRD-WAVE-OPEN-CORE.md`): stories S7–S9 added so a human's plain `docker compose up` — not just the CI smoke script — boots the local edition. **Depends on:** W6 (PRD-176 deployability), W12 (PRD-182 CI bar), W5 (PRD-175 auth decoupling — `AUTH_EDITION`, now landed), and **PRD-197 S5** (open-core local-RAG backend — the "serves far enough" hook; this PRD asserts the boot *reaches* it, it does not re-spec it; **landed** `9cfbf9005`).

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
- **The auth precondition is resolved — with one residual script bug.** The smoke lane's header says it is "blocked on W5"; but **W5 has landed** — `config.py:199-207` reads `AUTH_EDITION` (`local`⇒`REQUIRE_AUTH` forced false) and `validate_auth_edition()` is live. The smoke script sets `AUTH_EDITION=local` (`scripts/ci/smoke-fresh-clone.sh:41`) **but does NOT export `DEFAULT_WORKSPACE_ID`, which `validate_auth_edition()` hard-requires in `local`** — so even a fixed boot dies at the guard. Fix rides S5 (script) + S7 (compose defaults).
- **Compose is edition-blind — a human's `docker compose up` demands Clerk keys.** `docker-compose.yml` sets no `AUTH_EDITION`, no `DEFAULT_WORKSPACE_ID`, and uses no `env_file:` anywhere (verified @ 182cd6739: zero matches). The committed local defaults `envs/api.defaults` / `envs/frontend.defaults` / `envs/observability.defaults` have **no consumer** — and `api.defaults` still speaks the never-landed PRD-150 vocabulary (`EDITION=oss`, `AUTH_PROVIDER=local`) instead of the shipped flag (`AUTH_EDITION`). The backend therefore defaults to `saas` and `validate_auth_edition()` aborts on missing Clerk env. (→ S7)
- **QUICKSTART.md is false.** It claims "`docker-compose up --build` … That's it! No `.env` file needed" while compose hard-fails on three `:?`-guarded secrets (`POSTGRES_PASSWORD`, `REDIS_PASSWORD`, `API_KEY`) and, today, on Clerk env. First contact with the open-source promise is a lie. (→ S8)
- **Six drifted compose files shadow the canonical one.** `infrastructure/docker-compose{,.core,.data,.landing,.memory,.voice}.yml` (verified @ 182cd6739) predate the root stack; `.voice` references the decommissioned voice services (#625). PRD-153's cleanup, folded here. (→ S9)
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
| **OC-1 (open-core)** | Compose is edition-blind: no `AUTH_EDITION`, no `env_file`; `envs/*.defaults` unconsumed and speaking dead PRD-150 vocab → plain `docker compose up` demands Clerk. | Local edition becomes the compose default: fix `envs/api.defaults` to shipped vocab, wire `env_file:` + `DEFAULT_WORKSPACE_ID`. | **S7** |
| **OC-2 (open-core)** | `QUICKSTART.md` claims "no .env needed" — false on three `:?` secrets + Clerk. | Rewrite against the real flow; guard that quickstart env vars ⊆ documented set. | **S8** |
| **OC-3 (open-core)** | Six drifted `infrastructure/docker-compose*.yml` (incl. decommissioned voice) shadow the canonical stack. | Delete all six; guard asserts exactly one tracked compose file. (PRD-153's cleanup, absorbed.) | **S9** |

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
**Notes:** **Does not re-spec PRD-197 S5** — that PRD owns the local backend; this story only asserts the boot *reaches* it (197 S5 is now merged, `9cfbf9005`, so the probe has a live target). **Also owns the script bug:** export `DEFAULT_WORKSPACE_ID` in `scripts/ci/smoke-fresh-clone.sh` (the `validate_auth_edition()` local-mode requirement the script currently misses) and assert the entrypoint's seed step creates that workspace.

### S6 · One frontend lockfile (deterministic builds) — XS · _deployability F084_
**Files:** delete `frontend/yarn.lock` + `frontend/pnpm-lock.yaml` (keep `frontend/package-lock.json` — confirm the `frontend-ci` package manager at build, **Q5**); a guard.
**Test:** `test_single_frontend_lockfile` asserts exactly one lockfile is tracked under `frontend/` (source guard). Today: fails (three).
**Notes:** Only the lockfile relic belongs here (a fresh-clone determinism concern). The rest of the F084 cluster (`/api-control`, `/styleguide`, dead `api/anthropic_client.py`) is the un-authored PRD-184 dead-code kill-list — surfaced (**Q4**), not folded in.

### S7 · Local edition is the compose default — S · _open-core OC-1_
**Files:** `docker-compose.yml` (backend + frontend services gain `env_file:` pointing at `envs/api.defaults` / `envs/frontend.defaults`); `envs/api.defaults` (**fix to shipped vocabulary**: delete phantom `EDITION`/`AUTH_PROVIDER` keys, set `AUTH_EDITION=local`, `DEFAULT_WORKSPACE_ID` (**Q6**), keep `S3_VECTORS_ENABLED=false`, `REQUIRE_AUTH` handled by the flag); `envs/frontend.defaults` (`NEXT_PUBLIC_AUTH_EDITION=local`).
**Test:** a source guard asserts (a) `docker-compose.yml` declares `env_file` with `envs/api.defaults` on the backend service, (b) `envs/api.defaults` contains `AUTH_EDITION=local` + a `DEFAULT_WORKSPACE_ID`, and (c) contains **no** `EDITION=` / `AUTH_PROVIDER=` keys (dead vocab). The de-masked smoke lane (S3) then proves the behaviour end-to-end.
**Notes:** Compose is the *local* runbook — Railway never reads it, so making local the default costs SaaS nothing (per-key overrides remain via `.env` for anyone who wants a Clerk-mode compose). Per-service env precedence: `environment:` beats `env_file:` — keep the existing `:?` secrets in `environment:` and move edition/topology defaults to the file, so secrets stay explicit and defaults stay silent.

### S8 · QUICKSTART tells the truth — XS · _open-core OC-2_
**Files:** `QUICKSTART.md` (rewrite: the 3 required secrets, one optional LLM key, `docker compose up`, what you get — no login, local RAG, MinIO; troubleshooting pointers); `README.md` quickstart section aligned.
**Test:** a doc guard asserts every `${VAR:?}`-required variable in `docker-compose.yml` appears in `QUICKSTART.md` (parse both; today fails — QUICKSTART names none of the three).
**Notes:** The honest claim after S1–S7 is "3 secrets + 1 LLM key + `docker compose up`" — PRD-153's M1, finally true. Write against the merged reality, not aspiration.

### S9 · One canonical compose — XS · _open-core OC-3 / absorbs PRD-153_
**Files:** delete `infrastructure/docker-compose.yml`, `.core.yml`, `.data.yml`, `.landing.yml`, `.memory.yml`, `.voice.yml` (six files; `.voice` references services decommissioned by #625).
**Test:** `test_single_compose_file` asserts exactly one `docker-compose*.yml` is tracked repo-wide (source guard). Today: fails (seven).
**Notes:** No shims, no "kept for reference" — the canonical stack is the root file (§Conventions). Anything unique still referenced in `infrastructure/` copies gets folded into the root file first (expect nothing — they predate it).

---

## Sequencing

- **S1 + S2 are a coupled pair and land together.** The exec bit alone turns July's *silent schema drift* into a *hard boot abort* on the very fresh-clone path the wave opens — S2's clean head must be in the same PR (or land first). This is the single most important ordering constraint in the wave.
- **S3 (de-mask) lands after S1+S2 are green** — de-masking a known-broken boot just makes the lane honestly red (correct, but do it once the fix is in so the lane goes green, not red).
- **S4 (drift check) and S6 (lockfiles) are independent and parallel-safe.**
- **S5 depends on S3** (extends the same smoke assertion) **and on PRD-197 S5** (the backend it exercises — merged; the probe has a live target).
- **S7 lands with or immediately after S1+S2** — the compose defaults only matter once the container can exec and migrate; landing S7 first is harmless (the boot still dies at S1's exec bit) but the smoke lane can't go green until all three are in.
- **S8 is written last** (after S1+S2+S7 define the real flow) so the doc describes reality, not intent. **S9 is independent and parallel-safe.**

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
- **A human's plain `docker compose up` (3 secrets + optional LLM key, NO Clerk env) boots the local edition** — `env_file` consumed, `AUTH_EDITION=local` + `DEFAULT_WORKSPACE_ID` defaulted. Today: aborts demanding Clerk keys.
- **`QUICKSTART.md` names every required variable** (doc guard green). Today: claims none are needed.
- **Exactly one compose file tracked repo-wide.** Today: seven.

---

## Open questions — Gerard's call (§12)

1. **Stamp vs squash (S2).** Stamp `alembic_version` in `init_complete_schema.sql` (light; fixes the *compose* fresh-clone only, leaves the empty-DB `alembic-from-zero` lane still replaying) **or** the Step-2 single-baseline squash (heavier; fixes **both** lanes and satisfies the single-head invariant honestly). **Recommendation: squash** — the only path that makes both boot lanes honest. Confirm.
2. **Arm the floor — branch protection (F057 / deployability J1).** De-masking (S3) has teeth only when the lane is **required**. Flip `strict:true` and require `smoke-fresh-clone`, `alembic-from-zero`, and the S4 drift lane once green (ready command in `docs/runbooks/W12-BRANCH-PROTECTION.md`) — an owner repo-admin action. **Recommendation: require all three once green.** This is the highest-leverage-per-second action adjacent to the wave; do it here or note as owner follow-up?
3. **Converge schema truth to one writer (F051 / J5).** Retire `create_all`-at-boot + inline `ALTER`s in favour of alembic-only, now that S4 makes drift a red check. Larger separable refactor — do it in this wave or a follow-on? (Surfaced, not deferred.)
4. **PRD-184 dead-code kill-list.** The rest of the F084 cluster (`/api-control`, `/styleguide`, dead `api/anthropic_client.py`, `usePageAPI` no-ops) is un-authored. S6 takes only the lockfile relic; author 184 separately? (Owner call.)
5. **Lockfile winner (S6).** Keep `package-lock.json` (npm) — confirm the `frontend-ci` lane's package manager at build so the kept lockfile matches the CI installer.
6. **`DEFAULT_WORKSPACE_ID` convention (S7/S5).** A fixed well-known id shipped in `envs/api.defaults` (simplest; the entrypoint seed creates it idempotently) **or** seed-then-discover (entrypoint writes the id somewhere the config reads). **Recommendation: fixed well-known id** — one value, three consumers (defaults file, seed, smoke script), zero discovery machinery. Confirm the value/format against what `validate_auth_edition()` and the seed path expect (integer vs UUID — check `workspaces.id` type at build).

---

*Traceability: `reports/dossiers/deployability-open-core.md` (Defects 1–6; §J upgrade path J1/J2/J4/J5; F009/F010/F051/F057/F084/F092) and `reports/dossiers/thesis-T2-repo-topology.md` §6 (harden boundaries in-repo; "single compose bundle is the self-host shape"), under review id **P2-23** (`reports/PLATFORM_MODULE_DEEP_REVIEW_2026-07-04.md` §6 Wave 4, §5 relics). Reuses PRD-176/182 (deploy + CI scaffolding), PRD-175 (`AUTH_EDITION`, landed), and **PRD-197 S5** (local-RAG backend — the "serves far enough" hook, not re-spec'd here). All facts re-confirmed @ `origin/main 9dd4c848a` (entrypoint `100644`; worker entrypoint `100755`; 12 alembic heads / 158 versions; smoke + from-zero `continue-on-error`; three lockfiles; no schema-drift lane). `file:line` refs may have drifted — confirm by grep at build. North-Star framed (second-order/precondition); PILOT lens applied; no moat framing.*

*2026-08-29 addendum: re-confirmed @ `origin/main 182cd6739` — entrypoint `100644`; smoke lane `continue-on-error: true` (`smoke-fresh-clone.yml:51`); smoke script sets `AUTH_EDITION=local` (`:41`) but no `DEFAULT_WORKSPACE_ID`; compose has zero `env_file`/`AUTH_EDITION` matches; `envs/*.defaults` unconsumed, `api.defaults` speaks dead PRD-150 vocab; 168 files under `alembic/versions/`; six `infrastructure/docker-compose*.yml` tracked; QUICKSTART still claims no env needed. S7–S9 added under the open-core Phase-0 adoption (owner decision 2026-08-29; program doc `PRD-WAVE-OPEN-CORE.md`); S9 absorbs PRD-153's compose cleanup and S7 its US-002 defaults consumer — PRD-153 header updated accordingly.*

## S2 REVISION (2026-08-29 evening — from the first REAL local run)

The built S2 (stamped `init_complete_schema.sql`) shipped a **stale snapshot**: a live `docker compose up` produced a "healthy" instance with **107 of prod's ~152 tables** — `notifications`, `personas`, `channel_connections`, `marketplace_widgets` and ~40 more missing; every UI surface touching them 500'd. The from-zero Gate 2 stayed green because a stamp makes `upgrade heads` a no-op regardless of what the SQL created. A replay-from-empty spike then established the deeper fact: **the forest holds 41 revisions with `down_revision = None`** (hotfix-era orphan roots) whose cross-tree ordering makes a from-empty replay impossible without re-chaining them.

**Revised S2 (built, final):** fresh databases run `scripts/init_fresh_db.py` from the entrypoint → `scripts/generate_schema_baseline.build_schema()`: the model layer (`init_test_db.init_db()`: `create_all` + raw-DDL extras) **followed by a statement-tolerant replay of the migration forest** (each alembic op in a savepoint; a never-drop policy — models define the current shape, drops are history; a residual pass re-runs creators that lost an ordering race; the model layer re-asserted at the end). Both writers are required: ~8 core tables (`workspaces`, `chats`, `messages`, `system_settings`…) exist only as models, and ~50 (`notifications`, `deliverables`, `agent_skills`, `memory_items`…) exist only in migrations. Result on the live test: **149 tables** (the old snapshot gave 107; prod ≈152 — verify against prod `\dt` when convenient). **No schema snapshot is committed anywhere** — the generator IS the fresh path (boot and CI Gate 2 run the same code), so nothing can rot. `init_complete_schema.sql` is **deleted** everywhere (compose initdb mount, CI Gate 2, drift check, guards). One genuinely broken migration was repaired for fresh replays (`20250930_add_document_usage_tracking`: plain-string `server_default="'{}'::jsonb"` rendered as invalid JSON — now `sa.text(...)`; already applied on prod, so content-safe). The drift check now counts **models (`__tablename__`) as a first-class writer** — its baseline emptied, and it immediately caught three real orphans (`knowledge_items`, `tool_usage_logs`, `learning_outcomes` — live readers, no creator), now ported into `init_test_db` raw DDL. Constraint B is untouched: existing stamped databases never enter the fresh path.

**Also fixed by the live run:** the entrypoint's seed step was dead since PRD-176 (`python database/load_seed_data.py` — wrong path AND script-mode `sys.path` bug; now `python -m core.database.load_seed_data`, proven loading 418 credential types + models + skills); local boots fail closed if the `DEFAULT_WORKSPACE_ID` workspace can't be created; the frontend's 16-file Clerk-hook leak is closed by the `lib/auth-hooks.ts` edition seam; three widget-family models declared UUID FKs against `users.id` (Integer) — impossible FKs that crashed `create_all` on any fresh DB, aligned to the migrations' Integer.

**Recorded follow-on (owner-scheduled, not silently dropped):** re-chain the 41 orphan-root revisions (`depends_on`/lineage repair) so `alembic upgrade heads` replays from empty — that unlocks retiring `create_all`-at-boot (Q3's end state) and gives the drift check a migrations-built shadow to diff against.*
