# PRD-210: Phase 2 · Wave 4 — Arm the gates (CI teeth + branch protection) — P2-24

> **Status:** DRAFT — spec + runbook only, no build wave. **Grounded @ `origin/main 9dd4c848a`** (branch-protection state live-verified via `gh api` 2026-07-04, deployability C-defect-4; unchanged on this tree — re-confirmed during synthesis). This is the mostly-`[human]` repo-admin PRD: most of it is 30-second GitHub settings, not code.

**Framing (CLAUDE.md §3):** **Arming / hardening — enforce what already exists.** The CI lanes, the coverage script, the `.gitignore` line, and the ready branch-protection command are all in-tree and (mostly) green; nothing here is designed or built, it is *turned on*. Every action either flips a repo setting or lands a two-line CI-config change. **This is a runbook-shaped PRD by design — do not inflate it into a code project.**
**Build size:** **S**, and mostly *human* (3 of 5 stories are repo-admin / history actions only Gerard can take; 2 are tiny PRs a build agent opens). **Risk:** Low — every step only makes a merge *harder*; nothing widens access, loosens a gate, or touches runtime behaviour.

---

## Overview

This is the cheapest action in the entire Phase-2 program, and it is the one that gives **every other module's tests teeth.** Branch protection on `main` is `strict:false` with only two required contexts, so the whole W12 CI investment — the module-test collection, the frontend lane, CodeQL, gitleaks — *runs and reports* but **blocks nothing**. Judged against the **North Star** (*does this make Auto more autonomously capable / the agents' output higher-quality?*), this wave is **infrastructure, not intelligence**: it adds no capability. Its value is second-order and total — a required, honest CI floor is the mechanism by which every *other* wave's tests can actually stop a bad merge. **No moat framing; no new capability.** The deliverable is *enforcement*.

**Why now (the review's own root cause).** The July review attributes real red-main incidents to exactly this gap — a non-required frontend lane is what let the WSOD/red-main merges through, and `strict:false` is the stale-merge window behind them. CI teeth are what make the other 27 modules' tests mean something on a live merge instead of after it.

**PILOT lens (locked).** At pilot traffic the exposure window is small — the tracked JWTs were 60-second, and few merges happen — so this is not a fire *today*. But these gates must land **before the widget and channels go hot with real customers**: the day a real merge introduces a regression, a non-required lane is precisely the gap that lets it reach `main`. Arm the floor while it is cheap. (Quiet/empty CI at pilot is not a failure; the missing teeth are — see `feedback-pilot-usage-not-quality-signal`.)

**This PRD only *adds* enforcement — it loosens nothing.** The observability tier stays super-admin-only (PRD-143, deliberate); `strict:true` + required lanes only make merges harder, never widen a surface.

---

## Current reality (grounded @ `9dd4c848a`)

- **Branch protection on `main` does not bite (F057).** `strict:false`; required contexts only `["orchestrator-tests","ioc-scan"]`; `enforce_admins:false`; `required_approving_review_count:0`. So `orchestrator-module-tests`, `frontend-ci`, `codeql`, `gitleaks`, `alembic-from-zero`, and `smoke-fresh-clone` all **run but none block a merge**, and the stale-base merge window is open.
- **Most lanes are green-and-honest and can be required today:** `orchestrator-tests` (required; pytest + `--cov`), `orchestrator-module-tests` (collects the two formerly-orphaned test trees), `frontend-ci` (vitest hard gate + `tsc` baseline **554** + the route-contract step), `codeql` (python + js). Supply-chain CI runs green.
- **`gitleaks` (full-history, `fetch-depth:0`) is RED.** The tracked Clerk artifact sits in history; the lane cannot go green until it is purged (S2 + S3). So `gitleaks` **cannot** be required until then.
- **Two lanes are green-*while-broken*.** `smoke-fresh-clone` and the from-zero replay run under `continue-on-error`, reporting "success" while the fresh clone can't boot / the replay crashes on `marketplace_installs`. **De-masking + fixing the boot is the sibling Wave-4 PRD P2-23, not this one** — so these two lanes cannot honestly be required here.
- **The coverage ratchet is unarmed (F092 leg b).** `orchestrator/.coverage-baseline` holds the literal `SEED` token, so `orchestrator/scripts/check_coverage_baseline.py` (a step inside the already-required `orchestrator-tests` job) prints the number and exits 0 every run. The 80% doctrine is *measured, never enforced*.
- **The tracked Clerk artifact is exactly one path (F012).** Repo-root **`tests/e2e/.auth/user.json`** (blob `d342c5ad`) is tracked on live main; the `frontend/tests/e2e/.auth/user.json` variant **does not exist**. `.gitignore` names it at **lines 133–134** (`tests/e2e/.auth/`, `**/tests/e2e/.auth/`) — but the file was committed *before* the ignore, and **a gitignore is a no-op for an already-tracked file** (no `git rm --cached` was ever done). JWTs were 60-second → live-credential risk is low; the re-commit hazard + the red gitleaks lane are the open items.
- **The ready command is in-tree:** `docs/runbooks/W12-BRANCH-PROTECTION.md`.

---

## Findings → fix → story

| Finding | Issue (grounded @ `9dd4c848a`) | Fix | Story |
|---|---|---|---|
| **F057** / deployability C-defect-4 / Security §6.4 | `main` protection `strict:false`; required contexts only `["orchestrator-tests","ioc-scan"]`; `enforce_admins:false`. None of the W12 lanes block a merge; the stale-merge window is the named cause of the red-main/WSOD incidents. | Flip `strict:true` + `enforce_admins`; require the green-and-honest lanes. | **S1**, **S5** |
| **F012** / auth-identity C.7 / Security §6.1 | `tests/e2e/.auth/user.json` tracked despite `.gitignore` (no-op for a tracked file); `gitleaks` full-history lane red; re-commit hazard. | `git rm --cached` (untrack tip) + history purge + rotate. | **S2**, **S3** |
| **F092 leg b** / deployability C-defect-5 | Coverage ratchet unarmed — `.coverage-baseline` = `SEED`; the check prints and exits 0 every run. | Commit the measured floor (monotonic ratchet, never a hard 80% wall). | **S4** |
| *(cross-PRD dependency)* | `smoke-fresh-clone` + the from-zero replay are `continue-on-error`-masked green while the boot/replay is broken. | De-mask + fix boot = **P2-23**, not this PRD; it gates requiring those two lanes. | **S5** (gated on P2-23) |

---

## Stories

Each story is tagged by *who can do it*. That split is intrinsic (a PR cannot flip a GitHub setting or rewrite shared history), **not** a descope. Acceptance for code stories is a **pure** test; for human stories it is a verifiable repo end-state.

### S1 · Flip branch protection + require the green-and-honest lanes — **[human/repo-admin]** · XS
Using the ready command in `docs/runbooks/W12-BRANCH-PROTECTION.md`: set `strict:true`, `enforce_admins:true`, and expand required contexts to `["orchestrator-tests","ioc-scan","orchestrator-module-tests","frontend-ci","codeql"]`. **Do not** add `gitleaks` yet (red until S3) nor `alembic-from-zero`/`smoke-fresh-clone` yet (masked; P2-23) — requiring a red or masked lane would block every merge or enforce a lie.
**Acceptance (end-state):** a throwaway PR whose `orchestrator-module-tests` (or `frontend-ci`/`codeql`) run is red **can no longer be merged**; a PR based on a stale `main` must update before merge; `gh api repos/:owner/:repo/branches/main/protection` shows `strict:true`, `enforce_admins:true`, and the expanded required set.

### S2 · Untrack the Clerk artifact on tip — **[code/CI-config]** · XS
`git rm --cached tests/e2e/.auth/user.json` in a PR (file stays on disk for local e2e; it only leaves git). Confirm `.gitignore` already covers it (lines 133–134 — it does). Stops the re-commit hazard and is the precondition for `gitleaks` green; it does **not** alone clean history (gitleaks scans full history).
**Acceptance (pure / repo-state):** `git ls-files tests/e2e/.auth/user.json` is empty on the PR tip; a guard test asserts `git check-ignore tests/e2e/.auth/user.json` matches (a re-add is refused by `.gitignore`).

### S3 · Purge the artifact from history + rotate — **[human]** · XS
`git filter-repo` (or BFG) to strip blob `d342c5ad` (`tests/e2e/.auth/user.json`) across all refs; coordinated force-push; rotate anything that was ever a real credential (60-second JWTs → low risk, do it anyway). Human-only: rewrites shared history and force-pushes `main`.
**Acceptance (end-state):** `git rev-list --objects --all` no longer lists the blob; the full-history `gitleaks` lane is **green** on a fresh clone.

### S4 · Arm the coverage ratchet — **[code/CI-config]** · XS
Replace the `SEED` token in `orchestrator/.coverage-baseline` with the measured floor. The value comes from a **real CI coverage run** (never guessed) and its exact starting number is Gerard's call (§12, Q2). It is a **monotonic ratchet** — the lane fails only when coverage *drops below* the recorded floor; it is **never** a hard 80% wall that could red-gate the pilot.
**Acceptance (pure):** a synthetic PR that drops measured coverage below the committed floor **fails** the coverage step in the already-required `orchestrator-tests` job; a run at/above the floor exits 0; `check_coverage_baseline.py` reads the committed number, not `SEED`.

### S5 · Ratchet the required set upward as lanes go green — **[human/repo-admin]** · XS
The required set only ever grows. **After S3** greens `gitleaks`, add it as a required context. **After P2-23** de-masks and fixes the boot, add `alembic-from-zero` and `smoke-fresh-clone`. A lane joins the required set the moment it is green-and-honest, and never before.
**Acceptance (end-state):** `gitleaks` is a required, green, merge-blocking context (post-S3); the two boot lanes are added to the required set only once they pass without `continue-on-error` (post-P2-23), recorded as a dependency, not deferred.

---

## Sequencing

- **S1 lands first and independently** — `strict:true` + the today-green lanes; it does not wait on the artifact or coverage work.
- **S2 → S3** (untrack tip, then purge history) → **then S5 adds `gitleaks`** to the required set. `gitleaks`-required cannot precede S3 (it would block every merge while red).
- **S4** is independent; the floor number needs one real CI coverage measurement first.
- **S5's boot-lane step is gated on P2-23.** If P2-23 slips, S5 still ships the `gitleaks` addition; the boot lanes follow when P2-23 lands.
- **Human vs code:** S1/S3/S5 are repo-admin / history actions only Gerard can take; S2/S4 are PRs a build agent opens on `feat/p2-w4-arm-the-gates`. No code story can substitute for a setting flip or a history rewrite.

## Verification (CI is the only gate for the code parts — no local runs)

Per `feedback-no-local-servers`: the **code** stories (S2, S4) ship pure/repo-state tests and are verified by the PR checks — a `git check-ignore` assertion (S2) and a coverage-drop fixture (S4); no servers, builds, or local `pytest`. The **human** stories are verified by the resulting **repo settings and lane colour**, not a test: `gh api …/branches/main/protection` shows `strict:true` + the expanded set (S1/S5); a fresh-clone `gitleaks` run is green and the blob is gone from `git rev-list --objects --all` (S3). S1's real proof is behavioural — open a throwaway PR that fails a now-required lane and confirm the merge button is blocked. **The discipline of this whole PRD:** never require a lane that is red (blocks all merges) or `continue-on-error`-masked (a gate that lies) — a lane becomes required only when it is green-and-honest.

## Conventions (non-negotiable — see `automatos-ai/CLAUDE.md`)

- No code beyond the two tiny CI-config changes (untrack + baseline number). This is a runbook, not a build — do not inflate it.
- `git rm --cached`, never a delete — the fixture stays on disk for local e2e; it only leaves git.
- The coverage floor is a **ratchet**: only ever raised toward the 80% doctrine, never lowered to hide a regression (restore `SEED` to force a fresh measure).
- **Required contexts only ever grow** — a lane joins the required set the moment it is green-and-honest, and never before.
- Canonical vocab: **Command Center**, **Auto**, **Playbook**, **Deliverable**, **Knowledge Graph**.
- Adds enforcement only; respects PRD-143 (obs tier stays super-admin-only). Loosens nothing.

## Success metrics (the definition of "gates armed")

- **Branch protection on `main`: `strict:true`, `enforce_admins:true`** (S1).
- **The green-and-honest lanes are required and block a merge:** `orchestrator-tests`, `orchestrator-module-tests`, `frontend-ci` (route-contract included), `codeql`, and — post-S3 — `gitleaks` (S1/S5).
- **A PR that fails any required lane can no longer be merged; a stale-base PR must update first** (S1).
- **Coverage ratchet armed** — `.coverage-baseline` holds the measured floor (not `SEED`); a coverage drop red-gates the required job; the floor is monotonic (S4).
- **`gitleaks` green on a fresh full-history clone;** the Clerk artifact untracked on tip **and** purged from history; nothing re-commits it (S2/S3).
- **The required set is recorded as growing** to include `alembic-from-zero` + `smoke-fresh-clone` once **P2-23** de-masks and fixes the boot (S5, dependency on record).

## Open questions — Gerard's call (§12)

1. **Which lanes become blocking in S1's first turn of the key?** Recommendation: require `orchestrator-module-tests`, `frontend-ci`, `codeql` immediately (all green-and-honest today) on top of `orchestrator-tests`/`ioc-scan`. Also confirm: `enforce_admins:true` (recommended — admin-bypass is exactly how a red lane sneaks to `main`), and whether `required_approving_review_count` moves above 0 (recommended: leave 0 for a solo pilot — review count is a team control, not a CI-teeth control).
2. **Coverage floor starting value.** The first real coverage run is the floor. Set it exactly at measured, or measured-minus-a-small-margin to avoid flapping on non-deterministic lines? Recommendation: **measured − 1%** committed, ratchet up later.
3. **History purge now or scheduled?** S3 rewrites shared history and force-pushes `main` — disruptive across every clone/worktree/open PR (you run parallel sessions + worktrees). Do it in a quiet window now, or schedule it? Until it lands, `gitleaks` stays red and cannot be required. Recommendation: ship S1/S2/S4 first (arm the rest of the floor), schedule a single quiet-window purge for S3.
4. **The boot-lane requirement is gated on P2-23.** Confirm P2-23 (fresh-clone boot fix + de-mask) lands before S5's boot-lane step — or, if it slips, S5 ships with just the `gitleaks` addition and the boot lanes follow. (A recorded cross-PRD dependency, not a descope — §12.)

---

*Traceability: `reports/dossiers/deployability-open-core.md` (J1 branch-protection / J3 coverage-ratchet; C-defects 4 & 5; F057/F092), `reports/dossiers/auth-identity.md` (C.7 / J7; F012), and `reports/dossiers/security-hardening-appendix.md` (§6.4 branch protection / §6.1 secrets; backlog #10 & #16), under review id **P2-24** in `reports/PLATFORM_MODULE_DEEP_REVIEW_2026-07-04.md` §6 (Wave 4 — the honest-CI floor) + §7. Grounded @ `origin/main 9dd4c848a`; branch-protection state live-verified 2026-07-04 (unchanged on this tree); the tracked artifact re-confirmed via `git ls-tree origin/main -- tests/e2e/.auth/user.json` during synthesis (single path; the `frontend/` variant does not exist; `.gitignore` 133–134). Pairs with **P2-23** (fresh-clone boot fix — owns the two masked lanes). North-Star framed; PILOT lens; no moat framing; §12 — no unilateral descope (the boot-lane requirement is surfaced as a P2-23 dependency, not deferred).*
