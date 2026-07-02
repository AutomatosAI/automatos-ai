# PRD-173 — Secret & Supply-Chain Hygiene (Wave 3)

**Status:** Draft v1 — pending approval
**Type:** Security / Supply-chain hygiene
**Priority:** P0 — committed-secret + unauthenticated-service + supply-chain closure
**Owner:** Gerard Kavanagh
**Author:** Gerard Kavanagh + Claude (Opus 4.8)
**Date:** 2026-07-02
**Phase:** A — Coherence · **Size:** M · **Risk:** medium (**git history rewrite**)
**Parent:** [PLATFORM-OS-ROADMAP.md](./PLATFORM-OS-ROADMAP.md)
**Source:** review §4 (F011, F012, F058, F092 — all ADJUSTED), §13 Wave 3, §13 Security pillar, §14 (production unknowns)
**Findings register pinned to:** `37fdecc4e` — re-confirm each `file:line` on current `main` before editing
**Findings in scope:** F011, F012, F058, F092 (the SAST/dependabot/gitleaks slice only — see §4.5 scope note)
**Related repos:** `automatos-ai` (main), `automatos-mem0` (the mem0 fork — F011), `automatos-shopify` / `workspace.settings` (F058)

---

## Operating Principle

> **A secret in history is a secret leaked; an unauthenticated service is an open door; an unaudited
> dependency is an unlocked one.** This wave does not add features — it closes the three ways the platform
> currently leaks trust: a credential artifact committed to git history, a memory server that is
> unauthenticated if built from the wrong SHA, and a merchant token stored in plaintext while its docstring
> claims encryption. It also installs the standing supply-chain lanes (SAST, dependency audit, secret
> scanning) that would have caught all three before they shipped. Every fix here is *hygiene made
> permanent* — a scan that stays green, a probe that stays failing-closed, an encryption path that stays on.

---

## 1. Purpose

The enterprise-ready Security bar (roadmap §2, review §13) reads: **"Zero unauthenticated endpoints touching
tenant data; gitleaks-clean history; merchant/API tokens encrypted at rest; SAST + dependabot + gitleaks
lanes green; third-party tool and MCP output treated as untrusted input."** Wave 3 owns four of the gaps that
keep that bar failing:

- **Committed Clerk auth artifact (F012).** `tests/e2e/.auth/user.json` is git-tracked and not gitignored. It
  holds two `__session` JWTs plus a `__clerk_db_jwt` dev-browser token. Verification *refuted the headline* —
  these are 60-second Clerk tokens that expired **2026-02-17**, so this is a committed-artifact and
  re-commit hazard, **not** a live-to-2027 credential — but a secret in history is still a secret, and the
  file will be re-committed every test run until it is gitignored.
- **Plaintext 147-scope Shopify Admin token (F058).** `orchestrator/api/shopify.py` stores the merchant
  Shopify Admin access token (147 write-scopes) as **plaintext** in `workspace.settings.shopify_access_token`
  while the docstring claims it is "encrypted at rest via database-level encryption." The docstring lies; the
  bytes are clear.
- **Unauthenticated memory server if built from fork main (F011).** The mem0 fork's PRD-156 token auth and
  PRD-159 metadata preservation exist **only** on branch `fix/pool-exhaustion@16b27eb2`, which is not an
  ancestor of the fork's `origin/main`. If the deployed Railway image builds from fork main, the OpenMemory
  server is unauthenticated — the exact PRD-156 hole — and silently drops the typed metadata the
  orchestrator's taxonomy filtering assumes.
- **No SAST or gitleaks lanes (F092).** CodeQL is unconfigured, there is no `dependabot.yml`, and there is no
  secret-scanning job in CI. GitHub-native secret scanning **is** enabled (a real mitigation the review
  credits), but nothing runs in the pipeline to catch the next committed secret or vulnerable dependency
  before merge.

Wave 3 has **no dependencies** and runs in parallel with W1 (spine) and W2 (tenant isolation). Its work is
mechanical and largely Ralph-able (roadmap §5), with one operationally careful step — the F012 history purge
is a coordinated force-push. **Closing these four flips three of the Security-pillar sub-bars** —
gitleaks-clean history, tokens encrypted at rest, SAST + dependabot + gitleaks lanes green — and removes the
"unauthenticated endpoint touching tenant data" that F011 would create if the Railway image builds from the
wrong SHA. (The remaining Security sub-bar — zero unauthenticated *orchestrator* endpoints, F003/F007/F039/F045
— is W2's, not this wave's.)

---

## 2. Background

### 2.1 What's working today (must not break)

- **GitHub-native secret scanning is enabled.** The review credits this as a real mitigation — it is *why*
  F012 is a re-commit hazard rather than an active breach path going forward. W3 adds gitleaks in CI on top of
  it (defence in depth and history coverage), it does not replace it. Do not disable the native scanner.
- **The mem0 security patch already exists — it is just on the wrong branch.** PRD-156 token auth on every
  OpenMemory router and PRD-159 metadata preservation are **written, reviewed, and committed** at
  `16b27eb2` on `fix/pool-exhaustion`. W3 does not author new auth; it merges proven work to fork main and
  pins the deploy. (Confirmed: `16b27eb2` is not an ancestor of `automatos-mem0/origin/main`; it is reachable
  only from `fix/pool-exhaustion`.)
- **The Shopify connect endpoint works end-to-end.** `POST /api/shopify/connect` correctly finds/creates the
  workspace, persists the token, and hands it to Composio. The store path is sound; only the *at-rest
  protection* of the token is missing. W3 encrypts on write and decrypts on read — it does not re-plumb the
  connect flow.
- **The e2e auth fixture is functional.** `tests/e2e/.auth/user.json` is how the dev-browser suite
  authenticates. W3 keeps the fixture *mechanism*; it stops the artifact being tracked and purges the leaked
  blob — the file is regenerated locally per run, never committed.

### 2.2 What's broken / blocked

- **F012 — a credential artifact lives in git history and re-commits every run**
  (`tests/e2e/.auth/user.json`, not in `.gitignore`, git-tracked — both confirmed on current `main`). The
  committed JWTs are expired (2026-02-17), so this is a *re-commit hazard and hygiene failure*, not a live
  breach — but the dev-browser client should be revoked regardless, the blob purged from history, and CI must
  gain secret scanning so the next one is caught at the PR.
- **F058 — a 147-scope merchant token is stored in plaintext while claiming encryption**
  (`orchestrator/api/shopify.py`, the `POST /connect` handler). The docstring states "encrypted at rest via
  database-level encryption"; the handler writes `settings["shopify_access_token"] = request.access_token`
  verbatim. A docstring is not encryption. Any DB read, backup, or `workspace.settings` dump exposes a
  full-write Admin token.
- **F011 — the memory server is unauthenticated if the Railway image builds from fork main** (mem0 fork,
  patches only on `fix/pool-exhaustion@16b27eb2`). Two coupled failures: (1) no router token auth → the exact
  PRD-156 hole, an unauthenticated service that reads/writes tenant memory; (2) dropped typed metadata → the
  orchestrator's taxonomy filtering silently returns wrong recall. Latent-or-live depending on Railway config
  only the owner can read (§14) — the fix is correct regardless.
- **F092 (W3 slice) — no SAST, no dependency audit, no CI secret scanning.** CodeQL unconfigured, no
  `dependabot.yml` (confirmed absent on current `main`), no gitleaks job (confirmed absent). Nothing in the
  pipeline would catch the next committed secret (F012-class) or a known-vulnerable dependency before it
  merges.

### 2.3 Why now

W3 has **no dependencies** and is startable immediately alongside W1/W2. Three reasons it should not wait:

1. **F012 re-commits on every dev-browser run.** Until `tests/e2e/.auth/` is gitignored, each test run
   re-stages the artifact — the hazard compounds with time and the purge gets more expensive with every
   commit added on top.
2. **F011 is latent-or-live *now*.** If the current Railway mem0 image is built from fork main, an
   unauthenticated tenant-memory service is live in production today. The fix (merge + pin SHA + boot probe)
   is correct whether or not that config is confirmed (§14) — waiting only prolongs the exposure window.
3. **The supply-chain lanes are the standing guard for every later wave.** W4–W14 all add code; SAST +
   dependabot + gitleaks are what keep the next committed secret or vulnerable dependency out of the tree.
   Installing them in Phase A means they guard the largest, riskiest waves (W4 policy plane, W5 auth
   decoupling) from day one.

This wave closes the review's Security-pillar hygiene items; it is orthogonal to (and parallelisable with)
the tenancy work in W2.

---

## 3. Findings in scope

Register pinned to `37fdecc4e` — re-confirm each `file:line` on current `main` before editing. (`.auth/user.json`
tracked + un-gitignored, `dependabot.yml`/CodeQL/gitleaks absent, and `shopify.py` plaintext write on line 368
under the encryption-claiming docstring on lines 359–360 were all re-confirmed on `main` during authoring.)

| ID | Severity | Location (pinned `37fdecc4e`) | Defect | Fix |
|---|---|---|---|---|
| **F012** | Medium (ADJUSTED from Critical) | `tests/e2e/.auth/user.json` (git-tracked, not in `.gitignore`) | Committed Clerk auth artifact: two `__session` JWTs + a `__clerk_db_jwt` dev-browser token. JWTs are 60s tokens **expired 2026-02-17** — re-commit hazard, not a live credential | gitignore `tests/e2e/.auth/`; purge the blob from history; revoke the dev-browser client; add secret scanning (gitleaks) to CI regardless |
| **F058** | High | `orchestrator/api/shopify.py` — `POST /connect` handler (docstring claims encryption ~L359–360; plaintext write ~L368) | Merchant Shopify Admin token (147 write-scopes) stored **plaintext** in `workspace.settings.shopify_access_token`; docstring claims "encrypted at rest" | Encrypt the token at rest via the canonical encryption path on write; decrypt on read; make the docstring true |
| **F011** | High (ADJUSTED) | `automatos-mem0` fork — patches only on `fix/pool-exhaustion@16b27eb2` (**not** an ancestor of fork `origin/main`) | If Railway builds from fork main: OpenMemory server is unauthenticated (the PRD-156 hole) **and** drops typed metadata the orchestrator's taxonomy filtering assumes | Merge `fix/pool-exhaustion` → fork main; pin the Railway service to a verified SHA; rebuild; add a boot probe asserting 401-without-token |
| **F092** (W3 slice) | Medium (ADJUSTED) | `.github/workflows/test.yml:99-100`; **no** `.github/dependabot.yml`; **no** CodeQL/gitleaks workflow | No SAST (CodeQL unconfigured), no dependency audit (`dependabot.yml` absent), no CI secret scanning. GitHub-native secret scanning **is** enabled (real mitigation) | Add **CodeQL (SAST)** + **`dependabot.yml`** lanes, plus a **gitleaks** CI job (gitleaks also lands here via F012). *Migration-replay → W6; measured coverage ratchet → W12 (see §4.5).* |

---

## 4. Changes (concrete, per finding)

### 4.1 F012 — gitignore, purge history, revoke, and add secret scanning

Four coordinated actions; do them in this order:

1. **Stop tracking the artifact.** Add `tests/e2e/.auth/` to `.gitignore` and `git rm --cached
   tests/e2e/.auth/user.json` so the file remains locally (the fixture still works) but is no longer tracked.
   Confirm the dev-browser suite regenerates it on run and does not depend on the committed copy.
2. **Purge the blob from history.** Rewrite history to remove `tests/e2e/.auth/user.json` from every commit
   (e.g. `git filter-repo --path tests/e2e/.auth/user.json --invert-paths`, the maintained successor to
   BFG/`filter-branch`). **This is a coordinated force-push** — see §6 for the operational sequence
   (announce, force-push protected `main`, collaborators re-clone/reset, invalidate stale PR bases).
3. **Revoke the dev-browser client** in Clerk regardless of the expired TTL — the `__clerk_db_jwt` dev-browser
   token identifies a client that should be invalidated so a replay of the historical blob is inert.
4. **Add gitleaks to CI** (shared with §4.4) so the next committed secret is caught at the PR, not in a
   review six months later. The gitleaks job scans **full history** (`--log-opts` or `detect` over the whole
   repo), so it doubles as the verification gate for the purge in step 2.

> Note the honest severity (review §4): the committed JWTs **expired 2026-02-17**, so this is a
> committed-artifact / re-commit hazard, not the live-to-2027 credential the original headline claimed. The
> fix is unchanged — gitignore, purge, revoke, scan — because a secret in history plus a re-commit-every-run
> fixture is a standing hygiene failure regardless of the current token's TTL.

### 4.2 F058 — encrypt the Shopify Admin token at rest

In the `POST /api/shopify/connect` handler (`orchestrator/api/shopify.py`, ~L357–372 on the pinned commit —
re-confirm on `main`), the token is written verbatim:

```python
settings["shopify_access_token"] = request.access_token   # plaintext — the defect
```

while the docstring (~L359–360) claims "encrypted at rest via database-level encryption." **Make the docstring
true:**

- **On write:** encrypt `request.access_token` through the platform's canonical at-rest encryption path
  (the same mechanism already used for other injected secrets — reuse it per CLAUDE.md §2; do **not**
  hand-roll crypto or add a new dependency, CLAUDE.md §11). Store the ciphertext under
  `workspace.settings.shopify_access_token`.
- **On read:** decrypt at the point of use (the Composio bridge / any reader of
  `settings["shopify_access_token"]`). Audit every reader on `main` and route each through the decrypt path so
  no consumer receives ciphertext expecting plaintext. `integration_bridges/shopify.py` is a known consumer —
  confirm the full set with `grep -rn "shopify_access_token" orchestrator/`.
- **Docstring:** if the chosen mechanism is application-level (not literal DB-level TDE), correct the wording
  to describe what actually happens — an accurate docstring is part of the fix (no drift, CLAUDE.md §4).
- **Migration of any already-stored plaintext tokens** is an owner-input item: existing rows may hold
  plaintext under the old path. Surface it in §6 (do not silently skip it) — either a one-time re-encrypt
  migration or a documented "reconnect to re-store encrypted" path, Gerard's call.

### 4.3 F011 — merge the mem0 patch, pin the deploy SHA, add a boot probe

Two repos are involved; the fix lives mostly in `automatos-mem0`, the probe lands in both.

- **Merge the branch to fork main.** Merge `automatos-mem0` `fix/pool-exhaustion` (PRD-156 router token auth +
  PRD-159 metadata preservation, tip `16b27eb2`) into the fork's `origin/main`. This is proven, reviewed work —
  the finding is that it is *unmerged*, not that it is *wrong* (§2.1). Rebuild the Railway mem0 image from the
  merged main.
- **Pin Railway to a verified SHA.** Pin the mem0 service to the exact merged commit SHA (not a floating
  `main` tag), so the deployed image cannot silently drift back to an unauthenticated build. Record the pinned
  SHA in the deploy config/runbook.
- **Add a boot probe asserting 401-without-token.** On orchestrator startup (or a CI/deploy smoke step),
  issue an **unauthenticated** request to an OpenMemory router and assert it returns **401**. If it returns
  200, the deployed image is the unauthenticated build — fail the boot/smoke loudly rather than serve an open
  tenant-memory service. This probe is the F011 half of the wave's acceptance bar (§5.2) and is the standing
  guard against a future drift to the wrong SHA.

> Whether the current Railway image is actually built from fork main is a **production unknown** the owner must
> confirm (§14, surfaced in §6). The fix — merge, pin, probe — is correct regardless: it makes the
> unauthenticated state impossible to deploy silently, confirmed config or not.

### 4.4 F092 (W3 slice) — add CodeQL (SAST), `dependabot.yml`, and gitleaks lanes

Add three standing supply-chain lanes to `.github/workflows/` and `.github/`:

- **CodeQL (SAST).** Add a CodeQL workflow covering the repo's languages (Python + TypeScript). Run on PRs to
  `main` and on a schedule. Fail the PR on newly-introduced high-severity alerts (tune to avoid blocking on
  pre-existing findings so it does not red the tree on day one — see §6 risk note).
- **`dependabot.yml`.** Add `.github/dependabot.yml` covering the dependency ecosystems present (Python `pip`,
  frontend `npm`, and GitHub Actions). This opens automated dependency-update PRs — it is the "dependency
  audit" half of the F092 bar; the PRs themselves are triaged as they arrive, not part of this wave's diff.
- **gitleaks.** Add a gitleaks CI job (shared with §4.1 step 4) scanning **full history**, on PRs to `main`.
  This is the enforcement that keeps F012-class artifacts out and verifies the §4.1 purge landed clean.

All three are additive CI/config — no application-code change, no runtime impact. This is the mechanical,
Ralph-able core of the wave (roadmap §5).

### 4.5 Scope note — F092 is split across waves by the review's dependency order (CLAUDE.md §12)

F092 is a multi-part finding that the review §13 **deliberately sequences across three waves**. This is the
review's dependency order, stated as such — **not** a unilateral descope (CLAUDE.md §12; roadmap §5 "the wave
boundaries here are the review's dependency order, not silent deferral"):

- **Wave 3 (this PRD) owns:** add **SAST (CodeQL) + `dependabot.yml`** lanes, plus **gitleaks** (which also
  lands here via F012). Verbatim from review §13 Wave 3: *"add SAST and dependabot lanes."*
- **Wave 6 (PRD-176) owns** the **migration-replay lane** — CI runs `alembic upgrade heads` from an empty
  pgvector database and asserts exactly one head. Today CI initialises schema via `create_all` and never runs
  Alembic, which is why the four-heads state (F010) shipped undetected. *Referenced here, built there* — it is
  coupled to W6's alembic-baseline collapse and cannot land meaningfully without it.
- **Wave 12 (PRD-182) owns** the **measured coverage ratchet** (no aspirational 80% — measure the real
  baseline on code that runs, then ratchet), the **frontend CI lane**, **orphaned test-tree collection**, and
  the **route-contract test**. *Referenced here, built there.*

If re-confirmation on `main` shows any of the W3-owned lanes needs a piece this PRD doesn't name to actually
run and stay green, that piece is added here — not punted (CLAUDE.md §12).

---

## 5. Test-first acceptance

Write these **failing first**, then implement to green. The wave's definition of done (review §13 Wave 3):
**"gitleaks scans history clean and the boot probe returns 401 for an unauthenticated OpenMemory call."**

1. **gitleaks clean on cleaned history (F012 — headline acceptance).** The gitleaks CI job runs over **full
   history** and exits clean — no `tests/e2e/.auth/user.json` blob, no other detected secret. Written first,
   this **fails** against pre-purge history (the committed artifact is detected) and **passes** only after the
   purge (§4.1) lands. This is the exact gap that let the artifact ship, and it stays green as a permanent
   regression guard. Also assert `tests/e2e/.auth/` is in `.gitignore` and `git ls-files tests/e2e/.auth/`
   returns empty.
2. **mem0 boot probe returns 401 without a token (F011 — headline acceptance).** With a token configured, an
   **unauthenticated** request to an OpenMemory router asserts **401** (not 200). This fails against a
   fork-main (pre-merge) image and passes against the merged, SHA-pinned image (§4.3). Pair with a
   positive-path assertion that an authenticated request succeeds and that PRD-159 typed metadata round-trips
   (is not dropped), so the probe guards both halves of F011.
3. **Shopify token encrypted at rest (F058).** A test drives `POST /api/shopify/connect`, then asserts the
   stored `workspace.settings.shopify_access_token` is **not** the plaintext input (ciphertext at rest), and
   that the decrypt path round-trips back to the original token for the reader. Fails against the current
   plaintext write; passes once the encryption path (§4.2) is wired. Explicitly assert the raw plaintext does
   **not** appear anywhere under `workspace.settings`.
4. **CodeQL + dependabot lanes present and green (F092 W3 slice).** CI shows a CodeQL job running on PRs to
   `main` and green (no newly-introduced high-severity alerts on the change), and `.github/dependabot.yml`
   exists and validates. (Coverage ratchet and migration-replay are **not** asserted here — they are W12/W6
   per §4.5.)

**Wave-level bar:** gitleaks scans history clean **and** the boot probe returns 401 for an unauthenticated
OpenMemory call — plus the Shopify token round-trips through encryption (never plaintext at rest) and the SAST
+ dependabot + gitleaks lanes are present and green. At that point three Security-pillar sub-bars close:
gitleaks-clean history, tokens encrypted at rest, and SAST + dependabot + gitleaks lanes green.

---

## 6. Risks & rollback

- **History rewrite is the one operationally careful step (F012, §4.1).** Purging the blob rewrites every
  commit hash and requires a **force-push to protected `main`**. Sequence: (1) announce a freeze window; (2)
  merge/close open PRs first (their bases will be invalidated by the rewrite); (3) run the purge on a mirror,
  verify with gitleaks, then force-push; (4) all collaborators **re-clone or hard-reset** to the rewritten
  `main` — a plain `git pull` will conflict/diverge; (5) re-open any surviving PRs against the new base. This
  is coordinated, not casual. **Rollback:** keep a tagged backup of pre-rewrite `main` (`refs/backup/pre-f012`)
  until the team has re-synced; if anything goes wrong, restore from the backup ref before anyone rebases on
  top of the rewrite.
- **F058 reader-coverage risk.** Encrypting on write breaks any reader still expecting plaintext. Mitigate by
  auditing **all** consumers of `workspace.settings.shopify_access_token` on `main` (`grep -rn`) and routing
  each through the decrypt path in the same PR (CLAUDE.md §5 — migrate callers, don't leave two paths).
  Already-stored plaintext tokens need a one-time re-encrypt or a documented reconnect — an owner-input item
  (below), not silently skipped.
- **F011 deploy risk.** Merging the fork branch and re-pinning changes the running mem0 image. Mitigate with
  the boot probe (§4.3) as the go/no-go: if the new image does not 401-without-token, fail the deploy rather
  than serve it. **Rollback:** un-pin to the prior known-good SHA (still keeping the merge on fork main).
- **F092 CI-noise risk.** A freshly-added CodeQL lane can surface a backlog of pre-existing alerts and red the
  tree on day one. Mitigate by failing only on **newly-introduced** high-severity alerts (baseline the
  existing set), and let dependabot PRs queue for triage rather than gating merges. These lanes are additive
  and independently revertable — remove the workflow file to roll back with zero runtime impact.
- **Independent commits.** Each finding is a separate commit (F012 purge / F058 encryption / F011 merge+pin /
  F092 lanes). Revert individually. The two regression guards — gitleaks-clean history and the 401 boot probe
  — must stay green permanently regardless of the other fixes.

**Owner-decision & production-unknown dependencies (surface, do not silently resolve — CLAUDE.md §12; review §14):**

1. **Were the committed secrets rotated? (§14 production unknown).** Were the committed Clerk session material
   (F012) and the flagged AWS key **`AKIA3ZLYFH2WTHW2CMN6`** rotated? The review recommends **rotating both
   regardless**; until answered, treat them as live. (The Clerk dev-browser client revocation is in §4.1
   step 3; the AWS key rotation is an operational action outside this PRD's code diff but is called out here so
   it is not lost.)
2. **Is the Railway mem0 image actually built from fork `main`? (§14 production unknown, F011).** Only the
   owner can read the Railway config. The fix (merge + pin SHA + boot probe) is **correct regardless** — it
   makes the unauthenticated state impossible to deploy silently — but confirming the current build source
   tells us whether an unauthenticated tenant-memory service is live *today* or merely latent.
3. **Migration of already-stored plaintext Shopify tokens (F058).** Existing `workspace.settings` rows may hold
   plaintext tokens under the pre-fix path. One-time re-encrypt migration, or a documented "reconnect to
   re-store encrypted" path? **Gerard's call** — surfaced as an open question, not a settled deferral.

---

## 7. References

- Review §4 — F011 (mem0 fork unmerged, ADJUSTED), F012 (committed Clerk artifact, ADJUSTED down from
  live-credential), F058 (plaintext Shopify Admin token), F092 (no SAST/dependabot/migration-replay/coverage,
  ADJUSTED — GitHub-native secret scanning enabled): `reports/PLATFORM_OS_REVIEW_2026-07-01.md`
- Review §13 Wave 3 (acceptance: **"gitleaks scans history clean and the boot probe returns 401 for an
  unauthenticated OpenMemory call"**; scope: revoke/purge/gitleaks + merge mem0/pin SHA/boot probe + encrypt
  Shopify token + add SAST and dependabot lanes)
- Review §13 Security pillar (the pass/fail bar this wave closes toward)
- Review §14 — production unknowns (rotate committed Clerk material + AWS key `AKIA3ZLYFH2WTHW2CMN6` regardless;
  confirm Railway mem0 build source)
- Roadmap §2 (Security pillar bar), §3 (W3 row: F011/F012/F058/F092, M, medium/history-rewrite), §4 Phase A,
  §5 (Ralph option for mechanical waves; review's dependency order is not deferral)
- **Wave-split cross-references (F092):** migration-replay lane → **W6 / PRD-176**; measured coverage ratchet +
  frontend CI lane + orphaned test-tree collection + route-contract test → **W12 / PRD-182**
- Related repos: `automatos-mem0` (`fix/pool-exhaustion@16b27eb2` — F011); `automatos-ai/orchestrator/api/shopify.py`
  (F058); `automatos-ai/tests/e2e/.auth/user.json` (F012)
- CLAUDE.md §2 (reuse the canonical encryption path — don't hand-roll), §4 (no drift — make the docstring true),
  §5 (migrate all readers, don't leave two paths), §11 (no new crypto dependency), §12 (no unilateral descope —
  the F092 wave-split is the review's sequencing; owner-decisions surfaced, not resolved)
