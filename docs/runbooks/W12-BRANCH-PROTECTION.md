# Runbook — Branch Protection `strict=true` (PRD-182 W12-S5 / F057)

**Scope:** Enable "Require branches to be up to date before merging"
(`strict=true`) on the `main` branch's required status checks for
`AutomatosAI/automatos-ai`.

**Why this is a runbook, not a code change:** branch protection is a
GitHub **repo-admin** setting, not something a workflow or a PR can apply. It is
**surfaced here for the repo owner to run — deliberately not applied by the
wave** (Gerard's call, per the OS review's "the tradeoff against solo velocity
is yours").

---

## 1. The problem it fixes (F057)

`main` currently runs required checks with **`strict=false`**, zero required
reviews, and `enforce_admins` off. `strict=false` is the **stale-merge window**:
a PR can be merged while its branch is behind `main`, so a check that was green
against an *old* base merges code that was never tested against the *current*
base. The OS review ties this directly to **two red-`main` incidents** (PR #457
and the PRD-158 breakage): a PR merged green, but against a stale base, and
`main` went red the moment it landed.

`strict=true` closes the window: GitHub refuses the merge until the branch is
rebased/updated onto the latest `main` **and** the required checks pass against
*that* state. The harness / CI becomes the second reviewer of record.

---

## 2. The command (run by a repo admin)

> **Do not run this from CI or a PR.** Run it locally as a user with admin on
> `AutomatosAI/automatos-ai`, authenticated via `gh auth login`.

The exact setting the PRD specifies:

```bash
gh api -X PATCH \
  repos/AutomatosAI/automatos-ai/branches/main/protection/required_status_checks \
  -f strict=true
```

`strict=true` flips **only** the "up to date before merging" flag. It does **not**
change *which* checks are required — that list is preserved as-is.

---

## 3. Before you flip it — make the new lanes required first (recommended order)

The Wave-12 lanes are introduced **NON-REQUIRED** on purpose (the workflow's
standard posture — every lane in `test.yml` started non-required until proven
green). `strict=true` only enforces the checks that are already in the
**required** list, so it has no effect on a non-required lane. Recommended
sequence:

1. **Let the new lanes run green on a few PRs.** Watch, in Actions:
   - `orchestrator-tests` (already required) — now also emits coverage + runs the ratchet.
   - `frontend-ci` (new) — vitest gate, tsc baselined, eslint report, route-contract gate.
   - `orchestrator-module-tests` (new) — the two orphaned trees (F056).
2. **Arm the coverage ratchet (S4).** The first `orchestrator-tests` run prints
   the MEASURED coverage %. Read it from the "Coverage ratchet" step log, then
   commit that number into `orchestrator/.coverage-baseline` (replacing the
   `SEED` token). Until you do, the ratchet reports the number but does not
   enforce it (CI is ephemeral, so the floor must be committed to bite).
3. **Promote the lanes you want enforced to required**, then run the `strict`
   command. To add a check to the required list (example — do this per check
   once it is reliably green):

   ```bash
   # Fetch the current required contexts, add the new ones, PATCH the whole set.
   # `contexts` are the check "name:" strings, e.g.:
   #   "Frontend CI (tsc baselined, vitest, eslint, route-contract) (non-required)"
   #   "Orchestrator module + integration tests (F056) (non-required)"
   # (rename the jobs to drop "(non-required)" when you promote them, so the
   #  required-context name stays honest).
   gh api repos/AutomatosAI/automatos-ai/branches/main/protection/required_status_checks
   # then PATCH with -f 'contexts[]=<name>' for each check you want required.
   ```

> Order matters: promote a lane to required **only after** it is dependably
> green, or `strict=true` + a flaky-required check will block all merges. The
> two Wave-12 backend/module lanes may need their live-service gaps closed
> (Redis, pgvector, benchmark stack) before they are green enough to require.

---

## 4. Verify

```bash
gh api repos/AutomatosAI/automatos-ai/branches/main/protection/required_status_checks \
  --jq '{strict: .strict, contexts: .contexts}'
```

Expect `"strict": true` and the `contexts` list containing every check you
promoted.

---

## 5. Rollback

```bash
gh api -X PATCH \
  repos/AutomatosAI/automatos-ai/branches/main/protection/required_status_checks \
  -f strict=false
```

Reverts to the pre-Wave-12 posture. Use only if `strict=true` proves too costly
against solo velocity — the tradeoff the OS review flagged as the owner's call.
