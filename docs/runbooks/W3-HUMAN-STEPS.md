# W3 (PRD-173) — Human-Only Steps

**These steps are NOT automated and were deliberately left for a human to run.** They are
destructive-history / live-credential / cross-repo-deploy operations that must not be run
unattended by an agent. The **code parts** of W3 (Shopify token encryption, `.gitignore` entry,
CodeQL + Dependabot + gitleaks CI lanes) landed in the PR that references this runbook; what
remains below is the operational half.

Findings pinned to `37fdecc4e` (current `main` at authoring); re-confirm before running.

Order matters: **rotate credentials first** (so a leaked blob is already inert), then purge history,
then the mem0 deploy.

---

## 1. F012 — untrack + purge the committed Clerk auth blob from history

**What / why.** `tests/e2e/.auth/user.json` is git-tracked and lives in history (present since at
least PR #303). It holds two `__session` JWTs + a `__clerk_db_jwt` dev-browser token. The JWTs are
60-second tokens that **expired 2026-02-17**, so this is a re-commit hazard + hygiene failure, not a
live breach — but a secret in history is still a secret, and the file re-stages every dev-browser
run until it is untracked. The PR already added `tests/e2e/.auth/` to `.gitignore`; the steps below
stop tracking it and rewrite it out of history.

**Verification gate.** The new `.github/workflows/gitleaks.yml` scans full history. Before the purge
it will flag the blob; after the purge it goes and stays green. Use it as the go/no-go.

### 1a. Stop tracking (non-destructive — do this first, can go in a normal PR)

```bash
cd /path/to/automatos-ai
# Remove from the index but KEEP the local file (the e2e fixture regenerates it per run).
git rm --cached tests/e2e/.auth/user.json
git commit -m "chore(security): stop tracking tests/e2e/.auth/user.json (F012)"
# Confirm it is now ignored + untracked:
git check-ignore tests/e2e/.auth/user.json   # should print the path
git ls-files tests/e2e/.auth/                 # should print nothing
```

### 1b. Purge from ALL history (DESTRUCTIVE — coordinated force-push)

> This rewrites every commit hash. It requires a force-push to protected `main` and every
> collaborator to re-clone or hard-reset. Announce a freeze window first. Do it on a fresh mirror,
> verify, then push.

```bash
# 0) Announce a freeze. Merge or close all open PRs first — the rewrite invalidates their bases.

# 1) Tag a rollback point on the CURRENT main before touching anything.
cd /path/to/automatos-ai
git fetch origin
git tag backup/pre-f012 origin/main
git push origin backup/pre-f012        # keep until the team has re-synced

# 2) Fresh mirror clone to do the rewrite in isolation.
cd /tmp
git clone --mirror git@github-automatos:AutomatosAI/automatos-ai.git automatos-ai-purge.git
cd automatos-ai-purge.git

# 3) Purge the blob from every commit. git-filter-repo is the maintained successor to
#    BFG / filter-branch (install: `brew install git-filter-repo` or `pipx install git-filter-repo`).
git filter-repo --path tests/e2e/.auth/user.json --invert-paths --force

# 4) Verify the blob is gone from history BEFORE pushing.
git log --oneline --all -- tests/e2e/.auth/user.json    # must print NOTHING
#    Optional belt-and-braces: run gitleaks locally over the rewritten mirror.
#    docker run -v "$(pwd):/repo" zricethezav/gitleaks:latest detect --source=/repo --log-opts="--all"

# 5) Force-push the rewritten refs. (Temporarily lift branch protection / allow force-push on main,
#    then restore it immediately after.)
git push --force --mirror git@github-automatos:AutomatosAI/automatos-ai.git
```

### 1c. Every collaborator re-syncs (a plain `git pull` will diverge)

```bash
# Preferred: re-clone fresh.
# Or hard-reset an existing clone (loses un-pushed local work — stash/branch it first):
git fetch origin
git reset --hard origin/main
git for-each-ref --format="%(refname)" refs/original/ | xargs -n1 git update-ref -d 2>/dev/null || true
```

### 1d. Re-open any surviving PRs against the rewritten base, then remove the backup

```bash
# Once the team confirms everyone re-synced and CI (gitleaks) is green on the new main:
git push origin :refs/tags/backup/pre-f012   # delete the rollback tag
```

**Rollback:** if anything goes wrong before collaborators rebase on the rewrite, restore with
`git push --force origin backup/pre-f012:main`.

---

## 2. Credential rotation (live-cred ops — do BEFORE / alongside the purge)

The review (§14) recommends rotating both regardless of the expired TTL; until done, treat them as
live.

### 2a. Revoke the Clerk dev-browser client (F012)

The purged blob carries a `__clerk_db_jwt` dev-browser token that identifies a client. Revoke it so
a replay of the historical blob is inert.

- Clerk Dashboard → the relevant **application/instance** → **Sessions** (and **API Keys** if a dev
  instance key is implicated) → revoke the dev-browser client / rotate the instance's dev keys.
- Confirm the dev-browser suite still authenticates afterwards (it regenerates `user.json` on run).

### 2b. Rotate the flagged AWS key (§14 production unknown)

Access key `<FLAGGED_AWS_KEY_ID>` (the exact ID is in review §14 — do **not** re-commit the literal; it
trips secret scanning) was flagged in the review. Rotate it regardless of whether it was ever committed here:

```bash
# Identify the IAM user the key belongs to.
aws iam list-access-keys --output table   # or the console: IAM → Users → Security credentials

# Create a NEW key, roll it into the deployment secret store (Railway/env), verify the app works,
# THEN deactivate + delete the old one:
aws iam create-access-key --user-name <USER>
# ... update Railway/secret manager with the new key, redeploy, smoke-test ...
aws iam update-access-key --user-name <USER> --access-key-id <FLAGGED_AWS_KEY_ID> --status Inactive
aws iam delete-access-key --user-name <USER> --access-key-id <FLAGGED_AWS_KEY_ID>
```

---

## 3. F011 — merge the mem0 auth/metadata patch, pin the deploy, add a boot probe

**Repo:** `automatos-mem0` (the fork) — sibling checkout, remote
`git@github-automatos:AutomatosAI/automatos-mem0.git`.

**Confirmed state at authoring:** the PRD-156 router token auth + PRD-159 metadata preservation live
only on branch `fix/pool-exhaustion` at tip **`16b27eb26ad267eed9b6eafbeb18e524547c6f0e`**, which is
**NOT** an ancestor of `origin/main` (tip `5cbb4c1f8be921ce3b0f0e09df0fb11f4cbf8c31`). Re-verify:

```bash
cd /path/to/automatos-mem0
git fetch origin
git merge-base --is-ancestor 16b27eb26ad267eed9b6eafbeb18e524547c6f0e origin/main \
  && echo "already merged (nothing to do)" || echo "NOT merged — proceed"
```

### 3a. Merge `fix/pool-exhaustion` → fork `main`

```bash
cd /path/to/automatos-mem0
git checkout main
git pull origin main
git merge --no-ff fix/pool-exhaustion \
  -m "merge(security): PRD-156 router token auth + PRD-159 metadata preservation (F011)"
git push origin main
# Record the merged SHA — you pin Railway to THIS exact commit, not a floating main.
MEM0_PINNED_SHA=$(git rev-parse HEAD)
echo "Pin Railway mem0 to: $MEM0_PINNED_SHA"
```

### 3b. Pin the Railway mem0 service to the verified SHA + rebuild

- Railway → the **mem0 / OpenMemory** service → **Settings → Source**: pin the deploy to the exact
  merged commit `$MEM0_PINNED_SHA` (a fixed commit, **not** the `main` branch tracker), so the image
  cannot silently drift back to an unauthenticated build.
- Ensure the OpenMemory API token env var is set on the service (the value the orchestrator sends).
- Trigger a rebuild/redeploy from the pinned SHA.
- **Record the pinned SHA** in the Railway service config / deploy notes.

### 3c. Boot probe — assert 401-without-token

Verify the deployed image is the authenticated build. The OpenMemory server is `server/main.py` in
the fork. With a token configured, an **unauthenticated** request to a router must return **401**
(not 200):

```bash
# Replace with the deployed OpenMemory base URL and a real memory router path.
MEM0_URL="https://<your-mem0-service>.up.railway.app"
code=$(curl -s -o /dev/null -w "%{http_code}" "$MEM0_URL/api/v1/memories/")
echo "unauthenticated status: $code"
test "$code" = "401" && echo "PASS — failing closed" || echo "FAIL — image may be the UNAUTHENTICATED build; do NOT keep it live"
# Positive path: the same call WITH the token should succeed (200/2xx).
curl -s -o /dev/null -w "authed: %{http_code}\n" -H "Authorization: Bearer $OPENMEMORY_TOKEN" "$MEM0_URL/api/v1/memories/"
```

If the unauthenticated call returns **200**, the running image is built from the pre-merge (fork-main)
source — an open tenant-memory service. Do not serve it; re-pin to the merged SHA and rebuild.

**Rollback:** un-pin to the prior known-good SHA (the merge stays on fork `main`).

---

## 4. F058 — migration of already-stored plaintext Shopify tokens (OWNER DECISION)

The code fix encrypts the Shopify Admin token on write going forward and decrypts on read
(`orchestrator/api/shopify.py`). **Existing `workspace.settings` rows may still hold plaintext tokens
under the old path.** There is no runtime reader of `settings["shopify_access_token"]` in the
orchestrator today (the Composio bridge reads the separate n8n credential store), so nothing breaks —
but the stored plaintext should not linger.

**Gerard's call — pick one (surfaced, NOT decided here):**

- **(a) One-time re-encrypt migration:** a script/Alembic data migration that reads each
  `workspace.settings.shopify_access_token`, and if it is *not* already Fernet ciphertext (i.e. it
  still `startswith("shpat_")` or fails `decrypt`), re-writes it via the encryption service. Idempotent.
- **(b) Documented "reconnect to re-store encrypted":** merchants re-run `POST /api/shopify/connect`
  (which now encrypts), and any legacy plaintext value is treated as stale. Simpler, no migration, but
  leaves plaintext in old rows until each merchant reconnects.

Do not silently skip this — it is the one F058 open item that needs an owner decision.
