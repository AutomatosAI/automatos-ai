# Theme operations

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this page:

- [orchestrator/config.py](../../orchestrator/config.py)
- [orchestrator/services/themes/data/signing-keys.json](../../orchestrator/services/themes/data/signing-keys.json)
- [orchestrator/services/themes/signature_verify.py](../../orchestrator/services/themes/signature_verify.py)
- [orchestrator/services/themes/signature_state.py](../../orchestrator/services/themes/signature_state.py)
- [orchestrator/services/themes/defaults.py](../../orchestrator/services/themes/defaults.py)
- [orchestrator/services/themes/signing.py](../../orchestrator/services/themes/signing.py)
- [orchestrator/services/themes/review.py](../../orchestrator/services/themes/review.py)
- [orchestrator/services/themes/review_package.py](../../orchestrator/services/themes/review_package.py)
- [orchestrator/services/themes/submissions.py](../../orchestrator/services/themes/submissions.py)
- [orchestrator/services/themes/listings.py](../../orchestrator/services/themes/listings.py)
- [orchestrator/services/themes/listing_updates.py](../../orchestrator/services/themes/listing_updates.py)
- [orchestrator/services/themes/revocations.py](../../orchestrator/services/themes/revocations.py)
- [orchestrator/services/themes/revocation_admin.py](../../orchestrator/services/themes/revocation_admin.py)
- [orchestrator/services/themes/revocation_feed.py](../../orchestrator/services/themes/revocation_feed.py)
- [orchestrator/services/themes/revocation_pull.py](../../orchestrator/services/themes/revocation_pull.py)
- [orchestrator/services/theme_revocation_scheduler.py](../../orchestrator/services/theme_revocation_scheduler.py)
- [orchestrator/services/scheduler.py](../../orchestrator/services/scheduler.py)
- [orchestrator/services/themes/records.py](../../orchestrator/services/themes/records.py)
- [orchestrator/services/themes/settings.py](../../orchestrator/services/themes/settings.py)
- [orchestrator/services/themes/scan.py](../../orchestrator/services/themes/scan.py)
- [orchestrator/services/themes/review_stage.py](../../orchestrator/services/themes/review_stage.py)
- [orchestrator/services/themes/llm_stage.py](../../orchestrator/services/themes/llm_stage.py)
- [orchestrator/services/themes/mobile_contrast.py](../../orchestrator/services/themes/mobile_contrast.py)
- [orchestrator/services/themes/sdk_uses.py](../../orchestrator/services/themes/sdk_uses.py)
- [orchestrator/services/themes/drift.py](../../orchestrator/services/themes/drift.py)
- [orchestrator/services/themes/review_limits.py](../../orchestrator/services/themes/review_limits.py)
- [orchestrator/services/themes/updates.py](../../orchestrator/services/themes/updates.py)
- [orchestrator/api/admin_themes.py](../../orchestrator/api/admin_themes.py)
- [orchestrator/api/admin_theme_revocations.py](../../orchestrator/api/admin_theme_revocations.py)
- [orchestrator/api/admin_theme_revocation_sync.py](../../orchestrator/api/admin_theme_revocation_sync.py)
- [orchestrator/api/themes_revocations_feed.py](../../orchestrator/api/themes_revocations_feed.py)
- [orchestrator/main.py](../../orchestrator/main.py)
- [frontend/app/admin/themes/page.tsx](../../frontend/app/admin/themes/page.tsx)
- [frontend/components/admin/themes/review-queue.tsx](../../frontend/components/admin/themes/review-queue.tsx)
- [frontend/components/admin/themes/review-report.tsx](../../frontend/components/admin/themes/review-report.tsx)
- [frontend/components/admin/themes/report-note-form.tsx](../../frontend/components/admin/themes/report-note-form.tsx)
- [frontend/components/admin/themes/signature-panel.tsx](../../frontend/components/admin/themes/signature-panel.tsx)
- [frontend/components/admin/themes/deactivate-form.tsx](../../frontend/components/admin/themes/deactivate-form.tsx)
- [frontend/components/admin/themes/revocations-panel.tsx](../../frontend/components/admin/themes/revocations-panel.tsx)
- [frontend/components/admin/themes/revocation-form.tsx](../../frontend/components/admin/themes/revocation-form.tsx)
- [frontend/components/admin/themes/revocation-sync-status.tsx](../../frontend/components/admin/themes/revocation-sync-status.tsx)
- [frontend/components/settings/themes/scan-report.tsx](../../frontend/components/settings/themes/scan-report.tsx)

</details>

Runbooks for the people who run an Automatos server and its theme Marketplace. Each step names the screen and the
route behind it. The routes are in the [API reference](api-reference.md); the rules behind them are in the
[Security model](security-model.md).

The theme review queue, the revocations and the revocation sync are on `/admin/themes`, for platform admins. In the
local edition the operator is a platform admin. On the hosted edition the actions that publish, take down or vouch
for a version ask you to confirm your sign-in.

## The server's theme settings

Every setting is read from the environment by `orchestrator/config.py`.

| Setting | Default | What it does |
|---|---|---|
| `THEMES_PRIVATE_UPLOAD` | `auto` | Private uploads platform-wide: `auto`, `on`, `off` or `admins` (see [below](#switch-private-uploads-and-the-marketplace)) |
| `THEMES_MARKETPLACE` | `on` | The theme Marketplace: `on` or `off` |
| `THEMES_ONE_OPEN_SUBMISSION` | `true` | One pending Marketplace submission per author at a time |
| `THEME_MAX_PACKAGE_MB` | `10` | The `.autotheme` as uploaded |
| `THEME_MAX_CODE_KB` | `3072` | JS plus CSS together |
| `THEME_MAX_ENTRY_KB` | `300` | `dist/entry.js` |
| `THEME_MAX_CLIP_KB` | `1024` | Each mascot clip |
| `THEME_FILE_CACHE_SECONDS` | `86400` | How long a browser keeps a theme file |
| `STEP_UP_MAX_AGE_MINUTES` | `10` | How recent a sign-in the sensitive routes need (hosted edition) |
| `THEME_REVIEW_MAX_CONCURRENT` | `2` | Review stages running at once in one process; more is 503 |
| `THEME_UPLOADS_PER_DAY` | `20` | Private uploads per workspace in 24 hours; more is 429 |
| `THEME_SUBMISSIONS_PER_DAY` | `10` | Marketplace submissions per person in 24 hours; more is 429 |
| `THEME_REVIEW_DAILY_TOKENS` | `2000000` | LLM tokens every theme review may use in 24 hours |
| `THEME_LLM_CHUNK_TOKENS` | `8000` | Tokens per chunk of the AI code read |
| `THEME_LLM_MAX_CHUNKS` | `48` | Chunks per package; more and the AI code read does not run on all the code |
| `THEME_LLM_STAGE_SECONDS` | `300` | The whole AI code read's deadline |
| `THEME_LLM_TIMEOUT_SECONDS` | `60` | Each call's timeout |
| `PLUGIN_LLM_SCAN_MODEL` | `claude-haiku-4-5` | The model the AI code read uses (shared with the plugin scanner) |
| `ANTHROPIC_API_KEY` | unset | Without it the AI code read does not run ("LLM stage did not run") |
| `THEME_REVOCATIONS_URL` | `https://api.automatos.app/api/themes/revocations` | The feed the local edition pulls; empty turns the pull off |
| `THEME_REVOCATIONS_PULL_MINUTES` | `60` | How often the local edition pulls |
| `THEME_REVOCATIONS_LIFTED_DAYS` | `365` | How long the hosted feed keeps a lifted revocation |
| `THEME_REVOCATIONS_FEED_MAX_KB` | `1024` | The largest feed either side reads or serves |

A review limit or an AI code read setting of 0 or below falls back to its default (`review_limits.positive_setting`,
`llm_stage._setting`).

## Add the production signing key

Done on 2026-10-26: both files hold Automatos's theme-signing key, `ed25519:e138b6d0fa7aaaf3`, `active`. The steps
stay for a new key. A keys file with no active key verifies nothing: nothing is Official, no signed upload installs,
and a pasted signature is refused. The private key never goes near the orchestrator.

1. **In the Design Kit repository, set up the `theme-signing` environment** (Settings › Environments): deployment
   branches `main` only, and the named signers as required reviewers. The kit's
   [OFFICIAL-LISTING.md](https://github.com/AutomatosAI/automatos-design-kit/blob/main/docs/OFFICIAL-LISTING.md)
   says how.
2. **Create one Ed25519 key** and store it, as a PKCS#8 PEM, only as that environment's secret `THEME_SIGNING_KEY`.
   Keep an offline backup: losing the key means signing every official version again with a new one.
3. **Write its public half** as one entry of the keys file:

   ```json
   {"schema": 1, "keys": [{"keyId": "ed25519:<16 hex>", "spki": "<base64>", "status": "active", "added": "YYYY-MM-DD"}]}
   ```

   `spki` is the public key's SubjectPublicKeyInfo DER in base64. A **Theme sign** run prints both values in its
   summary (`Public key (SPKI, base64)` and `Key id`). To check a key id against its `spki` the way the server does
   (`signature_verify.key_id_of`):

   ```sh
   python3 -c 'import base64,hashlib,sys; print("ed25519:" + hashlib.sha256(base64.b64decode(sys.argv[1])).hexdigest()[:16])' '<spki>'
   ```

4. **Put the same bytes in both repositories**: the kit's `keys/theme-signing.json` and this repository's
   `orchestrator/services/themes/data/signing-keys.json`. Merge both, and deploy the core release that carries it.
   Until the kit's file holds an active key, the kit's Theme release workflow attaches no signature. In the same
   change, update the docs that name the shipped key: [Themes](../themes/_index.md#who-vouches-for-a-theme),
   [the signature file](../themes/installing-and-choosing.md#the-signature-file),
   [Marketplace › Themes](../community-marketplace/themes-tab.md),
   [Signatures and revocations](_index.md#11-signatures-and-revocations) and [Signatures](security-model.md#signatures).
5. **Check it loads.** A keys file that breaks a rule doesn't load, and then a signature paste and a signed upload
   answer 503, and on the hosted edition `GET /api/themes/revocations` answers 503 too. A first pasted signature that
   stores (`signatureVerified: true` and the key id in the report) proves the key end to end.

A signature made before the key was deployed can still be pasted afterwards: the approval's signature request stays
open, and a paste no active key verifies stores nothing (400).

### Retire or revoke a key

Change the key's `status` in both files and deploy:

- `retired`: versions it signed keep running, but are no longer Official, and a private upload it signed asks members
  to opt in again. If such an upload is a workspace default, the default stops applying (`defaults.default_install`
  skips it and logs a warning), so members who relied on it see Studio until the upload carries a signature an active
  key verifies or an admin sets another default.
- `revoked`: versions it signed stop running (read as `disable`) until they are signed again. The hosted feed lists
  the key in `revokedKeys`, so each local edition distrusts it on its next pull, before its own keys file changes.

To sign a version again, run Theme sign with the version's open signature request (the one **Approve** opened, shown
in its report) and paste the new signature: the admin queue takes a new signature for a version whose stored one no
longer verifies.

## List an official theme

A theme is **Official** when its publisher is the reserved `automatos` publisher and the version shown carries a
signature an active key verifies. The Design Kit's
[OFFICIAL-LISTING.md](https://github.com/AutomatosAI/automatos-design-kit/blob/main/docs/OFFICIAL-LISTING.md) is the
whole run, from the kit's release to the release's `.sig.json`. The server's side of it:

1. **Before you start**, the signing key is deployed (above), and this server's host and SDK versions meet the
   version's `engines`. Otherwise the listing shows it as needing a newer Automatos, and it can't be installed.
2. **Submit.** As a platform admin, in **Marketplace › Themes › Publish theme**, submit the release's `.autotheme`
   (`POST /api/marketplace/themes/submit`). Only a platform admin may submit under `automatos/…`. The package must ship
   `source/`. The version is pending with its automated verdict. "LLM stage did not run" (review required) is expected
   when the code is more than `THEME_LLM_MAX_CHUNKS` chunks; approval still works.
3. **Review.** In `/admin/themes`, open the version: its report shows its digest. Run the kit's **Theme review**
   workflow with `package_url` (the release asset, or the 15-minute link **Get the review job inputs** asks for,
   `GET /api/admin/themes/{packageId}/review-package`) and `expected_digest`. Paste the job's summary, choose
   **Pass**, and **Save review note** (`POST …/report-note`). A pass note must hold the kit's
   `Package digest:  <digest>` and `Outcome: pass` lines for this version. The local edition has no isolated job and
   skips this step.
4. **Approve** (`POST …/approve`). Approve opens a signature request. The answer, and the report's Signature section,
   carry it: `{requestId, payload, payloadSha256, algorithm, form, canonical}`. The version is now listed, and
   workspaces running an older Marketplace version of the theme are offered the update. Each of those installs that is
   not pinned moves to it at once (`updatesApplied` in the answer); one the move refuses keeps the offer, and the
   server logs why.
5. **Sign.** Copy the whole request and run the kit's **Theme sign** workflow on `main` with it as its `request`
   input. A required reviewer of `theme-signing` approves the run. Its summary prints the request id, the signature,
   the key id and the envelope.
6. **Paste the signature** and **Save signature** (`POST …/signature`, `{requestId, signature}`). It is stored only
   when an active key verifies it. The report then shows a verified signature by `ed25519:…`, and the listing turns
   Official.
7. **Check from another workspace.** Marketplace › Themes shows the theme with the **Official** badge. Add it and use
   it there.

To take a listed version down later, use **Deactivate** (next section).

## Revoke a theme, and lift a revocation

A revocation takes effect on each member's next load. Use the revocations panel under the review queue
(`/admin/themes`), or `POST /api/admin/themes/revocations`.

1. **Choose the target.**
   - A **digest** covers one package version, in every workspace that holds those bytes.
   - A **theme id** covers every version of that theme; add a **version range** (`1.2.0`, or `^1.0.0`) to narrow it.
2. **Choose the level.** `warn` keeps the theme running and shows the reason. `disable` sends members to the theme's
   base, and its files stop loading.
3. **Write the reason** members will read: one line of plain text, up to 2000 characters.
4. **Revoke.** The same target and level a second time answers the live row already there (200), not a second one.

To **take a Marketplace version off the Marketplace**, use **Deactivate** on its report (`POST …/deactivate`, with a
reason) rather than a revocation. It sets the version `deactivated`, writes a `disable` of its digest, withdraws the
update offers its approval made and moves each unpinned install that runs it back to the newest listed version below
it (`installsRolledBack` in the answer). With none listed, those installs stay on it and fall back to their base.
Lifting a revocation never undoes a deactivation: the last live local disable of a deactivated version can't be
lifted.

**Lift** a revocation from the panel, then **Confirm lift** (`POST /api/admin/themes/revocations/{revocationId}/lift`). A row written here
(`local`) can be lifted at any time. A row pulled from the feed (`feed`) is the hosted list's to lift; in the local
edition you may lift it only while sync is off or stale, and the server logs who did. A later pull never un-lifts it.

On the hosted edition, a revocation that covers a listed Marketplace version goes out in the feed, and each local
edition applies it on its next pull. A lifted one stays in the feed for `THEME_REVOCATIONS_LIFTED_DAYS`, so a local
edition that pulls within that time lifts its copy.

## The revocation feed and the local pull

**Hosted edition.** Nothing to configure. `GET /api/themes/revocations` serves the feed to anyone, with
`Cache-Control: public, max-age=300`. A 503 means the signing keys file doesn't load, or the feed is too large for a
local edition to read; the server logs which as an error.

**Local edition.** The pull copies the hosted feed into this server's revocations
(`services/themes/revocation_pull.py`):

- It runs in the local edition only, every `THEME_REVOCATIONS_PULL_MINUTES` on the shared scheduler, the first run 60 seconds after start
  (`services/theme_revocation_scheduler.py`). The shared scheduler starts only when `HEARTBEAT_ENABLED`,
  `RECIPE_SCHEDULER_ENABLED` or `COORDINATOR_ENABLED` is on. Without it, **Check now** is the only pull.
- Set `THEME_REVOCATIONS_URL` empty to turn the pull off. It must be `https://`, or `http://` to this machine, and
  carry no user name or password.
- **The sync panel** (`GET /api/admin/themes/revocations/sync`) shows `enabled`, the last attempt, the last success and
  the last error, when the hosted server built the feed, how many feed rows this server holds, the revoked keys, and
  `stale`: no successful pull in the last 7 days, or none yet.
- **Check now** (`POST /api/admin/themes/revocations/sync/pull`) pulls once. It answers the new status, or 502 with the
  error the pull recorded, or 409 while the URL is empty.

A failed pull changes no revocation and keeps the revoked keys it had. The errors it records:

| Error | What to check |
|---|---|
| "THEME_REVOCATIONS_URL must be an https:// URL (http:// only to this machine)." | The URL's scheme and host |
| "THEME_REVOCATIONS_URL must not carry a user name or password." | Remove the credentials from the URL |
| "The revocation feed couldn't be reached (…)." / "…didn't answer in time." | The network path from this server |
| "The revocation feed answered a redirect (…); redirects aren't followed." | Point the URL at the final address |
| "The revocation feed answered …." | The status the feed's server gave |
| "The revocation feed was sent encoded; only an unencoded body is read." | A proxy compressing the answer |
| "The revocation feed is larger than … KB." | `THEME_REVOCATIONS_FEED_MAX_KB` on both sides |
| A sentence naming a row, a field or the schema | The feed itself was refused whole; nothing was applied |
| "The pulled revocations couldn't be saved." | The database |

## Switch private uploads and the Marketplace

**Private uploads** have two layers (`services/themes/records.py`, `settings.py`):

| `THEMES_PRIVATE_UPLOAD` | Workspace setting unset | Workspace `on` | Workspace `off` |
|---|---|---|---|
| `auto` (default) | Local edition: on. Hosted: on only in a workspace of one person. | On | Off |
| `on` | On | On | Off |
| `admins` | Only platform admins upload | Only platform admins upload | Off |
| `off` | Off | Off (switching it on is refused, 409) | Off |

Any other value of `THEMES_PRIVATE_UPLOAD` reads as `off`, and the server logs a warning. A workspace owner or admin
changes the workspace setting in **Settings › Appearance › Themes**, or with `PUT /api/themes/settings`
(`{"privateUploads": "on" | "off" | null}`; `null` goes back to the default). `GET /api/themes/settings` answers what
applies: `privateUploads.effective` is `on`, `off` or `admins`, beside the stored setting, its default and the platform
switch.

**The Marketplace.** `THEMES_MARKETPLACE=off` makes the listings, a listing's detail, installs from it and
submissions answer 403 "The theme Marketplace is turned off on this platform." Installed themes, private uploads and
the review queue are not affected. Any other value than `on` or `off` reads as `off`.

**One open submission.** `THEMES_ONE_OPEN_SUBMISSION` (on by default) lets each author have one submission pending
at a time.

## Read a scan report

The scan report is stored with each package (`theme_packages.scan_report`). Where to see it:

| Where | What it shows |
|---|---|
| The answer of `POST /api/themes/import` and `POST /api/marketplace/themes/submit`, and the scan report in **Install theme** and **Publish theme** | The verdict and every check but the AI code read |
| `GET /api/themes/installs/{installId}/compare/{packageId}` | The two verdicts, and each top-level field that changed (still without the AI code read) |
| `GET /api/admin/themes/{packageId}/report` and the report in `/admin/themes` | The whole report |

The report (schema `automatos-theme-scan/v2`, `services/themes/review_stage.py`):

| Field | Meaning |
|---|---|
| `verdict` | `blocked` (any check is an error), `review_required` (any check is a warning), or `pass` |
| `summary` | How many checks are at each level (the reviewer's only) |
| `checks.<name>.level` | `pass`, `warning`, `error`, or `not_measured` (a check this host can't make for this package) |

The checks:

| Check | Fields | Reads as |
|---|---|---|
| `sizes` | `packageBytes`, `codeBytes`, `entryBytes`, `largestClipBytes`, `limits` | Always a pass: a package over a cap is refused before the report |
| `cssScope` | `status` (`pass`, or `absent` without `dist/theme.css`), `path` | Always a pass, for the same reason |
| `looks` | `presets` | Always a pass |
| `contrast` | `source`, `status`, `reason`, `problems`, and `schemes` or `presets`, each with its `pairs` and their `ratio` | An error when a pair of the `mobile` block is under its minimum or a painting gives no `--background` or `--foreground`; `not_measured` without a `mobile` block |
| `uses` | `usedAreas`, `usedNames`, `undeclaredAreas`, `notAreas`, `declaredUnused`, `unknownNames`, `wideImport`, `problems` | An error when the code imports a name the SDK doesn't export; a warning for undeclared areas, `uses` entries that are no area, a whole-SDK import, or a build with no readable SDK name list (`status: not_checked`) |
| `clonedFrom` | `clones`, `stale` (`{route, path, sha, releaseSha}`), `reason` | A warning when a cloned page's core file has changed since |
| `llm` (reviewer's only) | `status` (`ran`, `partial`, `did_not_run`), `reason`, `riskScore`, `findings`, `summary`, `model`, `tokensUsed`, `calls`, `chunks`, `read` (`dist` or `dist+source`), `files`, `thresholds` | An error at a score of 70 or more; a warning at 20 or more, or whenever it didn't read all the code |

When `llm` says "LLM stage did not run", its `reason` says why: no `ANTHROPIC_API_KEY`, the client library missing, a
call that failed, a reply that wasn't the JSON asked for, the stage's deadline, the day's token budget ("daily budget
reached"), or more chunks than `THEME_LLM_MAX_CHUNKS`.

What to do with each verdict:

- **`pass`**: nothing found. A Marketplace version still needs a person, and on the hosted edition the kit job's note.
- **`review_required`**: read the warnings. Approve when they are acceptable.
- **`blocked`**: the version can't be approved. Reject it with the reason the submitter will see. A private upload
  installs whatever its verdict, and stays private (members opt in).

A row stored before the review stage existed has the early report: schema `automatos-theme-scan/v1`, verdict
`not_run`, and a `status` per check instead of a `level`. Approve refuses a `not_run` version.

Related:

- [Theme host](_index.md)
- [API reference](api-reference.md)
- [Security model](security-model.md)
- [Managing themes](../themes/managing-themes.md) (for workspace admins)
