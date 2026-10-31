# The theme publish endpoint

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this page:

- [orchestrator/api/marketplace_themes_publish.py](../../orchestrator/api/marketplace_themes_publish.py)
- [orchestrator/services/themes/publish.py](../../orchestrator/services/themes/publish.py)
- [orchestrator/services/themes/publish_checks.py](../../orchestrator/services/themes/publish_checks.py)
- [orchestrator/services/themes/publish_key.py](../../orchestrator/services/themes/publish_key.py)
- [orchestrator/scripts/theme_publish_actor.py](../../orchestrator/scripts/theme_publish_actor.py)
- [orchestrator/services/themes/review.py](../../orchestrator/services/themes/review.py)
- [orchestrator/services/themes/auto_update.py](../../orchestrator/services/themes/auto_update.py)
- [orchestrator/services/themes/signing.py](../../orchestrator/services/themes/signing.py)
- [orchestrator/services/themes/data/signing-keys.json](../../orchestrator/services/themes/data/signing-keys.json)
- [orchestrator/services/themes/signed_upload.py](../../orchestrator/services/themes/signed_upload.py)
- [orchestrator/services/themes/submissions.py](../../orchestrator/services/themes/submissions.py)
- [orchestrator/config.py](../../orchestrator/config.py)

</details>

`POST /api/marketplace/themes/publish` lets the Design Kit's publish job put a theme version on the Marketplace with
no person in the loop: it lists the version, approves it and stores its signature, in one step. Merging a version bump
in the kit is the approval. The kit's `theme-publish.yml` calls it from its `publish` job (GitHub environment
`theme-publish`, `main` only) and makes the GitHub release only when this endpoint answers `published`, or
`unchanged` for a re-run.

## From a kit merge to the installs

1. A pull request in the Design Kit raises a theme's `version` in its `theme.json` and merges to `main`.
2. The kit's `theme-publish.yml` picks every theme whose version is above its newest `<slug>-v*` tag and whose
   publisher its `release/publishers.json` lists, builds and packs it, and runs the kit's review on the package with no network. Its publish job then signs
   `{id, version, digest, files}` with the Ed25519 theme-signing key, which only that job's Sign step reads (the
   `theme-publish` environment's `THEME_SIGNING_KEY`), and checks the envelope against the kit's
   `keys/theme-signing.json`.
3. It posts the package, the envelope and the review's text here.
4. This endpoint checks everything (below), then lists, approves and signs the version in one transaction. The
   approval moves every unpinned Marketplace install of an older version to it
   ([Updates and rollback](updates-and-rollback.md)).
5. On `published` or `unchanged`, the kit releases `<slug>-v<version>` with the `.autotheme`, its `.sha256` and
   `<version>.sig.json`.
6. Each local install that presses **Sync with Automatos Cloud** then finds the version in the catalogue, since the
   catalogue lists every signed Marketplace version
   ([Sync with Automatos Cloud](../community-marketplace/sync-with-automatos-cloud.md)).

The orchestrator never holds the private key. It holds the public keys, `services/themes/data/signing-keys.json`,
byte for byte the kit's `keys/theme-signing.json`, and verifies with them (`signature_verify.py`).

## What it does

The request is `multipart/form-data` with `Authorization: Bearer <THEME_PUBLISH_KEY>`:

| Part | What it is |
|---|---|
| `package` | The `.autotheme`, as a file part |
| `signature` | The kit's signature envelope (`.sig.json` text) for this package |
| `review_summary` | The kit review's summary for this pack (`scripts/review/main.ts`) |

Every check runs before anything is written, in this order:

1. The key matches one of the configured hashes (else 401), and the three settings are set (else 503).
2. The Marketplace is on (else 403).
   Only then is the body read: a malformed body, more than three parts or a text part over its cap is 400
   `PARTS_UNREADABLE`, a missing part 422 `PART_MISSING`.
3. The package passes the same checks as a submission and ships `source/` (else 422).
4. Its publisher is one of `THEME_PUBLISH_PUBLISHERS`, and is the service user's or not yet claimed (else 403). Under
   a reserved publisher such as `automatos` the service user submits as an admin would; nowhere else is it an admin.
5. The envelope verifies under an active Automatos signing key and names this id, version and digest (else 422).
6. The summary holds `Package digest:  <this digest>` and `Outcome: pass`, the rule a hosted approval note meets
   (else 422). No `disable` revocation covers the version (else 422).
7. These exact bytes, already approved with a signature that verifies: 200 `unchanged`, nothing written.
8. Otherwise the request takes a review slot (503 `REVIEWS_BUSY` when `THEME_REVIEW_MAX_CONCURRENT` are running) and
   the Marketplace's claim on the digest (409 `PUBLISH_BUSY` while another publish or submission of these bytes
   runs), and reads the row again.
9. **Bytes the Marketplace holds already** go on from their row: a pending version is approved with its stored
   verdict, then signed; an approved one whose signature no longer verifies (or that has none) is only signed; a
   rejected or deactivated one is 409 `VERSION_CLOSED`. This is how a backfill run of the kit signs a version an admin
   listed by hand, or signs again a version whose key was retired or revoked.
10. **New bytes**: the version is above the highest one held (else 409), within `THEME_PUBLISH_PER_DAY` (else 429),
    and core's own automated review gives `pass` or `review_required` (`blocked` is 422).

Then, in one transaction: the submission, the summary as its review note, the approval, the signature, the files and
the commit. If any step fails, all of it is rolled back. The answer is
`{"status": "published", "id", "version", "digest"}`. Refusals use the Marketplace error shape,
`{"detail": {"code", "message", "errors": [{"code", "path", "message"}]}}`.

The approval inside that transaction is `review.approve`, as an admin's is: it offers the version to every listing
install of an older one and moves the unpinned ones to it. Its `approved` event records `updatesOffered` and
`updatesApplied`; the publish answer doesn't carry them.

The key may do nothing else: no other route accepts it (hybrid auth refuses it), and it can't reject, deactivate or
revoke anything. The one-pending-submission rule (`THEMES_ONE_OPEN_SUBMISSION`) doesn't apply to it. A key is
compared by its sha256 with `hmac.compare_digest` against every configured hash, and nothing logs it: a refusal is
logged by its status and code alone.

## Refusals

| Status | `code` | When |
|---|---|---|
| 400 | `PARTS_UNREADABLE` | The multipart body can't be read, has more than three parts, or a text part is over its cap |
| 401 | `PUBLISH_KEY_REFUSED` | The bearer key matches no configured hash |
| 403 | `PUBLISHER_NOT_LISTED`, `PUBLISHER_TAKEN` | The publisher isn't in `THEME_PUBLISH_PUBLISHERS`, or another author owns it; the Marketplace is off |
| 409 | `VERSION_NOT_HIGHER`, `VERSION_CLOSED`, `PUBLISH_BUSY` | The version isn't above the highest held; these bytes were rejected or deactivated; another publish of these bytes runs |
| 422 | `PART_MISSING`, `PACKAGE_REFUSED`, `SOURCE_REQUIRED` | A part is missing; the package fails the reader's checks (first-paint ones included, [Theme host](_index.md#1-the-package)); no `source/` |
| 422 | the envelope's code, `SUMMARY_NOT_THIS_VERSION`, `DIGEST_DISABLED` | The signature doesn't verify or names other bytes; the summary isn't this digest's pass; a `disable` covers the bytes |
| 422 | `REVIEW_BLOCKED`, `REVIEW_NOT_RUN`, `REVIEW_REFUSED` | Core's review blocked the version or didn't run; the approval refused it |
| 429 | `PUBLISH_QUOTA` | `THEME_PUBLISH_PER_DAY` versions in the last 24 hours |
| 503 | `PUBLISH_NOT_CONFIGURED`, `REVIEWS_BUSY`, `STORE_FAILED`, `SIGNING_KEYS_UNREADABLE` | A setting or the service user is missing; every review slot is busy; the files couldn't be stored; the keys file doesn't load |

Every refusal is before the commit, so a refused publish leaves no pending, approved or half-signed row
(`test_theme_publish_atomic.py`).

## Its settings

| Setting | Default | What it holds |
|---|---|---|
| `THEME_PUBLISH_KEY_SHA256` | unset | The lower-case hex sha256 of the key string. Two, comma-separated, while a rotation runs. Never the key. |
| `THEME_PUBLISH_PUBLISHERS` | unset | The publishers the key may publish under, comma-separated (for example `automatos`) |
| `THEME_PUBLISH_ACTOR` | unset | The `users.id` the key acts as: the service user below |
| `THEME_PUBLISH_PER_DAY` | `10` | Versions the key may publish in 24 hours; more is 429 |

While any of the first three is unset or can't be read, the endpoint answers 503 `PUBLISH_NOT_CONFIGURED`.

## Set it up

1. **Make the service user.** In the backend service's shell, run `python -m scripts.theme_publish_actor`
   (`docker compose exec backend python -m scripts.theme_publish_actor` on the local edition). It creates one
   `users` row, or finds it on a second run, and prints its id. The row has no sign-in and no workspace.
2. **Make the key** where only you see it: 32 random bytes as hex, for example `openssl rand -hex 32`. Put it in the
   kit's `theme-publish` environment as the secret `THEME_PUBLISH_KEY`.
3. **Hash it**: `printf %s "<key>" | sha256sum` (`shasum -a 256` on macOS; no newline after the key).
4. **Set the variables** on the backend: `THEME_PUBLISH_KEY_SHA256` (the hash), `THEME_PUBLISH_PUBLISHERS` and
   `THEME_PUBLISH_ACTOR` (the id from step 1). Restart or redeploy the backend.

## Rotate the key

1. Make a new key and hash it.
2. Set `THEME_PUBLISH_KEY_SHA256` to `<old hash>,<new hash>` and redeploy. Both keys work.
3. Replace the kit's `THEME_PUBLISH_KEY` secret with the new key.
4. Set `THEME_PUBLISH_KEY_SHA256` to the new hash alone and redeploy. The old key is now refused (401).

## Revoke the key

Remove its hash from `THEME_PUBLISH_KEY_SHA256` and redeploy. With no hash left the endpoint answers 503 and accepts
nothing. Then [deactivate](operations.md#revoke-a-theme-and-lift-a-revocation) any version you don't trust. A stolen
key alone can't publish: every version also needs a signature from the signing key, which only the kit's `publish`
job holds.
