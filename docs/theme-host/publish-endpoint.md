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
- [orchestrator/services/themes/signed_upload.py](../../orchestrator/services/themes/signed_upload.py)
- [orchestrator/services/themes/submissions.py](../../orchestrator/services/themes/submissions.py)
- [orchestrator/config.py](../../orchestrator/config.py)

</details>

`POST /api/marketplace/themes/publish` lets the Design Kit's publish job put a theme version on the Marketplace with
no person in the loop: it lists the version, approves it and stores its signature, in one step. Merging a version bump
in the kit is the approval. The kit's `theme-publish.yml` calls it from its `publish` job (GitHub environment
`theme-publish`, `main` only) and makes the GitHub release only when this endpoint answers `published`, or
`unchanged` for a re-run.

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
8. The version is above the highest one held (else 409), within `THEME_PUBLISH_PER_DAY` (else 429), and core's own
   automated review gives `pass` or `review_required` (`blocked` is 422).

Then, in one transaction: the submission, the summary as its review note, the approval, the signature, the files and
the commit. If any step fails, all of it is rolled back. The answer is
`{"status": "published", "id", "version", "digest"}`. Refusals use the Marketplace error shape,
`{"detail": {"code", "message", "errors": [{"code", "path", "message"}]}}`.

The key may do nothing else: no other route accepts it (hybrid auth refuses it), and it can't reject, deactivate or
revoke anything. The one-pending-submission rule (`THEMES_ONE_OPEN_SUBMISSION`) doesn't apply to it.

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
