# Theme API reference

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this page:

- [orchestrator/reports/route-manifest.json](../../orchestrator/reports/route-manifest.json)
- [orchestrator/main.py](../../orchestrator/main.py)
- [orchestrator/api/themes_import.py](../../orchestrator/api/themes_import.py)
- [orchestrator/api/themes_install.py](../../orchestrator/api/themes_install.py)
- [orchestrator/api/themes_settings.py](../../orchestrator/api/themes_settings.py)
- [orchestrator/api/themes_looks.py](../../orchestrator/api/themes_looks.py)
- [orchestrator/api/themes_files.py](../../orchestrator/api/themes_files.py)
- [orchestrator/api/themes_mobile.py](../../orchestrator/api/themes_mobile.py)
- [orchestrator/api/themes_revocations_feed.py](../../orchestrator/api/themes_revocations_feed.py)
- [orchestrator/api/marketplace_themes.py](../../orchestrator/api/marketplace_themes.py)
- [orchestrator/api/marketplace_themes_submit.py](../../orchestrator/api/marketplace_themes_submit.py)
- [orchestrator/api/admin_themes.py](../../orchestrator/api/admin_themes.py)
- [orchestrator/api/admin_theme_revocations.py](../../orchestrator/api/admin_theme_revocations.py)
- [orchestrator/api/admin_theme_revocation_sync.py](../../orchestrator/api/admin_theme_revocation_sync.py)
- [orchestrator/core/auth/workspace_permission.py](../../orchestrator/core/auth/workspace_permission.py)
- [orchestrator/core/auth/step_up.py](../../orchestrator/core/auth/step_up.py)
- [orchestrator/core/auth/roles.py](../../orchestrator/core/auth/roles.py)
- [orchestrator/core/auth/device.py](../../orchestrator/core/auth/device.py)
- [orchestrator/modules/policy/roles.py](../../orchestrator/modules/policy/roles.py)
- [orchestrator/services/themes/membership.py](../../orchestrator/services/themes/membership.py)
- [orchestrator/services/themes/records.py](../../orchestrator/services/themes/records.py)
- [orchestrator/services/themes/settings.py](../../orchestrator/services/themes/settings.py)
- [orchestrator/services/themes/choices.py](../../orchestrator/services/themes/choices.py)
- [orchestrator/services/themes/optin.py](../../orchestrator/services/themes/optin.py)
- [orchestrator/services/themes/defaults.py](../../orchestrator/services/themes/defaults.py)
- [orchestrator/services/themes/updates.py](../../orchestrator/services/themes/updates.py)
- [orchestrator/services/themes/looks.py](../../orchestrator/services/themes/looks.py)
- [orchestrator/services/themes/serve.py](../../orchestrator/services/themes/serve.py)
- [orchestrator/services/themes/signed_upload.py](../../orchestrator/services/themes/signed_upload.py)
- [orchestrator/services/themes/review_limits.py](../../orchestrator/services/themes/review_limits.py)
- [orchestrator/services/themes/review_stage.py](../../orchestrator/services/themes/review_stage.py)
- [orchestrator/services/themes/submissions.py](../../orchestrator/services/themes/submissions.py)
- [orchestrator/services/themes/listings.py](../../orchestrator/services/themes/listings.py)
- [orchestrator/services/themes/review.py](../../orchestrator/services/themes/review.py)
- [orchestrator/services/themes/review_package.py](../../orchestrator/services/themes/review_package.py)
- [orchestrator/services/themes/signing.py](../../orchestrator/services/themes/signing.py)
- [orchestrator/services/themes/revocation_admin.py](../../orchestrator/services/themes/revocation_admin.py)
- [orchestrator/services/themes/revocation_feed.py](../../orchestrator/services/themes/revocation_feed.py)
- [orchestrator/services/themes/revocation_pull.py](../../orchestrator/services/themes/revocation_pull.py)

</details>

Every theme route of the backend, as `orchestrator/reports/route-manifest.json` lists them (36), grouped by what they
are for. The frontend calls them through `apiClient`, with two exceptions: the root layout fetches the boot route on
the server (`frontend/lib/theme-host/boot-server.ts`), and the browser loads theme files through the frontend's own
`/api/themes/files/...` proxy (see [Security model](security-model.md#the-files-proxy)).

## Who may call

The **Who** column of each table uses these words:

| Word | Meaning |
|---|---|
| **member** | `require_workspace_permission("members:read")`: any role in the caller's workspace (owner, admin, editor, viewer) |
| **manage** | `require_workspace_permission("workspace:manage")`: the workspace's owner or an admin |
| **who installs** | The workspace's "who may install" setting (`require_install_role`): owners and admins by default, or the owner only (stored as `workspace:delete`) |
| **uploads on** | The private-upload switch and the workspace's setting allow this caller to upload (`records.private_upload_refusal`) |
| **marketplace on** | `THEMES_MARKETPLACE` is `on` |
| **step-up** | `require_step_up(STEP_UP_MAX_AGE_MINUTES)`: on the hosted edition, a person signed in with Clerk whose first factor was verified within the last 10 minutes (the default). The local edition always passes. |
| **platform admin** | `caller_is_admin`: a caller whose system role is `admin` or `super_admin` |
| **workspace of the install** | A member or owner of the workspace that holds the install, read from the install row (`services/themes/membership.py`), never from `X-Workspace-ID` |
| **anyone** | No credentials needed |

Gates run in the order listed, so a caller refused by an earlier gate is never asked to confirm their sign-in.

How each kind of caller reaches these gates:

| Caller | Edition | Treated as |
|---|---|---|
| A person signed in with Clerk | Hosted | Their role in the workspace they name; the only caller step-up can pass on the hosted edition |
| The local operator (`AUTH_EDITION=local`, no sign-in) | Local | A super admin, and the owner of every workspace on the instance |
| A paired phone or browser (`Authorization: Bearer amd_…`) | Local only | The person it was paired for, in exactly the one workspace it is bound to; a request naming another workspace is 403 |
| An API key or SDK key | Both | No person: refused by step-up, belongs to no workspace on the boot, files and phone routes |

A platform `super_admin` passes every workspace-permission gate.

## Errors

| Shape | When |
|---|---|
| `{"detail": "<one plain sentence>"}` | Most refusals |
| `{"detail": {"message": "...", "errors": [{"code", "path", "message"}]}}` | A package or signature file refused by `POST /api/themes/import` or `POST /api/marketplace/themes/submit` |
| `{"detail": "...", "errors": [{"code", "path", "message"}]}` | A looks write refused by its schema |
| 403 `{"clerk_error": {...}, "code": "step_up_required", "detail": "Confirm it's you to continue: ..."}` | Step-up: the sign-in is too old. The frontend's `useReverification` reads `clerk_error`. |
| 403 `{"detail": {"code": "step_up_unavailable", "message": "..."}}` | Step-up: the caller has no person who can sign in again |
| 422 | The body is not the shape the route takes. Every body model forbids unknown fields. |

The install, settings and Marketplace routes answer 404 "Choose a workspace first." when the request names no
workspace; the looks routes answer 400. A route answers
404, never 403, for another workspace's install or package.

## Installing a theme

| Method and path | Who | Request | Answer |
|---|---|---|---|
| `POST /api/themes/import` | manage, who installs, uploads on, step-up | Multipart: `file` (the `.autotheme`), optional `signature` (its `.sig.json`) | 201, or 200 when these exact bytes are already in the workspace |
| `POST /api/marketplace/themes/{listingId}/install` | manage, marketplace on, who installs, step-up | Optional `{"packageId": "<uuid>"}`: a version from the listing's versions map. Omitted: the newest version this host can run. | 201 `{created: true, install}`, or 200 `{created: false, install}` when that version is installed already |

`POST /api/themes/import` answers:

```jsonc
{
  "package": { "id", "themeId", "version", "name", "digest", "ownerType": "workspace", "reviewed": false,
               "scanVerdict", "scanReport" },           // scanReport without the AI code read
  "install": { "id", "themeId", "packageId", "pendingPackageId", "isDefault" } | null,
  "status": "installed" | "update_available" | "stored",
  "duplicate": false
}
```

`status` is `installed` when the upload is the install's current package, `update_available` when it is the
install's pending update, and `stored` otherwise.

| Status | Reason |
|---|---|
| 400 | The package fails a check (one `{code, path, message}` each); the signature file is `SIGNATURE_BAD` or `SIGNATURE_MISMATCH` |
| 403 | "who installs", the upload switch or the workspace setting refuse it (each with its sentence); a reserved publisher (`automatos/…`) without a platform admin or a verified signature |
| 409 | This workspace holds the version with other files, or the version is not newer than the installed one; the same bytes are being uploaded right now |
| 413 | The package is over `THEME_MAX_PACKAGE_MB`; the signature file is over 4 KB (`SIGNATURE_TOO_LARGE`) |
| 429 | The workspace uploaded `THEME_UPLOADS_PER_DAY` themes in the last 24 hours |
| 503 | Every review slot is busy (`Retry-After: 30`); the signing keys file doesn't load; the files couldn't be stored |

The body of both upload routes is capped on the wire at the package cap plus 512 KB (`main.py`, `PATH_BODY_LIMITS`).

`POST /api/marketplace/themes/{listingId}/install` answers 404 for a listing or version that isn't listed, 409 when no
version runs on this host, when the asked-for version needs another host, when the workspace runs another version of
the theme ("Update it from Settings, Appearance, Themes.") or its own upload of it ("Uninstall it first…"), and 403
while the Marketplace is off. A listing id is the theme id with its `/` written as `.` (`automatos.play`).

## Installed themes and their versions

| Method and path | Who | Request | Answer |
|---|---|---|---|
| `GET /api/themes/installed` | member | none | `{installs: [<install row>]}` |
| `GET /api/themes/installs/{installId}/compare/{packageId}` | manage | none | `{installId, from, to, files, scan}` |
| `POST /api/themes/installs/{installId}/update` | manage, who installs, step-up | `{"packageId": "<uuid>"}` | the install row |
| `POST /api/themes/installs/{installId}/rollback` | manage, who installs, step-up | none | the install row |
| `DELETE /api/themes/installs/{installId}` | manage | none | `{uninstalled: "<installId>"}` |

An install row:

| Field | Meaning |
|---|---|
| `installId`, `themeId`, `name`, `packageId`, `version` | The install and the package it runs |
| `ownerType` | `marketplace` or `workspace` |
| `reviewed`, `private` | The package's review flag, and whether members must opt in to it |
| `isDefault`, `pinnedVersion` | The workspace default, and a version the install is pinned to |
| `previousPackageId` | Where a roll back goes |
| `pendingPackageId`, `pendingVersion` | An update waiting to be applied |
| `installedAt` | When it was installed |
| `optIn` | `{required, wording, packageId}` for this member |
| `revoked` | `null` or `{level, reason}` |
| `needsUpdate` | A cloned page copied a core file this release has changed |
| `signature` | `{state, keyId}`: `state` is `verified`, `invalid` or `unsigned` |

`compare` answers `from` and `to` (`{packageId, version, reviewed}`), `files` (`added`, `removed`, `changed` paths from
the two file manifests) and `scan` (`verdict: {from, to}` and each changed top-level field of the report, without the
AI code read).

| Status | Reason |
|---|---|
| 404 | The install or package isn't this workspace's; for `compare`, a Marketplace version that isn't approved |
| 409 | `update`: another theme, the same version, not newer, a Marketplace version not approved, a rejected or withdrawn version, a pinned install, a default moving to private code, a version a `disable` covers. `rollback`: no earlier version, and the same rules. |

An uninstall is soft: the install keeps its row with `uninstalled_at`, members' looks stay, and members who chose it
see the workspace default.

## Choosing a theme

| Method and path | Who | Request | Answer |
|---|---|---|---|
| `GET /api/themes/active` | member | none | The active answer |
| `PUT /api/themes/choice` | member | `{"choice": "classic" \| "studio" \| "<installId>"}` | The active answer for the new choice |
| `POST /api/themes/installs/{installId}/opt-in` | member, step-up | `{"packageId": "<uuid>"}` | `{installId, optIn: {required, wording, packageId}}` |
| `PUT /api/themes/default` | manage, step-up | `{"installId": "<uuid>" \| null}` | `{defaultInstallId}` |

The active answer (`services/themes/defaults.py`):

```jsonc
{
  "choice": "classic" | "studio" | "<installId>",
  "source": "choice" | "default" | "fallback",
  "theme": null | {
    "installId", "themeId", "packageId", "version", "digest", "manifest", "defaultLook", "reviewed",
    "optIn": { "required", "wording", "packageId" },
    "revoked": null | { "level", "reason" },
    "needsUpdate",
    "integrity": { "/api/themes/files/<digest>/dist/entry.js": "sha256-<base64>" }
  },
  "revoked": { "installId", "level": "disable", "reason" }   // only when a disable covers the chosen install
}
```

When a `disable` covers the chosen install, `choice` is the theme's base (`studio` or `classic`) and `theme` is
`null`.

| Route | Status | Reason |
|---|---|---|
| `choice` | 400 | Not `classic`, `studio` or a lowercase install id |
| `choice` | 403 | The workspace doesn't let members choose; the caller is not a signed-in member |
| `choice` | 404 | No such active install in this workspace |
| `choice` | 409 | The choice changed in another tab at the same moment |
| `opt-in` | 403 | The caller is not a signed-in member |
| `opt-in` | 409 | `packageId` is not the install's current package ("That version is no longer this theme's current one…") |
| `default` | 409 | Private code ("An unreviewed theme can't be the workspace default…"), a withdrawn version, or two admins setting it at once |

## Workspace theme settings

| Method and path | Who | Request | Answer |
|---|---|---|---|
| `GET /api/themes/settings` | member | none | The settings |
| `PUT /api/themes/settings` | manage, step-up | Any of `privateUploads` (`on`, `off`, `null`), `membersMayChoose` (`true`, `false`, `null`), `installRole` (`admins`, `owner`, `null`), `pins` (`[{installId, version \| null}]`, at most 50) | The new settings |

A key left out is not changed; `null` puts a setting back to its default. A `PUT` applies every change or none.

The settings:

| Field | Meaning |
|---|---|
| `privateUploads` | `{setting, source: workspace \| default, default, platform, effective: on \| off \| admins}` |
| `membersMayChoose` | `true` unless the workspace turned it off |
| `installRole`, `installRoles` | `admins` or `owner`, and the choices |
| `canInstall` | Whether this caller may install or update themes here |
| `marketplace` | `THEMES_MARKETPLACE`: `on` or `off` |
| `defaultInstallId` | The workspace default, or `null` |
| `pins` | `[{installId, themeId, version, pinnedVersion, pinnableVersions}]`: a pin names the install's current, pending or previous version |
| `updatedAt` | The last change, or `null` |

| Status | Reason |
|---|---|
| 400 | Nothing to change, a value out of its set, a pin version not `X.Y.Z`, one install pinned twice |
| 403 | Changing `installRole` without both the old and the new permission ("Only the workspace owner can change who may install themes here."); a pin by a caller "who installs" refuses |
| 404 | A pinned install that isn't this workspace's |
| 409 | `privateUploads: "on"` while the platform switch is `off`; a pin to a version the install doesn't hold |

## Looks

| Method and path | Who | Request | Answer |
|---|---|---|---|
| `GET /api/themes/looks/{themeId}` | member | none | `{themeId, looks, active, defaultLook, dropped}` |
| `PUT /api/themes/looks/{themeId}` | member | `{"looks": [<look>], "active": <look> \| null}`: both fields, every time | The same shape, as saved |

`themeId` is `publisher/slug`; the path takes the slash. Looks belong to the caller alone, in the caller's workspace.
A read fits each saved look to the theme's current `looks/schema.json`; `dropped` counts the looks that no longer
fit. A write replaces the whole list and refuses any look the schema refuses.

| Status | Reason |
|---|---|
| 400 | No workspace; a look the schema refuses (each error names its field, such as `looks[2].glow`); more than 100 saved looks (`TOO_MANY_LOOKS`); two looks with one id (`LOOK_DUPLICATE_ID`) |
| 403 | The request has no signed-in person |
| 404 | The theme isn't installed in this workspace |
| 503 | The theme's look schema can't be read from storage |

## Serving a theme

| Method and path | Who | Request | Answer |
|---|---|---|---|
| `GET /api/themes/files/{digest}/{path}` | An approved Marketplace version's files, while no `disable` covers it: anyone. Every other package: the workspace of an install that runs it, plus the caller's opt-in for private code. | Optional `Range`, `If-None-Match` | The file's bytes: 200, 206, 304 or 416 |
| `GET /api/themes/boot/{installId}` | workspace of the install | none | The install's boot |
| `GET /api/themes/mobile/{installId}` | workspace of the install | none | The theme as data for the phone: see [The phone's theme](mobile.md) |

Files carry `Cache-Control: private, max-age=<THEME_FILE_CACHE_SECONDS>, immutable` (86400 by default), an `ETag` of
the file's sha256, `Accept-Ranges: bytes`, `X-Content-Type-Options: nosniff` and
`Content-Security-Policy: default-src 'none'; sandbox`. A request with `Service-Worker: script` is 403. Anything the
caller may not load is 404, and a storage failure is 503.

The boot (`services/themes/serve.py`) carries `addClasses`, `removeClasses`, `tone: {policy, value}`, `prepaint`,
`lookStorage`, `lookGuard`, `looks: {defaultLook, presets}`, `integrity` and `entry`. A caller who may not run the
theme, a version a `disable` covers, or a package whose `boot/prepaint.json` fails its check on read, gets the base
boot: the base's classes, the member's own tone, and `null` for the rest. The boot and
the phone answer carry `Cache-Control: private, no-store`. An unknown, malformed or foreign install id is one and the
same 404.

## The Marketplace

| Method and path | Who | Request | Answer |
|---|---|---|---|
| `GET /api/marketplace/themes` | member, marketplace on | none | `{listings: [<card>]}` |
| `GET /api/marketplace/themes/{listingId}` | member, marketplace on | none | The card plus the detail |
| `POST /api/marketplace/themes/submit` | member, marketplace on, step-up | Multipart `file` | 201, or 200 when these bytes were submitted already |
| `POST /api/marketplace/themes/publish` | the theme publish key (`Authorization: Bearer`), marketplace on | Multipart `package`, `signature`, `review_summary` | 200 `published` or `unchanged`; see [The publish endpoint](publish-endpoint.md) |

A card (`services/themes/listings.py`): `listingId`, `themeId`, `name`, `description`, `publisher: {name, official}`,
`signature: {state, keyId}`, `version`, `packageId`, `compatible`, `uses`, `pricing` (the manifest's label, else
"Free"), `cover`, `revoked` and `installed` (`{installId, packageId, version, fromListing, updatePackageId}` or `null`).
The card shows the newest version this host can run, else the newest one.

The detail adds `presets` (`[{id, default, previews: [{screen, url}]}]`), `otherPreviews`, `readme`, `changelog`,
`base`, `tone`, `minWidth`, `pages`, `routes`, `latest` and `versions` (each version with `packageId`, `engines`,
`compatible`, `revoked`, `signature` and `createdAt`). Every URL is a files-route URL. A listing with no listed version
is 404 "That theme isn't in the Marketplace."

`POST /api/marketplace/themes/submit` answers `{package: {id, themeId, version, name, digest, ownerType, status,
reviewed, scanVerdict, scanReport}, duplicate}`, the report without the AI code read.

| Status | Reason |
|---|---|
| 400, 413 | The package fails a check, or is over the cap |
| 403 | The Marketplace is off; on the hosted edition, a caller with no user row; a reserved publisher without a platform admin |
| 409 | The publisher belongs to another author; a version already submitted with other files, or not higher than the theme's highest; the author already has a submission waiting (while `THEMES_ONE_OPEN_SUBMISSION` is on); the same bytes are being submitted right now |
| 422 | The package has no `source/` (`SOURCE_REQUIRED`) |
| 429 | The person submitted `THEME_SUBMISSIONS_PER_DAY` themes in the last 24 hours |
| 503 | Every review slot is busy (`Retry-After: 30`); the files couldn't be stored |

## The revocation feed

| Method and path | Who | Request | Answer |
|---|---|---|---|
| `GET /api/themes/revocations` | anyone | Optional `If-None-Match` | The feed, or 304 |

Hosted edition only: the local edition answers 404 (it pulls the feed). The feed:

```jsonc
{
  "schema": 1,
  "generatedAt": "<ISO 8601>",
  "revokedKeys": ["ed25519:<16 hex>"],
  "revocations": [
    { "id": "<uuid>", "digest": "<64 hex>" /* or "themeId" and "versionRange" */, "level": "warn" | "disable",
      "reason": "<one line>", "createdAt": "<ISO 8601>", "liftedAt": null | "<ISO 8601>" }
  ]
}
```

It carries only revocations written on the hosted server that cover a Marketplace version that was listed, and lifted
ones for `THEME_REVOCATIONS_LIFTED_DAYS` (365) after they were lifted. Headers: `Cache-Control: public, max-age=300`
and a weak `ETag` over everything but `generatedAt`. 503 when the signing keys file doesn't load, or when the feed is
larger than a local edition would read (`THEME_REVOCATIONS_FEED_MAX_KB`, or 10,000 rows).

## The admin review queue

Every route checks `caller_is_admin` first: 403 "Only an Automatos admin can review Marketplace themes." A version
that is not a Marketplace submission answers 404, as a missing one does.

| Method and path | Who | Request | Answer |
|---|---|---|---|
| `GET /api/admin/themes/pending` | platform admin | `?limit=` 1 to 200 (default 50) | `{pending: [<summary> + reportNote + approval]}`, oldest first |
| `GET /api/admin/themes/{packageId}/report` | platform admin | none | The summary plus `manifest`, `scanReport` (whole), `reportNote`, `approval`, `history`, `signatureRequest`, `signatureVerified`, `keyId` |
| `GET /api/admin/themes/{packageId}/review-package` | platform admin | none | `{url, digest, expiresAt}`: a presigned link to the version's rebuilt package, for 15 minutes |
| `POST /api/admin/themes/{packageId}/report-note` | platform admin, step-up | `{"summary": "<1 to 20000 characters>", "outcome": "pass" \| "fail"}` | The summary plus `reportNote` |
| `POST /api/admin/themes/{packageId}/approve` | platform admin, step-up | none | The summary plus `signatureRequest`, `updatesOffered` (installs offered it) and `updatesApplied` (the unpinned ones moved to it) |
| `POST /api/admin/themes/{packageId}/reject` | platform admin | `{"reason": "<1 to 2000 characters>"}` | The summary plus `reason` |
| `POST /api/admin/themes/{packageId}/deactivate` | platform admin, step-up | `{"reason": "<1 to 2000 characters>"}` | The summary plus `reason`, `revocation: "disable"` and `installsRolledBack` (the unpinned installs moved back to the previous listed version) |
| `POST /api/admin/themes/{packageId}/signature` | platform admin, step-up | `{"requestId": "<uuid>", "signature": "<base64>"}` | The summary plus `signature: {requestId, signature, algorithm, verified, keyId}` |

A summary: `id`, `themeId`, `version`, `name`, `publisher`, `digest`, `status`, `reviewed`, `scanVerdict`,
`submittedBy`, `signed`, `createdAt`. `approval` is `{allowed, reason}`: what approve would answer now.

The signature request that approve opens: `{requestId, payload: {id, version, digest, files}, payloadSha256,
algorithm: "ed25519", form, canonical}`. `canonical` is the exact text the Design Kit's sign workflow signs.

| Route | Status | Reason |
|---|---|---|
| `report-note` | 400 | Blank text, a NUL character, or a `pass` note whose header lacks `Package digest:  <this digest>` and `Outcome: pass` |
| `report-note`, `reject` | 409 | The version is not pending |
| `approve` | 409 | Not pending; the scan is `blocked` or `not_run`; a `disable` covers the digest; on the hosted edition, no isolated-job note yet or the latest one says `fail`; the file list doesn't add up to the digest |
| `deactivate` | 409 | The version is not approved |
| `signature` | 400 | Not a 64-byte Ed25519 signature in base64; no active Automatos key verifies it (nothing is stored) |
| `signature` | 409 | Not approved; a signature an active key verifies is stored already; no open request, or another request id |
| `review-package` | 409 | A stored file is missing or doesn't match its hash, or the file list doesn't add up to the digest |
| any | 503 | The signing keys file doesn't load; storage is down; storage gives no `https://` link |

## Admin revocations

Every route checks `caller_is_admin` first: 403 "Only an Automatos admin can revoke a theme or lift a revocation." (or
"…see or run the theme revocation sync." on the sync routes).

| Method and path | Who | Request | Answer |
|---|---|---|---|
| `GET /api/admin/themes/revocations` | platform admin | `?limit=` 1 to 500 (default 100) | `{revocations: [<row>]}`: live ones first, then the newest |
| `POST /api/admin/themes/revocations` | platform admin, step-up | `{"digest", "level", "reason"}` or `{"themeId", "versionRange"?, "level", "reason"}` | 201 `{revocation, created: true}`, or 200 `{revocation, created: false}` when a live row here already says the same |
| `POST /api/admin/themes/revocations/{revocationId}/lift` | platform admin, step-up | none | `{revocation}` |
| `GET /api/admin/themes/revocations/sync` | platform admin | none | `{enabled, lastAttemptAt, lastSuccessAt, lastError, feedGeneratedAt, rows, revokedKeys, stale}` |
| `POST /api/admin/themes/revocations/sync/pull` | platform admin | none | The sync status after one pull |

A row: `id`, `digest`, `themeId`, `versionRange`, `level`, `reason`, `origin` (`local` or `feed`), `createdAt`,
`liftedAt`.

| Route | Status | Reason |
|---|---|---|
| create | 400 | Not exactly one target; a range with a digest; a digest not 64 lower-case hex; a theme id not `publisher/slug`; a range not `X.Y.Z` or `^X.Y.Z`; a level not `warn` or `disable`; a reason not one line of 1 to 2000 characters |
| lift | 404 | No such revocation |
| lift | 409 | A `feed` row while revocation sync works; the last live `local` disable of a deactivated Marketplace version |
| sync, pull | 404 | The hosted edition, which serves the feed and pulls nothing |
| pull | 409 | `THEME_REVOCATIONS_URL` is empty |
| pull | 502 | The pull failed; the detail is the error it recorded |

Related:

- [Theme host](_index.md)
- [Security model](security-model.md)
- [Operations](operations.md)
