# Theme security model

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this page:

- [orchestrator/services/themes/choices.py](../../orchestrator/services/themes/choices.py)
- [orchestrator/services/themes/optin.py](../../orchestrator/services/themes/optin.py)
- [orchestrator/services/themes/defaults.py](../../orchestrator/services/themes/defaults.py)
- [orchestrator/services/themes/records.py](../../orchestrator/services/themes/records.py)
- [orchestrator/services/themes/settings.py](../../orchestrator/services/themes/settings.py)
- [orchestrator/services/themes/contract.py](../../orchestrator/services/themes/contract.py)
- [orchestrator/services/themes/manifest.py](../../orchestrator/services/themes/manifest.py)
- [orchestrator/services/themes/css_scope.py](../../orchestrator/services/themes/css_scope.py)
- [orchestrator/services/themes/submissions.py](../../orchestrator/services/themes/submissions.py)
- [orchestrator/services/themes/listings.py](../../orchestrator/services/themes/listings.py)
- [orchestrator/services/themes/sdk_uses.py](../../orchestrator/services/themes/sdk_uses.py)
- [orchestrator/services/themes/review_stage.py](../../orchestrator/services/themes/review_stage.py)
- [orchestrator/services/themes/llm_stage.py](../../orchestrator/services/themes/llm_stage.py)
- [orchestrator/services/themes/llm_guard.py](../../orchestrator/services/themes/llm_guard.py)
- [orchestrator/services/themes/review.py](../../orchestrator/services/themes/review.py)
- [orchestrator/services/themes/serve.py](../../orchestrator/services/themes/serve.py)
- [orchestrator/services/themes/membership.py](../../orchestrator/services/themes/membership.py)
- [orchestrator/services/themes/storage.py](../../orchestrator/services/themes/storage.py)
- [orchestrator/services/themes/signing.py](../../orchestrator/services/themes/signing.py)
- [orchestrator/services/themes/signature_verify.py](../../orchestrator/services/themes/signature_verify.py)
- [orchestrator/services/themes/signature_state.py](../../orchestrator/services/themes/signature_state.py)
- [orchestrator/services/themes/signed_upload.py](../../orchestrator/services/themes/signed_upload.py)
- [orchestrator/services/themes/data/signing-keys.json](../../orchestrator/services/themes/data/signing-keys.json)
- [orchestrator/services/themes/revocations.py](../../orchestrator/services/themes/revocations.py)
- [orchestrator/services/themes/revocation_admin.py](../../orchestrator/services/themes/revocation_admin.py)
- [orchestrator/services/themes/revocation_feed.py](../../orchestrator/services/themes/revocation_feed.py)
- [orchestrator/services/themes/revocation_pull.py](../../orchestrator/services/themes/revocation_pull.py)
- [orchestrator/services/themes/revocation_sync_state.py](../../orchestrator/services/themes/revocation_sync_state.py)
- [orchestrator/services/themes/mobile.py](../../orchestrator/services/themes/mobile.py)
- [orchestrator/api/themes_files.py](../../orchestrator/api/themes_files.py)
- [orchestrator/core/auth/step_up.py](../../orchestrator/core/auth/step_up.py)
- [orchestrator/config.py](../../orchestrator/config.py)
- [frontend/app/api/themes/files/[...path]/route.ts](../../frontend/app/api/themes/files/[...path]/route.ts)
- [frontend/lib/theme-host/import-map.ts](../../frontend/lib/theme-host/import-map.ts)
- [frontend/lib/theme-host/shared-modules.json](../../frontend/lib/theme-host/shared-modules.json)
- [frontend/lib/theme-host/host-modules.ts](../../frontend/lib/theme-host/host-modules.ts)
- [frontend/lib/theme-host/engines.ts](../../frontend/lib/theme-host/engines.ts)
- [frontend/lib/theme-host/loader.ts](../../frontend/lib/theme-host/loader.ts)
- [frontend/lib/theme-host/sources.ts](../../frontend/lib/theme-host/sources.ts)
- [frontend/lib/theme-host/constants.ts](../../frontend/lib/theme-host/constants.ts)
- [frontend/lib/theme-host/boot-sanitize.ts](../../frontend/lib/theme-host/boot-sanitize.ts)
- [frontend/lib/theme-host/safe-mode.ts](../../frontend/lib/theme-host/safe-mode.ts)
- [frontend/lib/theme-host/failure-ledger.ts](../../frontend/lib/theme-host/failure-ledger.ts)
- [frontend/lib/theme-sdk/index.ts](../../frontend/lib/theme-sdk/index.ts)
- [frontend/components/theme-host/theme-host-provider.tsx](../../frontend/components/theme-host/theme-host-provider.tsx)
- [frontend/components/theme-host/protected-surface.tsx](../../frontend/components/theme-host/protected-surface.tsx)
- [frontend/app/safe-mode/page.tsx](../../frontend/app/safe-mode/page.tsx)

</details>

What a theme can do, what limits it, and where each limit lives. The CSS rules and the protected surfaces have their
own page, [CSS layer model](css-layer-model.md); this page links to it rather than repeat it.

## What a theme's code can reach

A theme's JavaScript runs in the member's browser, in the app's page, with the member's session. The host does not
sandbox it. The server says this in the opt-in dialog's own words (`OPT_IN_WORDING`): "This theme was not reviewed.
Its code runs with your access to all your workspaces." The same is true of a reviewed theme: review decides whether
the code is trusted, not what it can reach.

A theme is meant to reach data only through `@automatos/theme-sdk` and to draw only the interface. The limits are:

| Limit | Where |
|---|---|
| Who may put code in a workspace | The upload switches, "who may install", the reserved publishers (below) |
| Who runs it | Private code runs only for members who opted in to that version (below) |
| What it is | The automated review stage on every package; on the Marketplace, the isolated kit review job and a person (below) |
| What it can style | The CSS scope check, and the guards on the approval cards and the AI label ([CSS layer model](css-layer-model.md)) |
| What it can finish silently | Step-up on the sensitive routes (below) |
| Whether it still runs | Signatures and revocations, checked on every serve (below) |
| A way out | Safe mode and the failure ledger (below) |

## Private code and the opt-in

`choices.is_private` decides which code needs an opt-in:

| Package | Private |
|---|---|
| A Marketplace version not reviewed yet | Yes |
| A reviewed (approved) Marketplace version | No |
| A workspace's own upload | Yes, unless its stored signature verifies with an active Automatos key that neither the keys file nor the hosted feed revoked (`signature_state.signature_vouches`) |

For private code:

- A member runs it only after opting in to that exact package. `theme_opt_ins` is keyed by workspace, member and
  package, and a new version is a new package, so every version asks again.
- The opt-in route takes the package id the dialog showed, and answers 409 when that is no longer the install's
  current package, so a dialog left open across an update can't opt a member in to code they never saw.
- It can never be the workspace default (409), because a default runs for members who never said yes. An install
  that is the default can't be updated or rolled back to private code either.
- Until the member opts in: `GET /api/themes/active` says `optIn.required` and the host stops before it asks for any
  file; the boot route answers the base boot; the files route answers 404 for every file; the phone answer's `mobile`
  is `null`.

Who may upload private code at all (`records.private_upload_refusal`, `settings.py`):

| Control | Values | Rule |
|---|---|---|
| `THEMES_PRIVATE_UPLOAD` (platform) | `auto` (default), `on`, `off`, `admins` | `off` is a ceiling. `admins` lets only a platform admin upload. `auto` and `on` defer to the workspace. Any other value reads as `off`. |
| The workspace's `privateUploads` | `on`, `off`, unset | Unset is the default: `on`, except under `auto` on the hosted edition, where it is `on` only in a workspace of one person (its active members plus its owner) |
| "Who may install" | `admins` (default) or `owner` | Narrows uploads, Marketplace installs, updates, roll backs and pins (`require_install_role`). Changing it needs both the old and the new permission. |

## CSS scope, reserved classes and publishers

Every rule in `dist/theme.css` must sit under the theme's own `html.<htmlClass>`, with no `!important`, no
`@import`, and no selector naming a protected surface or the host's AI label. The full list of what the scope check refuses, and how the
approval cards and the AI label hold against theme CSS, is in [CSS layer model](css-layer-model.md).

**The AI label at run time (PRD-264 EUAI-11a).** The host checks that a person can see the label: displayed,
visible and opaque; not clipped, masked or filtered; at least 9 px; its words at least 3:1 against what is
behind them at every point, `::first-line` and `::first-letter` included, the ground read from what the page
paints under them (any element, its background colour and gradients faded by its opacity; a background image that
is no gradient fails); its words' own box a real line of text inside the
viewport and the label's box; and nothing drawn over them, click-through or not (hit-testing with a transient
sheet that makes everything hittable). Anything it can't confirm reads as hidden, and a check that throws counts
the label as hidden too. A hidden label is drawn again in a closed shadow root on a guarded element of the
host's, in the browser's top layer, and that copy heals itself: removed, moved, edited or hidden by a theme's
code, it is rebuilt at once and then checked every animation frame for as long as the tampering goes on, and
anything that enters the top layer later is put back under it. Chat is never locked. What stays a review rule
(what a replaced element or another element's pseudo-element under the words draws, a backdrop-filter, a blank
font, a theme observer that races each frame's repair, code that patches browser methods before the copy exists)
is listed in
[CSS layer model](css-layer-model.md#guaranteed-at-run-time-and-what-stays-a-review-rule); the host does not
sandbox a theme's code.

**Reserved classes.** `htmlClass` is 3 to 32 characters of lower-case letters, digits and hyphens, starting with a
letter, and never one of the classes core owns on `<html>`: `studio`, `classic`, `dark`, `light`, `system`
(`contract.RESERVED_HTML_CLASSES`). A theme's `@keyframes` names and `@font-face` families must start with
`<htmlClass>-`.

**The first-paint data is checked twice.** `boot/prepaint.json` is held to a small CSS-value grammar on import and
again on every boot read (`css_value.py`, `serve.theme_prepaint`). The browser applies the boot only after
`sanitizeThemeBoot` drops any class, custom property name or value outside its own grammar: no `url()`, no `;`, `{`,
`}`, quotes, `<`, `!` or `@` (`frontend/lib/theme-host/boot-sanitize.ts`, over `css-value.ts` `isBootValue`). The
import also refuses a knob or pair cell's value that grammar would drop (`first_paint.py`), so what passes the import
is what the browser paints. The boot script uses `classList` and `style.setProperty` only.

**Reserved publishers.** A theme id is `publisher/slug`. The publisher `automatos` is reserved
(`contract.RESERVED_PUBLISHERS`). For Marketplace submissions and the Official badge, a `theme_publishers` row marked
`reserved` counts too:

- A Marketplace submission under a reserved publisher needs a platform admin (403 otherwise). Any other publisher
  belongs to the first person who submits under it; a submission by anyone else is 409.
- A private upload under `automatos/…` needs a platform admin, or a signature an active Automatos key verifies.
- A reserved name alone is never **Official**: the listing is Official only when the version shown also carries a
  signature an active key verifies (`listings._publisher`).

**Host-owned routes.** A theme can't register a page for sign-in and sign-up, `/settings`, `/team`, `/admin`,
`/marketplace`, `/looks`, `/safe-mode`, `/x`, `/api`, `/dev` or `/verify` (`contract.HOST_OWNED_ROUTES`).

## The SDK boundary

A theme imports one module from the host, `@automatos/theme-sdk` (`frontend/lib/theme-sdk/index.ts`). The
[SDK surface map](sdk-surface-map.md) lists every name and the core module each one replaces.

How the browser enforces the boundary:

- **The import map.** Its `imports` are built from `shared-modules.json` alone: `react`, `react/jsx-runtime`,
  `react-dom`, `react-dom/client` and `@automatos/theme-sdk`, each mapped to a shim under `/theme-sdk/v1/` that
  re-exports the host's own instance (`host-modules.ts`). Nothing from a request goes into `imports`.
- **Integrity.** For the active installed theme, the import map pins every JS module of its package by
  `sha256-<base64>`, from the file manifest. An entry is kept only when its URL is a module of the files route
  (`/api/themes/files/<64 hex>/<path>.js`, no `..`) and its value a sha256 (`import-map.ts`, `sanitizeIntegrity`).
- **The engines check.** The loader refuses a theme whose `engines.sdk` is not an exact version or a caret range that
  accepts the host's SDK version, before the entry is requested (`engines.ts`). Wildcards and `>=` are refused.
- **Registration.** `register(host)` may register only the host's slot ids, the nine core pages a theme may replace,
  and routes that match `THEME_ROUTE_PATTERN`. An unknown key, a value that isn't a component, or a second
  registration for the same key fails the load (`loader.ts`).

How the review checks it:

- The `uses` check reads `dist/` for imports from `@automatos/theme-sdk`. A name the host's SDK doesn't export is an
  error (the module can't link). A namespace or dynamic import of the SDK is a warning, since its names can't be
  checked one by one (`sdk_uses.py`).
- The `audioCapture` check reads `dist/` for a theme's own audio: `getUserMedia`, `SpeechRecognition` /
  `webkitSpeechRecognition` or `speechSynthesis` is an error naming each, so the version is blocked
  (`audio_capture.py`). A theme talks to Auto by voice only through the SDK's voice area, whose button the host draws
  with the AI voice caption and the stop control in a protected surface.
- The isolated kit review job runs the Design Kit's JS rules on `source/` (next section).

## The review

### The automated stage

It runs on every private upload and every Marketplace submission, before the route answers
(`review_stage.py`). The AI code read (`llm_stage.py`):

- reads the code the browser runs, `dist/entry.js`, `dist/chunks/*.js` and `dist/theme.css`, and also `source/` for a
  Marketplace submission; one score, the highest of every part;
- asks the model to report exfiltration, acting as the member, remote or dynamic code, deception (fake sign-in or
  approval forms, content over the approval cards, hiding, covering or restyling the host's AI label), abuse of the
  device, obfuscation, prompt injection, and audio captured or spoken in the theme's own code rather than through
  the SDK's voice area;
- wraps each part between delimiters carrying a fresh random nonce, so text in the package can't close the part or
  speak as the reviewer, and holds the score at the review threshold (20) whenever a file holds text aimed at an AI
  reviewer (`llm_guard.py`);
- lifts a chunk's score to at least 70 for a `critical` finding and 20 for a `high` one, whatever score the reply gave;
- is never safe by default: no key, a failed call, an unusable reply, the stage's deadline, the day's token budget or
  too many chunks make it at least a warning ("LLM stage did not run"), never a pass.

A score of 70 or more blocks the verdict; 20 or more needs review. A private upload installs whatever its verdict and
shows as private. A blocked Marketplace version can't be approved.

The person who uploaded or submitted a package never sees the AI code read's score, findings or summary, so nobody
can tune a package against the model's answers (`review_stage.member_view`).

### The isolated kit review job

A Marketplace version also goes through the Design Kit's **Theme review** workflow (`theme-review.yml`), started by
the reviewer. It downloads the package over `https://`, and does everything that touches the submission inside a
container with no network. It checks the package as core does, checks the digest against the one the admin queue
shows, runs the kit's JS rules (`@automatos/theme-rules`) on `source/`, and rebuilds `source/` with the kit's own
build preset. It has no secrets, and its token has no permissions. An error-level JS finding refuses the package;
warning-level ones are listed for the reviewer. The rules are listed in the kit's
[eslint-plugin README](https://github.com/AutomatosAI/automatos-design-kit/blob/main/packages/eslint-plugin/README.md).

The reviewer pastes the job's summary into the admin queue. A `pass` note is stored only when the kit's header block
holds `Package digest:  <this version's digest>` and exactly one `Outcome:` line, `Outcome: pass`
(`review.note_names_this_version`), so a note for another version, or a failed run, can't approve this one. On the
hosted edition, approve needs the latest note to pass. The local edition has no isolated job: there the admin's
approval is the review.

The orchestrator never runs the job, never runs submitter code, and never calls GitHub. The link it gives the job
(`GET /api/admin/themes/{packageId}/review-package`) is a presigned download of the version's package, rebuilt from
its stored files, valid for 15 minutes, and never logged.

### A person

Nothing moves a Marketplace version to `approved` but an admin calling approve (`review.py`). Approve refuses a
`blocked` or `not_run` scan, a version that isn't pending, and a digest a `disable` covers.

## The files proxy

The browser loads a theme's modules with a native `import()` and its clips with `<video src>`. Neither can carry a
Bearer token or an `X-Workspace-ID`, or take a cross-origin answer the import map's integrity pins. So the frontend
serves `/api/themes/files/<digest>/<path>` on the app's own origin (`frontend/app/api/themes/files/[...path]/route.ts`):

- Only a 64-hex digest and plain file-name segments (letters, digits, `.`, `_`, `-`; at most nine after the digest)
  reach the backend. Anything else is 404.
- The credential is the Clerk session's token on the hosted edition (signed out is 401), and the paired browser's
  device token from its `SameSite=Strict` cookie, or none, on the local edition.
- `Range` and `If-None-Match` go forward. Only the content and caching headers come back, never `Set-Cookie`.
  A refusal comes back with a fixed text, never the backend's body.
- A request with `Service-Worker: script` is refused (403): theme code never registers a service worker.

The backend's files route (`api/themes_files.py`, `serve.authorised_package`) decides, for one digest:

1. An approved, reviewed Marketplace version with that digest that no `disable` covers is served to any caller.
2. Otherwise the caller must belong to a workspace with an active install of a row with that digest, and that row
   must belong to the install: a Marketplace version (approved, or deactivated after it was listed), or a private
   upload of **that** workspace. A private package referenced from another workspace's install serves nothing.
3. Private code also needs the caller's opt-in to that version in that workspace.
4. A row a `disable` covers serves nothing, and the next authorised row is tried.

**Membership** (`services/themes/membership.py`) is read from the database, never from the request's workspace: a
person signed in with Clerk belongs to every usable workspace where they are an active member or the owner; the local
edition's operator to every workspace; a paired device to exactly the workspace it is bound to; anyone else to none.

Every read is checked against the package's file manifest: the path must be listed, the stored bytes must hash to the
listed sha256 (a mismatch is logged as tampering and serves nothing), and the listed type is the type served. SVG is
never served. Every answer carries `X-Content-Type-Options: nosniff` and
`Content-Security-Policy: default-src 'none'; sandbox`, so a file opened as a page runs nothing, and a `source/*.js`
file is served as plain text. Anything the caller may not load is 404, never 403, so a digest says nothing about
whether it exists.

Private uploads are stored per workspace (`themes/_workspace/<workspace id>/…`), so two workspaces that upload the
same bytes never share files.

## Step-up

`require_step_up` (`core/auth/step_up.py`) guards the routes that install, vouch for or take down code:

- `POST /api/themes/import`, `POST /api/marketplace/themes/{listingId}/install`, `POST /api/marketplace/themes/submit`;
- `POST /api/themes/installs/{installId}/opt-in`, `PUT /api/themes/default`, `POST …/update`, `POST …/rollback`,
  `PUT /api/themes/settings`;
- `POST /api/admin/themes/{packageId}/report-note`, `…/approve`, `…/deactivate`, `…/signature`;
- `POST /api/admin/themes/revocations` and `…/{revocationId}/lift`.

On the hosted edition a request passes only from a person signed in with Clerk whose token's `fva` claim says the
first factor was verified less than `STEP_UP_MAX_AGE_MINUTES` ago (10 by default), counting the token's own age. Every
other caller (an API key, an SDK key, an anonymous one) is refused, since none of them has a person who can sign in
again. The refusal is a 403 carrying Clerk's reverification error, which the frontend turns into a sign-in prompt.
The local edition always passes: it has one operator and no sign-in to repeat.

The rule is a window, not a per-action challenge: for `STEP_UP_MAX_AGE_MINUTES` after a sign-in, code in the page can
complete these routes without a prompt.

Each route lists its permission gate before step-up, so a caller refused anyway is never asked to sign in again.
Choosing a theme, uninstalling one and rejecting a submission don't ask.

## Signatures

A signature vouches for one theme version. It is an Ed25519 signature, by an Automatos signing key, of the canonical
text of `{id, version, digest, files}`: the theme id, the version, the package digest and each file's sha256, as JSON
with sorted keys, no spaces and ASCII only (`signing.canonical_text`).

The orchestrator never holds a signing key. The private key is a secret of the Design Kit's `theme-signing`
environment, read only by its **Theme sign** workflow; a person hands it each signature request and pastes the
signature back ([Operations](operations.md#list-an-official-theme)).

**The keys file.** `orchestrator/services/themes/data/signing-keys.json` holds the public keys, the same bytes as the
kit's `keys/theme-signing.json`:

```json
{"schema": 1, "keys": [{"keyId": "ed25519:<16 hex>", "spki": "<base64 SPKI DER>", "status": "active", "added": "YYYY-MM-DD"}]}
```

`keyId` is `ed25519:` and the first 16 hex characters of the sha256 of the SPKI DER. `status` is `active`,
`retired` or `revoked`. A file that breaks any rule, or names a key twice, doesn't load. The file ships one key,
Automatos's theme-signing key (`ed25519:e138b6d0fa7aaaf3`, `active`, added 2026-10-26); no version is signed with it
yet.

**Verdicts** (`signature_verify.verify`): `verified` when an active key verifies the signature; `invalid` when no key
does, or only a retired or revoked one does (that key's id is kept); `unsigned` when there is none. In the local
edition a key the hosted feed lists as revoked reads `invalid` too (`signature_state.feed_checked`).

What a key's status does to the versions it signed:

| Key status | The version runs | Official | A signed private upload counts as reviewed |
|---|---|---|---|
| `active` | Yes | Yes, under a reserved publisher | Yes |
| `retired` | Yes | No | No: members opt in again |
| `revoked` (in the keys file, or, in the local edition, by the feed) | No: read as `disable` until signed again | No | No |

A stored signature that no key verifies, or whose file list no longer adds up to its digest, also reads as `disable`
(`signature_state.signature_fails`). This disable is built on each read and never stored, so a new signature lifts it
on the next load. The admin queue takes a new signature for a version whose stored one no longer verifies.

Where signatures are checked:

| Where | Check |
|---|---|
| `POST /api/admin/themes/{packageId}/signature` | Stored only when an active key verifies it (400 otherwise) |
| `POST /api/themes/import` with a `signature` part | The `.sig.json` envelope must name the upload's id, version and digest, and the key it names must be the active key that verifies it |
| Every serve (`revocations.revocation_of`) | A signature that fails disables its version |
| The listings, the installed list, the admin report | `{state, keyId}` shown |

The on-load answers are cached, at most 1024 per cache, keyed by the package, its digest, its signature, a fingerprint
of its file list and a fingerprint of the key set (the keys file's bytes and the feed's revoked keys). A change to
any of them is checked again.

**A keys file that doesn't load** fails closed where it vouches and stays out of the way where it would only break
things: the listings and the installed list show every signature `invalid` (nothing is Official), a signed private
upload no longer counts as reviewed, a signature paste or a signed upload is 503, and the hosted feed is 503. The
on-load check is skipped (logged as an error), so signed versions keep running.

## Revocations

A revocation (`theme_revocations`, read by `revocations.py`) names a package digest, or a theme id with an optional
version range (`X.Y.Z` or `^X.Y.Z`), at one of two levels:

| Level | Effect |
|---|---|
| `warn` | The theme still runs; the host shows the reason |
| `disable` | Members see the theme's base; the files route answers 404 and the boot route the base boot; `active` answers the base with a top-level `revoked`; the listings hide the version; nothing may install, update, roll back or be made default to it |

Both readings fail closed: a version range the code can't read covers every version, and a level it doesn't know
counts as `disable`. A lifted row (`lifted_at` set) no longer counts. Rows are never deleted.

Deactivating a Marketplace version writes a `disable` of its digest, and the last live local disable of a deactivated
version can't be lifted.

### The hosted feed

The hosted edition publishes `GET /api/themes/revocations` (`revocation_feed.py`) with no credentials. It carries
only revocations written on the hosted server that cover a Marketplace version that was listed: a revocation that
covers only a workspace's private upload, or a submission never approved, never appears, since a private digest or id
says something about a workspace. It also carries `revokedKeys`, the keys file's revoked keys.

The feed is not signed. HTTPS and a strict parser protect it: the local edition refuses the whole feed for more than
`THEME_REVOCATIONS_FEED_MAX_KB` (1024), more than 10,000 rows or 1,000 keys, a schema other than 1, any malformed
row, a reason that isn't one line of plain text, a time without an offset or an id twice.

### The local pull

The local edition copies the feed into its own `theme_revocations` (`revocation_pull.py`):

- **Where from.** `THEME_REVOCATIONS_URL`, by default the hosted feed. It must be `https://`, or `http://` to a
  loopback host, and carry no user name or password. Empty turns the pull off. Only the environment sets it.
- **What is sent.** One `GET` with `Accept`, `Accept-Encoding: identity` and a `User-Agent`, from a client of its own:
  no `Authorization`, cookie or workspace header. Redirects are never followed. Each read times out after 10 seconds
  and the whole read after 30.
- **What changes.** Each row is written by its hosted id with `origin` `feed`: inserted when new, refreshed when it is
  already a feed row. It is lifted only when the feed says `liftedAt`, and never un-lifted. A row missing from the
  feed stays. A `local` row is never written. `revokedKeys` replaces the stored list.
- **Fail closed.** Any failure rolls back every change and records only the attempt and its error.

A feed row is the hosted list's to lift. An admin may lift one locally only while sync is off or stale (no
successful pull in 7 days), and the lift is logged with who did it.

## Safe mode and the failure ledger

**Safe mode** (`frontend/lib/theme-host/safe-mode.ts`) is `?safe_mode=themes` on any address, or the `/safe-mode`
page. It is remembered for the tab in `sessionStorage`. The host then loads no theme code, and takes the theme's
first paint (its classes and prepaint variables) off `<html>`, so what shows is Studio. The `/safe-mode` page stands
outside the theme's frame and calls no backend route. Safe mode can only turn theme loading off.

**The failure ledger** (`failure-ledger.ts`) records each theme's own failures in the browser's `localStorage`. The
third within 10 minutes sends the member back to Studio: the server's choice (`PUT /api/themes/choice`), the style
cookie and the page. Only the theme's own failures count (`render`, `load`, `register`, `look`, `engines`): a refused
session, a revocation, a missing opt-in, an outage of the files proxy and a manifest refused before any code runs do
not. Every failure, counted or not, falls back to Studio for that load.

## The phone gets data only

The phone app never loads a theme's JavaScript or CSS. `GET /api/themes/mobile/{installId}` answers colours as
tokens, and URLs only for the font and mascot files the block names, never for a JS or CSS file. The same membership
and opt-in rules as the files route apply. See [The phone's theme](mobile.md).

Related:

- [Theme host](_index.md)
- [API reference](api-reference.md)
- [Operations](operations.md)
- [CSS layer model](css-layer-model.md)
