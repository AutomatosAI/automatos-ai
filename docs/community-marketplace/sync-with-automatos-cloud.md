# Sync with Automatos Cloud

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this page:

- [orchestrator/api/marketplace_cloud_sync.py](../../orchestrator/api/marketplace_cloud_sync.py)
- [orchestrator/services/marketplace_cloud/fetch.py](../../orchestrator/services/marketplace_cloud/fetch.py)
- [orchestrator/services/marketplace_cloud/feed.py](../../orchestrator/services/marketplace_cloud/feed.py)
- [orchestrator/services/marketplace_cloud/pull.py](../../orchestrator/services/marketplace_cloud/pull.py)
- [orchestrator/services/marketplace_cloud/package.py](../../orchestrator/services/marketplace_cloud/package.py)
- [orchestrator/services/marketplace_cloud/updates.py](../../orchestrator/services/marketplace_cloud/updates.py)
- [orchestrator/services/marketplace_cloud/urls.py](../../orchestrator/services/marketplace_cloud/urls.py)
- [orchestrator/api/marketplace_catalog.py](../../orchestrator/api/marketplace_catalog.py)
- [orchestrator/api/public_marketplace.py](../../orchestrator/api/public_marketplace.py)
- [orchestrator/config.py](../../orchestrator/config.py)
- [envs/api.defaults](../../envs/api.defaults)
- [frontend/components/marketplace/cloud/sync-button.tsx](../../frontend/components/marketplace/cloud/sync-button.tsx)
- [frontend/components/marketplace/cloud/copy.ts](../../frontend/components/marketplace/cloud/copy.ts)
- [frontend/components/settings/automatos-cloud-panel.tsx](../../frontend/components/settings/automatos-cloud-panel.tsx)
- [frontend/lib/marketplace/cloud-files.ts](../../frontend/lib/marketplace/cloud-files.ts)
- [frontend/csp-sources.js](../../frontend/csp-sources.js)
- [frontend/components/marketplace/public/marketplace-page.tsx](../../frontend/components/marketplace/public/marketplace-page.tsx)
- [frontend/lib/marketplace/return-after-sign-in.ts](../../frontend/lib/marketplace/return-after-sign-in.ts)
- [frontend/app/robots.ts](../../frontend/app/robots.ts)

</details>

**Sync with Automatos Cloud** brings the themes Automatos Cloud lists into a local install's Marketplace. It is pull
only: the install reads Automatos Cloud's public theme catalogue and lists the versions an Automatos key signed. It
never publishes anything, and it runs only when someone presses the button.

This is the technical page. It covers the local edition's side (the button, what leaves your install, the checks,
the updates and the settings) and then the hosted edition's side: the catalogue it serves, its files, the public
reads and their rate limit, and the public Marketplace at `/marketplace`. For what a person can do and where, see
[The open Marketplace](open-marketplace.md).

Themes are the only type that syncs. An item of any other type in the catalogue is skipped, neither listed nor a
reason to fail the Sync.

---

## Who can sync, and where

- **The local edition only.** The hosted edition is Automatos Cloud: it serves the catalogue and has no Sync.
- **Workspace owners and admins.** The local operator is the workspace owner. Other members don't see the controls,
  and the server refuses them (403).
- **While the Sync is on.** `AUTOMATOS_CLOUD_URL` is set by default; empty, it hides the button and the Settings tab
  ([Turning it off](#turning-it-off)).

There are two places to press it:

- **Marketplace header:** **Sync with Automatos Cloud**, with a status line beside it: "Last synced 2m ago · 3
  themes", "Not synced yet", or the last error in plain words. While a Sync runs the button shows a spinner; after it,
  a toast says what it brought ("2 new, 1 updated from Automatos Cloud", or "Nothing new from Automatos Cloud").
- **Settings › Automatos Cloud:** the URL in use (read-only), the last success, the last check, the last error, the
  last Sync's counts (new, updated, unchanged, refused), the themes held here from Automatos Cloud, **Sync now**, and
  how to turn it off.

Nothing schedules a Sync. The button is the only way one runs, and a second press while one is running is answered
"A Sync with Automatos Cloud is already running. Try again when it has finished."

## What leaves your install

A Sync sends one anonymous request: `GET <AUTOMATOS_CLOUD_URL>/api/marketplace/catalog`.

- It carries exactly three headers: `Accept`, `Accept-Encoding: identity` and `User-Agent: Automatos/<version>
  (local)`. The version is the theme host's: it tells Automatos Cloud which Automatos is asking.
- It carries no cookie and no `Authorization`, and no account, workspace or install id. Nothing about who pressed the
  button, or what your workspace holds, is sent. Automatos Cloud sees the request's source address, as any web
  server does.
- Adding a Cloud theme sends one more request of the same kind, for that version's package
  ([Adding a Cloud theme](#adding-a-cloud-theme)).

What comes back is read with guards:

- **Where.** `AUTOMATOS_CLOUD_URL` must be an origin: `https://`, or `http://` to this machine only, with no user name
  or password and no path, query or fragment. Only the environment sets it, never the UI.
- **What is read.** A redirect is never followed (a 3xx fails the Sync), anything but 200 fails it, and an encoded
  (compressed) answer is refused. The size is capped, as declared and as streamed: `AUTOMATOS_CLOUD_FEED_MAX_KB` for
  the catalogue, `THEME_MAX_PACKAGE_MB` for a package. Each read times out after `AUTOMATOS_CLOUD_TIMEOUT_SECONDS`,
  and the whole read after four times that.

Screenshots are the exception. A Cloud listing's screenshots load **in the browser**, straight from Automatos
Cloud's catalogue files route (`/api/marketplace/catalog/files/<digest>/<path>`), and the install proxies none of
them. That request is the browser's own, as for any web image: it sends the browser's address and, under the app's
`strict-origin-when-cross-origin` referrer policy, only the origin of the page.

## What a Sync checks

The catalogue is checked whole first. It fails the Sync when it is larger than `AUTOMATOS_CLOUD_FEED_MAX_KB`, isn't
UTF-8 JSON, isn't schema 1, has no valid `generatedAt` time or lists more than 1000 items.

Then each item is checked on its own. One that fails is refused and counted, never listed:

1. Its type is `theme`. An item of another type is skipped, neither refused nor a reason to fail the Sync.
2. Its id, version and digest are ones a theme can have.
3. Its package is the catalogue's own files path for its digest, or an `https://` address under the Automatos Design
   Kit's GitHub releases. That value is only checked, never fetched.
4. Its file list is one a theme package can have, it holds `theme.json`, and the digest of that list is the item's
   digest.
5. Its `theme.json` hashes to the file list's entry, is a valid theme manifest with the item's id and version, and
   names an Automatos this install meets.
6. Its signature verifies over that id, version, digest and file list, with an active Automatos key this install
   ships, and the key isn't one the revocation feed revoked. `theme.json` is one of the signed files, so the name,
   description and screenshot list you see are covered by the signature too.

What passed is written in **one transaction**, as Marketplace listings marked **From Automatos Cloud**:

- **new**: a theme that had no Cloud listing before;
- **updated**: a newer version of a theme from Automatos Cloud;
- **unchanged**: a version whose exact bytes (its digest) are already listed here;
- **refused**: an item that failed a check, a version number already listed here with other bytes, or a theme id
  your own install lists.

What a Sync never touches:

- **Your own listings.** A theme id that a listing made on this install holds is skipped whole and counted refused,
  so its installs are never offered a Cloud version.
- **Private uploads.** A workspace's own upload of a theme is never read for a write.
- **What the catalogue drops.** A version Automatos Cloud no longer lists stays listed here. A revoked one is the
  revocation pull's to disable, as it already is.

A Sync stores no package byte. A Cloud listing holds the checked manifest, file list and signature; its files are
fetched when someone adds it.

When any step fails (the setting, the read, the catalogue, the signing keys file, the database), every row is rolled
back and only the attempt and its error are recorded: the last good Sync stays, and so do its listings.

## Adding a Cloud theme

A Cloud listing is added like any other: **Add to my workspace**, by an owner or admin (see
[Marketplace › Themes](themes-tab.md#adding-a-theme-to-the-workspace)). Before the install, when the version's files
are not stored here yet:

1. The package is fetched from `<AUTOMATOS_CLOUD_URL>/api/marketplace/catalog/files/<digest>/package.autotheme`. That
   address is built from your setting and the listing's digest, never taken from the catalogue, so no other host is
   ever asked.
2. It is capped at `THEME_MAX_PACKAGE_MB` and read as a theme package.
3. Its digest, its id and version, and its file list must be the listing's.
4. Its signature is verified again, over the downloaded files, so a key retired or revoked since the Sync refuses it.
5. Only then are the files stored, and the install goes ahead.

If any step fails, nothing is stored and nothing is installed. The answer says why, naming the theme and version:
409 when the bytes are refused (413 past the size cap) or the Sync is off, 502 when Automatos Cloud can't be reached
or doesn't answer 200, 503 when storage or the signing keys file fails. A person updating an install to a Cloud
version gets the same download and checks first.

## Updates after a Sync, and pinning

After every successful Sync, each theme from Automatos Cloud that an install on this server runs is looked at.
When its newest version this Automatos can run came from Automatos Cloud and an install runs an older version:

- if an install is **unpinned**, the new version's package is fetched and checked first (as in
  [Adding a Cloud theme](#adding-a-cloud-theme));
- every older install is then offered the new version, and the unpinned ones move to it;
- a **pinned** install keeps the offer (**Update available**) and stays where it is until an owner or admin applies
  it. See [Managing themes](../themes/managing-themes.md#updates).

A download that fails changes no install. The toast says how many themes could not be updated ("Their installs stay
as they were; the next Sync tries again."), and the next Sync does try again: the update follows what is listed, not
what one Sync brought. A theme your own install lists never moves from here.

This is `services/marketplace_cloud/updates.py` `apply_cloud_updates`, run by `POST /api/marketplace/cloud-sync`
after the pull commits. It calls the two functions an approval calls, `listing_updates.offer_listed_update` and
`auto_update.apply_listed_update`, but never `review.approve`, so a Cloud row gets no signature request and no event
row. Each theme is committed on its own, one pass runs at a time per process, and nothing is fetched for a theme whose
older installs are all pinned. See [Updates and rollback](../theme-host/updates-and-rollback.md) for the moves
themselves.

## Settings

All three are backend settings, read from the environment only (`envs/api.defaults`, overridden in
`envs/api.local`).

| Setting | Default | What it does |
|---|---|---|
| `AUTOMATOS_CLOUD_URL` | `https://api.automatos.app` | The Automatos Cloud this install syncs from and downloads packages from. An origin only: `https://`, or `http://` to this machine; no user name, password, path, query or fragment. Empty turns the Sync off. |
| `AUTOMATOS_CLOUD_FEED_MAX_KB` | `1024` | The largest catalogue a Sync reads. A larger one fails the Sync. |
| `AUTOMATOS_CLOUD_TIMEOUT_SECONDS` | `15` | Seconds each read may take, the catalogue or one package. The whole read may take four times that. |

A package's cap is the theme host's own `THEME_MAX_PACKAGE_MB` (32 by default), the same cap an upload has.

### Turning it off

Set `AUTOMATOS_CLOUD_URL=` (empty) in `envs/api.local`, then recreate the backend: `docker compose up -d backend`.

- The Marketplace's button and Settings › Automatos Cloud disappear. The Sync route answers 409, and the status
  route says `enabled: false`.
- Listings already synced stay listed, and themes already installed keep running from their stored files.
- Adding a Cloud theme whose files aren't stored here is refused: "… comes from Automatos Cloud, and this install
  doesn't sync with it now (AUTOMATOS_CLOUD_URL), so its files can't be fetched."

### Screenshots from another host

The web app shows a Cloud screenshot only from the catalogue files route of `https://api.automatos.app`, or of the
API origin the frontend was built with, and its Content-Security-Policy allows images from only those two. Point
`AUTOMATOS_CLOUD_URL` at another host and its listings show without screenshots unless the frontend is built for
that host. The Sync, the checks and installs work the same.

## The routes

Local edition only; the hosted edition answers 404 to both, for every caller. Each needs a workspace owner or admin
(`workspace:manage`).

| Route | What it does |
|---|---|
| `GET /api/marketplace/cloud-sync/status` | `{enabled, cloudUrl, lastAttemptAt, lastSuccessAt, lastError, generatedAt, counts, themes, stale}`. `themes` is the number of themes held from Automatos Cloud; `stale` is true when no Sync succeeded in the last 7 days. With `AUTOMATOS_CLOUD_URL` empty, `enabled` is false. |
| `POST /api/marketplace/cloud-sync` | Sync now, then apply the updates. 200 `{counts, updatesApplied, updatesFailed, status}`; 409 while `AUTOMATOS_CLOUD_URL` is empty or another Sync is running; 502 with the error the Sync recorded. |

---

## On the hosted edition: the catalogue and the public Marketplace

These routes and pages exist only on the hosted edition (a local install answers 404). None takes an account.

### The catalogue

`GET /api/marketplace/catalog` is the list every local install's Sync reads, schema 1, themes only. It holds one item
per listed Marketplace version that carries a signature an active Automatos key verifies: its id, version, digest,
name, description, publisher, engines, the signature envelope, the file list, `theme.json` (base64) and the paths
to its package, cover and screenshots. No private upload, pending, rejected or taken-off version is in it, and
nothing about a workspace, a submitter, a review or an install.

`GET /api/marketplace/catalog/files/<digest>/<path>` serves what the catalogue names and nothing else: a listed
version's `package.autotheme` and the screenshots its `theme.json` lists.

Both are public by design:

- cached (`Cache-Control: public, max-age=300` with an ETag for the catalogue; a file never changes, so a file is
  cached as immutable);
- rate-limited per client IP and per route (`api/public_rate_limit.py`): `MARKETPLACE_PUBLIC_RATE_PER_MINUTE` reads
  a minute on each (120 by default), over it 429 (see [The rate limit](#the-rate-limit));
- the catalogue is capped at `MARKETPLACE_CATALOG_FEED_MAX_KB` (1024 by default). A larger one is a logged 503,
  because served, it would fail every Sync;
- built at most once every 30 seconds for every caller (`catalog.catalog_snapshot`). While one caller builds, the
  others get the last catalogue this process holds; with none held, a caller waits up to 2 seconds, then gets 503
  with `Retry-After: 5`. A package is rebuilt from its stored files in one of two slots per process; none free within
  a second is 503 with `Retry-After: 5`.

### The public reads

`GET /api/public/marketplace/{type}` (`api/public_marketplace.py`, `services/marketplace_public/`) answers what a
visitor sees of every tab but Themes. `type` is one of `agents`, `playbooks`, `packages`, `applications`, `plugins`,
`skills` and `llms`; another is 404. `?limit` is 1 to 100 (50 by default) and `?offset` 0 or more (else 422). The
answer is `{type, items, limit, offset, more}`: each type's approved Marketplace rows mapped through that type's
field allowlist (`services/marketplace_public/fields.py`), with nothing from a workspace, a person or an install. It is
cached like the catalogue (`Cache-Control: public, max-age=300`, a weak ETag, 304). A database that fails the read is
a logged 503.

### The rate limit

The three public routes, the catalogue, its files and the public reads, each have their own bucket per client
address: `marketplace-catalog:ip`, `marketplace-catalog-files:ip` and `marketplace-public:ip`.

- A bucket is one IPv4 address or one IPv6 /64 (`core/security/ip_bucket.py`), counted in the widget API's Redis
  sliding window over 60 seconds (`api/widgets/rate_limit.py` `RateLimitStore`).
- Over the limit is 429 "Too many requests from this address. Try again in a minute.", with `Retry-After` and
  `Cache-Control: no-store`, so no cache keeps the refusal for everyone.
- With Redis unreachable the request goes through, and the store logs an error naming the limiter.
- The edition is checked first: a local install answers 404 before the limit counts anything.

| Setting (hosted edition) | Default | What it does |
|---|---|---|
| `MARKETPLACE_PUBLIC_RATE_PER_MINUTE` | `120` | Reads a minute per client address on each public route. At least 1. |
| `MARKETPLACE_CATALOG_FEED_MAX_KB` | `1024` | The largest catalogue the hosted edition serves. Keep it at or under every install's `AUTOMATOS_CLOUD_FEED_MAX_KB`. |

### The public Marketplace

On the hosted edition, `/marketplace` is public: anyone can browse it signed out. A signed-in member sees the same
page as before.

- **What a visitor sees:** every tab, read-only. The Themes storefront from the catalogue, with its web and phone
  screenshots, and the other tabs (Packages, Applications, Agents, Playbooks, LLMs, and Capabilities' plugins and
  skills) from `GET /api/public/marketplace/<type>`, which answers each type's approved Marketplace rows with
  display fields only.
  Nothing a workspace holds is shown: no installed list, no update badges, no Publish form.
- **The header:** the Automatos logo, **Sign in** and **Start free**.
- **Add to my workspace, signed out:** goes to sign-in (or sign-up) and remembers the item for 30 minutes, in that
  browser tab only. Back signed in, the Marketplace opens on that item with its Add ready. **Nothing installs until
  the person clicks Add**, and the install is the normal one, with its permission and confirmation rules. The only
  page it ever returns to is `/marketplace`.
- **Sharing:** a link to an item shows that item's name, description and picture (a theme's cover) when posted in a
  chat or on social media, with one canonical address per item.
- **Search engines:** `robots.txt` allows `/marketplace` (with the logo and built scripts the page needs) and
  nothing else of the app, not even the signed-in pages under `/marketplace/`. A local install allows nothing.

---

## Messages

| You see | What it means |
|---|---|
| Sync with Automatos Cloud is off: AUTOMATOS_CLOUD_URL is empty. | The setting is empty: the Sync is off on this install. |
| AUTOMATOS_CLOUD_URL must be an https:// address (http:// only to this machine). | The setting isn't an origin a Sync may use. Two messages like it name a user name or password, or a path, query or fragment, in the setting. |
| Automatos Cloud couldn't be reached (…). / Automatos Cloud didn't answer in time. | The read failed or timed out. The last good Sync stays. |
| Automatos Cloud answered a redirect (…); redirects aren't followed. | Point `AUTOMATOS_CLOUD_URL` at the origin that serves the catalogue itself. |
| Automatos Cloud's answer is larger than … KB. | Over `AUTOMATOS_CLOUD_FEED_MAX_KB` (or, for a package, `THEME_MAX_PACKAGE_MB`). |
| This install's signing keys couldn't be read, so nothing was synced. | The theme host's keys file didn't load; nothing can be verified, so nothing is listed. |
| … couldn't be fetched from Automatos Cloud, so nothing was installed. | Adding a Cloud theme: the download failed. Try again. |
| Automatos Cloud's package of … isn't signed by an active Automatos key, so nothing was installed. | Adding a Cloud theme: the downloaded package failed its signature check. |

For the Themes tab itself, see [Marketplace › Themes](themes-tab.md). For what a theme is, see
[Themes](../themes/_index.md).
