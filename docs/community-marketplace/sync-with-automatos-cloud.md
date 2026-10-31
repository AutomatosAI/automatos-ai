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
- [orchestrator/api/marketplace_catalog_items.py](../../orchestrator/api/marketplace_catalog_items.py)
- [orchestrator/api/marketplace_cloud_items.py](../../orchestrator/api/marketplace_cloud_items.py)
- [orchestrator/services/marketplace_bundles/schema.py](../../orchestrator/services/marketplace_bundles/schema.py)
- [orchestrator/services/marketplace_bundles/catalog_items.py](../../orchestrator/services/marketplace_bundles/catalog_items.py)
- [orchestrator/services/marketplace_bundles/items_feed.py](../../orchestrator/services/marketplace_bundles/items_feed.py)
- [orchestrator/services/marketplace_bundles/sync_items.py](../../orchestrator/services/marketplace_bundles/sync_items.py)
- [orchestrator/services/marketplace_bundles/plugin_import.py](../../orchestrator/services/marketplace_bundles/plugin_import.py)
- [orchestrator/services/marketplace_bundles/import_items.py](../../orchestrator/services/marketplace_bundles/import_items.py)
- [orchestrator/services/marketplace_bundles/disclosure.py](../../orchestrator/services/marketplace_bundles/disclosure.py)
- [orchestrator/services/marketplace_bundles/updates.py](../../orchestrator/services/marketplace_bundles/updates.py)
- [orchestrator/modules/tools/discovery/cloud_install_asks.py](../../orchestrator/modules/tools/discovery/cloud_install_asks.py)
- [orchestrator/api/public_marketplace.py](../../orchestrator/api/public_marketplace.py)
- [orchestrator/config.py](../../orchestrator/config.py)
- [envs/api.defaults](../../envs/api.defaults)
- [frontend/components/marketplace/cloud/sync-button.tsx](../../frontend/components/marketplace/cloud/sync-button.tsx)
- [frontend/components/marketplace/cloud/copy.ts](../../frontend/components/marketplace/cloud/copy.ts)
- [frontend/components/marketplace/cloud/disclosure-dialog.tsx](../../frontend/components/marketplace/cloud/disclosure-dialog.tsx)
- [frontend/components/marketplace/cloud/update-notices.tsx](../../frontend/components/marketplace/cloud/update-notices.tsx)
- [frontend/lib/marketplace/cloud-install-gate.ts](../../frontend/lib/marketplace/cloud-install-gate.ts)
- [frontend/components/settings/automatos-cloud-panel.tsx](../../frontend/components/settings/automatos-cloud-panel.tsx)
- [frontend/lib/marketplace/cloud-files.ts](../../frontend/lib/marketplace/cloud-files.ts)
- [frontend/csp-sources.js](../../frontend/csp-sources.js)
- [frontend/components/marketplace/public/marketplace-page.tsx](../../frontend/components/marketplace/public/marketplace-page.tsx)
- [frontend/lib/marketplace/return-after-sign-in.ts](../../frontend/lib/marketplace/return-after-sign-in.ts)
- [frontend/app/robots.ts](../../frontend/app/robots.ts)

</details>

**Sync with Automatos Cloud** brings what Automatos Cloud lists into a local install's Marketplace: its themes, and
the agents, Playbooks, packages and plugins its review approved. It is pull only: the install reads Automatos
Cloud's public catalogue, lists the theme versions an Automatos key signed, and lists the other items as **Reviewed
on Automatos Cloud**. It never publishes anything, and it runs only when someone presses the button.

This page covers the local edition's side (the button, what leaves your install, the checks, the updates, the
agents, Playbooks, packages and plugins, and the settings) and then the hosted edition's side: the catalogue it
serves and the public Marketplace at `/marketplace`.

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
  a toast says what it brought, themes and items together ("2 new, 1 updated from Automatos Cloud", or "Nothing new
  from Automatos Cloud").
- **Settings › Automatos Cloud:** the URL in use (read-only), the last success, the last check, the last error, the
  last Sync's counts (new, updated, unchanged, refused; themes and items together), the themes held here from
  Automatos Cloud, **Sync now**, and how to turn it off.

Nothing schedules a Sync. The button is the only way one runs, and a second press while one is running is answered
"A Sync with Automatos Cloud is already running. Try again when it has finished."

## What leaves your install

A Sync starts with one anonymous request: `GET <AUTOMATOS_CLOUD_URL>/api/marketplace/catalog`.

- It carries exactly three headers: `Accept`, `Accept-Encoding: identity` and `User-Agent: Automatos/<version>
  (local)`. The version is the theme host's: it tells Automatos Cloud which Automatos is asking.
- It carries no cookie and no `Authorization`, and no account, workspace or install id. Nothing about who pressed the
  button, or what your workspace holds, is sent. Automatos Cloud sees the request's source address, as any web
  server does.
- For each new or changed agent, Playbook, package or plugin, the Sync sends one more request of the same kind for
  its bundle, and for a plugin one more for its package ([How a Sync brings them](#how-a-sync-brings-them)). Which
  ones to fetch is decided here: nothing about what your install holds is sent.
- Adding a Cloud theme sends one more request of the same kind, for that version's package
  ([Adding a Cloud theme](#adding-a-cloud-theme)).

What comes back is read with guards:

- **Where.** `AUTOMATOS_CLOUD_URL` must be an origin: `https://`, or `http://` to this machine only, with no user name
  or password and no path, query or fragment. Only the environment sets it, never the UI.
- **What is read.** A redirect is never followed (a 3xx fails the Sync), anything but 200 fails it, and an encoded
  (compressed) answer is refused. The size is capped, as declared and as streamed: `AUTOMATOS_CLOUD_FEED_MAX_KB` for
  the catalogue, `THEME_MAX_PACKAGE_MB` for a theme package, `AUTOMATOS_CLOUD_BUNDLE_MAX_KB` for an item's bundle and
  `PLUGIN_MAX_UPLOAD_SIZE_MB` for a plugin's package. Each read times out after `AUTOMATOS_CLOUD_TIMEOUT_SECONDS`,
  and the whole read after four times that.

Screenshots are the exception. A Cloud listing's screenshots load **in the browser**, straight from Automatos
Cloud's catalogue files route (`/api/marketplace/catalog/files/<digest>/<path>`), and the install proxies none of
them. That request is the browser's own, as for any web image: it sends the browser's address and, under the app's
`strict-origin-when-cross-origin` referrer policy, only the origin of the page.

## What a Sync checks

The catalogue is checked whole first. It fails the Sync when it is larger than `AUTOMATOS_CLOUD_FEED_MAX_KB`, isn't
UTF-8 JSON, isn't schema 1, has no valid `generatedAt` time or lists more than 1000 items.

Then each item is checked on its own. One that fails is refused and counted, never listed:

1. Its type is `theme`. An agent, Playbook, package or plugin is read as a Cloud item instead
   ([Agents, Playbooks, packages and plugins](#agents-playbooks-packages-and-plugins)); an item of any other type is
   skipped, neither refused nor a reason to fail the Sync.
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

The themes that passed are written in **one transaction**, as Marketplace listings marked **From Automatos
Cloud**:

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

A Sync stores no theme package byte. A Cloud theme listing holds the checked manifest, file list and signature; its
files are fetched when someone adds it.

When any step of the themes fails (the setting, the read, the catalogue, the signing keys file, the database), every
row is rolled back and only the attempt and its error are recorded: the last good Sync stays, and so do its
listings, and no agent, Playbook, package or plugin is read.

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

## Agents, Playbooks, packages and plugins

A Sync also brings the agents, Playbooks, packages and plugins Automatos Cloud's review approved. Each one is listed
in the tab of its type, marked **From Automatos Cloud** and **Reviewed on Automatos Cloud**, and is added like any
other Marketplace item, after you have read what it will use.

### Reviewed on Automatos Cloud, never Official

**Official** means bytes an Automatos key signed in CI, and themes are the only items signed that way. An agent,
Playbook, package or plugin is checked by Automatos Cloud's review instead: a person approved it there, and a
plugin also passed Automatos Cloud's security scan. Nothing in its bundle is signed, so this install never calls
one Official: every catalogue entry says `trust: "reviewed"`, and an entry that says anything else is refused. The
cards and the detail header say **Reviewed on Automatos Cloud**.

A bundle reaches your install over HTTPS from `AUTOMATOS_CLOUD_URL`, and its integrity rests on that connection.
This install checks what it can on its own: each bundle's sha256, its strict format, and a plugin's files
([Plugins](#plugins)).

### What Automatos Cloud lists

Only what a local edition doesn't already seed itself, each item at its newest version:

- **Agents:** approved, active community agents of the Marketplace, with no seeding slug, not system agents and
  not limited to a role.
- **Playbooks:** approved, public community Playbooks that are neither system Playbooks nor one of the seeded ones.
- **Packages:** the packages an Automatos admin made (every package that isn't seeded), when every member names
  something a local install can find: a seeded agent or Playbook, another item the catalogue lists, a skill, a
  plugin, a model or an app.
- **Plugins:** approved, active plugins whose security scan said safe.

An item is followed by where it came from, not by its row: on Automatos Cloud a resubmission is a new row, and here
it is a new version of the same item. Its id in the catalogue (`externalId`) is a hash of that origin, so it names no
row of Automatos Cloud. [Two edges](#two-edges-a-deleted-source-and-a-deleted-row) says when an item starts again.

### What a bundle holds, and what it never holds

Each item is one JSON document, its **bundle**, the same bytes on both sides. Its **revision** is the sha256 of those
bytes, so any change to a bundle is a new revision. A strict format decides what can be in it: a key the format
doesn't define, at any depth, refuses the bundle, so only these cross:

- **Every item:** its type, its `externalId`, its version, name, description, category, icon (an emoji or a short
  name, never an image address), tags and its publisher's display name.
- **An agent:** its type, its prompt and whether it uses it, its persona by slug (only a shared persona), its model's
  id and provider, its skills by name and plugin, the Composio apps it uses, and its plugins by slug.
- **A Playbook:** its steps (each step's own fields and its prompt template; a document step's title, format,
  template and data), the agents its steps name (as a seeded agent's slug, another catalogue item, or a name only),
  its inputs and outputs, its run settings and its apps.
- **A package:** its slug, vertical tags, matching signals, members (each named by a seeded agent's slug or a
  seeded Playbook's template id, another catalogue item, a skill's name, a plugin's slug, a model's id or an app)
  and its setup: questions, connections, guide steps and report templates.
- **A plugin:** its slug, long description, author, licence, what it contains (how many skills, commands, agents
  and hooks), recommended models, token estimate, and Automatos Cloud's scan report: the verdict, the static
  findings (type, severity, file, line and description) and the LLM review's status, risk score, summary and
  findings.

Never in a bundle: an agent's configuration (it holds runtime, CLI and machine settings), a Playbook's schedule (a
trigger's webhook address is its only key), the id of any row, a workspace, an owner, an email address, a
credential, an OAuth token or connection, a file path, an install count, or a quote of a plugin file's content.

### How a Sync brings them

The catalogue is read once. Its themes go through the checks above, then its other entries go through two passes:

1. **The plugin pass** comes first, so an agent's plugins, and the skills they make, are here before the agent is
   ([Plugins](#plugins)).
2. **The item pass** writes the agents, then the Playbooks, then the packages, in **one transaction**, so a
   Playbook finds an agent, and a package a member, that the same Sync wrote.

Each entry is taken in four steps:

1. **The entry.** Its type, `externalId`, version, revision, display fields and trust are checked. An entry whose
   revision this install already holds is **unchanged**: nothing is fetched for it.
2. **The bundle.** It is fetched from `<AUTOMATOS_CLOUD_URL>/api/marketplace/catalog/bundles/<revision>`. That
   address is built from your setting and the revision, never taken from the catalogue, and the read is guarded as
   the catalogue's is, capped at `AUTOMATOS_CLOUD_BUNDLE_MAX_KB`.
3. **The checks.** Its sha256 must be the revision. It must be UTF-8 JSON in the strict format, in its canonical
   form, and of the entry's type, `externalId` and version.
4. **What it names.** Each persona, skill, plugin, model and agent it names is looked for here: a shared persona, a
   global skill, an approved plugin, a model in this install's list, a seeded agent or Playbook, or another Cloud
   item. An agent or Playbook is still written when one isn't found: what is missing is recorded and shown before
   it is added ("Needs the skill …", "needs a helper"), and the install warns as it does for any item. A
   **package** member that isn't found refuses the package, because its install would stop halfway.

Each type is counted **new**, **updated**, **unchanged** or **refused**. One item that fails is refused alone, and the
Sync goes on. The Sync's counts, and its toast, add every type's to the themes'.

An update changes the same row in place: there is one row per item. Its version reads as Automatos Cloud's version
and the revision's first 12 characters (`1.0.0+3f9c0a1b2d4e`), and every copy a workspace makes records it.

If the plugin pass can't start, or the item pass fails, that pass is rolled back whole and its error recorded; the
themes, and the plugins already saved, stand.

What a Sync never touches:

- **Rows it didn't write.** A seeded agent, Playbook, package or plugin, one made or uploaded on this install, and
  a workspace's own rows are never read for a write.
- **Workspace copies.** An item a workspace added is its own copy. A Sync never changes it: a newer version is
  offered as an [update notice](#update-notices).
- **Names and slugs held here.** A Cloud agent is refused when a Marketplace agent here already has its name (in any
  case) as its name or slug, because a Playbook's install finds its helpers by name and would pick either; and when
  a package or Playbook here that didn't come from Automatos Cloud names an agent so. A package or plugin is
  refused when a row here already has its slug.
- **What the catalogue drops.** An item Automatos Cloud no longer lists stays here as it was.

### Plugins

Plugins have their own pass, one plugin at a time, each saved on its own. A Sync downloads at most
`AUTOMATOS_CLOUD_PLUGINS_PER_SYNC` new or changed plugins (5 by default, by name); the rest wait for the next Sync,
and the Sync's answer counts them in `plugins.waiting`.

Nothing of a plugin is stored until every check has passed:

1. Its bundle, checked as above.
2. Its slug: a plugin that didn't come from Automatos Cloud and has the same slug refuses it. So does a changed
   bundle at a plugin version this install already holds, because its files are kept by version.
3. Its package, fetched from `<AUTOMATOS_CLOUD_URL>/api/marketplace/catalog/plugins/<slug>/<version>/package.zip`
   (built from your setting and the bundle's slug and version) and capped at `PLUGIN_MAX_UPLOAD_SIZE_MB`.
4. An upload's zip rules: at most 500 files, at most 10 MB a file and `PLUGIN_MAX_UPLOAD_SIZE_MB` in all,
   compressed at most 100 times, no absolute path, no `..` and no file named twice.
5. Its `manifest.json` names the bundle's slug and version.
6. **This install's static scan** of its text files, the pattern checks an upload gets: a critical finding refuses
   it, and so do a line too long to scan and more than 200 findings. The LLM review never runs here: a Sync makes
   no model call.

Then its files are stored where an upload's are, its scan record keeps Automatos Cloud's report beside this
install's static result, and the plugin is saved **approved and active**. Its `SKILL.md` files become global skills,
except one with a critical or high finding. A failure before the save removes what was stored and saves nothing.

- **Approved on arrival.** The local edition has no plugin review page, so Automatos Cloud's review is the approval.
- **Enabled per workspace.** Each workspace still chooses: enabling a Cloud plugin is the same step, with the same
  permission, as any plugin's, after its review ([Before Add](#before-add-what-it-will-use)).
- **Updated by reference.** When Automatos Cloud lists a new version, the same plugin moves to it once the new files
  are stored and checked, and the old version's files are removed. Every workspace that enabled it runs the new
  version from then on, as plugins do on Automatos Cloud, and each one gets a notice
  ([Update notices](#update-notices)). A plugin turned off here stays off.

### Before Add: what it will use

Adding a Cloud item opens its review first. It is read from this install's rows; nothing is asked of Automatos Cloud.

- **Every item:** its name, publisher, Automatos Cloud's version, when it was synced, **Reviewed on Automatos
  Cloud**, and what this install lacks ("Needs the skill …", "Needs the helper agent …").
- **An agent:** its prompt as plain text (or "No custom prompt: it uses the default for its type", or its persona's
  prompt), its model and whether this install has it, the Composio apps it uses and which **needs a connection**,
  and its skills and plugins, each **on this install** or **not on this install**.
- **A Playbook:** each step's helper and prompt (a step whose helper isn't here **needs a helper**), the agents its
  install adds, and its apps.
- **A package:** its members and whether each is here, the connections its setup asks for, its agents' prompts and
  how many steps each of its Playbooks has.
- **A plugin:** Automatos Cloud's scan report, this install's static check, what it contains, the skills it made
  here, and "Updates apply to every workspace that enabled it".

**Add to my workspace** sends the install; **Close** sends nothing ("Nothing was added: you closed the review.").
The review comes before each of the Marketplace's four adds of a Cloud item: an agent, a Playbook, a package, and a
plugin's enable. The install itself is the one every Marketplace item has, with the same permissions and the same
warnings (an app that needs a connection, a model or a helper this install doesn't have). Text from Automatos Cloud
is shown as text: a prompt is never rendered as markdown or HTML.

### When Auto adds one from chat

Auto can add a Marketplace agent, package or plugin from chat. When the item came from Automatos Cloud, Auto asks
first with the approval card gated actions use, in the chat and in the approvals queue. The card holds the review
as text: the prompt (its first 4000 characters; the rest is in the Marketplace) or "No custom prompt", the apps and
which needs a connection, the model, what this install lacks, a plugin's two scan results, and **Reviewed on
Automatos Cloud**.

- **Approve** adds the item the card showed, once. **Deny** adds nothing.
- Neither full autonomy nor the owner's own request skips the card.
- If a Sync changed the item after its card was shown, approving that card adds nothing, and Auto asks again with a
  card that shows the item as it is now.
- A Cloud Playbook reaches a chat install only inside a Cloud package, whose card lists it.

Seeded items, and items made on this install, are added from chat as they always were.

### Update notices

The notices sit above the Marketplace's tabs, for the workspace you are in.

- **A newer agent or Playbook.** Your workspace holds a copy of a Cloud agent or Playbook, and none of its copies
  is at the version this install now holds: "… has a newer version on Automatos Cloud. Your copy stays as it is."
  **Review the new version** opens its review, and **Add to my workspace** adds a new copy beside yours (an
  agent is named `<name>-copy` when the name is taken, a Playbook `<name> (Copy)`). Your old copy is yours to keep
  or delete.
- **An updated plugin.** A Cloud plugin your workspace enabled before its update now runs the new version: "… was
  updated to … on Automatos Cloud; this workspace now runs it." **Review** opens its review, with no
  Add. **Dismiss** hides the notice for you, in this browser, until the plugin moves again.

### Two edges: a deleted source and a deleted row

- **A deleted source on Automatos Cloud.** Automatos Cloud follows an agent or Playbook by the row it was submitted
  from. If the submitter deletes that row there, its versions become new items: the one held here stays as it
  was, with no more updates, and the next resubmission arrives as a new item. A new agent with the name of an agent
  held here is refused until that one is removed here.
- **A Cloud row deleted here.** Deleting a Cloud agent, Playbook, package or plugin on this install deletes, with
  it, the record of where it came from, so the next Sync brings the item back as new. The agent and Playbook copies
  workspaces already added stay theirs, but get no update notice from then on.

## Settings

All five are backend settings, read from the environment only (`envs/api.defaults`, overridden in
`envs/api.local`).

| Setting | Default | What it does |
|---|---|---|
| `AUTOMATOS_CLOUD_URL` | `https://api.automatos.app` | The Automatos Cloud this install syncs from and downloads packages from. An origin only: `https://`, or `http://` to this machine; no user name, password, path, query or fragment. Empty turns the Sync off. |
| `AUTOMATOS_CLOUD_FEED_MAX_KB` | `1024` | The largest catalogue a Sync reads. A larger one fails the Sync. |
| `AUTOMATOS_CLOUD_TIMEOUT_SECONDS` | `15` | Seconds each read may take: the catalogue, one bundle or one package. The whole read may take four times that. |
| `AUTOMATOS_CLOUD_BUNDLE_MAX_KB` | `256` | The largest item bundle (an agent, Playbook, package or plugin) a Sync reads. A larger one is refused and counted. |
| `AUTOMATOS_CLOUD_PLUGINS_PER_SYNC` | `5` | The new or changed plugins one Sync downloads. The rest wait for the next Sync and are counted in `plugins.waiting`; `0` downloads none. |

A theme package's cap is the theme host's own `THEME_MAX_PACKAGE_MB` (32 by default), and a plugin package's is
`PLUGIN_MAX_UPLOAD_SIZE_MB` (10 by default): the same caps an upload has.

### Turning it off

Set `AUTOMATOS_CLOUD_URL=` (empty) in `envs/api.local`, then recreate the backend: `docker compose up -d backend`.

- The Marketplace's button and Settings › Automatos Cloud disappear. The Sync route answers 409, and the status
  route says `enabled: false`.
- Listings already synced stay listed, and themes already installed keep running from their stored files.
- Agents, Playbooks, packages and plugins already synced stay listed and can still be added, after their review:
  their rows, and a plugin's files, are stored here.
- Adding a Cloud theme whose files aren't stored here is refused: "… comes from Automatos Cloud, and this install
  doesn't sync with it now (AUTOMATOS_CLOUD_URL), so its files can't be fetched."

### Screenshots from another host

The web app shows a Cloud screenshot only from the catalogue files route of `https://api.automatos.app`, or of the
API origin the frontend was built with, and its Content-Security-Policy allows images from only those two. Point
`AUTOMATOS_CLOUD_URL` at another host and its listings show without screenshots unless the frontend is built for
that host. The Sync, the checks and installs work the same.

## The routes

Local edition only; the hosted edition answers 404 to all five, for every caller. The Sync's status and the Sync
itself need a workspace owner or admin (`workspace:manage`). The three item reads need any member of the workspace;
they read this install's rows only, and `/updates` only the caller's workspace's copies.

| Route | What it does |
|---|---|
| `GET /api/marketplace/cloud-sync/status` | `{enabled, cloudUrl, lastAttemptAt, lastSuccessAt, lastError, generatedAt, counts, themes, items, stale}`. `themes` is the number of themes held from Automatos Cloud, `items` the number of Cloud items held, by type (`{agent, playbook, package, plugin}`); `stale` is true when no Sync succeeded in the last 7 days. With `AUTOMATOS_CLOUD_URL` empty, `enabled` is false. |
| `POST /api/marketplace/cloud-sync` | Sync now (themes, plugins, then agents, Playbooks and packages), then apply the theme updates. 200 `{counts, items, plugins, updatesApplied, updatesFailed, status}`: `counts` is the whole Sync's, `items` each type's `{new, updated, unchanged, refused}`, `plugins` `{waiting}`; 409 while `AUTOMATOS_CLOUD_URL` is empty or another Sync is running; 502 with the error the Sync recorded. |
| `GET /api/marketplace/cloud-sync/items` | The Cloud items held here: `{agents, playbooks, packages, plugins}`, each item `{id, externalId, version, revision, name}` (a package's `slug` in place of `id`), the key its install takes. The Marketplace marks these items and reviews them before an add. |
| `GET /api/marketplace/cloud-sync/items/{kind}/{key}/disclosure` | One Cloud item's review ([Before Add](#before-add-what-it-will-use)): `kind` is `agent`, `playbook`, `package` or `plugin`, `key` its id, slug or plugin id. 404 for anything a Sync didn't write. |
| `GET /api/marketplace/cloud-sync/updates` | The caller's workspace's [update notices](#update-notices): `{updates: [{kind, cloudId, name, installed, available}]}`. |

---

## On the hosted edition: the catalogue and the public Marketplace

These routes and pages exist only on the hosted edition (a local install answers 404). None takes an account.

### The catalogue

`GET /api/marketplace/catalog` is the list every local install's Sync reads, schema 1: its themes, then its agents,
Playbooks, packages and plugins. It holds one theme item per listed Marketplace version that carries a signature an
active Automatos key verifies: its id, version, digest, name, description, publisher, engines, the signature
envelope, the file list, `theme.json` (base64) and the paths to its package, cover and screenshots. No private upload, pending, rejected or taken-off version is in it, and
nothing about a workspace, a submitter, a review or an install.

Each agent, Playbook, package and plugin is an entry of metadata only: `{type, externalId, version, revision, name,
description, category, icon, tags, publisher, trust, bundle}`, where `trust` is always `"reviewed"` and `bundle` is
its bundle's path. The entries are what [What Automatos Cloud lists](#what-automatos-cloud-lists) says. A bundle
the strict format refuses is left out and logged as an error, and one larger than `MARKETPLACE_CATALOG_BUNDLE_MAX_KB`
(256 by default) is left out and logged as a warning.

`GET /api/marketplace/catalog/files/<digest>/<path>` serves what the catalogue names and nothing else: a listed
version's `package.autotheme` and the screenshots its `theme.json` lists.

`GET /api/marketplace/catalog/bundles/<revision>` serves the bundle the current catalogue lists with that revision,
its canonical bytes, and nothing else.

`GET /api/marketplace/catalog/plugins/<slug>/<version>/package.zip` serves a zip of a listed plugin's stored files at
that version (the same files always make the same zip), and nothing else: a pending, unsafe or inactive plugin, or
another version, is 404.

All four are public by design. An approved agent's prompt and model, a Playbook's steps and prompts, and a plugin's
files can be read without an account; a bundle carries no email address, id, credential or connection.

- cached (`Cache-Control: public, max-age=300` with an ETag for the catalogue; a file, a bundle or a plugin package
  never changes under its address, so each is cached as immutable, with an ETag);
- rate-limited per client IP and per route: `MARKETPLACE_PUBLIC_RATE_PER_MINUTE` reads a minute on each (120 by
  default), over it 429;
- the catalogue is capped at `MARKETPLACE_CATALOG_FEED_MAX_KB` (1024 by default). A larger one is a logged 503,
  because served, it would fail every Sync.

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
| The plugins from Automatos Cloud couldn't be read, so no plugin or item was added; the themes stand. | The plugin pass couldn't start. The themes of the same Sync are listed; the next Sync tries again. |
| The items from Automatos Cloud couldn't be read, so none was added; the themes and plugins stand. | This install's record of its Cloud items couldn't be read, so no agent, Playbook or package was fetched. The next Sync tries again. |
| The items from Automatos Cloud couldn't be saved, so none was added; the themes and plugins stand. / The items from Automatos Cloud failed unexpectedly, so none was added; the server log has the detail. | The item pass rolled back whole. The next Sync tries again. |
| The Sync's plugin and item counts couldn't be saved; what it added stands. | What the Sync added stands, but the counts it recorded (Settings › Automatos Cloud) are the themes' only. |
| Nothing was added: you closed the review. | You closed a Cloud item's review without adding it. |
| Couldn't check where this item comes from. Try again. | The Marketplace couldn't read which items came from Automatos Cloud, so it added nothing rather than skip the review. |
| Open it in the Marketplace to see what it will use, then add it there. | A Cloud item was added from a page that can't show its review. |
| Finish the review that is open first, then add this. | One review at a time. |
| … couldn't be fetched from Automatos Cloud, so nothing was installed. | Adding a Cloud theme: the download failed. Try again. |
| Automatos Cloud's package of … isn't signed by an active Automatos key, so nothing was installed. | Adding a Cloud theme: the downloaded package failed its signature check. |

For the Themes tab itself, see [Marketplace › Themes](themes-tab.md). For what a theme is, see
[Themes](../themes/_index.md).
