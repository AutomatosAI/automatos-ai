# PRD-270: Sync with Automatos Cloud — a local install pulls the hosted Marketplace

**Status:** Draft for Gerard's review, 30 Oct 2026. Pull only. Publishing from a local install to the Cloud
Marketplace is a later PRD, after a security review (§3).
**Why now:** Gerard set up a new self-hosted (VPS) Automatos on 30 Oct: "my marketplace has the defaults, as a open
source user I think I'd like a button Sync with Automatos cloud, that pulls what's in SaaS … for now have the pull."
**Origin:** PRD-233 (29 Aug, owner decision): "Marketplace = two things. The seeded local catalog … offline, no auth,
stays. The community hub … pulling needs no account and nothing is ever paywalled; publishing needs a free Automatos
account, and pushes are monitored and approved." The hub is "Planned, not shipped"
(`docs/getting-started/self-hosting.md:1030`). PRD-233 Q5 (the hub's form) is still open; §8-Q5 answers it.
**Scope:** automatos-ai (core and web). The Design Kit and Automatos Go need nothing: Go reads the install it is paired with.
**Depends on:** PRD-261 (theme signing keys shipped with every install, signed Official listings, the publish
pipeline `theme-publish.yml` → `POST /api/marketplace/themes/publish`, auto-update on approval, the storefront);
the theme revocation pull (`services/themes/revocation_pull.py`), which this copies.

## 1. Introduction

A local edition seeds the same Marketplace catalogue as SaaS at every boot (agents, packages, Playbooks, personas,
skills), so a fresh install is useful offline. What it never gets is what SaaS adds on top: Official themes listed
through the kit's publish pipeline, agents and Playbooks people submitted and an admin approved, plugins and skills
imported from GitHub. "Sync with Automatos Cloud" is one button on a local install that pulls that approved content
from SaaS, anonymously, and lists it locally. Nothing installs until someone chooses it.

## 2. Goals

- On a local install, an owner or admin clicks **Sync with Automatos Cloud**; within a few seconds the Themes
  storefront lists every Official theme SaaS lists, tagged **From Automatos Cloud**, with its previews and phone
  screenshots.
- Installing one verifies its digest and the kit's signature against the keys the install ships; a package that
  fails is refused and nothing is half-written.
- After a later Sync that brings a newer version, unpinned installs move to it (PRD-261's auto-update).
- Phase 2: approved agents, Playbooks, packages and plugins from SaaS appear locally the same way.
- The only thing that leaves the install is an anonymous HTTPS GET: no cookies, no account, no workspace data.
- An operator can turn it off (`AUTOMATOS_CLOUD_URL=` empty hides the button).

## 3. Non-goals

- **Submitting from a local install to the Cloud Marketplace.** Gerard wants it for every user, after a security
  review: account linking, moderation queue capacity, abuse and secret scanning, rate limits, licence terms for
  shared content. Its own PRD.
- Background sync by default (§8-Q2).
- Syncing workspace data, chats, knowledge or anything per-person.
- The LLM catalogue: it has its own provider sync (`core.services.provider_catalog_sync`).
- A Cloud account on a local install.

## 4. What exists today

### 4.1 Marketplace content, by type

| Tab | Rows | How a row exists on SaaS beyond the seeds | Portable today? | Install |
|---|---|---|---|---|
| Themes | `theme_packages` (`owner_type` marketplace, `status`, `scan_verdict`, `signature`) | the kit's publish pipeline; reviewed submissions | **Yes**: the `.autotheme` package, digest, Ed25519 signature (kit CI) | `theme_installations`, by package |
| Agents | `Agent` (`owner_type='marketplace'`, `is_approved`) | `POST /api/marketplace/submit` + admin approve (`api/marketplace.py:756`, `:888`) | No: an in-process clone (`cascade_installer.clone_agent_to_workspace`) | clone + dependency cascade, with warnings |
| Playbooks | `WorkflowTemplate` (`owner_type`, `is_approved`) | the same submit/approve | No | clone |
| Packages | `MarketplacePackage` (`setup_manifest`) | seed/admin only; no submit route | Partly: `setup_manifest` JSON references seeded items | installs the dependency closure |
| Plugins / Skills | `MarketplacePlugin` (S3 zip, `security_status`, `approval_status`), `Skill` | admin GitHub import → scan → approve (`api/admin_plugins.py`) | Partly: the plugin zip | enable by reference; skills materialised |
| LLMs | `LLMModel` | provider catalogue sync | n/a (non-goal) | — |

Every catalogue row is global; only installs are per workspace.

### 4.2 The pattern to copy

`services/themes/revocation_pull.py`: `THEME_REVOCATIONS_URL` (default `https://api.automatos.app/api/themes/revocations`,
empty = off); `checked_url()` (https only, http to loopback only, no embedded credentials); a streamed size cap; no
redirects; no encoded bodies; one transaction, rolled back with the error recorded on any failure; enabled only when
`IS_LOCAL_EDITION`. Status and **Check now** behind `api/admin_theme_revocation_sync.py` (admin-gated). The hosted side
serves the feed (`api/themes_revocations_feed.py`, 404 on a local edition).

### 4.3 Trust anchors a local install already has

`services/themes/data/signing-keys.json` (the public half of `ed25519:e138b6d0fa7aaaf3`) and
`services/themes/signature_verify.py`. **Core holds no private key**: the private key lives only in the kit's
`theme-publish` GitHub environment. So core cannot sign a feed without adding a new private key (§8-Q3).

## 5. Design

### 5.1 The flow

1. An owner or admin of a local install clicks **Sync with Automatos Cloud** (Marketplace header; also Settings ›
   System › Automatos Cloud).
2. `POST /api/marketplace/cloud-sync` (local edition only; owner/admin) fetches
   `GET {AUTOMATOS_CLOUD_URL}/api/marketplace/catalog` with the revocation pull's guards.
3. Each item is checked, then written in one transaction as a local Marketplace row marked `source = cloud`
   (§5.3). A failure rolls back the whole sync and records the error; the last good sync stays.
4. The storefront and the other tabs show the rows like any listing, with a **From Automatos Cloud** badge.
5. Install is the existing install flow. For a theme, its bytes are fetched on install (§5.4), not at sync.

### 5.2 The hosted catalogue feed (SaaS only)

`GET /api/marketplace/catalog` — public, no auth, cacheable (`Cache-Control: public, max-age=300`), 404 on a local
edition, in the route manifest and `OWN_AUTH_ROUTES` (public). Schema 1:

```json
{"schema": 1, "generatedAt": "…", "items": [
  {"type": "theme", "id": "automatos/arcade", "name": "Arcade", "description": "…", "version": "1.0.0",
   "digest": "…", "engines": {"…": "…"}, "official": true, "signature": {"…": "the .sig.json envelope"},
   "package": "https://api.automatos.app/api/marketplace/catalog/files/<digest>/package.autotheme",
   "cover": "…/files/<digest>/assets/previews/overworld.png", "phoneCover": "…",
   "presets": [{"id": "overworld", "previews": [{"screen": null, "url": "…"}], "phonePreviews": []}]}
]}
```

- **Phase 1 lists themes only:** every approved Marketplace version whose signature verifies under an active key
  (Official), the newest version per theme plus the versions installs may still need.
- `GET /api/marketplace/catalog/files/{digest}/{path}`: public, serves only (a) the packed `.autotheme` and (b)
  files listed in `previews`/`phonePreviews`, and only for a digest that is an approved, signature-verified
  Marketplace version. Same content-type, `nosniff` and sandbox headers as the theme files route. Everything else
  404s. These bytes are public already (the kit's GitHub releases).
- Size: the whole feed under a configured cap (`MARKETPLACE_CATALOG_FEED_MAX_KB`, default 1024).

### 5.3 The local side

- Settings (config.py): `AUTOMATOS_CLOUD_URL` (default `https://api.automatos.app`; empty = off, button hidden),
  `AUTOMATOS_CLOUD_FEED_MAX_KB` (1024), `AUTOMATOS_CLOUD_TIMEOUT_SECONDS` (15). Read from the environment only
  (an operator decision, like `THEME_REVOCATIONS_URL`).
- Fetch: `checked_url()` reused; no cookies, no `Authorization`, no workspace or install id;
  `User-Agent: Automatos/<version> (local)` so SaaS can answer by version if a schema changes.
- A status row `marketplace_cloud_sync` (last attempt, last success, counts new/updated/unchanged/refused, last
  error), mirroring `theme_revocation_sync`.
- **Themes become local Marketplace listings.** For each feed theme: write a `theme_packages` row,
  `owner_type='marketplace'`, `source='cloud'`, with manifest, digest and signature, `status='approved'`, only when
  the signature verifies under the shipped active keys and the digest matches what the signature binds. The
  package bytes are not stored yet (§5.4). A newer version arriving runs `review.approve`'s auto-update path, so
  unpinned installs move to it. A version the feed no longer lists is left as it is; a revoked one is handled by the
  revocation pull, which already runs.
- A local row with the same theme id but local origin (a private upload, the operator's own listing) is never
  overwritten.

### 5.4 Installing a Cloud theme

The existing install route, plus: when the package's files are not stored locally, download the `.autotheme` from
the feed's `package` URL (only that host or `github.com/AutomatosAI/automatos-design-kit/releases/download/…`;
never a URL from anywhere else), cap it at the package cap, read it with `read_theme_package`, check the digest,
check the signature (`signed_upload.check_upload_signature`), store the files, then install. Any failure: 4xx with
the code, nothing stored.

### 5.5 The UI (local edition only)

- Marketplace header: **Sync with Automatos Cloud** (owner/admin), with "Last synced 2 min ago · 3 themes" or the
  last error in plain words. While syncing: a spinner; after: a toast ("2 new, 1 updated from Automatos Cloud").
- Every synced listing: a **From Automatos Cloud** badge beside Official.
- Settings › System › Automatos Cloud: the URL in use (read-only), last sync, **Sync now**, and how to turn it off
  (the env var), mirroring the revocations panel.
- Hidden entirely when `AUTOMATOS_CLOUD_URL` is empty or on SaaS (`lib/auth-edition.ts`, by edition, not role).

### 5.6 Phase 2: agents, Playbooks, packages, plugins

Needs a portable format first: an **item bundle** (JSON, schema 1) per approved item, exported by SaaS and imported
by the local install:

- agent: its configuration, prompt text, model hint, skills and tools **by slug**, Composio apps by slug, persona by
  slug; never credentials, OAuth tokens or workspace ids.
- Playbook: its steps, with agents referenced by bundle-local ids that resolve to bundled or seeded agents.
- package: its `setup_manifest`, referencing bundled or seeded items.
- plugin: the stored zip plus its scan report; installs stay subject to the local edition's own enablement.

Import writes `owner_type='marketplace'`, `is_approved=true`, `source='cloud'` rows, upserted by
`(source, external_id, version)`. Install stays the existing clone with its dependency warnings ("needs a Composio
key", "needs model X"). Before install the user sees the prompt text and the tools and apps the item will use.
Updates arrive as new versions with an update notice; a sync never overwrites a workspace's installed copy.

### 5.7 Security

| Risk | Mitigation |
|---|---|
| Spoofed or intercepted feed | HTTPS only, `checked_url`, the URL from the environment only, no redirects |
| Tampered theme package | the kit's Ed25519 signature, checked against shipped keys at sync (metadata) and at install (bytes) |
| Tampered unsigned items (phase 2) | §8-Q3 |
| SSRF through feed URLs | package URLs only on the feed's host or the kit's release host; images proxied by nothing (the browser loads them from the Cloud host) |
| Oversized feed or package | streamed caps, timeouts |
| Malicious approved content | SaaS moderation (themes: review + signature; plugins: scan); local install still needs a click; prompt text and tools shown first |
| Privacy | anonymous GET; no install id; version in the User-Agent only |
| Overwriting local work | `source='cloud'` rows only; local-origin rows and installed copies are never touched |
| Revoked content | the existing revocation pull disables it |

## 6. User stories

### Phase 1: themes (one wave, two lanes)

- **US-270-01 (hosted) — The catalogue feed.** `GET /api/marketplace/catalog` per §5.2, themes only; tests: lists
  only approved signature-verified versions; private uploads and pending/rejected/deactivated rows never appear;
  404 on local; size cap; cache header; route manifest + authZ sweep.
- **US-270-02 (hosted) — Public Cloud files.** `GET /api/marketplace/catalog/files/{digest}/{path}` per §5.2;
  tests: package and listed previews served; any other path, any unlisted digest, a private upload → 404; headers.
- **US-270-03 (local) — Pull and list.** Settings, fetch with the revocation guards, status row, signature-gated
  `source='cloud'` listings, newer version → auto-update; one transaction; tests: tampered signature refused,
  digest mismatch refused, partial failure rolls back, local-origin rows untouched, off when the URL is empty, off on SaaS.
- **US-270-04 (local) — Install a Cloud theme.** §5.4; tests: download host allowlist, cap, digest and signature
  checks, nothing stored on failure.
- **US-270-05 (web) — The button and the badge.** §5.5; tests: local owner/admin sees it, members don't, SaaS never
  does, the toast counts, the badge, the Settings panel.
- **US-270-06 (docs) — Self-hosting guide.** `self-hosting.md` "community hub" row → Themes shipped, the rest
  planned; `docs/community-marketplace/` gains "Sync with Automatos Cloud".

### Phase 2: the other types (after Phase 1 merges; one wave)

- **US-270-10** the item bundle format, with fixtures and a parity test. **US-270-11** SaaS exports approved items to
  the feed. **US-270-12** local import and upsert. **US-270-13** pre-install disclosure (prompt, tools, apps) and
  update notices. **US-270-14** plugins with their scan report.

## 7. Rollout

Phase 1 ships dark on SaaS (the feed is public but harmless) and on by default on local editions (§8-Q6).
Gerard's VPS is the first test: Sync, install Arcade, see it update after the next kit merge and Sync.

## 8. Decisions for Gerard (each recommendation stands unless you change it)

1. **Who can press Sync** — workspace owners and admins of the local install (rec); every member; operator only.
2. **When it syncs** — the button only (rec, as asked); plus an optional daily schedule behind a setting.
3. **Trust for phase 2's unsigned items** — (a, rec) only items whose bytes are signed in CI become **Official**;
   approved community items come over HTTPS labelled **Reviewed on Automatos Cloud**; (b) give core a feed-signing
   key (a new private key on SaaS, new rotation duty); (c) phase 2 only syncs signed bundles from GitHub repos.
4. **Where Cloud themes live locally** — as local Marketplace listings (rec: the storefront, install and
   auto-update just work) vs straight to private installs.
5. **PRD-233 Q5, the hub's form** — hybrid (rec): official content from GitHub repos through CI (as themes now),
   community content through the SaaS review queue, one public SaaS feed that every install pulls.
6. **Default** — on for every local install, the button visible (rec); off until an operator sets the URL.

## 9. Later

**Submit to Automatos Cloud** (its own PRD): a free account linked from the local install, the moderation queue,
secret and abuse scanning on submit, rate limits, licence terms, and a verified-publisher tier.
