# Ralph Review Prompt — PRD-270 W1 The open Marketplace

You are a fresh-context **adversarial reviewer**. The build claims PRD-270 Wave 1 is complete (US-270-00, 01, 02, 07a, 07b, 03a, 03b, 04a, 04b, 03c, 05a, 05b, 08a, 08b, 09a, 09b, 10, 06, 05c). This wave opens unauthenticated routes on SaaS, makes `/marketplace` public, and has every local install fetch a catalogue and theme code from the network. Review it as someone trying to:
- read a private upload, a configuration, a prompt or an email from a public route;
- make a local install fetch a URL of the attacker's choosing, or install bytes no Automatos key signed;
- make a Sync overwrite an operator's own listing, or move a pinned install;
- make a visitor's return link an open redirect, or an install happen without a click;
- make any route but `/marketplace` public.

## What to look for

**The migration and models (US-270-00):**
- the revision doesn't chain onto the base's head, or does more than K6's two steps;
- a step without its guard (`IF NOT EXISTS`, the DO block's catalog checks);
- the downgrade deleting a `theme_packages` row (it would cascade to installs);
- the model's CHECK text not equal to the DDL's;
- `test_prd261_theme_models.py` grown, a test lost, or a pin loosened beyond K31;
- a setting read outside `config.py`, missing from `config-surface.json`, or `AUTOMATOS_CLOUD_URL` defaulting to anything but `https://api.automatos.app`.

**The hosted catalogue and its files (US-270-01, 02):**
- an item for a private upload, a pending, rejected, deactivated or disable-revoked version, an unsigned one, or one an active key doesn't verify;
- a `packageId`, submitter, workspace, scan report or review datum in an item;
- an item whose envelope doesn't verify over its `files`, or whose `themeJson` doesn't hash to `files['theme.json']`;
- absolute URLs built from a request header;
- the files route serving anything but `package.autotheme` and a listed screenshot (README, theme.json, the entry, source files);
- the files route serving for an unlisted or private digest;
- a missing nosniff or sandbox header;
- a rebuilt package that doesn't read back to the row's digest;
- either route answering on a local edition, carrying `get_request_context_hybrid`, or unthrottled;
- `listings.py` behaviour changed for the member routes.

**The public reads (US-270-07a, 07b):**
- any field outside the allowlist, nested ones included (configuration, model_config, prompts, steps, inputs, outputs, setup_manifest, scan or security reports, S3 keys, workspace, owner, creator, user or clone ids);
- an email anywhere in a body;
- `Agent.created_by` used as a publisher;
- an unapproved, inactive or workspace-owned row;
- a workspace's installed, enabled or connected state;
- a member route edited;
- no rate limit or no cache header;
- answering on a local edition.

**The pull (US-270-03a, 03b):**
- a redirect followed, an encoded or oversize body read;
- an `Authorization`, `Cookie`, `X-Workspace-ID` or install id sent;
- a URL taken from the catalogue fetched;
- an item listed without the signature, digest and theme.json checks;
- a key the revocation feed revoked accepted;
- a local-origin row written, or a theme id it holds getting a Cloud version;
- a private upload touched;
- rows written in more than one transaction, or a failure leaving some;
- the status row lying about the counts;
- a Sync running on SaaS, with the URL empty, or on a schedule;
- `revocation_pull.py` or its siblings edited;
- the sync routes not 404 on SaaS for every role, or reachable by a member;
- a route missing from the manifest or not a plain `def`.

**Install and auto-update (US-270-04a, 04b):**
- any byte stored before the digest, manifest and signature checks;
- the download not capped at `THEME_MAX_PACKAGE_MB`;
- a host other than the configured Cloud origin requested;
- a failed write leaving files;
- an install or update of a Cloud row whose files aren't stored;
- `review.approve` called, or a signature request recorded for a Cloud row;
- a pinned install moved;
- a local listing's installs moved;
- `install_listing`, `apply_update`, `offer_listed_update` or `apply_listed_update` changed.

**The local storefront (US-270-03c, 05a, 05b):**
- a Cloud row's screenshot pointing at this install's files route, or a Cloud URL for a path that isn't a listed screenshot;
- `themeFileUrl` accepting any origin or path outside the exact Cloud files prefix;
- the CSP widened beyond the Cloud and API origins in `img-src`;
- the Sync button or panel shown on SaaS, to a member, or with the URL empty;
- gated by role instead of edition;
- `SystemSettingsTab.tsx` touched;
- `api-client.ts` grown.

**The public /marketplace (US-270-08a, 08b, 09a, 09b, 10):**
- `isPublicRoute` gaining anything but exactly `/marketplace`;
- a signed-in member's page changed;
- the visitor view showing workspace-scoped UI (installed list, update badges, Publish form, admin switcher) or calling a workspace or install route;
- the return path accepting anything but `/marketplace` with its query (`//host`, a scheme, `/chat`, a backslash);
- a return used after its expiry or twice;
- the auth pages or forms edited;
- an install request made on load with `add=1`;
- a big tab component touched instead of the deep-link seam;
- the share metadata calling an authenticated route, leaking anything a public read doesn't, or building `og:image` from a non-public URL;
- `robots` allowing more than `/marketplace` on SaaS, or anything on a local edition.

**Docs and the browser lane (US-270-06, 05c):**
- a local path, partner name or secret in a doc;
- `self-hosting.md` grown;
- the spec untagged (it would never run), reaching the real Automatos Cloud, or adding a dependency;
- `browser-e2e.yml` changed.

**The wave as a whole:**
- a story's evidence doesn't exist;
- an AC is marked DONE that the code doesn't meet;
- Phase 2 work built (K28);
- another kit's file changed;
- a file over 800 lines grew (the K3 exceptions aside);
- a K31 pin file lost a test;
- a new dependency, HTTP client or model call;
- the PR undrafted before this review passed.

You fix NOTHING in the code yourself.

## Scope

```
BASE=$(bash scripts/ralph/acceptance-prd270w1.sh --print-base)
git diff --stat $BASE..HEAD
git diff $BASE..HEAD -- . ':!scripts/ralph'
```

Never file a finding against a line this diff doesn't touch, except where an AC says the base's behaviour had to change and it didn't.

Read beside the diff:
- `scripts/ralph/prd-270w1.json` (binding: its `decisions` and each story's ACs and `notes`);
- `docs/PRDS/PRD-270-SYNC-WITH-AUTOMATOS-CLOUD.md` §5 (all of it), §6 (US-270-01..10) and §8;
- `AGENTS.md` and `frontend/DESIGN_SYSTEM.md` §22;
- the `ownerTest` list.

## Verification

- **code-review** skill, or the code-reviewer agent, on the diff, **in the foreground**.
- **security-reviewer** agent, in the foreground, on:
  - `orchestrator/services/marketplace_cloud/` (all of it) and `orchestrator/services/marketplace_public/` (all of it);
  - `orchestrator/api/marketplace_catalog.py`, `public_marketplace.py` and `marketplace_cloud_sync.py`;
  - the changed lines of `orchestrator/services/themes/listings.py`, `api/marketplace_themes.py` and `api/themes_install.py`;
  - `frontend/middleware.ts`, `frontend/lib/marketplace/`, `frontend/components/auth/resume-after-sign-in.tsx`, `frontend/components/marketplace/public/`, `frontend/components/marketplace/deep-link/`, `frontend/app/marketplace/page.tsx`, `frontend/app/robots.ts`, `frontend/csp-sources.js` and `next.config.js`'s CSP.

  Ask it explicitly about:
  - what an unauthenticated caller can read (the allowlists, nested fields, emails, private uploads);
  - SSRF and redirects in the pull and the download;
  - bytes stored before verification;
  - the revoked-key path;
  - an operator's own listing overwritten or auto-updated;
  - the open-redirect surface of the return path;
  - exactly one public page;
  - the CSP change.
- **database-reviewer** agent, in the foreground, on `orchestrator/alembic/versions/prd270_w1_cloud_sync.py`, the model changes, the Sync's one transaction (`services/marketplace_cloud/pull.py`) and the update moves (`services/marketplace_cloud/updates.py`).
- Run `bash scripts/ralph/acceptance-prd270w1.sh` yourself. It waits for the wave's one CI run and the browser lane on the tested commit, and reads the PR body and its base.
- Spot-check five marked acceptance criteria at random against the code and the tests. Always include one from US-270-03a, one from US-270-04a and one from US-270-07a.
- **Nothing runs on the owner's machine.** Never end your turn with anything running in the background.

## Verdict

**No CRITICAL, HIGH or MEDIUM:** write a 7-line summary:
- (a) the hosted catalogue and its files: what is public, what is left out;
- (b) the public reads: the allowlists, the rate limit, SaaS only;
- (c) the pull: the guards, the checks, one transaction, local rows untouched;
- (d) install and auto-update: the derived URL, verification before storage, pins respected;
- (e) the local web: the Sync button, the badge, the panel, the CSP;
- (f) the public /marketplace: one public path, the visitor view, the return trip, share cards;
- (g) the owner's next step: the `ownerTest` list (images.yml first, then the SaaS deploy before a local Sync), then undraft the PR (`gh pr ready`).

Then `REVIEW_PASS` as the last line. REVIEW_PASS and REVIEW_FINDINGS appear on your last line only, never in prose. You never undraft the PR.

**Findings:**
1. Append `P270W1-RVW-n` stories to `scripts/ralph/prd-270w1.json`, n counting from 1. Same shape: id, title, description, acceptanceCriteria (with the cause, the change and the test), priority after the last story, passes false, notes.
2. Commit with `git commit -s -m 'chore(prd-270): W1 review findings → fix stories [skip ci]' -- scripts/ralph/prd-270w1.json`.
3. `REVIEW_FINDINGS` as the last line.
