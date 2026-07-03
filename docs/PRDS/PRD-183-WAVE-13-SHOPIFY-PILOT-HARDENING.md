# PRD-183: Wave 13 — Shopify Pilot Hardening

**Phase:** E — Vertical & flagship polish (weeks 32+)
**Branch(es):** `feat/w13-shopify-hardening` (automatos-ai) + `feat/w13-shopify-remix` (automatos-shopify)
**Dependencies:** Waves 1, 2, 7 (merged) + **Wave 11** (`erase_data_subject` entrypoint the GDPR webhooks call — in PR #499, merge first)
**Build size:** L (spans two repos) · **Risk:** Medium
**OS Review refs:** §10 (Shopify vertical), §12, roadmap Phase E
**Owner decision (locked 2026-07-03):** **BUILD the embedded Remix admin** (not retire/CDN-only).

---

## Overview

The proactive widget is the one genuinely-deployed autonomous leg (live on inbuilduk.com), but the pilot can't become a *reference* customer while the graph is manual-refresh-only, sync isn't a tool Auto can call, and the Shopify app doesn't build. This wave hardens the vertical on **both** sides of the seam and extracts the abstraction so vertical #2 doesn't fork `api/shopify.py`.

**Two repos, two worktrees, coordinated at one seam** — the Shopify Remix webhooks call platform endpoints:
- catalog webhook → platform `/events` (fixed in S1)
- GDPR `customers/redact` / `shop/redact` → platform `erase_data_subject` (built in Wave 11)

Fire the platform-side agent first (or together, but the Remix agent's webhook wiring targets the platform contracts). **Do not** let the Shopify agent touch `automatos-ai` or vice-versa.

---

## Part A — Platform-side (repo: `automatos-ai`, worktree `automatos-ai-prd183`)

### S1 · Catalog webhooks actually update the graph (F032) — S
**Files:** `orchestrator/api/shopify.py:491-518` (`/events`), the `_incremental_build` filter.
**Bug:** `/events` schedules pending dicts lacking the `type`/`id` keys `_incremental_build` filters on → every catalog webhook is a silent no-op. **Test:** `test_catalog_webhook_updates_graph` posts a product-update event and asserts the commerce graph changes incrementally. Fix the pending-dict shape so the incremental builder ingests it.

### S2 · Fix the detached-sync DB session (F033) — S
**Files:** `orchestrator/api/tools.py:270-288`.
**Bug:** first-connect auto-sync hands a request-scoped session to a detached background task, torn down mid-flight. **Test:** `test_autosync_uses_own_session` asserts the background task opens its own session (not the request's). Use a fresh session/`sessionmaker` in the task.

### S3 · Sync + freshness as platform tools (F088) — M
**Files:** the 3-file platform-tool registration pattern (`actions_*`, registry, permissions) for a Shopify **sync** tool + a **freshness/status** tool; authenticated (carries `RequestContext`, closes the F003 unauth-`/sync` gap on the tool path).
**Test:** `test_platform_sync_tool` — Auto can invoke "refresh the catalog graph" and "when did the graph last sync" through its own tool surface. This is the flagship "Auto, refresh the catalog and tell me what changed" moment.

### S4 · Codegraph reindex as a tool + working auto-reindex (F087, F022) — M
**Files:** `orchestrator/api/codegraph.py:115,683` (`auto_reindex` has no setter, only read), agent index/reindex platform tools.
**Test:** `test_codegraph_reindex_tool` — an agent can index/reindex a repo through a platform tool; `test_auto_reindex_setter` — the webhook path can fire a reindex (setter exists, no longer dead). Reach the commodity freshness bar (Cursor/Greptile have working push-reindex).

### S5 · VerticalProvisioner abstraction (F076) — L
**Files:** extract `api/shopify.py`'s lifecycle into `integrations/shopify/provision.py` behind a generic `VerticalProvisioner` interface + `/api/verticals/{v}/provision`; move the catalog/orders mappers (`modules/knowledge/graph_extraction.py:503,693`) behind the plugin registry; make proactive triggers plugin-declared; turn sync into a generic "graph source" tool.
**Test:** `test_vertical_provision_generic` — a second (mock) vertical provisions through the generic path without touching `api/shopify.py`. **Notes:** This is the big refactor — the write path never went through PRD-141 (only the widget read path did). Keep the CI vertical-isolation check (`scripts/ci/check-no-shopify-in-generic.sh`) green.

---

## Part B — Shopify-side (repo: `automatos-shopify`, worktree off its default branch)

### S6 · Make the Remix admin build + wire real handlers (F013) — L
**Repo:** `automatos-shopify` (react-router 7 + `@shopify/shopify-app-remix` + Polaris). **Current state:** `app/routes.ts` is MISSING → `npx react-router routes` fails → app can't compile; webhook URIs 404; install-time provisioning is dead code.
**Deliverable:**
1. Add `app/routes.ts` with `flatRoutes()` so the app builds (`npx react-router routes` + `npm run build` succeed).
2. Wire the **webhook handlers**: `PRODUCTS_*`/`COLLECTIONS_*` → call the platform `/events` (S1's fixed contract); the three **mandatory GDPR webhooks** — `customers/redact`, `shop/redact`, `customers/data_request` → call the platform `erase_data_subject`/export entrypoints (Wave 11). Register the webhook URIs in the app TOML (they 404 today).
3. OAuth redirect + install-time provisioning calling the platform `/api/verticals/shopify/provision` (S5).
4. Add a **CI build gate** (`npx react-router routes` + `npm run build` + `tsc`) so this can't regress.
**Test/verify:** `npx react-router routes` lists routes; `npm run build` succeeds; a webhook handler unit test asserts it forwards to the correct platform endpoint (mock the platform HTTP call). **NO `shopify app dev`, NO browser.**

---

## Fire model & verification
- **Two worktrees / two agents.** Platform agent owns Part A (`automatos-ai`); Shopify agent owns Part B (`automatos-shopify`). Neither crosses repos. The seam is the platform HTTP contracts (S1 `/events`, W11 `erase_data_subject`, S5 provision) — give the Shopify agent those contract shapes.
- **Platform verify:** `py_compile` + pure pytest (mock Shopify/Composio/PG). **Shopify verify:** `npx react-router routes` + `npm run build` + `tsc` — never `shopify app dev`/browser (kills the user's Chrome).
- Commit-per-story; **do not push / open PRs** until Gerard's call.

## Conventions
No `os.getenv()` outside `config.py` (platform); 3-file tool registration for S3/S4; keep the vertical-isolation CI check green; no backward-compat shims. Confirm `automatos-shopify`'s default branch before branching (currently on `fix/blog-widget-columns-setting`).

## Success metrics
- Catalog webhooks update the graph in <60s (PRD-009 freshness); autosync uses its own session.
- Auto can run + check catalog sync and codegraph reindex through its own tools.
- A second vertical provisions without forking `api/shopify.py`.
- The Remix admin builds, its webhooks (incl. all 3 GDPR) resolve and call the platform, guarded by a CI build gate.
