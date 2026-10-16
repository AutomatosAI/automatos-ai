# e2e/browser — the browser lane

PRD-261 A-00. Playwright drives real Chromium and WebKit against the local edition, in CI.
The rest of the frontend's tests are vitest on jsdom; this lane covers what only a real
browser shows: module loading, import maps and the page as it is actually drawn.

It is separate from the Python scenario runner in `e2e/` (`e2e/run.py`), which it does not touch.

**Nothing here runs on the maintainer's machine.** The lane runs in CI only
(`.github/workflows/browser-e2e.yml`).

## What the lane does

1. Boots the local backend the way `smoke-fresh-clone.yml` does: `docker compose up -d --build
   postgres redis minio backend`, with throwaway local secrets and no external credentials, then
   waits for `http://localhost:8000/health/ready` (`scripts/wait-for-url.sh`).
2. Installs and builds the frontend on the runner (`npm install`, `npm run build`, local edition),
   stages the standalone output as `frontend/Dockerfile` does, and serves it with
   `node .next/standalone/server.js` on `http://localhost:3000`, the backend's CORS default.
3. Installs this package (`npm install --no-package-lock`), the browsers
   (`npx playwright install --with-deps`), type-checks the specs (`npx tsc`) and runs them.
4. On a failure, uploads the Playwright report, traces, screenshots, videos and the backend and
   frontend server logs as `browser-e2e-<build>-report` (kept 7 days).

The package pins `@playwright/test`, `typescript` and `@types/node` exactly and commits no
lockfile, so `frontend/package-lock.json` never changes for it.

## The two builds

| Build | Spike flag | Runs |
|---|---|---|
| `default` | `AUTOMATOS_THEME_HOST_SPIKE=0`, as every shipped image | `@smoke`, `@off` |
| `spike` | `AUTOMATOS_THEME_HOST_SPIKE=1`, this lane only | `@smoke`, `@spike`, `@record` |

The `spike` build joins the matrix with the theme-host spike (PRD-261 A-01). Until then the lane
runs the `default` build only.

## Tags

Every test carries one tag through the `tag` option, for example
`test('…', { tag: '@smoke' }, async ({ page }) => { … })`. The workflow selects by tag with
`--grep`.

- `@smoke` — the lane itself works: the app loads, reaches the backend and draws the Studio shell. Both builds.
- `@off` — with the spike off, none of the theme host is on the page. Default build only.
- `@spike` — the theme host loads, falls back and stays off where it should. Spike build only.
- `@record` — writes measured facts (React version, import-map support) to `records/`. Spike build only.

Specs live in `specs/`, shared helpers in `support/`.

## When it runs

On a pull request (any base branch) or a push to `main` that changes one of the paths in the
workflow's `paths:` list: the root layout, `/agents`, the providers and workspace provider, the
layout components, the theme host, the Style axis files, `next.config.js`, the middleware, the
frontend's package files, the compose boot files and this folder. It does not watch the rest of
the backend or the other pages, so a change there can break the lane without running it.

Until the workflow is on `main`, it runs only on a `pull_request` whose HEAD commit has no
`[skip ci]`: GitHub dispatches (`workflow_dispatch`) only workflows that exist on the default
branch.

The job is named `(non-required)`; making it a required check is the repo admin's call.
