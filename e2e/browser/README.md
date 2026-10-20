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
3. Installs this package (`npm ci`), the browsers (`npx playwright install --with-deps`),
   type-checks the specs (`npx tsc`) and runs them.
4. Runs the `@package` specs in a step of their own (below), after the others.
5. On a failure, uploads the Playwright report, traces, screenshots, videos and the backend and
   frontend server logs as `browser-e2e-report` (kept 7 days).

## Play from the Design Kit (PRD-261 A-17a)

The workflow's `package` job checks out the Design Kit (`AutomatosAI/automatos-play`, public, so
no secret) at the commit the workflow's `KIT_REF` pins, installs it from its committed lockfile
(npm at the kit's own version, `npm ci --ignore-scripts`), builds and packs `themes/play`, and
uploads `automatos.play-1.0.0.autotheme` as the `play-package` artifact. Play's build fetches Auto's
clips from core at the commit its `mascot.lock.json` pins. The browser job waits for it, downloads
the package and names it in `E2E_KIT_PACKAGE`; when the package job failed the browser job still
runs, and the `@package` specs fail on the missing file.

`support/kit-package.ts` installs it the way Settings › Appearance › Themes does (A-07d): `POST
/api/themes/import`, `POST /api/themes/installs/{id}/opt-in`, `PUT /api/themes/choice` with the
install's id, then the style cookie. Those specs store the member's choice on the lane's one backend,
which every other spec's tab reads, so they run in their own step, one test at a time
(`--workers=1`), with their own report and output folders (`playwright-report-package`,
`test-results-package`). Each test puts the choice back on Studio and removes the install.

To move the kit commit, change `KIT_REF` in the workflow, in the pull request that brings the Play
change it carries (the mirror rule, `docs/frontend-architecture/play-theme.md`).

The package pins `@playwright/test`, `typescript`, `@types/node`, `@types/react` and
`@types/react-dom` exactly and commits its own `package-lock.json` (every package from
registry.npmjs.org with its integrity hash), so `frontend/package-lock.json` never changes for it.
To change a pin, change `package.json` and the lockfile together.

## One build

The lane builds the frontend once, exactly as every image does (PRD-261 A-04a): the theme host has
no build flag. `npm run build` runs `prebuild` first, which generates the theme-sdk shims under
`public/theme-sdk/v1/` from the React that install vendors (`scripts/theme-sdk/generate-shims.js`);
they are never committed. The host loads a theme only under an installed theme's style
(`theme:install:<uuid>`), so a browser on Studio or Classic meets none of it.

## The hello theme

`fixtures/hello-theme/` is the test theme: `theme.json`, an entry with a frame, one stylesheet
and one preset, plus the variants the specs need (`entry.throws.js`, `entry.render-throws.js`,
`theme.engines-mismatch.json`). It reaches the browser only through route interception
(`support/hello-theme.ts`), as an installed theme: `GET /api/themes/active` answers with the
install and the fixture's `theme.json` as its manifest, and `/api/themes/files/<digest>/**` serves
the fixture's files. A spec puts the browser on the install's style first (`chooseStyle`). The
lane type-checks the theme against `@types/react` 19 and the test SDK's types
(`types/theme-sdk.d.ts`) with `npx tsc -p fixtures/hello-theme/tsconfig.json`.

`support/hello-theme.ts` is read-only after A-04a (PA-17): a later story adds its own fixture under
`fixtures/hello-<story>/` and its own spec, and serves it with `serveInstalledTheme`.

## Tags

Every test carries one tag through the `tag` option, for example
`test('…', { tag: '@smoke' }, async ({ page }) => { … })`. The workflow selects by tag with
`--grep`.

- `@smoke` — the lane itself works: the app loads, reaches the backend and draws the Studio shell.
- `@off` — a fresh browser on Studio loads none of the theme host, and the server sends a page a
  ThemeSlot page entry wraps (`/agents`, `/chat`, `/deliverables`) with one Studio shell, in no
  outlined boundary. CI traces caught `/agents` and `/chat` coming down twice before the change;
  `/deliverables` was not captured then, so its check has not been seen to fail without it.
- `@host` — the theme host loads an installed theme, falls back on the theme's own failures, keeps
  the theme when the page inside its frame fails, and stays off where it should.
- `@package` — Play installed from the Design Kit's package (above). Selected only by the
  workflow's own step for it, never by the main run's `--grep`.
- `@record` — writes measured facts (React version, import-map support and integrity) to `records/`.
  The only tag Firefox runs (the `firefox` project has `grep: /@record/`).
  `scripts/summarise-records.mjs` turns the folder into the job summary's table; the folder is
  uploaded as `theme-host-record`.

Specs live in `specs/`, shared helpers in `support/`.

## When it runs

On a pull request (any base branch) or a push to `main` that changes one of the paths in the
workflow's `paths:` list: the root layout, the pages and components PRD-261 Phase A touches (PA-6:
the Q3 pages, the theme host, Play, the looks, the Style axis files, the theme routes and services
on the backend), `next.config.js`, the middleware, the frontend's package files, the compose boot
files and this folder. It does not watch the rest of
the backend or the other pages, so a change there can break the lane without running it.

Until the workflow is on `main`, it runs only on a `pull_request` whose HEAD commit has no
`[skip ci]`: GitHub dispatches (`workflow_dispatch`) only workflows that exist on the default
branch.

The job is named `(non-required)`; making it a required check is the repo admin's call.
