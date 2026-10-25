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
4. Runs the `@installed` spec in a step of its own (below): the run's kit theme installed by the
   kit's own CLI.
5. Runs the `@package` and `@parity` specs in a step of their own (below), after the others.
   Since W-A8 (PRD-261 A-19f4) these are core's only proof that Play still runs: builtin Play left
   core, and Play lives only in the Design Kit.
6. On a failure, uploads the Playwright report, traces, screenshots, videos and the backend and
   frontend server logs as `browser-e2e-report` (kept 7 days).

## Play from the Design Kit (PRD-261 A-17a)

The workflow's `package` job checks out the Design Kit (`AutomatosAI/automatos-design-kit`, public, so
no secret) at the commit the workflow's `KIT_REF` pins, installs it from its committed lockfile
(npm at the kit's own version, `npm ci --ignore-scripts`), builds and packs `themes/play`, and
uploads `automatos.play-1.1.1.autotheme` as the `kit-theme-package` artifact (the job's summary shows
its size, sha256 and the kit commit). A dispatch can name another kit commit and theme (below).
The build fetches nothing: since 1.1.0 Auto's clips are committed in the kit with the rest of
Play. Before it builds, the job fails when a `kitPath` of core's test ledger
(`frontend/lib/__tests__/fixtures/a19-test-ledger.json`, the tests W-A8 moved out of core) is
missing from the kit at `KIT_REF`, or when the kit's `tests/fixtures/signing-vector.json` and
`keys/theme-signing.json` differ from core's signing vector and keys (PRD-261 A-19f4). The two jobs
run side by side (no
`needs:`), so the kit's build overlaps the browser job's docker and frontend builds. Just before it
downloads the package, the browser job waits for the package job (`scripts/wait-for-job.sh`, at most
`KIT_PACKAGE_WAIT_SECONDS`), then names the file in `E2E_KIT_PACKAGE`. When the package job failed,
or did not finish in time, the browser job still runs every other spec, and the `@installed` and
`@package` specs fail on the missing file.

`support/kit-package.ts` installs it the way Settings › Appearance › Themes does (A-07d): `POST
/api/themes/import`, `POST /api/themes/installs/{id}/opt-in`, `PUT /api/themes/choice` with the
install's id, then the style cookie. Those specs store the member's choice on the lane's one backend,
which every other spec's tab reads, so they run in their own step, one test at a time
(`--workers=1`), with their own report and output folders (`playwright-report-package`,
`test-results-package`). Each test puts the choice back on Studio and removes the install.

To move the kit commit, change `KIT_REF` in the workflow, in the pull request that brings the kit
change it carries, and only forward in the kit's history: the new commit contains the old one
(`git -C <kit> merge-base --is-ancestor <old> <new>`, PB-17; Play's own rules are in the kit's
`themes/play/README.md`).

## Any kit theme, installed by the kit's CLI (PRD-261 B-03h)

`specs/installed-theme.spec.ts` (`@installed`, Chromium and WebKit, in a step of its own, one test
at a time) installs the run's package the way a theme author does, with the kit's own command:
`automatos-theme install <file> --to http://localhost:3000 --choose` (`support/kit-cli.ts`). `--to`
is the web app's URL: the CLI reads its `/automatos-host.json` (B-03j) to find the API, then
imports the package, opts in to it and chooses it, with no token (the local edition needs none).
The spec fails unless the CLI exits 0 and says it found the API through that host file. For the
CLI, the browser job checks the kit out at the package job's ref and installs it the same way (npm
at the kit's version, `npm ci --ignore-scripts`; nothing is built), then names the command in
`E2E_KIT_CLI` and the downloaded package's folder in `E2E_KIT_PACKAGE_DIR`. It does both after the
main run's specs: when either fails, the job fails on that step, the `@installed` spec is skipped
and the `@package` and `@parity` specs still run.

The spec knows nothing of the theme beyond what the server's active answer carries of its
`theme.json`. On `/agents` and on every page (a `[param]` segment as an id no workspace has) and
route (`/x/<route>`) the theme declares, it holds the theme to the host's contract: the host is
`active` with the theme's `htmlClass` on `<html>`, there is no fallback notice, and the page drew
with no error boundary (the page's own inside the frame, or Next's not-found one, where a route
the theme declares but never registered ends up) and no uncaught error. Pass or fail, it then puts
the choice back on Studio and removes every install the CLI added (each install not there when the
test began), so a failed check never leaves the theme installed for the steps after it.

A manual run (`workflow_dispatch`, once the workflow is on `main`) takes two inputs: `kit_ref`, a
kit commit, branch or tag (empty: `KIT_REF`), and `kit_theme`, the theme's folder in the kit
(default `themes/play`), for example `gh workflow run browser-e2e.yml -f kit_ref=<kit sha> -f
kit_theme=themes/<name>`. The package job packs that theme from that commit, and the `@installed`
spec runs on it. A run for a theme other than Play packs no Play, so the `@package` and `@parity`
step and the preview shots' upload are skipped; every other spec runs as on a pull request. The
inputs reach a shell only through a step's `env:`, never as an expression inside `run:`
(`orchestrator/tests/test_prd261_browser_lane_inputs.py` reads the workflow and holds it to that,
and to the kit installs without scripts, no secret and read-only permissions).

## Play's preview shots (PRD-261 C-06q1)

`specs/play-previews.spec.ts` (`@package`, Chromium only, in the package step) shoots the installed
Play for its listing previews (PB-24, PB-26). For each preset the page's boot carries (Ember,
Obsidian, Aurora, Workshop, Daylight), picked on Make it yours as a member picks it, it saves the frame
on `/chat` as `<preset>.png`, and under the default preset also `/deliverables` and `/looks` as
`ember-deliverables.png` and `ember-looks.png`, in `previews/`: 1440×900 at device scale 1, one image
pixel per CSS pixel (`scale: 'css'`). A shot is taken under reduced motion (Auto is his still, never a
clip frame), once his still and the fonts have loaded, Auto is at rest (the first open's hello is over)
and the look's variables are on `<html>`, with animations off, and only when two shots in a row are
equal. The conversations list is answered with none, so the shots show only the lane's own member and
workspace; the shelf is answered with one page of five made-up Deliverables (a report, a document, a
spreadsheet, slides and an image, named after the chat's example chips, none with a picture to fetch),
so `ember-deliverables.png` shows cards; the spec waits for every card. The lane has no LLM key, so
the host's local-edition key nudge
(`components/local/first-run-nudge.tsx`) is dismissed before the first load, as a member dismisses it
(its `localStorage` key), and a chat shot that shows it fails. Play's empty chat scrolls on a short
window (P260-FX-14; 800 px high, three presets pushed the message box below the fold), so a chat shot
also fails unless the whole message box is inside the viewport. The spec fails when a file is missing,
is not a PNG of that size, is over 512 KB (`package.py` `IMAGE_MAX_BYTES`), or has a name the
manifest's `previews` pattern refuses or no preset claims by the listing's longest-match rule
(`manifest.py` `PREVIEW_PATTERN`, `listings.py` `_owner_preset`). The step after it uploads
`previews/` as the `play-previews` artifact, also when the specs failed; a run that saved none fails
there. Nothing here commits a shot: the owner picks from a run's artifact and the kit commits the
picks into `themes/play/assets/previews/` (C-06q2), so a package's bytes, and its signature, stay the
same from build to build.

## Play parity (PRD-261 A-17b, A-19f4)

`specs/play-parity/*.spec.ts` hold the installed Play package to Play's assertions. Until W-A8 they
ran each test twice, under builtin Play (compiled into core) and under the package, and held the two
to the same assertions; W-A8 deleted builtin Play, so `PLAY_VARIANTS` (`support/play-parity.ts`) is
the package alone and each test runs once. Each spec still writes its tests inside `describeParity`,
which installs the kit's package (uploaded, opted in to and chosen through `support/kit-package.ts`).
They cover Auto's 12 states on the style sheet (each still and clip served from the package's own
files), approvals in the chat (a tool card and a Mission card, approve, deny and reject), P260-FX-19
(Make it yours readable on Paper, in forced colours and with a long name), P260-FX-22 (focus after
Try again, the asleep screen and This look's toolbar; Studio's `/looks` note names no Play switch),
P260-FX-30 (Stop keeps the reply), Play below 1024 px (the core chrome in the look's colours, the
look's tone forced, a stored look painted from the first frame) and Play's pages and route through
ThemeSlot (`play-routes.spec.ts`: its chat in its frame, the core chat below its width, one frame
mount on a cold load, Conversations at `/x/conversations`). Play's chat runs on recorded turns
(`fixtures/play-parity/chat-turns.json`, through `support/chat-stream.ts`); no model is called.

They store the member's choice and Play's looks on the lane's one backend, so they carry `@parity`
and run in the `@package` step, one test at a time. Each test starts and ends on Studio, with no
install of Play's package and no looks kept for Play on the server: the server holds Play's looks
only while the package is installed and keeps them across an uninstall, so `describeParity` empties
them right after each install and again before it removes the package. Two more Play specs run
there as `@package` on the same harness: `looks-server.spec.ts` (a look saved in one browser shows in
another) and the CSS layer model's check in `protected-surfaces.spec.ts` (the body takes Play's page
background).

The checks no browser lane can make are in `parity-checklist.md`, the manual pass (Q22).

## The core styles (PRD-261 A-19f4)

`first-paint`, `default-build` and `smoke` run under the Classic and under the Studio style cookie
(`CORE_STYLES` in `support/hello-theme.ts`, each with the desktop chrome only it draws), on Chromium
and WebKit. Under either, a page asks for none of Play: no stylesheet that declares Play's tokens,
none of Play's face variables, nothing from builtin Play's old mascot folder or from a theme's files.
The cookies builtin Play ran under (the legacy `play`, and its builtin theme style) load Studio, with
no `play` class on `<html>` from the first frame, and builtin Play's old `/conversations` page
answers the 404. The specs build those retired names from their parts, so a search of `e2e/` for
them finds nothing.

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
`fixtures/hello-<story>/` and its own spec, and serves it with `serveInstalledTheme`. A-19f4 added
beside it, leaving the serving as it was, the two core styles (`CORE_STYLES`, `chooseCoreStyle`) and
the mount counter the route specs share (`countMounts`, `mountsOf`).

## Tags

Every test carries one tag through the `tag` option, for example
`test('…', { tag: '@smoke' }, async ({ page }) => { … })`. The workflow selects by tag with
`--grep`.

- `@smoke` — the lane itself works: the app loads, reaches the backend and draws the Studio shell,
  and under each core style's cookie that style's chrome (A-19f4).
- `@off` — a fresh browser on Studio, and a browser on either core style's cookie, loads none of the
  theme host, and the server sends a page a ThemeSlot page entry wraps (`/agents`, `/chat`,
  `/deliverables`) with one shell of its style, in no outlined boundary. CI traces caught `/agents` and `/chat` coming down twice before the change;
  `/deliverables` was not captured then, so its check has not been seen to fail without it.
- `@host` — the theme host loads an installed theme, falls back on the theme's own failures, keeps
  the theme when the page inside its frame fails, and stays off where it should.
- `@installed` — the run's kit theme installed by the kit's CLI (above). Selected only by the
  workflow's own step for it, never by the main run's `--grep`.
- `@package` — Play installed from the Design Kit's package (above). Selected only by the
  workflow's own step for it, never by the main run's `--grep`.
- `@parity` — the Play parity specs (above), run against the installed package.
  Selected by the same step as `@package`.
- `@record` — writes measured facts (React version, import-map support and integrity) to `records/`.
  The only tag Firefox runs (the `firefox` project has `grep: /@record/`).
  `scripts/summarise-records.mjs` turns the folder into the job summary's table; the folder is
  uploaded as `theme-host-record`.

Specs live in `specs/`, shared helpers in `support/`.

## When it runs

On a pull request (any base branch) or a push to `main` that changes one of the paths in the
workflow's `paths:` list: the root layout, the pages and components PRD-261 Phase A touches (PA-6:
the Q3 pages, the theme host, the looks pages, the Style axis files, the theme routes and services
on the backend; since A-17b also the chat code the Play specs read through the SDK, `lib/chat/` and
`components/widgets/`; since B-03h the web app's host file the kit's CLI reads,
`app/automatos-host.json/`), `next.config.js`, the middleware, the frontend's package files, the
compose boot files and this folder. Builtin Play's folders left the list when W-A8 deleted them
(A-19f4): Play is the kit's now, and the lane installs it from `KIT_REF`. It does not watch the
rest of the backend or the other pages, so a change there can break the lane without running it.

Until the workflow is on `main`, it runs only on a `pull_request` whose HEAD commit has no
`[skip ci]`: GitHub dispatches (`workflow_dispatch`) only workflows that exist on the default
branch.

The job is named `(non-required)`; making it a required check is the repo admin's call.
