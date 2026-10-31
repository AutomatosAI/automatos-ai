# Changing the theme host

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this page:

- [orchestrator/services/themes/contract.py](../../orchestrator/services/themes/contract.py)
- [orchestrator/services/themes/drift.py](../../orchestrator/services/themes/drift.py)
- [orchestrator/services/themes/data/core-pages.json](../../orchestrator/services/themes/data/core-pages.json)
- [orchestrator/services/themes/data/sdk-surface.json](../../orchestrator/services/themes/data/sdk-surface.json)
- [orchestrator/tests/test_prd261_core_pages_drift.py](../../orchestrator/tests/test_prd261_core_pages_drift.py)
- [orchestrator/tests/test_prd261_sdk_surface_drift.py](../../orchestrator/tests/test_prd261_sdk_surface_drift.py)
- [orchestrator/tests/test_prd261_theme_package.py](../../orchestrator/tests/test_prd261_theme_package.py)
- [orchestrator/tests/test_prd261_mobile_contrast.py](../../orchestrator/tests/test_prd261_mobile_contrast.py)
- [orchestrator/tests/test_prd261_b2_wire.py](../../orchestrator/tests/test_prd261_b2_wire.py)
- [orchestrator/tests/test_prd261_b2_step_up_routes.py](../../orchestrator/tests/test_prd261_b2_step_up_routes.py)
- [orchestrator/tests/test_prd261_no_builtin_play.py](../../orchestrator/tests/test_prd261_no_builtin_play.py)
- [orchestrator/tests/test_route_manifest.py](../../orchestrator/tests/test_route_manifest.py)
- [orchestrator/tests/fixtures/themes/cases.json](../../orchestrator/tests/fixtures/themes/cases.json)
- [orchestrator/reports/route-manifest.json](../../orchestrator/reports/route-manifest.json)
- [frontend/lib/theme-sdk/index.ts](../../frontend/lib/theme-sdk/index.ts)
- [frontend/lib/theme-sdk/surface](../../frontend/lib/theme-sdk/surface)
- [frontend/lib/theme-sdk/__tests__/surface.test.ts](../../frontend/lib/theme-sdk/__tests__/surface.test.ts)
- [frontend/lib/theme-sdk/__tests__/surface-map.test.ts](../../frontend/lib/theme-sdk/__tests__/surface-map.test.ts)
- [frontend/lib/theme-sdk/__tests__/types-match-runtime.test.ts](../../frontend/lib/theme-sdk/__tests__/types-match-runtime.test.ts)
- [frontend/lib/theme-host/shared-modules.json](../../frontend/lib/theme-host/shared-modules.json)
- [frontend/lib/theme-host/__tests__/validator-parity.test.ts](../../frontend/lib/theme-host/__tests__/validator-parity.test.ts)
- [frontend/lib/look-kit/__tests__/contrast-parity.test.ts](../../frontend/lib/look-kit/__tests__/contrast-parity.test.ts)
- [frontend/lib/__tests__/prd261-no-theme-in-core.test.ts](../../frontend/lib/__tests__/prd261-no-theme-in-core.test.ts)
- [frontend/packages/theme-sdk/package.json](../../frontend/packages/theme-sdk/package.json)
- [frontend/scripts/theme-sdk/generate-shims.js](../../frontend/scripts/theme-sdk/generate-shims.js)
- [frontend/scripts/check-route-contract.js](../../frontend/scripts/check-route-contract.js)
- [frontend/scripts/__tests__/frontend-ci-lane-paths.test.ts](../../frontend/scripts/__tests__/frontend-ci-lane-paths.test.ts)
- [.github/workflows/test.yml](../../.github/workflows/test.yml)
- [.github/workflows/theme-sdk-cdn.yml](../../.github/workflows/theme-sdk-cdn.yml)
- [AGENTS.md](../../AGENTS.md)

</details>

This page is for people who change the theme host itself. Several promises the host makes to themes are written down
twice, once for the server and once for the browser, or are committed as data because the running image has no
`frontend/` folder. Guards hold each copy to the other. When you change one side, this page says what else to update
and which guard fails if you don't.

The host's own tests are `orchestrator/tests/test_prd261_*.py` and the `__tests__` folders under
`frontend/lib/theme-host/`, `frontend/lib/theme-sdk/`, `frontend/components/theme-host/` and
`frontend/components/settings/themes/`. CI runs them.

---

## A core page a theme can replace or clone

The pages a theme may replace, and the files behind each, are `PAGE_SCOPE` and `PAGE_SOURCES` in
`orchestrator/services/themes/contract.py`. The sha256 of every `PAGE_SOURCES` file is committed in
`orchestrator/services/themes/data/core-pages.json`. The "Needs update" check (`services/themes/drift.py`) and the
review's `clonedFrom` check read it.

| You change | Also do | Guard |
|---|---|---|
| A file listed in `PAGE_SOURCES` | Regenerate the hashes from the repository root: `python3 orchestrator/tests/test_prd261_core_pages_drift.py --write` (a bare `python3`, nothing installed) | `test_prd261_core_pages_drift.py` fails until the JSON matches the files |
| The lists themselves (`PAGE_SCOPE`, `PAGE_SOURCES`) | The browser's copy of the lists, then regenerate the hashes as above | `validator-parity.test.ts` reads `contract.py` as text and compares them |

Once the new hashes ship, every installed theme that cloned a changed file shows **Needs update**, and a new upload of
it gets a `clonedFrom` warning. That is what the check is for: the theme's copy is behind the core page.

## A name in the theme SDK

`@automatos/theme-sdk` (`frontend/lib/theme-sdk/index.ts`) exports the union of its areas. Each area's names are
planned in `frontend/lib/theme-sdk/surface/<area>.json`.

| You change | Also do | Guard |
|---|---|---|
| An area's exports | The area's `surface/<area>.json` names | `surface.test.ts`: each area exports exactly its planned names, and no name sits in two areas |
| A surface file | The row for each name in the [SDK surface map](sdk-surface-map.md) | `surface-map.test.ts`: every surface name has a row, and every row's name is planned |
| A surface file | The server's copy: `python3 orchestrator/tests/test_prd261_sdk_surface_drift.py --write` writes `services/themes/data/sdk-surface.json`, which the review's `uses` check reads | `test_prd261_sdk_surface_drift.py` |
| An area's names or types | The types package `frontend/packages/theme-sdk` (`index.d.ts` and `areas/<area>.d.ts`) | `types-match-runtime.test.ts`: every runtime name is declared, and nothing more |

The shims a theme imports from `/theme-sdk/v1/` are generated from `shared-modules.json` and the surface files by
`frontend/scripts/theme-sdk/generate-shims.js`, before every `next dev` and `next build`. They are never committed.

### The SDK and host versions

| Value | Where | Held equal by |
|---|---|---|
| The SDK version | `sdkVersion` in `frontend/lib/theme-host/shared-modules.json`, `SDK_VERSION` in `contract.py`, `version` in `frontend/packages/theme-sdk/package.json` | `validator-parity.test.ts` and `types-match-runtime.test.ts` |
| The host contract version | `hostVersion` in `shared-modules.json`, `THEME_HOST_VERSION` in `contract.py` | `validator-parity.test.ts` |

The **Theme SDK types to CDN (main)** workflow (`.github/workflows/theme-sdk-cdn.yml`) runs on `main` when
`frontend/packages/theme-sdk/**` or `frontend/lib/theme-sdk/surface/**` changes. It publishes the types package under
`theme-sdk/v<version>/`, once per version: a run that finds its version already published stops with "This Theme SDK
version is already on the CDN: nothing to ship. Bump frontend/packages/theme-sdk/package.json and sdkVersion together
to publish." Until its deploy secrets exist, the job prints a notice and exits 0.

## The package contract and its validators

The server checks a package with `services/themes/manifest.py`, `look_schema.py`, `css_value.py` and
`mobile_manifest.py`, against the lists in `contract.py`. The browser has TypeScript twins under
`frontend/lib/theme-host/`: `package-manifest.ts`, `look-schema.ts`, `css-value.ts`, `mobile-manifest.ts`,
`constants.ts` and `boot-sanitize.ts`.

| You change | Also do | Guard |
|---|---|---|
| A rule in a Python validator | The TypeScript twin, and a case in the shared corpus `orchestrator/tests/fixtures/themes/cases.json` | `test_prd261_theme_package.py` runs the corpus in Python and `validator-parity.test.ts` in TypeScript: both must give the same verdict and code |
| A list or limit in `contract.py` (reserved classes, slots, pages, error codes, the mobile lists, the contrast pairs, `PREPAINT_VALUE_MAX`) | The TypeScript constant | `validator-parity.test.ts` reads the Python sources as text |
| The browser's first-paint gate (`css_value.py` `BOOT_VALUE_CHARS`, `BOOT_VALUE_FUNCTIONS`, `boot_value_problem`) | Its one TypeScript copy, `css-value.ts` (`boot-sanitize.ts` imports `isBootValue` from it), and a case in the corpus's `prepaintDrops` section | `validator-parity.test.ts` reads the function list from `css_value.py` and runs `prepaintDrops` with `prepaintDropErrors`; `test_prd261_theme_package.py` runs it with `prepaint_drop_errors` |
| The contrast maths | Both sides, and the corpus's `contrast` section | `test_prd261_mobile_contrast.py` and `contrast-parity.test.ts` |

The [Design Kit](https://github.com/AutomatosAI/automatos-design-kit) vendors these TypeScript validators and the
corpus (`packages/theme-cli/src/core` and `packages/theme-cli/test/fixtures/core/cases.json`), so that its `pack`
refuses what core refuses. A contract change has to be copied there too.

## A theme route

| You change | Also do | Guard |
|---|---|---|
| A backend route the frontend calls | Regenerate `orchestrator/reports/route-manifest.json`: `cd orchestrator && python -m scripts.dump_routes` (see [AGENTS.md](../../AGENTS.md)) | `frontend/scripts/check-route-contract.js` in CI; `test_prd261_b2_wire.py` holds the routes of the five theme routers (`api/themes_import.py`, `themes_install.py`, `themes_files.py`, `themes_looks.py`, `themes_settings.py`) to the manifest |
| Which theme routes ask for step-up | The route's step-up dependency | `test_prd261_b2_wire.py` (which routes carry it) and `test_prd261_b2_step_up_routes.py` (the refusals) |
| A theme route's behaviour | Its entry in the [API reference](api-reference.md) | None: keep it by hand |

## Which CI lane runs

`.github/workflows/test.yml` starts the `frontend-ci` lane on a pull request only when a changed file matches
`FRONTEND_PATHS`. Several frontend tests read server files as text (`scan-report.test.ts` reads `review_stage.py` and
`scan.py`, for example), so those server files are in `FRONTEND_PATHS` too.

| You change | Also do | Guard |
|---|---|---|
| A frontend test, so that it reads another `orchestrator/services/themes/*.py` module | Add the module to `FRONTEND_PATHS` | `frontend-ci-lane-paths.test.ts`: every such module a frontend test names must match the lane |

## No theme in core

Play and HUD are installed packages like any other. `orchestrator/tests/test_prd261_no_builtin_play.py` and
`frontend/lib/__tests__/prd261-no-theme-in-core.test.ts` fail when core names what Play left behind when it moved out
of core. The second one also reads `docs/` (all but `docs/PRDS`).

Related:

- [Theme host](_index.md)
- [SDK surface map](sdk-surface-map.md)
- [API reference](api-reference.md)
