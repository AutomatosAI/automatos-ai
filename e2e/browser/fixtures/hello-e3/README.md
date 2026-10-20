# hello-e3 — the theme that keeps its state (PRD-261 E3c)

The fixture behind `specs/theme-storage.spec.ts`. Like A-08c's, its packages are not served through
route interception: the host publishes a theme's storage scope only for an installed theme with a
registry entry, and the entry comes from the install's boot, which the root layout fetches on the
server (`frontend/lib/theme-host/boot-server.ts` → `GET /api/themes/boot/{installId}`). So both
packages are uploaded to the local backend and opted in to.

| File | What it is |
|---|---|
| `entry.js` | The theme's `dist/entry.js`. Its frame keeps one value, `greeting`, through the SDK's storage area (`useThemeState`) and shows it on `data-testid="hello-e3-state"`: `data-ready`, `data-value` (the value as JSON), `data-persisted` and `data-last-write`. Its button (`hello-e3-save`) keeps `"kept by hello-e3"`. It never touches the browser's storage itself. |
| `build.py` | Builds A-06a's `hello` package with `entry.js` as `dist/entry.js`, `uses: ["storage"]` and `engines` `^1.1.0`, under one of two theme ids: `--variant hello-e3` (`automatos-e2e/hello-e3`) or `--variant hello-e3-other` (`automatos-e2e/hello-e3-other`). Each id is this fixture's own, so it never meets another spec's upload (the same id and version with other files is a 409). The spec runs it with `python3 build.py --variant <variant> --out <file>`. |

Only the browser's `GET /api/themes/active` is intercepted: it names the install the spec is on, with
a manifest in the shape the client loader reads (as A-08c's spec does), so the spec never changes the
local operator's stored choice, which other specs read.
