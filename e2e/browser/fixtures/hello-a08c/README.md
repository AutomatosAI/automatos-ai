# hello-a08c — the installed package behind first paint (PRD-261 A-08c)

The fixture behind `specs/installed-first-paint.spec.ts`. Like A-08b's, it is not served through
route interception: the root layout fetches the install's boot on the server
(`frontend/lib/theme-host/boot-server.ts` → `GET /api/themes/boot/{installId}`), which no browser
route can intercept, so the package is uploaded to the local backend and opted in to.

| File | What it is |
|---|---|
| `build.py` | Builds A-06a's `hello` package under the id `automatos-e2e/hello-a08c`, so it never meets another spec's upload (the same id and version with other files is a 409). The spec runs it with `python3 build.py --out <file>`. |

Only the browser's `GET /api/themes/active` is intercepted, and only for the tampered-chunk
tests: it names this install (the spec never changes the local operator's stored choice, which
other specs read) with a manifest in the shape the client loader reads today.
