# hello-a04c — the theme that keeps failing (PRD-261 A-04c)

The fixture behind `specs/three-failures.spec.ts`. Like A-08c's, the package is uploaded to the
local backend and opted in to, so its files come through the app's own files proxy and the
revert's `PUT /api/themes/choice` reaches the real backend.

| File | What it is |
|---|---|
| `build.py` | Builds A-06a's `hello-throws` variant (`orchestrator/tests/fixtures/themes/build.py`: the hello theme whose `dist/entry.js` throws while it is evaluated) under the id `automatos-e2e/hello-a04c`, so it never meets another spec's upload (the same id and version with other files is a 409). The spec runs it with `python3 build.py --out <file>`. |

Only the browser's `GET /api/themes/active` is intercepted: it names this install, with a manifest
in the shape the client loader reads today (as A-08c's spec does), so the spec never sets the local
operator's stored choice to a broken theme that other specs would read. The revert's
`PUT /api/themes/choice` is not intercepted, and the backend's answer to it is the proof that the
server's choice is Studio.
