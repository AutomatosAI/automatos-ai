# hello-a08b — the clip package behind the files proxy (PRD-261 A-08b)

The fixture behind `specs/files-proxy.spec.ts`. Unlike the other `hello-<story>` fixtures it is
not served through route interception: the spec proves the real path a browser takes, so the
package is uploaded to the local backend and read back through the app's own origin
(`frontend/app/api/themes/files/[...path]/route.ts` → `GET /api/themes/files/{digest}/{path}`).

| File | What it is |
|---|---|
| `build.py` | Builds A-06a's `hello-clip` variant (`orchestrator/tests/fixtures/themes/build.py`: the hello theme with PRD-260's `idle.mp4` and `idle.png` as `assets/mascot/`) under the id `automatos-e2e/hello-a08b`, so it never meets another spec's `automatos-e2e/hello` upload (the same id and version with other files is a 409). The spec runs it with `python3 build.py --out <file>`. |

The clip is HEVC with alpha. Where a browser build cannot decode it, the spec still checks that
the `<video>` loaded it over 206 through the proxy, and records the playback as unproven.
