# hello-a08b — the clip package behind the files proxy (PRD-261 A-08b)

The fixture behind `specs/files-proxy.spec.ts`. Unlike the other `hello-<story>` fixtures it is
not served through route interception: the spec proves the real path a browser takes, so the
package is uploaded to the local backend and read back through the app's own origin
(`frontend/app/api/themes/files/[...path]/route.ts` → `GET /api/themes/files/{digest}/{path}`).

| File | What it is |
|---|---|
| `build.py` | Builds A-06a's `hello-clip` variant (`orchestrator/tests/fixtures/themes/build.py`: the hello theme with PRD-260's `idle.mp4` and `idle.png` as `assets/mascot/`) under the id `automatos-e2e/hello-a08b`, so it never meets another spec's `automatos-e2e/hello` upload (the same id and version with other files is a 409). It also appends a second clip, `assets/mascot/idle-vp9.webm` (`automatos-design-kit`'s `public/auto/idle.webm`, copied to `orchestrator/tests/fixtures/themes/variants/idle.webm`), as an extra package member — not a declared mascot state. The spec runs it with `python3 build.py --out <file>`. |

The first clip is HEVC with alpha; the second is the same clip in VP9. Where a browser build
cannot decode a clip, the spec checks only what that clip proves (the HEVC clip: the 206/Range
path; the VP9 clip: nothing) and records the rest as unproven. Decodability is decided by
`navigator.mediaCapabilities.decodingInfo` where the browser has it, falling back to
`canPlayType(...) === 'probably'` — not `canPlayType(...) !== ''`, which 1.64.1's WebKit answers
for HEVC without actually being able to play it (#1363).
