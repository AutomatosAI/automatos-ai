# @automatos/theme-sdk — types

The type declarations of `@automatos/theme-sdk`, the one module an Automatos theme imports from the host
(PRD-261 A-10b). **Types only.** A theme never bundles the SDK: at runtime the host serves its own instance through
the import map (`frontend/lib/theme-host/shared-modules.json`, shim `/theme-sdk/v1/sdk.js`), built from
`frontend/lib/theme-sdk/index.ts`. A theme's bundler keeps `@automatos/theme-sdk` external.

## What is in it

| File | What it declares |
|---|---|
| `index.d.ts` | Re-exports every area. Not edited again after A-10b (PA-17). |
| `areas/<area>.d.ts` | One area of the SDK, as `frontend/lib/theme-sdk/<area>.ts` builds it (`pages/agents` is `areas/pages-agents.d.ts`). |
| `internal/stub.d.ts` | `NotInThisBuild`: the type of a planned name whose area has not landed (calling it throws "not in this SDK build"). |
| `internal/host.d.ts` | Host types the areas read but the SDK does not export by name (`SlotId`). |

Built in v1: `theme`, `context`, `navigation`, `notices`, `events`. Stubbed (every planned name declared as
`NotInThisBuild` until its lane lands): `chat`, `deliverables`, `cards`, `looks`, `assets` (W-A4) and
`pages/{agents,assignments,documents,playbooks,command-center}` (W-A4P). The planned names are in
`frontend/lib/theme-sdk/surface/<area>.json` and `docs/theme-host/sdk-surface-map.md`.

The declarations import only `react` (types) and each other, so they work outside this repository. They use only
React type names that `@types/react` 18 and 19 share: core typechecks them with its own `@types/react` 18.2; a theme
uses React 19's types (the peer), because the host's React is the one Next.js compiles in.

## How it is kept true

`frontend/lib/theme-sdk/__tests__/types-match-runtime.test.ts` (vitest) checks, for every area:

- every runtime export is declared as a value in the area's `.d.ts`, and no declared value is missing at runtime;
- every type the runtime area exports is declared;
- a stub area declares each name `NotInThisBuild`, a built area declares none that way;
- `index.d.ts` re-exports exactly the areas, and no declaration file imports anything but `react` and its own files;
- `package.json`'s `version` is `sdkVersion` (`shared-modules.json`), and `@types/react` ^19 is the peer.

The same file assigns the host's module to `typeof` this package's index, so CI's baselined `tsc` fails if any runtime
export does not fit its declaration.

**The lane that fills an area** replaces that area's `NotInThisBuild` declarations with the real types in the same
change, and sets its surface file's `status` to `built`. Names never change (A-10c). **A change to the surface after
W-A3 merges** bumps `sdkVersion` in `shared-modules.json` and `version` here together (PRD §4.3 item 7).

## Publishing path — recommended, Gerard's call (D2-b)

Recommended for v1 (PRD §4.3 item 9): **no npm org and no registry publish** (`"private": true` stops an accidental
`npm publish`).

1. Core CI runs `npm pack` in `frontend/packages/theme-sdk` and attaches the tarball
   (`automatos-theme-sdk-<sdkVersion>.tgz`) to the wave's run as an artefact.
2. The theme kit vendors the `.d.ts` files by version under its own `packages/theme-sdk/`, from that tarball, and
   records the core SHA it came from.
3. A theme declares `engines.sdk` against that version; the host refuses a theme whose range its `sdkVersion` does
   not meet.

Not decided until Gerard answers D2-b: whether to publish under an npm org instead, and the CI step that attaches the
tarball (it belongs to the workflow's owner, not to this package). Until then the kit copies these files at a pinned
core SHA.
