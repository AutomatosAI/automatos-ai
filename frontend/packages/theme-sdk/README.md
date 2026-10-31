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

## Install it (D2-b: the widget SDK's CDN)

```sh
npm install https://widgets.automatos.app/theme-sdk/v1/theme-sdk.tgz
```

That is the newest `1.x`. For a lockfile that never changes under you, pin the exact version instead:

```sh
npm install https://widgets.automatos.app/theme-sdk/v1.3.0/theme-sdk.tgz
```

`package-lock.json` records the tarball's integrity hash. The `v1` alias moves when a new `1.x` ships, so a lock
made against it fails `npm ci` with `EINTEGRITY` until you run `npm install` again; the `v<version>` URL never
changes. Either way the package name is `@automatos/theme-sdk`, and a theme imports from it as usual (types only:
keep it external in the bundler, the host provides the runtime).

## How it is published

No npm org and no registry publish (`"private": true` stops an accidental `npm publish`; `npm pack` ignores it).
The widget SDK's distribution is reused instead (Gerard, D2-b): its S3 bucket and CloudFront distribution
(`widgets.automatos.app`), its immutable-version + major-alias layout and its five `AWS_SDK_DEPLOY_*` secrets.

- `.github/workflows/theme-sdk-cdn.yml` runs on a push to `main` that touches `frontend/packages/theme-sdk/**` or
  `frontend/lib/theme-sdk/surface/**`, and on a manual run. It calls `scripts/theme-sdk/deploy-cdn.sh`.
- The script runs `npm pack` here and uploads the tarball twice, then writes a marker:

  | Key | Cache | Rule |
  |---|---|---|
  | `theme-sdk/v<version>/theme-sdk.tgz` | `public, max-age=31536000, immutable` | never overwritten |
  | `theme-sdk/v<major>/theme-sdk.tgz` | `public, max-age=3600` | replaced on every deploy; the only path CloudFront invalidates |
  | `theme-sdk/v<version>/published.json` | `no-cache` | written last; its presence stops the next deploy of this version |

- The marker goes up only after the CloudFront invalidation has completed and both URLs serve the tarball's exact
  bytes (sha256). A run that fails part way leaves no marker, so the next run resumes: it downloads the immutable
  copy and redoes the alias, the invalidation and the checks, never re-packing over a published version.

- `<version>` is this file's `version`, which is `sdkVersion`. **To ship a change, bump both together**; a push
  that does not bump them is a no-op ("already published", the job exits 0 with a notice).
- Until the five secrets (`AWS_SDK_DEPLOY_ACCESS_KEY_ID`, `_SECRET_ACCESS_KEY`, `_REGION`, `_BUCKET`,
  `_DISTRIBUTION_ID`) are set on this repository or the org, the job prints a notice and exits 0.
- A manual deploy is the same script: `scripts/theme-sdk/deploy-cdn.sh --dry-run` first, then without the flag,
  with AWS credentials in the shell.

A theme declares `engines.sdk` against the version it was built with; the host refuses a theme whose range its
`sdkVersion` does not meet.
