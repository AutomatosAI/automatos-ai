/**
 * PRD-261 A-10b — the types of `@automatos/theme-sdk` v1, the one module a theme imports from the host.
 *
 * Types only. At runtime a theme never bundles the SDK: the host serves its own instance through the import map
 * (frontend/lib/theme-host/shared-modules.json, the shim /theme-sdk/v1/sdk.js), built from
 * frontend/lib/theme-sdk/index.ts. This package's version is that build's `sdkVersion`.
 *
 * One file per area, as the runtime has (areas/<area>.d.ts; `pages/agents` is `areas/pages-agents.d.ts`), so the
 * lane that fills an area edits only its own file (PA-17). This index re-exports every area; it changes only when
 * an area is added (E3b added `storage`, with the SDK at 1.1.0), never for an area's own names.
 * frontend/lib/theme-sdk/__tests__/types-match-runtime.test.ts holds each area's declarations to its runtime names,
 * and tsc holds the host's module to these types.
 */
export * from './areas/theme'
export * from './areas/context'
export * from './areas/navigation'
export * from './areas/notices'
export * from './areas/events'
export * from './areas/chat'
export * from './areas/deliverables'
export * from './areas/cards'
export * from './areas/looks'
export * from './areas/assets'
export * from './areas/pages-agents'
export * from './areas/pages-assignments'
export * from './areas/pages-documents'
export * from './areas/pages-playbooks'
export * from './areas/pages-command-center'
export * from './areas/storage'
