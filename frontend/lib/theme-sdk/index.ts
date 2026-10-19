/**
 * PRD-261 A-10a — `@automatos/theme-sdk` v1: the one module a theme imports from the host.
 *
 * A theme never imports `@/…`, `next/*`, zustand stores, react-query, `api-client` or sonner
 * (PRD §4.3 item 7). It imports this, through the import map, as the host's own instance: the
 * loader chunk puts this module's namespace on the host global (lib/theme-host/host-modules.ts)
 * and the generated shim /theme-sdk/v1/sdk.js re-exports it by name. Core code reaches it as
 * '@automatos/theme-sdk' too (tsconfig.json paths, vitest.config.ts alias).
 *
 * One file per area. Each area's runtime names are listed in surface/<area>.json, and
 * __tests__/surface.test.ts holds this module to exactly their union, so an export nobody
 * planned fails CI. Areas built by A-10a: theme, context, navigation, notices, events. The rest
 * are stubs that throw "not in this SDK build" until their lane lands (chat, deliverables, cards,
 * looks and assets in W-A4; the pages/* data hooks in W-A4P).
 */
export * from './theme'
export * from './context'
export * from './navigation'
export * from './notices'
export * from './events'
export * from './chat'
export * from './deliverables'
export * from './cards'
export * from './looks'
export * from './assets'
export * from './pages/agents'
export * from './pages/assignments'
export * from './pages/documents'
export * from './pages/playbooks'
export * from './pages/command-center'
