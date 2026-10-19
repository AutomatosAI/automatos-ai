/**
 * PRD-261 A-01, A-10a — the SDK version and defineTheme. They live here, not in lib/theme-sdk,
 * because the loader (sources.ts) and the manifest check (package-manifest.ts) read them. A theme
 * reaches them through '@automatos/theme-sdk': lib/theme-sdk/theme.ts re-exports both, and the
 * SDK the host exposes is lib/theme-sdk/index.ts. The shim's names are not listed in
 * shared-modules.json: generate-shims.js (namesFromSurface) takes the union of
 * lib/theme-sdk/surface/*.json, and surface.test.ts holds the runtime to those files.
 */
import sharedModules from './shared-modules.json'
import type { ThemeDefinition } from './types'

export type { ThemeDefinition, ThemeRegistrationHost } from './types'

export const SDK_VERSION: string = sharedModules.sdkVersion

function isThemeDefinition(value: unknown): value is ThemeDefinition {
  return typeof value === 'object' && value !== null && typeof (value as { register?: unknown }).register === 'function'
}

/** A theme's definition, checked and frozen: it must carry a register(host) function. */
export function defineTheme(definition: ThemeDefinition): Readonly<ThemeDefinition> {
  if (!isThemeDefinition(definition)) {
    throw new TypeError('defineTheme needs a register(host) function.')
  }
  return Object.freeze({ ...definition })
}
