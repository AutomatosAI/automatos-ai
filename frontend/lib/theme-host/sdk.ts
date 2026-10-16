/**
 * PRD-261 A-01 — the spike's theme SDK, shared with a theme as '@automatos/theme-sdk'.
 *
 * Two exports only: the SDK version and defineTheme. Its runtime exports must equal the
 * `exports` list of its entry in shared-modules.json, because the shim re-exports exactly those
 * names (shim-generator.test.ts checks it). The SDK proper is PRD-261 A-10.
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
