/**
 * PRD-261 A-01 — the import map that resolves a theme's bare imports ('react', 'react-dom', the
 * SDK…) to the host's shims under /theme-sdk/v1/. Built from shared-modules.json alone; nothing
 * from a request goes into its `imports`.
 *
 * PRD-261 A-03 (PA-13): the map is on every page. Its `integrity` block pins every module of the
 * active installed theme; it is present only when the boot carries integrity (A-08c fills it
 * from the package's file manifest), and every entry is checked here before it is written: the
 * URL must be one of the theme files route's module paths and the value a sha256 digest.
 *
 * The body is scriptSafeJson's output as a whole, with nothing put around it. That helper
 * escapes '<' (so no '</script>' or '<!--' can form), U+2028 and U+2029, which is enough inside a
 * script element.
 */
import { scriptSafeJson } from './script-safe-json'
import sharedModules from './shared-modules.json'

export interface ThemeImportMapTable {
  readonly imports: Readonly<Record<string, string>>
  readonly integrity?: Readonly<Record<string, string>>
}

/** A module of an installed theme, served by the theme files route (A-08a). */
export const INTEGRITY_URL_PATTERN = /^\/api\/themes\/files\/[0-9a-f]{64}\/[A-Za-z0-9._/-]+\.js$/
/** Subresource integrity for one module: sha256, base64. */
export const INTEGRITY_VALUE_PATTERN = /^sha256-[A-Za-z0-9+/]{43}=$/

export const THEME_IMPORT_MAP: ThemeImportMapTable = Object.freeze({
  imports: Object.freeze(
    Object.fromEntries(sharedModules.modules.map((entry) => [entry.specifier, sharedModules.publicPath + entry.file])),
  ),
})

export const IMPORT_MAP_SCRIPT_BODY = scriptSafeJson(THEME_IMPORT_MAP)

/** The entries that pass both patterns (no '..' in a URL); null when none is left. */
export function sanitizeIntegrity(integrity: unknown): Readonly<Record<string, string>> | null {
  if (integrity === null || typeof integrity !== 'object' || Array.isArray(integrity)) return null
  const kept = Object.entries(integrity).filter(
    (entry): entry is [string, string] =>
      INTEGRITY_URL_PATTERN.test(entry[0]) &&
      !entry[0].includes('..') &&
      typeof entry[1] === 'string' &&
      INTEGRITY_VALUE_PATTERN.test(entry[1]),
  )
  return kept.length === 0 ? null : Object.fromEntries(kept)
}

/** The map's script body: the constant one, or the same imports with the checked integrity block. */
export function importMapScriptBody(integrity: unknown): string {
  const checked = sanitizeIntegrity(integrity)
  return checked === null ? IMPORT_MAP_SCRIPT_BODY : scriptSafeJson({ ...THEME_IMPORT_MAP, integrity: checked })
}
