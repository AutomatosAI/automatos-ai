/**
 * PRD-261 A-01 — the import map that resolves a theme's bare imports ('react', 'react-dom', the
 * SDK…) to the host's shims under /theme-sdk/v1/. Built from shared-modules.json alone; nothing
 * from a request goes into it.
 *
 * The body is scriptSafeJson's output as a whole, with nothing put around it. That helper
 * escapes '<' (so no '</script>' or '<!--' can form), U+2028 and U+2029, which is enough inside a
 * script element.
 */
import { scriptSafeJson } from '@/lib/looks/prepaint'
import sharedModules from './shared-modules.json'

export interface ThemeImportMapTable {
  readonly imports: Readonly<Record<string, string>>
}

export const THEME_IMPORT_MAP: ThemeImportMapTable = Object.freeze({
  imports: Object.freeze(
    Object.fromEntries(sharedModules.modules.map((entry) => [entry.specifier, sharedModules.publicPath + entry.file])),
  ),
})

export const IMPORT_MAP_SCRIPT_BODY = scriptSafeJson(THEME_IMPORT_MAP)
