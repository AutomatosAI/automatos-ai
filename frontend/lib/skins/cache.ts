// PRD-259 §4 — the active skin's stylesheet, cached in this browser so the pre-paint script
// (prepaint.ts) can apply it before first paint. Per-browser convenience only: the provider
// regenerates it from the installed skin after hydration, so a missing, blocked or corrupt
// entry only costs a flash of the base style. Storage can throw (private mode, blocked site
// data), so every access is guarded and logged.

export const SKIN_CSS_CACHE_KEY = 'automatos-skin-css'
const LOG_PREFIX = '[Skins]'

export interface SkinCssCache {
  readonly id: string
  readonly version: string
  readonly css: string
}

function isCache(value: unknown): value is SkinCssCache {
  if (typeof value !== 'object' || value === null) return false
  const entry = value as Record<string, unknown>
  return typeof entry.id === 'string' && typeof entry.version === 'string' && typeof entry.css === 'string'
}

/** The cached entry, or null when there is none, it isn't one, or storage refuses. */
export function readSkinCache(): SkinCssCache | null {
  try {
    const raw = window.localStorage.getItem(SKIN_CSS_CACHE_KEY)
    if (raw === null) return null
    const parsed: unknown = JSON.parse(raw)
    return isCache(parsed) ? { id: parsed.id, version: parsed.version, css: parsed.css } : null
  } catch (error) {
    console.warn(`${LOG_PREFIX} Could not read the cached skin stylesheet.`, error)
    return null
  }
}

/** Stores the entry; false when storage refuses (the page still works, it just can't pre-paint). */
export function writeSkinCache(entry: SkinCssCache): boolean {
  try {
    window.localStorage.setItem(SKIN_CSS_CACHE_KEY, JSON.stringify({ id: entry.id, version: entry.version, css: entry.css }))
    return true
  } catch (error) {
    console.warn(`${LOG_PREFIX} Could not cache the skin stylesheet.`, error)
    return false
  }
}

export function clearSkinCache(): void {
  try {
    window.localStorage.removeItem(SKIN_CSS_CACHE_KEY)
  } catch (error) {
    console.warn(`${LOG_PREFIX} Could not clear the cached skin stylesheet.`, error)
  }
}
