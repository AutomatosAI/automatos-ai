/**
 * A link that came from outside the app (a mail tool's attachment URL, a field in a tool
 * result), as a URL the app may open: http(s) only, else null.
 *
 * window.open on a `javascript:` or `data:` URL runs it in a window the app opened, with the
 * app's origin. Such a link is never opened, and the page that does open is given no
 * `window.opener` (EXTERNAL_WINDOW_FEATURES), so it cannot navigate the app's tab.
 */
const OPENABLE_PROTOCOLS = new Set(['http:', 'https:'])

export const EXTERNAL_WINDOW_FEATURES = 'noopener,noreferrer'

export function externalHttpUrl(url: unknown): string | null {
  if (typeof url !== 'string' || !url.trim()) return null
  try {
    const parsed = new URL(url.trim())
    return OPENABLE_PROTOCOLS.has(parsed.protocol) ? parsed.toString() : null
  } catch {
    return null
  }
}

/** Opens an outside link in a new tab, or does nothing when it is not http(s). */
export function openExternalUrl(url: unknown): boolean {
  const safe = externalHttpUrl(url)
  if (!safe) return false
  window.open(safe, '_blank', EXTERNAL_WINDOW_FEATURES)
  return true
}
