/**
 * PRD-270 K15 — where a Cloud listing's files load from. A version a Sync with Automatos Cloud listed
 * is not stored on this install until it is added, so its screenshots load from Automatos Cloud's
 * public catalogue files route (`services/marketplace_cloud/urls.py` `cloud_file_url`:
 * `<origin>/api/marketplace/catalog/files/<digest>/<path>`), straight from the browser, proxied by
 * nothing (§5.8).
 *
 * Only two origins count: Automatos Cloud's API, and the API this build talks to (NEXT_PUBLIC_API_URL,
 * baked in at build time: the hosted edition's own catalogue). The app's Content-Security-Policy lets
 * images load from the files route on exactly those two (csp-sources.js `cloudImageSources`). A URL
 * is kept only when it parses to one of them and its path, after the browser's own normalising, is
 * still a file of one version under the catalogue files route.
 */
import { THEME_DIGEST_PATTERN } from '@/lib/theme-host/constants'

/** Automatos Cloud's API origin; csp-sources.js `AUTOMATOS_CLOUD_ORIGIN` names the same one. */
export const AUTOMATOS_CLOUD_ORIGIN = 'https://api.automatos.app'

/** The catalogue's files route (urls.py `CATALOG_FILES_ROUTE`), then `<digest>/<path>`. */
export const CLOUD_FILES_PATH = '/api/marketplace/catalog/files/'

const WEB_PROTOCOLS: ReadonlySet<string> = new Set(['http:', 'https:'])
/** A percent-encoded dot, slash or backslash: the browser leaves it as it is, and a server may decode it. */
const ENCODED_SEPARATOR = /%(2e|2f|5c)/i

function parseUrl(raw: string | undefined): URL | null {
  if (!raw) return null
  try {
    return new URL(raw)
  } catch {
    return null
  }
}

function webOrigin(raw: string | undefined): string {
  const url = parseUrl(raw)
  return url && WEB_PROTOCOLS.has(url.protocol) ? url.origin : ''
}

/** Automatos Cloud's origin and the build's API origin, the only two a Cloud file may load from. */
export function cloudFileOrigins(): string[] {
  return [AUTOMATOS_CLOUD_ORIGIN, webOrigin(process.env.NEXT_PUBLIC_API_URL)].filter(Boolean)
}

function isCatalogFilePath(pathname: string): boolean {
  if (!pathname.startsWith(CLOUD_FILES_PATH) || ENCODED_SEPARATOR.test(pathname)) return false
  const [digest, ...file] = pathname.slice(CLOUD_FILES_PATH.length).split('/')
  return THEME_DIGEST_PATTERN.test(digest) && file.join('/').length > 0
}

/**
 * True for `<origin>/api/marketplace/catalog/files/<64 hex>/<path>` written out on one of
 * `cloudFileOrigins()` exactly: never another host, a user name, a protocol-relative URL, a path
 * that climbs out of the files route once the browser resolves it, or one with a percent-encoded
 * dot, slash or backslash a server could decode into a climb.
 */
export function isCloudFileUrl(url: string): boolean {
  const parsed = parseUrl(url)
  if (!parsed || !isCatalogFilePath(parsed.pathname)) return false
  return cloudFileOrigins().some((origin) => parsed.origin === origin && url.startsWith(`${origin}${CLOUD_FILES_PATH}`))
}
