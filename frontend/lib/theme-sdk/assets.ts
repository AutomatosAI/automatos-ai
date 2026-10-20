/**
 * PRD-261 A-13c — SDK area `assets`: the URL of a file inside the active theme's package.
 *
 * An installed theme's files are served by the same-origin files route under its package digest
 * (`/api/themes/files/<digest>/<path>`, A-08a, proxied by A-08b), so a theme never builds that URL
 * itself and never learns where the host keeps it. The host names the active install in the style
 * cookie and leaves the digest it loaded for that install in the digest cookie
 * (lib/theme-host/sources.ts rememberActiveDigest) before the theme's entry module is imported, so
 * by the time theme code runs both are there. The path is checked as the manifest's file paths are
 * (lib/theme-host/manifest.ts resolveUnderRoot): plain, relative, lowercase, no '..' and no
 * '//', and the URL must stay on this origin and under the package's folder.
 *
 * Null when the path is refused, when the style is not an installed theme (Classic, Studio, a
 * builtin theme, whose files are the app's own), when the digest is not known for that install,
 * and on the server, where no installed theme renders.
 */
import { resolveUnderRoot, themeFilesRoot } from '@/lib/theme-host/manifest'
import { isInstalledThemeStyle, themeRefOf } from '@/lib/theme-host/registry'
import { THEME_DIGEST_COOKIE, THEME_DIGEST_HINT_SEPARATOR } from '@/lib/theme-host/types'
import { UI_STYLE_COOKIE } from '@/lib/ui-style'

/** The ref of an installed theme: 'install:' and its install id. */
const INSTALL_REF_PREFIX = 'install:'
const COOKIE_SEPARATOR = '; '

/** One cookie's value from a `document.cookie` string; undefined when it is not set. */
function cookieValue(cookies: string, name: string): string | undefined {
  const prefix = `${name}=`
  return cookies.split(COOKIE_SEPARATOR).find((entry) => entry.startsWith(prefix))?.slice(prefix.length)
}

/** The install the style cookie names, or null for Classic, Studio, a builtin theme or no cookie. */
function activeInstallId(cookies: string): string | null {
  const style = cookieValue(cookies, UI_STYLE_COOKIE)
  if (style === undefined || !isInstalledThemeStyle(style)) return null
  const ref = themeRefOf(style)
  return ref === null ? null : ref.slice(INSTALL_REF_PREFIX.length)
}

/** The digest the host last loaded for `installId`, as its digest cookie says; null for another install or none. */
function digestOf(cookies: string, installId: string): string | null {
  const hint = cookieValue(cookies, THEME_DIGEST_COOKIE)
  if (hint === undefined) return null
  const at = hint.lastIndexOf(THEME_DIGEST_HINT_SEPARATOR)
  return at >= 0 && hint.slice(0, at) === installId ? hint.slice(at + 1) : null
}

/** The files folder of the active installed theme's package, or null when there is none to name. */
function activeFilesRoot(cookies: string): string | null {
  const installId = activeInstallId(cookies)
  if (installId === null) return null
  const digest = digestOf(cookies, installId)
  return digest === null ? null : themeFilesRoot(digest)
}

/** The URL of `path` (for example 'assets/mascot/wave.mp4') in the active theme's package, or null. */
export function assetUrl(path: string): string | null {
  if (typeof document === 'undefined') return null
  const root = activeFilesRoot(document.cookie)
  return root === null ? null : resolveUnderRoot(path, root)
}
