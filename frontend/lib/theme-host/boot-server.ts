/**
 * PRD-261 A-08c — first paint for an installed theme. The root layout's ThemeBootResolver for
 * `theme:install:<uuid>` (plan §2.2): it asks the backend's boot route (A-08a,
 * GET /api/themes/boot/{installId}) for the install's ThemeBoot with the user's own session, so
 * the theme's classes, its prepaint table and its module integrity are in the first HTML.
 *
 * Why a server-side fetch and not apiClient: this runs in the root layout on the server, before
 * any page code, and apiClient is the browser's client (its token comes from the mounted Clerk
 * session and its workspace from the page). The boot route is not workspace-scoped (it resolves
 * the workspace from the install, §2.2), so no X-Workspace-ID is needed. In SaaS the Clerk
 * session's token goes as Bearer and a signed-out request asks nothing; in the local edition no
 * token is sent and the backend serves the local operator. On a local install in
 * LOCAL_REMOTE_ACCESS=paired mode that lane is closed: the boot route answers 401
 * (device_pairing_required), which paints the base boot quietly, and the client host then loads
 * the theme with the browser's own device token (PRD-266 K8: the layout reads no device cookie).
 *
 * The fetch has 300 ms (INSTALLED_BOOT_TIMEOUT_MS). When it fails, times out or answers
 * anything but a boot, the resolver throws and createThemeBootResolver (boot.ts) paints the base
 * boot (Studio's classes); the client host then loads the theme as before. A 401, 403 or 404 (an
 * expired session, a stale cookie, an install removed or of another workspace) is no theme to
 * paint, not a failure: the base boot, with nothing logged.
 *
 * A per-process cache keeps a boot for 60 s keyed by (user, install, digest): one slot per user
 * and install holds the boot and the one package digest its integrity names, and it is served
 * only while that digest is the one the client host last saw active for the install (the
 * THEME_DIGEST_COOKIE hint sources.ts writes from GET /api/themes/active). So a boot of one user
 * is never served to another, one boot's integrity never mixes two digests, and after an update
 * or a rollback the first load the client host sees it on leaves the new digest, and the next
 * request fetches the new boot instead of serving the old one for the rest of its 60 s. The hint
 * can only make the server ask again; what is painted always comes from the backend. Only a boot
 * of a theme that runs (one with integrity) is kept: a base boot means the user may not run it
 * yet (no opt-in, a revoked version), which an opt-in followed by a reload must change at once.
 *
 * The answer passes sanitizeThemeBoot before it is kept or written: an integrity entry whose URL
 * is not a module of the files route or whose value is not `sha256-<base64>` is dropped
 * (import-map.ts), and the import map is scriptSafeJson of the whole object (A0 D16).
 */
import { isSaaS } from '@/lib/auth-edition'

import { baseThemeBoot } from './boot'
import { sanitizeThemeBoot } from './boot-sanitize'
import type { ThemeBoot, ThemeBootResolver } from './boot-types'
import { THEME_DIGEST_PATTERN, THEME_FILES_PATH } from './constants'
import { isInstalledThemeStyle, themeRefOf } from './registry'
import type { UiStyle } from './registry-types'
import { THEME_DIGEST_COOKIE, THEME_DIGEST_HINT_SEPARATOR } from './types'

/** The time the boot fetch may take before the page paints the base boot. */
export const INSTALLED_BOOT_TIMEOUT_MS = 300
/** How long a fetched boot is kept per user and install. */
export const INSTALLED_BOOT_CACHE_TTL_MS = 60_000
/** The cache's bound, so a long-lived process never grows it without end. */
export const INSTALLED_BOOT_CACHE_MAX_ENTRIES = 1_000
/** The backend's boot route (A-08a); the install id is appended. */
export const THEME_BOOT_PATH = '/api/themes/boot/'

const BACKEND_URL = // Server-side: BACKEND_INTERNAL_URL (container DNS, local edition) beats the
// browser-facing NEXT_PUBLIC_API_URL — inside the frontend container 'localhost'
// is the frontend itself. Unset in SaaS ⇒ identical to before. (PRD-209)
process.env.BACKEND_INTERNAL_URL || process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

const INSTALL_REF_PREFIX = 'install:'
/** The local edition has one identity, the local operator (hybrid.py's anonymous fallback). */
const LOCAL_OPERATOR = 'local-operator'
/** Answers that mean "no theme to paint for this caller", not a broken theme. */
const NO_THEME_STATUSES: readonly number[] = [401, 403, 404]
const LOG_PREFIX = '[theme-boot]'
const JSON_TYPE = 'application/json'

/** The credential a boot is fetched with: a Bearer header (SaaS) or none (local); not ok: ask nothing. */
export type BootCredential = { readonly ok: true; readonly authorization: string | null } | { readonly ok: false }

/** Who the boot is for. The credential is read only on a cache miss. */
export interface BootCaller {
  readonly user: string
  readonly credential: () => Promise<BootCredential>
}

interface CachedBoot {
  readonly digest: string
  readonly boot: ThemeBoot
  readonly expiresAt: number
}

/** Per (user, install): the boot, the digest its integrity names (the key's third part), and when it expires. */
export type InstalledBootCache = Map<string, CachedBoot>

export interface InstalledBootDeps {
  readonly caller: () => Promise<BootCaller | null>
  /** The digest the client host last saw active for the install (THEME_DIGEST_COOKIE), or null. */
  readonly digestHint: (installId: string) => Promise<string | null>
  readonly fetchBoot: (url: string, init: RequestInit) => Promise<Response>
  readonly now: () => number
  readonly timeoutMs: number
  readonly ttlMs: number
  readonly maxEntries: number
  readonly cache: InstalledBootCache
}

/** The one per-process cache the root layout's resolver uses. */
const PROCESS_BOOT_CACHE: InstalledBootCache = new Map()

/**
 * SaaS: the Clerk session (null when signed out or unreadable). Local: the local operator, no
 * token. Clerk's server module is imported only here, in SaaS, so the local edition and any
 * render outside a request (where auth() refuses to run) never load it with the layout.
 */
export async function sessionBootCaller(): Promise<BootCaller | null> {
  if (!isSaaS) return { user: LOCAL_OPERATOR, credential: async () => ({ ok: true, authorization: null }) }
  try {
    const { auth } = await import('@clerk/nextjs/server')
    const session = await auth()
    if (!session.userId) return null
    const credential = async (): Promise<BootCredential> => {
      const token = await session.getToken()
      return token ? { ok: true, authorization: `Bearer ${token}` } : { ok: false }
    }
    return { user: session.userId, credential }
  } catch (error) {
    console.error(LOG_PREFIX, 'could not read the session', error)
    return null
  }
}

/** The digest a THEME_DIGEST_COOKIE value names for `installId`; null for another install or a malformed value. */
export function parseDigestHint(value: string | undefined, installId: string): string | null {
  if (value === undefined) return null
  const at = value.lastIndexOf(THEME_DIGEST_HINT_SEPARATOR)
  if (at < 0 || value.slice(0, at) !== installId) return null
  const digest = value.slice(at + 1)
  return THEME_DIGEST_PATTERN.test(digest) ? digest : null
}

/** The request's digest hint for `installId`; null without the cookie or outside a request. */
export async function requestDigestHint(installId: string): Promise<string | null> {
  try {
    const { cookies } = await import('next/headers')
    const store = await cookies()
    return parseDigestHint(store.get(THEME_DIGEST_COOKIE)?.value, installId)
  } catch (error) {
    console.error(LOG_PREFIX, 'could not read the digest cookie', error)
    return null
  }
}

/** The install a style names, or null for anything but `theme:install:<uuid>`. */
export function installIdOfStyle(style: UiStyle): string | null {
  if (!isInstalledThemeStyle(style)) return null
  const ref = themeRefOf(style)
  return ref === null ? null : ref.slice(INSTALL_REF_PREFIX.length)
}

/** The slot of a user's install: both values, unambiguously (the digest is checked against the entry's). */
export function bootCacheKey(user: string, installId: string): string {
  return JSON.stringify([user, installId])
}

/**
 * The one package digest every integrity URL names; null when the boot has no integrity (the
 * theme does not run). Throws when the URLs name more than one digest: no boot mixes two packages.
 */
export function bootDigest(boot: ThemeBoot): string | null {
  if (boot.integrity === null) return null
  const digests = new Set(Object.keys(boot.integrity).map((url) => url.slice(THEME_FILES_PATH.length).split('/')[0]))
  if (digests.size !== 1) throw new Error(`the boot's integrity names ${digests.size} package digests`)
  return Array.from(digests)[0]
}

/** The answer as a boot that may be written into the page; throws when it is not an object. */
export function parseInstalledBoot(value: unknown): ThemeBoot {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('the boot route did not answer with an object')
  }
  // Every field is checked by sanitizeThemeBoot, which reads each one as unknown.
  return sanitizeThemeBoot(value as ThemeBoot)
}

/** `work` with a deadline: past it the request is aborted and the promise rejects. */
async function withDeadline<T>(work: (signal: AbortSignal) => Promise<T>, timeoutMs: number): Promise<T> {
  const controller = new AbortController()
  let timer: ReturnType<typeof setTimeout> | undefined
  const deadline = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      controller.abort()
      reject(new Error(`GET ${THEME_BOOT_PATH}… did not answer within ${timeoutMs} ms`))
    }, timeoutMs)
  })
  try {
    return await Promise.race([work(controller.signal), deadline])
  } finally {
    clearTimeout(timer)
  }
}

function discardBody(response: Response): void {
  response.body?.cancel().catch((error: unknown) => console.error(LOG_PREFIX, 'could not discard a body', error))
}

/** The install's boot from the backend; null when it answers that there is no theme to paint. */
export async function fetchInstalledBoot(
  installId: string,
  authorization: string | null,
  deps: Pick<InstalledBootDeps, 'fetchBoot' | 'timeoutMs'>,
): Promise<ThemeBoot | null> {
  const headers: Record<string, string> =
    authorization === null ? { Accept: JSON_TYPE } : { Accept: JSON_TYPE, Authorization: authorization }
  const url = `${BACKEND_URL}${THEME_BOOT_PATH}${encodeURIComponent(installId)}`
  return withDeadline(async (signal) => {
    const response = await deps.fetchBoot(url, { headers, cache: 'no-store', redirect: 'manual', signal })
    if (NO_THEME_STATUSES.includes(response.status)) {
      discardBody(response)
      return null
    }
    if (!response.ok) {
      discardBody(response)
      throw new Error(`GET ${THEME_BOOT_PATH}… answered ${response.status}`)
    }
    return parseInstalledBoot(await response.json())
  }, deps.timeoutMs)
}

/** Keeps a boot of a theme that runs; drops the key otherwise. Expired entries go first, then the oldest. */
function remember(deps: InstalledBootDeps, key: string, boot: ThemeBoot): void {
  const digest = bootDigest(boot)
  deps.cache.delete(key)
  if (digest === null) return
  const now = deps.now()
  deps.cache.forEach((entry, held) => {
    if (entry.expiresAt <= now) deps.cache.delete(held)
  })
  while (deps.cache.size >= deps.maxEntries) {
    const oldest = deps.cache.keys().next()
    if (oldest.done) break
    deps.cache.delete(oldest.value)
  }
  deps.cache.set(key, { digest, boot, expiresAt: now + deps.ttlMs })
}

/** The kept boot while it is fresh and its digest is the one the client last saw active (when it names one). */
async function cachedBoot(deps: InstalledBootDeps, key: string, installId: string): Promise<ThemeBoot | null> {
  const cached = deps.cache.get(key)
  if (cached === undefined || cached.expiresAt <= deps.now()) return null
  const hint = await deps.digestHint(installId)
  return hint === null || hint === cached.digest ? cached.boot : null
}

function defaultDeps(): InstalledBootDeps {
  return {
    caller: sessionBootCaller,
    digestHint: requestDigestHint,
    fetchBoot: (url, init) => fetch(url, init),
    now: () => Date.now(),
    timeoutMs: INSTALLED_BOOT_TIMEOUT_MS,
    ttlMs: INSTALLED_BOOT_CACHE_TTL_MS,
    maxEntries: INSTALLED_BOOT_CACHE_MAX_ENTRIES,
    cache: PROCESS_BOOT_CACHE,
  }
}

/**
 * The resolver for `theme:install:*` that createThemeBootResolver takes as `installed`. It
 * throws on a failed or slow fetch, so the caller paints the base boot and logs why.
 */
export function createInstalledBootResolver(overrides: Partial<InstalledBootDeps> = {}): ThemeBootResolver {
  const deps: InstalledBootDeps = { ...defaultDeps(), ...overrides }
  return async (style) => {
    const installId = installIdOfStyle(style)
    if (installId === null) return baseThemeBoot(style)
    const caller = await deps.caller()
    if (caller === null) return baseThemeBoot(style)
    const key = bootCacheKey(caller.user, installId)
    const cached = await cachedBoot(deps, key, installId)
    if (cached !== null) return cached
    const credential = await caller.credential()
    if (!credential.ok) return baseThemeBoot(style)
    const boot = await fetchInstalledBoot(installId, credential.authorization, deps)
    if (boot === null) {
      deps.cache.delete(key)
      return baseThemeBoot(style)
    }
    remember(deps, key, boot)
    return boot
  }
}

/** The root layout's resolver for installed themes, with the per-process cache. */
export const installedThemeBootResolver: ThemeBootResolver = createInstalledBootResolver()
