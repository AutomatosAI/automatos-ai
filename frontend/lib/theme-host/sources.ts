/**
 * PRD-261 A-04a — where a theme comes from (plan §2.2 `ThemeSource`).
 *
 * - **installed** (`install:<uuid>`): reads GET /api/themes/active through the backend client,
 *   checks that it names this install and carries a sha256 digest, checks its manifest with every
 *   file under `/api/themes/files/<digest>/` (a package's by its fixed layout and the answer's
 *   `defaultLook`, manifest.ts), then imports the entry with the browser's own
 *   import(), so its bare imports resolve through the page's import map to the host's modules.
 *   It is the only source: no theme is compiled into the app since A-19f1 (builtin Play left
 *   core; Play ships as a package from the theme kit).
 *
 * The active-theme request goes through apiClient's named method (getActiveTheme), so the
 * route-contract check sees it. When it names no theme for this install (a stale
 * `theme:install:` cookie) there is no theme to load, which is not a failure: A-07b answers 200
 * with `theme: null` (no choice, the install was removed, or a `disable` revocation, which also
 * carries a top-level `revoked`) or with another install (the workspace default), and a 404 is a
 * backend without the route or no workspace. The preset fetch below is a same-origin static file
 * under the theme's files folder, not the backend client, so it is a plain fetch.
 *
 * A-08b: a refused session is not the theme's failure. A 401 or 403 on the active-theme read, on
 * the preset or on the entry (the same-origin files proxy answers 401 when signed out, as after
 * an expired session) fails the load with reason `auth`, which the failure ledger never counts.
 * import() reports no status, so after a failed entry import one byte of the entry is fetched
 * again (a same-origin file of the proxy, not the backend client) to read its status.
 *
 * A-08c: the digest the active-theme answer names is left in a cookie for the root layout's boot
 * cache (boot-server.ts), which fetches a boot of any other digest again.
 *
 * The answer's other fields: `optIn.required` (private code this member has not opted in to)
 * fails the load as `opt-in` before any theme file is asked for, since the files proxy would
 * 404 each one; a `warn` revocation (`theme.revoked`) loads the theme and hands the warning on
 * (ThemeLoadOk.revoked) for the host's notice. A 5xx from the files proxy, or a network failure,
 * fails as `unavailable`: the files could not be reached, which is not the theme's fault.
 */
import { apiClient } from '@/lib/api-client'
import { writeCookie } from '@/lib/cookies'
import { UI_STYLE_COOKIE_MAX_AGE_S, type UiStyle } from '@/lib/ui-style'

import { THEME_ACTIVE_ENDPOINT, THEME_LOAD_TIMEOUT_MS } from './constants'
import { isRecord } from './css-value'
import { exposeHostModules } from './host-modules'
import { loadFromManifest, settleLoad, step, withTimeout } from './loader'
import { parseActiveInstall, parseThemeManifest } from './manifest'
import { importThemeModule } from './native-import'
import { isInstalledThemeStyle, themeRefOf } from './registry'
import type { ThemeRef } from './registry-types'
import { THEME_DIGEST_COOKIE, THEME_DIGEST_HINT_SEPARATOR } from './types'
import type {
  LoaderDeps,
  SpikeManifest,
  ThemeLoadAbsent,
  ThemeLoadOk,
  ThemeLoadResult,
  ThemeRevocationWarning,
  ThemeSource,
} from './types'

const INSTALL_REF_PREFIX = 'install:'
/** What GET /api/themes/active answers when the workspace has no active install. */
const NO_ACTIVE_INSTALL_STATUS = 404

/** The statuses that mean the session was refused, not that the theme is broken (A-08b). */
const AUTH_REFUSED_STATUSES: readonly number[] = [401, 403]
/** apiClient's 401 error carries no status; its message starts with this (api-client.ts httpErrorOf). */
const API_CLIENT_401_PREFIX = 'HTTP 401'
/** The status probe after a failed import asks for one byte, so it never downloads the entry again. */
const STATUS_PROBE_RANGE = 'bytes=0-0'
/** From this status up the files proxy (or the backend behind it) could not serve the file. */
const UNAVAILABLE_FROM_STATUS = 500
/** The revocation level that still runs the theme (orchestrator/services/themes/revocations.py). */
const REVOKE_WARN = 'warn'

/** The HTTP status apiClient puts on the error it throws, if it has one. */
function httpStatusOf(error: unknown): unknown {
  return (error as { status?: unknown } | null)?.status
}

/** Whether a request failed because the session was refused: a 401 or 403 (apiClient's 401 by its message). */
export function isAuthRefusal(error: unknown): boolean {
  const status = httpStatusOf(error)
  if (typeof status === 'number') return AUTH_REFUSED_STATUSES.includes(status)
  return error instanceof Error && error.message.startsWith(API_CLIENT_401_PREFIX)
}

/** Fails as `reason` with `error`, whatever step it runs inside (step keeps the first reason). */
function failAs<T>(reason: 'auth' | 'opt-in' | 'unavailable', error: unknown): Promise<T> {
  return step<T>(reason, () => {
    throw error
  })
}

/** A file status that means the files could not be reached: a 5xx, or no answer at all (null). */
export function isUnavailableStatus(status: number | null): boolean {
  return status === null || status >= UNAVAILABLE_FROM_STATUS
}

/** Runs `run`; a refused session fails as `auth` rather than as the reason of the step around it. */
export async function refusedAsAuth<T>(run: () => Promise<T>): Promise<T> {
  try {
    return await run()
  } catch (error) {
    if (!isAuthRefusal(error)) throw error
    return failAs<T>('auth', error)
  }
}

/** GET /api/themes/active through apiClient; null when it answers 404 (no route or no workspace). */
export async function fetchActiveInstall(): Promise<unknown> {
  try {
    return await apiClient.getActiveTheme()
  } catch (error) {
    if (httpStatusOf(error) === NO_ACTIVE_INSTALL_STATUS) return null
    throw error
  }
}

function noActiveInstall(detail: string): ThemeLoadAbsent {
  return { ok: false, reason: null, detail }
}

/**
 * Why the active-theme answer names no theme for `installId` (none, a revocation or another
 * install), or null when it names this one or is malformed (parseActiveInstall refuses that).
 */
export function absenceOf(answer: unknown, installId: string): string | null {
  const answered = `GET ${THEME_ACTIVE_ENDPOINT}`
  if (answer === null) return `${answered} answered ${NO_ACTIVE_INSTALL_STATUS}: no active install`
  if (!isRecord(answer)) return null
  const theme = answer.theme
  if (theme === null) {
    return answer.revoked == null
      ? `${answered} names no installed theme: no active install`
      : `${answered} names no installed theme: the install's version is revoked`
  }
  const other = isRecord(theme) ? theme.installId : undefined
  if (typeof other === 'string' && other !== installId) return `${answered} names the install ${other}, not ${installId}`
  return null
}

/** The `theme` object of the active-theme answer, or null. */
function activeThemeOf(answer: unknown): Record<string, unknown> | null {
  return isRecord(answer) && isRecord(answer.theme) ? answer.theme : null
}

/** Whether the answer says this member has not opted in to the theme's private code (Q6). */
export function optInRequired(answer: unknown): boolean {
  const optIn = activeThemeOf(answer)?.optIn
  return isRecord(optIn) && optIn.required === true
}

/** The answer's `warn` revocation of the running version, or null (a `disable` one names no theme). */
export function revocationWarningOf(answer: unknown): ThemeRevocationWarning | null {
  const revoked = activeThemeOf(answer)?.revoked
  if (!isRecord(revoked) || revoked.level !== REVOKE_WARN) return null
  return { level: REVOKE_WARN, reason: typeof revoked.reason === 'string' ? revoked.reason : '' }
}

type ActiveManifest =
  | { readonly manifest: SpikeManifest; readonly revoked: ThemeRevocationWarning | null }
  | { readonly absent: string }

/**
 * A theme file's JSON, from the files proxy on the page's own origin; a failure carries its
 * status. A 5xx or a network failure fails as `unavailable`.
 */
async function fetchSameOriginJson(url: string): Promise<unknown> {
  let response: Response
  try {
    response = await fetch(url, { credentials: 'same-origin', cache: 'no-store' })
  } catch (error) {
    return failAs('unavailable', error)
  }
  if (!response.ok) {
    const failure = Object.assign(new Error(`GET ${url} answered ${response.status}`), { status: response.status })
    if (isUnavailableStatus(response.status)) return failAs('unavailable', failure)
    throw failure
  }
  return response.json()
}

/**
 * The status a theme file answers with, read with a one-byte same-origin GET of the files proxy;
 * null when the probe itself fails, so the caller keeps the original error.
 */
export async function themeFileStatus(url: string): Promise<number | null> {
  try {
    const response = await fetch(url, { credentials: 'same-origin', cache: 'no-store', headers: { Range: STATUS_PROBE_RANGE } })
    // Only the status is wanted; a body that will not cancel changes nothing about it.
    response.body?.cancel().catch(() => undefined)
    return response.status
  } catch (error) {
    console.error('[theme-host]', 'status probe', url, error)
    return null
  }
}

/**
 * Imports an installed theme's entry. When the import fails and the entry answers 401 or 403,
 * the load fails as `auth` (the session, not the theme); a 5xx or no answer to the probe fails as
 * `unavailable`; any other failure is rethrown as it was.
 */
export async function importInstalledEntry(
  url: string,
  importModule: (url: string) => Promise<unknown> = importThemeModule,
  probeStatus: (url: string) => Promise<number | null> = themeFileStatus,
): Promise<unknown> {
  try {
    return await importModule(url)
  } catch (error) {
    const status = await probeStatus(url)
    if (isUnavailableStatus(status)) return failAs('unavailable', error)
    if (status === null || !AUTH_REFUSED_STATUSES.includes(status)) throw error
    return refusedAsAuth(() => Promise.reject(Object.assign(new Error(`GET ${url} answered ${status}`), { status })))
  }
}

/**
 * A-08c: leaves the digest the active-theme answer named for `installId` in THEME_DIGEST_COOKIE,
 * so the next request's boot cache refetches a boot of another digest. Writes only on a change.
 */
export function rememberActiveDigest(installId: string, digest: string): void {
  const value = `${installId}${THEME_DIGEST_HINT_SEPARATOR}${digest}`
  if (document.cookie.split('; ').includes(`${THEME_DIGEST_COOKIE}=${value}`)) return
  writeCookie(THEME_DIGEST_COOKIE, value, UI_STYLE_COOKIE_MAX_AGE_S)
}

export function defaultLoaderDeps(): LoaderDeps {
  return {
    exposeHost: exposeHostModules,
    fetchActive: fetchActiveInstall,
    fetchJson: (url) => refusedAsAuth(() => fetchSameOriginJson(url)),
    importModule: (url) => importInstalledEntry(url),
    rememberDigest: rememberActiveDigest,
    doc: document,
    timeoutMs: THEME_LOAD_TIMEOUT_MS,
  }
}

/** The active install's checked manifest, with its files resolved under the package's folder, or why there is none to load. */
async function readActiveManifest(installId: string, deps: LoaderDeps): Promise<ActiveManifest> {
  const timedOut = `GET ${THEME_ACTIVE_ENDPOINT} did not answer within ${deps.timeoutMs} ms`
  const answer = await withTimeout(refusedAsAuth(() => deps.fetchActive()), deps.timeoutMs, timedOut)
  const absent = absenceOf(answer, installId)
  if (absent !== null) return { absent }
  if (optInRequired(answer)) {
    return failAs<ActiveManifest>('opt-in', new Error(`GET ${THEME_ACTIVE_ENDPOINT}: the member has not opted in to the install ${installId}`))
  }
  const active = parseActiveInstall(answer, installId)
  if (!active.ok) throw new Error(active.detail)
  deps.rememberDigest?.(installId, active.digest)
  const parsed = parseThemeManifest(active.manifest, active.filesRoot, active.defaultLook)
  if (!parsed.ok) throw new Error(parsed.detail)
  return { manifest: parsed.manifest, revoked: revocationWarningOf(answer) }
}

/** The theme installed in the workspace as `installId` (a lowercase uuid). */
export function installedThemeSource(installId: string): ThemeSource {
  const ref: ThemeRef = `${INSTALL_REF_PREFIX}${installId}`
  return {
    ref,
    load: (deps) =>
      settleLoad(async (): Promise<ThemeLoadOk | ThemeLoadAbsent> => {
        const host = await step('load', () => deps.exposeHost())
        const active = await step('manifest', () => readActiveManifest(installId, deps))
        if ('absent' in active) return noActiveInstall(active.absent)
        const loaded = await loadFromManifest(ref, active.manifest, host.sdkVersion, deps)
        return active.revoked === null ? loaded : { ...loaded, revoked: active.revoked }
      }),
  }
}

/** The install a style names, or null for Classic and Studio. */
export function installIdOf(style: UiStyle): string | null {
  if (!isInstalledThemeStyle(style)) return null
  const ref = themeRefOf(style)
  return ref === null ? null : ref.slice(INSTALL_REF_PREFIX.length)
}

/** The source the host loads for a style: an installed theme's; Classic and Studio have none. */
export function sourceForStyle(style: UiStyle): ThemeSource | null {
  const installId = installIdOf(style)
  return installId === null ? null : installedThemeSource(installId)
}

/** Loads the theme a style names. Never throws; a style with no source is a 'load' failure. */
export async function loadThemeForStyle(style: UiStyle, deps: LoaderDeps = defaultLoaderDeps()): Promise<ThemeLoadResult> {
  const source = sourceForStyle(style)
  if (source === null) return { ok: false, reason: 'load', detail: `no theme source for the style ${style}` }
  return source.load(deps)
}
