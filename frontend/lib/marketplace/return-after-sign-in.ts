/**
 * PRD-270 US-270-09a (§5.7, §8-Q8, K21) — the way back to a Marketplace item after sign-in or
 * sign-up.
 *
 * Nothing in the app honours a return path: Clerk's sign-in, the sign-up form and the SSO
 * callback all land on '/' (their pages stay untouched). So a signed-out visitor's Add, Sign in
 * and Start free remember where they were, in this tab's sessionStorage for 30 minutes, and
 * ResumeAfterSignIn (components/auth/resume-after-sign-in.tsx) takes it once the person is signed
 * in with a workspace. The same path rides `redirect_url`, the PRD's URL contract, for Clerk's own
 * flows.
 *
 * The only path ever remembered, sent or resumed is '/marketplace' with its tab, item and add
 * query, rebuilt from those three: another path, '//', a scheme, a backslash, a fragment, an
 * overlong value or any other parameter is dropped, so neither a crafted link nor a tampered store can send the person
 * anywhere else. Storage errors (blocked site data, a full store) mean nothing is remembered,
 * never a throw: the person signs in and lands where Clerk sends them.
 */

export const MARKETPLACE_PATH = '/marketplace'
export const SIGN_IN_PATH = '/sign-in'
export const SIGN_UP_PATH = '/sign-up'

/** This tab's sessionStorage key: the remembered path and when it expires. */
export const RETURN_PATH_STORAGE_KEY = 'automatos-marketplace-return'
/** How long a remembered path waits for the sign-in or sign-up: 30 minutes. */
export const RETURN_PATH_TTL_MS = 30 * 60 * 1000

/** `add` opens the item's add dialog only with this value (US-270-09b). */
const ADD_READY = '1'
const ADD_PARAM = 'add'
const KEPT_PARAMS = ['tab', 'item', ADD_PARAM] as const
/** A tab or item longer than this is dropped: no tab value, listing id or item key comes close. */
const MAX_PARAM_LENGTH = 256
const REDIRECT_PARAM = 'redirect_url'
const LOG_PREFIX = '[marketplace]'

/** '/marketplace' alone or with a query: no further path, fragment, backslash or whitespace. */
const MARKETPLACE_URL = /^\/marketplace(?:\?([^#\\\s]*))?$/

export type ReturnPathStorage = Pick<Storage, 'getItem' | 'setItem' | 'removeItem'>

interface Remembered {
  readonly path: string
  readonly expiresAt: number
}

/** This tab's sessionStorage, or null where reading the property itself throws (blocked site data). */
function tabStorage(): ReturnPathStorage | null {
  try {
    return window.sessionStorage
  } catch (error) {
    console.warn(LOG_PREFIX, 'sessionStorage is not available', error)
    return null
  }
}

function keptValue(given: URLSearchParams, name: (typeof KEPT_PARAMS)[number]): string | null {
  const value = given.get(name)
  if (!value || value.length > MAX_PARAM_LENGTH) return null
  return name === ADD_PARAM && value !== ADD_READY ? null : value
}

/**
 * `path` as a Marketplace return path, rebuilt from its tab, item and add, or null when it is
 * anything but '/marketplace' with a query.
 */
export function safeReturnPath(path: unknown): string | null {
  if (typeof path !== 'string') return null
  const match = MARKETPLACE_URL.exec(path)
  if (match === null) return null
  const given = new URLSearchParams(match[1] ?? '')
  const kept = KEPT_PARAMS.flatMap((name) => {
    const value = keptValue(given, name)
    return value === null ? [] : [[name, value]]
  })
  const query = new URLSearchParams(kept).toString()
  return query ? `${MARKETPLACE_PATH}?${query}` : MARKETPLACE_PATH
}

/** The Marketplace link that opens `item` on `tab` with its add dialog ready (US-270-09b). */
export function marketplaceItemPath(tab: string, item: string): string {
  return `${MARKETPLACE_PATH}?${new URLSearchParams({ tab, item, add: ADD_READY })}`
}

function withRedirect(page: string, returnPath: string): string {
  const redirect = safeReturnPath(returnPath) ?? MARKETPLACE_PATH
  return `${page}?${new URLSearchParams({ [REDIRECT_PARAM]: redirect })}`
}

/** `/sign-in?redirect_url=<the encoded return path>`. */
export function signInUrl(returnPath: string): string {
  return withRedirect(SIGN_IN_PATH, returnPath)
}

/** `/sign-up?redirect_url=<the encoded return path>`. */
export function signUpUrl(returnPath: string): string {
  return withRedirect(SIGN_UP_PATH, returnPath)
}

/** Remembers `path` for 30 minutes, in this tab; an unsafe path is not remembered. */
export function remember(
  path: string,
  storage: ReturnPathStorage | null = tabStorage(),
  now: number = Date.now(),
): void {
  const safe = safeReturnPath(path)
  if (safe === null || storage === null) return
  const remembered: Remembered = { path: safe, expiresAt: now + RETURN_PATH_TTL_MS }
  try {
    storage.setItem(RETURN_PATH_STORAGE_KEY, JSON.stringify(remembered))
  } catch (error) {
    console.warn(LOG_PREFIX, 'the way back to the Marketplace could not be kept', error)
  }
}

function parseRemembered(raw: string): Remembered | null {
  try {
    const value: unknown = JSON.parse(raw)
    if (typeof value !== 'object' || value === null) return null
    const { path, expiresAt } = value as Record<string, unknown>
    if (typeof path !== 'string' || typeof expiresAt !== 'number') return null
    return { path, expiresAt }
  } catch (error) {
    console.warn(LOG_PREFIX, 'the remembered way back to the Marketplace was malformed', error)
    return null
  }
}

/** Whether `remembered` is still in its 30 minutes (never longer, whatever the store says). */
function isCurrent(remembered: Remembered, now: number): boolean {
  return now < remembered.expiresAt && remembered.expiresAt <= now + RETURN_PATH_TTL_MS
}

/**
 * Once: the remembered path, or null when none was remembered, it expired or it is unsafe. Taking
 * it forgets it, whatever it held.
 */
export function take(storage: ReturnPathStorage | null = tabStorage(), now: number = Date.now()): string | null {
  if (storage === null) return null
  try {
    const raw = storage.getItem(RETURN_PATH_STORAGE_KEY)
    if (raw === null) return null
    storage.removeItem(RETURN_PATH_STORAGE_KEY)
    const remembered = parseRemembered(raw)
    return remembered !== null && isCurrent(remembered, now) ? safeReturnPath(remembered.path) : null
  } catch (error) {
    console.warn(LOG_PREFIX, 'the way back to the Marketplace could not be read', error)
    return null
  }
}
