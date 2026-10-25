/**
 * PRD-261 A-01 (§6 "Kill switch and recovery") — safe mode: `?safe_mode=themes` boots Studio with
 * no theme code for that tab. The request is remembered in this tab's sessionStorage, so the next
 * page in the tab stays safe. Safe mode can only turn theme loading off; nothing here turns it on.
 *
 * Storage errors (blocked site data, a full store) mean "not remembered", never a throw.
 */
import { SAFE_MODE_PARAM, SAFE_MODE_STORAGE_KEY, SAFE_MODE_VALUE } from './constants'

export type SafeModeStorage = Pick<Storage, 'getItem' | 'setItem'>

const REMEMBERED = '1'
const LOG_PREFIX = '[theme-host]'

function remember(storage: SafeModeStorage | null): void {
  if (storage === null) return
  try {
    storage.setItem(SAFE_MODE_STORAGE_KEY, REMEMBERED)
  } catch (error) {
    console.warn(LOG_PREFIX, 'safe mode could not be remembered for this tab', error)
  }
}

function isRemembered(storage: SafeModeStorage | null): boolean {
  if (storage === null) return false
  try {
    return storage.getItem(SAFE_MODE_STORAGE_KEY) === REMEMBERED
  } catch (error) {
    console.warn(LOG_PREFIX, 'safe mode could not be read for this tab', error)
    return false
  }
}

/** True when this page asks for safe mode, or an earlier page in this tab did. */
export function isSafeModeRequested(search: string, storage: SafeModeStorage | null): boolean {
  if (new URLSearchParams(search).getAll(SAFE_MODE_PARAM).includes(SAFE_MODE_VALUE)) {
    remember(storage)
    return true
  }
  return isRemembered(storage)
}

/** This tab's sessionStorage, or null where reading the property itself throws (blocked site data). */
export function safeSessionStorage(): SafeModeStorage | null {
  try {
    return window.sessionStorage
  } catch (error) {
    console.warn(LOG_PREFIX, 'sessionStorage is not available', error)
    return null
  }
}

/** The page that turns safe mode on for this tab (app/safe-mode/page.tsx). */
export const SAFE_MODE_PATH = '/safe-mode'
/** The query that turns safe mode on, as a link writes it. */
export const SAFE_MODE_QUERY = `?${SAFE_MODE_PARAM}=${SAFE_MODE_VALUE}`

/** Turns safe mode on for this tab, as `?safe_mode=themes` does; the /safe-mode page calls it. */
export function requestSafeModeForThisTab(): void {
  isSafeModeRequested(SAFE_MODE_QUERY, safeSessionStorage())
}

/**
 * Safe mode in this tab: asked for by the query, by being the /safe-mode page, or by an earlier page in this tab.
 * Browser only: false where there is no window (a server render).
 */
export function isSafeModeHere(): boolean {
  if (typeof window === 'undefined') return false
  const { pathname, search } = window.location
  return isSafeModeRequested(pathname === SAFE_MODE_PATH ? SAFE_MODE_QUERY : search, safeSessionStorage())
}
