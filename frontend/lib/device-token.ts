/**
 * PRD-266 US-005b (§4.3 "Pair this browser", K8) — a paired browser's device token.
 *
 * On a local install with LOCAL_REMOTE_ACCESS=paired the anonymous lane is closed, so a browser
 * pairs like a phone (POST /api/v1/mobile/pair, platform "web") and keeps the `amd_…` token the
 * server returns once. It lives in localStorage, never only in memory, so a reload, a browser
 * restart and a server restart all keep the browser paired. Beside it is the browser's install
 * id, made once with crypto.randomUUID and kept when the token is cleared, so pairing again
 * reuses this browser's one device row. A browser offers randomUUID only in a secure context
 * (https, localhost); one that reaches the install over plain http on the LAN gets 16 random
 * bytes from getRandomValues as hex instead.
 *
 * The token reaches the API through the existing seams: apiClient's token getter
 * (components/local-auth-provider.tsx) and the local edition's useAuth().getToken
 * (lib/auth-hooks.ts). The one Next.js server surface that calls the API for the browser without
 * its headers and needs the token, the theme files proxy (app/api/themes/files/[...path]/route.ts),
 * reads it from a cookie written beside it (K8): SameSite=Strict, its Path the proxy's, Secure on
 * https, read only by the Next.js server and only in the local edition (the backend reads no
 * cookie). The cookie is a session cookie, written again from localStorage whenever the pairing
 * gate starts, so it never outlives the stored token.
 *
 * Pairing also makes the token's workspace the active one: the device lane refuses (403) a
 * request that names another workspace. Nothing here logs a token.
 */
import { THEME_FILES_PATH } from '@/lib/theme-host/constants'

export const DEVICE_TOKEN_STORAGE_KEY = 'automatos.device_token'
export const BROWSER_INSTALL_ID_STORAGE_KEY = 'automatos.browser_install_id'
export const DEVICE_TOKEN_COOKIE = 'automatos_device_token'
/** The key apiClient reads the active workspace (X-Workspace-ID) from. */
export const ACTIVE_WORKSPACE_STORAGE_KEY = 'last_active_workspace'
export const DEVICE_REVOKED = 'device_revoked'
export const DEVICE_PAIRING_REQUIRED = 'device_pairing_required'
/** The 401 codes that send a browser back to pairing (core/auth/device.py, core/auth/bearer_registry.py). */
export const DEVICE_REFUSAL_CODES: readonly string[] = [DEVICE_REVOKED, DEVICE_PAIRING_REQUIRED]

/** What services/mobile/pairing.py mints: `amd_` then token_urlsafe. Anything else is never stored or sent. */
const DEVICE_TOKEN_PATTERN = /^amd_[A-Za-z0-9_-]{20,128}$/
/** The install ids POST /pair accepts (api/mobile_devices.py INSTALL_ID_PATTERN). */
const INSTALL_ID_PATTERN = /^[A-Za-z0-9_-]{8,128}$/
const INSTALL_ID_BYTES = 16
const COOKIE_PATH = THEME_FILES_PATH.replace(/\/$/, '')
const NOT_KEPT = 'This browser could not save its pairing: its site storage is full or blocked.'
const COOKIE_EXPIRED = 'Thu, 01 Jan 1970 00:00:00 GMT'
const BEARER = 'Bearer '

/** The browser's localStorage, or null on the server or when the browser blocks site storage. */
function browserStorage(): Storage | null {
  if (typeof window === 'undefined') return null
  try {
    return window.localStorage
  } catch (error) {
    console.error('[device-token] browser storage is unavailable', error)
    return null
  }
}

function isDeviceToken(value: string | null | undefined): value is string {
  return typeof value === 'string' && DEVICE_TOKEN_PATTERN.test(value)
}

/** The cookie the theme files proxy reads: the token, or an expired empty value to remove it. */
function writeDeviceCookie(token: string | null): void {
  if (typeof document === 'undefined') return
  const secure = window.location.protocol === 'https:' ? '; Secure' : ''
  const value = token === null ? `=; Expires=${COOKIE_EXPIRED}` : `=${token}`
  document.cookie = `${DEVICE_TOKEN_COOKIE}${value}; Path=${COOKIE_PATH}; SameSite=Strict${secure}`
}

/** This browser's device token, or null when it has none (or holds something that is not one). */
export function readDeviceToken(): string | null {
  const token = browserStorage()?.getItem(DEVICE_TOKEN_STORAGE_KEY)
  return isDeviceToken(token) ? token : null
}

/** Keep a pairing: the token, its workspace as the active one, and the proxy's cookie. Throws when it cannot. */
export function storeDevicePairing(paired: { readonly token: string; readonly workspace_id: string }): void {
  if (!isDeviceToken(paired.token)) throw new Error('The server answered with a token this browser cannot use.')
  const storage = browserStorage()
  if (storage === null) throw new Error(NOT_KEPT)
  try {
    storage.setItem(ACTIVE_WORKSPACE_STORAGE_KEY, paired.workspace_id)
    storage.setItem(DEVICE_TOKEN_STORAGE_KEY, paired.token)
  } catch (error) {
    console.error('[device-token] could not keep the pairing', error)
    clearDeviceToken()
    throw new Error(NOT_KEPT)
  }
  writeDeviceCookie(paired.token)
}

/** Forget the token and its cookie; the install id stays, so pairing again reuses the device row. */
export function clearDeviceToken(): void {
  browserStorage()?.removeItem(DEVICE_TOKEN_STORAGE_KEY)
  writeDeviceCookie(null)
}

/** Write the proxy's cookie again from the stored token (a session cookie ends with the browser). */
export function restoreDeviceCookie(): void {
  const token = readDeviceToken()
  if (token !== null) writeDeviceCookie(token)
}

/** crypto.randomUUID where the browser offers it; else (plain http) 16 random bytes as hex. */
function newInstallId(): string {
  if (typeof crypto.randomUUID === 'function') return crypto.randomUUID()
  const bytes = crypto.getRandomValues(new Uint8Array(INSTALL_ID_BYTES))
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('')
}

/** This browser's install id, made once and kept beside the token. */
export function browserInstallId(): string {
  const storage = browserStorage()
  const kept = storage?.getItem(BROWSER_INSTALL_ID_STORAGE_KEY)
  if (typeof kept === 'string' && INSTALL_ID_PATTERN.test(kept)) return kept
  const made = newInstallId()
  storage?.setItem(BROWSER_INSTALL_ID_STORAGE_KEY, made)
  return made
}

/** The device refusal a 401's body names (`{"detail": {"code": …}}`), or null for any other 401. */
export function deviceRefusalCode(body: unknown): string | null {
  const detail = body !== null && typeof body === 'object' ? (body as { detail?: unknown }).detail : undefined
  const code = detail !== null && typeof detail === 'object' ? (detail as { code?: unknown }).code : undefined
  return typeof code === 'string' && DEVICE_REFUSAL_CODES.includes(code) ? code : null
}

/** Next.js server side: the Authorization a request's device cookie carries, or null without one. */
export function deviceAuthorizationFromCookies(cookieHeader: string | null): string | null {
  if (!cookieHeader) return null
  const prefix = `${DEVICE_TOKEN_COOKIE}=`
  const pair = cookieHeader.split(';').map((part) => part.trim()).find((part) => part.startsWith(prefix))
  const token = pair?.slice(prefix.length)
  return isDeviceToken(token) ? `${BEARER}${token}` : null
}
