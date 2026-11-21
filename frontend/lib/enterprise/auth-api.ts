/**
 * PRD-277 Wave 5 — the enterprise web app's calls to the API's sign-in routes, through apiClient's
 * generic methods (the lib/marketplace/cloud-items.ts convention, so api-client.ts does not grow).
 *
 *   GET  /api/v1/auth/me                       who the session's person is (api/auth_web.py)
 *   GET  /api/v1/auth/sign-in-page             the provider's name for the one button (public)
 *   POST /api/v1/auth/sign-out                 204, or 200 {redirect_to} at the provider
 *   GET  /api/workspaces/{id}/team/waiting     a page of people waiting, oldest first (install admin)
 *   POST /api/workspaces/{id}/team/waiting/{u} place one of them here (step-up)
 *   POST /api/v1/auth/people/{u}/sessions/revoke, /deactivate  end access (step-up)
 *
 * The start route is not called: the browser is sent there (`signInStartUrl`), and the API's own
 * rule (`safe_return_path`) decides what `return_to` may be. Nothing here builds a redirect from a
 * query parameter.
 */
import { apiClient } from '@/lib/api-client'

export const ME_PATH = '/api/v1/auth/me'
export const SIGN_IN_PAGE_PATH = '/api/v1/auth/sign-in-page'
export const SIGN_OUT_PATH = '/api/v1/auth/sign-out'
/** The API's start route (api/auth_oidc.py); a step-up's start address must be this path. */
export const SIGN_IN_START_PATH = '/api/v1/auth/oidc/start'
export const RETURN_TO_PARAM = 'return_to'
/** What the sign-in page says before the API has answered with the provider's name. */
export const DEFAULT_DISPLAY_NAME = 'your organisation'

export type SystemRole = 'super_admin' | 'admin' | 'user'

export interface HeldWorkspace {
  readonly id: string
  readonly name: string
  /** The role the API's permission gate uses here; null when the member row holds none. */
  readonly role: string | null
}

/** GET /api/v1/auth/me. */
export interface WhoAmI {
  readonly edition: string
  readonly person: {
    readonly id: number
    readonly email: string | null
    readonly name: string | null
    readonly system_role: SystemRole
  }
  readonly workspaces: readonly HeldWorkspace[]
  readonly in_no_workspace: boolean
  readonly default_workspace_id: string | null
}

export interface WaitingPerson {
  readonly user_id: number
  readonly email: string | null
  readonly name: string | null
  readonly signed_in_first_at: string | null
}

/** The roles a waiting person may be given (never owner: api/team_waiting.py). */
export const PLACEABLE_ROLES = ['viewer', 'editor', 'admin'] as const
export type PlaceableRole = (typeof PLACEABLE_ROLES)[number]

export function fetchWhoAmI(): Promise<WhoAmI> {
  return apiClient.get<WhoAmI>(ME_PATH)
}

/** The provider's name for the button; the default when the API cannot be asked. */
export async function fetchDisplayName(): Promise<string> {
  try {
    const answer = await apiClient.get<{ display_name?: unknown }>(SIGN_IN_PAGE_PATH)
    const name = answer?.display_name
    return typeof name === 'string' && name.trim() ? name : DEFAULT_DISPLAY_NAME
  } catch (error) {
    console.warn('[sign-in] the provider name could not be read', error)
    return DEFAULT_DISPLAY_NAME
  }
}

const HTTPS_PREFIX_LENGTH = 'https://'.length
const LAST_CONTROL_CHAR = 0x1f
const DELETE_CHAR = 0x7f

/** Whitespace or a control character anywhere: a plain web address has none. */
function hasSpaceOrControl(value: string): boolean {
  if (/\s/.test(value)) return true
  return Array.from(value).some((char) => {
    const code = char.charCodeAt(0)
    return code <= LAST_CONTROL_CHAR || code === DELETE_CHAR
  })
}

/**
 * The provider's end-session address as the browser may be sent to it, or null: it must parse as
 * an absolute `https:` address with a host, no user-info (not even an empty one), and no
 * whitespace, control character or backslash. Anything else (another scheme, `https:` with no `//`,
 * credentials before the host) is null, and sign-out goes to the sign-in page instead.
 */
export function safeEndSessionAddress(value: unknown): string | null {
  if (typeof value !== 'string' || !value || hasSpaceOrControl(value)) return null
  // A backslash, which a browser reads as a slash, has no place in a plain address.
  if (!/^https:\/\//i.test(value) || value.includes('\\')) return null
  // Any `@` before the path is user-info, even an empty one.
  const authority = value.slice(HTTPS_PREFIX_LENGTH).split(/[/?#]/)[0]
  if (authority.includes('@')) return null
  let url: URL
  try {
    url = new URL(value)
  } catch {
    return null
  }
  if (url.protocol !== 'https:' || !url.hostname || url.username || url.password) return null
  return url.href
}

/** Sign out at the API; the provider's end-session address when it has a safe one, else null. */
export async function signOutAtApi(): Promise<string | null> {
  const answer = await apiClient.post<{ redirect_to?: unknown } | null>(SIGN_OUT_PATH)
  return safeEndSessionAddress(answer?.redirect_to)
}

/** The browser's address to start a sign-in at, bringing the person back to `returnTo`. */
export function signInStartUrl(returnTo: string | null | undefined, start: string = SIGN_IN_START_PATH): string {
  const query = new URLSearchParams(start.includes('?') ? start.slice(start.indexOf('?') + 1) : '')
  if (returnTo) query.set(RETURN_TO_PARAM, returnTo)
  const path = start.includes('?') ? start.slice(0, start.indexOf('?')) : start
  const search = query.toString()
  return `${apiClient.getBaseUrl()}${path}${search ? `?${search}` : ''}`
}

/** Where the browser is now: its path and query, the way back after a sign-in. */
export function currentPath(): string {
  if (typeof window === 'undefined') return '/'
  return `${window.location.pathname}${window.location.search}`
}

/** A page of the people waiting, oldest first (api/team_waiting.py), and how many wait in all. */
export interface WaitingPage {
  readonly people: readonly WaitingPerson[]
  readonly total: number
  readonly limit: number
  readonly offset: number
}

/** How many people one page of the waiting list shows (the API allows 1 to 100). */
export const WAITING_PAGE_SIZE = 50

export function fetchWaitingPeople(workspaceId: string, offset = 0, limit = WAITING_PAGE_SIZE): Promise<WaitingPage> {
  const query = new URLSearchParams({ limit: String(limit), offset: String(Math.max(0, offset)) })
  return apiClient.get<WaitingPage>(`/api/workspaces/${workspaceId}/team/waiting?${query.toString()}`)
}

export function placeWaitingPerson(workspaceId: string, userId: number, role: PlaceableRole): Promise<unknown> {
  return apiClient.post(`/api/workspaces/${workspaceId}/team/waiting/${userId}`, { role })
}

export function endSessionsOf(userId: number): Promise<{ revoked: number }> {
  return apiClient.post<{ revoked: number }>(`/api/v1/auth/people/${userId}/sessions/revoke`)
}

export function deactivatePerson(userId: number): Promise<{ sessions_revoked: number; devices_revoked: number }> {
  return apiClient.post(`/api/v1/auth/people/${userId}/deactivate`)
}
