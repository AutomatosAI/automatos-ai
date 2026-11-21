/**
 * PRD-277 US-302, US-501 — how a browser request carries the enterprise session.
 *
 * In enterprise the API keeps the person's session in its `__Host-` cookie, which scripts cannot
 * read (orchestrator/core/auth/session_cookie.py). The web app never holds a token: apiClient's
 * token getter returns null, the page and the API share one address (PRD-277 §5.3), and the
 * browser attaches the cookie itself. What the web app adds is the header the API's cross-site
 * check asks for on every request that changes something (`SESSION_CSRF_HEADER`, default
 * `X-Automatos-CSRF`): another site's form cannot send a custom header, and the API's CORS
 * settings do not allow it, so a script on another site cannot either.
 *
 * `withSession` is applied in ONE place, lib/api-client.ts, to every request it sends. In `local`
 * and `saas` it hands back the very same init object, so those editions send exactly what they
 * sent before.
 *
 * The names are the API's defaults. An install that changes SESSION_COOKIE_NAME or
 * SESSION_CSRF_HEADER on the API must change these to match.
 */
import { isEnterprise } from '@/lib/edition-name'

/** The API's session cookie (SESSION_COOKIE_NAME's default). */
export const SESSION_COOKIE_NAME = '__Host-automatos_session'
/** The header the API's cross-site check asks for (SESSION_CSRF_HEADER's default). */
export const SESSION_CSRF_HEADER = 'X-Automatos-CSRF'
/** Any non-blank value passes: the header's presence is what another site cannot forge. */
export const SESSION_CSRF_VALUE = '1'
const SAFE_METHODS: ReadonlySet<string> = new Set(['GET', 'HEAD', 'OPTIONS'])

/** Whether `method` may change something (the API checks those only); GET when none is given. */
export function isUnsafeMethod(method: string | undefined): boolean {
  return !SAFE_METHODS.has((method || 'GET').toUpperCase())
}

/** The cross-site header, in enterprise and for an unsafe method; none otherwise. */
export function csrfHeaders(method: string | undefined, enterprise: boolean = isEnterprise): Record<string, string> {
  return enterprise && isUnsafeMethod(method) ? { [SESSION_CSRF_HEADER]: SESSION_CSRF_VALUE } : {}
}

function headerRecord(headers: HeadersInit | undefined): Record<string, string> {
  if (headers === undefined) return {}
  if (headers instanceof Headers) return Object.fromEntries(headers.entries())
  if (Array.isArray(headers)) return Object.fromEntries(headers)
  return { ...headers }
}

/**
 * `init` as the browser should send it. Enterprise: a copy that sends credentials, with the
 * cross-site header added when the method may change something. Every other edition: `init` itself.
 */
export function withSession(init: RequestInit, enterprise: boolean = isEnterprise): RequestInit {
  if (!enterprise) return init
  return {
    ...init,
    credentials: 'include',
    headers: { ...headerRecord(init.headers), ...csrfHeaders(init.method, true) },
  }
}

/**
 * Headers for a caller that builds its own request and does not say its method
 * (`apiClient.getAuthHeaders`): enterprise always adds the cross-site header, which a GET simply
 * does not need. Every other edition: none.
 */
export function sessionHeadersForAnyMethod(enterprise: boolean = isEnterprise): Record<string, string> {
  return enterprise ? { [SESSION_CSRF_HEADER]: SESSION_CSRF_VALUE } : {}
}

/** The stored workspace as a header, in enterprise only (a request outside apiClient names it too). */
export function enterpriseWorkspaceHeaders(enterprise: boolean = isEnterprise): Record<string, string> {
  if (!enterprise || typeof window === 'undefined') return {}
  try {
    const workspaceId = window.localStorage.getItem('last_active_workspace')
    return workspaceId ? { 'X-Workspace-ID': workspaceId } : {}
  } catch (error) {
    console.warn('[session] the stored workspace could not be read', error)
    return {}
  }
}
