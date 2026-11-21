/**
 * PRD-277 US-503 — the web app's own server routes, in enterprise, call the API as the browser's
 * person by forwarding the session cookie, and nothing else that identifies anyone.
 *
 * A route handler (chat, workflow stream, theme files) and the root layout's theme boot run on the
 * web app's server, so the browser's cookie reaches them, not the API. They pass on:
 *
 *   - the session cookie alone, rebuilt as `<name>=<value>` (never the browser's other cookies);
 *   - for a request that changes something, the cross-site header and the browser's Origin, so the
 *     API's own check (session_cookie.assert_cookie_request_allowed) judges the browser's request
 *     exactly as if it had come straight to the API;
 *   - the workspace the request names.
 *
 * No Authorization header and no API key: the API reads the cookie only when neither is present.
 * None of these values is ever logged.
 */
import { SESSION_COOKIE_NAME, SESSION_CSRF_HEADER, isUnsafeMethod } from './session-transport'

const WORKSPACE_HEADERS = ['x-workspace-id', 'x-workspace'] as const

/** The session cookie's value in a `Cookie` header, or null. */
export function sessionCookieFrom(cookieHeader: string | null | undefined): string | null {
  if (!cookieHeader) return null
  for (const part of cookieHeader.split(';')) {
    const at = part.indexOf('=')
    if (at < 0 || part.slice(0, at).trim() !== SESSION_COOKIE_NAME) continue
    const value = part.slice(at + 1).trim()
    return value || null
  }
  return null
}

/** The `Cookie` header that carries only the session (the value is passed on as the browser sent it). */
export function sessionCookieHeader(value: string): string {
  return `${SESSION_COOKIE_NAME}=${value}`
}

/** A key for the session that is not the session: its SHA-256, hex (a per-person cache key). */
export async function sessionKey(value: string): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(value))
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('')
}

/** The workspace the browser's request names, or null. */
export function namedWorkspace(request: Request): string | null {
  for (const name of WORKSPACE_HEADERS) {
    const value = request.headers.get(name)?.trim()
    if (value) return value
  }
  return null
}

/**
 * What a route handler sends the API for `request`, whose own call to the API uses `method`: the
 * session cookie, the workspace, and for an unsafe `method` the browser's cross-site header and
 * Origin. Empty when the browser sent no session cookie.
 */
export function forwardedSession(request: Request, method: string): Record<string, string> {
  const value = sessionCookieFrom(request.headers.get('cookie'))
  if (value === null) return {}
  const headers: Record<string, string> = { Cookie: sessionCookieHeader(value) }
  const workspace = namedWorkspace(request)
  if (workspace) headers['X-Workspace-ID'] = workspace
  if (!isUnsafeMethod(method)) return headers
  const csrf = request.headers.get(SESSION_CSRF_HEADER)
  const origin = request.headers.get('origin')
  if (csrf) headers[SESSION_CSRF_HEADER] = csrf
  if (origin) headers.Origin = origin
  return headers
}
