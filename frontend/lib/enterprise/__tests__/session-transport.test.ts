// @vitest-environment node
/**
 * PRD-277 US-302, US-503 — the session travels by itself; the web app adds only the cross-site header.
 *
 *   - Browser (session-transport.ts): in enterprise every request sends credentials and an
 *     unsafe one carries `X-Automatos-CSRF`; in local and saas the init comes back untouched
 *     (the same object), so those editions send what they sent before.
 *   - Server (server-session.ts): a route handler passes on the session cookie alone (never the
 *     browser's other cookies), the workspace, and for an unsafe call the browser's cross-site
 *     header and Origin; never an Authorization header or an API key.
 */
import { describe, expect, it } from 'vitest'

import {
  SESSION_COOKIE_NAME,
  SESSION_CSRF_HEADER,
  csrfHeaders,
  isUnsafeMethod,
  sessionHeadersForAnyMethod,
  withSession,
} from '@/lib/enterprise/session-transport'
import { forwardedSession, sessionCookieFrom, sessionCookieHeader, sessionKey } from '@/lib/enterprise/server-session'

const SESSION = 'session' + '-' + 'v'.repeat(32)
const OTHER_COOKIES = 'theme=dark; automatos-tone=light'

describe('withSession (the browser)', () => {
  it('hands back the very same init outside enterprise', () => {
    const init: RequestInit = { method: 'POST', headers: { 'Content-Type': 'application/json' } }
    expect(withSession(init, false)).toBe(init)
    expect(csrfHeaders('POST', false)).toEqual({})
    expect(sessionHeadersForAnyMethod(false)).toEqual({})
  })

  it.each(['POST', 'PUT', 'PATCH', 'DELETE', 'post'])('sends credentials and the cross-site header on %s', (method) => {
    const sent = withSession({ method, headers: { 'Content-Type': 'application/json' } }, true)
    expect(sent.credentials).toBe('include')
    expect(sent.headers).toEqual({ 'Content-Type': 'application/json', [SESSION_CSRF_HEADER]: '1' })
  })

  it.each([undefined, 'GET', 'HEAD', 'OPTIONS'])('sends credentials but no cross-site header on %s', (method) => {
    const sent = withSession({ method, headers: new Headers({ Accept: 'application/json' }) }, true)
    expect(sent.credentials).toBe('include')
    expect(sent.headers).toEqual({ accept: 'application/json' })
  })

  it('never puts a token on a request', () => {
    const sent = withSession({ method: 'POST' }, true)
    expect(Object.keys(sent.headers as Record<string, string>)).toEqual([SESSION_CSRF_HEADER])
    expect(isUnsafeMethod('GET')).toBe(false)
  })
})

describe('forwardedSession (a route handler)', () => {
  const browser = (method: string, extra: Record<string, string> = {}) =>
    new Request('https://automatos.example.test/api/chat', {
      method,
      headers: {
        cookie: `${OTHER_COOKIES}; ${SESSION_COOKIE_NAME}=${SESSION}`,
        authorization: 'Bearer ' + 'x'.repeat(20),
        'x-api-key': 'k'.repeat(20),
        origin: 'https://automatos.example.test',
        [SESSION_CSRF_HEADER]: '1',
        'x-workspace-id': 'ws-1',
        ...extra,
      },
    })

  it('reads the session cookie out of the others', () => {
    expect(sessionCookieFrom(`${OTHER_COOKIES}; ${SESSION_COOKIE_NAME}=${SESSION}`)).toBe(SESSION)
    expect(sessionCookieFrom(OTHER_COOKIES)).toBeNull()
    expect(sessionCookieFrom(`${SESSION_COOKIE_NAME}=`)).toBeNull()
    expect(sessionCookieFrom(null)).toBeNull()
  })

  it('passes on the session alone, the workspace, the cross-site header and Origin for an unsafe call', () => {
    expect(forwardedSession(browser('POST'), 'POST')).toEqual({
      Cookie: sessionCookieHeader(SESSION),
      'X-Workspace-ID': 'ws-1',
      [SESSION_CSRF_HEADER]: '1',
      Origin: 'https://automatos.example.test',
    })
  })

  it('passes on only the session and the workspace for a GET', () => {
    expect(forwardedSession(browser('POST'), 'GET')).toEqual({ Cookie: sessionCookieHeader(SESSION), 'X-Workspace-ID': 'ws-1' })
  })

  it('passes on nothing when the browser sent no session', () => {
    expect(forwardedSession(browser('POST', { cookie: OTHER_COOKIES }), 'POST')).toEqual({})
  })

  it('keys a cache by a hash of the session, never the session', async () => {
    const key = await sessionKey(SESSION)
    expect(key).toMatch(/^[0-9a-f]{64}$/)
    expect(key).not.toContain(SESSION)
    expect(await sessionKey(SESSION)).toBe(key)
  })
})
