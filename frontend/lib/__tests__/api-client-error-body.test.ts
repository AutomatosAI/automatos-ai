/**
 * PRD-261 W-A2 opening commit (plan §2.2):
 *
 * 1. On a non-2xx response apiClient attaches the parsed JSON body to the error it throws as
 *    `body`. Step-up's 403 (orchestrator/core/auth/step_up.py) carries Clerk's `clerk_error` at
 *    the TOP level, and the step-up helper (lib/step-up) reads it from there. The messages and
 *    the `status` callers already read are unchanged.
 * 2. Every theme route of §2.2 is a named apiClient method with a literal path, so the
 *    route-contract check (scripts/check-route-contract.js, which reads only api-client.ts) sees it.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { apiClient, type ApiHttpError } from '@/lib/api-client'

// step_up.py step_up_body(10): Clerk's reverificationError at the top level, plus code and detail.
const STEP_UP_BODY = {
  clerk_error: {
    type: 'forbidden',
    reason: 'reverification-error',
    metadata: { reverification: { level: 'first_factor', afterMinutes: 10 } },
  },
  code: 'step_up_required',
  detail: "Confirm it's you to continue: this action needs a sign-in from the last 10 minutes.",
}

const JSON_HEADERS = { 'Content-Type': 'application/json' }

function jsonResponse(status: number, body: unknown, statusText = ''): Response {
  return new Response(JSON.stringify(body), { status, statusText, headers: JSON_HEADERS })
}

function stubFetch(response: Response) {
  const fetchMock = vi.fn(async () => response)
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

async function caught(promise: Promise<unknown>): Promise<ApiHttpError> {
  try {
    await promise
  } catch (error) {
    return error as ApiHttpError
  }
  throw new Error('expected the call to throw')
}

beforeEach(() => {
  vi.spyOn(console, 'error').mockImplementation(() => {})
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('apiClient error body', () => {
  it('keeps the status and the top-level clerk_error body of a step-up 403', async () => {
    stubFetch(jsonResponse(403, STEP_UP_BODY, 'Forbidden'))

    const error = await caught(apiClient.putThemeSettings({ membersMayChoose: false }))

    expect(error).toBeInstanceOf(Error)
    expect(error.status).toBe(403)
    expect(error.body).toEqual(STEP_UP_BODY)
    expect(error.message).toBe(STEP_UP_BODY.detail)
  })

  it('keeps the message of a string detail and adds the body beside the status', async () => {
    const body = { detail: 'This install is pinned to version 1.0.0. Unpin it in the theme settings first.' }
    stubFetch(jsonResponse(409, body, 'Conflict'))

    const error = await caught(apiClient.updateThemeInstall('install-1', 'package-2'))

    expect(error.message).toBe(body.detail)
    expect(error.status).toBe(409)
    expect(error.body).toEqual(body)
  })

  it('still names a structured detail by its JSON (step_up_unavailable nests under detail)', async () => {
    const body = { detail: { code: 'step_up_unavailable', message: 'keys cannot confirm it' } }
    stubFetch(jsonResponse(403, body, 'Forbidden'))

    const error = await caught(apiClient.setDefaultTheme(null))

    expect(error.message).toBe(JSON.stringify(body.detail))
    expect(error.status).toBe(403)
    expect(error.body).toEqual(body)
  })

  it('falls back to the status text and attaches no body when the response is not JSON', async () => {
    stubFetch(new Response('<html>bad gateway</html>', { status: 502, statusText: 'Bad Gateway' }))

    const error = await caught(apiClient.getInstalledThemes())

    expect(error.message).toBe('Bad Gateway')
    expect(error.status).toBe(502)
    expect('body' in error).toBe(false)
  })

  it('keeps the 401 message (and no status) unchanged, with the body attached', async () => {
    stubFetch(jsonResponse(401, { detail: 'Not authenticated' }, 'Unauthorized'))

    const error = await caught(apiClient.getThemeSettings())

    expect(error.message).toBe(
      'HTTP 401: Unauthorized (missing/invalid Clerk token). ' +
        'Make sure you are signed in and the API client is configured with Clerk.',
    )
    expect(error.status).toBeUndefined()
    expect(error.body).toEqual({ detail: 'Not authenticated' })
  })

  it('still answers a 2xx with its JSON and a 204 with null', async () => {
    stubFetch(jsonResponse(200, { installs: [] }))
    await expect(apiClient.getInstalledThemes()).resolves.toEqual({ installs: [] })

    stubFetch(new Response(null, { status: 204 }))
    await expect(apiClient.getThemeSettings()).resolves.toBeNull()
  })
})

describe('apiClient theme methods (plan §2.2)', () => {
  function spyRequest() {
    return vi.spyOn(apiClient, 'request').mockResolvedValue({} as never)
  }

  it('reads the active theme, the installs and the settings with GET', async () => {
    const spy = spyRequest()
    await apiClient.getActiveTheme()
    await apiClient.getInstalledThemes()
    await apiClient.getThemeSettings()
    expect(spy.mock.calls).toEqual([['/api/themes/active'], ['/api/themes/installed'], ['/api/themes/settings']])
  })

  it('PUTs the choice, the default and the settings as JSON', async () => {
    const spy = spyRequest()
    await apiClient.putThemeChoice('builtin:play')
    await apiClient.setDefaultTheme('install-1')
    await apiClient.putThemeSettings({ privateUploads: 'off', pins: [{ installId: 'install-1', version: null }] })
    expect(spy.mock.calls).toEqual([
      ['/api/themes/choice', { method: 'PUT', body: JSON.stringify({ choice: 'builtin:play' }) }],
      ['/api/themes/default', { method: 'PUT', body: JSON.stringify({ installId: 'install-1' }) }],
      [
        '/api/themes/settings',
        { method: 'PUT', body: JSON.stringify({ privateUploads: 'off', pins: [{ installId: 'install-1', version: null }] }) },
      ],
    ])
  })

  it('uploads the .autotheme as multipart form data', async () => {
    const spy = spyRequest()
    const file = new File(['zip'], 'hello.autotheme')
    await apiClient.importTheme(file)
    const [path, init] = spy.mock.calls[0] as [string, RequestInit]
    expect(path).toBe('/api/themes/import')
    expect(init.method).toBe('POST')
    expect(init.body).toBeInstanceOf(FormData)
    expect(((init.body as FormData).get('file') as File).name).toBe('hello.autotheme')
  })

  it('acts on one install: opt-in, compare, update, rollback, uninstall', async () => {
    const spy = spyRequest()
    await apiClient.optInTheme('install-1', 'package-1')
    await apiClient.compareThemeUpdate('install-1', 'package-2')
    await apiClient.updateThemeInstall('install-1', 'package-2')
    await apiClient.rollbackTheme('install-1')
    await apiClient.uninstallTheme('install-1')
    expect(spy.mock.calls).toEqual([
      ['/api/themes/installs/install-1/opt-in', { method: 'POST', body: JSON.stringify({ packageId: 'package-1' }) }],
      ['/api/themes/installs/install-1/compare/package-2'],
      ['/api/themes/installs/install-1/update', { method: 'POST', body: JSON.stringify({ packageId: 'package-2' }) }],
      ['/api/themes/installs/install-1/rollback', { method: 'POST' }],
      ['/api/themes/installs/install-1', { method: 'DELETE' }],
    ])
  })

  it('keeps a theme id as path segments for the looks (the route takes {theme_id:path})', async () => {
    const spy = spyRequest()
    const looks = { looks: [{ name: 'Dusk' }], active: null }
    await apiClient.getThemeLooks('acme/hello world')
    await apiClient.putThemeLooks('acme/hello world', looks)
    expect(spy.mock.calls).toEqual([
      ['/api/themes/looks/acme/hello%20world'],
      ['/api/themes/looks/acme/hello%20world', { method: 'PUT', body: JSON.stringify(looks) }],
    ])
  })

  it('encodes an install id that is not a plain uuid', async () => {
    const spy = spyRequest()
    await apiClient.rollbackTheme('../settings')
    expect(spy).toHaveBeenCalledWith('/api/themes/installs/..%2Fsettings/rollback', { method: 'POST' })
  })
})
