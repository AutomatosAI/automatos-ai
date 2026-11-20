// @vitest-environment node
/**
 * PRD-277 US-503 — the web app's own server code calls the API with the browser's session cookie
 * in enterprise, and with nothing else that identifies anyone:
 *
 *   - the chat route (POST, PATCH) and the workflow stream route (POST, GET): the session cookie
 *     alone, the workspace, and for an unsafe call the cross-site header and Origin; no
 *     Authorization and no API key, even when the server holds one. A 401 hands back the API's
 *     own body, so the chat can say the session ended.
 *   - the theme files route: the session cookie, never a bearer; no cookie → 401 from the handler.
 *   - the theme boot (boot-server.ts): the session cookie, cached by its hash.
 *
 * Each module is loaded as the enterprise build loads it (lib/edition-name mocked to enterprise).
 */
import { afterEach, describe, expect, it, vi, type Mock } from 'vitest'

const COOKIE_NAME = '__Host-automatos_session'
const SESSION = 'session' + '-' + 'w'.repeat(32)
const ORIGIN = 'https://automatos.example.test'
const WORKSPACE = '00000000-0000-0000-0000-0000000277f5'
const EXECUTION = '11111111-2222-4333-8444-555555555555'
const DIGEST = 'b'.repeat(64)

type BackendFetch = Mock<(url: string, init?: RequestInit) => Promise<Response>>

function asEnterprise(): void {
  vi.resetModules()
  vi.doMock('@/lib/edition-name', () => ({ isEnterprise: true, resolveEdition: () => 'enterprise' }))
  vi.doMock('@/lib/auth-edition', () => ({ isSaaS: false, isLocal: false, isEnterprise: true }))
  vi.doMock('@clerk/nextjs/server', () => ({ auth: vi.fn(() => Promise.reject(new Error('Clerk was asked'))) }))
  vi.stubEnv('API_KEY', 'server' + '-' + 'k'.repeat(16))
}

function stubBackend(answer: () => Response): BackendFetch {
  const backend = vi.fn(async () => answer()) as BackendFetch
  vi.stubGlobal('fetch', backend)
  return backend
}

function browser(url: string, method: string, body?: string): Request {
  return new Request(url, {
    method,
    body,
    headers: {
      cookie: `theme=dark; ${COOKIE_NAME}=${SESSION}`,
      authorization: 'Bearer ' + 'b'.repeat(20),
      'x-api-key': 'k'.repeat(20),
      origin: ORIGIN,
      'X-Automatos-CSRF': '1',
      'x-workspace-id': WORKSPACE,
      'content-type': 'application/json',
    },
  })
}

function sent(backend: BackendFetch): Record<string, string> {
  const headers = backend.mock.calls[0][1]?.headers
  return headers instanceof Headers ? Object.fromEntries(headers.entries()) : { ...(headers as Record<string, string>) }
}

function lower(headers: Record<string, string>): Record<string, string> {
  return Object.fromEntries(Object.entries(headers).map(([name, value]) => [name.toLowerCase(), value]))
}

afterEach(() => {
  vi.unstubAllGlobals()
  vi.unstubAllEnvs()
  vi.restoreAllMocks()
  vi.resetModules()
})

describe('the chat route', () => {
  it('POSTs with the session cookie alone, the cross-site header and Origin, and no key or bearer', async () => {
    asEnterprise()
    const backend = stubBackend(() => new Response('ok', { status: 200 }))
    const { POST } = await import('@/app/api/chat/route')
    await POST(browser(`${ORIGIN}/api/chat`, 'POST', '{"id":""}') as never)

    const headers = lower(sent(backend))
    expect(headers.cookie).toBe(`${COOKIE_NAME}=${SESSION}`)
    expect(headers['x-automatos-csrf']).toBe('1')
    expect(headers.origin).toBe(ORIGIN)
    expect(headers['x-workspace-id']).toBe(WORKSPACE)
    expect(headers).not.toHaveProperty('authorization')
    expect(headers).not.toHaveProperty('x-api-key')
  })

  it('does not warn about a missing API key or Authorization: enterprise sends neither', async () => {
    asEnterprise()
    vi.stubEnv('API_KEY', '')
    stubBackend(() => new Response('ok', { status: 200 }))
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const { POST } = await import('@/app/api/chat/route')
    const request = new Request(`${ORIGIN}/api/chat`, {
      method: 'POST',
      body: '{"id":""}',
      headers: { cookie: `${COOKIE_NAME}=${SESSION}`, origin: ORIGIN, 'X-Automatos-CSRF': '1', 'content-type': 'application/json' },
    })
    await POST(request as never)

    expect(warn.mock.calls.filter((call) => String(call[0]).includes('No API_KEY'))).toEqual([])
  })

  it("hands a 401 back with the API's own body", async () => {
    asEnterprise()
    const ended = JSON.stringify({ detail: { code: 'session_ended', reason: 'expired', message: 'Sign in again.' } })
    stubBackend(() => new Response(ended, { status: 401 }))
    vi.spyOn(console, 'error').mockImplementation(() => {})
    const { PATCH } = await import('@/app/api/chat/route')
    const answer = await PATCH(browser(`${ORIGIN}/api/chat`, 'PATCH', '{}') as never)

    expect(answer.status).toBe(401)
    expect(await answer.json()).toEqual(JSON.parse(ended))
  })
})

describe('the workflow stream route', () => {
  it.each(['POST', 'GET'])('%s streams with the session cookie and the workspace, and no key or bearer', async (method) => {
    asEnterprise()
    const backend = stubBackend(() => new Response('0:"x"\n', { status: 200 }))
    const { GET, POST } = await import('@/app/api/workflows/stream/route')
    const request =
      method === 'POST'
        ? browser(`${ORIGIN}/api/workflows/stream`, 'POST', JSON.stringify({ executionId: EXECUTION }))
        : browser(`${ORIGIN}/api/workflows/stream?executionId=${EXECUTION}`, 'GET')
    await (method === 'POST' ? POST : GET)(request as never)

    const headers = lower(sent(backend))
    expect(headers.cookie).toBe(`${COOKIE_NAME}=${SESSION}`)
    expect(headers['x-workspace-id']).toBe(WORKSPACE)
    expect(headers).not.toHaveProperty('authorization')
    expect(headers).not.toHaveProperty('x-api-key')
  })
})

describe('the theme files route', () => {
  type Get = (request: Request, context: { params: Promise<{ path: string[] }> }) => Promise<Response>
  const path = [DIGEST, 'dist', 'entry.js']

  it('forwards the session cookie, never a bearer', async () => {
    asEnterprise()
    const backend = stubBackend(() => new Response('export {}', { status: 200, headers: { 'content-type': 'text/javascript' } }))
    const { GET } = (await import('@/app/api/themes/files/[...path]/route')) as { GET: Get }
    const answer = await GET(browser(`${ORIGIN}/api/themes/files/${path.join('/')}`, 'GET'), { params: Promise.resolve({ path }) })

    expect(answer.status).toBe(200)
    const headers = sent(backend)
    expect(headers.cookie).toBe(`${COOKIE_NAME}=${SESSION}`)
    expect(headers).not.toHaveProperty('authorization')
  })

  it('answers 401 itself with no session cookie, and never asks the API', async () => {
    asEnterprise()
    const backend = stubBackend(() => new Response('', { status: 200 }))
    const { GET } = (await import('@/app/api/themes/files/[...path]/route')) as { GET: Get }
    const request = new Request(`${ORIGIN}/api/themes/files/${path.join('/')}`, { headers: { cookie: 'theme=dark' } })
    const answer = await GET(request, { params: Promise.resolve({ path }) })

    expect(answer.status).toBe(401)
    expect(backend).not.toHaveBeenCalled()
  })
})

describe("the theme boot's caller", () => {
  it('is the session cookie, keyed by its hash; none without one', async () => {
    asEnterprise()
    const { enterpriseBootCaller, fetchInstalledBoot } = await import('@/lib/theme-host/boot-server')
    const caller = await enterpriseBootCaller(`theme=dark; ${COOKIE_NAME}=${SESSION}`)

    expect(caller?.user).toMatch(/^session:[0-9a-f]{64}$/)
    expect(caller?.user).not.toContain(SESSION)
    const credential = await caller!.credential()
    expect(credential).toEqual({ ok: true, authorization: null, cookie: `${COOKIE_NAME}=${SESSION}` })
    expect(await enterpriseBootCaller('theme=dark')).toBeNull()

    const fetchBoot = vi.fn(async () => new Response('', { status: 401 }))
    await fetchInstalledBoot('install-1', null, { fetchBoot, timeoutMs: 1000 }, `${COOKIE_NAME}=${SESSION}`)
    expect((fetchBoot.mock.calls[0] as unknown as [string, RequestInit])[1].headers).toEqual({
      Accept: 'application/json',
      Cookie: `${COOKIE_NAME}=${SESSION}`,
    })
  })
})
