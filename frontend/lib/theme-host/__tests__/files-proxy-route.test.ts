// @vitest-environment node
// A route handler runs on Node, so its Headers are Node's: jsdom's Headers drops `range` on every
// set() or append() (its privileged no-CORS cleanup runs under every guard), which no server does.
/**
 * PRD-261 A-08b — GET /api/themes/files/<digest>/<path>, the same-origin proxy a native import()
 * and a <video src> reach an installed theme's files through. SaaS: the Clerk session is read in
 * the handler (signed out → 401 from here, never Clerk's 404) and its token goes to the backend
 * as Bearer; local: no token. Range and If-None-Match go forward; 200/206/304 come back with
 * their content and caching headers and never a Set-Cookie; a service-worker registration and
 * any path that is not a digest and plain file names are refused before the backend is asked.
 */
import { afterEach, describe, expect, it, vi, type Mock } from 'vitest'

const clerk = vi.hoisted(() => ({ auth: vi.fn() }))

const DIGEST = 'a'.repeat(64)
const CLIP = ['assets', 'mascot', 'idle.mp4']
const ENTRY = ['dist', 'entry.js']
const CSP = "default-src 'none'; sandbox"
/** A paired browser's device token (PRD-266: `amd_` then token_urlsafe(32)). */
const DEVICE_TOKEN = `amd_${'d'.repeat(43)}`

type BackendCall = (url: string, init?: RequestInit) => Promise<Response>
type BackendFetch = Mock<BackendCall>
type Get = (request: Request, context: { params: Promise<{ path: string[] }> }) => Promise<Response>

/** The route module under one edition (lib/auth-edition reads the edition once, at import). */
async function routeGet(saas: boolean): Promise<Get> {
  vi.resetModules()
  vi.doMock('@/lib/auth-edition', () => ({ isSaaS: saas }))
  vi.doMock('@clerk/nextjs/server', () => ({ auth: clerk.auth }))
  return (await import('@/app/api/themes/files/[...path]/route')).GET as Get
}

function call(get: Get, path: string[], headers: Record<string, string> = {}): Promise<Response> {
  const request = new Request(`http://app.test/api/themes/files/${path.join('/')}`, { headers })
  return get(request, { params: Promise.resolve({ path }) })
}

/** What the backend answers, with any headers (a plain Headers keeps Set-Cookie, as a real answer can carry it). */
function backendAnswer(status: number, headers: Record<string, string> = {}, body: string | null = null): Response {
  const stream = body === null ? null : new Response(body).body
  return { status, headers: new Headers(headers), body: stream } as unknown as Response
}

function stubBackend(answer: Response | Error): BackendFetch {
  const backend = vi.fn<BackendCall>(async () => {
    if (answer instanceof Error) throw answer
    return answer
  })
  vi.stubGlobal('fetch', backend)
  return backend
}

function signedIn(token: string | null = 'session-token'): void {
  clerk.auth.mockResolvedValue({ userId: 'user_1', getToken: vi.fn(async () => token) })
}

function sentHeaders(backend: BackendFetch): Headers {
  return backend.mock.calls[0][1]?.headers as Headers
}

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  clerk.auth.mockReset()
})

describe('PRD-261 A-08b — SaaS: the handler authenticates', () => {
  it('answers 401 itself when signed out, and never asks the backend', async () => {
    clerk.auth.mockResolvedValue({ userId: null, getToken: vi.fn(async () => null) })
    const backend = stubBackend(backendAnswer(200))
    const response = await call(await routeGet(true), [DIGEST, ...ENTRY])
    expect(response.status).toBe(401)
    expect(response.headers.get('x-content-type-options')).toBe('nosniff')
    expect(backend).not.toHaveBeenCalled()
  })

  it('answers 401 when the session has no token, or cannot be read', async () => {
    const backend = stubBackend(backendAnswer(200))
    signedIn(null)
    expect((await call(await routeGet(true), [DIGEST, ...ENTRY])).status).toBe(401)
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    clerk.auth.mockRejectedValue(new Error('clerkMiddleware did not run'))
    expect((await call(await routeGet(true), [DIGEST, ...ENTRY])).status).toBe(401)
    expect(backend).not.toHaveBeenCalled()
  })

  it("sends the session's token as Bearer to the backend's files route, with the Range", async () => {
    signedIn()
    const backend = stubBackend(backendAnswer(206, { 'Content-Range': 'bytes 0-99/1000', 'Content-Type': 'video/mp4' }, 'x'))
    await call(await routeGet(true), [DIGEST, ...CLIP], { Range: 'bytes=0-99', Cookie: '__session=abc' })
    expect(backend).toHaveBeenCalledTimes(1)
    expect(String(backend.mock.calls[0][0])).toMatch(new RegExp(`/api/themes/files/${DIGEST}/assets/mascot/idle\\.mp4$`))
    const sent = sentHeaders(backend)
    expect(sent.get('authorization')).toBe('Bearer session-token')
    expect(sent.get('range')).toBe('bytes=0-99')
    expect(sent.get('cookie')).toBeNull()
  })
})

describe('PRD-261 A-08b — local edition', () => {
  it('reads no session and sends no token', async () => {
    const backend = stubBackend(backendAnswer(200, { 'Content-Type': 'text/javascript' }, 'export default {}'))
    const response = await call(await routeGet(false), [DIGEST, ...ENTRY])
    expect(response.status).toBe(200)
    expect(clerk.auth).not.toHaveBeenCalled()
    expect(sentHeaders(backend).get('authorization')).toBeNull()
  })

  it("PRD-266 K8: sends a paired browser's device cookie as Bearer, and nothing for any other cookie", async () => {
    const get = await routeGet(false)
    const paired = stubBackend(backendAnswer(200, { 'Content-Type': 'text/javascript' }, 'export default {}'))
    await call(get, [DIGEST, ...ENTRY], { cookie: `theme=x; automatos_device_token=${DEVICE_TOKEN}; other=1` })
    expect(sentHeaders(paired).get('authorization')).toBe(`Bearer ${DEVICE_TOKEN}`)
    expect(clerk.auth).not.toHaveBeenCalled()
    for (const cookie of ['automatos_device_token=not-a-device-token', 'automatos_device_token=', `session=${DEVICE_TOKEN}`]) {
      const backend = stubBackend(backendAnswer(200, { 'Content-Type': 'text/javascript' }, 'export default {}'))
      await call(get, [DIGEST, ...ENTRY], { cookie })
      expect(sentHeaders(backend).get('authorization')).toBeNull()
    }
  })
})

describe('PRD-266 K8 — SaaS never reads the device cookie', () => {
  it("sends the Clerk session's token, whatever device cookie the request carries", async () => {
    signedIn('session-token')
    const backend = stubBackend(backendAnswer(200, { 'Content-Type': 'text/javascript' }, 'export default {}'))
    await call(await routeGet(true), [DIGEST, ...ENTRY], { cookie: `automatos_device_token=${DEVICE_TOKEN}` })
    expect(sentHeaders(backend).get('authorization')).toBe('Bearer session-token')
  })
})

describe("PRD-261 A-08b — the backend's answer", () => {
  it('passes a 206 with its body, Content-Range, Accept-Ranges, Content-Type and Cache-Control, and drops Set-Cookie', async () => {
    stubBackend(backendAnswer(206, {
      'Content-Range': 'bytes 0-3/10',
      'Accept-Ranges': 'bytes',
      'Content-Type': 'video/mp4',
      'Cache-Control': 'private, max-age=3600, immutable',
      'Content-Length': '4',
      ETag: '"abc"',
      'Set-Cookie': 'tracker=1',
      Server: 'uvicorn',
    }, 'abcd'))
    const response = await call(await routeGet(false), [DIGEST, ...CLIP], { Range: 'bytes=0-3' })
    expect(response.status).toBe(206)
    expect(await response.text()).toBe('abcd')
    expect(response.headers.get('content-range')).toBe('bytes 0-3/10')
    expect(response.headers.get('accept-ranges')).toBe('bytes')
    expect(response.headers.get('content-type')).toBe('video/mp4')
    expect(response.headers.get('cache-control')).toBe('private, max-age=3600, immutable')
    expect(response.headers.get('content-length')).toBe('4')
    expect(response.headers.get('etag')).toBe('"abc"')
    expect(response.headers.get('set-cookie')).toBeNull()
    expect(response.headers.get('server')).toBeNull()
    expect(response.headers.get('x-content-type-options')).toBe('nosniff')
    expect(response.headers.get('content-security-policy')).toBe(CSP)
  })

  it('forwards If-None-Match and passes a 304 with no body', async () => {
    const backend = stubBackend(backendAnswer(304, { ETag: '"abc"', 'Cache-Control': 'private, max-age=3600, immutable' }))
    const response = await call(await routeGet(false), [DIGEST, ...ENTRY], { 'If-None-Match': '"abc"' })
    expect(sentHeaders(backend).get('if-none-match')).toBe('"abc"')
    expect(response.status).toBe(304)
    expect(response.body).toBeNull()
    expect(response.headers.get('etag')).toBe('"abc"')
  })

  it("passes a 404 with its own text, never the backend's body", async () => {
    stubBackend(backendAnswer(404, { 'Content-Type': 'application/json' }, '{"detail":"internal detail"}'))
    const response = await call(await routeGet(false), [DIGEST, ...ENTRY])
    expect(response.status).toBe(404)
    expect(await response.text()).not.toContain('internal detail')
    expect(response.headers.get('cache-control')).toBe('no-store')
  })

  it('answers 502 for a backend error it does not pass, or a backend it cannot reach', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    stubBackend(backendAnswer(500, {}, 'Traceback'))
    const failed = await call(await routeGet(false), [DIGEST, ...ENTRY])
    expect(failed.status).toBe(502)
    expect(await failed.text()).not.toContain('Traceback')
    stubBackend(new TypeError('fetch failed'))
    expect((await call(await routeGet(false), [DIGEST, ...ENTRY])).status).toBe(502)
  })
})

describe('PRD-261 A-08b — refused before the backend is asked', () => {
  it('refuses a service-worker registration with 403', async () => {
    signedIn()
    const backend = stubBackend(backendAnswer(200))
    const response = await call(await routeGet(true), [DIGEST, ...ENTRY], { 'Service-Worker': 'script' })
    expect(response.status).toBe(403)
    expect(backend).not.toHaveBeenCalled()
    expect(clerk.auth).not.toHaveBeenCalled()
  })

  const deep = [DIGEST, 'source', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'x.ts']
  it.each([
    ['no file', [DIGEST]],
    ['a digest that is not sha256 hex', ['not-a-digest', ...ENTRY]],
    ['an upper-case digest', [DIGEST.toUpperCase(), ...ENTRY]],
    ['a parent segment', [DIGEST, '..', '..', 'health']],
    ['a current segment', [DIGEST, '.', 'theme.json']],
    ['a dotfile', [DIGEST, '.env']],
    ['a backslash', [DIGEST, 'dist\\entry.js']],
    ['a slash inside a segment', [DIGEST, 'dist/../../health']],
    ['an encoded segment', [DIGEST, '%2e%2e', 'health']],
    ['a space', [DIGEST, 'dist', 'en try.js']],
    ['a path deeper than a package holds', deep],
  ])('answers 404 for %s', async (_case, path) => {
    const backend = stubBackend(backendAnswer(200))
    expect((await call(await routeGet(false), path)).status).toBe(404)
    expect(backend).not.toHaveBeenCalled()
  })

  it('forwards the deepest path a package holds', async () => {
    const backend = stubBackend(backendAnswer(200, {}, 'x'))
    const deepest = [DIGEST, 'source', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'x.ts']
    expect((await call(await routeGet(false), deepest)).status).toBe(200)
    expect(backend).toHaveBeenCalledTimes(1)
  })
})
