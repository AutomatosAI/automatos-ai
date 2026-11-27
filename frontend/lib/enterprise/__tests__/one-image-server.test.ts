// @vitest-environment node
/**
 * PRD-277 US-505 — the web app's own server code, in an image that carries no API address.
 *
 *   - The workflow stream route's GET reaches the API at BACKEND_INTERNAL_URL in enterprise, as its
 *     POST does (the browser-facing address is relative there, so there is nothing else to use);
 *     in local and saas it reads NEXT_PUBLIC_API_URL, as before.
 *   - /automatos-host.json tells the theme CLI where the API is: in enterprise the origin the request
 *     came in on (the API answers on the web app's own host), from its Host and X-Forwarded-Proto,
 *     never X-Forwarded-Host, with `Vary: Host, X-Forwarded-Proto`; a Host that is not a plain host
 *     name with a port from 1 to 65535 gets the 500 sentence, never an address built from it.
 *
 * Each module is loaded as each edition's build loads it (lib/edition-name mocked).
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

const EXECUTION = '42'  // execution ids are integers (#1664)
const INTERNAL = 'http://api.internal.test:8000'
const PUBLIC_API = 'https://api.example.test'

/** The CLI's request as the handler reads it: its headers (Host set as a proxy passes it on). */
function cliRequest(headers: Record<string, string>): Request {
  return { headers: new Headers(headers) } as unknown as Request
}

function asEdition(enterprise: boolean): void {
  vi.resetModules()
  vi.doMock('@/lib/edition-name', () => ({ isEnterprise: enterprise, resolveEdition: () => (enterprise ? 'enterprise' : 'saas') }))
}

afterEach(() => {
  vi.unstubAllGlobals()
  vi.unstubAllEnvs()
  vi.doUnmock('@/lib/edition-name')
  vi.restoreAllMocks()
  vi.resetModules()
})

describe('the workflow stream route’s GET', () => {
  async function streamedTo(enterprise: boolean): Promise<string> {
    asEdition(enterprise)
    vi.stubEnv('BACKEND_INTERNAL_URL', INTERNAL)
    vi.stubEnv('NEXT_PUBLIC_API_URL', enterprise ? '' : PUBLIC_API)
    const backend = vi.fn(async (_url: string, _init?: RequestInit) => new Response('', { status: 200 }))
    vi.stubGlobal('fetch', backend)
    const { GET } = await import('@/app/api/workflows/stream/route')
    await GET(new Request(`https://automatos.example.test/api/workflows/stream?executionId=${EXECUTION}`) as never)
    return backend.mock.calls[0][0]
  }

  it('calls BACKEND_INTERNAL_URL in enterprise', async () => {
    expect(await streamedTo(true)).toBe(`${INTERNAL}/api/workflows/executions/${EXECUTION}/stream/aisdk`)
  })

  it('calls NEXT_PUBLIC_API_URL outside enterprise, as before', async () => {
    expect(await streamedTo(false)).toBe(`${PUBLIC_API}/api/workflows/executions/${EXECUTION}/stream/aisdk`)
  })
})

describe('/automatos-host.json in enterprise', () => {
  async function hostFile(headers: Record<string, string>): Promise<Response> {
    asEdition(true)
    vi.stubEnv('NEXT_PUBLIC_API_URL', '')
    const { GET } = await import('@/app/automatos-host.json/route')
    return GET(cliRequest(headers))
  }

  it('answers the origin the request came in on', async () => {
    const answer = await hostFile({ host: 'automatos.example.test', 'x-forwarded-proto': 'https' })
    expect(answer.status).toBe(200)
    expect(answer.headers.get('cache-control')).toBe('no-store')
    expect(answer.headers.get('vary')).toBe('Host, X-Forwarded-Proto')
    expect(await answer.json()).toStrictEqual({ schema: 1, apiUrl: 'https://automatos.example.test' })
  })

  it('never reads X-Forwarded-Host', async () => {
    const answer = await hostFile({ host: 'automatos.example.test', 'x-forwarded-host': 'evil.example.test' })
    expect(await answer.json()).toStrictEqual({ schema: 1, apiUrl: 'https://automatos.example.test' })
  })

  it.each([
    ['an empty label', 'automatos..example.test'],
    ['a trailing dot', 'automatos.example.test.'],
    ['port zero', 'automatos.example.test:0'],
    ['a port above 65535', 'automatos.example.test:65536'],
    ['no host', ''],
  ])('answers the 500 sentence, saying what it varies by, for a Host with %s', async (_why, host) => {
    const answer = await hostFile({ host })
    expect(answer.status).toBe(500)
    expect(answer.headers.get('vary')).toBe('Host, X-Forwarded-Proto')
    expect(answer.headers.get('cache-control')).toBe('no-store')
  })

  it('takes the highest port', async () => {
    const answer = await hostFile({ host: 'automatos.example.test:65535' })
    expect(await answer.json()).toStrictEqual({ schema: 1, apiUrl: 'https://automatos.example.test:65535' })
  })

  it('keeps the port and the scheme the proxy names', async () => {
    const answer = await hostFile({ host: 'automatos.example.test:8443', 'x-forwarded-proto': 'http' })
    expect(await answer.json()).toStrictEqual({ schema: 1, apiUrl: 'http://automatos.example.test:8443' })
  })

  it('answers the 500 sentence, never an address, for a Host that is not a plain host name', async () => {
    const answer = await hostFile({ host: 'evil.example.test/path' })
    expect(answer.status).toBe(500)
    const text = await answer.text()
    expect(text).not.toContain('evil.example.test')
    expect(text).not.toMatch(/[{}<>\n]/)
  })

  it('answers the build’s address outside enterprise, ignoring the request, as before', async () => {
    asEdition(false)
    vi.stubEnv('NEXT_PUBLIC_API_URL', PUBLIC_API)
    const { GET } = await import('@/app/automatos-host.json/route')
    const answer = GET(cliRequest({ host: 'other.example.test' }))
    expect(answer.headers.get('vary')).toBeNull()
    expect(await answer.json()).toStrictEqual({ schema: 1, apiUrl: PUBLIC_API })
  })
})
