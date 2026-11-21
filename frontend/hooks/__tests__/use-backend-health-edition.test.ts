// @vitest-environment node
/**
 * PRD-277 US-505 — what the connection indicator asks, by edition.
 *
 *   - local and saas: GET /health through apiClient, exactly as before (path and options).
 *   - enterprise: `/health` is not exposed on the install's host, so the indicator asks the public
 *     sign-in page route `/api/v1/auth/sign-in-page` (api/auth_web.py) with a
 *     plain fetch, never `/health`, never the discovery document and never apiClient. Any HTTP
 *     answer is reachable (null, no body); a network failure, a timeout, or a gateway's 502, 503 or
 *     504 is not.
 *
 * Each edition loads the hook as its build loads it (NEXT_PUBLIC_AUTH_EDITION read when
 * lib/edition-name.ts loads).
 */
import { afterEach, describe, expect, it, vi, type Mock } from 'vitest'

import { API_REACHABLE_PATH, fetchApiReachable } from '@/lib/enterprise/api-reachable'

const TIMEOUT_MS = 5_000

type FetchMock = Mock<(url: string, init?: RequestInit) => Promise<Response>>

function answering(status: number): FetchMock {
  const fetchMock: FetchMock = vi.fn(async (_url: string, _init?: RequestInit) => new Response(null, { status }))
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

afterEach(() => {
  vi.unstubAllGlobals()
  vi.unstubAllEnvs()
  vi.doUnmock('@/lib/api-client')
  vi.restoreAllMocks()
  vi.resetModules()
})

describe('fetchApiReachable', () => {
  it('asks the sign-in page route, public, with no credentials and no cache', async () => {
    const fetchMock = answering(200)
    await expect(fetchApiReachable(TIMEOUT_MS)).resolves.toBeNull()
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('/api/v1/auth/sign-in-page')
    expect(url).toBe(API_REACHABLE_PATH)
    expect(url).not.toBe('/.well-known/automatos.json')
    expect(init?.credentials).toBe('omit')
    expect(init?.cache).toBe('no-store')
    expect(init?.signal).toBeInstanceOf(AbortSignal)
  })

  it.each([200, 301, 401, 403, 404, 500])('reads an answer of %i as reachable', async (status) => {
    answering(status)
    await expect(fetchApiReachable(TIMEOUT_MS)).resolves.toBeNull()
  })

  it.each([502, 503, 504])('reads a gateway’s %i as not reachable', async (status) => {
    answering(status)
    await expect(fetchApiReachable(TIMEOUT_MS)).rejects.toThrow(String(status))
  })

  it('reads a network failure as not reachable', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => Promise.reject(new TypeError('Failed to fetch'))))
    await expect(fetchApiReachable(TIMEOUT_MS)).rejects.toThrow('Failed to fetch')
  })
})

describe('fetchBackendHealth, as each edition builds it', () => {
  async function healthIn(edition: string) {
    vi.resetModules()
    const request = vi.fn(async () => ({ status: 'healthy' }))
    vi.doMock('@/lib/api-client', () => ({ apiClient: { request, getBaseUrl: () => '' } }))
    vi.stubEnv('NEXT_PUBLIC_AUTH_EDITION', edition)
    const fetchMock = answering(200)
    const { fetchBackendHealth } = await import('@/hooks/use-backend-health')
    const answer = await fetchBackendHealth()
    return { answer, request, fetchMock }
  }

  it.each(['local', 'saas'])('asks GET /health through apiClient in %s, as before', async (edition) => {
    const { answer, request, fetchMock } = await healthIn(edition)
    expect(request).toHaveBeenCalledTimes(1)
    const [endpoint, options] = request.mock.calls[0] as unknown as [string, RequestInit]
    expect(endpoint).toBe('/health')
    expect(Object.keys(options)).toEqual(['signal'])
    expect(options.signal).toBeInstanceOf(AbortSignal)
    expect(answer).toEqual({ status: 'healthy' })
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('asks the sign-in page route in enterprise, never /health and never apiClient, and has no body', async () => {
    const { answer, request, fetchMock } = await healthIn('enterprise')
    expect(request).not.toHaveBeenCalled()
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(fetchMock.mock.calls[0][0]).toBe(API_REACHABLE_PATH)
    expect(answer).toBeNull()
  })
})
