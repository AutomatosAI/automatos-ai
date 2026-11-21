/**
 * PRD-277 US-505 — the enterprise web app calls the API by relative address, so one image works on
 * any install's address; local and saas keep their build-time address.
 *
 *   - apiBaseUrl: '' in enterprise whatever the build or the page says; in local and saas the same
 *     answer, in the same order, as apiClient's constructor gave before (and the same warning when
 *     there is none).
 *   - apiClient, loaded as each edition's build loads it: enterprise sends `/api/…` on the page's
 *     own origin, with the session's credentials; saas sends to the build's API address.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { apiBaseUrl } from '@/lib/api-base-url'

const API = 'https://api.example.test'
type PageGlobals = { __NEXT_PUBLIC_API_URL__?: string; NEXT_PUBLIC_API_URL?: string }

beforeEach(() => {
  vi.spyOn(console, 'error').mockImplementation(() => {})
})

afterEach(() => {
  const page = window as unknown as PageGlobals
  delete page.__NEXT_PUBLIC_API_URL__
  delete page.NEXT_PUBLIC_API_URL
  vi.unstubAllEnvs()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  vi.resetModules()
})

describe('apiBaseUrl', () => {
  it('is relative in enterprise, whatever the build or the page names', () => {
    vi.stubEnv('NEXT_PUBLIC_API_URL', API)
    ;(window as unknown as PageGlobals).__NEXT_PUBLIC_API_URL__ = 'https://elsewhere.example.test'
    expect(apiBaseUrl(true)).toBe('')
    expect(console.error).not.toHaveBeenCalled()
  })

  it('is the build-time address outside enterprise, as before', () => {
    vi.stubEnv('NEXT_PUBLIC_API_URL', API)
    expect(apiBaseUrl(false)).toBe(API)
  })

  it('prefers a runtime-injected address, then the build, then the page fallback, as before', () => {
    const page = window as unknown as PageGlobals
    page.NEXT_PUBLIC_API_URL = 'https://fallback.example.test'
    vi.stubEnv('NEXT_PUBLIC_API_URL', '')
    expect(apiBaseUrl(false)).toBe('https://fallback.example.test')
    vi.stubEnv('NEXT_PUBLIC_API_URL', API)
    expect(apiBaseUrl(false)).toBe(API)
    page.__NEXT_PUBLIC_API_URL__ = 'https://injected.example.test'
    expect(apiBaseUrl(false)).toBe('https://injected.example.test')
  })

  it('warns, as before, when local or saas has no address in the browser', () => {
    vi.stubEnv('NEXT_PUBLIC_API_URL', '')
    expect(apiBaseUrl(false)).toBe('')
    expect(console.error).toHaveBeenCalledWith('❌ NEXT_PUBLIC_API_URL is not set! API calls will fail.')
  })
})

describe('apiClient, as each edition builds it', () => {
  async function clientFor(edition: string) {
    vi.stubEnv('NEXT_PUBLIC_AUTH_EDITION', edition)
    vi.stubEnv('NEXT_PUBLIC_API_URL', edition === 'enterprise' ? '' : API)
    const fetchMock = vi.fn(async (_url: string, _init?: RequestInit) => new Response('{}', { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
    const { apiClient } = await import('@/lib/api-client')
    return { apiClient, fetchMock }
  }

  it('calls /api/… on the page’s own origin in enterprise, with the session', async () => {
    const { apiClient, fetchMock } = await clientFor('enterprise')
    await apiClient.post('/api/workspaces/current/onboarding/reset', {})

    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('/api/workspaces/current/onboarding/reset')
    expect(init?.credentials).toBe('include')
    expect(apiClient.getBaseUrl()).toBe('')
  })

  it('calls the build-time API address in saas, as before', async () => {
    const { apiClient, fetchMock } = await clientFor('saas')
    await apiClient.get('/api/workspaces/current')

    expect(fetchMock.mock.calls[0][0]).toBe(`${API}/api/workspaces/current`)
    expect(apiClient.getBaseUrl()).toBe(API)
  })
})
