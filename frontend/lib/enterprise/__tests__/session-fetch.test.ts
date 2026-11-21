// @vitest-environment node
/**
 * PRD-277 US-502 — the session on a request made outside apiClient (lib/enterprise/session-fetch.ts).
 *
 *   - The page's own origin: an unsafe method gets the cross-site header, a path under /api the
 *     stored workspace, both with credentials; a header the caller set is kept as it is.
 *   - A 401 from the page's own API is reported as apiClient reports one.
 *   - Neither header ever reaches another origin: an absolute address to another host, a
 *     protocol-relative address, and a Request object for another host all pass through untouched.
 *   - A safe method to a path outside /api passes through untouched; a Request is never changed.
 *   - Never installed outside enterprise; taken off again by the function install returns.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

import { onUnauthorized } from '@/lib/api-unauthorized'
import { installSessionFetch, sessionFetch, type FetchHost } from '@/lib/enterprise/session-fetch'
import { SESSION_CSRF_HEADER } from '@/lib/enterprise/session-transport'

const PAGE = 'https://automatos.example.test'
const WORKSPACE = '00000000-0000-0000-0000-0000000000aa'

function host(): FetchHost & { readonly calls: Array<{ input: RequestInfo | URL; init?: RequestInit }> } {
  const calls: Array<{ input: RequestInfo | URL; init?: RequestInit }> = []
  const original = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    calls.push({ input, init })
    return new Response('{}', { status: 200 })
  })
  return {
    calls,
    fetch: original as unknown as typeof fetch,
    location: { href: `${PAGE}/knowledge?tab=docs`, origin: PAGE },
    localStorage: { getItem: (key: string) => (key === 'last_active_workspace' ? WORKSPACE : null) },
  }
}

function sentHeaders(call: { input: RequestInfo | URL; init?: RequestInit }): Headers {
  return call.input instanceof Request ? call.input.headers : new Headers(call.init?.headers)
}

afterEach(() => vi.restoreAllMocks())

describe('the page’s own requests', () => {
  it('adds the cross-site header and the workspace to an unsafe call under /api, with credentials', async () => {
    const page = host()
    await sessionFetch(page.fetch, page)('/api/documents/upload', { method: 'POST', body: 'x' })

    const headers = sentHeaders(page.calls[0])
    expect(headers.get(SESSION_CSRF_HEADER)).toBe('1')
    expect(headers.get('X-Workspace-ID')).toBe(WORKSPACE)
    expect(page.calls[0].init?.credentials).toBe('include')
    expect(page.calls[0].init?.body).toBe('x')
  })

  it("adds the workspace to a GET under /api, and keeps a header the caller set", async () => {
    const page = host()
    await sessionFetch(page.fetch, page)(`${PAGE}/api/knowledge`, { headers: { 'X-Workspace-ID': 'chosen' } })

    const headers = sentHeaders(page.calls[0])
    expect(headers.get('X-Workspace-ID')).toBe('chosen')
    expect(headers.has(SESSION_CSRF_HEADER)).toBe(false)
  })

  it('reports a 401 from the API as apiClient does', async () => {
    const page = host()
    page.fetch = (async () => new Response('{"detail":{"code":"session_ended","reason":"expired"}}', { status: 401 })) as typeof fetch
    const heard = vi.fn()
    const stop = onUnauthorized(heard)
    const answer = await sessionFetch(page.fetch, page)('/api/examples', { method: 'DELETE' })
    stop()

    expect(answer.status).toBe(401)
    expect(heard).toHaveBeenCalledWith({ detail: { code: 'session_ended', reason: 'expired' } })
  })

  it('builds a new Request from a Request, and never changes the one it was given', async () => {
    const page = host()
    const given = new Request(`${PAGE}/api/examples`, { method: 'PUT', headers: { Accept: 'application/json' } })
    await sessionFetch(page.fetch, page)(given)

    const sent = page.calls[0].input as Request
    expect(sent).not.toBe(given)
    expect(sent.headers.get(SESSION_CSRF_HEADER)).toBe('1')
    expect(sent.headers.get('Accept')).toBe('application/json')
    expect(given.headers.has(SESSION_CSRF_HEADER)).toBe(false)
    expect(given.headers.has('X-Workspace-ID')).toBe(false)
  })
})

describe('everything else passes through untouched', () => {
  it.each([
    ['an absolute address to another host', 'https://elsewhere.example.test/api/upload'],
    ['a protocol-relative address', '//elsewhere.example.test/api/upload'],
  ])('%s gets neither header', async (_label, address) => {
    const page = host()
    const init = { method: 'POST', body: 'x' }
    await sessionFetch(page.fetch, page)(address, init)

    expect(page.calls[0].input).toBe(address)
    expect(page.calls[0].init).toBe(init)
  })

  it('a Request object for another host is passed as it is', async () => {
    const page = host()
    const given = new Request('https://elsewhere.example.test/api/upload', { method: 'POST', body: 'x' })
    await sessionFetch(page.fetch, page)(given)

    expect(page.calls[0].input).toBe(given)
    expect(page.calls[0].init).toBeUndefined()
    expect(given.headers.has(SESSION_CSRF_HEADER)).toBe(false)
    expect(given.headers.has('X-Workspace-ID')).toBe(false)
  })

  it('a safe method to a path outside /api', async () => {
    const page = host()
    await sessionFetch(page.fetch, page)('/_next/data/page.json')

    expect(page.calls[0]).toEqual({ input: '/_next/data/page.json', init: undefined })
  })

  it('does not count /apiary as the API', async () => {
    const page = host()
    await sessionFetch(page.fetch, page)('/apiary')
    expect(page.calls[0].init).toBeUndefined()
  })
})

describe('installing it', () => {
  it('is never installed outside enterprise', () => {
    const page = host()
    const before = page.fetch
    const restore = installSessionFetch(page, false)
    expect(page.fetch).toBe(before)
    restore()
    expect(page.fetch).toBe(before)
  })

  it('is installed once in enterprise and taken off again', async () => {
    const page = host()
    const before = page.fetch
    const restore = installSessionFetch(page, true)
    const wrapped = page.fetch
    expect(wrapped).not.toBe(before)
    expect(installSessionFetch(page, true)).toBe(restore)
    expect(page.fetch).toBe(wrapped)

    await page.fetch('/api/x', { method: 'POST' })
    expect(sentHeaders(page.calls[0]).get(SESSION_CSRF_HEADER)).toBe('1')
    restore()
    expect(page.fetch).toBe(before)
  })
})
