/**
 * PRD-266 US-018d — /go/pair, where a pairing QR opens in a phone's browser.
 *
 * - A valid link shows the code (XXXX-XXXX, any case in) and the server's host only.
 * - A malformed code, or an api that is not an http(s) URL (javascript:), shows the one
 *   incomplete-link message and renders neither value.
 * - It never pairs: apiClient and fetch are mocked and never called, no form, no storage, and no
 *   link is built from the query.
 * - In SaaS the real middleware (Clerk's own path matcher, signed out, as in US-018c's test) lets
 *   /go/pair through and still protects /go, /go/pairing and /gopher. next/server is stubbed: the
 *   saas branch never touches it, and this file runs in jsdom to render the page.
 *
 * The page header is stubbed (its theme slots are not under test), as in verify-page.test.tsx.
 */
import { readFileSync } from 'fs'
import path from 'path'
import React from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'

const nav = vi.hoisted(() => ({ search: '' }))
const api = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn(), put: vi.fn(), patch: vi.fn(), delete: vi.fn() }))
const clerk = vi.hoisted(() => {
  const protect = vi.fn(async () => {
    throw new Error('auth.protect() answered 404: signed out, and not a page')
  })
  const auth = Object.assign(vi.fn(async () => ({ userId: null, getToken: async () => null })), { protect })
  return { auth, protect }
})

vi.mock('next/navigation', () => ({ useSearchParams: () => new URLSearchParams(nav.search) }))
vi.mock('next/server', () => ({ NextResponse: { next: () => 'next' } }))
vi.mock('@/lib/api-client', () => ({ apiClient: api }))
vi.mock('@/lib/auth-edition', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/lib/auth-edition')>()),
  isSaaS: true,
}))
vi.mock('@clerk/nextjs/server', async () => {
  const { createPathMatcher } = await import('@clerk/shared/pathMatcher')
  type Patterns = Parameters<typeof createPathMatcher>[0]
  type Callback = (auth: typeof clerk.auth, request: { nextUrl: URL }) => Promise<unknown>
  return {
    auth: clerk.auth,
    createRouteMatcher: (routes: Patterns) => {
      const matches = createPathMatcher(routes)
      return (request: { nextUrl: URL }) => matches(request.nextUrl.pathname)
    },
    clerkMiddleware: (callback: Callback) => (request: { nextUrl: URL }) => callback(clerk.auth, request),
  }
})
vi.mock('@/components/shared/page-header', () => ({
  PageHeader: ({ title, titleAccent }: { title: string; titleAccent: string }) => (
    <h1>
      {title} {titleAccent}
    </h1>
  ),
}))

import GoPairPage from '@/app/go/pair/page'
import middleware from '@/middleware'
import { INCOMPLETE_LINK_MESSAGE, TRUST_WARNING } from '@/components/go-pair/go-pair-panel'
import { GET_AUTOMATOS_GO } from '@/components/go-pair/get-automatos-go'
import { SAAS_ONLY_ROUTES } from '@/lib/auth-edition'

const CODE = 'K7QM-X4PZ'
const HOST = 'automatos.local:8000'
const API = `http://${HOST}/some/path?token=never-shown`
const fetchSpy = vi.fn()
type Middleware = (request: { nextUrl: URL }) => Promise<unknown>
type Query = Readonly<Record<string, string>>

function openLink(query: Query) {
  nav.search = new URLSearchParams(query).toString()
  return render(<GoPairPage />)
}

function runMiddleware(pathname: string): Promise<unknown> {
  return (middleware as unknown as Middleware)({ nextUrl: new URL(pathname, 'http://app.test') })
}

function expectNothingCalled() {
  for (const method of Object.values(api)) expect(method).not.toHaveBeenCalled()
  expect(fetchSpy).not.toHaveBeenCalled()
}

beforeEach(() => {
  vi.stubGlobal('fetch', fetchSpy)
})

afterEach(() => {
  cleanup()
  nav.search = ''
  fetchSpy.mockReset()
  clerk.protect.mockClear()
  vi.unstubAllGlobals()
  window.localStorage.clear()
  window.sessionStorage.clear()
})

describe('PRD-266 US-018d — /go/pair repeats a valid link', () => {
  it.each([CODE, 'k7qm-x4pz', 'K7QMX4PZ'])('shows the code %s as XXXX-XXXX and the server as its host only', (code) => {
    const { container } = openLink({ code, api: API })

    expect(screen.getByTestId('go-pair-code')).toHaveTextContent(CODE)
    expect(screen.getByTestId('go-pair-server')).toHaveTextContent(HOST)
    expect(container).not.toHaveTextContent('never-shown')
    expect(container).not.toHaveTextContent('/some/path')
    expect(screen.queryByTestId('go-pair-incomplete')).toBeNull()
    expect(screen.getByTestId('go-pair-trust')).toHaveTextContent(TRUST_WARNING)
    expect(screen.getByText(/scan the QR code again, from inside the app/)).toBeInTheDocument()
    expect(screen.getByText(/‘Connect to your own server’/)).toBeInTheDocument()
  })

  it('says how to get Automatos Go from the one constant', () => {
    openLink({ code: CODE, api: API })
    expect(screen.getByRole('heading', { name: GET_AUTOMATOS_GO.heading })).toBeInTheDocument()
    for (const way of GET_AUTOMATOS_GO.ways) expect(screen.getByText(way)).toBeInTheDocument()
    expect(GET_AUTOMATOS_GO.ways.join(' ')).toMatch(/TestFlight/)
  })
})

describe('PRD-266 US-018d — anything else is an incomplete link, and shows neither value', () => {
  it.each<[string, Query]>([
    ['a malformed code', { code: 'K0QM-X4PZ', api: API }],
    ['a code of the wrong length', { code: 'K7QM-X4P', api: API }],
    ['a javascript: api', { code: CODE, api: 'javascript:alert(document.domain)' }],
    ['an api that is not a URL', { code: CODE, api: HOST }],
    ['no api', { code: CODE }],
    ['no code', { api: API }],
  ])('%s', (_case, query) => {
    const { container } = openLink(query)

    expect(screen.getByTestId('go-pair-incomplete')).toHaveTextContent(INCOMPLETE_LINK_MESSAGE)
    expect(screen.queryByTestId('go-pair-code')).toBeNull()
    expect(screen.queryByTestId('go-pair-server')).toBeNull()
    expect(container).not.toHaveTextContent(CODE)
    expect(container).not.toHaveTextContent(HOST)
    expect(container).not.toHaveTextContent('javascript')
  })
})

describe('PRD-266 US-018d — it never pairs', () => {
  it.each<[string, Query]>([
    ['a valid link', { code: CODE, api: API }],
    ['an incomplete link', { code: 'nope', api: 'javascript:alert(1)' }],
  ])('%s calls no API, has no form, stores nothing and builds no link from the query', (_case, query) => {
    const { container } = openLink(query)

    expectNothingCalled()
    expect(container.querySelector('form')).toBeNull()
    expect(window.localStorage.length).toBe(0)
    expect(window.sessionStorage.length).toBe(0)
    for (const anchor of Array.from(container.querySelectorAll('a'))) {
      const href = anchor.getAttribute('href') ?? ''
      expect(href).not.toContain(HOST)
      expect(href).not.toMatch(/javascript:|automatos[a-z_-]*:\/\//i)
    }
  })

  it('wears the shared page frame', () => {
    const page = readFileSync(path.resolve(__dirname, '..', 'go', 'pair', 'page.tsx'), 'utf8')
    expect(page).toContain('<PageHeader')
  })
})

describe('PRD-266 US-018d — public in SaaS at the exact path, and served by the local edition', () => {
  it('lets a signed-out /go/pair through without auth.protect()', async () => {
    await expect(runMiddleware('/go/pair')).resolves.toBeUndefined()
    expect(clerk.protect).not.toHaveBeenCalled()
  })

  it.each(['/go', '/go/pairing', '/gopher', '/go/pair/extra', '/go/other'])('still protects %s', async (pathname) => {
    await expect(runMiddleware(pathname)).rejects.toThrow(/auth\.protect\(\) answered 404/)
    expect(clerk.protect).toHaveBeenCalledTimes(1)
  })

  it('is no hosted-only route, so the local edition serves it', () => {
    expect(SAAS_ONLY_ROUTES.some((prefix) => '/go/pair' === prefix || '/go/pair'.startsWith(`${prefix}/`))).toBe(false)
  })
})
