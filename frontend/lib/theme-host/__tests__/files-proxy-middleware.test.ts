// @vitest-environment node
/**
 * PRD-261 A-08b — in SaaS the middleware lets the theme files proxy through to its handler.
 *
 * auth.protect() answers a signed-out request that is not a page (a module import(), a <video>
 * range request) with a 404 before any handler runs (@clerk/nextjs server/protect.js), which the
 * theme host would read as a broken theme. So '/api/themes/files(.*)' is in isPublicRoute (maintainer
 * OK at D1-g) and the handler answers 401 itself. middleware.ts is imported for real with isSaaS
 * true; clerkMiddleware hands its callback a signed-out auth, and createRouteMatcher is Clerk's own
 * matching (createPathMatcher over the request's pathname, as server/routeMatcher.js does).
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

const clerk = vi.hoisted(() => {
  const protect = vi.fn(async () => {
    throw new Error('auth.protect() answered 404: signed out, and not a page')
  })
  const auth = Object.assign(vi.fn(async () => ({ userId: null, getToken: async () => null })), { protect })
  return { auth, protect }
})

vi.mock('@/lib/auth-edition', () => ({ isSaaS: true }))
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

import middleware, { config } from '@/middleware'
import { GET } from '@/app/api/themes/files/[...path]/route'

const DIGEST = 'b'.repeat(64)
const FILES = `/api/themes/files/${DIGEST}`
type Middleware = (request: { nextUrl: URL }) => Promise<unknown>

function runMiddleware(pathname: string): Promise<unknown> {
  return (middleware as unknown as Middleware)({ nextUrl: new URL(pathname, 'http://app.test') })
}

afterEach(() => {
  clerk.protect.mockClear()
  clerk.auth.mockClear()
  vi.unstubAllGlobals()
})

describe('PRD-261 A-08b — the files proxy is a public route in SaaS', () => {
  it.each([`${FILES}/dist/entry.js`, `${FILES}/dist/chunks/a.js`, `${FILES}/assets/mascot/idle.mp4`])(
    'lets a signed-out %s through without auth.protect()',
    async (pathname) => {
      await expect(runMiddleware(pathname)).resolves.toBeUndefined()
      expect(clerk.protect).not.toHaveBeenCalled()
    },
  )

  it.each(['/chat', '/api/themes/active', '/api/themes/import', '/api/themes/boot/x', '/api/themes'])(
    'still protects %s',
    async (pathname) => {
      await expect(runMiddleware(pathname)).rejects.toThrow(/auth\.protect\(\) answered 404/)
      expect(clerk.protect).toHaveBeenCalledTimes(1)
    },
  )

  it('runs clerkMiddleware on API paths, so the handler can read the session', () => {
    expect(config.matcher).toContain('/(api|trpc)(.*)')
  })

  it('reaches the handler, which answers the signed-out request with 401 and asks no backend', async () => {
    const backend = vi.fn()
    vi.stubGlobal('fetch', backend)
    const pathname = `${FILES}/dist/entry.js`
    await runMiddleware(pathname)
    const response = await GET(new Request(`http://app.test${pathname}`), {
      params: Promise.resolve({ path: [DIGEST, 'dist', 'entry.js'] }),
    })
    expect(response.status).toBe(401)
    expect(clerk.auth).toHaveBeenCalledTimes(1)
    expect(backend).not.toHaveBeenCalled()
  })
})
