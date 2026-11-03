// @vitest-environment node
/**
 * PRD-270 US-270-08a (K19) — in SaaS the middleware lets a signed-out visitor open /marketplace.
 *
 * isPublicRoute gains exactly '/marketplace'. Clerk's createRouteMatcher matches the pathname only,
 * so a tab, item or add query rides along, while /marketplace/publish, /marketplace/developer,
 * /marketplace/widgets and /marketplaces stay protected. middleware.ts is imported for real with
 * isSaaS true; clerkMiddleware hands its callback a signed-out auth, and createRouteMatcher is
 * Clerk's own matching (createPathMatcher over the request's pathname, as server/routeMatcher.js
 * does), as lib/theme-host/__tests__/files-proxy-middleware.test.ts sets it up.
 */
import { readFileSync } from 'fs'
import path from 'path'
import { afterEach, describe, expect, it, vi } from 'vitest'

const clerk = vi.hoisted(() => {
  const protect = vi.fn(async () => {
    throw new Error('auth.protect() redirected a signed-out visitor to sign-in')
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

import middleware from '@/middleware'

type Middleware = (request: { nextUrl: URL }) => Promise<unknown>

function runMiddleware(href: string): Promise<unknown> {
  return (middleware as unknown as Middleware)({ nextUrl: new URL(href, 'https://app.test') })
}

afterEach(() => {
  clerk.protect.mockClear()
  clerk.auth.mockClear()
})

describe('PRD-270 US-270-08a — /marketplace is a public route in SaaS', () => {
  it.each(['/marketplace', '/marketplace?tab=themes&item=automatos.arcade&add=1'])(
    'lets a signed-out %s through without auth.protect()',
    async (href) => {
      await expect(runMiddleware(href)).resolves.toBeUndefined()
      expect(clerk.protect).not.toHaveBeenCalled()
    },
  )

  it.each(['/marketplace/publish', '/marketplace/developer', '/marketplace/widgets', '/marketplaces', '/chat'])(
    'still protects %s',
    async (href) => {
      await expect(runMiddleware(href)).rejects.toThrow(/auth\.protect\(\) redirected/)
      expect(clerk.protect).toHaveBeenCalledTimes(1)
    },
  )

  it('adds only the one marketplace pattern to the public list', () => {
    const source = readFileSync(path.resolve(__dirname, '..', '..', 'middleware.ts'), 'utf8')
    const publicList = source.slice(source.indexOf('createRouteMatcher(['), source.indexOf(']);'))
    const patterns = Array.from(publicList.matchAll(/^\s*"([^"]+)",/gm), (match) => match[1])
    expect(patterns.filter((pattern) => pattern.includes('marketplace'))).toEqual(['/marketplace'])
  })
})
