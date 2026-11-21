// @vitest-environment node
/**
 * PRD-277 US-502 — the public routes are one list (lib/public-routes.ts): the saas middleware
 * hands it to Clerk's `createRouteMatcher`, and the enterprise middleware and session provider read
 * it with `isPublicPath`. Both readings agree on every path, so a page is public in every edition
 * or in none. Clerk's own matcher (`createPathMatcher`, what `createRouteMatcher` runs) is the
 * reference, as lib/__tests__/marketplace-public-middleware.test.ts uses it.
 */
import { readFileSync } from 'fs'
import path from 'path'
import { createPathMatcher } from '@clerk/shared/pathMatcher'
import { describe, expect, it } from 'vitest'

import { PUBLIC_ROUTES, isPublicPath } from '@/lib/public-routes'

const PATHS = [
  '/', '/chat', '/agents', '/sign-in', '/sign-in/factor-one', '/sign-inx', '/sign-up', '/reset-password',
  '/sso-callback', '/accept-invitation/abc', '/api/webhooks', '/api/webhooks/clerk', '/api/themes/files/a/b.js',
  '/api/themes/files', '/api/chat', '/verify', '/verify/abc', '/verify-email', '/verification',
  '/.well-known/apple-app-site-association', '/.well-known/assetlinks.json', '/.well-known/other',
  '/go/pair', '/go/pairing', '/go', '/marketplace', '/marketplace/publish', '/marketplaces', '/team',
]

describe('the public routes', () => {
  it("are read by isPublicPath exactly as Clerk's matcher reads them", () => {
    const clerk = createPathMatcher([...PUBLIC_ROUTES])
    expect(PATHS.map((each) => [each, isPublicPath(each)])).toEqual(PATHS.map((each) => [each, clerk(each)]))
  })

  it('use only the two shapes isPublicPath reads: an exact path, or a prefix and (.*)', () => {
    for (const pattern of PUBLIC_ROUTES) {
      const prefix = pattern.endsWith('(.*)') ? pattern.slice(0, -4) : pattern
      expect(prefix, pattern).not.toMatch(/[()*:?[\]]/)
    }
  })

  it('are the list both middlewares use', () => {
    const middleware = readFileSync(path.resolve(__dirname, '..', '..', 'middleware.ts'), 'utf8')
    expect(middleware).toContain('createRouteMatcher([...PUBLIC_ROUTES])')
    expect(middleware).toContain('isPublicPath(request.nextUrl.pathname)')
    expect(isPublicPath(null)).toBe(false)
  })
})
