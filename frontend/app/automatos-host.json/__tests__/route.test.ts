// @vitest-environment node
/**
 * PRD-261 B-03j — GET /automatos-host.json tells the theme CLI where this web app's API is.
 *
 * The CLI's `install --to <web app URL>` reads `{schema: 1, apiUrl}` from here (the kit's
 * host-resolve, B-03g). The answer is exactly those two keys, never cached, NEXT_PUBLIC_API_URL read
 * per request; with the variable unset it is a 500 with a plain sentence. The file needs no sign-in
 * because middleware.ts never runs for it: each `config.matcher` pattern is run here against the path
 * (as prd266-universal-links.test.ts reads the matcher) and none matches, since the first one skips a
 * path with an extension. If one ever does, this fails and the story stops for a D-point (DA-8),
 * rather than the path joining the public list: middleware.ts never names it.
 */
import { readFileSync } from 'fs'
import path from 'path'
import { afterEach, describe, expect, it, vi } from 'vitest'

// middleware.ts builds Clerk's middleware when it loads; only its exported matcher is read here.
vi.mock('@clerk/nextjs/server', () => ({
  createRouteMatcher: () => () => false,
  clerkMiddleware: (callback: unknown) => callback,
}))

import { config } from '@/middleware'
import * as route from '../route'

const HOST_FILE = '/automatos-host.json'
const API_URL_ENV = 'NEXT_PUBLIC_API_URL'
const MIDDLEWARE_SOURCE = path.resolve(__dirname, '..', '..', '..', 'middleware.ts')
// A transpiled module may carry this marker; it is no export (lib/theme-sdk/__tests__/shim.test.ts skips it too).
const ES_MODULE_FLAG = '__esModule'

/** Each matcher pattern as the anchored expression the middleware runs on (the universal-links reading). */
function matcherPatterns(): RegExp[] {
  return config.matcher.map((pattern) => new RegExp(`^${pattern}$`))
}

afterEach(() => {
  vi.unstubAllEnvs()
})

describe('PRD-261 B-03j — the host file answers exactly {schema, apiUrl}, never cached', () => {
  it('answers 200 JSON with schema 1 and the API URL, and Cache-Control: no-store', async () => {
    vi.stubEnv(API_URL_ENV, 'http://localhost:8000')
    const response = route.GET()

    expect(response.status).toBe(200)
    expect(response.headers.get('content-type')).toBe('application/json')
    expect(response.headers.get('cache-control')).toBe('no-store')
    expect(await response.json()).toStrictEqual({ schema: 1, apiUrl: 'http://localhost:8000' })
  })

  it('reads the variable per request', async () => {
    vi.stubEnv(API_URL_ENV, 'https://api.example.test')
    expect(await route.GET().json()).toStrictEqual({ schema: 1, apiUrl: 'https://api.example.test' })
    vi.stubEnv(API_URL_ENV, 'http://127.0.0.1:18000')
    expect(await route.GET().json()).toStrictEqual({ schema: 1, apiUrl: 'http://127.0.0.1:18000' })
  })

  it.each([
    ['unset', undefined],
    ['empty', ''],
  ])('is a 500 with a plain sentence naming the variable when it is %s', async (_state, value) => {
    vi.stubEnv(API_URL_ENV, value)
    const response = route.GET()

    expect(response.status).toBe(500)
    expect(response.headers.get('content-type')).toBe('text/plain; charset=utf-8')
    expect(response.headers.get('cache-control')).toBe('no-store')
    const text = await response.text()
    expect(text).toContain(API_URL_ENV)
    expect(text).not.toMatch(/[{}<>\n]/) // one plain line: no JSON, no markup, no stack
  })

  it('exports only GET and its route config, as a Next.js route handler must', () => {
    const exported = Object.keys(route).filter((name) => name !== ES_MODULE_FLAG)
    expect(exported.sort()).toStrictEqual(['GET', 'dynamic'])
    expect(route.dynamic).toBe('force-dynamic')
  })
})

describe('PRD-261 B-03j — no middleware runs for /automatos-host.json, so no auth surface changes', () => {
  it('matches none of middleware.ts config.matcher patterns', () => {
    expect(config.matcher.length).toBeGreaterThan(0)
    expect(matcherPatterns().filter((pattern) => pattern.test(HOST_FILE))).toStrictEqual([])
  })

  it.each(['/', '/chat', '/api/themes/active', '/.well-known/apple-app-site-association'])(
    'while the same patterns still match %s, so the check above can fail',
    (pathname) => {
      expect(matcherPatterns().some((pattern) => pattern.test(pathname))).toBe(true)
    },
  )

  it('is not in the SaaS public list: middleware.ts never names it', () => {
    expect(readFileSync(MIDDLEWARE_SOURCE, 'utf8')).not.toContain('automatos-host')
  })
})
