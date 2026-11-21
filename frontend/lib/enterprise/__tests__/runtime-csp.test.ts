// @vitest-environment node
/**
 * PRD-279 US-402 — media origins read at start-up; PRD-277 US-505 — no address built into the
 * enterprise image.
 *
 * Next evaluates next.config.js's `headers()` in `next build`, so the policy it writes is fixed in
 * the image. In the enterprise build:
 *   - the middleware sets the policy on every answer it gives, with the media origins the running
 *     server was given (MEDIA_ORIGINS), read on each answer;
 *   - the build keeps every other security header on every path, and puts its own policy only on
 *     the paths the middleware never runs for, so no answer gets two policies and every page gets one;
 *   - the build refuses an API or WebSocket address of its own, naming the variable, not its value;
 *   - with no media origin the middleware's policy is exactly the build's: never looser, and
 *     wider only by the origins the operator named.
 * In local and saas the config is what it was: one rule on every path, and the same policy string
 * byte for byte (the strings below are the directive list as next.config.js wrote it before).
 */
import { createRequire } from 'module'
import { NextRequest } from 'next/server'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { CSP_HEADER, enterprisePolicy, runningPolicy } from '@/lib/enterprise/content-security-policy'
import { enterpriseMiddleware } from '@/lib/enterprise/session-middleware'
import { SESSION_COOKIE_NAME } from '@/lib/enterprise/session-transport'

// middleware.ts builds Clerk's middleware when it loads; only its exported matcher is read here.
vi.mock('@clerk/nextjs/server', () => ({
  createRouteMatcher: () => () => false,
  clerkMiddleware: (callback: unknown) => callback,
}))

import { config as middlewareConfig } from '@/middleware'

type Header = { key: string; value: string }
type Rule = { source: string; headers: Header[] }

const load = createRequire(import.meta.url)
const CONFIG = load.resolve('../../../next.config.js')
const BUILD_ENV = [
  'NEXT_PUBLIC_AUTH_EDITION', 'NEXT_PUBLIC_API_URL', 'NEXT_PUBLIC_WS_URL', 'NEXT_PUBLIC_MEDIA_ORIGINS',
  'NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY',
] as const
const MEDIA = 'https://media.example.test'
const OTHER_MEDIA = 'https://*.storage.example.test'
const ORIGIN = 'https://automatos.example.test'

const SCRIPT_NO_BILLING =
  "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://*.clerk.accounts.dev https://*.protect.clerk.com https://challenges.cloudflare.com https://cdn.jsdelivr.net"
const IMG_BASE =
  "img-src 'self' data: blob: https://*.clerk.accounts.dev https://img.clerk.com https://*.googleusercontent.com https://logos.composio.dev https://api.automatos.app/api/marketplace/catalog/files/"
const FRAME_NO_BILLING =
  "frame-src 'self' blob: https://*.clerk.accounts.dev https://*.protect.clerk.com https://challenges.cloudflare.com"
const TAIL = ["worker-src 'self' blob:", 'object-src blob:', "base-uri 'self'", "form-action 'self'", "frame-ancestors 'none'"]

/** The saas build with nothing set (no key, no API address, no media origin). */
const SAAS_POLICY = [
  "default-src 'self'",
  `${SCRIPT_NO_BILLING} https://js.stripe.com https://*.js.stripe.com https://maps.googleapis.com`,
  "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net",
  `${IMG_BASE} https://*.link.com`,
  "media-src 'self' blob:",
  "font-src 'self' data: https://cdn.jsdelivr.net",
  "connect-src 'self'  https://*.automatos.app https://*.clerk.accounts.dev https://*.protect.clerk.com:* https://api.clerk.com https://cdn.jsdelivr.net https://api.stripe.com https://maps.googleapis.com https://link.com https://*.link.com wss: ws:",
  `${FRAME_NO_BILLING} https://js.stripe.com https://*.js.stripe.com https://hooks.stripe.com https://link.com https://*.link.com`,
  ...TAIL,
].join('; ')

/** The published local image's build arguments (images.yml). */
const LOCAL_POLICY = [
  "default-src 'self'",
  SCRIPT_NO_BILLING,
  "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net",
  `${IMG_BASE} http://localhost:8000/api/marketplace/catalog/files/ http://localhost:9000`,
  "media-src 'self' blob: http://localhost:9000",
  "font-src 'self' data: https://cdn.jsdelivr.net",
  "connect-src 'self' http://localhost:8000 https://*.automatos.app https://*.clerk.accounts.dev https://*.protect.clerk.com:* https://api.clerk.com https://cdn.jsdelivr.net wss: ws:",
  FRAME_NO_BILLING,
  ...TAIL,
].join('; ')

/**
 * The enterprise build (PRD-277 D-12, #1605): no Clerk, Cloudflare, Automatos or payment host, no
 * media origin, and Cloud sync off (enterprise-own-hosts-csp.test.ts pins it with Cloud sync on).
 */
const ENTERPRISE_IMG = "img-src 'self' data: blob: https://*.googleusercontent.com https://logos.composio.dev"
const ENTERPRISE_POLICY = [
  "default-src 'self'",
  "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net",
  "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net",
  ENTERPRISE_IMG,
  "media-src 'self' blob:",
  "font-src 'self' data: https://cdn.jsdelivr.net",
  "connect-src 'self' https://cdn.jsdelivr.net wss: ws:",
  "frame-src 'self' blob:",
  ...TAIL,
].join('; ')

function loadConfig(env: Partial<Record<(typeof BUILD_ENV)[number], string>>): { headers: () => Promise<Rule[]> } {
  for (const name of BUILD_ENV) vi.stubEnv(name, env[name] ?? '')
  delete load.cache[CONFIG]
  return load(CONFIG)
}

function policyOf(rule: Rule | undefined): string | undefined {
  return rule?.headers.find((header) => header.key === CSP_HEADER)?.value
}

function directives(policy: string): Map<string, string> {
  return new Map(policy.split('; ').map((part) => [part.split(' ')[0], part]))
}

const middleware = enterpriseMiddleware(() => false)

function visit(path: string, cookie?: string): Response {
  const headers: Record<string, string> = cookie ? { cookie } : {}
  return middleware(new NextRequest(new URL(path, ORIGIN), { headers }))
}

afterEach(() => {
  vi.unstubAllEnvs()
  vi.restoreAllMocks()
  delete load.cache[CONFIG]
})

describe('the middleware sets the policy with the running server’s media origins', () => {
  it('on a page it lets through and on a redirect to sign in', () => {
    vi.stubEnv('MEDIA_ORIGINS', MEDIA)
    const through = visit('/chat', `${SESSION_COOKIE_NAME}=${'s'.repeat(40)}`)
    const redirected = visit('/settings')

    expect(redirected.status).toBe(307)
    for (const answer of [through, redirected]) {
      const policy = directives(answer.headers.get(CSP_HEADER) ?? '')
      expect(policy.get('img-src')).toBe(`${ENTERPRISE_IMG} ${MEDIA}`)
      expect(policy.get('media-src')).toBe(`media-src 'self' blob: ${MEDIA}`)
    }
  })

  it('on the API paths it leaves alone, as the build put it there before', () => {
    vi.stubEnv('MEDIA_ORIGINS', MEDIA)
    expect(visit('/api/chat').headers.get(CSP_HEADER)).toBe(enterprisePolicy([MEDIA]))
  })

  it('reads the setting the server has, not one fixed when the code was built', () => {
    vi.stubEnv('MEDIA_ORIGINS', MEDIA)
    expect(visit('/sign-in').headers.get(CSP_HEADER)).toContain(MEDIA)
    vi.stubEnv('MEDIA_ORIGINS', OTHER_MEDIA)
    const policy = visit('/sign-in').headers.get(CSP_HEADER) ?? ''
    expect(policy).toContain(OTHER_MEDIA)
    expect(policy).not.toContain(MEDIA)
  })

  it('names no media origin, and logs the setting without its value, for a bad value', () => {
    const logged = vi.spyOn(console, 'error').mockImplementation(() => {})
    expect(runningPolicy("https://media.example.test 'unsafe-inline'")).toBe(ENTERPRISE_POLICY)
    expect(String(logged.mock.calls[0][0])).toContain('MEDIA_ORIGINS')
    expect(String(logged.mock.calls[0][0])).not.toContain('media.example.test')
  })

  it('parses once per distinct value, and logs a bad value once per process however often it is served', async () => {
    vi.resetModules()
    const parse = vi.fn()
    vi.doMock('@/lib/enterprise/media-origins', async (importOriginal) => {
      const actual = await importOriginal<typeof import('@/lib/enterprise/media-origins')>()
      parse.mockImplementation(actual.parseMediaOrigins)
      return { ...actual, parseMediaOrigins: parse }
    })
    const fresh = await import('@/lib/enterprise/content-security-policy')
    const logged = vi.spyOn(console, 'error').mockImplementation(() => {})

    for (let n = 0; n < 3; n += 1) fresh.runningPolicy(MEDIA)
    expect(parse).toHaveBeenCalledTimes(1)
    expect(fresh.runningPolicy(OTHER_MEDIA)).toContain(OTHER_MEDIA)
    expect(parse).toHaveBeenCalledTimes(2)
    for (const bad of ['https://a.example.test/x', 'data:', 'https://a.example.test/x']) {
      for (let n = 0; n < 3; n += 1) expect(fresh.runningPolicy(bad)).toBe(ENTERPRISE_POLICY)
    }
    expect(logged).toHaveBeenCalledTimes(1)
    vi.doUnmock('@/lib/enterprise/media-origins')
  })

  it('with no media origin is exactly the enterprise build’s policy, and adds only img-src and media-src', () => {
    expect(runningPolicy('')).toBe(ENTERPRISE_POLICY)
    const base = directives(enterprisePolicy([]))
    const wider = directives(enterprisePolicy([MEDIA, OTHER_MEDIA]))
    for (const [name, value] of base) {
      if (name === 'img-src' || name === 'media-src') expect(wider.get(name)).toBe(`${value} ${MEDIA} ${OTHER_MEDIA}`)
      else expect(wider.get(name)).toBe(value)
    }
  })
})

describe('the enterprise build', () => {
  const ENTERPRISE_BUILD = { NEXT_PUBLIC_AUTH_EDITION: 'enterprise' } as const

  it('keeps every other security header on every path, without the policy', async () => {
    const [everyPath, ...rest] = await loadConfig(ENTERPRISE_BUILD).headers()
    expect(everyPath.source).toBe('/(.*)')
    expect(policyOf(everyPath)).toBeUndefined()
    expect(everyPath.headers.map((header) => header.key)).toEqual([
      'X-Frame-Options', 'X-Content-Type-Options', 'Referrer-Policy', 'Permissions-Policy',
      'Strict-Transport-Security', 'X-DNS-Prefetch-Control',
    ])
    expect(rest.map((rule) => rule.source)).toEqual(['/:path(_next.*)', '/:path((?!api|trpc).+\\.\\w+)'])
  })

  it('puts its own policy, the middleware’s with no media origin, on the paths the middleware skips', async () => {
    const [, ...rest] = await loadConfig(ENTERPRISE_BUILD).headers()
    for (const rule of rest) expect(policyOf(rule)).toBe(ENTERPRISE_POLICY)
    expect(enterprisePolicy([])).toBe(ENTERPRISE_POLICY)
  })

  it('reads neither the build’s media origins nor a Clerk key', async () => {
    // A key of Clerk's documented shape (pk_ + kind + base64 of "<host>$"), not a real key.
    const key = `pk_${'live'}_${Buffer.from('clerk.example.test$').toString('base64')}`
    const env = { ...ENTERPRISE_BUILD, NEXT_PUBLIC_MEDIA_ORIGINS: 'https://*.amazonaws.com', NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY: key }
    const [, staticPaths] = await loadConfig(env).headers()
    expect(policyOf(staticPaths)).toBe(ENTERPRISE_POLICY)
  })

  it.each(['NEXT_PUBLIC_API_URL', 'NEXT_PUBLIC_WS_URL'] as const)('refuses %s, naming it and not its value', (name) => {
    const value = 'https://address.example.test'
    expect(() => loadConfig({ ...ENTERPRISE_BUILD, [name]: value })).toThrow(name)
    expect(() => loadConfig({ ...ENTERPRISE_BUILD, [name]: value })).not.toThrow(value)
  })

  it.each(['NEXT_PUBLIC_API_URL', 'NEXT_PUBLIC_WS_URL'] as const)('refuses %s of spaces alone, untrimmed', (name) => {
    expect(() => loadConfig({ ...ENTERPRISE_BUILD, [name]: '  ' })).toThrow(name)
  })

  it('makes an unset address the empty string before Next inlines it, never the text undefined', () => {
    loadConfig(ENTERPRISE_BUILD)
    vi.stubEnv('NEXT_PUBLIC_API_URL', undefined)
    vi.stubEnv('NEXT_PUBLIC_WS_URL', undefined)
    delete load.cache[CONFIG]
    load(CONFIG)
    expect(process.env.NEXT_PUBLIC_API_URL).toBe('')
    expect(process.env.NEXT_PUBLIC_WS_URL).toBe('')
  })

  it('leaves an unset address unset outside enterprise', () => {
    loadConfig({})
    vi.stubEnv('NEXT_PUBLIC_API_URL', undefined)
    delete load.cache[CONFIG]
    load(CONFIG)
    expect(process.env.NEXT_PUBLIC_API_URL).toBeUndefined()
  })

  it('gives each path the policy from exactly one place: the build or the middleware', async () => {
    const [, ...rest] = await loadConfig(ENTERPRISE_BUILD).headers()
    const buildPaths = rest.map((rule) => new RegExp(`^${rule.source.replace('/:path(', '/(')}$`, 'i'))
    const middlewarePaths = middlewareConfig.matcher.map((pattern) => new RegExp(`^${pattern}$`))
    const paths = [
      '/', '/chat', '/chat/abc', '/sign-in', '/.well-known/apple-app-site-association', '/.env', '/api', '/api/chat',
      '/api/themes/files/d/dist/entry.js', '/trpc/x.json', '/_next/static/chunks/app.js', '/_next/image',
      '/favicon.ico', '/brand/automatos-mark-hi.png', '/automatos-host.json', '/.well-known/assetlinks.json',
      '/deliverables/report.pdf',
    ]
    for (const path of paths) {
      const fromBuild = buildPaths.some((pattern) => pattern.test(path))
      const fromMiddleware = middlewarePaths.some((pattern) => pattern.test(path))
      expect(fromBuild !== fromMiddleware, path).toBe(true)
    }
  })
})

describe('local and saas are what they were', () => {
  it('saas: one rule on every path, with the same policy', async () => {
    const rules = await loadConfig({})
      .headers()
    expect(rules).toHaveLength(1)
    expect(rules[0].source).toBe('/(.*)')
    expect(policyOf(rules[0])).toBe(SAAS_POLICY)
  })

  it('local, with the published image’s build arguments: one rule, the same policy', async () => {
    const rules = await loadConfig({
      NEXT_PUBLIC_AUTH_EDITION: 'local',
      NEXT_PUBLIC_API_URL: 'http://localhost:8000',
      NEXT_PUBLIC_WS_URL: 'ws://localhost:8000/ws',
      NEXT_PUBLIC_MEDIA_ORIGINS: 'http://localhost:9000',
    }).headers()
    expect(rules).toHaveLength(1)
    expect(policyOf(rules[0])).toBe(LOCAL_POLICY)
  })
})
