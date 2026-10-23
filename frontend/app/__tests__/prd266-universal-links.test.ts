// @vitest-environment node
/**
 * PRD-266 US-018c — ui.automatos.app declares Automatos Go for /go/*, Command Center and chat links.
 *
 * iOS fetches /.well-known/apple-app-site-association and Android /.well-known/assetlinks.json,
 * signed out. middleware.ts is imported for real with isSaaS true (the files-proxy-middleware
 * pattern): clerkMiddleware hands its callback a signed-out auth, and createRouteMatcher is Clerk's
 * own matching, so only the two exact paths skip auth.protect(). The route handlers answer 200 JSON
 * with a public Cache-Control; assetlinks reads ANDROID_APP_SHA256_CERT_FINGERPRINTS per request.
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
import { GET as getAppleAppSiteAssociation } from '@/app/.well-known/apple-app-site-association/route'
import { GET as getAssetLinks } from '@/app/.well-known/assetlinks.json/route'

const AASA = '/.well-known/apple-app-site-association'
const ASSET_LINKS = '/.well-known/assetlinks.json'
const FINGERPRINTS_ENV = 'ANDROID_APP_SHA256_CERT_FINGERPRINTS'
// A made-up SHA-256 in the AA:BB:… form (32 upper-case hex pairs); the malformed one is lower case.
const VALID_FINGERPRINT = Array.from({ length: 32 }, (_, i) => (i * 7).toString(16).padStart(2, '0').toUpperCase()).join(':')
const MALFORMED_FINGERPRINT = VALID_FINGERPRINT.toLowerCase()
type Middleware = (request: { nextUrl: URL }) => Promise<unknown>

function runMiddleware(pathname: string): Promise<unknown> {
  return (middleware as unknown as Middleware)({ nextUrl: new URL(pathname, 'http://app.test') })
}

afterEach(() => {
  clerk.protect.mockClear()
  clerk.auth.mockClear()
  vi.unstubAllEnvs()
})

describe('PRD-266 US-018c — the association files are public in SaaS, nothing else under /.well-known', () => {
  it.each([AASA, ASSET_LINKS])('lets a signed-out %s through without auth.protect()', async (pathname) => {
    await expect(runMiddleware(pathname)).resolves.toBeUndefined()
    expect(clerk.protect).not.toHaveBeenCalled()
  })

  it.each(['/.well-known/other', '/.well-known', `${AASA}/extra`, `${AASA}-x`, '/chat'])(
    'still protects %s',
    async (pathname) => {
      await expect(runMiddleware(pathname)).rejects.toThrow(/auth\.protect\(\) answered 404/)
      expect(clerk.protect).toHaveBeenCalledTimes(1)
    },
  )

  it('needs the AASA entry: the matcher skips assetlinks.json (an extension) but not the extensionless AASA', () => {
    const middlewareRunsOn = new RegExp(`^${config.matcher[0]}$`)
    expect(middlewareRunsOn.test(AASA)).toBe(true)
    expect(middlewareRunsOn.test(ASSET_LINKS)).toBe(false)
  })
})

describe('PRD-266 US-018c — each file answers 200 JSON, cacheable, never a redirect', () => {
  it.each([
    ['apple-app-site-association', getAppleAppSiteAssociation],
    ['assetlinks.json', getAssetLinks],
  ])('%s answers 200 with content-type application/json and a public max-age', (_name, get) => {
    const response = get()
    expect(response.status).toBe(200)
    expect(response.headers.get('content-type')).toBe('application/json')
    expect(response.headers.get('cache-control')).toMatch(/^public, max-age=[1-9]\d*$/)
    expect(response.headers.get('location')).toBeNull()
  })

  it('the AASA body is exactly the app id and the five link paths', async () => {
    expect(await getAppleAppSiteAssociation().json()).toStrictEqual({
      applinks: {
        details: [
          {
            appIDs: ['KGVA9933BG.com.automatos.app'],
            components: [
              { '/': '/go/*' },
              { '/': '/command-center' },
              { '/': '/command-center/*' },
              { '/': '/chat' },
              { '/': '/chat/*' },
            ],
          },
        ],
      },
    })
  })
})

describe('PRD-266 US-018c — assetlinks.json lists the Android app only with a well-formed fingerprint', () => {
  it('is an empty statement list when the env is unset', async () => {
    expect(process.env[FINGERPRINTS_ENV]).toBeUndefined()
    expect(await getAssetLinks().json()).toStrictEqual([])
  })

  it('is an empty statement list when nothing in the env is well-formed', async () => {
    vi.stubEnv(FINGERPRINTS_ENV, `${MALFORMED_FINGERPRINT}, not-a-fingerprint`)
    expect(await getAssetLinks().json()).toStrictEqual([])
  })

  it('lists exactly the valid fingerprint of one valid and one malformed, read at request time', async () => {
    vi.stubEnv(FINGERPRINTS_ENV, `${MALFORMED_FINGERPRINT}, ${VALID_FINGERPRINT}`)
    expect(await getAssetLinks().json()).toStrictEqual([
      {
        relation: ['delegate_permission/common.handle_all_urls'],
        target: {
          namespace: 'android_app',
          package_name: 'com.automatos.app',
          sha256_cert_fingerprints: [VALID_FINGERPRINT],
        },
      },
    ])
  })

  it('lists a repeated fingerprint once', async () => {
    vi.stubEnv(FINGERPRINTS_ENV, `${VALID_FINGERPRINT},${VALID_FINGERPRINT}`)
    const [statement] = await getAssetLinks().json()
    expect(statement.target.sha256_cert_fingerprints).toStrictEqual([VALID_FINGERPRINT])
  })
})
