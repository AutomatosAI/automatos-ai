/**
 * Clerk production readiness (19 Oct 2026, "Automatos billing options" section 4, bug 1): the
 * Content-Security-Policy allowed clerk-js only from *.clerk.accounts.dev, the development
 * instance. A production key (pk_live_) loads it from the instance's own Frontend API host
 * (clerk.automatos.app), so the sign-in page would not render after the cutover. The policy now
 * names the host the build's publishable key points at, plus the hosts Clerk's CSP guide lists
 * (clerk.com/docs/guides/secure/best-practices/csp-headers): its abuse and fraud protection hosts
 * in script-src and frame-src, and in connect-src on any port.
 *
 * Loads next.config.js the way csp-media-sources.test.ts does. The keys are built here from their
 * documented shape (pk_live_ + base64 of "<host>$"); they are not real keys.
 */
import { createRequire } from 'module'
import { afterEach, describe, expect, it } from 'vitest'

const load = createRequire(import.meta.url)
const { clerkFrontendApiOrigin } = load('../../csp-sources.js') as {
  clerkFrontendApiOrigin: (key: unknown) => string
}
const CONFIG = load.resolve('../../next.config.js')
const BEFORE = process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY

const PRODUCTION_HOST = 'clerk.automatos.app'
const DEVELOPMENT_HOST = 'example-instance-00.clerk.accounts.dev'

function keyFor(kind: 'test' | 'live', decoded: string): string {
  return `pk_${kind}_${Buffer.from(decoded).toString('base64')}`
}

async function policyWith(key: string | undefined): Promise<Record<string, string>> {
  if (key === undefined) delete process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY
  else process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY = key
  delete load.cache[CONFIG]
  const config = load(CONFIG) as { headers: () => Promise<Array<{ headers: Array<{ key: string; value: string }> }>> }
  const all = (await config.headers()).flatMap((rule) => rule.headers)
  const csp = all.find((header) => header.key === 'Content-Security-Policy')?.value ?? ''
  return Object.fromEntries(csp.split('; ').map((part) => [part.split(' ')[0], part]))
}

function sources(directive: string): string[] {
  return directive.split(/\s+/).slice(1)
}

afterEach(() => {
  if (BEFORE === undefined) delete process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY
  else process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY = BEFORE
  delete load.cache[CONFIG]
})

describe("the Clerk Frontend API origin a publishable key points at", () => {
  it('decodes production and development keys to their host', () => {
    expect(clerkFrontendApiOrigin(keyFor('live', `${PRODUCTION_HOST}$`))).toBe(`https://${PRODUCTION_HOST}`)
    expect(clerkFrontendApiOrigin(keyFor('test', `${DEVELOPMENT_HOST}$`))).toBe(`https://${DEVELOPMENT_HOST}`)
  })

  it('gives nothing for no key, a secret key, or a key that is not a bare host', () => {
    expect(clerkFrontendApiOrigin(undefined)).toBe('')
    expect(clerkFrontendApiOrigin('')).toBe('')
    expect(clerkFrontendApiOrigin(`sk_live_${Buffer.from(`${PRODUCTION_HOST}$`).toString('base64')}`)).toBe('')
    expect(clerkFrontendApiOrigin(keyFor('live', PRODUCTION_HOST))).toBe('')
    expect(clerkFrontendApiOrigin(keyFor('live', 'evil.example/path$'))).toBe('')
    expect(clerkFrontendApiOrigin(keyFor('live', "* 'unsafe-inline'$"))).toBe('')
  })
})

describe('the Content-Security-Policy with a production key', () => {
  it("loads clerk-js from the instance's Frontend API and Clerk's protection hosts", async () => {
    const policy = await policyWith(keyFor('live', `${PRODUCTION_HOST}$`))
    expect(sources(policy['script-src'])).toEqual(
      expect.arrayContaining([`https://${PRODUCTION_HOST}`, 'https://*.protect.clerk.com', 'https://challenges.cloudflare.com']),
    )
  })

  it('lets Clerk call its Frontend API and its protection hosts on any port', async () => {
    const policy = await policyWith(keyFor('live', `${PRODUCTION_HOST}$`))
    expect(sources(policy['connect-src'])).toEqual(
      expect.arrayContaining([`https://${PRODUCTION_HOST}`, 'https://*.protect.clerk.com:*', 'https://api.clerk.com']),
    )
  })

  it("keeps the rest of Clerk's guide: frames, images, workers and inline styles", async () => {
    const policy = await policyWith(keyFor('live', `${PRODUCTION_HOST}$`))
    expect(sources(policy['frame-src'])).toEqual(
      expect.arrayContaining(['https://*.protect.clerk.com', 'https://challenges.cloudflare.com']),
    )
    expect(sources(policy['img-src'])).toContain('https://img.clerk.com')
    expect(policy['worker-src']).toBe("worker-src 'self' blob:")
    expect(sources(policy['style-src'])).toContain("'unsafe-inline'")
  })
})

describe('the Content-Security-Policy without a production key', () => {
  it('keeps the development instance working until the cutover (and for a rollback)', async () => {
    const policy = await policyWith(keyFor('test', `${DEVELOPMENT_HOST}$`))
    expect(sources(policy['script-src'])).toContain('https://*.clerk.accounts.dev')
    expect(sources(policy['connect-src'])).toContain('https://*.clerk.accounts.dev')
  })

  it('adds no Frontend API host when there is no key (the local edition)', async () => {
    const policy = await policyWith(undefined)
    expect(policy['script-src']).toBe(
      "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://*.clerk.accounts.dev https://*.protect.clerk.com https://challenges.cloudflare.com https://cdn.jsdelivr.net",
    )
    expect(policy['connect-src']).not.toContain(PRODUCTION_HOST)
  })
})
