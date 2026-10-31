// @vitest-environment node
/**
 * PRD-270 US-270-10 (§5.7 Search engines, K23) — /robots.txt in each edition.
 *
 * SaaS: the public Marketplace is allowed, with the logo its share cards show and the built assets
 * its storefront draws with; everything else, the Marketplace's signed-in sub-pages included, is
 * disallowed. Local: everything is disallowed.
 * The rules are also read the way a crawler reads them (the longest matching path wins, an Allow
 * on a tie), so the test proves what is indexable, not only the literal lists.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { MetadataRoute } from 'next'

const edition = vi.hoisted(() => ({ saas: true }))

vi.mock('@/lib/auth-edition', () => ({
  get isSaaS() {
    return edition.saas
  },
  get isLocal() {
    return !edition.saas
  },
}))

import robots from '@/app/robots'

type Rule = { userAgent?: string | string[]; allow?: string | string[]; disallow?: string | string[] }

function onlyRule(): Rule {
  const { rules } = robots() as MetadataRoute.Robots
  expect(Array.isArray(rules)).toBe(false)
  return rules as Rule
}

function paths(value: string | string[] | undefined): string[] {
  return value === undefined ? [] : Array.isArray(value) ? value : [value]
}

function longestMatch(prefixes: readonly string[], path: string): number {
  return Math.max(-1, ...prefixes.filter((prefix) => path.startsWith(prefix)).map((prefix) => prefix.length))
}

/** robots.txt's own reading: the longest matching rule decides, and Allow wins a tie. */
function crawlable(rule: Rule, path: string): boolean {
  const allowed = longestMatch(paths(rule.allow), path)
  const disallowed = longestMatch(paths(rule.disallow), path)
  return allowed >= disallowed
}

afterEach(() => {
  edition.saas = true
})

describe('/robots.txt on SaaS', () => {
  it('allows /marketplace, the share-card logo and the built assets, and disallows everything else', () => {
    expect(onlyRule()).toEqual({
      userAgent: '*',
      allow: ['/marketplace', '/brand/automatos-mark-hi.png', '/_next/static/'],
      disallow: ['/', '/marketplace/'],
    })
  })

  it.each([
    '/marketplace',
    '/marketplace?tab=themes&item=automatos.arcade',
    '/marketplace?tab=agents&item=research-assistant',
    '/brand/automatos-mark-hi.png',
    '/_next/static/chunks/app/marketplace/page.js',
  ])(
    'lets a crawler read %s',
    (path) => {
      expect(crawlable(onlyRule(), path)).toBe(true)
    },
  )

  it.each([
    '/',
    '/chat',
    '/settings',
    '/marketplace/publish',
    '/marketplace/developer',
    '/marketplace/widgets/1',
    '/sign-in',
    '/brand/automatos-logo.png',
    '/api/themes/files/x',
  ])(
    'keeps a crawler out of %s',
    (path) => {
      expect(crawlable(onlyRule(), path)).toBe(false)
    },
  )
})

describe('/robots.txt on the local edition', () => {
  it('disallows everything', () => {
    edition.saas = false

    const rule = onlyRule()

    expect(rule).toEqual({ userAgent: '*', disallow: '/' })
    expect(crawlable(rule, '/marketplace')).toBe(false)
  })
})
