/**
 * PRD-233 S7 — the edition seam's route allowlist.
 *
 * SaaS-only surfaces are hidden in `local` by ONE explicit list of route
 * prefixes (`SAAS_ONLY_ROUTES`), never by role: the local operator is
 * deliberately super_admin. In `saas` every helper is the identity so the
 * running product renders exactly as before.
 *
 * The seam reads NEXT_PUBLIC_AUTH_EDITION once at import, so each case stubs the
 * env and re-imports the module (same pattern as the PRD-175 seam test).
 */
import { describe, it, expect, vi, afterEach } from 'vitest'

afterEach(() => {
  vi.unstubAllEnvs()
  vi.resetModules()
})

async function seam(edition: 'local' | 'saas') {
  vi.stubEnv('NEXT_PUBLIC_AUTH_EDITION', edition)
  return import('@/lib/auth-edition')
}

// The owner's list (decisions recorded 2026-08-29): Workspace Admin, Team,
// invitations, sign-in/up (+ the Clerk-bound sign-in sub-flows), the hosted
// hub's plugin-moderation queue and the Clerk-only dev reset page.
const EXPECTED_LIST = [
  '/admin/workspaces',
  '/team',
  '/accept-invitation',
  '/sign-in',
  '/sign-up',
  '/reset-password',
  '/sso-callback',
  '/admin/plugins',
  '/dev/reset-onboarding',
]

describe('SAAS_ONLY_ROUTES — the explicit allowlist', () => {
  it('names exactly the hosted-edition surfaces, as path prefixes', async () => {
    const { SAAS_ONLY_ROUTES } = await seam('local')
    expect([...SAAS_ONLY_ROUTES]).toEqual(EXPECTED_LIST)
    for (const prefix of SAAS_ONLY_ROUTES) {
      expect(prefix.startsWith('/')).toBe(true)
      expect(prefix.endsWith('/')).toBe(false)
    }
  })

  it('is the same list in saas (the list is data; the edition decides whether it applies)', async () => {
    const { SAAS_ONLY_ROUTES } = await seam('saas')
    expect([...SAAS_ONLY_ROUTES]).toEqual(EXPECTED_LIST)
  })
})

describe('isRouteAvailableInEdition', () => {
  it('saas: every route is available, listed or not', async () => {
    const { isRouteAvailableInEdition } = await seam('saas')
    for (const href of [...EXPECTED_LIST, '/team/members', '/chat', '/analytics', '/settings']) {
      expect(isRouteAvailableInEdition(href)).toBe(true)
    }
  })

  it('local: a listed prefix is unavailable — exact, nested, with query or hash', async () => {
    const { isRouteAvailableInEdition } = await seam('local')
    expect(isRouteAvailableInEdition('/team')).toBe(false)
    expect(isRouteAvailableInEdition('/team/invitations')).toBe(false)
    expect(isRouteAvailableInEdition('/admin/workspaces')).toBe(false)
    expect(isRouteAvailableInEdition('/admin/workspaces?page=2')).toBe(false)
    expect(isRouteAvailableInEdition('/accept-invitation?token=abc')).toBe(false)
    expect(isRouteAvailableInEdition('/sign-in#top')).toBe(false)
    expect(isRouteAvailableInEdition('/sign-up/verify-email')).toBe(false)
    expect(isRouteAvailableInEdition('/reset-password')).toBe(false)
    expect(isRouteAvailableInEdition('/sso-callback')).toBe(false)
  })

  it('local: everything else stays available — including near-misses of a listed prefix', async () => {
    const { isRouteAvailableInEdition } = await seam('local')
    for (const href of ['/', '/chat', '/analytics', '/documents', '/marketplace', '/settings', '/settings/profile']) {
      expect(isRouteAvailableInEdition(href)).toBe(true)
    }
    // Prefix matching is on a segment boundary: '/team' must not swallow '/teams'.
    expect(isRouteAvailableInEdition('/teams')).toBe(true)
    // Only Workspace Admin is listed under /admin — the plugin console is not.
    expect(isRouteAvailableInEdition('/admin/settings')).toBe(true)
    // External links are never on the list.
    expect(isRouteAvailableInEdition('https://docs.automatos.app')).toBe(true)
  })
})

describe('filterNavForEdition', () => {
  const items = [
    { id: 'chat', href: '/chat' },
    { id: 'team', href: '/team' },
    { id: 'analytics', href: '/analytics' },
    { id: 'admin', href: '/admin/workspaces' },
    { id: 'docs', href: 'https://docs.automatos.app' },
  ]

  it('saas: returns the same items in the same order', async () => {
    const { filterNavForEdition } = await seam('saas')
    expect(filterNavForEdition(items)).toEqual(items)
  })

  it('local: drops exactly the listed hrefs, keeps order, does not mutate the input', async () => {
    const { filterNavForEdition } = await seam('local')
    const before = [...items]
    const result = filterNavForEdition(items)
    expect(result.map((i) => i.id)).toEqual(['chat', 'analytics', 'docs'])
    expect(items).toEqual(before)
    expect(result).not.toBe(items)
  })

  it('preserves the item type (extra fields survive the filter)', async () => {
    const { filterNavForEdition } = await seam('local')
    const typed = [{ href: '/chat', label: 'Chat', requiredExposure: 'analytics' as const }]
    const [first] = filterNavForEdition(typed)
    expect(first.label).toBe('Chat')
    expect(first.requiredExposure).toBe('analytics')
  })
})

describe('navExposureForEdition — plan-tier exposure as nav gating must honour it', () => {
  // What GET /api/workspaces/current returns for the plan-less local workspace
  // today: exposure_for_plan(plan or "basic") ⇒ the basic tier's nav profile.
  const localBasic = { plan: 'basic', nav: { analytics: false, team: false } }

  it('saas: passes the exposure through untouched (null and undefined included)', async () => {
    const { navExposureForEdition } = await seam('saas')
    expect(navExposureForEdition(localBasic)).toBe(localBasic)
    expect(navExposureForEdition(null)).toBeNull()
    expect(navExposureForEdition(undefined)).toBeUndefined()
  })

  it('local: there is no plan — exposure is unknown, so nav-exposure fails open', async () => {
    const { navExposureForEdition } = await seam('local')
    expect(navExposureForEdition(localBasic)).toBeUndefined()
    expect(navExposureForEdition(null)).toBeUndefined()
  })
})
