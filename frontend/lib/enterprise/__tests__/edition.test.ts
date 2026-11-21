/**
 * PRD-277 US-501 — `resolveEdition()` knows `enterprise`, and the edition seam answers for it.
 *
 * The seam reads NEXT_PUBLIC_AUTH_EDITION once at import, so each case stubs the env and imports
 * the modules again (the lib/__tests__/auth-edition-routes.test.ts pattern).
 *   - enterprise is its own edition (not saas, not local); an unknown value is still saas.
 *   - enterprise has the Team page and its own sign-in page; every other hosted-only surface
 *     (Workspace Admin, invitations, sign-up, password reset, SSO callback) stays hidden.
 *   - a theme reads enterprise as 'saas': the theme SDK's contract names two editions.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

afterEach(() => {
  vi.unstubAllEnvs()
  vi.resetModules()
})

async function seam(edition: string) {
  vi.stubEnv('NEXT_PUBLIC_AUTH_EDITION', edition)
  return import('@/lib/auth-edition')
}

describe('resolveEdition', () => {
  it.each([
    ['enterprise', 'enterprise'],
    [' Enterprise ', 'enterprise'],
    ['local', 'local'],
    ['saas', 'saas'],
    ['', 'saas'],
    ['enterprize', 'saas'],
  ])('reads %j as %s', async (raw, expected) => {
    const edition = await seam(raw)
    expect(edition.authEdition).toBe(expected)
    expect(edition.resolveEdition()).toBe(expected)
    expect(edition.isEnterprise).toBe(expected === 'enterprise')
    expect(edition.isSaaS).toBe(expected === 'saas')
    expect(edition.isLocal).toBe(expected === 'local')
  })

  it('is the same value from lib/edition-name, which tests never mock', async () => {
    vi.stubEnv('NEXT_PUBLIC_AUTH_EDITION', 'enterprise')
    expect((await import('@/lib/edition-name')).isEnterprise).toBe(true)
  })
})

describe('isRouteAvailableInEdition in enterprise', () => {
  it.each(['/team', '/team/members', '/sign-in', '/sign-in?return_to=%2Fchat'])('keeps %s', async (href) => {
    expect((await seam('enterprise')).isRouteAvailableInEdition(href)).toBe(true)
  })

  it.each(['/admin/workspaces', '/accept-invitation', '/sign-up', '/reset-password', '/sso-callback', '/admin/plugins'])(
    'hides %s',
    async (href) => {
      expect((await seam('enterprise')).isRouteAvailableInEdition(href)).toBe(false)
    },
  )

  it('leaves local and saas exactly as they were', async () => {
    expect((await seam('local')).isRouteAvailableInEdition('/team')).toBe(false)
    vi.resetModules()
    expect((await seam('local')).isRouteAvailableInEdition('/sign-in')).toBe(false)
    vi.resetModules()
    expect((await seam('saas')).isRouteAvailableInEdition('/accept-invitation')).toBe(true)
  })

  it('fails the plan exposure open, as local does (no plans in enterprise)', async () => {
    expect((await seam('enterprise')).navExposureForEdition({ nav: { analytics: false } })).toBeUndefined()
  })
})

describe("the theme SDK's useEdition", () => {
  it("reads enterprise as 'saas'", async () => {
    const { renderHook } = await import('@testing-library/react')
    vi.stubEnv('NEXT_PUBLIC_AUTH_EDITION', 'enterprise')
    const { useEdition } = await import('@/lib/theme-sdk/context')
    expect(renderHook(() => useEdition()).result.current).toBe('saas')
  })
})
