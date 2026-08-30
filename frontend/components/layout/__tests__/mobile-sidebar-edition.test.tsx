/**
 * PRD-233 S7 — edition-aware navigation (MobileSidebar mirror).
 *
 * Same contract as the desktop rail: `local` drops the SaaS-only entries by the
 * explicit list (Team) and neutralises the plan-tier exposure (Analytics stays);
 * `saas` renders exactly as today.
 */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'

// Dynamic imports transform the whole rail (framer-motion, lucide) on first use;
// under CI/parallel load that can exceed vitest's 5s default and a timed-out
// render would leak DOM into the next case. Generous budget + explicit cleanup.
vi.setConfig({ testTimeout: 30_000 })

vi.mock('next/navigation', () => ({ usePathname: () => '/chat' }))
vi.mock('@/hooks/use-system-config-api', () => ({
  useSystemIcons: () => ({ data: {} }),
}))

const roleRef = vi.hoisted(() => ({ current: { systemRole: 'super_admin', isAdmin: true } }))
vi.mock('@/contexts/role-context', () => ({
  useSystemRole: () => roleRef.current,
}))

const workspaceRef = vi.hoisted(() => ({ current: null as any }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: workspaceRef.current }),
}))

const LOCAL_BASIC_EXPOSURE = { plan: 'basic', families: {}, marketplace_depth: 1, nav: { analytics: false, team: false } }
const BUSINESS_EXPOSURE = { plan: 'business', families: {}, marketplace_depth: 3, nav: { analytics: true, team: true } }

// Today's mobile rail, in order (the saas invariant).
const SAAS_RAIL = [
  '/chat', '/assignments', '/deliverables', '/command-center',
  '/agents', '/tools', '/marketplace', '/documents',
  '/team', '/analytics',
]

async function renderMobile(edition: 'local' | 'saas') {
  vi.stubEnv('NEXT_PUBLIC_AUTH_EDITION', edition)
  const { MobileSidebar } = await import('../mobile-sidebar')
  return render(<MobileSidebar onNavigate={() => {}} />)
}

function railHrefs(container: HTMLElement): string[] {
  return Array.from(container.querySelectorAll('nav a')).map((a) => a.getAttribute('href') ?? '')
}

afterEach(() => {
  cleanup()
  vi.unstubAllEnvs()
  vi.resetModules()
  workspaceRef.current = null
  roleRef.current = { systemRole: 'super_admin', isAdmin: true }
})

describe('MobileSidebar — local edition (PRD-233 S7)', () => {
  it('hides Team, keeps Analytics under the basic-tier exposure of the plan-less local workspace', async () => {
    workspaceRef.current = { exposure: LOCAL_BASIC_EXPOSURE }
    const { container } = await renderMobile('local')

    expect(container.querySelector('a[href="/team"]')).toBeNull()
    expect(screen.queryByText('Team')).toBeNull()
    expect(container.querySelector('a[href="/analytics"]')).not.toBeNull()
    expect(screen.getByText('Analytics')).toBeInTheDocument()
    expect(screen.getByText('Knowledge')).toBeInTheDocument()
    expect(screen.getByText('Marketplace')).toBeInTheDocument()
    expect(railHrefs(container)).toEqual(SAAS_RAIL.filter((h) => h !== '/team'))
  })

  it('keeps Settings (the operator is admin locally)', async () => {
    const { container } = await renderMobile('local')
    expect(container.querySelector('a[href="/settings"]')).not.toBeNull()
  })
})

describe('MobileSidebar — saas edition renders exactly as today', () => {
  it('business: full rail in today\'s order', async () => {
    workspaceRef.current = { exposure: BUSINESS_EXPOSURE }
    const { container } = await renderMobile('saas')
    expect(railHrefs(container)).toEqual(SAAS_RAIL)
  })

  it('basic: US-024 still hides Team + Analytics', async () => {
    workspaceRef.current = { exposure: LOCAL_BASIC_EXPOSURE }
    const { container } = await renderMobile('saas')
    expect(railHrefs(container)).toEqual(SAAS_RAIL.filter((h) => h !== '/team' && h !== '/analytics'))
  })
})
