/**
 * PRD-233 S7 — edition-aware navigation (desktop Sidebar).
 *
 * `local`: the SaaS-only surfaces (Team Management, Workspace Admin) are absent
 * from the rail by the explicit SAAS_ONLY_ROUTES list — NOT by role: the local
 * operator is deliberately super_admin, so the role gate alone would show
 * Workspace Admin. Plan-tier exposure is neutralised locally: the backend derives
 * the `basic` profile from the plan-less local workspace (nav.analytics=false),
 * which must not hide Analytics on the operator's own instance.
 *
 * `saas`: the rail is exactly today's — role gate + US-024 exposure gating.
 *
 * The real seam is exercised: the edition env is stubbed and the module graph is
 * reset per test (the seam reads NEXT_PUBLIC_AUTH_EDITION once at import).
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

// The local operator is super_admin (role-context.tsx) ⇒ isAdmin is TRUE locally.
const roleRef = vi.hoisted(() => ({ current: { systemRole: 'super_admin', isAdmin: true } }))
vi.mock('@/contexts/role-context', () => ({
  useSystemRole: () => roleRef.current,
}))

const workspaceRef = vi.hoisted(() => ({ current: null as any }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: workspaceRef.current }),
}))

// The exact exposure GET /api/workspaces/current returns for the plan-less local
// workspace (exposure_for_plan(None or "basic")) — the trap S7 must not fall into.
const LOCAL_BASIC_EXPOSURE = {
  plan: 'basic',
  families: { codegraph: false, nl2sql: false, team: false, voice: false },
  marketplace_depth: 1,
  nav: { analytics: false, team: false },
}
const BUSINESS_EXPOSURE = {
  plan: 'business',
  families: { codegraph: true, nl2sql: true, team: true, voice: true },
  marketplace_depth: 3,
  nav: { analytics: true, team: true },
}

// Today's desktop rail, in order (the saas invariant).
const SAAS_RAIL = [
  '/chat', '/command-center', '/assignments', '/deliverables',
  '/agents', '/tools', '/documents', '/marketplace',
  '/team', '/analytics', '/admin/workspaces',
]

async function renderSidebar(edition: 'local' | 'saas') {
  vi.stubEnv('NEXT_PUBLIC_AUTH_EDITION', edition)
  const { Sidebar } = await import('../sidebar')
  return render(<Sidebar collapsed={false} onToggle={() => {}} />)
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

describe('Sidebar — local edition (PRD-233 S7)', () => {
  it('hides Team Management + Workspace Admin even though the operator is super_admin', async () => {
    workspaceRef.current = { exposure: BUSINESS_EXPOSURE }
    const { container } = await renderSidebar('local')

    expect(screen.queryByText('Team Management')).toBeNull()
    expect(screen.queryByText('Workspace Admin')).toBeNull()
    expect(container.querySelector('a[href="/team"]')).toBeNull()
    expect(container.querySelector('a[href="/admin/workspaces"]')).toBeNull()
  })

  it('keeps Analytics despite the basic-tier exposure the plan-less local workspace gets', async () => {
    workspaceRef.current = { exposure: LOCAL_BASIC_EXPOSURE }
    const { container } = await renderSidebar('local')

    expect(screen.getByText('Analytics')).toBeInTheDocument()
    expect(container.querySelector('a[href="/analytics"]')).not.toBeNull()
    expect(screen.getByText('Knowledge Base')).toBeInTheDocument()
    expect(screen.getByText('Marketplace')).toBeInTheDocument()
    expect(screen.getByText('Chat')).toBeInTheDocument()
    expect(screen.queryByText('Team Management')).toBeNull()
    expect(screen.queryByText('Workspace Admin')).toBeNull()
  })

  it('renders today\'s rail minus exactly the SaaS-only entries, in order (workspace still loading)', async () => {
    workspaceRef.current = null
    const { container } = await renderSidebar('local')

    expect(railHrefs(container)).toEqual(SAAS_RAIL.filter((h) => h !== '/team' && h !== '/admin/workspaces'))
  })

  it('still offers Settings in the footer', async () => {
    const { container } = await renderSidebar('local')
    expect(container.querySelector('a[href="/settings"]')).not.toBeNull()
  })
})

describe('Sidebar — saas edition renders exactly as today', () => {
  it('admin on business: full rail in today\'s order', async () => {
    workspaceRef.current = { exposure: BUSINESS_EXPOSURE }
    const { container } = await renderSidebar('saas')

    expect(railHrefs(container)).toEqual(SAAS_RAIL)
    expect(screen.getByText('Team Management')).toBeInTheDocument()
    expect(screen.getByText('Workspace Admin')).toBeInTheDocument()
    expect(screen.getByText('Analytics')).toBeInTheDocument()
  })

  it('non-admin on basic: US-024 exposure still hides Analytics + Team, role gate hides Workspace Admin', async () => {
    roleRef.current = { systemRole: 'user', isAdmin: false }
    workspaceRef.current = { exposure: LOCAL_BASIC_EXPOSURE }
    const { container } = await renderSidebar('saas')

    expect(screen.queryByText('Analytics')).toBeNull()
    expect(screen.queryByText('Team Management')).toBeNull()
    expect(screen.queryByText('Workspace Admin')).toBeNull()
    expect(railHrefs(container)).toEqual(SAAS_RAIL.filter((h) => !['/team', '/analytics', '/admin/workspaces'].includes(h)))
  })

  it('unknown exposure fails open (loading / solo) — unchanged', async () => {
    workspaceRef.current = null
    await renderSidebar('saas')

    expect(screen.getByText('Analytics')).toBeInTheDocument()
    expect(screen.getByText('Team Management')).toBeInTheDocument()
  })
})
