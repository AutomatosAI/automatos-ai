/**
 * PRD-233 S7 — edition-aware navigation (Studio rail, the studio-menu consumer).
 *
 * `local`: Team Management + Workspace Admin are absent by the explicit list;
 * Analytics survives the basic-tier exposure of the plan-less local workspace.
 * `saas`: the 11-item rail + footer render exactly as today.
 */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'

// Dynamic imports transform the whole rail (framer-motion, lucide) on first use;
// under CI/parallel load that can exceed vitest's 5s default and a timed-out
// render would leak DOM into the next case. Generous budget + explicit cleanup.
vi.setConfig({ testTimeout: 30_000 })

vi.mock('next/navigation', () => ({ usePathname: () => '/chat' }))

const workspaceRef = vi.hoisted(() => ({ current: null as any }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspaceOptional: () => ({ workspace: workspaceRef.current }),
}))
// PRD-244 W0: the rail gates Workspace Admin + Settings by system role, like
// the classic rail. Admin by default here so the edition cases read as before;
// the role cases below flip it.
const roleRef = vi.hoisted(() => ({ current: { isAdmin: true } as { isAdmin: boolean } | null }))
vi.mock('@/contexts/role-context', () => ({
  useSystemRoleOptional: () => roleRef.current,
}))

const LOCAL_BASIC_EXPOSURE = { plan: 'basic', families: {}, marketplace_depth: 1, nav: { analytics: false, team: false } }
const BUSINESS_EXPOSURE = { plan: 'business', families: {}, marketplace_depth: 3, nav: { analytics: true, team: true } }

const SAAS_RAIL = [
  '/chat', '/command-center', '/assignments', '/deliverables',
  '/agents', '/tools', '/documents', '/marketplace',
  '/team', '/analytics', '/admin/workspaces',
]

async function renderStudio(edition: 'local' | 'saas') {
  vi.stubEnv('NEXT_PUBLIC_AUTH_EDITION', edition)
  const { StudioSidebar } = await import('../studio-sidebar')
  return render(<StudioSidebar />)
}

function railHrefs(container: HTMLElement): string[] {
  return Array.from(container.querySelectorAll('nav a')).map((a) => a.getAttribute('href') ?? '')
}

afterEach(() => {
  cleanup()
  vi.unstubAllEnvs()
  vi.resetModules()
  workspaceRef.current = null
  roleRef.current = { isAdmin: true }
})

describe('PRD-244 W0 — role gates match the classic rail', () => {
  it('a member sees neither Workspace Admin nor Settings', async () => {
    roleRef.current = { isAdmin: false }
    workspaceRef.current = { exposure: BUSINESS_EXPOSURE }
    const { container } = await renderStudio('saas')
    expect(railHrefs(container)).toEqual(SAAS_RAIL.filter((h) => h !== '/admin/workspaces'))
    expect(container.querySelector('a[href="/settings"]')).toBeNull()
    expect(container.querySelector('a[href="https://docs.automatos.app"]')).not.toBeNull()
  })

  it('an unknown role (no provider) is not an admin', async () => {
    roleRef.current = null
    workspaceRef.current = { exposure: BUSINESS_EXPOSURE }
    const { container } = await renderStudio('saas')
    expect(screen.queryByText('Workspace Admin')).toBeNull()
    expect(container.querySelector('a[href="/settings"]')).toBeNull()
  })
})

describe('StudioSidebar — local edition (PRD-233 S7)', () => {
  it('drops Team Management + Workspace Admin, keeps Analytics under the basic-tier exposure', async () => {
    workspaceRef.current = { exposure: LOCAL_BASIC_EXPOSURE }
    const { container } = await renderStudio('local')

    expect(screen.queryByText('Team Management')).toBeNull()
    expect(screen.queryByText('Workspace Admin')).toBeNull()
    expect(screen.getByText('Analytics')).toBeInTheDocument()
    expect(screen.getByText('Knowledge Base')).toBeInTheDocument()
    expect(screen.getByText('Marketplace')).toBeInTheDocument()
    expect(railHrefs(container)).toEqual(SAAS_RAIL.filter((h) => h !== '/team' && h !== '/admin/workspaces'))
  })

  it('keeps the footer (Docs + Settings)', async () => {
    const { container } = await renderStudio('local')
    expect(container.querySelector('a[href="/settings"]')).not.toBeNull()
    expect(container.querySelector('a[href="https://docs.automatos.app"]')).not.toBeNull()
  })
})

describe('StudioSidebar — saas edition renders exactly as today', () => {
  it('business: full rail in today\'s order', async () => {
    workspaceRef.current = { exposure: BUSINESS_EXPOSURE }
    const { container } = await renderStudio('saas')
    expect(railHrefs(container)).toEqual(SAAS_RAIL)
    expect(screen.getByText('Workspace Admin')).toBeInTheDocument()
  })

  it('basic: US-024 still hides Team + Analytics (Workspace Admin is not exposure-gated on this rail)', async () => {
    workspaceRef.current = { exposure: LOCAL_BASIC_EXPOSURE }
    const { container } = await renderStudio('saas')
    expect(railHrefs(container)).toEqual(SAAS_RAIL.filter((h) => h !== '/team' && h !== '/analytics'))
  })

  it('no provider (studio shell in isolation) fails open — unchanged', async () => {
    workspaceRef.current = null
    const { container } = await renderStudio('saas')
    expect(railHrefs(container)).toEqual(SAAS_RAIL)
  })
})
