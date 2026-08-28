/**
 * PRD-222 W2·S1b (US-024) — the desktop Sidebar renders nav from plan-tier
 * exposure: basic hides Analytics + Team Management from the rail (their routes
 * still resolve — D5), business shows them, and an unknown exposure fails open.
 * Role/icon/workspace/navigation hooks are stubbed to isolate the nav filter.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'

vi.mock('next/navigation', () => ({ usePathname: () => '/chat' }))
vi.mock('@/contexts/role-context', () => ({
  useSystemRole: () => ({ systemRole: 'user', isAdmin: false }),
}))
vi.mock('@/hooks/use-system-config-api', () => ({
  useSystemIcons: () => ({ data: {} }),
}))

const workspaceRef = vi.hoisted(() => ({ current: null as any }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: workspaceRef.current }),
}))

import { Sidebar } from '../sidebar'

describe('Sidebar — plan-tier exposure gating (US-024)', () => {
  beforeEach(() => {
    workspaceRef.current = null
  })

  it('basic hides Analytics + Team Management, keeps ungated items', () => {
    workspaceRef.current = { exposure: { nav: { analytics: false, team: false } } }
    render(<Sidebar collapsed={false} onToggle={() => {}} />)

    expect(screen.queryByText('Analytics')).toBeNull()
    expect(screen.queryByText('Team Management')).toBeNull()
    // Ungated surfaces remain on the rail.
    expect(screen.getByText('Chat')).toBeInTheDocument()
    expect(screen.getByText('Command Center')).toBeInTheDocument()
    expect(screen.getByText('Marketplace')).toBeInTheDocument()
  })

  it('business shows Analytics + Team Management', () => {
    workspaceRef.current = { exposure: { nav: { analytics: true, team: true } } }
    render(<Sidebar collapsed={false} onToggle={() => {}} />)

    expect(screen.getByText('Analytics')).toBeInTheDocument()
    expect(screen.getByText('Team Management')).toBeInTheDocument()
  })

  it('fails open when exposure is unknown (loading / solo)', () => {
    workspaceRef.current = null
    render(<Sidebar collapsed={false} onToggle={() => {}} />)

    expect(screen.getByText('Analytics')).toBeInTheDocument()
    expect(screen.getByText('Team Management')).toBeInTheDocument()
  })
})
