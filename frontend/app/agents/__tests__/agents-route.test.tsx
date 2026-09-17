/** PRD-244 W5a (two styles) — Studio desktop renders the Studio page; Classic keeps AgentManagement; below 1024 px both use Classic. */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'

const state = vi.hoisted(() => ({ tabletOrBelow: false, studio: true }))
vi.mock('@/hooks/use-mobile', () => ({ useIsMobile: () => false, useIsTabletOrBelow: () => state.tabletOrBelow }))
vi.mock('@/hooks/use-studio-theme', () => ({ useIsStudio: () => state.studio }))
vi.mock('@/hooks/use-page-api', () => ({ usePageAPI: () => {} }))
vi.mock('@/components/layout/main-layout', () => ({ MainLayout: ({ children }: { children: React.ReactNode }) => <div>{children}</div> }))
vi.mock('@/components/agents/studio/agent-management-studio', () => ({ AgentManagementStudio: () => <div data-testid="studio" /> }))
vi.mock('@/components/agents/agent-management', () => ({ AgentManagement: () => <div data-testid="classic" /> }))

import AgentsPage from '@/app/agents/page'

afterEach(() => { cleanup(); state.tabletOrBelow = false; state.studio = true })

describe('Agents route', () => {
  it('Studio desktop → the Studio page', () => {
    render(<AgentsPage />)
    expect(screen.getByTestId('studio')).toBeInTheDocument()
    expect(screen.queryByTestId('classic')).toBeNull()
  })
  it('Classic desktop → the classic page, never the Studio one', () => {
    state.studio = false
    render(<AgentsPage />)
    expect(screen.getByTestId('classic')).toBeInTheDocument()
    expect(screen.queryByTestId('studio')).toBeNull()
  })
  it('below 1024 px → the classic page in every style', () => {
    state.tabletOrBelow = true
    render(<AgentsPage />)
    expect(screen.getByTestId('classic')).toBeInTheDocument()
  })
})
