/** PRD-244 W5a — the Studio Agent Management page: in-page tabs from ?tab=, honest stats, the shared bodies. */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'

const nav = vi.hoisted(() => ({ replace: vi.fn(), search: '' }))
const data = vi.hoisted(() => ({ agents: [] as any[], stats: undefined as any, canEdit: true }))
vi.mock('next/navigation', () => ({
  useRouter: () => ({ replace: nav.replace, push: vi.fn() }),
  useSearchParams: () => new URLSearchParams(nav.search),
}))
vi.mock('@/hooks/use-agent-api', () => ({
  useAgents: () => ({ data: data.agents, isLoading: false, refetch: vi.fn(), error: null }),
  useAgentStats: () => ({ data: data.stats }),
}))
vi.mock('@/components/workspace-provider', () => ({ useWorkspace: () => ({ canEdit: data.canEdit }) }))
vi.mock('@/hooks/use-view-mode', () => ({ useViewMode: () => ['grid', vi.fn()] }))
vi.mock('@/components/shared/search-input', () => ({ SearchInput: () => <input aria-label="Search" /> }))
vi.mock('@/components/agents/agent-roster', () => ({ AgentRoster: () => <div data-testid="body-roster" /> }))
vi.mock('@/components/agents/fleet-tab', () => ({ FleetTab: () => <div data-testid="body-fleet" /> }))
vi.mock('@/components/agents/org-chart-tab', () => ({ OrgChartTab: () => <div data-testid="body-org-chart" /> }))
vi.mock('@/components/agents/agent-configuration', () => ({ AgentConfiguration: () => <div data-testid="body-configuration" /> }))
vi.mock('@/components/agents/skills/workspace-skills-tab', () => ({ WorkspaceSkillsTab: () => <div data-testid="body-skills" /> }))
vi.mock('@/components/agents/create-agent-modal', () => ({ CreateAgentModal: () => null }))
vi.mock('@/components/agents/agent-details-modal', () => ({ AgentDetailsModal: () => null }))

import { AgentManagementStudio } from '@/components/agents/studio/agent-management-studio'

beforeEach(() => { nav.replace.mockClear(); nav.search = ''; data.agents = []; data.stats = undefined; data.canEdit = true })
afterEach(cleanup)

describe('AgentManagementStudio', () => {
  it('renders the editorial head, five in-page tabs and the roster by default, with honest zero stats', () => {
    const { container } = render(<AgentManagementStudio />)
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Agent Management')
    expect(screen.getByText('Workforce · Roster · 0 agents')).toBeInTheDocument()
    expect(container.querySelectorAll('nav.cc-tabs button.cc-tab')).toHaveLength(5)
    expect(screen.getByTestId('body-roster')).toBeInTheDocument()
    expect(screen.getByText('—')).toBeInTheDocument() // no success rate fabricated
    expect(screen.getByText('no data yet')).toBeInTheDocument()
  })

  it('reads ?tab= for the body and writes it back on click', () => {
    nav.search = 'tab=fleet'
    render(<AgentManagementStudio />)
    expect(screen.getByTestId('body-fleet')).toBeInTheDocument()
    expect(screen.queryByTestId('body-roster')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Skills' }))
    expect(nav.replace).toHaveBeenCalledWith('/agents?tab=skills')
  })

  it('shows the roster count on the tab and the stats from the reads', () => {
    data.agents = [{ id: 1, status: 'active' }, { id: 2, status: 'failed' }]
    data.stats = { total_agents: 2, active_agents: 1, average_performance: 95 }
    render(<AgentManagementStudio />)
    expect(screen.getByText('Workforce · Roster · 2 agents')).toBeInTheDocument()
    expect(screen.getByText('95.0%')).toBeInTheDocument()
    expect(screen.getByText('1 failing')).toBeInTheDocument()
  })

  it('a viewer cannot create an agent', () => {
    data.canEdit = false
    render(<AgentManagementStudio />)
    expect(screen.getByRole('button', { name: /Create agent/ })).toBeDisabled()
  })
})
