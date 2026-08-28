/**
 * PRD-228 US-004 — the fleet view tab.
 *
 * The tab renders one row per agent with the right live line (working / idle /
 * blocked: awaiting answer), queue depth, rolling-24h cost, and a watch badge;
 * clicking a row opens the EXISTING agent details modal via onViewDetails (no
 * new modal). The data hook is mocked — plumbing is covered in
 * use-fleet-state.test.tsx.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import type { FleetAgentRow, FleetStateResponse } from '@/lib/api-client'

const useFleetStateMock = vi.hoisted(() => vi.fn())

vi.mock('@/hooks/use-agent-api', () => ({
  useFleetState: useFleetStateMock,
}))

import { FleetTab } from '../fleet-tab'

function agent(overrides: Partial<FleetAgentRow> = {}): FleetAgentRow {
  return {
    agent_id: 1,
    name: 'Agent',
    current: null,
    queue_depth: 0,
    blocked: { count: 0, open_asks: [] },
    watches: { active: 0, needs_attention: 0 },
    last_activity_at: null,
    cost_24h: { tokens: 0, usd: 0 },
    ...overrides,
  }
}

function setFleet(agents: FleetAgentRow[], extra: Partial<FleetStateResponse> = {}) {
  const data: FleetStateResponse = {
    version: 1,
    generated_at: '2026-08-28T12:00:00Z',
    window_hours: 24,
    cost_available: true,
    cost_source: 'llm_usage',
    agents,
    ...extra,
  }
  useFleetStateMock.mockReturnValue({
    data,
    isLoading: false,
    isError: false,
    refetch: vi.fn(),
  })
}

describe('FleetTab — PRD-228', () => {
  beforeEach(() => {
    useFleetStateMock.mockReset()
  })

  it('renders working / idle / blocked live lines', () => {
    setFleet([
      agent({
        agent_id: 1, name: 'Builder',
        current: { kind: 'board_task', id: 7, title: 'Ship the widget', since: null },
        cost_24h: { tokens: 1200, usd: 0.34 },
      }),
      agent({ agent_id: 2, name: 'Bench' }),
      agent({ agent_id: 3, name: 'Stuck', blocked: { count: 1, open_asks: [99] } }),
    ])
    render(<FleetTab onViewDetails={vi.fn()} />)

    expect(screen.getByText('working: Ship the widget')).toBeInTheDocument()
    expect(screen.getByText('idle')).toBeInTheDocument()
    expect(screen.getByText('blocked: awaiting answer')).toBeInTheDocument()
    // Rolling-24h cost is labelled as such.
    expect(screen.getByText(/last 24h · 1,200 tok · \$0\.34/)).toBeInTheDocument()
  })

  it('a blocked agent with no open ask still shows blocked, not idle (P228-RVW-5)', () => {
    // Approval-/manually-blocked task: blocked.count>0 with open_asks empty.
    setFleet([
      agent({ agent_id: 8, name: 'Approval', current: null, blocked: { count: 1, open_asks: [] } }),
    ])
    render(<FleetTab onViewDetails={vi.fn()} />)
    const line = screen.getByText('blocked')
    expect(line).toBeInTheDocument()
    expect(line.className).toContain('destructive') // blocked tone, not idle
    expect(screen.queryByText('idle')).not.toBeInTheDocument()
  })

  it('a blocked agent that is also mid-task shows blocked (awaiting answer wins)', () => {
    setFleet([
      agent({
        agent_id: 5, name: 'Torn',
        current: { kind: 'board_task', id: 1, title: 'Half done', since: null },
        blocked: { count: 0, open_asks: [42] },
      }),
    ])
    render(<FleetTab onViewDetails={vi.fn()} />)
    expect(screen.getByText('blocked: awaiting answer')).toBeInTheDocument()
    expect(screen.queryByText('working: Half done')).not.toBeInTheDocument()
  })

  it('renders queue depth and a watch badge', () => {
    setFleet([
      agent({ agent_id: 1, name: 'Busy', queue_depth: 3, watches: { active: 2, needs_attention: 0 } }),
    ])
    render(<FleetTab onViewDetails={vi.fn()} />)
    expect(screen.getByText('queue 3')).toBeInTheDocument()
    expect(screen.getByText('2')).toBeInTheDocument() // watch badge count
  })

  it('row click opens the existing details modal via onViewDetails', () => {
    const onViewDetails = vi.fn()
    setFleet([agent({ agent_id: 42, name: 'Clickable' })])
    render(<FleetTab onViewDetails={onViewDetails} />)
    fireEvent.click(screen.getByTestId('fleet-row'))
    expect(onViewDetails).toHaveBeenCalledWith('42')
  })

  it('shows cost n/a when the cost source is unavailable', () => {
    const a = agent({ agent_id: 1, name: 'NoCost' })
    delete (a as any).cost_24h
    setFleet([a], { cost_available: false, cost_source: null })
    render(<FleetTab onViewDetails={vi.fn()} />)
    expect(screen.getByText('cost n/a')).toBeInTheDocument()
    expect(screen.getByText('cost unavailable')).toBeInTheDocument()
  })

  it('renders an honest empty state', () => {
    setFleet([])
    render(<FleetTab onViewDetails={vi.fn()} />)
    expect(screen.getByText(/No agents in this workspace/)).toBeInTheDocument()
  })

  it('renders a loading state', () => {
    useFleetStateMock.mockReturnValue({ data: undefined, isLoading: true, isError: false, refetch: vi.fn() })
    render(<FleetTab onViewDetails={vi.fn()} />)
    expect(screen.getByText(/Loading fleet/)).toBeInTheDocument()
  })
})
