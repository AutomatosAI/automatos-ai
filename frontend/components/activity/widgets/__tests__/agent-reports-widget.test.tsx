/**
 * F157 — the Agent Reports panel shows reports on every workspace.
 *
 * Local showed nothing where SaaS showed live reports. With nothing pinned the
 * endpoint returns the most recent reports (and the header says "N most
 * recent"), but the body rendered only the pin picker. Pins that name no agent
 * of this workspace (deleted, recreated with a new id, or pinned on another
 * workspace under the one browser-wide key) came back with no entry, and the
 * grid was empty with no word why. Pins are now per workspace, a pin with no
 * agent is dropped, the most recent reports show when nothing is pinned, and
 * pinned agents that never reported say so.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'

type Report = { agent_id: number; agent_name: string; status: string; summary: string }

const reportsFor = vi.fn<(ids: number[]) => Report[]>()

vi.mock('@/hooks/use-activity-api', () => ({
  useAgentReports: (ids: number[]) => ({
    data: { reports: reportsFor(ids).map((r) => ({ agent_icon: null, last_run: null, execution_id: null,
                                                   latest_file_path: null, ...r })) },
    isLoading: false,
  }),
}))
vi.mock('@/hooks/use-agent-api', () => ({
  useAgents: () => ({
    data: [{ id: 3, name: 'Accountant' }, { id: 7, name: 'Club Secretary' }],
    isLoading: false,
  }),
}))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspaceOptional: () => ({ workspace: { id: 'ws-1' } }),
}))
vi.mock('next/navigation', () => ({ useRouter: () => ({ push: vi.fn(), replace: vi.fn() }) }))
vi.mock('@/components/shared', () => ({ PremiumIcon: () => null }))

import { AgentReportsWidget, livePins, pinnedAgentsKey } from '../agent-reports-widget'

const SALES = { agent_id: 3, agent_name: 'Accountant', status: 'completed', summary: 'Weekly sales are up 4%.' }
const NEVER = { agent_id: 7, agent_name: 'Club Secretary', status: 'no_data', summary: '' }

function pinned(): number[] {
  return JSON.parse(localStorage.getItem(pinnedAgentsKey('ws-1')) ?? '[]')
}

beforeEach(() => {
  localStorage.clear()
  reportsFor.mockReset()
})

describe('AgentReportsWidget (F157)', () => {
  it('shows the most recent reports when nothing is pinned', () => {
    reportsFor.mockImplementation((ids) => (ids.length === 0 ? [SALES] : []))
    render(<AgentReportsWidget />)
    expect(screen.getByText(/Weekly sales are up 4%/)).toBeInTheDocument()
    expect(screen.queryByText('No agents pinned')).not.toBeInTheDocument()
  })

  it('drops a pin that names no agent of this workspace', async () => {
    localStorage.setItem(pinnedAgentsKey('ws-1'), JSON.stringify([3, 99]))
    reportsFor.mockImplementation((ids) => (ids.includes(3) ? [SALES] : []))
    render(<AgentReportsWidget />)
    await waitFor(() => expect(pinned()).toEqual([3]))
    expect(screen.getByText(/Weekly sales are up 4%/)).toBeInTheDocument()
  })

  it('says when the pinned agents have no reports', () => {
    localStorage.setItem(pinnedAgentsKey('ws-1'), JSON.stringify([7]))
    reportsFor.mockImplementation((ids) => (ids.length ? [NEVER] : [SALES]))
    render(<AgentReportsWidget />)
    expect(screen.getByText('Your pinned agents have no reports')).toBeInTheDocument()
  })

  it('keeps pins per workspace', () => {
    localStorage.setItem(pinnedAgentsKey('ws-2'), JSON.stringify([7]))
    reportsFor.mockImplementation((ids) => (ids.length === 0 ? [SALES] : [NEVER]))
    render(<AgentReportsWidget />)
    expect(reportsFor).not.toHaveBeenCalledWith([7])
    expect(screen.getByText(/Weekly sales are up 4%/)).toBeInTheDocument()
  })

  it('keeps the live pins in pin order', () => {
    expect(livePins([9, 3, 7], [7, 3])).toEqual([3, 7])
  })
})
