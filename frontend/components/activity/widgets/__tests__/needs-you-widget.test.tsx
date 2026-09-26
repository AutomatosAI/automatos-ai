/** PRD-244 review batch 2 — "Needs you": questions, approvals and decisions in one place, each row landing on its tab. */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'

const data = vi.hoisted(() => ({
  decisions: { data: undefined as any, isLoading: false },
  gates: { data: undefined as any, isLoading: false },
  questions: { data: undefined as any, isLoading: false },
  reviews: { data: undefined as any, isLoading: false },
}))
vi.mock('@/hooks/use-kpi-api', () => ({ useDecisionsNeeded: () => data.decisions, useApprovalGates: () => data.gates }))
vi.mock('@/hooks/use-approval-grants', () => ({ useQuestions: () => data.questions }))
// The review family (night-1, 95f261a1f) reads the board's review column.
vi.mock('@/hooks/use-board-tasks-api', () => ({ useBoardTasksList: () => data.reviews }))

import { NeedsYouWidget } from '@/components/activity/widgets/needs-you-widget'

beforeEach(() => {
  data.decisions.data = undefined; data.gates.data = undefined
  data.questions.data = undefined; data.reviews.data = undefined
  ;(data.decisions as any).isError = false
})
afterEach(cleanup)

describe('NeedsYouWidget', () => {
  it('with nothing waiting, says so and fabricates no count', () => {
    render(<NeedsYouWidget period="1d" />)
    expect(screen.getByText('Nothing on your plate. Grab a tea.')).toBeInTheDocument()
    expect(screen.queryByText(/waiting$/)).toBeNull()
  })

  it('shows the three families with honest counts and links to the owning tab', () => {
    data.questions.data = { grants: [{ id: 41, question_md: '## Which vendor?', asked_by_agent_id: 3, requested_at: null }] }
    data.gates.data = { pending_count: 1, pending_missions: [{ id: 'm1', goal: 'Ship the invoice run', created_at: null, waiting_since: null }] }
    data.decisions.data = { total: 2, items: [{ kind: 'report', id: 'r1', title: 'Q3 review', escalation_level: 3, agent_name: 'OPS', created_at: null }, { kind: 'mission', id: 'm2', title: 'Budget', escalation_level: 0, created_at: null }] }
    render(<NeedsYouWidget period="1d" />)
    expect(screen.getByText('4 waiting')).toBeInTheDocument()
    expect(screen.getByText('Questions · 1')).toBeInTheDocument()
    expect(screen.getByText('Approvals · 1')).toBeInTheDocument()
    expect(screen.getByText('Decisions · 2')).toBeInTheDocument()
    expect(screen.getByText('Which vendor?').closest('a')).toHaveAttribute('href', '/command-center?tab=questions')
    expect(screen.getByText('Ship the invoice run').closest('a')).toHaveAttribute('href', '/command-center?tab=governance')
    expect(screen.getByText('Q3 review').closest('a')).toHaveAttribute('href', '/command-center?tab=governance')
    expect(screen.getByText('L3 URGENT')).toBeInTheDocument()
  })

  it('counts a ticket waiting in review as its own family, linked to the board', () => {
    data.reviews.data = { total: 1, tasks: [{ id: 760, title: 'Monday dispatch', agent_name: 'CLUB SECRETARY', completed_at: null }] }
    render(<NeedsYouWidget period="1d" />)
    expect(screen.getByText('1 waiting')).toBeInTheDocument()
    expect(screen.getByText('In review · 1')).toBeInTheDocument()
    expect(screen.getByText('Monday dispatch').closest('a')).toHaveAttribute('href', '/command-center?tab=board')
  })

  it('says the decisions could not be loaded instead of "nothing on your plate" (F207)', () => {
    data.decisions.data = { total: 0, reports_count: 0, missions_count: 0, items: [],
      error: 'The decisions waiting for you could not be loaded. Try again shortly.' }
    render(<NeedsYouWidget period="1d" />)
    expect(screen.queryByText('Nothing on your plate. Grab a tea.')).toBeNull()
    expect(screen.getByRole('status')).toHaveTextContent('could not be loaded')
  })

  it('treats a decisions request that failed the same way (F207)', () => {
    ;(data.decisions as any).isError = true
    render(<NeedsYouWidget period="1d" />)
    expect(screen.queryByText('Nothing on your plate. Grab a tea.')).toBeNull()
    expect(screen.getByRole('status')).toHaveTextContent('could not be loaded')
  })

  it('hides a family that has nothing', () => {
    data.gates.data = { pending_count: 1, pending_missions: [{ id: 'm1', goal: 'Approve me', created_at: null, waiting_since: null }] }
    render(<NeedsYouWidget period="1d" />)
    expect(screen.queryByText(/^Questions ·/)).toBeNull()
    expect(screen.getByText('Approvals · 1')).toBeInTheDocument()
  })
})
